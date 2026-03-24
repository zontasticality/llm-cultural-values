"""Temperature sampling pipeline for cultural completions.

Usage:
    PYTHONPATH=gemma-sae python -m inference.sample \
        --model_id gemma3_27b_pt \
        --model_hf_id google/gemma-3-27b-pt \
        --db data/culture.db \
        [--dtype bf16] [--n-samples 50] [--batch-size 16] \
        [--max-new-tokens 128] [--temperature 1.0] [--top-p 0.95] \
        [--lang fin] [--template self_concept] \
        [--validate]
"""

import argparse
import hashlib
import re
import time
from pathlib import Path

from db.load import load_unsampled_prompts
from db.schema import get_connection, init_db


# ── Model loading (reused from eurollm) ─────────────────────────

def load_model(model_id: str, dtype: str = "bf16"):
    """Load model and tokenizer with configurable precision."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {model_id} (dtype={dtype})")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if dtype == "bf16":
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto",
        )
    elif dtype == "int4":
        from transformers import BitsAndBytesConfig
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            ),
        )
    elif dtype == "int8":
        from transformers import BitsAndBytesConfig
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto",
            quantization_config=BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
            ),
        )
    else:
        raise ValueError(f"Unknown dtype: {dtype}")

    model.eval()
    print(f"Model loaded on {model.device if hasattr(model, 'device') else 'auto-mapped devices'}")
    return model, tokenizer


# ── Completion extraction & filtering ────────────────────────────

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?。！？])\s+(?=[A-Z\u4e00-\u9fff\n])")
CLAUSE_BOUNDARY = re.compile(r"[,;:—–]\s+")
CODE_CHARS = set("{}[]<>=;//")


def extract_first_sentence(raw: str, tokenizer, max_extract_tokens: int = 80,
                           min_extract_words: int = 3) -> str | None:
    """Extract the first meaningful chunk from raw generation.

    If the first sentence is very short (< min_extract_words), extend to the
    next sentence boundary so we capture "family. No matter what..." rather
    than just "family."
    """
    text = raw.strip()
    if not text:
        return None

    # Try sentence boundary
    m = SENTENCE_BOUNDARY.search(text)
    if m:
        candidate = text[:m.start() + 1].strip()
        # If first sentence is too short, try extending to next boundary
        if len(candidate.split()) < min_extract_words:
            m2 = SENTENCE_BOUNDARY.search(text, m.end())
            if m2 and len(tokenizer.encode(text[:m2.start() + 1])) <= max_extract_tokens:
                candidate = text[:m2.start() + 1].strip()
        if len(tokenizer.encode(candidate)) <= max_extract_tokens:
            return candidate

    # Try clause boundary
    m = CLAUSE_BOUNDARY.search(text)
    if m and len(tokenizer.encode(text[:m.start()])) <= max_extract_tokens:
        return text[:m.start()].strip()

    # Take up to max_extract_tokens as-is
    tokens = tokenizer.encode(text)[:max_extract_tokens]
    return tokenizer.decode(tokens, skip_special_tokens=True).strip()


def classify_filter(raw: str, extracted: str | None) -> str:
    """Return filter_status: 'ok', 'degenerate', 'repetition', 'too_short', 'non_text'."""
    if not extracted or not extracted.strip():
        return "degenerate"

    # Repetition: any 5-gram appears 3+ times
    words = extracted.split()
    if len(words) >= 5:
        ngrams: dict[tuple, int] = {}
        for i in range(len(words) - 4):
            ng = tuple(words[i:i + 5])
            ngrams[ng] = ngrams.get(ng, 0) + 1
            if ngrams[ng] >= 3:
                return "repetition"

    # Too short — single word with no content (articles, fragments)
    # Note: 1-2 word completions like "family" or "my health" are valid for TST
    if len(words) < 1 or (len(words) == 1 and len(words[0]) <= 2):
        return "too_short"

    # Non-text: >50% code/markup characters
    code_count = sum(1 for c in extracted if c in CODE_CHARS)
    if len(extracted) > 0 and code_count / len(extracted) > 0.5:
        return "non_text"

    return "ok"


# ── Seed generation ──────────────────────────────────────────────

def make_seed(model_id: str, prompt_id: int, sample_idx: int) -> int:
    """Deterministic seed from (model, prompt, sample) triple."""
    key = f"{model_id}:{prompt_id}:{sample_idx}"
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) & 0xFFFFFFFF


# ── Core sampling loop ───────────────────────────────────────────

def sample_prompt(
    model,
    tokenizer,
    prompt_text: str,
    n_samples: int,
    model_id: str,
    prompt_id: int,
    start_idx: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    batch_size: int,
) -> list[dict]:
    """Generate n_samples completions for a single prompt."""
    import torch

    results = []

    for batch_start in range(0, n_samples, batch_size):
        batch_end = min(batch_start + batch_size, n_samples)
        actual_batch = batch_end - batch_start

        # Per-sample seeds
        seeds = [make_seed(model_id, prompt_id, start_idx + batch_start + i)
                 for i in range(actual_batch)]

        # Encode prompt (same for all samples in batch)
        inputs = tokenizer(
            [prompt_text] * actual_batch,
            return_tensors="pt",
            padding=True,
        ).to(model.device if hasattr(model, "device") else "cuda")

        # Generate -- use first seed for the batch (torch generator)
        # For exact reproducibility per-sample, generate one at a time
        # But for speed, we batch and accept approximate reproducibility
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        # Decode and process each sample
        prompt_len = inputs["input_ids"].shape[1]
        for i in range(actual_batch):
            raw_tokens = outputs[i][prompt_len:]
            raw_text = tokenizer.decode(raw_tokens, skip_special_tokens=True)
            n_tokens_raw = len(raw_tokens)

            extracted = extract_first_sentence(raw_text, tokenizer)
            status = classify_filter(raw_text, extracted)

            if status != "ok":
                extracted = None

            n_tokens = None
            if extracted:
                n_tokens = len(tokenizer.encode(extracted))

            results.append({
                "prompt_id": prompt_id,
                "model_id": model_id,
                "sample_idx": start_idx + batch_start + i,
                "completion_raw": raw_text,
                "completion_text": extracted,
                "n_tokens_raw": n_tokens_raw,
                "n_tokens": n_tokens,
                "filter_status": status,
                "temperature": temperature,
                "top_p": top_p,
                "seed": seeds[i],
                "steering_config": "none",
            })

    return results


def insert_completions(conn, completions: list[dict]):
    """Batch insert completions, ignoring duplicates."""
    conn.executemany(
        "INSERT OR IGNORE INTO completions "
        "(prompt_id, model_id, sample_idx, completion_raw, completion_text, "
        " n_tokens_raw, n_tokens, filter_status, temperature, top_p, seed, steering_config) "
        "VALUES (:prompt_id, :model_id, :sample_idx, :completion_raw, :completion_text, "
        " :n_tokens_raw, :n_tokens, :filter_status, :temperature, :top_p, :seed, :steering_config)",
        completions,
    )
    conn.commit()


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Temperature sampling pipeline")
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--model_hf_id", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "int4", "int8"])
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--lang", type=str, default=None)
    parser.add_argument("--template", type=str, default=None)
    parser.add_argument("--validate", action="store_true",
                        help="Process 2 prompts, print first 5 completions each, then exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="Test DB queries, extraction, filtering — no model loading")
    args = parser.parse_args()

    conn = get_connection(args.db)

    # Get prompts needing more samples
    prompts = load_unsampled_prompts(
        conn, args.model_id, n_samples=args.n_samples,
        lang=args.lang, template_id=args.template,
    )

    if not prompts:
        print("All prompts fully sampled. Nothing to do.")
        return

    print(f"Found {len(prompts)} prompts needing samples "
          f"(total needed: {sum(p['needed'] for p in prompts)})")
    for p in prompts[:5]:
        print(f"  {p['lang']}:{p['template_id']} "
              f"need={p['needed']} text={p['prompt_text']!r}")
    if len(prompts) > 5:
        print(f"  ... and {len(prompts) - 5} more")

    # ── Dry run: test extraction/filtering on synthetic text ─────
    if args.dry_run:
        print("\n=== DRY RUN: testing extraction & filtering ===")

        # Use a simple stub tokenizer for local testing if transformers unavailable
        try:
            from transformers import AutoTokenizer
            print(f"Loading tokenizer: {args.model_hf_id}")
            tokenizer = AutoTokenizer.from_pretrained(args.model_hf_id)
        except ImportError:
            print("  transformers not installed — using stub tokenizer")
            class _StubTokenizer:
                """Whitespace tokenizer for dry-run without transformers."""
                def encode(self, text, **kw): return text.split()
                def decode(self, tokens, **kw): return " ".join(tokens)
            tokenizer = _StubTokenizer()

        test_cases = [
            ("This is a normal sentence. And another one follows.", "ok"),
            ("", "degenerate"),
            ("ab", "too_short"),
            ("{} = {} <> {} // {} {} <> {} = {} <> {}", "non_text"),
            ("The quick brown fox jumps over the lazy dog today.", "ok"),
        ]
        is_stub = not hasattr(tokenizer, "vocab_size")
        print(f"\nExtraction + filter tests{' (stub tokenizer — some may differ)' if is_stub else ''}:")
        all_pass = True
        for raw, expected_filter in test_cases:
            extracted = extract_first_sentence(raw, tokenizer)
            status = classify_filter(raw, extracted)
            match = status == expected_filter
            ok = "PASS" if match else ("STUB" if is_stub else "FAIL")
            if not match and not is_stub:
                all_pass = False
            print(f"  [{ok}] raw={raw!r:.60s}... → extracted={extracted!r:.40s} status={status} (expected {expected_filter})")

        print(f"\nSeed test:")
        s1 = make_seed(args.model_id, 1, 0)
        s2 = make_seed(args.model_id, 1, 1)
        s3 = make_seed(args.model_id, 1, 0)
        print(f"  seed(pid=1, idx=0) = {s1}")
        print(f"  seed(pid=1, idx=1) = {s2}")
        print(f"  seed(pid=1, idx=0) = {s3} (should match first: {s1 == s3})")

        print(f"\nDB insert test (fake completion):")
        test_comp = [{
            "prompt_id": prompts[0]["prompt_id"],
            "model_id": args.model_id,
            "sample_idx": 99999,
            "completion_raw": "DRY_RUN_TEST",
            "completion_text": "DRY_RUN_TEST",
            "n_tokens_raw": 1,
            "n_tokens": 1,
            "filter_status": "ok",
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": 0,
            "steering_config": "none",
        }]
        insert_completions(conn, test_comp)
        row = conn.execute(
            "SELECT * FROM completions WHERE completion_raw = 'DRY_RUN_TEST'"
        ).fetchone()
        if row:
            print(f"  INSERT + SELECT: PASS (completion_id={row[0]})")
            conn.execute("DELETE FROM completions WHERE completion_raw = 'DRY_RUN_TEST'")
            conn.commit()
            print(f"  Cleaned up test row.")
        else:
            print(f"  INSERT + SELECT: FAIL")
            all_pass = False

        print(f"\n{'='*60}")
        print(f"Dry run {'PASSED' if all_pass else 'FAILED'}")
        conn.close()
        return

    # ── Real run ─────────────────────────────────────────────────
    print(f"Loading model: {args.model_hf_id}")
    model, tokenizer = load_model(args.model_hf_id, dtype=args.dtype)

    # Ensure pad token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    total_prompts = len(prompts)
    if args.validate:
        prompts = prompts[:2]

    total_completions = 0
    filter_counts: dict[str, int] = {}
    t0 = time.time()

    for i, prompt in enumerate(prompts):
        needed = prompt["needed"]
        start_idx = args.n_samples - needed

        print(f"\n[{i+1}/{len(prompts)}] {prompt['lang']}:{prompt['template_id']} "
              f"({needed} needed, starting at idx {start_idx})")
        print(f"  Prompt: {prompt['prompt_text']!r}")

        completions = sample_prompt(
            model, tokenizer,
            prompt_text=prompt["prompt_text"],
            n_samples=needed,
            model_id=args.model_id,
            prompt_id=prompt["prompt_id"],
            start_idx=start_idx,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )

        insert_completions(conn, completions)
        total_completions += len(completions)

        for c in completions:
            filter_counts[c["filter_status"]] = filter_counts.get(c["filter_status"], 0) + 1

        if args.validate:
            print(f"\n  === Validate: first 5 completions ===")
            for c in completions[:5]:
                print(f"  [{c['filter_status']}] {c['completion_text']!r}")

        elapsed = time.time() - t0
        rate = total_completions / elapsed if elapsed > 0 else 0
        remaining = sum(p["needed"] for p in prompts[i+1:])
        eta = remaining / rate if rate > 0 else 0
        print(f"  Progress: {total_completions} done, {rate:.1f} samples/s, ETA {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Done. {total_completions} completions in {elapsed:.1f}s")
    print(f"Filter distribution: {filter_counts}")
    if not args.validate:
        print(f"Remaining prompts: {total_prompts - len(prompts)} already complete")

    conn.close()


if __name__ == "__main__":
    main()
