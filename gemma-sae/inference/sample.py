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

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from db.load import load_unsampled_prompts
from db.schema import get_connection, init_db


# ── Model loading (reused from eurollm) ─────────────────────────

def load_model(model_id: str, dtype: str = "bf16"):
    """Load model and tokenizer with configurable precision."""
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


def extract_first_sentence(raw: str, tokenizer, max_extract_tokens: int = 80) -> str | None:
    """Extract the first sentence/clause from raw generation."""
    text = raw.strip()
    if not text:
        return None

    # Try sentence boundary
    m = SENTENCE_BOUNDARY.search(text)
    if m and len(tokenizer.encode(text[:m.start() + 1])) <= max_extract_tokens:
        return text[:m.start() + 1].strip()

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

    # Too short
    if len(words) < 3:
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
