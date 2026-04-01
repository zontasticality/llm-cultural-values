"""Local LLM classification using vLLM with logprob extraction.

Generates JSON classifications, then extracts probability distributions
over dimension scores (1-5) and content categories from token-level
logprobs. This gives continuous expected values and uncertainty estimates
instead of point-estimate Likert scores.

Usage:
    PYTHONPATH=gemma-sae python -m classify.classify_local \
        --db data/culture.db \
        --model google/gemma-3-27b-it \
        --classifier-name gemma3_27b_it \
        [--batch-size 256] [--limit 100] [--validate]
"""

import argparse
import json
import math
import time

from analysis.constants import CONTENT_CATEGORIES
from classify.prompts import (
    CLASSIFIER_SYSTEM,
    make_classifier_prompt,
    parse_classification,
)
from db.load import load_unclassified_completions
from db.schema import get_connection, migrate_classifications_probs


# ── vLLM model loading ─────────────────────────────────────────

def load_classifier(model_hf_id: str, dtype: str = "bf16"):
    """Load classifier model with vLLM."""
    from vllm import LLM

    vllm_dtype = {"bf16": "bfloat16", "int4": "auto", "int8": "auto"}[dtype]
    llm = LLM(
        model=model_hf_id,
        dtype=vllm_dtype,
        max_model_len=2048,
        gpu_memory_utilization=0.90,
    )
    tokenizer = llm.get_tokenizer()
    return llm, tokenizer


# ── Token ID helpers ────────────────────────────────────────────

def get_digit_token_ids(tokenizer) -> dict[int, set[int]]:
    """Map each digit 1-5 to its possible token IDs (with/without leading space)."""
    digit_ids = {}
    for d in range(1, 6):
        ids = set()
        for text in [str(d), f" {d}"]:
            toks = tokenizer.encode(text, add_special_tokens=False)
            ids.add(toks[-1])
        digit_ids[d] = ids
    return digit_ids


def get_category_first_tokens(tokenizer) -> dict[str, set[int]]:
    """Map each content category to plausible first-token IDs.

    After '"content_category": "', the first generated token determines the
    category. We check several contexts to capture tokenizer variations.
    """
    cat_tokens = {}
    for cat in CONTENT_CATEGORIES:
        ids = set()
        for prefix in [f'"{cat}"', f' "{cat}"', cat]:
            toks = tokenizer.encode(prefix, add_special_tokens=False)
            # Find first token that starts the category name (skip quotes/spaces)
            for tid in toks:
                decoded = tokenizer.decode([tid]).strip().strip('"').lower()
                if decoded and cat.lower().startswith(decoded):
                    ids.add(tid)
                    break
        cat_tokens[cat] = ids
    return cat_tokens


# ── Logprob extraction ─────────────────────────────────────────

DIM_KEYS = ["dim_indiv_collect", "dim_trad_secular", "dim_surv_selfexpr"]


def extract_dim_probs(completion_output, digit_ids: dict[int, set[int]]) -> dict[str, list[float] | None]:
    """Extract P(1..5) for each dimension from output logprobs.

    Strategy: the only digit tokens 1-5 in the JSON output are dimension
    scores. Find them in order — they map to IC, TS, SS respectively.
    """
    token_ids = list(completion_output.token_ids)
    logprobs_list = completion_output.logprobs or []

    all_digit_ids = set()
    for ids in digit_ids.values():
        all_digit_ids.update(ids)

    # Find positions where a digit 1-5 was generated
    digit_positions = [i for i, tid in enumerate(token_ids) if tid in all_digit_ids]

    result = {}
    for dim_idx, dk in enumerate(DIM_KEYS):
        if dim_idx >= len(digit_positions):
            result[dk] = None
            continue

        pos = digit_positions[dim_idx]
        if pos >= len(logprobs_list) or logprobs_list[pos] is None:
            result[dk] = None
            continue

        lp = logprobs_list[pos]

        probs = []
        for d in range(1, 6):
            p = 0.0
            for tid in digit_ids[d]:
                if tid in lp:
                    p += math.exp(lp[tid].logprob)
            probs.append(p)

        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
        result[dk] = probs

    return result


def extract_cat_probs(
    completion_output,
    tokenizer,
    cat_first_tokens: dict[str, set[int]],
) -> dict[str, float] | None:
    """Extract approximate P(category) from first-token logprobs.

    After '"content_category": "', the first category-discriminating token
    determines the category. We find that position and read logprobs.
    """
    token_ids = list(completion_output.token_ids)
    logprobs_list = completion_output.logprobs or []
    decoded = [tokenizer.decode([tid]) for tid in token_ids]

    all_cat_tids = set()
    for ids in cat_first_tokens.values():
        all_cat_tids.update(ids)

    # Build running text to find the category value position
    running = ""
    for i, tok_text in enumerate(decoded):
        running += tok_text
        # Look for the opening quote of the category value
        if '"content_category":' in running or "'content_category':" in running:
            # Scan forward from here for a token matching a category
            for j in range(i, min(len(token_ids), i + 5)):
                if token_ids[j] in all_cat_tids and j < len(logprobs_list) and logprobs_list[j] is not None:
                    lp = logprobs_list[j]
                    probs = {}
                    for cat in CONTENT_CATEGORIES:
                        p = 0.0
                        for tid in cat_first_tokens[cat]:
                            if tid in lp:
                                p += math.exp(lp[tid].logprob)
                        probs[cat] = p
                    total = sum(probs.values())
                    if total > 0:
                        return {k: v / total for k, v in probs.items()}
                    return None
            return None

    return None


# ── Prompt building ─────────────────────────────────────────────

def build_prompts(completions: list[dict], tokenizer) -> list[str]:
    """Build chat-templated prompts for classification."""
    prompts = []
    for comp in completions:
        user_msg = make_classifier_prompt(
            comp["completion_text"], comp["lang"], comp["template_id"],
        )
        messages = [
            {"role": "user", "content": CLASSIFIER_SYSTEM + "\n\n" + user_msg},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        prompts.append(text)
    return prompts


# ── DB insertion ────────────────────────────────────────────────

def insert_classifications(conn, classifier_name: str, results: list[dict]):
    """Batch insert classifications with optional probability distributions."""
    conn.executemany(
        "INSERT OR IGNORE INTO classifications "
        "(completion_id, classifier_model, content_category, "
        " dim_indiv_collect, dim_trad_secular, dim_surv_selfexpr, "
        " raw_response, dim_ic_probs, dim_ts_probs, dim_ss_probs, cat_probs) "
        "VALUES (:completion_id, :classifier, :content_category, "
        " :dim_indiv_collect, :dim_trad_secular, :dim_surv_selfexpr, "
        " :raw_response, :dim_ic_probs, :dim_ts_probs, :dim_ss_probs, :cat_probs)",
        [{"classifier": classifier_name, **r} for r in results],
    )
    conn.commit()


# ── Main pipeline ───────────────────────────────────────────────

def classify_batch_vllm(
    llm,
    tokenizer,
    completions: list[dict],
    digit_ids: dict[int, set[int]],
    cat_first_tokens: dict[str, set[int]],
    max_new_tokens: int = 200,
) -> list[dict]:
    """Classify a batch via vLLM with logprob extraction."""
    from vllm import SamplingParams

    prompts = build_prompts(completions, tokenizer)
    params = SamplingParams(
        temperature=0,
        max_tokens=max_new_tokens,
        logprobs=10,
    )

    outputs = llm.generate(prompts, params)

    results = []
    for comp, output in zip(completions, outputs):
        raw = output.outputs[0].text.strip()

        # Parse JSON classification (greedy picks)
        parsed = parse_classification(raw)
        if not parsed:
            # Try extracting JSON substring
            try:
                start = raw.index("{")
                end = raw.rindex("}") + 1
                parsed = parse_classification(raw[start:end])
            except (ValueError, TypeError):
                pass

        if not parsed:
            continue

        # Extract logprob distributions
        dim_probs = extract_dim_probs(output.outputs[0], digit_ids)
        cat_probs = extract_cat_probs(output.outputs[0], tokenizer, cat_first_tokens)

        results.append({
            "completion_id": comp["completion_id"],
            "raw_response": raw,
            **parsed,
            "dim_ic_probs": json.dumps(dim_probs.get("dim_indiv_collect")) if dim_probs.get("dim_indiv_collect") else None,
            "dim_ts_probs": json.dumps(dim_probs.get("dim_trad_secular")) if dim_probs.get("dim_trad_secular") else None,
            "dim_ss_probs": json.dumps(dim_probs.get("dim_surv_selfexpr")) if dim_probs.get("dim_surv_selfexpr") else None,
            "cat_probs": json.dumps(cat_probs) if cat_probs else None,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Local vLLM classification with logprob extraction")
    parser.add_argument("--db", required=True)
    parser.add_argument("--model", default="google/gemma-3-27b-it")
    parser.add_argument("--classifier-name", default="gemma3_27b_it")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "int4", "int8"])
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Completions per vLLM batch (vLLM handles internal scheduling)")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    conn = get_connection(args.db)
    migrate_classifications_probs(conn)

    completions = load_unclassified_completions(
        conn, args.classifier_name, limit=args.limit,
    )
    if not completions:
        print("No unclassified completions. Nothing to do.")
        conn.close()
        return

    print(f"Found {len(completions)} unclassified completions for {args.classifier_name}")

    if args.validate:
        completions = completions[:20]
        print(f"Validate mode: classifying {len(completions)} samples")

    # Load model
    print(f"Loading {args.model} with vLLM...")
    llm, tokenizer = load_classifier(args.model, dtype=args.dtype)

    # Precompute token ID maps
    digit_ids = get_digit_token_ids(tokenizer)
    cat_first_tokens = get_category_first_tokens(tokenizer)

    print(f"Digit token IDs: { {d: list(ids) for d, ids in digit_ids.items()} }")
    print(f"Category first tokens: { {c: list(ids) for c, ids in cat_first_tokens.items()} }")

    total_classified = 0
    total_failed = 0
    probs_extracted = 0
    t0 = time.time()

    for batch_start in range(0, len(completions), args.batch_size):
        batch = completions[batch_start:batch_start + args.batch_size]

        results = classify_batch_vllm(
            llm, tokenizer, batch, digit_ids, cat_first_tokens,
            max_new_tokens=args.max_new_tokens,
        )

        total_classified += len(results)
        total_failed += len(batch) - len(results)
        probs_extracted += sum(1 for r in results if r.get("dim_ic_probs"))

        if not args.validate:
            insert_classifications(conn, args.classifier_name, results)

        elapsed = time.time() - t0
        rate = total_classified / elapsed if elapsed > 0 else 0
        remaining = len(completions) - batch_start - len(batch)
        eta = remaining / rate if rate > 0 else 0

        print(f"  [{batch_start + len(batch):>6}/{len(completions)}] "
              f"{total_classified} ok, {total_failed} failed, "
              f"{probs_extracted} w/ probs, "
              f"{rate:.1f}/s, ETA {eta / 60:.1f}m")

    elapsed = time.time() - t0
    print(f"\nDone. {total_classified}/{len(completions)} classified "
          f"in {elapsed:.1f}s ({total_classified / elapsed:.1f}/s)")
    print(f"Parse failures: {total_failed}")
    print(f"Logprob extraction: {probs_extracted}/{total_classified} "
          f"({probs_extracted / total_classified * 100:.0f}%)" if total_classified else "")

    if args.validate:
        print(f"\n{'=' * 72}")
        print("Validation results:\n")
        for r in results:
            comp = next(c for c in completions if c["completion_id"] == r["completion_id"])
            print(f"  [{comp['lang']}:{comp['template_id']}] "
                  f"{comp['completion_text'][:70]!r}")
            print(f"    greedy: {r['content_category']} "
                  f"IC={r['dim_indiv_collect']} TS={r['dim_trad_secular']} SS={r['dim_surv_selfexpr']}")

            if r.get("dim_ic_probs"):
                ic = json.loads(r["dim_ic_probs"])
                ts = json.loads(r["dim_ts_probs"])
                ss = json.loads(r["dim_ss_probs"])
                ic_ev = sum((i + 1) * p for i, p in enumerate(ic))
                ts_ev = sum((i + 1) * p for i, p in enumerate(ts))
                ss_ev = sum((i + 1) * p for i, p in enumerate(ss))
                print(f"    probs:  IC={[f'{p:.2f}' for p in ic]} E={ic_ev:.2f}")
                print(f"            TS={[f'{p:.2f}' for p in ts]} E={ts_ev:.2f}")
                print(f"            SS={[f'{p:.2f}' for p in ss]} E={ss_ev:.2f}")

            if r.get("cat_probs"):
                cp = json.loads(r["cat_probs"])
                top3 = sorted(cp.items(), key=lambda x: -x[1])[:3]
                print(f"    cat_p:  {', '.join(f'{c}={p:.2f}' for c, p in top3)}")

            print()

    conn.close()


if __name__ == "__main__":
    main()
