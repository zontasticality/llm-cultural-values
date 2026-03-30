"""Local LLM classification using transformers for GPU inference.

Runs Gemma 3 27B IT (or smaller) locally on GPU to classify completions.
No API costs, no rate limits.

Usage:
    PYTHONPATH=gemma-sae python -m classify.classify_local \
        --db data/culture.db \
        --model google/gemma-3-27b-it \
        --classifier-name gemma3_27b_it \
        [--batch-size 8] [--limit 100] [--validate]
"""

import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from classify.prompts import (
    CLASSIFIER_SYSTEM,
    make_classifier_prompt,
    parse_classification,
)
from db.load import load_unclassified_completions
from db.schema import get_connection


def classify_batch(model, tokenizer, completions, max_new_tokens=128):
    """Classify a batch of completions. Returns list of parsed results."""
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

    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True,
        max_length=2048,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated tokens (not the prompt)
    results = []
    for i, (comp, output) in enumerate(zip(completions, outputs)):
        prompt_len = inputs["input_ids"][i].shape[0]
        generated = output[prompt_len:]
        raw = tokenizer.decode(generated, skip_special_tokens=True).strip()

        parsed = parse_classification(raw)
        if not parsed:
            # Try extracting JSON from response
            try:
                start = raw.index("{")
                end = raw.rindex("}") + 1
                parsed = parse_classification(raw[start:end])
            except (ValueError, TypeError):
                pass

        if parsed:
            results.append({
                "completion_id": comp["completion_id"],
                "raw_response": raw,
                **parsed,
            })

    return results


def insert_classifications(conn, classifier_name, results):
    """Insert classification results into the DB."""
    conn.executemany(
        "INSERT OR IGNORE INTO classifications "
        "(completion_id, classifier_model, content_category, "
        " dim_indiv_collect, dim_trad_secular, dim_surv_selfexpr, raw_response) "
        "VALUES (:completion_id, :classifier, :content_category, "
        " :dim_indiv_collect, :dim_trad_secular, :dim_surv_selfexpr, :raw_response)",
        [{"classifier": classifier_name, **r} for r in results],
    )
    conn.commit()


def main():
    parser = argparse.ArgumentParser(description="Local transformers classification")
    parser.add_argument("--db", required=True)
    parser.add_argument("--model", default="google/gemma-3-27b-it")
    parser.add_argument("--classifier-name", default="gemma3_27b_it")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    conn = get_connection(args.db)

    completions = load_unclassified_completions(
        conn, args.classifier_name, limit=args.limit,
    )

    if not completions:
        print("No unclassified completions. Nothing to do.")
        conn.close()
        return

    print(f"Found {len(completions)} unclassified completions for {args.classifier_name}")

    if args.validate:
        completions = completions[:10]
        print(f"Validate mode: classifying {len(completions)} samples")

    # Load model
    print(f"Loading {args.model} (bf16)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    print(f"Model loaded on {model.device}")

    total_classified = 0
    total_failed = 0
    t0 = time.time()

    for batch_start in range(0, len(completions), args.batch_size):
        batch = completions[batch_start:batch_start + args.batch_size]

        try:
            results = classify_batch(model, tokenizer, batch, args.max_new_tokens)
        except torch.cuda.OutOfMemoryError:
            # Fall back to one-at-a-time
            print(f"  OOM on batch of {len(batch)}, falling back to single")
            torch.cuda.empty_cache()
            results = []
            for comp in batch:
                try:
                    r = classify_batch(model, tokenizer, [comp], args.max_new_tokens)
                    results.extend(r)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    total_failed += 1

        total_classified += len(results)
        total_failed += len(batch) - len(results)

        if not args.validate:
            insert_classifications(conn, args.classifier_name, results)

        elapsed = time.time() - t0
        rate = total_classified / elapsed if elapsed > 0 else 0
        remaining = len(completions) - batch_start - len(batch)
        eta = remaining / rate if rate > 0 else 0

        print(f"  [{batch_start + len(batch):>6}/{len(completions)}] "
              f"{total_classified} ok, {total_failed} failed, "
              f"{rate:.1f}/s, ETA {eta/3600:.1f}h")

    elapsed = time.time() - t0
    print(f"\nDone. {total_classified}/{len(completions)} classified "
          f"in {elapsed:.1f}s ({total_classified/elapsed:.1f}/s)")
    print(f"Parse failures: {total_failed}")

    if args.validate:
        print(f"\n{'='*60}")
        print("Re-running for display...")
        results = []
        for comp in completions:
            r = classify_batch(model, tokenizer, [comp], args.max_new_tokens)
            results.extend(r)
        for r in results:
            comp = next(c for c in completions if c["completion_id"] == r["completion_id"])
            print(f"  [{comp['lang']}:{comp['template_id']}] "
                  f"{comp['completion_text'][:80]!r}")
            print(f"    -> {r['content_category']} IC={r['dim_indiv_collect']} "
                  f"TS={r['dim_trad_secular']} SS={r['dim_surv_selfexpr']}")

    conn.close()


if __name__ == "__main__":
    main()
