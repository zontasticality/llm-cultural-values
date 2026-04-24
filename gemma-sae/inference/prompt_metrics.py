"""Compute prompt-level metrics: perplexity and next-token entropy.

For each (prompt, model) pair, runs a single forward pass to get:
  - prompt_logprob: sum of log-probs for all prompt tokens (excluding BOS)
  - prompt_ppl: perplexity = exp(-prompt_logprob / n_tokens)
  - prompt_n_tokens: number of tokens in the prompt
  - next_token_entropy: Shannon entropy of the next-token distribution
    at the completion boundary (bits)

Usage:
    PYTHONPATH=gemma-sae python -m inference.prompt_metrics \
        --db data/culture.db \
        --models gemma3_27b_pt gemma3_12b_pt eurollm22b \
        [--trimmed-only] [--validate]
"""

import argparse
import math
import time

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from analysis.constants import TRIMMED_VARIANT_MIN
from db.schema import get_connection, create_tables


def compute_metrics_batch(model, tokenizer, texts: list[str], device) -> list[dict]:
    """Compute perplexity and next-token entropy for a batch of prompts."""
    # Tokenize with padding
    encoded = tokenizer(
        texts, return_tensors="pt", padding=True, add_special_tokens=True,
    ).to(device)

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits  # (batch, seq_len, vocab_size)

    results = []
    for i in range(len(texts)):
        mask = attention_mask[i].bool()
        ids = input_ids[i][mask]
        seq_logits = logits[i][mask]
        n_tokens = int(mask.sum().item())

        if n_tokens < 2:
            results.append({
                "prompt_logprob": 0.0,
                "prompt_ppl": 1.0,
                "prompt_n_tokens": n_tokens,
                "next_token_entropy": None,
            })
            continue

        # Log-probs for prompt tokens (teacher-forced)
        # For token at position t, the logit at position t-1 predicts it
        shift_logits = seq_logits[:-1]  # (n_tokens-1, vocab)
        shift_labels = ids[1:]          # (n_tokens-1,)

        log_probs = torch.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)

        total_logprob = token_log_probs.sum().item()
        n_scored = len(token_log_probs)
        ppl = math.exp(-total_logprob / n_scored) if n_scored > 0 else 1.0

        # Next-token entropy: distribution at the last token position
        last_logits = seq_logits[-1]  # (vocab_size,)
        last_probs = torch.softmax(last_logits, dim=-1)
        # Shannon entropy in bits
        log2_probs = torch.log2(last_probs + 1e-12)
        entropy = -(last_probs * log2_probs).sum().item()

        results.append({
            "prompt_logprob": total_logprob,
            "prompt_ppl": ppl,
            "prompt_n_tokens": n_tokens,
            "next_token_entropy": entropy,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Compute prompt perplexity and entropy metrics")
    parser.add_argument("--db", required=True)
    parser.add_argument("--models", nargs="+", required=True,
                        help="Model IDs to evaluate (looked up in models table for HF ID)")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "int4", "int8"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--trimmed-only", action="store_true",
                        help=f"Only compute metrics for trimmed prompts (variant_idx >= {TRIMMED_VARIANT_MIN})")
    parser.add_argument("--validate", action="store_true",
                        help="Process 5 prompts per model, print results, don't save")
    args = parser.parse_args()

    conn = get_connection(args.db)
    create_tables(conn)

    # Load prompts
    where = ""
    if args.trimmed_only:
        where = f" WHERE variant_idx >= {TRIMMED_VARIANT_MIN}"
    prompts = conn.execute(
        f"SELECT prompt_id, template_id, lang, prompt_text FROM prompts{where} ORDER BY prompt_id"
    ).fetchall()
    print(f"Loaded {len(prompts)} prompts")

    for model_id in args.models:
        # Get HF ID
        row = conn.execute(
            "SELECT model_hf_id FROM models WHERE model_id = ?", (model_id,)
        ).fetchone()
        if not row:
            print(f"  WARNING: model_id '{model_id}' not found in DB, skipping")
            continue
        hf_id = row[0]

        # Check which prompts already have metrics
        existing = set()
        for r in conn.execute(
            "SELECT prompt_id FROM prompt_metrics WHERE model_id = ?", (model_id,)
        ).fetchall():
            existing.add(r[0])

        todo = [(pid, tid, lang, text) for pid, tid, lang, text in prompts
                if pid not in existing]

        if not todo:
            print(f"  {model_id}: all prompts already computed, skipping")
            continue

        if args.validate:
            todo = todo[:5]

        print(f"\n{'='*60}")
        print(f"Model: {model_id} ({hf_id})")
        print(f"  Prompts to compute: {len(todo)}")
        print(f"  Loading model...")

        # Load model
        if args.dtype == "bf16":
            model = AutoModelForCausalLM.from_pretrained(
                hf_id, torch_dtype=torch.bfloat16, device_map="auto",
            )
        elif args.dtype == "int4":
            from transformers import BitsAndBytesConfig
            model = AutoModelForCausalLM.from_pretrained(
                hf_id, device_map="auto",
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                ),
            )
        else:
            raise ValueError(f"Unsupported dtype: {args.dtype}")

        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        device = next(model.parameters()).device
        print(f"  Model loaded on {device}")

        t0 = time.time()
        total_done = 0

        for batch_start in range(0, len(todo), args.batch_size):
            batch = todo[batch_start:batch_start + args.batch_size]
            texts = [text for _, _, _, text in batch]

            metrics = compute_metrics_batch(model, tokenizer, texts, device)

            rows_to_insert = []
            for (pid, tid, lang, text), m in zip(batch, metrics):
                rows_to_insert.append({
                    "prompt_id": pid,
                    "model_id": model_id,
                    "prompt_ppl": m["prompt_ppl"],
                    "prompt_logprob": m["prompt_logprob"],
                    "prompt_n_tokens": m["prompt_n_tokens"],
                    "next_token_entropy": m["next_token_entropy"],
                })

                if args.validate:
                    print(f"  [{lang}:{tid}] {text!r}")
                    print(f"    ppl={m['prompt_ppl']:.2f}  logprob={m['prompt_logprob']:.2f}  "
                          f"n_tok={m['prompt_n_tokens']}  entropy={m['next_token_entropy']:.2f} bits")

            if not args.validate:
                conn.executemany(
                    "INSERT OR IGNORE INTO prompt_metrics "
                    "(prompt_id, model_id, prompt_ppl, prompt_logprob, "
                    " prompt_n_tokens, next_token_entropy) "
                    "VALUES (:prompt_id, :model_id, :prompt_ppl, :prompt_logprob, "
                    " :prompt_n_tokens, :next_token_entropy)",
                    rows_to_insert,
                )
                conn.commit()

            total_done += len(batch)
            elapsed = time.time() - t0
            rate = total_done / elapsed if elapsed > 0 else 0
            print(f"  [{total_done}/{len(todo)}] {rate:.1f} prompts/s")

        # Free model memory before loading next
        del model
        del tokenizer
        torch.cuda.empty_cache()

        elapsed = time.time() - t0
        print(f"  Done: {total_done} prompts in {elapsed:.1f}s")

    conn.close()
    print("\nAll models complete.")


if __name__ == "__main__":
    main()
