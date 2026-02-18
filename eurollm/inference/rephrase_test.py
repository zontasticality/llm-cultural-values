"""Prompt sensitivity experiment: compare original vs rephrased question distributions.

Loads rephrasings.json, constructs prompts with variant text but original options,
runs forward + reversed logprob extraction, and outputs per-variant distributions.

Reuses TokenMap, extract_logprobs_for_question, renormalize, remap_reversed from
extract_logprobs.py, and clean_label / formatting logic from prompt_templates.py.

Usage:
    python eurollm/rephrase_test.py \
        --model_id HPLT/hplt2c_eng_checkpoints \
        --output eurollm/results/rephrase_test_hplt2c_eng.parquet \
        [--validate]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from inference.extract_logprobs import (
    TokenMap,
    extract_logprobs_for_question,
    renormalize,
    remap_reversed,
)
from prompting.prompt_templates import clean_label, clean_text


def build_prompt_from_variant(question: dict, variant_text: str, reverse: bool) -> dict:
    """Build a prompt using variant text but original options from questions.json.

    Mirrors the logic of format_prompt() / _format_standard() / _format_likert10()
    but substitutes variant_text for the original question text.

    Args:
        question: A question dict from questions.json (must have eng translation).
        variant_text: The rephrased question text to use.
        reverse: If True, reverse option ordering for position-bias control.

    Returns:
        dict with keys: prompt, valid_values, value_map, is_likert10
    """
    trans = question["translations"]["eng"]
    answer_cue = trans.get("answer_cue", "Answer")
    options = trans["options"]
    is_likert10 = question["response_type"] == "likert10"
    n = len(options)

    # Clean the variant text the same way we clean original text
    text = clean_text(variant_text)

    if is_likert10:
        return _build_likert10(text, answer_cue, options, reverse)
    else:
        return _build_standard(text, answer_cue, options, reverse, n)


def _build_standard(text, answer_cue, options, reverse, n):
    """Standard (non-likert10) prompt with numbered options."""
    if reverse:
        reversed_options = list(reversed(options))
        value_map = {}
        lines = []
        for i, opt in enumerate(reversed_options, 1):
            label = clean_label(opt["label"])
            original_value = str(opt["value"])
            value_map[str(i)] = original_value
            if label:
                lines.append(f"{i}. {label}")
            else:
                lines.append(f"{i}.")
        valid_values = [str(i) for i in range(1, n + 1)]
    else:
        value_map = {}
        lines = []
        for opt in options:
            pos = opt["value"]
            label = clean_label(opt["label"])
            value_map[str(pos)] = str(pos)
            if label:
                lines.append(f"{pos}. {label}")
            else:
                lines.append(f"{pos}.")
        valid_values = [str(opt["value"]) for opt in options]

    prompt = f"{text}\n" + "\n".join(lines) + f"\n{answer_cue}: "
    return {
        "prompt": prompt,
        "valid_values": valid_values,
        "value_map": value_map,
        "is_likert10": False,
    }


def _build_likert10(text, answer_cue, options, reverse):
    """Likert10 prompt with anchors at endpoints."""
    opts_by_value = {opt["value"]: opt for opt in options}

    left_anchor = ""
    right_anchor = ""
    left_label = clean_label(opts_by_value.get(1, {}).get("label", ""))
    right_label = clean_label(opts_by_value.get(10, {}).get("label", ""))
    if left_label and not left_label.isdigit():
        left_anchor = left_label
    if right_label and not right_label.isdigit():
        right_anchor = right_label

    if reverse:
        left_anchor, right_anchor = right_anchor, left_anchor
        value_map = {str(i): str(11 - i) for i in range(1, 11)}
    else:
        value_map = {str(i): str(i) for i in range(1, 11)}

    lines = []
    for i in range(1, 11):
        if i == 1 and left_anchor:
            lines.append(f"1. {left_anchor}")
        elif i == 10 and right_anchor:
            lines.append(f"10. {right_anchor}")
        else:
            lines.append(f"{i}.")

    prompt = f"{text}\n" + "\n".join(lines) + f"\n{answer_cue}: "
    valid_values = [str(i) for i in range(1, 11)]

    return {
        "prompt": prompt,
        "valid_values": valid_values,
        "value_map": value_map,
        "is_likert10": True,
    }


def process_variant(
    model, tokenizer, token_map, question, variant_text, variant_id,
    variant_type="original",
):
    """Process one variant: forward + reversed, average, compute bias.

    Returns a list of row dicts (one per response value).
    """
    # Forward
    fwd = build_prompt_from_variant(question, variant_text, reverse=False)
    logprobs_fwd, p_valid_fwd = extract_logprobs_for_question(
        model, tokenizer, token_map, fwd
    )
    probs_fwd = renormalize(logprobs_fwd)

    # Reversed
    rev = build_prompt_from_variant(question, variant_text, reverse=True)
    logprobs_rev, p_valid_rev = extract_logprobs_for_question(
        model, tokenizer, token_map, rev
    )
    probs_rev_raw = renormalize(logprobs_rev)
    probs_rev = remap_reversed(probs_rev_raw, rev["value_map"])

    # Collect all semantic values
    all_values = sorted(
        set(list(probs_fwd.keys()) + list(probs_rev.keys())),
        key=lambda x: int(x),
    )

    # Position bias magnitude
    pos_bias = max(
        abs(probs_fwd.get(v, 0.0) - probs_rev.get(v, 0.0)) for v in all_values
    ) if all_values else 0.0

    rows = []
    for v in all_values:
        pf = probs_fwd.get(v, 0.0)
        pr = probs_rev.get(v, 0.0)
        rows.append({
            "question_id": question["canonical_id"],
            "variant_id": variant_id,
            "variant_type": variant_type,
            "response_value": int(v),
            "prob_forward": pf,
            "prob_reversed": pr,
            "prob_averaged": (pf + pr) / 2.0,
            "p_valid_forward": p_valid_fwd,
            "p_valid_reversed": p_valid_rev,
            "position_bias_magnitude": pos_bias,
        })

    return rows


def validate(model, tokenizer, token_map, rephrasings, questions_by_id):
    """Validate mode: process 3 questions with original + 1 variant each."""
    print("=" * 70)
    print("VALIDATION MODE — Rephrase Test")
    print("=" * 70)
    token_map.report()

    for entry in rephrasings[:3]:
        qid = entry["canonical_id"]
        question = questions_by_id.get(qid)
        if not question:
            print(f"\nWARNING: {qid} not found in questions.json, skipping")
            continue

        print(f"\n{'—'*60}")
        print(f"Question {qid} ({question['response_type']}) — issue: {entry['issue_type']}")
        print(f"{'—'*60}")

        # Original
        print(f"\n  [ORIGINAL] {entry['original_text'][:80]}...")
        rows_orig = process_variant(
            model, tokenizer, token_map, question,
            entry["original_text"], f"{qid}_original",
            variant_type="original",
        )
        if rows_orig:
            pv = rows_orig[0]["p_valid_forward"]
            bias = rows_orig[0]["position_bias_magnitude"]
            dist = {r["response_value"]: f'{r["prob_averaged"]:.3f}' for r in rows_orig}
            print(f"  P_valid(fwd): {pv:.4f}  Bias: {bias:.4f}")
            print(f"  Distribution: {dist}")

        # All variants
        for var in entry["variants"]:
            vtype = var.get("variant_type", "unknown")
            print(f"\n  [{vtype.upper()} {var['id']}] {var['text'][:80]}...")
            rows_var = process_variant(
                model, tokenizer, token_map, question,
                var["text"], var["id"],
                variant_type=vtype,
            )
            if rows_var:
                pv = rows_var[0]["p_valid_forward"]
                bias = rows_var[0]["position_bias_magnitude"]
                dist = {r["response_value"]: f'{r["prob_averaged"]:.3f}' for r in rows_var}
                print(f"  P_valid(fwd): {pv:.4f}  Bias: {bias:.4f}")
                print(f"  Distribution: {dist}")

    print(f"\n{'='*70}")
    print("Validation complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Prompt sensitivity experiment: original vs rephrased questions"
    )
    parser.add_argument(
        "--model_id", required=True,
        help="HuggingFace model ID (e.g. HPLT/hplt2c_eng_checkpoints)"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output parquet file path"
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Run validation mode (3 questions with diagnostics)"
    )
    parser.add_argument(
        "--rephrasings", default=None,
        help="Path to rephrasings.json (default: eurollm/rephrasings.json)"
    )
    parser.add_argument(
        "--questions", default=None,
        help="Path to questions.json (default: eurollm/questions.json)"
    )
    args = parser.parse_args()

    # Load rephrasings
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    rephrasings_path = args.rephrasings or str(PROJECT_ROOT / "data" / "rephrasings.json")
    with open(rephrasings_path) as f:
        rephrasings = json.load(f)
    print(f"Loaded {len(rephrasings)} rephrasing entries")

    # Load questions.json for option metadata
    questions_path = args.questions or str(PROJECT_ROOT / "data" / "questions.json")
    with open(questions_path) as f:
        data = json.load(f)
    questions_by_id = {q["canonical_id"]: q for q in data["questions"]}

    # Verify all rephrasings reference valid questions with English translations
    for entry in rephrasings:
        qid = entry["canonical_id"]
        if qid not in questions_by_id:
            print(f"ERROR: {qid} not found in questions.json")
            sys.exit(1)
        if "eng" not in questions_by_id[qid]["translations"]:
            print(f"ERROR: {qid} has no English translation")
            sys.exit(1)

    # Count total variants (including originals)
    n_variants = sum(1 + len(e["variants"]) for e in rephrasings)
    print(f"Total prompts to process: {n_variants} ({len(rephrasings)} originals + variants)")

    # Load model and tokenizer
    print(f"Loading model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"Model loaded on {model.device if hasattr(model, 'device') else 'auto-mapped devices'}")

    # Build token map
    token_map = TokenMap.from_tokenizer(tokenizer)

    if args.validate:
        validate(model, tokenizer, token_map, rephrasings, questions_by_id)
        return

    # Full run
    all_rows = []
    processed = 0

    for entry in rephrasings:
        qid = entry["canonical_id"]
        question = questions_by_id[qid]

        # Process original text
        rows = process_variant(
            model, tokenizer, token_map, question,
            entry["original_text"], f"{qid}_original",
            variant_type="original",
        )
        all_rows.extend(rows)
        processed += 1

        # Process each variant
        for var in entry["variants"]:
            rows = process_variant(
                model, tokenizer, token_map, question,
                var["text"], var["id"],
                variant_type=var.get("variant_type", "unknown"),
            )
            all_rows.extend(rows)
            processed += 1

        print(f"  {qid}: processed original + {len(entry['variants'])} variants")

    print(f"\nProcessed {processed} total prompts → {len(all_rows)} rows")

    # Save to parquet
    df = pd.DataFrame(all_rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}")

    # Summary
    print(f"\nSummary:")
    print(f"  Mean P_valid (fwd): {df['p_valid_forward'].mean():.4f}")
    print(f"  Mean P_valid (rev): {df['p_valid_reversed'].mean():.4f}")
    per_variant = df.groupby("variant_id").agg(
        p_valid=("p_valid_forward", "first"),
        bias=("position_bias_magnitude", "first"),
    )
    print(f"  P_valid range: [{per_variant['p_valid'].min():.4f}, {per_variant['p_valid'].max():.4f}]")
    print(f"  Bias range: [{per_variant['bias'].min():.4f}, {per_variant['bias'].max():.4f}]")


if __name__ == "__main__":
    main()
