"""Extract survey response distributions from base language models via logprobs.

Runs EVS survey questions through a causal LM, extracts next-token logprobs
for valid response digits, and outputs per-question probability distributions
with position-bias debiasing (forward + reversed ordering, or K-permutation averaging).

Requirements (install in venv or via module on GPU node):
    torch, transformers, numpy, pandas, pyarrow

Usage:
    python eurollm/inference/extract_logprobs.py \
        --model_id HPLT/hplt2c_eng_checkpoints \
        --lang eng \
        --output eurollm/results/hplt2c_eng_checkpoints_eng.parquet \
        [--validate] [--permutations 6] [--prompt-config config.json] [--dtype bf16]
"""

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompting.prompt_templates import (
    format_prompt,
    format_prompt_permuted,
    format_prompt_custom,
)


@dataclass
class TokenMap:
    """Pre-computed token ID mappings for a tokenizer.

    Attributes:
        digit_tokens: mapping from digit string to set of token IDs
                      (bare + space-prefixed variants)
        zero_tokens: set of token IDs for "0" (for two-step "10" resolution)
        terminator_tokens: set of token IDs for answer-terminating tokens
    """
    digit_tokens: dict[str, set[int]] = field(default_factory=dict)
    zero_tokens: set[int] = field(default_factory=set)
    terminator_tokens: set[int] = field(default_factory=set)

    @classmethod
    def from_tokenizer(cls, tokenizer) -> "TokenMap":
        tm = cls()
        vocab = tokenizer.get_vocab()

        # Initialize digit sets with direct encoding (bare + space-prefixed)
        for d in range(0, 10):
            digit_str = str(d)
            ids = set()
            for variant in [digit_str, f" {d}"]:
                encoded = tokenizer.encode(variant, add_special_tokens=False)
                if len(encoded) == 1:
                    ids.add(encoded[0])
            if d >= 1:
                tm.digit_tokens[digit_str] = ids
            else:
                tm.zero_tokens = ids

        # Single pass over vocab to find byte-fallback tokens (e.g. <0x31> for "1")
        # that also decode to bare digits
        target_digits = {str(d) for d in range(0, 10)}
        for tok_id in vocab.values():
            decoded = tokenizer.decode([tok_id])
            stripped = decoded.strip()
            if stripped in target_digits and len(decoded) <= 2:
                if stripped == "0":
                    tm.zero_tokens.add(tok_id)
                else:
                    tm.digit_tokens[stripped].add(tok_id)

        # Terminator tokens: EOS, newline, space, period, comma
        for t in ["\n", " ", ".", ",", "\n\n"]:
            encoded = tokenizer.encode(t, add_special_tokens=False)
            if len(encoded) == 1:
                tm.terminator_tokens.add(encoded[0])
        if tokenizer.eos_token_id is not None:
            tm.terminator_tokens.add(tokenizer.eos_token_id)

        return tm

    def report(self):
        """Print diagnostic info about token mappings."""
        print("=== Token Map Report ===")
        for d in range(1, 10):
            ids = self.digit_tokens.get(str(d), set())
            print(f"  Digit '{d}': {len(ids)} token(s) — IDs: {sorted(ids)}")
        print(f"  Zero '0': {len(self.zero_tokens)} token(s) — IDs: {sorted(self.zero_tokens)}")
        print(f"  Terminators: {len(self.terminator_tokens)} token(s) — IDs: {sorted(self.terminator_tokens)}")
        # Verify all digits 1-9 have at least one token
        missing = [str(d) for d in range(1, 10) if not self.digit_tokens.get(str(d))]
        if missing:
            print(f"  WARNING: No tokens found for digits: {missing}")
        else:
            print("  All digits 1-9 have at least one token.")
        print()


def extract_logprobs_for_question(
    model,
    tokenizer,
    token_map: TokenMap,
    formatted: dict,
    chat_template: bool = False,
) -> tuple[dict[str, float], float]:
    """Extract logprobs for valid response tokens at the answer position.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        token_map: Pre-computed token ID mappings.
        formatted: Output of format_prompt() with keys: prompt, valid_values, is_likert10.
        chat_template: If True, wrap prompt with tokenizer.apply_chat_template().

    Returns:
        (logprobs, p_valid) where logprobs maps value string to log-probability,
        and p_valid is total probability mass on valid tokens before renormalization.
    """
    prompt = formatted["prompt"]
    valid_values = formatted["valid_values"]
    is_likert10 = formatted["is_likert10"]

    if chat_template:
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)
    else:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]
    log_probs = torch.log_softmax(logits, dim=-1)

    result = {}
    for val_str in valid_values:
        if val_str == "10":
            continue  # handled via two-step below
        token_ids = token_map.digit_tokens.get(val_str, set())
        if token_ids:
            # Sum probabilities across token variants (bare + space-prefixed)
            prob_sum = sum(torch.exp(log_probs[tid]).item() for tid in token_ids)
            if prob_sum > 0:
                result[val_str] = np.log(prob_sum)

    # Two-step conditional resolution for "10"
    if is_likert10 and "1" in result:
        p_raw_1 = np.exp(result["1"])

        # Append bare "1" token for second forward pass
        bare_one_id = tokenizer.encode("1", add_special_tokens=False)[0]
        extended_ids = torch.cat([
            inputs["input_ids"],
            torch.tensor([[bare_one_id]], device=model.device),
        ], dim=1)

        # Handle attention mask if present
        if "attention_mask" in inputs:
            extended_mask = torch.cat([
                inputs["attention_mask"],
                torch.ones((1, 1), dtype=inputs["attention_mask"].dtype, device=model.device),
            ], dim=1)
        else:
            extended_mask = None

        with torch.no_grad():
            if extended_mask is not None:
                out2 = model(input_ids=extended_ids, attention_mask=extended_mask)
            else:
                out2 = model(input_ids=extended_ids)
        logits2 = out2.logits[0, -1, :]
        lp2 = torch.log_softmax(logits2, dim=-1)

        # P("0" | ..., "1") — model continues to form "10"
        p_zero = sum(torch.exp(lp2[tid]).item() for tid in token_map.zero_tokens)

        # P(terminator | ..., "1") — answer is complete at "1"
        p_term = sum(torch.exp(lp2[tid]).item() for tid in token_map.terminator_tokens)

        # Decompose P_raw("1") into P(answer=1) and P(answer=10)
        denom = p_zero + p_term
        if denom > 0.01:
            p_answer_1 = p_raw_1 * p_term / denom
            p_answer_10 = p_raw_1 * p_zero / denom
            if p_answer_1 > 0:
                result["1"] = np.log(p_answer_1)
            if p_answer_10 > 0:
                result["10"] = np.log(p_answer_10)

    # Compute p_valid: total probability mass on valid response tokens
    p_valid = sum(np.exp(lp) for lp in result.values())

    return result, p_valid


def renormalize(logprobs: dict[str, float]) -> dict[str, float]:
    """Convert logprobs to a normalized probability distribution."""
    if not logprobs:
        return {}
    values = list(logprobs.values())
    max_lp = max(values)
    # Subtract max for numerical stability
    exp_vals = {k: np.exp(v - max_lp) for k, v in logprobs.items()}
    total = sum(exp_vals.values())
    if total == 0:
        return {k: 0.0 for k in logprobs}
    return {k: v / total for k, v in exp_vals.items()}


def remap_reversed(probs: dict[str, float], value_map: dict[str, str]) -> dict[str, float]:
    """Remap reversed-order probabilities back to semantic values.

    Args:
        probs: Position-keyed probabilities from reversed prompt.
        value_map: Maps position string to semantic value string.

    Returns:
        Semantic-value-keyed probabilities.
    """
    remapped = {}
    for pos_str, prob in probs.items():
        semantic_val = value_map.get(pos_str, pos_str)
        remapped[semantic_val] = prob
    return remapped


def generate_permutations(n_options: int, k: int, question_id: str) -> list[list[int]]:
    """Generate K permutations of n_options items, seeded by question_id.

    Always includes identity (forward) as first and reversed as second.
    Additional permutations are random but deterministic per question.
    If n_options! < k, returns all n_options! unique permutations (capped).
    """
    import math
    identity = list(range(n_options))
    reversed_perm = list(reversed(identity))
    perms = [identity, reversed_perm]

    if k <= 2:
        return perms[:k]

    # Cap K at n! to avoid infinite loop for small option counts
    max_perms = math.factorial(n_options)
    k = min(k, max_perms)

    if k <= 2:
        return perms[:k]

    rng = random.Random(hash(question_id) & 0xFFFFFFFF)
    seen = {tuple(identity), tuple(reversed_perm)}
    while len(perms) < k:
        perm = list(range(n_options))
        rng.shuffle(perm)
        t = tuple(perm)
        if t not in seen:
            seen.add(t)
            perms.append(perm)

    return perms


def load_prompt_config(config_path: str) -> dict:
    """Load prompt format configuration from JSON file."""
    with open(config_path) as f:
        return json.load(f)


def get_format_kwargs(prompt_config: dict, response_type: str) -> dict:
    """Extract format keyword arguments from prompt config for a given response type."""
    cfg = dict(prompt_config.get("default", {}))
    overrides = prompt_config.get("overrides_by_response_type", {}).get(response_type, {})
    cfg.update(overrides)
    # Remove n_perms — handled separately
    cfg.pop("n_perms", None)
    return {
        "cue_style": cfg.get("cue_style", "lang"),
        "opt_format": cfg.get("opt_format", "numbered_dot"),
        "scale_hint": cfg.get("scale_hint", False),
        "embed_style": cfg.get("embed_style", "separate"),
    }


def process_question(
    model,
    tokenizer,
    token_map: TokenMap,
    question: dict,
    lang: str,
    n_permutations: int = 2,
    prompt_config: dict | None = None,
    chat_template: bool = False,
) -> list[dict]:
    """Process a single question with K-permutation debiasing.

    For K=2 with no prompt_config, uses the original format_prompt() for full
    backward compatibility. For K>2 or with prompt_config, uses permutation-
    based formatting.

    Returns a list of row dicts (one per response value) for the output DataFrame.
    """
    rtype = question["response_type"]
    n_options = 10 if rtype == "likert10" else len(question["translations"][lang]["options"])
    perms = generate_permutations(n_options, n_permutations, question["canonical_id"])

    fmt_kwargs = get_format_kwargs(prompt_config, rtype) if prompt_config else None

    all_probs = []
    all_p_valids = []

    for perm in perms:
        if prompt_config:
            formatted = format_prompt_custom(question, lang, perm, **fmt_kwargs)
        elif n_permutations == 2:
            # Backward compatible: use original format_prompt
            is_reversed = (perm == list(reversed(range(n_options))))
            formatted = format_prompt(question, lang, reverse=is_reversed)
        else:
            formatted = format_prompt_permuted(question, lang, perm)

        logprobs, p_valid = extract_logprobs_for_question(
            model, tokenizer, token_map, formatted, chat_template=chat_template
        )
        probs_raw = renormalize(logprobs)
        probs = remap_reversed(probs_raw, formatted["value_map"])
        all_probs.append(probs)
        all_p_valids.append(p_valid)

    # Collect all semantic values
    all_values = sorted(
        set(v for p in all_probs for v in p.keys()),
        key=lambda x: int(x),
    )

    # Position bias: max across values of (max - min prob across permutations)
    pos_bias = 0.0
    for v in all_values:
        probs_for_v = [p.get(v, 0.0) for p in all_probs]
        pos_bias = max(pos_bias, max(probs_for_v) - min(probs_for_v))

    # Build the forward (identity) prompt for reference
    trans = question["translations"][lang]
    question_text = trans["text"]
    option_labels = {str(opt["value"]): opt.get("label", "") for opt in trans["options"]}
    if rtype == "likert10":
        # Fill in 2-9 with empty labels
        for i in range(1, 11):
            option_labels.setdefault(str(i), "")
    # Get the actual prompt string from the identity permutation
    identity_perm = list(range(n_options))
    if prompt_config:
        fwd_formatted = format_prompt_custom(question, lang, identity_perm, **fmt_kwargs)
    else:
        fwd_formatted = format_prompt(question, lang, reverse=False)
    prompt_text = fwd_formatted["prompt"]

    rows = []
    for v in all_values:
        probs_per_perm = [p.get(v, 0.0) for p in all_probs]
        avg = sum(probs_per_perm) / len(probs_per_perm)
        rows.append({
            "question_id": question["canonical_id"],
            "response_type": rtype,
            "question_text": question_text,
            "prompt": prompt_text,
            "response_value": int(v),
            "option_label": option_labels.get(v, ""),
            "prob_forward": all_probs[0].get(v, 0.0),
            "prob_reversed": all_probs[1].get(v, 0.0) if len(all_probs) > 1 else all_probs[0].get(v, 0.0),
            "prob_averaged": avg,
            "p_valid_forward": all_p_valids[0],
            "p_valid_reversed": all_p_valids[1] if len(all_p_valids) > 1 else all_p_valids[0],
            "position_bias_magnitude": pos_bias,
            "n_permutations": n_permutations,
        })

    return rows


def validate(model, tokenizer, token_map: TokenMap, questions: list[dict], lang: str,
             chat_template: bool = False):
    """Run validation mode: process 10 questions with detailed diagnostics."""
    print("=" * 70)
    print("VALIDATION MODE")
    print("=" * 70)

    token_map.report()

    # Process first 10 questions that have this language
    processed = 0
    p_valids_fwd = []
    p_valids_rev = []
    biases = []

    for q in questions:
        if lang not in q["translations"]:
            continue
        if processed >= 10:
            break

        print(f"\n{'—'*60}")
        print(f"Question {q['canonical_id']} ({q['response_type']})")
        print(f"{'—'*60}")

        # Forward
        fwd = format_prompt(q, lang, reverse=False)
        print(f"\nPrompt (forward):\n{fwd['prompt'][:200]}...")
        logprobs_fwd, p_valid_fwd = extract_logprobs_for_question(
            model, tokenizer, token_map, fwd, chat_template=chat_template
        )
        probs_fwd = renormalize(logprobs_fwd)
        print(f"\nRaw logprobs (fwd): { {k: f'{v:.4f}' for k, v in sorted(logprobs_fwd.items())} }")
        print(f"P_valid (fwd): {p_valid_fwd:.4f}")
        print(f"Renormalized (fwd): { {k: f'{v:.4f}' for k, v in sorted(probs_fwd.items())} }")

        # Reversed
        rev = format_prompt(q, lang, reverse=True)
        logprobs_rev, p_valid_rev = extract_logprobs_for_question(
            model, tokenizer, token_map, rev, chat_template=chat_template
        )
        probs_rev_raw = renormalize(logprobs_rev)
        probs_rev = remap_reversed(probs_rev_raw, rev["value_map"])
        print(f"\nP_valid (rev): {p_valid_rev:.4f}")
        print(f"Renormalized (rev, remapped): { {k: f'{v:.4f}' for k, v in sorted(probs_rev.items())} }")

        # Likert10 two-step details
        if fwd["is_likert10"]:
            print(f"\n  [Likert10 two-step] P(1): {probs_fwd.get('1', 0):.4f}, P(10): {probs_fwd.get('10', 0):.4f}")

        # Position bias
        all_vals = sorted(set(list(probs_fwd.keys()) + list(probs_rev.keys())), key=lambda x: int(x))
        bias = max(abs(probs_fwd.get(v, 0) - probs_rev.get(v, 0)) for v in all_vals) if all_vals else 0
        print(f"Position bias magnitude: {bias:.4f}")

        # Non-uniformity check
        if probs_fwd:
            uniform = 1.0 / len(probs_fwd)
            max_dev = max(abs(p - uniform) for p in probs_fwd.values())
            print(f"Non-uniformity (max dev from uniform): {max_dev:.4f}")

        p_valids_fwd.append(p_valid_fwd)
        p_valids_rev.append(p_valid_rev)
        biases.append(bias)
        processed += 1

    # Summary
    print(f"\n{'='*70}")
    print("VALIDATION SUMMARY")
    print(f"{'='*70}")
    print(f"Questions processed: {processed}")
    print(f"P_valid (fwd): mean={np.mean(p_valids_fwd):.4f}, min={np.min(p_valids_fwd):.4f}, max={np.max(p_valids_fwd):.4f}")
    print(f"P_valid (rev): mean={np.mean(p_valids_rev):.4f}, min={np.min(p_valids_rev):.4f}, max={np.max(p_valids_rev):.4f}")
    print(f"Position bias: mean={np.mean(biases):.4f}, max={np.max(biases):.4f}")

    # Pass/fail
    low_pvalid = sum(1 for p in p_valids_fwd if p < 0.10)
    high_bias = sum(1 for b in biases if b > 0.20)
    print(f"\nLow P_valid (<0.10): {low_pvalid}/{processed}")
    print(f"High position bias (>0.20): {high_bias}/{processed}")

    if low_pvalid > processed // 2:
        print("\nFAIL: Majority of questions have P_valid < 0.10")
    elif np.mean(p_valids_fwd) < 0.20:
        print("\nWARN: Mean P_valid is low — model may not engage with task format")
    else:
        print("\nPASS: Validation looks reasonable")


def load_model(model_id: str, dtype: str = "bf16"):
    """Load model and tokenizer with configurable precision.

    Args:
        model_id: HuggingFace model ID.
        dtype: "bf16" (default), "int4" (4-bit NF4), "int8" (8-bit), or "fp8" (8-bit proxy).

    Returns:
        (model, tokenizer) tuple.
    """
    print(f"Loading model: {model_id} (dtype={dtype})")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if dtype == "bf16":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map="auto",
        )
    elif dtype == "int4":
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
        )
    elif dtype == "int8":
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
        )
    elif dtype == "fp8":
        # FP8 via bitsandbytes 8-bit quantization as proxy.
        # For true FP8 on H100, adjust to use native torch.float8 or auto-fp8.
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
        )
        print("Note: Using int8 quantization as FP8 proxy. "
              "For true FP8, use hardware with native FP8 support (H100).")
    else:
        raise ValueError(f"Unknown dtype: {dtype}")

    model.eval()
    print(f"Model loaded on {model.device if hasattr(model, 'device') else 'auto-mapped devices'}")
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Extract survey response distributions from a base LM"
    )
    parser.add_argument(
        "--model_id", required=True,
        help="HuggingFace model ID (e.g. HPLT/hplt2c_eng_checkpoints)"
    )
    parser.add_argument(
        "--lang", required=True,
        help="Language code (e.g. eng, deu)"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output parquet file path"
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Run validation mode (10 questions with diagnostics)"
    )
    parser.add_argument(
        "--questions", default=None,
        help="Path to questions.json (default: eurollm/data/questions.json)"
    )
    parser.add_argument(
        "--permutations", type=int, default=2,
        help="Number of option-order permutations for debiasing (default: 2 = forward+reversed)"
    )
    parser.add_argument(
        "--prompt-config", default=None,
        help="Path to prompt_config.json for optimized prompt formatting"
    )
    parser.add_argument(
        "--dtype", default="bf16", choices=["bf16", "int4", "int8", "fp8"],
        help="Model precision: bf16 (default), int4 (NF4), int8, fp8"
    )
    parser.add_argument(
        "--chat-template", action="store_true",
        help="Wrap prompts with tokenizer.apply_chat_template() (for instruction-tuned models)"
    )
    args = parser.parse_args()

    # Load questions
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    questions_path = args.questions or str(PROJECT_ROOT / "data" / "questions.json")
    with open(questions_path) as f:
        data = json.load(f)
    questions = data["questions"]

    # Filter to questions available in this language
    available = [q for q in questions if args.lang in q["translations"]]
    print(f"Loaded {len(available)}/{len(questions)} questions for language '{args.lang}'")

    # Load prompt config if specified
    prompt_config = None
    if args.prompt_config:
        prompt_config = load_prompt_config(args.prompt_config)
        # Override n_perms from config if not explicitly set on CLI
        config_perms = prompt_config.get("default", {}).get("n_perms")
        if config_perms and args.permutations == 2:
            args.permutations = config_perms
        print(f"Loaded prompt config from {args.prompt_config}")

    n_perms = args.permutations
    print(f"Using {n_perms} permutation(s) per question")

    # Load model and tokenizer
    model, tokenizer = load_model(args.model_id, args.dtype)

    # Build token map
    t0 = time.time()
    token_map = TokenMap.from_tokenizer(tokenizer)
    print(f"Token map built in {time.time() - t0:.1f}s")

    if args.validate:
        validate(model, tokenizer, token_map, available, args.lang,
                 chat_template=args.chat_template)
        return

    # Full run
    all_rows = []
    t_start = time.time()
    for i, q in enumerate(available):
        t_q = time.time()
        rows = process_question(
            model, tokenizer, token_map, q, args.lang,
            n_permutations=n_perms,
            prompt_config=prompt_config,
            chat_template=args.chat_template,
        )
        all_rows.extend(rows)
        if i == 0:
            print(f"  First question took {time.time() - t_q:.2f}s")
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (len(available) - i - 1) / rate
            print(f"  Processed {i + 1}/{len(available)} questions "
                  f"({elapsed:.0f}s elapsed, {rate:.1f} q/s, ETA {eta:.0f}s)")

    print(f"Processed {len(available)} questions → {len(all_rows)} rows")

    # Save to parquet
    df = pd.DataFrame(all_rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}")

    # Print summary stats
    print(f"\nSummary:")
    if len(df) > 0 and "p_valid_forward" in df.columns:
        print(f"  Mean P_valid (fwd): {df['p_valid_forward'].mean():.4f}")
        print(f"  Mean P_valid (rev): {df['p_valid_reversed'].mean():.4f}")
        print(f"  Mean position bias: {df.groupby('question_id')['position_bias_magnitude'].first().mean():.4f}")
        low_pvalid = (df.groupby("question_id")["p_valid_forward"].first() < 0.10).sum()
        print(f"  Questions with P_valid < 0.10: {low_pvalid}/{len(available)}")
    else:
        print(f"  WARNING: No rows produced — model may have returned NaN logits (check GPU health)")
    print(f"  Permutations per question: {n_perms}")


if __name__ == "__main__":
    main()
