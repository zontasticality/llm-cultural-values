"""Investigate space-prefixed digit tokens and prompt format effects on P_valid."""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompting.prompt_templates import format_prompt, clean_text, clean_label


def investigate_tokenizer(tokenizer):
    """Deep investigation of digit tokenization."""
    print("=" * 70)
    print("TOKENIZER INVESTIGATION")
    print("=" * 70)
    print(f"Tokenizer class: {type(tokenizer).__name__}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print()

    # 1. Check bare digits and space-prefixed variants
    print("--- Digit token encoding ---")
    for d in range(0, 10):
        bare = tokenizer.encode(str(d), add_special_tokens=False)
        space = tokenizer.encode(f" {d}", add_special_tokens=False)
        # Also check what "Answer: " + digit looks like
        ctx = tokenizer.encode(f"Answer: {d}", add_special_tokens=False)
        ctx_no_space = tokenizer.encode(f"Answer:{d}", add_special_tokens=False)
        print(f"  '{d}' → {bare}  |  ' {d}' → {space}  |  'Answer: {d}' → {ctx}  |  'Answer:{d}' → {ctx_no_space}")

    print()

    # 2. Check what the last token is when "Answer:" is tokenized
    print("--- Answer cue tokenization ---")
    for cue in ["Answer:", "Answer: ", "Answer:\n", "Answer: \n"]:
        tokens = tokenizer.encode(cue, add_special_tokens=False)
        decoded = [tokenizer.decode([t]) for t in tokens]
        print(f"  {repr(cue):20s} → tokens={tokens}, decoded={decoded}")

    print()

    # 3. Reverse lookup: find ALL tokens containing digit characters
    print("--- Tokens containing bare digits (searching vocab) ---")
    for d in range(1, 10):
        digit_str = str(d)
        matching = []
        # Search vocab for tokens that are just the digit (with possible space/special prefix)
        for tok_str, tok_id in tokenizer.get_vocab().items():
            # Check if the token decodes to just the digit or space+digit
            decoded = tokenizer.decode([tok_id])
            if decoded.strip() == digit_str and len(decoded) <= 3:
                matching.append((tok_id, repr(tok_str), repr(decoded)))
        print(f"  Digit '{d}': {len(matching)} matching tokens")
        for tid, ts, dec in matching[:5]:
            print(f"    ID={tid}, vocab_key={ts}, decoded={dec}")

    print()

    # 4. Check the specific token that appears after "Answer:" in context
    print("--- What token follows 'Answer:' in full prompt context ---")
    test_prompt = "How important is work?\n1. very important\n2. not important\nAnswer:"
    prompt_tokens = tokenizer.encode(test_prompt, add_special_tokens=False)
    for d in ["1", "2", " 1", " 2"]:
        full = tokenizer.encode(test_prompt + d, add_special_tokens=False)
        new_tokens = full[len(prompt_tokens):]
        print(f"  '{test_prompt[-8:]}' + {repr(d):5s} → new token(s): {new_tokens} = {[tokenizer.decode([t]) for t in new_tokens]}")


def investigate_prompt_formats(model, tokenizer, questions, lang="eng"):
    """Test different prompt format variations and measure P_valid."""
    print("\n" + "=" * 70)
    print("PROMPT FORMAT INVESTIGATION")
    print("=" * 70)

    # Pick 5 diverse questions
    target_ids = ["v1", "v7", "v31", "v38", "v64"]
    test_qs = []
    for q in questions:
        if q["canonical_id"] in target_ids and lang in q["translations"]:
            test_qs.append(q)

    # Define prompt format variations
    def make_prompt_standard(q, lang):
        """Current format: 'Answer:'"""
        result = format_prompt(q, lang, reverse=False)
        return result["prompt"], result["valid_values"], result["is_likert10"]

    def make_prompt_space_after_colon(q, lang):
        """'Answer: ' (with trailing space)"""
        result = format_prompt(q, lang, reverse=False)
        prompt = result["prompt"] + " "
        return prompt, result["valid_values"], result["is_likert10"]

    def make_prompt_newline_after_colon(q, lang):
        """'Answer:\\n'"""
        result = format_prompt(q, lang, reverse=False)
        prompt = result["prompt"] + "\n"
        return prompt, result["valid_values"], result["is_likert10"]

    def make_prompt_no_cue(q, lang):
        """No answer cue at all — prompt ends with last option"""
        result = format_prompt(q, lang, reverse=False)
        # Strip the answer cue line
        lines = result["prompt"].rsplit("\n", 1)
        prompt = lines[0] + "\n"
        return prompt, result["valid_values"], result["is_likert10"]

    def make_prompt_the_answer_is(q, lang):
        """'The answer is'"""
        result = format_prompt(q, lang, reverse=False)
        # Replace the answer cue
        lines = result["prompt"].rsplit("\n", 1)
        prompt = lines[0] + "\nThe answer is"
        return prompt, result["valid_values"], result["is_likert10"]

    def make_prompt_select_one(q, lang):
        """'Select one: '"""
        result = format_prompt(q, lang, reverse=False)
        lines = result["prompt"].rsplit("\n", 1)
        prompt = lines[0] + "\nSelect one:"
        return prompt, result["valid_values"], result["is_likert10"]

    formats = {
        "Answer:": make_prompt_standard,
        "Answer: ": make_prompt_space_after_colon,
        "Answer:\\n": make_prompt_newline_after_colon,
        "(no cue)": make_prompt_no_cue,
        "The answer is": make_prompt_the_answer_is,
        "Select one:": make_prompt_select_one,
    }

    # Build digit token map (include all variants found)
    digit_tokens = {}
    for d in range(1, 10):
        ids = set()
        for variant in [str(d), f" {d}"]:
            encoded = tokenizer.encode(variant, add_special_tokens=False)
            if len(encoded) == 1:
                ids.add(encoded[0])
        # Also search vocab for any other representations
        for tok_str, tok_id in tokenizer.get_vocab().items():
            decoded = tokenizer.decode([tok_id])
            if decoded.strip() == str(d) and len(decoded) <= 3:
                ids.add(tok_id)
        digit_tokens[str(d)] = ids

    zero_tokens = set()
    for variant in ["0", " 0"]:
        encoded = tokenizer.encode(variant, add_special_tokens=False)
        if len(encoded) == 1:
            zero_tokens.add(encoded[0])

    terminator_tokens = set()
    for t in ["\n", " ", ".", ",", "\n\n"]:
        encoded = tokenizer.encode(t, add_special_tokens=False)
        if len(encoded) == 1:
            terminator_tokens.add(encoded[0])
    if tokenizer.eos_token_id is not None:
        terminator_tokens.add(tokenizer.eos_token_id)

    print(f"\nDigit tokens (expanded search):")
    for d in range(1, 10):
        print(f"  '{d}': {sorted(digit_tokens[str(d)])}")

    # Test each format × question
    print(f"\n{'Question':<8} {'Format':<20} {'P_valid':>8} {'Top1':>6} {'Top1_val':>8} {'Prompt_end'}")
    print("-" * 80)

    for q in test_qs:
        for fmt_name, fmt_fn in formats.items():
            prompt, valid_values, is_likert10 = fmt_fn(q, lang)

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1)

            # Extract probabilities for valid digits
            result = {}
            for val_str in valid_values:
                if val_str == "10":
                    continue
                token_ids = digit_tokens.get(val_str, set())
                if token_ids:
                    prob_sum = sum(torch.exp(log_probs[tid]).item() for tid in token_ids)
                    if prob_sum > 0:
                        result[val_str] = prob_sum

            # Handle "10" for likert10
            if is_likert10 and "1" in result:
                p_raw_1 = result["1"]
                bare_one_id = tokenizer.encode("1", add_special_tokens=False)[0]
                extended_ids = torch.cat([
                    inputs["input_ids"],
                    torch.tensor([[bare_one_id]], device=model.device),
                ], dim=1)
                with torch.no_grad():
                    out2 = model(input_ids=extended_ids)
                logits2 = out2.logits[0, -1, :]
                lp2 = torch.log_softmax(logits2, dim=-1)
                p_zero = sum(torch.exp(lp2[tid]).item() for tid in zero_tokens)
                p_term = sum(torch.exp(lp2[tid]).item() for tid in terminator_tokens)
                denom = p_zero + p_term
                if denom > 0.01:
                    result["1"] = p_raw_1 * p_term / denom
                    result["10"] = p_raw_1 * p_zero / denom

            p_valid = sum(result.values())
            if result:
                top_val = max(result, key=result.get)
                top_prob = result[top_val] / p_valid if p_valid > 0 else 0
            else:
                top_val = "?"
                top_prob = 0

            prompt_end = repr(prompt[-30:])
            print(f"{q['canonical_id']:<8} {fmt_name:<20} {p_valid:>8.4f} {top_prob:>6.2%} {top_val:>8} {prompt_end}")

            # For the first question only, show top-20 tokens at answer position
            if q["canonical_id"] == "v1" and fmt_name == "Answer:":
                print(f"\n  Top-20 tokens at answer position for v1 'Answer:' format:")
                top_k = torch.topk(log_probs, 20)
                for i in range(20):
                    tid = top_k.indices[i].item()
                    lp = top_k.values[i].item()
                    decoded = repr(tokenizer.decode([tid]))
                    print(f"    #{i+1}: ID={tid}, logprob={lp:.4f}, prob={np.exp(lp):.4f}, token={decoded}")
                print()

        print()

    # 5. Also try: what if we look at the top-k AFTER "Answer: " (with space)?
    print("\n--- Top-20 tokens comparison: 'Answer:' vs 'Answer: ' ---")
    q = test_qs[0]  # v1
    for suffix in [":", ": "]:
        result = format_prompt(q, lang, reverse=False)
        if suffix == ": ":
            prompt = result["prompt"] + " "
        else:
            prompt = result["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
        log_probs = torch.log_softmax(logits, dim=-1)
        print(f"\n  After 'Answer{suffix}' — Top 20:")
        top_k = torch.topk(log_probs, 20)
        for i in range(20):
            tid = top_k.indices[i].item()
            lp = top_k.values[i].item()
            decoded = repr(tokenizer.decode([tid]))
            print(f"    #{i+1}: ID={tid}, logprob={lp:.4f}, prob={np.exp(lp):.4f}, token={decoded}")


def main():
    model_id = "HPLT/hplt2c_eng_checkpoints"
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    questions_path = PROJECT_ROOT / "data" / "questions.json"

    with open(questions_path) as f:
        data = json.load(f)
    questions = data["questions"]

    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    investigate_tokenizer(tokenizer)

    print(f"\nLoading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    print(f"Model on {next(model.parameters()).device}")

    investigate_prompt_formats(model, tokenizer, questions)


if __name__ == "__main__":
    main()
