# Inference Module

## Purpose
Run survey prompts through causal language models and extract next-token logprob distributions with position-bias debiasing.

## Inputs
- `../data/questions.json` — Canonical question set
- `../data/rephrasings.json` — Rephrased question variants (for sensitivity testing)
- HuggingFace model IDs (HPLT or EuroLLM)

## Outputs
- `../results/*.parquet` — Per-question probability distributions with forward/reversed/averaged probs, P_valid, and position bias magnitude

## Key Functions / Public API

### `extract_logprobs.py`
- CLI: `python inference/extract_logprobs.py --model_id <id> --lang <code> --output <path> [--validate] [--permutations K] [--prompt-config config.json] [--dtype bf16|int8|fp8]`
- `TokenMap.from_tokenizer(tokenizer)` — Pre-compute digit/zero/terminator token ID mappings
- `extract_logprobs_for_question(model, tokenizer, token_map, formatted)` — Core logprob extraction with two-step "10" resolution
- `renormalize(logprobs)` — Convert logprobs to normalized probability distribution
- `remap_reversed(probs, value_map)` — Remap reversed-order positions back to semantic values
- `generate_permutations(n_options, k, question_id)` — Generate K deterministic permutations for position debiasing
- `load_prompt_config(config_path)` — Load prompt format config JSON
- `load_model(model_id, dtype)` — Load model with configurable precision (bf16/int8/fp8)

### `optimize_prompts.py`
- CLI: `python inference/optimize_prompts.py --model_id <id> --lang <code> --output <path> --questions <path> [--human <path>]`
- Grid search over prompt format dimensions (cue_style, opt_format, scale_hint, embed_style, n_perms)
- Tests 72 format variants × 30 selected questions, computing JSD to human data

### `rephrase_test.py`
- CLI: `python inference/rephrase_test.py --model_id <id> --output <path> [--validate]`
- Tests prompt sensitivity by comparing original vs rephrased question distributions

### `investigate.py`
- CLI: `python inference/investigate.py` (hardcoded to HPLT English)
- Diagnostic tool for tokenizer behavior and prompt format effects on P_valid

## Dependencies
- `prompting.prompt_templates` — `format_prompt`, `clean_text`, `clean_label`
- External: `torch`, `transformers`, `numpy`, `pandas`
