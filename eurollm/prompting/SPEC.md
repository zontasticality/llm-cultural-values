# Prompting Module

## Purpose
Format survey questions into completion prompts for base language models, with forward and reversed option orderings for position-bias debiasing.

## Inputs
- Question dicts from `../data/questions.json`
- Language code (e.g. "eng", "deu")

## Outputs
- Formatted prompt strings with valid response values and value maps

## Key Functions / Public API

### `prompt_templates.py`
- `format_prompt(question, lang, reverse=False) -> dict` — Main entry point. Returns `{prompt, valid_values, value_map, is_likert10}`
- `format_prompt_permuted(question, lang, permutation) -> dict` — Generalized permutation-based formatting. `permutation` is a list of 0-indexed positions into the original options list.
- `format_prompt_custom(question, lang, permutation, cue_style, opt_format, scale_hint, embed_style) -> dict` — Full format customization for optimization grid search. Supports configurable answer cues, option formats, scale hints, and option embedding styles.
- `clean_text(text) -> str` — Strip interviewer instructions and embedded option lists
- `clean_label(label) -> str` — Clean option labels (strip number prefixes, trailing punctuation)

### Response types handled
- `likert4`, `likert5` — Standard numbered options
- `likert10` — 1-10 scale with endpoint anchors
- `likert3` — 3-point scale
- `categorical`, `frequency` — Named options

## Dependencies
- No cross-module imports (stdlib only: `json`, `re`, `pathlib`)
