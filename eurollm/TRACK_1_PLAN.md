# Track 1: WVS Survey Elicitation — Detailed Implementation Plan

## Context

We want to measure the cultural values encoded in pretrained (non-finetuned) European language models by having them "answer" World Values Survey / European Values Study questions and comparing response distributions across languages. The models are:

- **22 HPLT monolingual reference models** (`HPLT/hplt2c_{lang}_checkpoints`, 2.15B params each, LLaMA architecture, Gemma-3 tokenizer with 262K vocab, trained on 100B tokens of single-language web data)
- **EuroLLM-22B-2512** (`utter-project/EuroLLM-22B-2512`, 22.6B params, multilingual across 35 languages, custom 128K BPE tokenizer)

Each monolingual model gets the EVS/WVS questions in its own language. The multilingual model gets every language.

**Key related work**: Santurkar et al. (2023, OpinionsQA), Durmus et al. (2023, GlobalOpinionQA), Kabir et al. (2025, "Break the Checkbox"), Shen et al. (2025, "Revisiting LLM Value Probing Strategies"). See [Appendix A](#appendix-a-key-references) for full citations.

---

## Step 1: Extract Questions from EVS/WVS Questionnaires into JSON

**What**: Extract survey questions from official EVS 2017 field questionnaires (which share a ~70% common core with WVS Wave 7/8) into structured JSON, one file per language.

**Source documents**: Official EVS 2017 field questionnaire PDFs from GESIS (no registration required, direct download). These are the actual human-translated questionnaires used in the field. Downloaded to `eurollm/evs_questionnaires/`.

| Language | GESIS URL | Local file |
|----------|-----------|------------|
| Bulgarian | https://access.gesis.org/dbk/65200 | `evs_questionnaires/bul.pdf` |
| Croatian | https://access.gesis.org/dbk/65207 | `evs_questionnaires/hrv.pdf` |
| Czech | https://access.gesis.org/dbk/65203 | `evs_questionnaires/ces.pdf` |
| Danish | https://access.gesis.org/dbk/66248 | `evs_questionnaires/dan.pdf` |
| Dutch | https://access.gesis.org/dbk/66257 | `evs_questionnaires/nld.pdf` |
| English | https://access.gesis.org/dbk/66252 | `evs_questionnaires/eng.pdf` |
| Estonian | https://access.gesis.org/dbk/66249 | `evs_questionnaires/est.pdf` |
| Finnish | https://access.gesis.org/dbk/66250 | `evs_questionnaires/fin.pdf` |
| French | https://access.gesis.org/dbk/66251 | `evs_questionnaires/fra.pdf` |
| German | https://access.gesis.org/dbk/66247 | `evs_questionnaires/deu.pdf` |
| Greek | https://access.gesis.org/dbk/67588 | `evs_questionnaires/ell.pdf` |
| Hungarian | https://access.gesis.org/dbk/66253 | `evs_questionnaires/hun.pdf` |
| Italian | https://access.gesis.org/dbk/66255 | `evs_questionnaires/ita.pdf` |
| Latvian | https://access.gesis.org/dbk/72394 | `evs_questionnaires/lvs.pdf` |
| Lithuanian | https://access.gesis.org/dbk/66256 | `evs_questionnaires/lit.pdf` |
| Polish | https://access.gesis.org/dbk/65210 | `evs_questionnaires/pol.pdf` |
| Portuguese | https://access.gesis.org/dbk/69458 | `evs_questionnaires/por.pdf` |
| Romanian | https://access.gesis.org/dbk/66259 | `evs_questionnaires/ron.pdf` |
| Slovak | https://access.gesis.org/dbk/65213 | `evs_questionnaires/slk.pdf` |
| Slovenian | https://access.gesis.org/dbk/65212 | `evs_questionnaires/slv.pdf` |
| Spanish | https://access.gesis.org/dbk/65205 | `evs_questionnaires/spa.pdf` |
| Swedish | https://access.gesis.org/dbk/66261 | `evs_questionnaires/swe.pdf` |
| EVS Master (English) | https://access.gesis.org/dbk/69554 | `evs_questionnaires/evs_master_en.pdf` |

**Extraction method**: Use Gemini Flash to process each PDF. For each language, prompt Gemini Flash with the PDF and a schema specification to output structured JSON.

**Prompt for Gemini Flash** (applied to each PDF):
```
Extract all opinion/values survey questions from this questionnaire PDF into JSON.
Skip demographic questions (age, sex, education, income, etc.) and interviewer instructions.

For each question, output:
{
  "id": "Q1",           // EVS question number as it appears in the PDF
  "text": "...",         // Full question text in the original language
  "stem": "...",         // Shared stem if this is part of a battery (null otherwise)
  "item": "...",         // Sub-item label if part of a battery (null otherwise)
  "response_type": "likert4|likert5|likert10|binary|categorical|frequency|action|confidence",
  "options": [
    {"value": 1, "label": "..."},  // In the original language
    {"value": 2, "label": "..."}
  ],
  "answer_cue": "..."   // The word "Answer" in this language (for prompt construction)
}

Output a JSON array. Preserve the original language for all text fields.
```

**Post-processing**: After Gemini Flash extraction, align questions across languages using the EVS question IDs. Build a master question list from the English extraction, then verify each language file has matching IDs.

**Schema per question** (final output):
```json
{
  "id": "Q1",
  "section": "SOCIAL_VALUES",
  "text": "For each of the following, indicate how important it is in your life. Would you say it is... Family",
  "stem": "For each of the following, indicate how important it is in your life. Would you say it is...",
  "item": "Family",
  "response_type": "likert4",
  "options": [
    {"value": 1, "label": "Very important"},
    {"value": 2, "label": "Rather important"},
    {"value": 3, "label": "Not very important"},
    {"value": 4, "label": "Not at all important"}
  ],
  "answer_cue": "Answer"
}
```

**Response types to handle**:
| Type | Scale | Count (approx) |
|------|-------|-----------------|
| `likert4` | 4-point importance/agree/confidence/favorability | ~100 |
| `likert5` | 5-point agree/disagree | ~15 |
| `likert10` | 1-10 scale with anchors | ~50 |
| `binary` | yes/no or mentioned/not | ~20 |
| `categorical` | pick one from list | ~10 |
| `frequency` | how often | ~15 |
| `action` | have done / might do / never | 8 |

**Questions to exclude**:
- Country-specific questions (head of state, political parties, regional organizations)
- List experiments (Q78A/B) — complex format incompatible with logprob extraction
- Election integrity questions (Q180-Q188) — not meaningful for LLMs
- Demographic questions (age, sex, education, income, etc.)

**Output**: `eurollm/translations/{lang_code}.json` for each of 22 languages, plus `eurollm/questions.json` as the English master list.

**EVS vs WVS alignment note**: The EVS 2017 and WVS Wave 7/8 share a common core of questions but are not identical. The EVS question IDs may differ from WVS IDs. We use EVS question IDs as our canonical reference and note which questions also appear in WVS Wave 8 (using the WVS-8 master questionnaire PDF for cross-referencing).

---

## Step 2: Prompt Template Design

**Design principle**: Minimize framing. No system prompt, no mention of "survey" or "EVS". Present as a bare completion task.

### 2a. Templates

**Template for Likert-scale questions (4-point, 5-point, frequency, action, etc.)**:
```
{question_text}
{option_1_value}. {option_1_label}
{option_2_value}. {option_2_label}
...
{answer_cue}:
```

Example (English):
```
How important is family in your life?
1. Very important
2. Rather important
3. Not very important
4. Not at all important
Answer:
```

Example (German):
```
Wie wichtig ist Familie in Ihrem Leben?
1. Sehr wichtig
2. Ziemlich wichtig
3. Nicht sehr wichtig
4. Überhaupt nicht wichtig
Antwort:
```

**Template for 1-10 scale questions**:
```
{question_text}
1. {left_anchor}
2.
3.
4.
5.
6.
7.
8.
9.
10. {right_anchor}
{answer_cue}:
```

All 10 options are listed explicitly. See Step 2b for how "10" is handled at the logprob level.

**For binary (mentioned/not mentioned) questions**: Convert to agree/disagree format since the "show card and select" interaction doesn't translate to text completion.

### 2b. The "10" Token Problem — Two-Step Conditional Resolution

The Gemma-3 tokenizer (used by HPLT models) uses SentencePiece with `split_digits` enabled:
- Digits "1" through "9" are each single tokens.
- "10" is tokenized as **two tokens** ("1" + "0").

**Solution**: Use two-step conditional logprob extraction to disambiguate "1" (standalone answer) from the start of "10":

1. **First pass**: Extract logprobs at the answer position for tokens "1"–"9" as usual. Call the raw probability for "1" `P_raw("1")`.
2. **Second pass**: Append the token "1" to the prompt and run one additional forward pass. From the resulting next-token distribution, extract:
   - `P("0" | "1")` — probability the model continues to "10"
   - `P(terminator | "1")` — probability the answer is complete (sum over EOS, newline, space, period, comma tokens)
3. **Decompose**:
   - `P(answer=1) = P_raw("1") × P(terminator | "1") / (P("0"|"1") + P(terminator|"1"))`
   - `P(answer=10) = P_raw("1") × P("0" | "1") / (P("0"|"1") + P(terminator|"1"))`
4. Digits 2–9 are unambiguous — their probabilities come directly from the first pass.

**Cost**: One extra forward pass per 1-10 question (~50 questions). Negligible relative to the full run.

**Validation**: During the sanity check, verify that `P("0"|"1") + P(terminator|"1")` accounts for the vast majority of the conditional distribution (>0.80). If not, there may be other continuations we need to handle.

### 2c. Position Bias Mitigation

The literature shows that LLMs exhibit significant position bias in survey tasks — both primacy bias (favoring earlier options) and recency bias (favoring later options), depending on the model (Zheng et al. 2024; Kabir et al. 2025). Quantified effects include 3-15% alignment shifts from option reordering (Kabir et al. 2025) and 20× selection frequency changes (arxiv 2507.07188). For 2.15B base models, we should expect meaningful position effects.

**Approach**: Run every question in **two orderings** — original and reversed — and average the resulting probability distributions. This doubles the compute cost (88 conditions instead of 44) but is the minimum credible debiasing strategy per the literature.

For a 4-point Likert question, the two orderings are:
```
# Original order
1. Very important
2. Rather important
3. Not very important
4. Not at all important

# Reversed order
1. Not at all important
2. Rather important
3. Not very important
4. Very important
```

Note: We reverse the mapping of labels to numbers, not the semantic content. Option "1" always means "the first listed option". After logprob extraction, we remap back to the original semantic scale before averaging.

**Output**: For each question, store both the original-order and reversed-order distributions, plus the averaged distribution. Report the magnitude of position bias (max absolute difference between orderings) as a quality metric.

### 2d. Leading-Space Token Variant

The Gemma-3 tokenizer may encode `" 1"` (with a leading space) differently from `"1"` (bare digit). Since the answer cue ends with `:` or `: `, the next token prediction could plausibly target either variant.

**Action**: During the tokenizer verification step, check both `tokenizer.encode("1", add_special_tokens=False)` and `tokenizer.encode(" 1", add_special_tokens=False)`. If they differ, sum the logprobs of both variants for each digit. Similarly for the EuroLLM 128K tokenizer.

### 2e. Key Design Decisions

- No system prompt at all (base models don't use them)
- BOS token must be prepended (both model architectures are LLaMA-based and expect it — `AutoTokenizer` handles this by default)
- The `answer_cue` field from the JSON (extracted by Gemini Flash per language) is used as the prompt suffix
- We extract logprobs for tokens "1"–"9" (plus "10" via two-step method, plus space-prefixed variants) at the next-token position
- The prompt fits in <200 tokens (well within the 2048-token HPLT context limit)

### 2f. Validation

Before the full run, test 10 questions on the English HPLT model to verify:
1. Logprob distributions are non-uniform (model has preferences)
2. Distributions make directional sense (e.g., "stealing" skews toward "never justifiable")
3. The tokenizer encodes "1"–"9" as single tokens; check leading-space variants
4. Two-step "10" resolution works: `P("0"|"1") + P(terminator|"1") > 0.80`
5. Position bias check: run forward vs. reversed on 5 questions, measure magnitude
6. `P_valid` (see Step 3) is meaningfully above uniform random for most questions

---

## Step 3: Logprob Extraction Pipeline

**File**: `eurollm/extract_logprobs.py`

**Core logic**:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, json
import numpy as np

def get_digit_token_ids(tokenizer):
    """Get token IDs for digits 1-9, including leading-space variants."""
    digit_tokens = {}
    for d in range(1, 10):
        ids = set()
        for variant in [str(d), f" {d}"]:
            encoded = tokenizer.encode(variant, add_special_tokens=False)
            if len(encoded) == 1:
                ids.add(encoded[0])
        digit_tokens[str(d)] = ids
    return digit_tokens

def get_terminator_token_ids(tokenizer):
    """Get token IDs for answer-terminating tokens (EOS, newline, space, period, comma)."""
    terminators = set()
    for t in [tokenizer.eos_token, "\n", " ", ".", ",", "\n\n"]:
        if t is not None:
            encoded = tokenizer.encode(t, add_special_tokens=False)
            if len(encoded) == 1:
                terminators.add(encoded[0])
    return terminators

def extract_logprobs(model, tokenizer, prompt, valid_values, digit_token_ids,
                     terminator_ids=None, has_ten=False):
    """Extract logprobs for valid response tokens at the answer position.

    For 1-10 scale questions (has_ten=True), runs a second forward pass
    to disambiguate "1" (standalone) from "10" (two-digit).

    Returns:
        logprobs: dict mapping value string to logprob
        p_valid: total probability mass on valid tokens (before renormalization)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]
    log_probs = torch.log_softmax(logits, dim=-1)

    result = {}
    for val_str in valid_values:
        if val_str == "10":
            continue  # handled below
        token_ids = digit_token_ids.get(val_str, set())
        if token_ids:
            probs_for_val = torch.tensor([
                torch.exp(log_probs[tid]).item() for tid in token_ids
            ])
            result[val_str] = torch.log(probs_for_val.sum()).item()

    # Two-step resolution for "10"
    if has_ten and "1" in result and terminator_ids:
        p_raw_1 = np.exp(result["1"])

        # Second forward pass: append "1" token and get conditional distribution
        one_ids = list(digit_token_ids["1"])
        # Use the bare "1" token (not space-prefixed) as the appended token
        bare_one = tokenizer.encode("1", add_special_tokens=False)[0]
        extended_ids = torch.cat([
            inputs["input_ids"],
            torch.tensor([[bare_one]], device=model.device)
        ], dim=1)
        with torch.no_grad():
            out2 = model(extended_ids)
        logits2 = out2.logits[0, -1, :]
        lp2 = torch.log_softmax(logits2, dim=-1)

        # P("0" | ..., "1") — continuing to "10"
        zero_ids = tokenizer.encode("0", add_special_tokens=False)
        p_zero = sum(torch.exp(lp2[tid]).item() for tid in zero_ids if len(zero_ids) == 1)

        # P(terminator | ..., "1") — answer complete at "1"
        p_term = sum(torch.exp(lp2[tid]).item() for tid in terminator_ids)

        # Normalize and decompose
        denom = p_zero + p_term
        if denom > 0.01:
            result["1"] = np.log(p_raw_1 * p_term / denom).item()
            result["10"] = np.log(p_raw_1 * p_zero / denom).item()
        # else: ambiguous, keep P(1) as-is and set P(10) ≈ 0

    p_valid = sum(np.exp(lp) for lp in result.values())
    return result, p_valid

def run_survey(model_id, questions_file, lang_code, output_file):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    digit_token_ids = get_digit_token_ids(tokenizer)
    terminator_ids = get_terminator_token_ids(tokenizer)
    questions = json.load(open(questions_file))
    results = []

    for q in questions:
        valid_values = [str(opt["value"]) for opt in q["options"]]
        has_ten = "10" in valid_values

        # Forward order
        prompt_fwd = format_prompt(q, lang_code, reverse=False)
        logprobs_fwd, p_valid_fwd = extract_logprobs(
            model, tokenizer, prompt_fwd, valid_values, digit_token_ids,
            terminator_ids, has_ten
        )
        probs_fwd = softmax_dict(logprobs_fwd)

        # Reversed order
        prompt_rev = format_prompt(q, lang_code, reverse=True)
        logprobs_rev, p_valid_rev = extract_logprobs(
            model, tokenizer, prompt_rev, valid_values, digit_token_ids,
            terminator_ids, has_ten
        )
        probs_rev = remap_reversed(softmax_dict(logprobs_rev), q["options"])

        # Average forward and reversed distributions
        probs_avg = {k: (probs_fwd.get(k, 0) + probs_rev.get(k, 0)) / 2
                     for k in set(list(probs_fwd) + list(probs_rev))}

        # Position bias magnitude
        all_keys = set(list(probs_fwd) + list(probs_rev))
        pos_bias = max(
            abs(probs_fwd.get(k, 0) - probs_rev.get(k, 0)) for k in all_keys
        )

        results.append({
            "question_id": q["id"],
            "probs_forward": probs_fwd,
            "probs_reversed": probs_rev,
            "probs_averaged": probs_avg,
            "p_valid_forward": p_valid_fwd,
            "p_valid_reversed": p_valid_rev,
            "position_bias_magnitude": pos_bias,
        })

    save_results(results, output_file)
```

**Output format**: One parquet file per model-language pair:
```
eurollm/results/{model_name}_{lang_code}.parquet
```

Columns: `question_id, response_value, prob_forward, prob_reversed, prob_averaged, p_valid_forward, p_valid_reversed, position_bias_magnitude`

**Quality metric — `P_valid`**: The total probability mass the model places on valid answer tokens *before* renormalization. This indicates whether the model "understands" the prompt as a question requiring one of the listed responses. Questions where `P_valid < 0.10` should be flagged as unreliable — the renormalized distribution is likely noise. Report `P_valid` distributions in the paper.

---

## Step 4: SLURM Job Scripts

**File**: `eurollm/run_slurm.sh`

**Strategy**: One SLURM job per model-language pair. Each job loads the model once and runs all questions × 2 orderings sequentially (plus one extra forward pass per 1-10 question for the "10" resolution).

```bash
#!/bin/bash
#SBATCH --job-name=wvs-survey
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/wvs_%j.out
#SBATCH --error=logs/wvs_%j.err

# Load environment (adjust for your cluster)
# module load cuda/12.x
# source activate llm-culture

MODEL_ID=$1   # e.g., HPLT/hplt2c_eng_checkpoints
LANG=$2       # e.g., eng
QUESTIONS=eurollm/translations/${LANG}.json

export HF_HOME=/scratch4/workspace/zevwilson_umass_edu-culture-llm/.hf_cache

python eurollm/extract_logprobs.py \
    --model_id "$MODEL_ID" \
    --questions "$QUESTIONS" \
    --lang "$LANG" \
    --output "eurollm/results/${MODEL_ID##*/}_${LANG}.parquet"
```

**Launcher script** (`eurollm/launch_all.sh`):
```bash
#!/bin/bash
set -euo pipefail

LANGS="bul hrv ces dan nld eng est fin fra deu ell hun ita lvs lit pol por ron slk slv spa swe"

# Submit all 22 monolingual HPLT models (2.15B, ~5GB VRAM each)
for lang in $LANGS; do
    sbatch eurollm/run_slurm.sh "HPLT/hplt2c_${lang}_checkpoints" "$lang"
done

# Submit EuroLLM-22B for each language (22.6B, ~45GB VRAM in BF16)
for lang in $LANGS; do
    sbatch --mem=96G --gres=gpu:a100:2 --time=02:00:00 eurollm/run_slurm.sh \
        "utter-project/EuroLLM-22B-2512" "$lang"
done

echo "Submitted 22 monolingual + 22 multilingual = 44 jobs"
```

---

## Step 5: Analysis Pipeline

**File**: `eurollm/analyze.py`

### 5a. Data Aggregation

Load all 44 parquet files into a single DataFrame with columns:
`model_id, model_type (monolingual|multilingual), lang_code, question_id, response_value, prob_forward, prob_reversed, prob_averaged, p_valid_forward, p_valid_reversed, position_bias_magnitude`

One row per (model, language, question, response_value) tuple. Total rows ≈ 44 conditions × ~200 questions × ~5 avg options ≈ 44,000 rows.

### 5b. Quality Filtering

- **Per-question filter**: Drop questions where `P_valid < 0.10` for >50% of models. These are questions where the prompt format doesn't elicit survey-like responses — the model is putting <10% of its probability mass on valid answer tokens, so the renormalized distribution is noise.
- **Per-model flag**: Flag models where median `P_valid < 0.20`. This indicates the model may be too small or too poorly trained to engage with the task format at all.
- **Output**: A quality report table showing % of questions passing the P_valid threshold per model. This goes in the paper's methods section.

### 5c. Position Bias Analysis

- Compute the distribution of `position_bias_magnitude` across all questions and models.
- Flag question-model pairs where position bias > 0.20 (the forward and reversed distributions differ by more than 20 percentage points on at least one option).
- Report: Does position bias correlate with model size? (Expect 2.15B HPLT models to show more bias than 22B EuroLLM.) Does it vary by question type? (Expect more bias on longer Likert scales.)
- This is itself a finding worth reporting, even if it's a methodological result rather than a cultural one.

### 5d. Per-Question Summary Statistics

For each (model, language, question), compute from `probs_averaged`:
- **Expected value**: `E[X] = Σ (value × prob)` — the "average response" on the scale
- **Entropy**: `H = -Σ (prob × log2(prob))` — how uncertain/spread the distribution is
- **Mode**: the response option with highest probability
- **Concentration**: probability of the mode (how peaked the distribution is)

### 5e. Cross-Model Distance Matrix

For each question, compute **Jensen-Shannon divergence (JSD)** between every pair of (model, language) distributions. This produces a 44×44 symmetric distance matrix per question, which we average across questions to get an overall distance matrix.

Apply dimensionality reduction (MDS or t-SNE) to visualize which model-language pairs cluster together. Key hypotheses to test:
- Do HPLT monolingual models cluster by **cultural geography**? (Nordic cluster: Danish/Swedish/Finnish/Norwegian; Mediterranean: Italian/Spanish/Portuguese/Greek; Central European: German/Czech/Slovak/Polish; Baltic: Estonian/Latvian/Lithuanian)
- Does EuroLLM-22B in language X cluster with the HPLT monolingual model for language X, or does it form its own cluster (reflecting a "EuroLLM average culture")?
- Quantify: what fraction of the variance is explained by language vs. model architecture?

### 5f. Inglehart-Welzel Cultural Map

The Inglehart-Welzel cultural map positions countries on two dimensions:
- **Traditional vs. Secular-rational values** (Y-axis): composed from questions about importance of God, national pride, respect for authority, obedience as a child value, justifiability of abortion
- **Survival vs. Self-expression values** (X-axis): composed from questions about life satisfaction, happiness, trust in people, tolerance of outgroups, political action, gender equality

The specific WVS variables that feed into each dimension are published in Inglehart & Welzel (2005) and the WVS cultural map methodology docs. We compute each model-language pair's position on these two axes using the standard composite indices.

**Comparison to real data**: Download the WVS Wave 7 aggregate country-level response distributions from worldvaluessurvey.org (free, no registration for trend data). Compute real country positions on the same Inglehart-Welzel dimensions. Plot model-language pairs alongside real countries on the same cultural map.

**This is the key figure for the paper.** It shows at a glance whether LLM cultural priors align with real-world cultural geography.

### 5g. Model vs. Real Country Comparison

For each (language, question), compute:
- **JSD** between the model's response distribution and the real country's survey response distribution
- **1-Wasserstein distance** (earth mover's distance) — more interpretable for ordinal scales since it accounts for the ordering of response options (disagreeing by 1 point is better than disagreeing by 3 points)
- **KL divergence** — asymmetric, measures information loss from approximating the real distribution with the model distribution

Report: Which languages/countries have LLM distributions closest to real survey data? Are there systematic patterns? (e.g., do models for high-resource languages like English/German match better than low-resource ones like Latvian/Lithuanian?)

### 5h. Monolingual vs. Multilingual Comparison

The core research question: **Does training on only one language's data produce value distributions closer to that country's real survey responses?**

For each language:
- Compute distance (JSD) between HPLT monolingual model and real country data
- Compute distance (JSD) between EuroLLM-22B (in same language) and real country data
- Paired comparison: Is the monolingual model systematically closer?

If yes, this supports the hypothesis that cultural values are encoded in language-specific training data. If no (multilingual model is equally close or closer), it suggests that model scale or architecture matters more than data isolation.

### 5i. Visualization

All figures go in `eurollm/figures/`:

1. **cultural_map.png** — Inglehart-Welzel cultural map with model-language pairs and real countries
2. **p_valid_heatmap.png** — Heatmap of P_valid by model × question type
3. **position_bias_dist.png** — Distribution of position bias magnitude across models
4. **jsd_heatmap.png** — 44×44 distance matrix heatmap showing model-language clustering
5. **mono_vs_multi.png** — Paired bar chart: monolingual vs. multilingual distance to real country data, per language
6. **example_distributions.png** — Selected questions showing response distributions across a few languages (for qualitative illustration)

---

## Compute Requirements

### Per-model VRAM

| Model | Params | BF16 VRAM | GPU needed |
|-------|--------|-----------|------------|
| HPLT monolingual | 2.15B | ~5 GB | Any GPU with 8+ GB |
| EuroLLM-22B-2512 | 22.6B | ~45 GB | 2x A100 40GB or 1x A100 80GB |

### Total runtime

With position-bias debiasing (2× orderings per question, plus one extra pass per 1-10 question for "10" resolution), each HPLT job runs ~500 forward passes and each EuroLLM job runs ~500 forward passes.

| Task | # Jobs | Time per job | Total wall-clock |
|------|--------|-------------|-----------------|
| 22 HPLT monolingual models | 22 | ~10 min each | ~10 min (parallel) |
| EuroLLM-22B × 22 languages | 22 | ~45 min each | ~45 min (parallel) |
| **Total** | **44** | | **~45 min** (if 22+ GPUs) or ~20 hrs (serial on 1 GPU) |

### Storage
- Model weights: ~4.3 GB per HPLT model × 22 = ~95 GB (downloaded to shared cache)
- EuroLLM-22B: ~45 GB (BF16)
- Results: <20 MB total (logprob numbers + metadata)
- **Total disk**: ~140 GB for model cache

---

## File Structure

```
eurollm/
├── WVS-8_QUESTIONNAIRE_V11_FINAL_Jan_2024.pdf   # WVS-8 master (for cross-referencing)
├── TRACK_1_PLAN.md                                # This plan
├── evs_questionnaires/                            # Downloaded EVS 2017 field questionnaire PDFs
│   ├── eng.pdf
│   ├── deu.pdf
│   ├── fra.pdf
│   ├── ... (22 languages + EVS master)
│   └── evs_master_en.pdf
├── extract_questions.py                           # Gemini Flash PDF→JSON extraction script
├── questions.json                                 # English master question list
├── translations/
│   ├── eng.json                                   # English
│   ├── deu.json                                   # German
│   ├── fra.json                                   # French
│   └── ...                                        # 22 languages total
├── extract_logprobs.py                            # Main inference script
├── prompt_templates.py                            # Prompt formatting logic
├── analyze.py                                     # Analysis and visualization
├── run_slurm.sh                                   # Per-model SLURM job
├── launch_all.sh                                  # Submit all jobs
├── results/
│   ├── hplt2c_eng_checkpoints_eng.parquet
│   ├── hplt2c_deu_checkpoints_deu.parquet
│   ├── EuroLLM-22B-2512_eng.parquet
│   └── ...
└── figures/
    ├── cultural_map.png
    ├── p_valid_heatmap.png
    ├── position_bias_dist.png
    ├── jsd_heatmap.png
    ├── mono_vs_multi.png
    └── example_distributions.png
```

---

## Implementation Order

1. **Tokenizer verification** — confirm digit tokenization for both Gemma-3 (262K) and EuroLLM (128K) tokenizers; check leading-space variants; verify "10" two-step approach works
2. **Extract questions via Gemini Flash** → `translations/*.json` from EVS PDFs
3. **Align and validate** — cross-language question alignment, build master `questions.json`
4. **Write prompt template code** → `prompt_templates.py` (including reverse-order generation)
5. **Write logprob extraction** → `extract_logprobs.py` (with P_valid, position-bias debiasing, two-step "10", space-variant handling)
6. **Validate on English HPLT model locally** — sanity check (10 questions, both orderings, check P_valid, test "10" resolution)
7. **Write SLURM scripts** → `run_slurm.sh`, `launch_all.sh`
8. **Run on cluster** → submit 44 jobs
9. **Write analysis** → `analyze.py`
10. **Generate figures** → `figures/`

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| 2.15B models too small to produce meaningful survey responses | High | Validate with sanity checks first. Use `P_valid` to quantify how well the model engages with the task. If distributions are near-uniform, the model has no opinion — still an interesting null result. |
| Position bias dominates cultural signal | High | Run every question in forward and reversed order; average the distributions; report bias magnitude. If bias > cultural effect, this is a methodological finding worth reporting. |
| Two-step "10" resolution unreliable | Medium | Validate that P("0"\|"1") + P(terminator\|"1") > 0.80 during sanity check. If not, fall back to 1-9 remapping for those questions. |
| Leading-space token variants split probability mass | Medium | Sum logprobs across `"1"` and `" 1"` variants for each digit. Verify during tokenizer check. |
| EVS questions don't fully align with WVS Wave 8 | Medium | ~70% common core. Cross-reference with WVS-8 master PDF. Report which questions are EVS-only. The Inglehart-Welzel composite indices are well-covered in the common core. |
| Gemini Flash misextracts questions from PDFs | Medium | Validate English extraction against EVS master questionnaire. Spot-check 2-3 other languages manually. |
| `P_valid` too low — model doesn't treat prompt as a question | Medium | Report P_valid per question. If consistently low across models, try alternative prompt suffixes (e.g., "The answer is" instead of "Answer:"). |
| HPLT and EuroLLM use different tokenizers | Low | Build tokenizer-specific digit-token mappings at startup. The `get_digit_token_ids()` function handles this generically. |
| EuroLLM-22B needs 2× A100 GPUs | Low | Use `device_map="auto"` with `torch_dtype=torch.bfloat16` to shard across 2 GPUs. |

---

## Verification Plan

1. **Tokenizer check**: Verify that digit tokens "1"–"9" (and their space-prefixed variants) are single tokens in *both* the Gemma-3 tokenizer (HPLT) and the EuroLLM tokenizer
2. **BOS token check**: Confirm that `AutoTokenizer` prepends BOS for both model families
3. **Two-step "10" validation**: Verify P("0"|"1") + P(terminator|"1") > 0.80 on English HPLT model
4. **Prompt sanity check**: Run 10 English questions through HPLT English model, inspect logprob distributions and `P_valid`
5. **Directional validation**: Verify that for clear-cut questions (e.g., "stealing property" justifiability), the model's peak probability is in the expected direction
6. **Position bias quantification**: Run 10 questions in both orderings, measure magnitude of position bias
7. **Cross-language consistency check**: For EuroLLM-22B, verify that English and French produce more similar distributions than English and Greek (different cultural clusters)
8. **Gemini Flash extraction validation**: Compare English extraction output against EVS master questionnaire; spot-check German and Polish extractions
9. **Full run**: Submit all 44 SLURM jobs, collect results
10. **Reproducibility**: Run English model twice to verify deterministic logprobs (greedy, no sampling)

---

## Appendix A: Key References

### Foundational methodology

- **Santurkar et al. (2023). "Whose Opinions Do Language Models Reflect?"** ICML 2023. ([paper](https://proceedings.mlr.press/v202/santurkar23a/santurkar23a.pdf))
  Established the logprob-extraction-and-renormalize methodology we use. Showed LLM opinions skew toward liberal/educated US demographics. Their "refusal detection" (measuring P(refusal) separately) isn't needed for base models. Validates our core approach of extracting next-token logprobs for answer tokens and renormalizing.

- **Durmus et al. (2023). "Towards Measuring the Representation of Subjective Global Opinions in Language Models."** Anthropic. ([arxiv:2306.16388](https://arxiv.org/abs/2306.16388))
  Built GlobalOpinionQA from WVS and Pew Global Attitudes data. Showed that prompting in a target language shifts responses but introduces stereotypes, and does NOT reliably match that language's actual survey respondents. Sets expectations: our monolingual models (trained on only one language's data) might do better than simply translating prompts for a multilingual model. Key comparison to make.

### Position bias and debiasing

- **Zheng et al. (2024). "Large Language Models Are Not Robust Multiple Choice Selectors."** ICLR 2024. ([paper](https://openreview.net/forum?id=shr9PXz7T0))
  Proved token bias > position bias — models have a priori preferences for certain option-ID tokens ("A" > "D"). Introduced the PriDe debiasing method. Our use of numeric IDs ("1", "2") instead of letters partially mitigates token bias. Justifies our forward+reversed debiasing approach.

- **Kabir et al. (2025). "Break the Checkbox: Rethinking Cultural Value Evaluation of LLMs."** EMNLP 2025. ([arxiv:2502.08045](https://arxiv.org/abs/2502.08045))
  Directly tested WVS elicitation with 4 probing methods on 9 instruction-tuned models. Found closed-style probing is "inadequate" — reversed ordering improved alignment 3-15%. All statistically significant cross-country correlations came from unconstrained methods, not checkbox-style probing. Our strongest motivation for position-bias control. We can report whether base models show the same issues.

### Probing methodology

- **Shen et al. (2025). "Revisiting LLM Value Probing Strategies."** EMNLP 2025. ([arxiv:2507.13490](https://arxiv.org/abs/2507.13490))
  Compared token-logit, sequence-perplexity, and text-generation probing. Found sequence perplexity most robust but all methods have high variance under perturbation. Critical finding: leading-space tokens (" A" vs "A") affect results. Justifies our leading-space token handling. If we had more compute, we'd also run sequence perplexity as a robustness check.

- **"On the Credibility of Evaluating LLMs using Survey Questions" (2025).** ([arxiv:2602.04033](https://arxiv.org/abs/2602.04033))
  Showed greedy decoding "systematically misestimates" both average alignment and response distributions. Recommends 100+ samples for generation-based methods and multi-metric evaluation. Validates our logprob approach (no sampling needed). Motivates using multiple distance metrics (JSD, KL, Wasserstein) in Step 5.

### Cultural values in LLMs

- **"Randomness, Not Representation" (2025).** ([arxiv:2503.08688](https://arxiv.org/abs/2503.08688))
  Warns that LLM survey outputs may be "artifacts of evaluation design" rather than evidence of inherent biases. Motivates our position-bias controls and our framing as "cultural priors in training data" rather than "LLM beliefs."
