# Track 1: WVS Survey Elicitation — Detailed Implementation Plan

## Context

We want to measure the cultural values encoded in pretrained (non-finetuned) European language models by having them "answer" World Values Survey Wave 8 questions and comparing response distributions across languages. The models are:

- **38 HPLT monolingual reference models** (`HPLT/hplt2c_{lang}_checkpoints`, 2.15B params each, LLaMA architecture, Gemma-3 tokenizer with 262K vocab, trained on 100B tokens of single-language web data)
- **EuroLLM-9B-2512** (`utter-project/EuroLLM-9B-2512`, 9.15B params, multilingual across 35 languages)

Each monolingual model gets the WVS questions in its own language. The multilingual model gets every language.

---

## Step 1: Manually Transcribe WVS-8 Questions to JSON

**What**: Transcribe Q1–Q227 (the opinion/values questions, skipping demographics Q228–Q250) from `eurollm/WVS-8_QUESTIONNAIRE_V11_FINAL_Jan_2024.pdf` into a structured `eurollm/questions.json`.

**Schema per question**:
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
  "reverse_coded": false,
  "country_specific": false,
  "notes": ""
}
```

**Response types to handle**:
| Type | Scale | Example Qs | Count (approx) |
|------|-------|------------|-----------------|
| `likert4` | 4-point agree/disagree or importance | Q1-Q6, Q23-Q29 | ~60 |
| `likert5` | 5-point agree/disagree | Q30-Q33, Q119 | ~15 |
| `likert10` | 1-10 scale with anchors | Q38-Q42, Q107-Q111, Q130-Q145 | ~50 |
| `likert9` | 1-9 trust scale | Q164-Q168 | 5 |
| `binary` | yes/no or mentioned/not | Q7-Q22, Q99 | ~20 |
| `categorical` | pick one from list | Q34, Q116, Q122, Q128, Q204 | ~10 |
| `ranked` | first + second choice | Q86-Q89 | 4 |
| `frequency` | how often | Q126-Q127, Q156-Q163 | ~15 |
| `action` | have done / might do / never | Q169-Q176 | 8 |
| `favorability` | 4-point favorable/unfavorable | Q100-Q106 | 7 |
| `confidence` | 4-point confidence in institutions | Q50-Q58, Q69-Q77, Q79-Q85 | ~30 |

**Questions to flag as country-specific** (need adaptation or exclusion):
- Q73 (Head of State) — refers to country leader
- Q78A/B (trust in world leaders) — list experiment, complex format
- Q79 (regional organization) — EU/AU/ASEAN etc.
- Q179 (party vote) — country-specific party list
- Q180-Q188 (election integrity) — may not make sense for LLMs

**Estimated effort**: ~3-4 hours of careful manual work for 227 questions.

**Output**: `eurollm/questions.json`

---

## Step 2: Translate Questions into Target Languages

**The problem**: HPLT monolingual models only understand their own language. We need WVS questions in each of the target languages.

**Available official translations (EVS 2017 / WVS Wave 7 PDFs from GESIS)**:
22 of 24 EU languages have official WVS/EVS field questionnaire translations. Missing: Irish, Maltese. These are also the two languages without HPLT models, so they're excluded anyway.

**Approach — two-phase**:

**Phase A (fast, for initial results)**: Machine-translate the English JSON using a high-quality MT system.
- Use EuroLLM-9B-Instruct or Google Translate API
- Output: `eurollm/translations/{lang_code}.json` for each language
- This introduces some cultural bias from the MT system, which we note as a limitation

**Phase B (higher quality, optional follow-up)**: Download official EVS 2017 PDFs from GESIS, extract text, and align with our English questions.
- Register at GESIS (free), download 22 PDFs
- Use pdfplumber to extract text
- Manually align each question with the English source
- Replaces machine translations with validated human translations

**For the initial run, Phase A is sufficient.** The primary signal we're measuring is the model's training-data cultural prior, not translation quality. Phase B can validate findings later.

**Languages covered** (matching HPLT models to EU official languages):

| EU Language | HPLT Model | WVS Translation | Status |
|-------------|-----------|-----------------|--------|
| Bulgarian | `hplt2c_bul` | YES | Ready |
| Croatian | `hplt2c_hrv` | YES | Ready |
| Czech | `hplt2c_ces` | YES | Ready |
| Danish | `hplt2c_dan` | YES | Ready |
| Dutch | `hplt2c_nld` | YES | Ready |
| English | `hplt2c_eng` | YES (source) | Ready |
| Estonian | `hplt2c_est` | YES | Ready |
| Finnish | `hplt2c_fin` | YES | Ready |
| French | `hplt2c_fra` | YES | Ready |
| German | `hplt2c_deu` | YES | Ready |
| Greek | `hplt2c_ell` | YES | Ready |
| Hungarian | `hplt2c_hun` | YES | Ready |
| Irish | NONE | NONE | EXCLUDED |
| Italian | `hplt2c_ita` | YES | Ready |
| Latvian | `hplt2c_lvs` | YES | Ready |
| Lithuanian | `hplt2c_lit` | YES | Ready |
| Maltese | NONE | NONE | EXCLUDED |
| Polish | `hplt2c_pol` | YES | Ready |
| Portuguese | `hplt2c_por` | YES | Ready |
| Romanian | `hplt2c_ron` | YES | Ready |
| Slovak | `hplt2c_slk` | YES | Ready |
| Slovenian | `hplt2c_slv` | YES | Ready |
| Spanish | `hplt2c_spa` | YES | Ready |
| Swedish | `hplt2c_swe` | YES | Ready |

**= 22 monolingual models + EuroLLM-9B tested in all 22 languages = 44 total experimental conditions**

---

## Step 3: Prompt Template Design

**Design principle**: Minimize framing. No system prompt, no mention of "survey" or "WVS". Present as a bare completion task.

**Template for Likert-scale questions**:
```
{question_text}
{option_1_value}. {option_1_label}
{option_2_value}. {option_2_label}
...
Answer:
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
1 = {left_anchor} ... 10 = {right_anchor}
Answer:
```

**For binary (mentioned/not mentioned) questions**: Convert to agree/disagree format since the "show card and select" interaction doesn't translate to text completion.

**Key design decisions**:
- No system prompt at all (base models don't use them)
- The word "Answer:" (translated) is the final token before we measure logprobs
- We extract logprobs for tokens "1", "2", "3", ... at the next-token position
- The prompt fits in <200 tokens (well within model context limits)

**Validation**: Before the full run, test 10 questions on the English HPLT model to verify:
1. Logprob distributions are non-uniform (model has preferences)
2. Distributions make directional sense (e.g., "stealing" skews toward "never justifiable")
3. The tokenizer encodes "1", "2", etc. as single tokens

---

## Step 4: Logprob Extraction Pipeline

**File**: `eurollm/extract_logprobs.py`

**Core logic**:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, json

def extract_logprobs(model, tokenizer, prompt, valid_tokens):
    """Extract logprobs for valid response tokens at the answer position."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    # Get logits at the last token position (predicting next token)
    logits = outputs.logits[0, -1, :]
    # Extract logprobs for valid response token IDs
    log_probs = torch.log_softmax(logits, dim=-1)
    result = {}
    for token_str in valid_tokens:
        token_id = tokenizer.encode(token_str, add_special_tokens=False)
        # Handle multi-token numbers (shouldn't happen for 1-10)
        if len(token_id) == 1:
            result[token_str] = log_probs[token_id[0]].item()
    return result

def run_survey(model_id, questions_file, lang_code, output_file):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    questions = json.load(open(questions_file))
    results = []
    for q in questions:
        prompt = format_prompt(q, lang_code)
        valid_tokens = [str(opt["value"]) for opt in q["options"]]
        logprobs = extract_logprobs(model, tokenizer, prompt, valid_tokens)
        # Normalize to probability distribution
        probs = softmax_dict(logprobs)
        results.append({"question_id": q["id"], "logprobs": logprobs, "probs": probs})
    save_results(results, output_file)
```

**Output format**: One parquet file per model-language pair:
```
eurollm/results/{model_name}_{lang_code}.parquet
```

Columns: `question_id, response_value, logprob, prob`

---

## Step 5: SLURM Job Scripts

**File**: `eurollm/run_slurm.sh`

**Strategy**: One SLURM job per model. Each job loads the model once and runs all 226 questions sequentially (takes ~2-5 minutes per model).

```bash
#!/bin/bash
#SBATCH --job-name=wvs-{model_name}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=24G          # 2.15B model needs ~8GB VRAM + overhead
#SBATCH --time=00:30:00    # 30 min is generous for 226 questions
#SBATCH --output=logs/%x_%j.out

MODEL_ID=$1   # e.g., HPLT/hplt2c_eng_checkpoints
LANG=$2       # e.g., eng
QUESTIONS=eurollm/translations/${LANG}.json

python eurollm/extract_logprobs.py \
    --model_id $MODEL_ID \
    --questions $QUESTIONS \
    --lang $LANG \
    --output eurollm/results/${MODEL_ID##*/}_${LANG}.parquet
```

**Launcher script** (`eurollm/launch_all.sh`):
```bash
# Submit all 22 monolingual models
for lang in bul hrv ces dan nld eng est fin fra deu ell hun ita lvs lit pol por ron slk slv spa swe; do
    sbatch run_slurm.sh "HPLT/hplt2c_${lang}_checkpoints" "$lang"
done

# Submit EuroLLM-9B for each language (needs more VRAM)
for lang in bul hrv ces dan nld eng est fin fra deu ell hun ita lvs lit pol por ron slk slv spa swe; do
    sbatch --mem=48G --gres=gpu:a100:1 run_slurm.sh "utter-project/EuroLLM-9B-2512" "$lang"
done
```

---

## Step 6: Analysis Pipeline

**File**: `eurollm/analyze.py`

1. **Aggregate results**: Load all parquet files into a single DataFrame
2. **Compute summary statistics**: Mean response, entropy, per question per model
3. **Cross-model comparison**: For each question, compare monolingual vs multilingual response distributions
4. **Inglehart-Welzel mapping**: Compute Traditional/Secular-rational and Survival/Self-expression scores from WVS composite indices
5. **Visualization**: Plot cultural map of model-language pairs, compare to real WVS country data
6. **Statistical tests**: KL divergence between model distributions and real WVS country distributions

---

## Compute Requirements

### Per-model VRAM

| Model | Params | BF16 VRAM | INT8 VRAM | GPU needed |
|-------|--------|-----------|-----------|------------|
| HPLT monolingual | 2.15B | ~5 GB | ~3 GB | Any GPU with 8+ GB |
| EuroLLM-9B-2512 | 9.15B | ~20 GB | ~12 GB | 1x A100 40GB |

### Total runtime estimate

| Task | # Jobs | Time per job | Total wall-clock |
|------|--------|-------------|-----------------|
| 22 HPLT monolingual models | 22 | ~5 min each | ~5 min (parallel) |
| EuroLLM-9B × 22 languages | 22 | ~10 min each | ~10 min (parallel) |
| **Total** | **44** | | **~15 min** (if 22+ GPUs) or ~3.5 hrs (serial on 1 GPU) |

### Storage
- Model weights: ~4.3 GB per HPLT model × 22 = ~95 GB (downloaded to shared cache)
- EuroLLM-9B: ~18 GB
- Results: <10 MB total (just logprob numbers)
- **Total disk**: ~115 GB for model cache

### Network
- First run downloads all models from HuggingFace (~115 GB)
- Subsequent runs use cached weights

**Bottom line**: This entire experiment fits comfortably on a single A100 40GB running jobs serially in ~3.5 hours, or on a cluster with 22 GPUs in ~15 minutes. No multi-GPU parallelism needed for any single model.

---

## File Structure

```
eurollm/
├── WVS-8_QUESTIONNAIRE_V11_FINAL_Jan_2024.pdf   # Source document
├── questions.json                                 # Manually transcribed English questions
├── translations/
│   ├── eng.json                                   # English (= questions.json)
│   ├── deu.json                                   # German
│   ├── fra.json                                   # French
│   └── ...                                        # 22 languages total
├── extract_logprobs.py                            # Main inference script
├── translate_questions.py                         # MT script for Phase A translations
├── prompt_templates.py                            # Prompt formatting logic
├── analyze.py                                     # Analysis and visualization
├── run_slurm.sh                                   # Per-model SLURM job
├── launch_all.sh                                  # Submit all jobs
├── results/
│   ├── hplt2c_eng_checkpoints_eng.parquet
│   ├── hplt2c_deu_checkpoints_deu.parquet
│   ├── EuroLLM-9B-2512_eng.parquet
│   └── ...
└── figures/
    └── cultural_map.png
```

---

## Implementation Order

1. **Transcribe questions** → `questions.json` (~3-4 hrs manual work)
2. **Write prompt template code** → `prompt_templates.py` (~30 min)
3. **Write translation script** → `translate_questions.py` (~1 hr)
4. **Generate translations** → `translations/*.json` (~30 min compute)
5. **Write logprob extraction** → `extract_logprobs.py` (~1 hr)
6. **Validate on English model locally** → sanity check (~30 min)
7. **Write SLURM scripts** → `run_slurm.sh`, `launch_all.sh` (~30 min)
8. **Run on cluster** → submit jobs (~15 min - 3.5 hrs depending on parallelism)
9. **Write analysis** → `analyze.py` (~2 hrs)
10. **Generate figures** → `figures/` (~1 hr)

**Total estimated effort**: ~10-12 hours of implementation + compute time.

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| 2.15B models too small to produce meaningful survey responses | High | Validate with sanity checks first. If distributions are near-uniform, the model has no opinion — still an interesting null result. |
| Token "1", "2" etc. not single tokens in Gemma-3 tokenizer | Medium | Check tokenizer encoding before running. Fall back to constrained generation if needed. |
| Machine translations distort cultural content | Medium | Note as limitation. Phase B with official translations can validate. |
| HPLT checkpoint format incompatible | Low | The `_checkpoints` repos suggest intermediate checkpoints; verify the final checkpoint is accessible. |
| EuroLLM-9B-2512 stored in FP32 on HuggingFace (~36 GB download) | Low | Load with `torch_dtype=torch.bfloat16` to keep VRAM manageable. |

---

## Verification Plan

1. **Tokenizer check**: Verify that digit tokens "1"-"10" are single tokens in the Gemma-3 tokenizer
2. **Prompt sanity check**: Run 10 English questions through HPLT English model, inspect logprob distributions
3. **Directional validation**: Verify that for clear-cut questions (e.g., "stealing property" justifiability), the model's peak probability is in the expected direction
4. **Cross-language consistency check**: For EuroLLM-9B, verify that English and French (closely related cultural cluster) produce more similar distributions than English and Chinese
5. **Full run**: Submit all 44 SLURM jobs, collect results
6. **Reproducibility**: Run English model twice to verify deterministic logprobs (temperature=0, same results)
