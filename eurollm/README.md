# LLM Cultural Values — Track 1: Survey Elicitation

Extract cultural value distributions from base language models using EVS survey questions, and compare to real human survey data.

## Models
- **HPLT-2.15B**: 22 monolingual Gemma-3-based models (`HPLT/hplt2c_{lang}_checkpoints`)
- **EuroLLM-22B**: 1 multilingual model (`utter-project/EuroLLM-22B-2512`)

## Data Flow

```
questionnaires/*.pdf
        |
   extraction/          Gemini Flash extracts structured questions
        |
 data/translations/*.json
        |
   extraction/align_questions.py    Three-pass cross-language alignment
        |
  data/questions.json ──────────────────────┐
        |                                   |
   prompting/ ← inference/           human_data/
                    |                       |
            results/*.parquet    human_data/data/*.parquet
                    |                       |
               analysis/ ←─────────────────┘
                    |
             figures/*.png
```

## Directory Structure

| Directory | Purpose | SPEC |
|---|---|---|
| `extraction/` | PDF → structured JSON | [SPEC.md](extraction/SPEC.md) |
| `prompting/` | Question → prompt string | [SPEC.md](prompting/SPEC.md) |
| `inference/` | Prompt → logprob distributions | [SPEC.md](inference/SPEC.md) |
| `human_data/` | Real EVS survey data processing | [SPEC.md](human_data/SPEC.md) |
| `analysis/` | All analysis + comparison | [SPEC.md](analysis/SPEC.md) |
| `slurm/` | SLURM job scripts | [SPEC.md](slurm/SPEC.md) |
| `data/` | Shared data artifacts (questions.json, translations/) | — |
| `questionnaires/` | Source EVS PDFs | — |
| `results/` | Output parquets | — |
| `figures/` | Generated figures | — |

## Quick Start

```bash
# 1. Extract logprobs (on GPU node via SLURM)
bash eurollm/slurm/launch_all.sh

# 2. Analyze results
eurollm/.venv/bin/python eurollm/analysis/analyze.py all

# 3. Compare to human survey data (requires EVS download)
eurollm/.venv/bin/python eurollm/human_data/process_evs.py \
    --input eurollm/human_data/data/ZA7500.sav \
    --output eurollm/human_data/data/human_distributions.parquet

eurollm/.venv/bin/python eurollm/analysis/compare_to_human.py \
    --human eurollm/human_data/data/human_distributions.parquet
```

## 22 Languages
bul ces dan deu ell eng est fin fra hrv hun ita lit lvs nld pol por ron slk slv spa swe
