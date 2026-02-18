# Human Data Module

## Purpose
Process real EVS 2017 survey data into country-level response distributions for comparison with LLM outputs.

## Inputs
- `data/ZA7500.sav` — EVS 2017 Integrated Dataset (SPSS format, user must download from GESIS)
- `../data/questions.json` — Canonical question set (for v-code mapping)

## Outputs
- `data/human_distributions.parquet` — Country-level weighted response distributions

### Output schema
| Column | Type | Description |
|---|---|---|
| lang | str | Language code (e.g. "eng", "deu") |
| question_id | str | Canonical v-code (e.g. "v1", "v39") |
| response_value | int | Response option value |
| prob_human | float | Weighted probability |
| n_respondents | int | Total respondents for this country |
| n_valid | int | Respondents with valid (non-missing) responses |

## Key Functions / Public API

### `process_evs.py`
- CLI: `python human_data/process_evs.py --input <sav_path> --questions <json_path> --output <parquet_path>`
- Reads SPSS via `pyreadstat`, filters by country, drops missing values, computes weighted distributions
- Uses `gweight` for cross-national comparison (falls back to `dweight` or unweighted)

## Dependencies
- External: `pyreadstat`, `pandas`, `numpy`
- No cross-module imports
