# Analysis Module

## Purpose
Analyze LLM cultural value distributions and compare to human survey ground truth.

## Inputs
- `../results/*.parquet` — LLM extraction results (44 model-lang pairs)
- `../data/questions.json` — Question metadata
- `../data/rephrasings.json` — Rephrase experiment metadata
- `../human_data/data/human_distributions.parquet` — EVS human survey distributions

## Outputs
- `../figures/*.png` — All generated figures
- `../summary_stats.parquet` — Per-question summary statistics
- `../jsd_matrix.npz` — 44x44 JSD distance matrix

## Key Functions / Public API

### `analyze.py`
- CLI: `python analysis/analyze.py <command> [--results-dir ...] [--questions ...] [--figures-dir ...]`
- Commands: `quality`, `bias`, `summary`, `distance`, `pca`, `tsne`, `umap`, `examples`, `rephrase`, `all`
- `load_all_results(results_dir)` — Load and combine all parquet result files
- `compute_jsd_matrix(df)` — Pairwise JSD between all model-lang pairs
- `compute_summary_stats(df, questions)` — Expected value, entropy, mode per question

### `compare_to_human.py`
- CLI: `python analysis/compare_to_human.py --results-dir ... --human ... --questions ... --figures-dir ...`
- Compares LLM distributions to human survey data
- Generates 5 figures: scatter, JSD by qtype, mono vs multi, heatmap, best/worst examples

### `constants.py`
- Shared constants: `LANG_NAMES`, `CULTURAL_CLUSTERS`, `LANG_TO_CLUSTER`, `CLUSTER_COLORS`
- Model display constants: `MODEL_COLORS`, `MODEL_LABELS`, `MODEL_MARKERS`, `MODEL_SIZES`, `MODEL_SIZES_SMALL`
- `ORDINAL_TYPES` — set of ordinal response type strings

## Dependencies
- External: `matplotlib`, `seaborn`, `scipy`, `sklearn`, `numpy`, `pandas`
- Internal: `analysis.constants` (shared constants between analyze.py and compare_to_human.py)
