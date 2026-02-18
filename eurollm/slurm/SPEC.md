# SLURM Module

## Purpose
Job submission scripts for running inference on GPU cluster nodes.

## Scripts

### `run_slurm.sh`
- Main SBATCH script for single model-lang extraction
- Args: `$1=MODEL_ID $2=LANG $3=OUTPUT`
- Requests: 4 cores, 32G RAM, 1 GPU, 1 hour

### `launch_all.sh`
- Submits all 44 jobs (22 HPLT + 22 EuroLLM)
- EuroLLM jobs request 96G RAM and A100/H100 constraint

### `run_rephrase.sh`
- Submits 2 rephrase sensitivity experiment jobs (HPLT + EuroLLM, English only)

### `run_validate.sh`
- Submits 1 validation job (HPLT English, 10 questions with diagnostics)

### `run_investigate.sh`
- Submits 1 tokenizer/prompt format investigation job

## Environment
- CUDA 12.6 via module system
- HF cache at `/scratch4/workspace/zevwilson_umass_edu-culture-llm/.hf_cache`
- `PYTHONPATH` includes `eurollm/` for cross-module imports
- Python from `eurollm/.venv/bin/python`
