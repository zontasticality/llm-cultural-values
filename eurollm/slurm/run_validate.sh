#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -p gpu-preempt
#SBATCH -G 1
#SBATCH -t 01:00:00
#SBATCH --constraint="a100|l40s|h100"
#SBATCH -o eurollm/logs/validate_%j.out
#SBATCH -e eurollm/logs/validate_%j.err
#SBATCH --job-name=evs-validate

module purge
module load cuda/12.6

cd /scratch4/workspace/zevwilson_umass_edu-culture-llm/llm-cultural-values

export HF_HOME=/scratch4/workspace/zevwilson_umass_edu-culture-llm/.hf_cache
export PYTHONPATH="${PYTHONPATH}:eurollm"

eurollm/.venv/bin/python eurollm/inference/extract_logprobs.py \
    --model_id HPLT/hplt2c_eng_checkpoints \
    --lang eng \
    --output eurollm/results/validate_test.parquet \
    --validate
