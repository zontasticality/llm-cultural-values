#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -p gpu-preempt
#SBATCH --constraint="a100|l40s|h100"
#SBATCH -t 04:00:00
#SBATCH -o eurollm/logs/db_extract_%j.out
#SBATCH -e eurollm/logs/db_extract_%j.err
#SBATCH --job-name=evs-db

module purge
module load cuda/12.6

cd /scratch4/workspace/zevwilson_umass_edu-culture-llm/llm-cultural-values

export HF_HOME=/scratch4/workspace/zevwilson_umass_edu-culture-llm/.hf_cache
export PYTHONPATH="${PYTHONPATH}:eurollm"
export PYTHONUNBUFFERED=1

MODEL_ID=$1
MODEL_HF_ID=$2
DB_PATH=${3:-eurollm/data/survey.db}
EXTRA_ARGS="${@:4}"

echo "Model ID: $MODEL_ID | HF ID: $MODEL_HF_ID | DB: $DB_PATH | Extra: $EXTRA_ARGS"

eurollm/.venv/bin/python eurollm/inference/extract_logprobs.py db \
    --model_id "$MODEL_ID" \
    --model_hf_id "$MODEL_HF_ID" \
    --db "$DB_PATH" \
    $EXTRA_ARGS
