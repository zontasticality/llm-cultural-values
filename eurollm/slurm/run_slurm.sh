#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -p gpu-preempt
#SBATCH -G 1
#SBATCH --constraint="a100|l40s|h100"
#SBATCH -t 01:00:00
#SBATCH -o eurollm/logs/extract_%j.out
#SBATCH -e eurollm/logs/extract_%j.err
#SBATCH --job-name=evs-extract

module purge
module load cuda/12.6

cd /scratch4/workspace/zevwilson_umass_edu-culture-llm/llm-cultural-values

export HF_HOME=/scratch4/workspace/zevwilson_umass_edu-culture-llm/.hf_cache
export PYTHONPATH="${PYTHONPATH}:eurollm"
export PYTHONUNBUFFERED=1

MODEL_ID=$1
LANG=$2
OUTPUT=$3
shift 3
EXTRA_ARGS="$@"

echo "Model: $MODEL_ID | Lang: $LANG | Output: $OUTPUT | Extra: $EXTRA_ARGS"

eurollm/.venv/bin/python eurollm/inference/extract_logprobs.py \
    --model_id "$MODEL_ID" \
    --lang "$LANG" \
    --output "$OUTPUT" \
    $EXTRA_ARGS
