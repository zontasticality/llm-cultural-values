#!/bin/bash
#SBATCH -c 8
#SBATCH --mem=192G
#SBATCH -p gpu-preempt
#SBATCH --gres=gpu:a100:2
#SBATCH --constraint="a100-80g"
#SBATCH -t 04:00:00
#SBATCH --requeue
#SBATCH -o eurollm/logs/qwen_%j.out
#SBATCH -e eurollm/logs/qwen_%j.err
#SBATCH --job-name=evs-qwen

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
