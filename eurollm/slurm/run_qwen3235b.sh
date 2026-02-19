#!/bin/bash
#SBATCH -c 8
#SBATCH --mem=320G
#SBATCH -p gpu-preempt
#SBATCH -G 4
#SBATCH --constraint="a100"
#SBATCH -t 08:00:00
#SBATCH -o eurollm/logs/qwen3235b_%j.out
#SBATCH -e eurollm/logs/qwen3235b_%j.err
#SBATCH --job-name=evs-qwen3235b

# Qwen3-235B-A22B: MoE model, 235B total / 22B active params
# Uses int4 (NF4) quantization (~120GB weights + buffers) → needs 4x A100 80GB
# Uses device_map="auto" for tensor parallelism across GPUs

module purge
module load cuda/12.6

cd /scratch4/workspace/zevwilson_umass_edu-culture-llm/llm-cultural-values

export HF_HOME=/scratch4/workspace/zevwilson_umass_edu-culture-llm/.hf_cache
export PYTHONPATH="${PYTHONPATH}:eurollm"

LANG=$1
CONFIG="eurollm/data/prompt_config.json"
MODEL_ID="Qwen/Qwen3-235B-A22B"
OUTPUT="eurollm/results/qwen3235b_${LANG}.parquet"

echo "Qwen3-235B-A22B | Lang: $LANG | Output: $OUTPUT"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

eurollm/.venv/bin/python eurollm/inference/extract_logprobs.py \
    --model_id "$MODEL_ID" \
    --lang "$LANG" \
    --output "$OUTPUT" \
    --permutations 6 \
    --prompt-config "$CONFIG" \
    --dtype int4
