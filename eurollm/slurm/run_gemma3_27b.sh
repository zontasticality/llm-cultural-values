#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -p gpu-preempt
#SBATCH -G 1
#SBATCH --constraint="a100|h100"
#SBATCH -t 02:00:00
#SBATCH -o eurollm/logs/gemma3_27b_%j.out
#SBATCH -e eurollm/logs/gemma3_27b_%j.err
#SBATCH --job-name=evs-gemma3-27b

# Gemma-3-27B: 27B params, ~54GB in bf16 → needs A100 80GB or H100
# Variants: pt (base) or it (instruction-tuned with chat template)

module purge
module load cuda/12.6

cd /scratch4/workspace/zevwilson_umass_edu-culture-llm/llm-cultural-values

export HF_HOME=/scratch4/workspace/zevwilson_umass_edu-culture-llm/.hf_cache
export PYTHONPATH="${PYTHONPATH}:eurollm"
export PYTHONUNBUFFERED=1

LANG=$1
VARIANT=${2:-pt}  # "pt" (base) or "it" (instruction-tuned)
MODEL_ID="google/gemma-3-27b-${VARIANT}"
OUTPUT="eurollm/results/gemma3_27b_${VARIANT}_${LANG}.parquet"

echo "Gemma-3-27B-${VARIANT} | Lang: $LANG | Output: $OUTPUT"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

EXTRA_ARGS=""
if [ "$VARIANT" = "it" ]; then
    EXTRA_ARGS="--chat-template"
fi

eurollm/.venv/bin/python eurollm/inference/extract_logprobs.py \
    --model_id "$MODEL_ID" \
    --lang "$LANG" \
    --output "$OUTPUT" \
    $EXTRA_ARGS
