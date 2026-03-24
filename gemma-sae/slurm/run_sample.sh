#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -p gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH -t 04:00:00
#SBATCH -o gemma-sae/logs/sample_%j.out
#SBATCH -e gemma-sae/logs/sample_%j.err
#SBATCH --job-name=culture-sample

# Overridable via sbatch flags: --mem, --constraint, -t, etc.

# Load CUDA if module system is available (not all nodes have it)
module purge 2>/dev/null && module load cuda/12.6 2>/dev/null || true

cd /scratch4/workspace/zevwilson_umass_edu-culture-llm/llm-cultural-values

export HF_HOME=/scratch4/workspace/zevwilson_umass_edu-culture-llm/.hf_cache
export PYTHONPATH="${PYTHONPATH}:gemma-sae"
export PYTHONUNBUFFERED=1

MODEL_ID=$1
MODEL_HF_ID=$2
DB_PATH=${3:-gemma-sae/data/culture.db}
EXTRA_ARGS="${@:4}"

echo "Model ID: $MODEL_ID | HF ID: $MODEL_HF_ID | DB: $DB_PATH | Extra: $EXTRA_ARGS"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"

mkdir -p gemma-sae/logs

gemma-sae/.venv/bin/python -m inference.sample \
    --model_id "$MODEL_ID" \
    --model_hf_id "$MODEL_HF_ID" \
    --db "$DB_PATH" \
    $EXTRA_ARGS
