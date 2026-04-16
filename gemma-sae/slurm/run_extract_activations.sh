#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -p gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH -t 02:00:00
#SBATCH -o gemma-sae/logs/extract_activations_%j.out
#SBATCH -e gemma-sae/logs/extract_activations_%j.err
#SBATCH --job-name=extract-acts

module purge 2>/dev/null && module load cuda/12.6 2>/dev/null || true

cd /scratch4/workspace/zevwilson_umass_edu-culture-llm/llm-cultural-values

export HF_HOME=/scratch4/workspace/zevwilson_umass_edu-culture-llm/.hf_cache
export PYTHONPATH="${PYTHONPATH}:gemma-sae"
export PYTHONUNBUFFERED=1

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"

mkdir -p gemma-sae/logs

# Step 1: Extract activations (GPU)
echo "=== Extracting activations ==="
gemma-sae/.venv/bin/python -m steering.extract_activations \
    --model google/gemma-3-27b-pt \
    --flores-dir gemma-sae/data/probes/flores_200 \
    --output-dir gemma-sae/data/activations \
    --layers 31 40 53 \
    --batch-size 8 \
    --pool last

echo "=== Activation extraction complete ==="

# Step 2: Score features (CPU-bound, SAE encoding is just matmuls)
echo "=== Scoring features ==="
gemma-sae/.venv/bin/python -m steering.score_features \
    --activations-dir gemma-sae/data/activations \
    --output-dir gemma-sae/data/feature_scores \
    --layers 31 40 53 \
    --sae-width 256k \
    --sae-l0 medium \
    --top-k 20

echo "=== Done ==="
