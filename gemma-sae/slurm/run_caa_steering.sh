#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -p gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH -t 02:00:00
#SBATCH -o gemma-sae/logs/caa_steering_%j.out
#SBATCH -e gemma-sae/logs/caa_steering_%j.err
#SBATCH --job-name=caa-steer

module purge 2>/dev/null && module load cuda/12.6 2>/dev/null || true

cd /scratch4/workspace/zevwilson_umass_edu-culture-llm/llm-cultural-values

export HF_HOME=/scratch4/workspace/zevwilson_umass_edu-culture-llm/.hf_cache
export PYTHONPATH="${PYTHONPATH}:gemma-sae"
export PYTHONUNBUFFERED=1

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"

mkdir -p gemma-sae/logs

# CAA sanity check: sweep layers 25-60 (middle-to-late), scales 1-20
# Test 3 target languages: Finnish, Chinese, Arabic
gemma-sae/.venv/bin/python -m steering.caa_steering \
    --model google/gemma-3-27b-pt \
    --flores-dir gemma-sae/data/probes/flores_200 \
    --source-lang eng \
    --target-langs fin zho ara \
    --layers 25 30 35 40 45 50 55 60 \
    --scales 3 10 20 \
    --max-sentences 100 \
    --prompt "The most important thing in life is" \
    --n-samples 2 --max-tokens 50

echo "=== Done ==="
