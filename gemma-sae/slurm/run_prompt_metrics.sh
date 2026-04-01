#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -p gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH -t 08:00:00
#SBATCH -o gemma-sae/logs/prompt_metrics_%j.out
#SBATCH -e gemma-sae/logs/prompt_metrics_%j.err
#SBATCH --job-name=prompt-metrics

module purge 2>/dev/null && module load cuda/12.6 2>/dev/null || true

cd /scratch4/workspace/zevwilson_umass_edu-culture-llm/llm-cultural-values

export HF_HOME=/scratch4/workspace/zevwilson_umass_edu-culture-llm/.hf_cache
export PYTHONPATH="${PYTHONPATH}:gemma-sae"
export PYTHONUNBUFFERED=1

DB_PATH=${1:-gemma-sae/data/culture.db}

# Use node-local DB
LOCAL_DB="/tmp/prompt_metrics_${SLURM_JOB_ID}.db"
OUTDIR="gemma-sae/data/job_dbs"
OUTFILE="${OUTDIR}/prompt_metrics_${SLURM_JOB_ID}.db"

cleanup() {
    echo "Copying DB back to NFS..."
    mkdir -p "$OUTDIR"
    cp "$LOCAL_DB" "$OUTFILE" 2>/dev/null && echo "Saved to $OUTFILE" || echo "Copy failed"
    rm -f "$LOCAL_DB" "${LOCAL_DB}-wal" "${LOCAL_DB}-shm"
}
trap cleanup EXIT

echo "DB: $DB_PATH"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "Local DB: $LOCAL_DB"

mkdir -p gemma-sae/logs
cp "$DB_PATH" "$LOCAL_DB"

# Run all three multilingual models sequentially (largest first while GPU memory is clean)
# Each model loads, computes metrics for all prompts, then unloads
gemma-sae/.venv/bin/python -m inference.prompt_metrics \
    --db "$LOCAL_DB" \
    --models gemma3_27b_pt gemma3_12b_pt eurollm22b \
    --batch-size 64
