#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -p gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH -t 04:00:00
#SBATCH -o gemma-sae/logs/sample_%j.out
#SBATCH -e gemma-sae/logs/sample_%j.err
#SBATCH --job-name=culture-sample

# Load CUDA if module system is available
module purge 2>/dev/null && module load cuda/12.6 2>/dev/null || true

cd /scratch4/workspace/zevwilson_umass_edu-culture-llm/llm-cultural-values

export HF_HOME=/scratch4/workspace/zevwilson_umass_edu-culture-llm/.hf_cache
export PYTHONPATH="${PYTHONPATH}:gemma-sae"
export PYTHONUNBUFFERED=1

MODEL_ID=$1
MODEL_HF_ID=$2
DB_PATH=${3:-gemma-sae/data/culture.db}
EXTRA_ARGS="${@:4}"

# ── Local-write strategy ──────────────────────────────────────────
LOCAL_DB="/tmp/culture_${MODEL_ID}_${SLURM_JOB_ID}.db"
OUTDIR="gemma-sae/data/job_dbs"
OUTFILE="${OUTDIR}/${MODEL_ID}_${SLURM_JOB_ID}.db"

# Copy DB back on ANY exit (normal, error, SIGTERM from preemption/timeout)
cleanup() {
    echo "Copying local DB back to NFS..."
    mkdir -p "$OUTDIR"
    cp "$LOCAL_DB" "$OUTFILE" 2>/dev/null && echo "Saved to $OUTFILE" || echo "Copy failed"
    rm -f "$LOCAL_DB" "${LOCAL_DB}-wal" "${LOCAL_DB}-shm"
}
trap cleanup EXIT

echo "Model ID: $MODEL_ID | HF ID: $MODEL_HF_ID | DB: $DB_PATH | Extra: $EXTRA_ARGS"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "Local DB: $LOCAL_DB"

mkdir -p gemma-sae/logs

# Copy shared DB to node-local storage
cp "$DB_PATH" "$LOCAL_DB"

# Run sampling against local DB
gemma-sae/.venv/bin/python -m inference.sample \
    --model_id "$MODEL_ID" \
    --model_hf_id "$MODEL_HF_ID" \
    --db "$LOCAL_DB" \
    $EXTRA_ARGS
