#!/bin/bash
#SBATCH -c 2
#SBATCH --mem=96G
#SBATCH -p gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH -t 24:00:00
#SBATCH -o gemma-sae/logs/classify_%j.out
#SBATCH -e gemma-sae/logs/classify_%j.err
#SBATCH --job-name=culture-classify

module purge 2>/dev/null && module load cuda/12.6 2>/dev/null || true

cd /scratch4/workspace/zevwilson_umass_edu-culture-llm/llm-cultural-values

export HF_HOME=/scratch4/workspace/zevwilson_umass_edu-culture-llm/.hf_cache
export PYTHONPATH="${PYTHONPATH}:gemma-sae"
export PYTHONUNBUFFERED=1

DB_PATH=${1:-gemma-sae/data/culture.db}
MODEL=${2:-google/gemma-3-27b-it}
CLASSIFIER=${3:-gemma3_27b_it}

echo "DB: $DB_PATH | Model: $MODEL | Classifier: $CLASSIFIER"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"

# Use node-local DB to avoid NFS issues
LOCAL_DB="/tmp/culture_classify_${SLURM_JOB_ID}.db"
OUTDIR="gemma-sae/data/job_dbs"
OUTFILE="${OUTDIR}/classify_${CLASSIFIER}_${SLURM_JOB_ID}.db"

cleanup() {
    echo "Copying classified DB back to NFS..."
    mkdir -p "$OUTDIR"
    cp "$LOCAL_DB" "$OUTFILE" 2>/dev/null && echo "Saved to $OUTFILE" || echo "Copy failed"
    rm -f "$LOCAL_DB"
}
trap cleanup EXIT

cp "$DB_PATH" "$LOCAL_DB"

gemma-sae/.venv/bin/python -m classify.classify_local \
    --db "$LOCAL_DB" \
    --model "$MODEL" \
    --classifier-name "$CLASSIFIER" \
    --batch-size 4
