#!/bin/bash
# Launch all DB-based inference jobs.
# HPLT models: 1h, 1 GPU, 32G (2.15B params, ~1K prompts per model)
# EuroLLM-22B: 4h, 1 GPU, 48G (22B params, ~21K prompts)
# Gemma-3-27B-PT: 4h, 1 GPU, 48G (27B params, ~21K prompts)
# Qwen3-235B: 8h, 4x A100 (single node), 320G (MoE 235B, int4 quantization)
#
# Usage: bash eurollm/slurm/launch_db.sh [--dry-run]

set -e
cd /scratch4/workspace/zevwilson_umass_edu-culture-llm/llm-cultural-values

DRY_RUN=false
if [ "$1" = "--dry-run" ]; then
    DRY_RUN=true
    echo "=== DRY RUN — not actually submitting ==="
fi

DB=eurollm/data/survey.db
SCRIPT=eurollm/slurm/run_db.sh

submit() {
    local slurm_args=$1
    local model_id=$2
    local hf_id=$3
    local extra="${@:4}"
    if $DRY_RUN; then
        echo "  [dry-run] sbatch $slurm_args $SCRIPT $model_id $hf_id $DB $extra"
    else
        JOB=$(sbatch --parsable $slurm_args $SCRIPT "$model_id" "$hf_id" "$DB" $extra)
        echo "  Submitted $model_id → job $JOB"
    fi
}

# HPLT: 2.15B params, ~1K prompts → 1h, 32G is plenty
HPLT_ARGS="-G 1 --mem=32G -t 01:00:00"
# Large models: 22-27B params, ~21K prompts → 4h, 48G
LARGE_ARGS="-G 1 --mem=48G -t 04:00:00"
# Qwen MoE: 235B total, int4 quant → 4x A100 80GB on a single node, 8h
QWEN_ARGS="-G 4 --nodes=1 --mem=320G -t 08:00:00 -c 8 --constraint=a100"

echo "=== HPLT 2.15B monolingual models (22 jobs) ==="
for lang in bul ces dan deu ell eng est fin fra hrv hun ita lit lvs nld pol por ron slk slv spa swe; do
    submit "$HPLT_ARGS" "hplt2c_${lang}" "HPLT/hplt2c_${lang}_checkpoints" --lang "$lang"
done

echo ""
echo "=== EuroLLM-22B (1 job, all languages) ==="
submit "$LARGE_ARGS" eurollm22b "utter-project/EuroLLM-22B-2512"

echo ""
echo "=== Gemma-3-27B-PT (1 job, all languages) ==="
submit "$LARGE_ARGS" gemma3_27b_pt "google/gemma-3-27b-pt"

echo ""
echo "=== Qwen3-235B-A22B (1 job, all languages, int4) ==="
submit "$QWEN_ARGS" qwen3235b "Qwen/Qwen3-235B-A22B" --dtype int4

echo ""
echo "Total: 25 jobs"
