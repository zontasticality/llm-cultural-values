#!/bin/bash
# Launch full sampling run on Unity.
# One job per model group — each loops through all its languages via DB queries.
# The sampling pipeline is resumable: pull + merge + resubmit to continue
# from where preempted jobs left off.
#
# Usage: bash gemma-sae/slurm/launch_full.sh [--dry-run]

set -uo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN ==="
fi

SCRIPT="gemma-sae/slurm/run_sample.sh"
DB="gemma-sae/data/culture.db"
N_SAMPLES="--n-samples 200"

EU_LANGS=(bul ces dan deu ell eng est fin fra hrv hun ita lit lvs nld pol por ron slk slv spa swe)

submit() {
    local name="$1" mem="$2" time="$3" constraint="$4"
    shift 4
    local cmd="sbatch --job-name=$name --mem=$mem -t $time"
    if [[ -n "$constraint" ]]; then
        cmd="$cmd --constraint=\"$constraint\""
    fi
    cmd="$cmd $SCRIPT $@"

    if $DRY_RUN; then
        echo "  $cmd"
    else
        echo -n "  $name: "
        eval "$cmd"
    fi
}

# ── Multilingual models ──────────────────────────────────────────

echo "=== Gemma 3 27B PT (27 langs) ==="
submit "full-g27b" "96G" "12:00:00" "a100" \
    gemma3_27b_pt google/gemma-3-27b-pt "$DB" \
    $N_SAMPLES --temperature 1.0

echo "=== Gemma 3 12B PT (27 langs) ==="
submit "full-g12b" "48G" "08:00:00" "a100|l40s" \
    gemma3_12b_pt google/gemma-3-12b-pt "$DB" \
    $N_SAMPLES --temperature 1.0

echo "=== EuroLLM-22B (all langs — expanded langs will produce junk, filtered in analysis) ==="
submit "full-euro" "48G" "04:00:00" "a100|h100" \
    eurollm22b utter-project/EuroLLM-22B-2512 "$DB" \
    $N_SAMPLES --temperature 1.0 --backend vllm

# ── HPLT monolingual models ─────────────────────────────────────

echo "=== HPLT monolingual (22 models) ==="
for lang in "${EU_LANGS[@]}"; do
    submit "full-hplt-$lang" "32G" "01:00:00" "" \
        "hplt2c_${lang}" "HPLT/hplt2c_${lang}_checkpoints" "$DB" \
        $N_SAMPLES --temperature 0.8 --lang "$lang"
done

echo ""
echo "Submitted: 3 multilingual + 22 HPLT = 25 jobs total."
echo "Use 'squeue -u \$USER' to monitor."
echo ""
echo "After completion:"
echo "  make pull && make merge   # combine per-job DBs into culture.db"
echo "  make classify-full        # run classifier on all completions"
