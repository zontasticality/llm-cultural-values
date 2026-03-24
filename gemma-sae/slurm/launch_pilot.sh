#!/bin/bash
# Launch pilot sampling jobs on Unity.
# Usage: bash gemma-sae/slurm/launch_pilot.sh [--dry-run]

set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN ==="
fi

SCRIPT="gemma-sae/slurm/run_sample.sh"
DB="gemma-sae/data/culture.db"
PILOT_ARGS="--n-samples 50"

submit() {
    local name="$1" mem="$2" time="$3" constraint="$4"
    shift 4
    local cmd="sbatch --job-name=$name --mem=$mem -t $time"
    if [[ -n "$constraint" ]]; then
        cmd="$cmd --constraint=$constraint"
    fi
    cmd="$cmd $SCRIPT $@"

    if $DRY_RUN; then
        echo "  $cmd"
    else
        echo -n "  $name: "
        eval "$cmd"
    fi
}

echo "=== Gemma 3 27B PT (pilot) ==="
submit "pilot-g27b" "96G" "01:00:00" "a100" \
    gemma3_27b_pt google/gemma-3-27b-pt "$DB" \
    $PILOT_ARGS --temperature 1.0

echo "=== Gemma 3 12B PT (pilot) ==="
submit "pilot-g12b" "48G" "00:30:00" "a100|l40s" \
    gemma3_12b_pt google/gemma-3-12b-pt "$DB" \
    $PILOT_ARGS --temperature 1.0

echo "=== EuroLLM-22B (pilot, EU langs only) ==="
submit "pilot-euro" "48G" "00:30:00" "a100|h100" \
    eurollm22b utter-project/EuroLLM-22B-2512 "$DB" \
    $PILOT_ARGS --temperature 1.0

echo "=== HPLT monolingual (pilot, 4 langs) ==="
for lang in eng fin pol ron; do
    submit "pilot-hplt-$lang" "32G" "00:15:00" "" \
        "hplt2c_${lang}" "HPLT/hplt2c_${lang}_checkpoints" "$DB" \
        $PILOT_ARGS --temperature 0.8 --lang "$lang"
done

echo ""
echo "Done. Use 'squeue -u \$USER' to monitor jobs."
