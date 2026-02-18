#!/bin/bash
# Launch optimization grid search on ALL 22 languages x 2 model families.
# HPLT: 22 monolingual models (2.15B, ~5GB VRAM, ~2.5 min each)
# EuroLLM: 22 languages on EuroLLM-22B (22.6B, ~45GB VRAM, ~15 min each)
# Total: 44 jobs. Skips any that already have output files.
set -euo pipefail

cd /scratch4/workspace/zevwilson_umass_edu-culture-llm/llm-cultural-values
mkdir -p eurollm/results/optimization eurollm/logs

ALL_LANGS="bul ces dan deu ell eng est fin fra hrv hun ita lit lvs nld pol por ron slk slv spa swe"
HPLT_COUNT=0
EURO_COUNT=0
SKIP_COUNT=0

# HPLT monolingual models
for lang in $ALL_LANGS; do
    OUTPUT="eurollm/results/optimization/hplt2c_${lang}_grid.parquet"
    if [ -f "$OUTPUT" ]; then
        echo "SKIP: $OUTPUT already exists"
        SKIP_COUNT=$((SKIP_COUNT + 1))
        continue
    fi
    sbatch eurollm/slurm/run_optimize.sh \
        "HPLT/hplt2c_${lang}_checkpoints" \
        "$lang" \
        "$OUTPUT"
    HPLT_COUNT=$((HPLT_COUNT + 1))
done

echo "Submitted $HPLT_COUNT HPLT optimization jobs"

# EuroLLM-22B
for lang in $ALL_LANGS; do
    OUTPUT="eurollm/results/optimization/eurollm22b_${lang}_grid.parquet"
    if [ -f "$OUTPUT" ]; then
        echo "SKIP: $OUTPUT already exists"
        SKIP_COUNT=$((SKIP_COUNT + 1))
        continue
    fi
    sbatch --mem=96G \
           --constraint="a100|h100" \
           -t 02:00:00 \
           --job-name=evs-opt-euro \
           eurollm/slurm/run_optimize.sh \
        "utter-project/EuroLLM-22B-2512" \
        "$lang" \
        "$OUTPUT"
    EURO_COUNT=$((EURO_COUNT + 1))
done

echo "Submitted $EURO_COUNT EuroLLM optimization jobs"
echo "Skipped $SKIP_COUNT already-complete jobs"
echo "Total submitted: $((HPLT_COUNT + EURO_COUNT)) optimization jobs"
