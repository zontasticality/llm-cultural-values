#!/bin/bash
set -euo pipefail

cd /scratch4/workspace/zevwilson_umass_edu-culture-llm/llm-cultural-values
mkdir -p eurollm/results eurollm/logs

LANGS="bul ces dan deu ell eng est fin fra hrv hun ita lit lvs nld pol por ron slk slv spa swe"
HPLT_COUNT=0
EURO_COUNT=0

# Submit 22 HPLT monolingual models (2.15B, ~5GB VRAM)
for lang in $LANGS; do
    sbatch eurollm/slurm/run_slurm.sh \
        "HPLT/hplt2c_${lang}_checkpoints" \
        "$lang" \
        "eurollm/results/hplt2c_${lang}.parquet"
    HPLT_COUNT=$((HPLT_COUNT + 1))
done

echo "Submitted $HPLT_COUNT HPLT monolingual jobs"

# Submit EuroLLM-22B for each language (22.6B, ~45GB VRAM in BF16)
# Needs A100 80GB or H100
for lang in $LANGS; do
    sbatch --mem=96G \
           --constraint="a100|h100" \
           -t 02:00:00 \
           --job-name=evs-euro \
           eurollm/slurm/run_slurm.sh \
        "utter-project/EuroLLM-22B-2512" \
        "$lang" \
        "eurollm/results/eurollm22b_${lang}.parquet"
    EURO_COUNT=$((EURO_COUNT + 1))
done

echo "Submitted $EURO_COUNT EuroLLM-22B jobs"
echo "Total: $((HPLT_COUNT + EURO_COUNT)) jobs"
