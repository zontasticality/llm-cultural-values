#!/bin/bash
# Launch all 44 model-language pairs with optimized prompt config and K=6 permutations.
# Run AFTER analyzing Phase 2 results and updating data/prompt_config.json.
set -euo pipefail

cd /scratch4/workspace/zevwilson_umass_edu-culture-llm/llm-cultural-values
mkdir -p eurollm/results/optimized eurollm/logs

LANGS="bul ces dan deu ell eng est fin fra hrv hun ita lit lvs nld pol por ron slk slv spa swe"
CONFIG="eurollm/data/prompt_config.json"
HPLT_COUNT=0
EURO_COUNT=0

# Submit 22 HPLT monolingual models (2.15B, ~5GB VRAM)
for lang in $LANGS; do
    sbatch eurollm/slurm/run_slurm.sh \
        "HPLT/hplt2c_${lang}_checkpoints" \
        "$lang" \
        "eurollm/results/optimized/hplt2c_${lang}.parquet" \
        --permutations 6 \
        --prompt-config "$CONFIG"
    HPLT_COUNT=$((HPLT_COUNT + 1))
done

echo "Submitted $HPLT_COUNT HPLT monolingual jobs"

# Submit EuroLLM-22B for each language (22.6B, ~45GB VRAM in BF16)
for lang in $LANGS; do
    sbatch --mem=96G \
           --constraint="a100|h100" \
           -t 02:00:00 \
           --job-name=evs-euro-opt \
           eurollm/slurm/run_slurm.sh \
        "utter-project/EuroLLM-22B-2512" \
        "$lang" \
        "eurollm/results/optimized/eurollm22b_${lang}.parquet" \
        --permutations 6 \
        --prompt-config "$CONFIG"
    EURO_COUNT=$((EURO_COUNT + 1))
done

echo "Submitted $EURO_COUNT EuroLLM-22B jobs"
echo "Total: $((HPLT_COUNT + EURO_COUNT)) jobs with optimized prompts"
