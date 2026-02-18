#!/bin/bash
# Launch Qwen2.5-72B on all 22 languages with optimized prompt config.
# Each job needs 2x A100-80GB (~145GB in bf16).
set -euo pipefail

cd /scratch4/workspace/zevwilson_umass_edu-culture-llm/llm-cultural-values
mkdir -p eurollm/results/optimized eurollm/logs

LANGS="bul ces dan deu ell eng est fin fra hrv hun ita lit lvs nld pol por ron slk slv spa swe"
CONFIG="eurollm/data/prompt_config.json"
COUNT=0

for lang in $LANGS; do
    OUTPUT="eurollm/results/optimized/qwen2572b_${lang}.parquet"
    if [ -f "$OUTPUT" ]; then
        echo "Skipping $lang — $OUTPUT already exists"
        continue
    fi
    sbatch eurollm/slurm/run_slurm_qwen.sh \
        "Qwen/Qwen2.5-72B" \
        "$lang" \
        "$OUTPUT" \
        --permutations 6 \
        --prompt-config "$CONFIG"
    COUNT=$((COUNT + 1))
done

echo "Submitted $COUNT Qwen2.5-72B jobs"
