#!/bin/bash
# Launch Gemma-3-27B on all 22 languages × 2 variants (pt + it).
# Each job needs 1x A100 80GB or H100 (~54GB in bf16).
set -euo pipefail

cd /scratch4/workspace/zevwilson_umass_edu-culture-llm/llm-cultural-values
mkdir -p eurollm/results eurollm/logs

LANGS="bul ces dan deu ell eng est fin fra hrv hun ita lit lvs nld pol por ron slk slv spa swe"
COUNT=0

for variant in pt it; do
    for lang in $LANGS; do
        sbatch eurollm/slurm/run_gemma3_27b.sh "$lang" "$variant"
        COUNT=$((COUNT + 1))
    done
done

echo "Submitted $COUNT Gemma-3-27B jobs (22 languages × 2 variants)"
