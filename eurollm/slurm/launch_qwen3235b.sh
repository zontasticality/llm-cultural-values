#!/bin/bash
# Launch Qwen3-235B-A22B on all 22 languages.
# Each job needs 4x A100 80GB with int8 quantization.
set -euo pipefail

cd /scratch4/workspace/zevwilson_umass_edu-culture-llm/llm-cultural-values
mkdir -p eurollm/results eurollm/logs

LANGS="bul ces dan deu ell eng est fin fra hrv hun ita lit lvs nld pol por ron slk slv spa swe"
COUNT=0

for lang in $LANGS; do
    sbatch eurollm/slurm/run_qwen3235b.sh "$lang"
    COUNT=$((COUNT + 1))
done

echo "Submitted $COUNT Qwen3-235B-A22B jobs"
