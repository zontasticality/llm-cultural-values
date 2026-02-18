#!/bin/bash
set -euo pipefail

cd /scratch4/workspace/zevwilson_umass_edu-culture-llm/llm-cultural-values
mkdir -p eurollm/results eurollm/logs

# HPLT English (2.15B, ~5GB VRAM)
sbatch \
    -c 4 --mem=32G -p gpu-preempt -G 1 -t 01:00:00 \
    -o eurollm/logs/rephrase_hplt_eng_%j.out \
    -e eurollm/logs/rephrase_hplt_eng_%j.err \
    --job-name=rephrase-hplt \
    --wrap="
module purge && module load cuda/12.6
export HF_HOME=/scratch4/workspace/zevwilson_umass_edu-culture-llm/.hf_cache
export PYTHONPATH=\"\${PYTHONPATH:-}:eurollm\"
eurollm/.venv/bin/python eurollm/inference/rephrase_test.py \
    --model_id HPLT/hplt2c_eng_checkpoints \
    --output eurollm/results/rephrase_test_hplt2c_eng.parquet
"

echo "Submitted HPLT English rephrase job"

# EuroLLM-22B English (22.6B, ~45GB VRAM in BF16)
sbatch \
    -c 4 --mem=96G -p gpu-preempt -G 1 -t 02:00:00 \
    --constraint="a100|h100" \
    -o eurollm/logs/rephrase_euro_eng_%j.out \
    -e eurollm/logs/rephrase_euro_eng_%j.err \
    --job-name=rephrase-euro \
    --wrap="
module purge && module load cuda/12.6
export HF_HOME=/scratch4/workspace/zevwilson_umass_edu-culture-llm/.hf_cache
export PYTHONPATH=\"\${PYTHONPATH:-}:eurollm\"
eurollm/.venv/bin/python eurollm/inference/rephrase_test.py \
    --model_id utter-project/EuroLLM-22B-2512 \
    --output eurollm/results/rephrase_test_eurollm22b_eng.parquet
"

echo "Submitted EuroLLM-22B English rephrase job"
echo "Total: 2 jobs"
