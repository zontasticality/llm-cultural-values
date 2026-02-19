#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -p cpu
#SBATCH -t 04:00:00
#SBATCH -o eurollm/logs/download_%j.out
#SBATCH -e eurollm/logs/download_%j.err
#SBATCH --job-name=dl-models

cd /scratch4/workspace/zevwilson_umass_edu-culture-llm/llm-cultural-values

export HF_HOME=/scratch4/workspace/zevwilson_umass_edu-culture-llm/.hf_cache
export PYTHONUNBUFFERED=1

eurollm/.venv/bin/python -c "
from huggingface_hub import snapshot_download
import time

models = [
    'google/gemma-3-27b-pt',
    'google/gemma-3-27b-it',
    'Qwen/Qwen3-235B-A22B',
]

for m in models:
    print(f'\\n=== Downloading {m} ===')
    t0 = time.time()
    snapshot_download(m, max_workers=4)
    elapsed = time.time() - t0
    print(f'DONE: {m} ({elapsed/60:.1f} min)')

print('\\n=== All models downloaded ===')
"
