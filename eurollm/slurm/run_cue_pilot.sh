#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -G 1
#SBATCH -p gpu-preempt
#SBATCH -t 08:00:00
#SBATCH -o eurollm/logs/cue_pilot_%j.out
#SBATCH -e eurollm/logs/cue_pilot_%j.err
#SBATCH --job-name=cue-pilot

module purge
module load cuda/12.6

cd /scratch4/workspace/zevwilson_umass_edu-culture-llm/llm-cultural-values

export HF_HOME=/scratch4/workspace/zevwilson_umass_edu-culture-llm/.hf_cache
export PYTHONPATH="${PYTHONPATH}:eurollm"
export PYTHONUNBUFFERED=1

echo "=== Cue-hint A/B pilot: 22 HPLT models × 300 prompts ==="
echo "Start: $(date)"

for lang in bul ces dan deu ell eng est fin fra hrv hun ita lit lvs nld pol por ron slk slv spa swe; do
    echo ""
    echo "--- hplt2c_${lang} ($(date)) ---"
    eurollm/.venv/bin/python eurollm/inference/extract_logprobs.py db \
        --model_id "hplt2c_${lang}" \
        --model_hf_id "HPLT/hplt2c_${lang}_checkpoints" \
        --db eurollm/data/survey.db \
        --config cue_hint \
        --limit 300
    echo "--- hplt2c_${lang} done ($(date)) ---"
done

echo ""
echo "=== Cue-hint pilot complete: $(date) ==="
