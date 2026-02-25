# Iteration Plan: DB-Based Inference & Analysis

## Background

We're extracting cultural value distributions from language models by running 187 EVS (European Values Study) survey questions through them and measuring response probabilities via next-token logprobs. The goal: do LLMs trained on language X produce value distributions that resemble country X's actual survey responses?

### Models
| Model | Params | Type | Languages | Dtype |
|-------|--------|------|-----------|-------|
| HPLT-2.15B (×22) | 2.15B | Monolingual | 1 each | bf16 |
| EuroLLM-22B | 22B | Multilingual | All 22 | bf16 |
| Gemma-3-27B-PT | 27B | Multilingual | All 22 | bf16 |
| Qwen3-235B | 235B | Multilingual | All 22 | int4 |

### Pipeline
1. **questions.json** → 187 EVS questions across 22 European languages
2. **db.populate** → generates 6 option-order permutations per (question, language) → 20,863 prompts in `survey.db`
3. **extract_logprobs.py db** → queries unevaluated prompts from DB, runs logprob extraction, writes results back. Commits every 50 prompts (crash-resumable).
4. **analysis pipeline** → loads aggregated results from DB (`--db` flag), generates cultural maps, JSD matrices, human comparison figures

---

## What's Already Done

### Infrastructure (complete)
- [x] `eurollm/db/schema.py` — SQLite schema (questions, prompts, models, evaluations, human_distributions)
- [x] `eurollm/db/populate.py` — prompt generation, model registration, human data import
- [x] `eurollm/db/load.py` — query DB → backward-compatible DataFrames for analysis
- [x] `eurollm/inference/extract_logprobs.py` — added `db` subcommand with resumable inference
- [x] `eurollm/slurm/run_db.sh` — SLURM job script for DB-based inference
- [x] `eurollm/slurm/launch_db.sh` — master launcher for all models
- [x] `eurollm/analysis/analyze.py` — added `--db` flag
- [x] `eurollm/analysis/compare_to_human.py` — added `--db` flag
- [x] `eurollm/analysis/constants.py` — updated model registry
- [x] `eurollm/human_data/process_evs.py` — added `--db` flag

### Database populated (complete)
- [x] 187 questions loaded
- [x] 20,863 prompts generated (20,767 standard + 96 rephrase)
- [x] 26 models registered
- [x] 20,181 human distribution rows imported

### HPLT inference (complete)
- [x] All 22 HPLT-2.15B monolingual models: **100% done** (~40s each, all prompts evaluated)
- Total: ~22K prompts evaluated, ~120K evaluation rows

### EuroLLM-22B: resubmitted
- [x] First attempt (job 52544123): OOM at 48GB, failed after 100/20,863 prompts
- [x] Resubmitted (job 52554809): 96GB RAM, 4h, A100/H100 — currently pending
- 100 prompts already committed (will resume from 101)

---

## What Needs To Be Done

### Step 1: Monitor & resubmit EuroLLM-22B until complete
EuroLLM-22B resubmitted with 96GB (job 52554809). Once it starts:
- Verify it doesn't OOM again by checking logs: `tail -f eurollm/logs/db_extract_52554809.out`
- At ~1.5 p/s, 20,763 remaining prompts ≈ 3.8 hours — should fit in 4h limit
- If preempted or fails, resubmit same command:
```bash
sbatch --mem=96G -t 04:00:00 -G 1 --constraint="a100|h100" \
    eurollm/slurm/run_db.sh eurollm22b utter-project/EuroLLM-22B-2512
```

### Step 2: Monitor & resubmit Gemma-3-27B-PT until complete
Job 52544124 running at ~0.7 p/s. ETA ~9h exceeds 4h limit — will be preempted.
- Check progress after preemption
- Resubmit with 8h limit:
```bash
sbatch --mem=48G -t 08:00:00 -G 1 --constraint="a100|l40s|h100" \
    eurollm/slurm/run_db.sh gemma3_27b_pt google/gemma-3-27b-pt
```
- May need 2-3 resubmissions total

### Step 3: Monitor & resubmit Qwen3-235B until complete
Job 52544634 pending (waiting for A100-80GB node). Once it starts:
- Verify int4 quantization loads: check for bitsandbytes errors in logs
- Expected throughput: ~0.3-0.5 p/s → 12-19h for 20,863 prompts
- Will need multiple 8h resubmissions:
```bash
sbatch --mem=192G -t 08:00:00 -G 1 --constraint=a100-80g \
    eurollm/slurm/run_db.sh qwen3235b Qwen/Qwen3-235B \
    eurollm/data/survey.db --dtype int4
```

### Step 4: Sanity-check HPLT results
Before running analysis, verify HPLT data looks reasonable:
```bash
PYTHONPATH=eurollm eurollm/.venv/bin/python3 -c "
from db.load import load_results
df = load_results('eurollm/data/survey.db', model_ids=['hplt2c_eng','hplt2c_deu','hplt2c_fra'])
print(df.groupby('model_type')[['prob_averaged','p_valid_forward']].describe())
print()
print('Low p_valid questions:')
low = df[df['p_valid_forward'] < 0.10].groupby(['model_type','question_id']).size()
print(low if len(low) > 0 else '  None')
"
```

### Step 5: Run analysis pipeline (once all models finish)
```bash
# Main analysis: quality, bias, JSD, PCA, t-SNE, UMAP, example distributions
PYTHONPATH=eurollm eurollm/.venv/bin/python -m analysis.analyze all \
    --db eurollm/data/survey.db \
    --questions eurollm/data/questions.json \
    --figures-dir eurollm/figures

# Human comparison: scatter, JSD by type, model comparison, heatmap, best/worst
PYTHONPATH=eurollm eurollm/.venv/bin/python -m analysis.compare_to_human \
    --db eurollm/data/survey.db \
    --questions eurollm/data/questions.json \
    --figures-dir eurollm/figures
```

### Step 6: Run rephrase analysis (once HPLT eng is done — already complete)
Can run this now since hplt2c_eng has evaluated the rephrase prompts:
```bash
PYTHONPATH=eurollm eurollm/.venv/bin/python -m analysis.analyze rephrase \
    --db eurollm/data/survey.db \
    --questions eurollm/data/questions.json \
    --figures-dir eurollm/figures
```

---

## Monitoring Commands

```bash
# Job queue
squeue -u $USER

# DB progress — prompts evaluated per model
PYTHONPATH=eurollm eurollm/.venv/bin/python -c "
from db.schema import get_connection
conn = get_connection('eurollm/data/survey.db')
r = conn.execute('''
    SELECT e.model_id, COUNT(DISTINCT e.prompt_id) as done
    FROM evaluations e GROUP BY e.model_id ORDER BY done DESC
''').fetchall()
for m, d in r: print(f'  {m}: {d}/20863')
conn.close()
"

# Tail a running job
tail -f eurollm/logs/db_extract_<JOBID>.out

# Check job exit status
sacct -j <JOBID> --format=JobID,State,ExitCode,Elapsed,MaxRSS
```
