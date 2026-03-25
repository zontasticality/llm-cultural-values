# gemma-sae — Implementation Plan

## Overview

DB-backed pipeline for cultural completion comparison via temperature sampling + LLM classification. Mirrors eurollm/ architecture (SQLite/WAL, resumable SLURM jobs, batch commits).

Two decoupled stages: **sampling** (GPU-bound, Unity SLURM cluster) → **classification** (API-based: GPT-4.1 mini batch or Claude Haiku 4.5 batch).

**Development workflow**: Edit locally → rsync to Unity → sbatch → poll → rsync results back. Classification runs via API from local machine.

See `RESEARCH.md` for scientific motivation, literature review, and the Phase 3 SAE hypothesis.

---

## Pilot-First Strategy

Iterate on a small slice before scaling to the full matrix. The pilot tests the full pipeline end-to-end: sampling → extraction → filtering → classification → analysis.

### Pilot Slice

| Model | Langs | Templates | Samples/each | Completions |
|-------|-------|-----------|-------------|-------------|
| Gemma 3 27B PT | eng, fin, pol, ron, zho | self_concept, values | 50 | 500 |
| Gemma 3 12B PT | eng, fin, pol, ron, zho | self_concept, values | 50 | 500 |
| EuroLLM-22B | eng, fin, pol, ron | self_concept, values | 50 | 400 |
| HPLT 2.15B × 4 | eng, fin, pol, ron (1 each) | self_concept, values | 50 | 400 |
| **Total** | | | | **1,800** |

After filtering (~85% survive) → ~1,530 completions for classification.

### Classifier Comparison

Run **both** classifiers on all pilot completions:

| Classifier | Est. cost | Notes |
|------------|-----------|-------|
| GPT-4.1 mini batch | ~$0.13 | Native JSON schema constraints, 24h async SLA |
| Claude Haiku 4.5 batch | ~$0.35 | High quality, 50% batch discount |

Hand-label ~50 completions (10/lang), compute Cohen's κ against each classifier, pick the winner.

### Pilot Go/No-Go

1. **Classification quality**: κ > 0.7 on content categories vs hand-labels for at least one classifier
2. **Cultural signal**: χ² p < 0.01 for content category distributions between any two IW clusters, OR Cohen's d > 0.3 for fin vs ron on trad_secular
3. **Pipeline health**: filter rate < 30% for 27B, all models produce completions in all target langs

If gates fail: iterate prompts/classifier/templates before scaling. **Phase 3 code should NOT be written until pilot passes.**

### Compute & Cost

| Item | Where | Cost |
|------|-------|------|
| Pilot sampling (~30 min A100) | Unity SLURM | $0 |
| Full sampling (~13h A100) | Unity SLURM | $0 |
| Pilot classification (both classifiers) | API | ~$0.48 |
| Full classification (winner) | API | $96 (mini) or $270 (haiku) |
| **Pilot total** | | **~$0.48** |
| **Full run total** | | **$96–$270** |

---

## Directory Structure

```
gemma-sae/
  RESEARCH.md              # Research plan & methodology
  PLAN.md                  # This file
  __init__.py
  db/
    __init__.py
    schema.py              # DDL + get_connection/init_db
    populate.py            # CLI: populate templates, prompts, models
    load.py                # Query helpers: unsampled, unclassified, results
  inference/
    __init__.py
    sample.py              # Temperature sampling (GPU jobs)
  classify/
    __init__.py
    classify.py            # LLM classification pipeline
    prompts.py             # Classifier prompt template + JSON parser
  analysis/
    __init__.py
    constants.py           # Languages, clusters, model metadata
    analyze.py             # Subcommand-based analysis
  steering/                # Phase 3: SAE feature identification + steering
    __init__.py
    extract_activations.py # 3A: Run probes through model, collect SAE activations
    score_features.py      # 3B: Compute monolinguality/monoculturality metrics
    select_features.py     # 3B: Feature selection with null distribution + ablation
    build_vectors.py       # 3C: Construct CAA, SAE, and FGAA steering vectors
    steer.py               # 3D: Forward hook for steering during generation
    analyze_steering.py    # 3E: Compare steered vs baseline cultural profiles
  data/
    prompt_templates.json  # 8 template definitions
    translations/          # Per-language prompt files ({lang}.json)
    culture.db             # Main database (gitignored, regenerated via populate)
    probes/                # Probe text corpora for Phase 3
      flores_200/          # Parallel text (downloaded, same content per lang)
      cultural_wikipedia/  # English Wikipedia passages per IW cluster
      wvs_stems/           # Reuse Phase 2 completions
    sae_cache/             # Downloaded Gemma Scope 2 weights (gitignored)
    activations/           # Sparse activation matrices (.npz)
      lang/                # e.g., layer40_256k_L050_flores_fin.npz
      culture/             # e.g., layer40_256k_L050_wikiculture_protestant.npz
    feature_scores/        # Exported per-feature metric CSVs
    vectors/               # Steering vectors (safetensors)
      sae/
      caa/
      fgaa/
  slurm/
    run_sample.sh          # Parameterized sampling job
    run_classify.sh        # Classification job
    launch_sample.sh       # Submit all sampling jobs
    launch_classify.sh     # Submit classification job
    run_extract_activations.sh  # Phase 3A: activation extraction job
    run_score_features.sh       # Phase 3B: feature scoring (CPU-only)
    run_build_vectors.sh        # Phase 3C: vector construction
    run_steer_sample.sh         # Phase 3D: steered generation job
    launch_phase3.sh            # Submit all Phase 3 jobs
  figures/
    main/
    supplementary/
    diagnostic/
```

---

## Database Schema

5 tables. Core pattern from eurollm: "uncompleted work" queries for resumability, WAL mode for concurrent SLURM job access, batch commits.

```sql
CREATE TABLE IF NOT EXISTS templates (
    template_id     TEXT PRIMARY KEY,       -- "self_concept", "values", etc.
    template_name   TEXT NOT NULL,          -- Human-readable name
    cultural_target TEXT NOT NULL            -- Which cultural dimension it targets
);

CREATE TABLE IF NOT EXISTS prompts (
    prompt_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    template_id     TEXT NOT NULL REFERENCES templates(template_id),
    lang            TEXT NOT NULL,           -- ISO 639-3
    variant_idx     INTEGER NOT NULL DEFAULT 0,  -- 0=primary, 1+=grammatical controls
    prompt_text     TEXT NOT NULL,           -- Actual text fed to model
    is_control      INTEGER NOT NULL DEFAULT 0,
    UNIQUE(template_id, lang, variant_idx)
);

CREATE TABLE IF NOT EXISTS models (
    model_id        TEXT PRIMARY KEY,       -- "hplt2c_eng", "gemma3_27b_pt"
    model_hf_id     TEXT NOT NULL,          -- HuggingFace ID
    model_family    TEXT NOT NULL,          -- For grouping in analysis
    is_multilingual INTEGER NOT NULL,
    dtype           TEXT DEFAULT 'bf16',
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS completions (
    completion_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt_id       INTEGER NOT NULL REFERENCES prompts(prompt_id),
    model_id        TEXT NOT NULL REFERENCES models(model_id),
    sample_idx      INTEGER NOT NULL,       -- 0..199
    completion_raw  TEXT NOT NULL,           -- Full generation (up to max_new_tokens)
    completion_text TEXT,                    -- Extracted first sentence/clause (NULL if filtered out)
    n_tokens_raw    INTEGER NOT NULL,        -- Token count of raw generation
    n_tokens        INTEGER,                 -- Token count of extracted text (NULL if filtered)
    filter_status   TEXT NOT NULL DEFAULT 'ok',  -- "ok", "degenerate", "repetition", "too_short", "non_text"
    temperature     REAL NOT NULL DEFAULT 1.0,
    top_p           REAL NOT NULL DEFAULT 0.95,
    seed            INTEGER,
    steering_config TEXT NOT NULL DEFAULT 'none',  -- "none" for Phase 1-2; JSON for Phase 3
    UNIQUE(prompt_id, model_id, sample_idx, steering_config)
);

CREATE TABLE IF NOT EXISTS classifications (
    completion_id       INTEGER NOT NULL REFERENCES completions(completion_id),
    classifier_model    TEXT NOT NULL,       -- "gemma3_27b_it", "claude-sonnet", etc.
    content_category    TEXT NOT NULL,
    dim_indiv_collect   INTEGER NOT NULL,   -- 1 (individualist) to 5 (collectivist)
    dim_trad_secular    INTEGER NOT NULL,   -- 1 (traditional) to 5 (secular-rational)
    dim_surv_selfexpr   INTEGER NOT NULL,   -- 1 (survival) to 5 (self-expression)
    raw_response        TEXT,               -- Full classifier output for debugging
    UNIQUE(completion_id, classifier_model)
);

CREATE INDEX IF NOT EXISTS idx_comp_model_prompt ON completions(model_id, prompt_id);
CREATE INDEX IF NOT EXISTS idx_prompt_template_lang ON prompts(template_id, lang);

-- Phase 3 tables: SAE feature identification + steering vectors

CREATE TABLE IF NOT EXISTS probe_sets (
    probe_set_id    TEXT PRIMARY KEY,       -- "flores_200", "wiki_culture", "wvs_stems"
    description     TEXT NOT NULL,
    source_url      TEXT,
    n_texts         INTEGER NOT NULL,
    text_type       TEXT NOT NULL            -- "parallel" (same content, diff lang) or "contrastive" (diff content, same lang)
);

CREATE TABLE IF NOT EXISTS feature_scores (
    feature_score_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id         TEXT NOT NULL,
    sae_id           TEXT NOT NULL,          -- "resid_post_layer40_256k_L050"
    feature_idx      INTEGER NOT NULL,
    probe_set_id     TEXT NOT NULL REFERENCES probe_sets(probe_set_id),
    group_key        TEXT NOT NULL,          -- lang code or IW cluster name
    metric           TEXT NOT NULL,          -- "monolinguality", "monoculturality", "output_score", "feature_class"
    score            REAL NOT NULL,
    mean_activation  REAL,
    mean_other       REAL,
    rank_in_group    INTEGER,
    UNIQUE(model_id, sae_id, feature_idx, probe_set_id, group_key, metric)
);

CREATE TABLE IF NOT EXISTS steering_vectors (
    vector_id       TEXT PRIMARY KEY,       -- "caa_orthodox_layer31-53_a2.0"
    model_id        TEXT NOT NULL,
    method          TEXT NOT NULL,           -- "sae", "caa", "fgaa"
    target_cluster  TEXT NOT NULL,           -- IW cluster being steered toward
    layer_or_range  TEXT NOT NULL,           -- "40" or "31-53"
    alpha           REAL NOT NULL,           -- steering strength
    feature_ids     TEXT,                   -- JSON array of SAE feature indices (NULL for CAA)
    n_features      INTEGER,
    vector_path     TEXT NOT NULL,           -- path to safetensors file
    notes           TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_fs_sae_group ON feature_scores(sae_id, group_key, metric);
CREATE INDEX IF NOT EXISTS idx_fs_model_metric ON feature_scores(model_id, metric, score);
CREATE INDEX IF NOT EXISTS idx_sv_method_cluster ON steering_vectors(method, target_cluster);
```

**Design notes:**
- `completions.steering_config`: `"none"` for baseline (Phases 1-2). In Phase 3, a `vector_id` string referencing the `steering_vectors` table (e.g., `"caa_orthodox_layer31-53_a2.0"`) or inline JSON like `{"method": "sae", "layer": 40, "feature_ids": [1234], "alpha": 1.5}`. Uses a sentinel string instead of NULL so SQLite's UNIQUE constraint works correctly (NULL ≠ NULL in SQLite, which would allow duplicate baseline rows).
- `classifications.classifier_model` in UNIQUE: allows re-classification with a different classifier without re-sampling.

---

## Key DB Query Patterns (db/load.py)

Mirrors `eurollm/db/load.py:load_unevaluated_prompts()`:

- **`load_unsampled_prompts(db, model_id, n_samples=200, steering_config='none')`** — prompts with fewer than N completions for this model and steering config (WHERE steering_config = ?). Core resumability query for sampling.
- **`load_unclassified_completions(db, classifier_model)`** — completions with `filter_status = 'ok'` and `completion_text IS NOT NULL` that lack a classification row for this classifier. Core resumability query for classification.
- **`load_results(db, model_ids, langs, template_ids)`** — JOIN completions → classifications → prompts → templates → models. Returns the full classified dataset for analysis.
- **`get_sampling_progress(db)`** — Per-model completion summary (like `eurollm/db/load.py:get_evaluation_progress()`).

---

## Sampling Pipeline (inference/sample.py)

```
PYTHONPATH=gemma-sae python -m inference.sample \
    --model_id gemma3_27b_pt \
    --model_hf_id google/gemma-3-27b-pt \
    --db data/culture.db \
    [--dtype bf16] [--n-samples 200] [--batch-size 20] \
    [--max-new-tokens 128] [--temperature 1.0] [--top-p 0.95] \
    [--lang fin] [--template self_concept] \
    [--validate]
```

Core loop (mirrors `eurollm/inference/extract_logprobs.py` DB mode):
1. Query `load_unsampled_prompts()` → list of prompts needing more samples
2. Load model via `load_model()` (reused from eurollm — bf16/int4/int8/fp8)
3. For each prompt: generate remaining samples via `model.generate(do_sample=True, temperature, top_p, max_new_tokens)` with **`eos_token_id=None`** (base models have unreliable EOS — some emit it after 1 token, some never)
4. **Post-process** each raw generation (see "Completion extraction and filtering" below)
5. Batch INSERT into completions table every `batch_size` prompts (store both `completion_raw` and `completion_text`)
6. Print progress with rate/ETA (same format as eurollm); also report filter-status distribution

### Completion extraction and filtering

Base models are document completers, not instruction followers. A stem like "Minä olen" may trigger blog paragraphs, dialogue, lists, boilerplate, or degenerate output. The pipeline generates a generous 128 tokens, then extracts a "unit of meaning" and filters bad completions.

**Extraction** (applied to raw generation → `completion_text`):
1. Look for a sentence boundary: `(?<=[.!?。！？])\s+` followed by uppercase, newline, or CJK character. Take everything before the boundary.
2. If no sentence boundary within the first 80 tokens, look for a clause boundary: `, ; : — –` or equivalent. Take the first clause.
3. If no clause boundary either, take up to 80 tokens as-is.
4. Strip leading/trailing whitespace.

**Filtering** (sets `filter_status` and `completion_text = NULL` for bad completions):
- `"degenerate"`: empty or whitespace-only after extraction
- `"repetition"`: any 5-gram appears 3+ times in the raw output
- `"too_short"`: fewer than 3 tokens after extraction
- `"non_text"`: >50% of characters are code/markup indicators (`{ } < > = ; //` etc.) or URLs

Filtered completions are stored (`completion_raw` preserved) but excluded from classification. The **filter rate** per (model, lang, template) is a diagnostic — high filter rates indicate the model struggles with that language or stem.

**Why 128 tokens**: Base models need room to reach a natural stopping point. 50 tokens often cuts off mid-thought, especially in agglutinative languages (Finnish, Hungarian, Turkish) where more morphological information is packed per token. 128 is generous enough that the first sentence almost always completes, while still being fast to generate (~0.25s on A100).

**Why not use EOS for stopping**: Base model EOS behavior is unreliable. Some models (especially smaller ones) emit EOS immediately after a period; others never emit it. Disabling EOS and using post-hoc extraction gives consistent behavior across model families.

### Temperature strategy

**Default: T=1.0 for large models (27B, 22B, 12B), T=0.8 for small models (4B, HPLT 2.15B). Top-p=0.95 throughout.**

**Rationale**: T=1.0 samples from the model's learned distribution — the theoretically correct choice for measuring "what cultural patterns did pretraining encode." Lower temperatures sharpen the distribution, which systematically suppresses minority cultural patterns: a Finnish-specific completion at 3% probability under T=1.0 drops to ~0.5% at T=0.5. Since the training data is English-dominated, low-T biases all languages toward Western/English cultural modes. This is the opposite of what we want.

The concern that T=1.0 is "too noisy" is addressed by N=200: the standard error of the mean on a 1-5 Likert scale is at most 1.15/√200 ≈ 0.08, giving 95% CIs of ±0.16 points — tight enough to detect cross-cultural differences (typical IW effect sizes are 0.5-1.5 scale points).

Small models (4B, HPLT 2.15B) use T=0.8 because they degrade faster at high temperature (Li et al. 2025 — larger models have higher "mutation temperature" thresholds). T=0.8 is a 1.25× logit amplification, enough to reduce incoherent completions without heavy cultural bias.

Top-p=0.95 acts as a safety net across all temperatures, trimming the extreme tail without materially affecting the distribution.

**Temperature sensitivity check (pilot)**: Run 5 pilot languages × {T=0.7, T=1.0, T=1.3} × 1 template × 50 samples to verify that cultural profiles (mean dimension scores per language) are stable across temperatures. If they are, this is strong evidence of robustness against the Rupprecht et al. (FAccT 2025) critique that "cultural alignment is not a stable property of LLMs." If profiles shift systematically (e.g., cross-language variance decreases at lower T), report this as evidence of temperature-dependent cultural suppression. Either finding is publishable.

**Key references**:
- Renze & Guven (EMNLP 2024): T ∈ [0, 1] doesn't significantly change *what* models know on MCQA — it mainly modulates *diversity*
- Sivaprasad et al. (ACL 2025): LLM sampling has a "prescriptive" component — even at T=1.0, samples deviate from the empirical training distribution toward implicit norms
- Rupprecht et al. (FAccT 2025): Cultural alignment is sensitive to minor methodological variations; temperature stability analysis directly addresses this
- Dominguez-Olmedo et al. (NeurIPS 2024): After correcting for ordering/labeling biases, models trend toward uniform responses on constrained MCQ formats — motivates our open-ended generation approach over forced-choice

**Seeding:** `seed = int(hashlib.sha256(f"{model_id}:{prompt_id}:{sample_idx}".encode()).hexdigest(), 16) & 0xFFFFFFFF` for reproducibility. Must use `hashlib`, NOT Python's built-in `hash()` (which is randomized across processes since Python 3.3). Stored in DB.

**`--validate` mode:** Process 2 prompts, print first 5 completions each. Quick sanity check before full SLURM submission.

---

## Classification Pipeline (classify/classify.py)

```
PYTHONPATH=gemma-sae python -m classify.classify \
    --db data/culture.db \
    --classifier gemma3_27b_it \
    --classifier-hf-id google/gemma-3-27b-it \
    [--batch-size 50] [--lang fin] [--model-id gemma3_27b_pt] \
    [--api-mode] [--validate]
```

### Classifier prompt (classify/prompts.py)

Asks for structured JSON output with content category + 3 cultural dimensions.

**Critical**: The classifier must NOT see the source language or language name — this would bias cultural ratings (e.g., "Arabic" → traditional regardless of content). Non-English completions are machine-translated to English before classification so the classifier operates on content only.

```
You are classifying a text completion for cross-cultural research.

A person was asked to complete the sentence: "{stem_english}"
Their response: "{completion_english}"

Classify this completion on the following dimensions.
Respond ONLY with a JSON object in this exact format.

1. Content category (choose one):
   - family_social, occupation_achievement, personality_trait,
     material_physical, abstract_philosophical, religious_spiritual,
     national_civic, other

2. Individualist (1) vs Collectivist (5)
3. Traditional (1) vs Secular-Rational (5)
4. Survival (1) vs Self-Expression (5)

{"category": "...", "indiv_collect": N, "trad_secular": N, "surv_selfexpr": N}
```

**Translation step**: For non-English completions, prepend a batch translation pass (can use the same classifier model or an API). Store the English translation in a new `completions.completion_english` column (NULL for native English). The classifier always operates on English text. This adds a pipeline stage but eliminates the language→culture confound.

**Alternative (simpler, less robust)**: Skip translation, pass the raw completion text without any language label. The classifier will often infer the language from script/vocabulary, but at least it isn't *told* the language. Use this for the pilot; upgrade to translation-based if classifier κ on non-English completions is substantially lower than on English.

### Core loop
1. Query `load_unclassified_completions()` — only returns completions with `filter_status = 'ok'` and `completion_text IS NOT NULL`
2. Load classifier model, or use `--api-mode` for Claude/Gemini API
3. For each completion: build prompt from `completion_text` (extracted first sentence, not raw), generate, parse JSON
4. Batch INSERT into classifications every `batch_size` completions
5. Failed JSON parses → logged and skipped (retried on next run)

**`--validate` mode:** Classify 20 completions, print raw output + parsed result, show category/dimension distributions.

---

## Data Files

### `data/prompt_templates.json`

8 culturally diagnostic templates:

| template_id | Canonical English text | Cultural target |
|---|---|---|
| self_concept | "I am" / "I am a" | identity |
| values | "The most important thing in life is" | core values |
| moral | "A good person is someone who" | moral norms |
| childrearing | "Children should learn to" | socialization |
| family | "In my family, the most important thing is" | family values |
| success | "Success means" | achievement |
| decision | "When making a difficult decision, I" | agency |
| belief | "I believe that" | worldview |

### `data/translations/{lang}.json`

Per-language translations. Format:

```json
{
  "lang": "fin",
  "prompts": [
    {"template_id": "self_concept", "variant_idx": 0, "prompt_text": "Minä olen", "is_control": false},
    {"template_id": "values", "variant_idx": 0, "prompt_text": "Elämässä tärkeintä on", "is_control": false},
    {"template_id": "values", "variant_idx": 1, "prompt_text": "Eniten elämässä merkitsee", "is_control": true}
  ]
}
```

- Grammatical control variants (variant_idx > 0, is_control=true) only for 5 control languages: eng, deu, fra, zho, tur
- All translations should be natural sentence-start fragments, not word-for-word translations
- Native speaker review for 10 core languages before full run

### Translation Validation & Completion Dynamics Analysis

Each set of pilot translations (fin, pol, ron, zho) is accompanied by a **completion dynamics document** (`data/translations/completion_dynamics.md`) that analyzes, for each (language, template) pair:

1. **Grammatical constraints**: How the language's morphology/syntax restricts what can follow the stem. E.g., Finnish partitive case after "tärkeintä on" constrains to noun phrases; Chinese topic-comment structure after 我是 invites broader continuations.
2. **Web corpus expectations**: What a base LLM trained on this language's web text would most commonly produce. E.g., Polish "Jestem" on forums likely triggers self-introductions; Chinese 我相信 on Zhihu triggers analytical statements.
3. **Cultural confounds**: Where grammatical structure might masquerade as cultural signal. E.g., if Romanian "Copiii ar trebui să învețe" preferentially elicits education-related completions due to the subjunctive construction, this is a grammatical artifact, not cultural.
4. **Predicted category distributions**: Expected content category proportions per (language, template) before running any model, serving as a prior against which to compare actual results.
5. **Cross-language stem equivalence assessment**: Whether each pair of translations across languages is truly parallel in what it invites, or whether structural differences create systematic biases.

This document serves as a pre-registered prediction and a diagnostic reference for interpreting Phase 2 results.

---

## Models

| model_id | HuggingFace ID | Size | Type | Languages |
|---|---|---|---|---|
| hplt2c_{lang} × 22 | HPLT/hplt2c_{lang}_checkpoints | 2.15B | Monolingual | 1 each |
| eurollm22b | utter-project/EuroLLM-22B-2512 | 22B | Multilingual EU | 22 EU |
| gemma3_12b_pt | google/gemma-3-12b-pt | 12B | Multilingual | All 27 |
| gemma3_27b_pt | google/gemma-3-27b-pt | 27B | Multilingual | All 27 |

Phase 3 adds: Gemma 3 27B PT + SAE steering (same model, `steering_config` column distinguishes).

### Classifiers (API-based)

| Classifier | API | Pricing (batch) | Notes |
|------------|-----|-----------------|-------|
| gpt-4.1-mini | OpenAI Batch API | $0.20/$0.80 per M tok | Native JSON schema constraints |
| claude-haiku-4.5 | Anthropic Batch API | $0.50/$2.50 per M tok | 50% batch discount |

Pilot runs both; full run uses the winner.

---

## Languages & Cultural Clusters

### European (22) — all models
bul, ces, dan, deu, ell, eng, est, fin, fra, hrv, hun, ita, lit, lvs, nld, pol, por, ron, slk, slv, spa, swe

### Expanded (5) — Gemma 3 only
zho, jpn, ara, hin, tur

### Inglehart-Welzel clusters

| Cluster | Languages |
|---|---|
| Protestant Europe | dan, fin, swe, nld, deu |
| Catholic Europe | fra, ita, spa, por, ces, hun, pol, slk, slv, hrv |
| English-speaking | eng |
| Orthodox | bul, ron, ell |
| Baltic | est, lit, lvs |
| East Asian | zho, jpn |
| South Asian | hin |
| Middle Eastern | ara, tur |

---

## SLURM Jobs (Unity Cluster)

Development is local; sampling runs on Unity via SSH. Classification is API-based (runs locally).

### Remote workflow

```bash
# One-time: set up SSH multiplexing in ~/.ssh/config
# Host unity
#     ControlMaster auto
#     ControlPath ~/.ssh/sockets/%r@%h-%p
#     ControlPersist 600

# Push code → submit → poll → pull results
make push              # rsync gemma-sae/ to unity
make submit-pilot      # sbatch pilot jobs
make status            # squeue check
make pull              # rsync culture.db back
```

### `slurm/run_sample.sh`

Parameterized by `MODEL_ID`, `MODEL_HF_ID`, `DB_PATH`, `EXTRA_ARGS`. Same env setup as eurollm (CUDA 12.6, HF_HOME, PYTHONPATH).

### `slurm/launch_pilot.sh [--dry-run]`

Submits pilot sampling jobs (2 templates × 5 langs × 50 samples):

| Model | Jobs | Mem | Time | GPU constraint |
|---|---|---|---|---|
| Gemma 3 27B PT | 1 | 96G | 1h | a100 |
| Gemma 3 12B PT | 1 | 48G | 30m | a100\|l40s |
| EuroLLM-22B | 1 | 48G | 30m | a100\|h100 |
| HPLT 2.15B × 4 | 4 (eng, fin, pol, ron) | 32G | 15m | any |

### `slurm/launch_full.sh [--dry-run]`

Full sampling matrix (~25 jobs):

| Model | Jobs | Mem | Time | GPU constraint |
|---|---|---|---|---|
| HPLT 2.15B × 22 | 22 (one per lang) | 32G | 1h | any |
| EuroLLM-22B | 1 | 48G | 4h | a100\|h100 |
| Gemma 3 12B PT | 1 | 48G | 2h | a100\|l40s |
| Gemma 3 27B PT | 1 | 96G | 4h | a100 |

### Classification (API-based, runs locally)

```bash
python -m classify.classify \
    --db data/culture.db \
    --classifier gpt-4.1-mini \
    --batch-mode                   # Submit to OpenAI/Anthropic Batch API
```

No SLURM needed — classification happens via API from local machine.

---

## Analysis Pipeline (analysis/analyze.py)

```
PYTHONPATH=gemma-sae python -m analysis.analyze {subcommand} \
    --db data/culture.db [--figure-dir figures/]
```

### Subcommands

| Subcommand | What it does |
|---|---|
| quality | Sampling coverage heatmap: (model, lang) → completions/classifications count + **filter rate** per (model, lang, template) |
| profiles | Aggregate cultural profiles: mean + bootstrap 95% CI per dimension per (model, lang) |
| umap | UMAP of cultural profiles, colored by IW cluster |
| known_groups | Cohen's d for expected contrasts: Nordic vs Orthodox, East Asian vs Western |
| stability | Cross-prompt Spearman correlation per (model, lang) — low = grammatical artifact |
| temp_sensitivity | Cultural profiles at T=0.7 vs T=1.0 vs T=1.3 for pilot subset — Spearman rank correlation of language orderings per dimension |
| categories | Content category distributions: stacked bars per lang, chi-square tests |
| size_effect | Cultural differentiation vs model size (Gemma 3 12B/27B + HPLT 2.15B + EuroLLM-22B) |
| all | Run all of the above |

**Primary outcome: content categories.** Content category distributions (family_social, occupation_achievement, etc.) are categorical, objective, and less susceptible to classifier Likert bias than the 1-5 cultural dimensions. If Chinese completions of "I am" skew 40% family_social vs English at 15%, that's a clear finding regardless of Likert dimension noise. The `categories` subcommand is the most likely to produce positive results and should be treated as the anchor behavioral finding. The three Likert cultural dimensions are secondary/exploratory.

**Go/no-go metrics (two gates, either sufficient):**
1. `known_groups` shows Cohen's d > 0.3 for Finnish vs Romanian on trad_secular, OR
2. `categories` shows χ² p < 0.01 for content category distributions between any two IW clusters (e.g., Protestant vs Orthodox, European vs East Asian)

If neither gate passes on the pilot, revise prompts/classifier before scaling. **Phase 3 code should NOT be written until at least one gate passes** — without a validated behavioral signal, SAE steering has no ground truth to validate against.

---

## Data Volume Estimates

| Stage | Records | DB size |
|---|---|---|
| Templates | 8 | — |
| Prompts (pilot: 5 langs) | ~52 | — |
| Prompts (full: 27 langs) | ~280 | — |
| **Pilot completions** (4 model groups × ~10 prompts × 50) | **1,800** | ~2 MB |
| **Pilot classifications** (×2 classifiers) | **~3,060** | ~0.5 MB |
| Completions (full: 25 models × 280 prompts × 200) | ~1,400,000 | ~1.4 GB |
| Classifications (~85% survive, 1 classifier) | ~1,190,000 | ~200 MB |
| **Total DB (full)** | | **~1.7 GB** |

---

## Reused Code from eurollm

| What | Source | How |
|---|---|---|
| `load_model()` (bf16/int4/int8/fp8) | `eurollm/inference/extract_logprobs.py:480-539` | Copy verbatim |
| `get_connection()` (WAL/FK/timeout) | `eurollm/db/schema.py:68-74` | Same pattern |
| SLURM env setup | `eurollm/slurm/run_db.sh` | Same CUDA, HF_HOME, PYTHONPATH |
| Cultural clusters, lang names | `eurollm/analysis/constants.py` | Extend with expanded langs |

---

## Implementation Order

| Step | What | Depends on |
|------|------|------------|
| 0 | Scaffolding: dirs, `__init__.py`, venv, Makefile | — |
| 1 | `db/schema.py` + `db/populate.py` + `analysis/constants.py` | 0 |
| 2 | ~~Translations for pilot langs (fin, pol, ron, zho)~~ **DONE** — see `data/translations/` + `completion_dynamics.md` | 1 |
| 3 | `inference/sample.py` + `db/load.py` | 1 |
| 4 | `classify/prompts.py` + `classify/classify.py` (API-based: GPT-4.1 mini + Haiku 4.5 batch) | 1 |
| 5 | SLURM scripts + Makefile targets (push/submit-pilot/status/pull) | 3 |
| 6 | **Pilot sampling** on Unity: 4 model groups × 2 templates × 5 langs × 50 samples | 2, 5 |
| 7 | **Pilot classification**: run both classifiers on all ~1,530 completions | 4, 6 |
| 8 | Hand-label ~50, compute κ, pick winning classifier, check go/no-go | 7 |
| 9 | `analysis/constants.py` + `analyze.py` (quality, categories, known_groups) on pilot data | 8 |
| 10 | Full translations (27 langs), full sampling + classification matrix | 8, 9 |
| 11 | Remaining analysis subcommands + human IW comparison | 10 |

---

## Verification Checklist

- **Step 1**: `python -m db.populate --db data/culture.db` → DB with 8 templates, prompts, models
- **Step 3**: `python -m inference.sample --validate` → coherent completions with both `completion_raw` and `completion_text` populated; filter rate < 30% for 27B model; kill + restart resumes
- **Step 4**: `python -m classify.classify --validate` → valid JSON classifications; only classifies `filter_status = 'ok'` rows
- **Step 6**: Pilot: `SELECT COUNT(*) FROM completions` ≈ 1,800; filter rate < 30% for 27B
- **Step 7**: Both classifiers return valid JSON for all pilot completions; no systematic failures per language
- **Step 8**: κ > 0.7 for at least one classifier; go/no-go gates checked
- **Step 9**: `python -m analysis.analyze categories` → χ² significant or Cohen's d > 0.3

---

## Phase 3: SAE-Based Cultural Feature Discovery and Steering

### Overview

Use pre-trained Gemma Scope 2 SAEs to identify cultural features in Gemma 3 27B's representations, then steer generation to produce cross-cultural behavioral shifts. Three steering methods compared: CAA (baseline), SAE, and FGAA (hybrid).

The schema already supports Phase 3 via `completions.steering_config` (`"none"` for baseline, vector_id or JSON for steered). Classification + analysis pipelines work unchanged — steered completions are just more rows.

### Key Design Decisions From Literature Review

1. **Detection features ≠ steering features**: Features at 50-65% depth detect cultural content; features at 66-100% depth steer generation. These are uncorrelated. Use crosscoders to bridge the gap.
2. **CAA is the baseline to beat**: SAE steering underperforms CAA on known concepts (DeepMind 2025). FGAA hybrid is the best-case SAE-informed method. **CAA baselines should run in parallel with SAE feature scoring** — they use the same Phase 2 activation data and can complete while 3B is still running. This saves ~1 week on the critical path and ensures we have working steering results even if SAE analysis stalls.
3. **Expect a spectrum, not a clean partition**: Feature hedging merges correlated culture/language features at narrow widths. The culture-language boundary will be fuzzy.
4. **~25% of active features are task-relevant**: Permutation null distributions + cross-source validation are required to control false discovery rate.
5. **Linear probe baseline is diagnostic**: A logistic regression probe trained on the same activations (sklearn, minutes to run) answers "does the model encode cultural information at this layer?" independently of whether SAEs can decompose it. If the probe separates IW clusters but SAEs don't → information exists but SAEs can't extract it (a claim about SAEs). If neither works → information doesn't exist at that layer (a claim about the model). This distinction is essential for interpreting Phase 3 results.
5. **12B is a size control**: Should show fewer cultural features than 27B. Gemma Scope 2 covers both (`google/gemma-scope-2-12b-pt`, `google/gemma-scope-2-27b-pt`).

### Starting SAE Configuration

| Purpose | SAE Type | Width | L0 | Layers (27B = 62 layers) |
|---------|----------|-------|----|--------------------------|
| Cultural feature discovery | Residual SAE | 256k | 50 | 31, 40, 53 (50/65/85% depth) |
| Language feature identification | Residual SAE | 256k | 50 | 31, 40, 53 |
| Circuit analysis (exploratory) | Skip transcoder | 256k | 50-100 | 31-53 |
| Cross-layer persistence | Crosscoder | 256k | 50 | {16, 31, 40, 53} |
| Steering | Residual SAE (output features) | 256k | 50 | 40-53 |

---

### Phase 3A: Probe Corpus Construction + Activation Extraction

**Goal**: Collect SAE activation vectors for carefully chosen probe texts.

#### 4 probe corpora

| Probe Set | Purpose | Content | Size | Variable |
|-----------|---------|---------|------|----------|
| `flores_200` | Language feature ID | Flores-200 devtest (parallel text) | ~1012 sentences × 27 langs | Language varies, content held constant |
| `wiki_culture` | Cultural topic features | Wikipedia passages on cultural practices per IW cluster, translated into all 27 langs | ~100 passages × 8 clusters × 27 langs | Cultural topic varies; language varies independently |
| `literary_corpus` | Cultural voice features | Native literary texts per language (novels, essays, folk tales) | ~200 passages × 27 langs | Both language and culture vary (authentically) |
| `wvs_stems` | Entangled signal | Top-50 Phase 2 completions per (lang, template) | ~10,800 texts | Both language and culture vary (model-generated) |

**Why four**: Each probe set controls different confounds:
- `flores_200` isolates **language** (same content, different languages → pure language features)
- `wiki_culture` isolates **cultural topic** (same topics translated into all languages → features that respond to "Protestant practices" regardless of language). But these are encyclopedic descriptions *about* a culture, not text *from* it.
- `literary_corpus` captures **cultural voice** — the style, framing, assumptions, and values embedded in authentic native text. A Finnish novel carries Finnish cultural priors in *how* it frames family and duty, not just *whether* it mentions them. Language and culture are fully confounded, but this is the most naturalistic signal.
- `wvs_stems` captures **model-generated** cultural signal from Phase 2 completions.

**Cross-source validation** is now a 3-way intersection: a feature is "validated cultural" only if significant in at least 2 of {`wvs_stems`, `wiki_culture`, `literary_corpus`}. Features found in all 3 are highest confidence. Features found only in `literary_corpus` + `wvs_stems` but not `wiki_culture` are likely culture-language entangled (the feature responds to cultural voice but not encyclopedic cultural topics).

**`wiki_culture` construction**: Curate ~100 English Wikipedia passages per IW cluster describing culturally distinctive practices (e.g., Protestant work ethic, Orthodox religious traditions, Confucian family values). Translate each passage into all 27 languages using an MT model or API. Monoculturality on `wiki_culture` is then computed by grouping activations by IW cluster label, **averaging across languages within each cluster**. This ensures a feature scored as "monocultural" responds to the cultural *content*, not the language it's written in.

**`literary_corpus` construction**: ~200 native passages per language sourced from:
- Project Gutenberg (public domain novels/essays — good coverage for major EU languages + English + Chinese)
- National digital libraries (e.g., Kansalliskirjasto for Finnish, Polona for Polish)
- OPUS/InterCorp parallel literary corpora (useful for finding comparable literary works across languages, though we use the native text, not translations)
- Folk tale collections (culturally grounded narratives, available for most languages)

Selection criteria: passages should be ~100-300 words, from works written originally in the target language (not translations), spanning genres (novels, essays, folk tales, religious texts) and time periods. No MT involved — this is the key advantage over `wiki_culture`.

**Interpreting disagreements between probe sets:**

| `wiki_culture` | `literary_corpus` | Interpretation |
|---|---|---|
| Significant | Significant | **High-confidence cultural feature** — responds to both cultural topics and cultural voice |
| Significant | Not significant | **Topic feature** — responds to encyclopedic cultural content but not authentic cultural expression |
| Not significant | Significant | **Culture-language entangled** — responds to authentic cultural voice but can't separate culture from language |
| Not significant | Not significant | Not cultural (discard) |

#### Activation extraction (`steering/extract_activations.py`)

```
python -m steering.extract_activations \
    --model_id gemma3_27b_pt \
    --sae_repo google/gemma-scope-2-27b-pt \
    --sae_id resid_post_layer40_256k_L050 \
    --probe_set flores_200 \
    --lang fin \
    [--batch_size 8] [--validate]
```

Core loop:
1. Load Gemma 3 27B PT (bf16)
2. Load SAE from HuggingFace (cache in `data/sae_cache/`)
3. For each probe text: forward pass → extract residual stream at target layer → encode through SAE encoder → sparse activation vector
4. Save as `.npz` (scipy.sparse): rows = texts, columns = SAE features, values = activations
5. Aggregate to per-text mean activation: shape `(n_texts, n_features)` sparse matrix

**SAE loading** (from Gemma Scope 2 docs):
```python
from huggingface_hub import hf_hub_download
import numpy as np
path = hf_hub_download("google/gemma-scope-2-27b-pt",
    "layer40/width256k/average_l0_50/params.npz")
params = dict(np.load(path))
# W_enc, W_dec, b_enc, b_dec, threshold (JumpReLU)
```

**Data volume**: ~110GB total across all (layer, probe_set, lang) combos. Sparse storage (L0≈50 nonzeros per token out of 256k) keeps this manageable.

**SLURM**: One job per (layer, probe_set, lang). 3 layers × (27 flores + 27 wiki_culture + 27 literary + 27 wvs_stems) = 324 jobs. 1× A100 80GB, ~1h each.

---

### Phase 3B: Feature Scoring and Selection

**Goal**: Compute monolinguality/monoculturality metrics, validate with null distributions, select feature sets.

#### Step 3B.1: Monolinguality scoring (`steering/score_features.py --mode language`)

Replicate Deng et al. (ACL 2025) directly on Flores-200 activations:
```
ν_s^L = μ_s^L - γ_s^L
```
where μ_s^L = mean activation of feature s on language L, γ_s^L = mean across all other languages.

**Expected**: 1-4 features per language with ν > 10 (based on Deng et al. on Gemma 2). INSERT into `feature_scores` with `metric = 'monolinguality'`.

#### Step 3B.2: Monoculturality scoring (`steering/score_features.py --mode culture`)

Same metric grouped by IW cluster:
```
ν_s^C = μ_s^C - γ_s^C
```
Computed separately on `wvs_stems` (entangled) and `wiki_culture` (language-controlled).

#### Step 3B.3: Null distribution + FDR control

1. Randomly shuffle language/cluster labels 1000 times, recompute ν → null distribution
2. Benjamini-Hochberg FDR correction at q < 0.01
3. **Cross-source validation**: feature is "validated cultural" only if significant in at least 2 of {`wvs_stems`, `wiki_culture`, `literary_corpus`}. Features significant in all 3 are highest confidence. `wiki_culture` monoculturality is computed across all 27 languages, grouped by IW cluster. `literary_corpus` monoculturality is grouped by language-to-cluster mapping (language confounded, but authentic).

**Expected yield**: ~dozens to low hundreds of validated cultural features per cluster (after FDR + cross-source filtering). The 3-way validation is stricter but produces more interpretable feature categories (topic-only, voice-only, both).

#### Step 3B.4: Feature partition

Classify validated features into a spectrum (not a clean partition):

| Category | Definition | Expected % |
|----------|-----------|------------|
| Pure language | High monolinguality, low monoculturality | 5-15% |
| Pure culture | Low monolinguality, high monoculturality (cross-validated) | 5-20% |
| Culture-language entangled | High on both | 30-50% |
| Non-specific | Low on both | 20-40% |

#### Step 3B.5: Size control (12B vs 27B)

Repeat 3B.1-3B.4 for Gemma 3 12B PT. Expected: fewer validated cultural features than 27B. Gemma Scope 2 has SAEs for 12B (`google/gemma-scope-2-12b-pt`).

**SLURM**: CPU-only, 16GB RAM, 2-4h per layer.

#### Step 3B.5.1: Linear probe baseline (`steering/probe_baseline.py`)

**Purpose**: Diagnostic — determine whether cultural information exists in the activations independently of whether SAEs can decompose it.

**Method**: For each layer's activation data (from 3A), train a simple sklearn `LogisticRegressionCV` (L2-regularized, 5-fold CV) to classify IW cluster membership from mean activation vectors.

```python
# Pseudocode — runs on the same .npz activation matrices from 3A
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score

X = load_activations(probe_set="wiki_culture", layer=40)  # (n_texts, hidden_dim) — raw residual, not SAE-encoded
y = cluster_labels  # IW cluster per text

probe = LogisticRegressionCV(Cs=10, cv=5, max_iter=1000, class_weight="balanced")
probe.fit(X_train, y_train)
acc = balanced_accuracy_score(y_test, probe.predict(X_test))
```

**Key distinction**: The probe operates on **raw residual stream activations** (hidden_dim = 3584 for 27B), NOT SAE-encoded features. This tests whether the information exists in the representation at all. SAE scoring (3B.2) tests whether it survives sparse decomposition.

**Interpretation matrix**:

| Probe accuracy | SAE monoculturality | Conclusion |
|---|---|---|
| High (>60%) | High | Cultural info exists AND SAEs capture it |
| High (>60%) | Low | Cultural info exists but SAEs lose it (feature hedging, sparsity) |
| Low (<40%) | Low | Cultural info not linearly separable at this layer |
| Low (<40%) | High | Spurious SAE features — methodology problem |

Run for all 3 target layers (31, 40, 53) on both 27B and 1B. Also run on Flores-200 with language labels as a positive control (should achieve >90% — languages are easily separable).

**SLURM**: CPU-only, 8GB RAM, <30min per layer. Trivial compute cost.

#### Step 3B.6: Output score extraction (`steering/score_features.py --mode output`)

**Why this step exists**: Detection features (which features activate when reading cultural text) are NOT the same as steering features (which features influence generation when active at the output position). Monoculturality from 3B.2 identifies detection features. Steering feature selection (3C.2) needs output scores — measured at the last-token position during autoregressive generation.

**Method**: Hook into Gemma 3 27B during `model.generate()` on a subset of `wvs_stems` (e.g., 20 completions per cluster × 8 clusters = 160 generations). At each generation step, extract the SAE encoding of the residual stream at the last-token position for layers 40-53. Aggregate per-feature mean activation across generation steps and across texts in each cluster.

```
output_score(s, C) = mean_{texts in C, generation steps}(SAE_encode(h_last_token)[s])
```

INSERT into `feature_scores` with `metric = 'output_score'`. Features are ranked by output_score for steering vector construction in 3C.2.

**SLURM**: 1× A100, ~2h (small subset, but requires GPU for generation + SAE encoding).

---

### Phase 3C: Steering Vector Construction

**Goal**: Build three types of steering vectors for each IW cluster.

#### Step 3C.1: CAA baselines (`steering/build_vectors.py --method caa`)

```
v_caa(C) = mean(h_cultural(C)) - mean(h_other)
```
Residual stream activations from `wvs_stems` completions, applied at layers 31-53 simultaneously (following Veselovsky et al.'s finding that cultural knowledge spans mid-to-late layers).

One CAA vector per (cluster, layer_range). Store as safetensors in `data/vectors/caa/`.

#### Step 3C.2: SAE culture vectors (`steering/build_vectors.py --method sae`)

Using validated cultural features from 3B, decoded to model space:
```
v_sae(C) = Σ_{top-K features for C} (mean_act_C - mean_act_overall) × decoder_direction
```

**Critical**: Select features by **output score** from step 3B.6, not by monoculturality metric. Detection features ≠ steering features (features at 50-65% depth detect cultural content; features at 66-100% depth steer generation). Focus on layers 40-53 (66-85% depth). Sweep K ∈ {5, 10, 25, 50}.

Store in `data/vectors/sae/`.

#### Step 3C.3: FGAA hybrid (`steering/build_vectors.py --method fgaa`)

Project CAA vector onto the subspace spanned by top-K SAE cultural decoder directions:
```
v_fgaa(C) = proj(v_caa(C), span(d_1, ..., d_K))
```
Interpretable (lives in SAE feature subspace) + benefits from CAA's dense optimization. Store in `data/vectors/fgaa/`.

#### Step 3C.4: Alpha tuning

Sweep α ∈ {0.5, 1.0, 1.5, 2.0, 3.0, 5.0} on validation set (2 templates × 2 langs × 20 samples = 80 completions per α).

**Coherence guard**: perplexity of steered completions under the unsteered model must not exceed 2× baseline. Register best (vector_id, α) in `steering_vectors` table.

---

### Phase 3D: Steered Generation and Evaluation

**Goal**: Generate steered completions and classify them using the Phase 2 pipeline.

#### Step 3D.1: Steering hook (`steering/steer.py`)

```python
def steering_hook(module, input, output, vector, alpha):
    # Modify last token position only (autoregressive generation)
    output[:, -1, :] += alpha * vector
    return output
```

For CAA/FGAA: add dense vector at each layer in range. For SAE: clamp specific features via encode → modify sparse activations → decode.

#### Step 3D.2: Extend `inference/sample.py`

Add `--steering-config` argument: accepts a `vector_id` from `steering_vectors` table or inline JSON. Stored in `completions.steering_config`.

#### Step 3D.3: Generation matrix

3 methods × 8 clusters × 5 pilot langs × 8 templates × 200 samples ≈ **210,000 completions**.

Plus **4 cross-language transfer tests** (the core novelty):
1. Orthodox culture in English → expect shift toward traditional/survival
2. Protestant Europe in Arabic → expect shift toward secular/self-expression
3. East Asian culture in German → expect shift toward collectivist
4. English-speaking culture in Chinese → expect shift toward individualist

**SLURM**: One job per (method, target_cluster). 3 × 8 = 24 jobs, 1× A100, ~4h each.

#### Step 3D.4: Classification

Run Phase 2 classifier unchanged on all steered completions.

---

### Phase 3E: Analysis and Validation

#### Step 3E.1: Steering effect sizes (extend `analysis/analyze.py` with `steered` subcommand)

Cohen's d (steered vs baseline) per method per IW dimension. **Success threshold**: mean d > 0.3.

**Key comparison table**:

| Method | Mean d (IW dims) | Cross-lang transfer d | Coherence (PPL ratio) |
|--------|-----------------|----------------------|----------------------|
| CAA | ? | ? | ? |
| SAE | ? | ? | ? |
| FGAA | ? | ? | ? |
| Baseline | 0 | 0 | 1.0 |

#### Step 3E.2: Ablation validation

Zero out each top-K cultural feature during generation. Feature passes if ablation reduces the cultural shift observed in baseline completions.

**Expected**: ~50-75% of features pass ablation validation.

#### Step 3E.3: Culture vector geometry

PCA/UMAP of culture vectors from all 8 clusters. Three convergence checks:
1. SAE vectors vs CAA vectors (should correlate if both capture cultural signal)
2. Culture vectors vs Phase 2 behavioral UMAP (internal representations should match behavioral profiles)
3. Culture vectors vs human IW map positions (the gold standard)

#### Step 3E.4: Crosscoder analysis (exploratory)

If residual SAE results are promising, use crosscoders at {16, 31, 40, 53} to check:
- Do cultural features persist from layer 31 (detection) to layer 53 (output)?
- Features that bridge detection→output are the strongest candidates for interpretable steering

#### Step 3E.5: Size gradient (12B / 27B)

| Model | Expected cultural features | Expected steering effect |
|-------|--------------------------|------------------------|
| 12B | Fewer / weaker | Weaker |
| 27B | Strongest | Strongest |

If gradient doesn't appear, methodology has a problem.

---

### Phase 3 Data Volume Estimates

| Data | Records/Size |
|------|-------------|
| Probe texts (all sets) | ~60,000 texts (multilingual wiki_culture adds ~21k) |
| Activation files | ~243 npz files, ~80 GB total |
| Feature scores (3 layers × 256k × 27 groups × 3 metrics) | ~62M rows |
| Steering vectors | ~72 safetensors (3 methods × 8 clusters × 3 alpha values) |
| Steered completions | ~210,000 new rows in `completions` |
| Steered classifications | ~210,000 new rows in `classifications` |

---

### Phase 3 Implementation Order

| Step | What | Depends on |
|------|------|------------|
| 3A.0 | Schema fix (steering_config `"none"` sentinel) | Phase 2 complete |
| 3A.1 | Probe corpus preparation (download Flores, curate wiki passages) | — |
| 3A.2 | `steering/extract_activations.py` + SAE loading utilities | 3A.0, 3A.1 |
| 3A.3 | SLURM activation extraction (165 jobs) | 3A.2 |
| 3B.1 | `steering/score_features.py` (monolinguality + monoculturality) | 3A.3 |
| 3B.2 | Null distribution + FDR + cross-source validation | 3B.1 |
| 3B.3 | 1B negative control (extract + score) | 3A.2, 3B.1 |
| 3B.5.1 | **Linear probe baseline** (sklearn logistic regression on raw activations) | 3A.3 (raw activations, not SAE-encoded) |
| 3B.6 | Output score extraction (last-token SAE activations during generation) | 3A.2, Phase 2 completions |
| 3C.1 | CAA baselines (**can run in parallel with 3B**) | 3A.3 |
| 3C.2 | SAE vectors | 3B.2, 3B.6 |
| 3C.3 | FGAA hybrid vectors | 3C.1, 3C.2 |
| 3C.4 | Alpha tuning (small validation set) | 3C.1-3 |
| 3D.1 | `steering/steer.py` (hook implementation) | 3C.1 |
| 3D.2 | Extend `inference/sample.py` with `--steering-config` | 3D.1 |
| 3D.3 | Full steered generation (24 SLURM jobs) | 3D.2, 3C.4 |
| 3D.4 | Classify steered completions | 3D.3 |
| 3E.1 | Effect sizes + comparison table | 3D.4 |
| 3E.2 | Ablation validation | 3D.1, 3B.2 |
| 3E.3 | Culture vector geometry (PCA/UMAP) | 3C.1-3, 3E.1 |
| 3E.4 | Crosscoder analysis (exploratory) | 3E.1, if results promising |
| 3E.5 | 1B/4B/27B size gradient | 3B.3, 3E.1 |

**Critical path**: 3A.1 → 3A.2 → 3A.3 → 3B.1 → 3B.2 → 3C.2 → 3C.4 → 3D.2 → 3D.3 → 3D.4 → 3E.1

**Parallelism opportunity**: CAA baseline (3C.1 → 3D) can run in parallel with SAE feature scoring (3B), saving ~1 week.

---

### Phase 3 Verification Checklist

- **3A.2**: `--validate` → activation sparsity ~L0=50 nonzeros per token
- **3B.1**: Top monolinguality features for eng/fin/zho are distinct; ν > 10 for top-1
- **3B.2**: >90% of 256k features non-significant; <1% significant after FDR correction
- **3B.3**: 12B has fewer validated cultural features than 27B
- **3B.5.1**: Linear probe on Flores-200 with language labels achieves >90% balanced accuracy (positive control); probe on wiki_culture with IW cluster labels achieves >45% (above 8-class chance = 12.5%) for 27B layer 40
- **3C.1**: CAA vectors for Protestant vs Orthodox have cosine similarity < 0.9
- **3C.4**: Steered completions at best α are coherent (PPL ratio < 2.0)
- **3E.1**: At least one method achieves mean Cohen's d > 0.3 on steered vs baseline
- **3E.3**: Culture vector UMAP cluster topology agrees with IW map (Protestant near Catholic, both far from East Asian)

---

### Realistic Expectations

| Prediction | Confidence | Rationale |
|-----------|-----------|-----------|
| Language features replicable on Gemma 3 27B | High | Direct extension of Deng et al., well-established |
| CAA steering produces measurable cultural shifts | High | Already demonstrated by Veselovsky et al. |
| Some SAE monoculturality features exist | Moderate-high | Abstract features exist in large models (Anthropic scaling monosemanticity) |
| Clean culture-language partition | Low | Feature hedging + deep correlation predict fuzzy spectrum |
| SAE steering outperforms CAA | Low | DeepMind negative results + input/output feature split |
| FGAA competitive with CAA while interpretable | Moderate | Published 2-3× improvement over raw SAE steering |
| Cross-language cultural transfer | Moderate | CAA vectors shown language-agnostic; SAE-based unproven |
| 12B shows fewer cultural features than 27B | Moderate-high | Smaller capacity, less cultural exposure in training |

**A negative result is publishable**: The first systematic attempt to disentangle culture from language using SAEs, with well-characterized failure modes, would be a significant contribution regardless of outcome.
