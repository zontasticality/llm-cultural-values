# Active Analysis Plan — Phase 2 Ablation & Phase 3 SAE

Status as of 2026-04-20. This doc tracks what we've found, what's deferred, and what's next.

## Phase 2: Where We Are

### Data
- **401,722 classified completions under v2 classifier** (`gemma3_27b_it_v2`, post-B6-fix)
- **379,374 under v1** (`gemma3_27b_it`, retained for B6 comparison)
- Trimmed-only subset: 202,556 completions
- 100% logprob coverage on all three dimensions (v2)
- Models: gemma3_27b_pt, gemma3_12b_pt, eurollm22b, 22 HPLT monolingual
- 25 languages (22 EU + ara, hin, tur; zho/jpn via expanded Gemma set)
- Human comparison: 21 EU languages via EVS survey data in `../eurollm/data/survey.db`
- **30 v200 alternative prompts** sampled + classified (April 2026) for the 15 low-quality stems

### Status milestones
- **2026-04-17**: B6 template leak identified. `make_classifier_prompt` was passing `Prompt type: {template_id}` in user message. Fixed at `classify/prompts.py`; dropped `template_id` arg entirely. Bumped classifier version to `gemma3_27b_it_v2`.
- **2026-04-17**: Alternative prompt strategy. `prompt_quality.py` scored all 235 trimmed prompts; 15 low-quality stems got 2 alternatives each written to `prompts.quality_notes` via `write_quality_notes.py`. Alternatives inserted as variant_idx 200–201 via `sync_translations.py`.
- **2026-04-17**: Translation JSONs became authoritative. `scripts/sync_translations.py` dumps prompts table back to `data/translations/{lang}.json` so `db.populate` on a fresh DB reproduces all 532 prompts (v0 + v100 + v200). JSONs live in git; DB is derived.
- **2026-04-18**: Full v200 sampling job run on Unity. ~22k completions.
- **2026-04-18**: v2 classify job submitted as 55752753 (gpu-preempt, `constraint=a100-80g|h100`). COMPLETED in 4h3m on A100-80g. 401,722/402,006 classified (99.93% parse success).
- **2026-04-20**: Pulled + merged. All four analysis scripts rerun: `prompt_quality.py` (quality scores in DB), `quality_figures.py` (4 figures), `compare_to_wvs.py` (`iw_comparison_v2.png`), `ablation_analysis.py all` (B1–B6 v2), `analyze.py all` (chi-square + Cohen's d). Typst report updated with v2 numbers, new Section 7 (v1→v2 comparison), new Summary bullets. PDF rebuilt.
- **2026-04-20**: Manual inspection of 97 random completions across all (model, lang) pairs. Found ~6% classifier category errors and a clear pattern of **silent language drift** — wrong-language completions receiving real content labels instead of `other`.
- **2026-04-21**: Added `completions.detected_lang` + `detected_lang_conf` columns via `migrate_completions_lang()`. `scripts/detect_lang.py` uses `lingua-language-detector` over all 402,006 completions (8.5min walltime). 98.9% detected, 95.9% match prompt lang, **2.6% silent drift in trimmed subset** (5,283 / 204,000). 57% of drifted rows carried a real content category label (not `other`) — the silent contaminant the B6 fix and quality_score missed.
- **2026-04-21**: Added `--lang-match` flag to `ablation_analysis.py`, `compare_to_wvs.py`, `analyze.py`, and `db/load.py::load_results`. Filter: `(c.detected_lang IS NULL OR c.detected_lang = p.lang)`. Re-ran all v2 analyses with the filter. **Every headline signal survives**: EuroLLM-SE ρ=0.562→0.568 (p=0.008→0.007); chi-squares all nudge up (Gate 2 stronger on cleaner data); values×v82 0.703→0.697; EuroLLM English-vs-Polish Cohen's d unchanged at 0.35/0.34/0.39. v2 findings are robust to language-drift contamination. Section 7 of typst report gets a new subsection documenting the robustness check.

### Ablation Results (v2, `scripts/ablation_analysis.py --classifier gemma3_27b_it_v2`)

**B1: Logprob validation** — PASS. 99.9%+ argmax agreement on all three dimensions. Logprobs reliable.

**B2: Logprob EVs vs greedy (v2)** — EV and greedy give near-identical Spearman rho against EVS (max delta 0.02). EuroLLM-22B SE is the only significant result (greedy ρ=0.562, EV ρ=0.542, both p<0.01). Dimension scores still compressed to ~0.3-unit range around 3.

**B3: Per-prompt signal (v2)** — Now `values:100` SE ρ=0.560 (p=0.008 ***) is the strongest composite. `self_concept:200` (v200 alternatives, 10 langs) shows SE ρ=0.636 (p=0.048). `moral:100` SE ρ=0.417 (p=0.060 trend).

**B3b: Per-prompt × per-question (v2)** — Same qualitative story, essentially identical numbers:
- **"values" template** tracks specific SE questions strongly: v82 (gay couples, ρ=0.70 ***), v154 (abortion, 0.63 ***), v153 (homosexuality, 0.57 ***), v63 (importance of God, 0.54 **), v98 (petition signing, 0.48 **)
- **"childrearing"** maps to religion questions (v6: 0.51, v63: 0.50, v154: 0.50)
- **v86 (independence) and v95 (obedience)** — no prompt tracks these; uncapturable from TST-style completions
- **"family" and "success"** carry almost no signal — unchanged from v1

**B4: Variance decomposition (v2)** — Language eta²=0.010 TS, 0.010 SE. Template 1.6% TS, 5.0% SE. Lang×template interaction 4.8–8.7%. Signal remains prompt-dependent.

**B5: Categories vs EVS (v2)** — *All dropped to non-significance.* `emotional_state` ↔ TS (ρ=0.41, p=0.066 *trend*), `other` ↔ SE (ρ=-0.39, p=0.081 *trend*). v1's `personality_trait` (ρ=0.47, p=0.03) and `family_social` (-0.44, p=0.045) correlations were the B6 leak leaking into EVS alignment.

**B6: Template label bias (v2)** — All categories still flagged BIAS?, but dominance values reduced vs v1:
- `family_social`: 98.5 → 72.4 (-27%)
- `personality_trait`: 27.0 → 14.7 (-46%)
- `physical_attribute`: 10.6 → 3.4 (-68%)
- `occupation_achievement`: 18.5 → 14.2 (-23%)

Residual dominance is partly genuine (family-template prompts naturally elicit family-content completions); the v2 numbers are the honest floor under the current classifier.

### Infrastructure Built This Cycle

- `classify/prompts.py` — dropped `template_id` from `make_classifier_prompt` (B6 fix); added classifier version-history docstring block as canonical map from `classifier_model` name → prompt config
- `classify/classify_local.py` — default `--classifier-name gemma3_27b_it_v2`
- `slurm/run_classify.sh` — `constraint=a100-80g|h100` (40GB A100s OOM on Gemma-3-27B-IT)
- `db/schema.py` — `migrate_prompts_quality()`, `migrate_completions_lang()` (auto-run by `init_db()`)
- `db/load.py::load_results()` — `lang_match` kwarg: when True, filter with `(detected_lang IS NULL OR detected_lang = p.lang)`
- `scripts/prompt_quality.py` — per-prompt quality scoring via fraction-`other`; populates `quality_score` and samples 20 completions per prompt
- `scripts/write_quality_notes.py` — writes 15 low-quality prompts' alternatives as structured JSON to `quality_notes`
- `scripts/sync_translations.py` — bidirectional prompts ↔ translation-JSON sync (idempotent on prompt_text, not variant_idx)
- `scripts/quality_figures.py` — 4 figures in `figures/trimmed/quality/`
- `scripts/inspect_samples.py` — stratified random sampler for manual classifier audit (1 per model×lang pair by default)
- `scripts/detect_lang.py` — lingua-based LID over all completions, populates `detected_lang` + `detected_lang_conf`, reports mismatch per model and top drift destinations
- `scripts/ablation_analysis.py`, `compare_to_wvs.py`, `analysis/analyze.py` — added `--lang-match` flag

### Deferred Analysis Work

These remain non-blocking:

1. **Hand-labeled gold set for Cohen's κ (Gate 3)** — A stratified 30 completions × (language, model) sample, hand-labeled on all four fields (category + 3 Likerts), would give per-pair confusion matrices and an honest agreement number. Never done. Becomes more compelling under v2 given the lost v1 signals need independent verification.

2. **Quality-weighted F4 aggregation** — Rerun `compare_to_wvs.py` with `--weight-by-quality` (not yet implemented) or hard `--min-quality 0.8` filter. Quick robustness check; would confirm whether EuroLLM-SE ρ=0.56 survives prompt quality filtering.

3. **Per-question distributional analysis** — Instead of collapsing logprob [p1..p5] to E[x], compare full 5-bin distribution shape across languages. B2 showed EV≈greedy, so low priority.

4. **Compare v100 vs v200 quality at the cell level** — 30 v200 alternatives have 22k completions + 22k classifications. Heatmap of (lang, template): v100 quality vs v200 quality would confirm the alternatives actually fix the short-stem problem at the (model, language) level, not just at the global-quality level.

---

## Phase 3: SAE Plan

### Go/No-Go Assessment

Phase 2 pilot gates (re-evaluated under v2 classifier 2026-04-20):
- Gate 1 (Cohen's d): Finnish vs Romanian d > 0.3 on dim_trad_secular — **FAILS** (best: 0.12 Gemma-27B). EuroLLM English vs Polish crosses threshold on all three dims (|d|=0.35/0.33/0.39).
- Gate 2 (chi-square): Cross-cluster category differences p < 0.01 — **PASSES** (p < 10⁻¹³⁵ for all three multilingual models under v2, all pairwise comparisons significant)

Gate 2 robustness under v2 clears Phase 3. Primary outcome is the content-category signal; Likert dimensions are a secondary signal and only the EuroLLM-SE axis (ρ=0.56, p=0.008) holds up as human-aligned under the honest classifier.

### Strategy (from RESEARCH.md)

1. **Load Gemma 3 27B + Gemma Scope 2 residual SAEs**
   - Layers 31, 40, 53 (50/65/85% depth of 62 layers)
   - Width: 256k, L0: 50 (recommended starting config)
   - `resid_post` site

2. **Identify language-specific features**
   - Deng et al. monolinguality metric: nu_s^L = mu_s^L - gamma_s^L
   - Feed parallel text (Flores-200) in all languages through the model
   - Rank features by nu for each language
   - Prior work: 1-4 features per language capture most signal

3. **Identify culture-specific features**
   - Feed culturally-loaded text, compute "monoculturality" metric grouped by IW cluster
   - Two independent text sources for cross-validation:
     a. WVS translations in each language (culture+language confounded)
     b. English Wikipedia about different cultures (language controlled, culture varies)
   - Features with high nu for any culture cluster = candidates

4. **Build culture vectors**
   - culture_only(C) = mean activations for cluster C - projection onto language subspace
   - Validate: culture vectors from different languages for same IW cluster should be similar
   - Validate: should activate on text about Nordic values regardless of language

5. **Steer and evaluate**
   - h_steered = h_base + alpha * culture_vector(C)
   - Re-run Phase 2 sampling pipeline on steered model
   - Compare steered cultural profiles against Phase 2 profiles

6. **Baselines**
   - CAA (Contrastive Activation Addition) — run in parallel
   - Linear probes on raw residual stream — diagnostic: if probe works but SAEs don't, information exists but SAEs can't extract it

### Realistic Expectations (from lit review)

- Language feature subtraction: **high confidence** it works
- Some cultural features identifiable: **moderate confidence**
- Clean culture-language disentanglement: **low confidence** (feature hedging = correlated features merge)
- SAE steering for culture transfer: **likely weak** (CAA/FGAA may outperform)
- 1B model = negative control (too small for abstract cultural features)
- **Negative result is publishable** — first systematic attempt at SAE-based cultural feature identification

### First Implementation Step

Get SAE loading + Flores activation extraction working on the cluster:
- Download Flores-200 parallel sentences for all 27 languages
- Forward pass through Gemma 3 27B, extract residual stream activations at target layers
- Load Gemma Scope 2 SAEs, encode activations to get feature activations
- Compute monolinguality scores as proof of concept
