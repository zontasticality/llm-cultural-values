# Robustness Plan: Getting More Resilient Cultural Signal

## Problem Statement

Current results show model architecture (HPLT vs EuroLLM) dominates over
language in the PCA/t-SNE/JSD analyses. Position bias is large (median
0.35-0.50) and our forward+reversed averaging is a crude 2-sample correction.
The rephrase experiment shows JSD ~0.08 just from rewording — comparable to
cross-language differences. We need to determine whether cultural signal is
genuinely weak or whether our elicitation method is too noisy to detect it.

## Phase 1: Multi-Permutation Position Debiasing

### What
Instead of just forward + reversed (2 orderings), run K random permutations of
the option order for each question and average across all K.

### Why
With 4 options there are 24 possible orderings; with 5 there are 120. Our
current 2-sample average leaves a lot of position-interaction variance on the
table. Random permutation averaging is the standard approach in the survey
elicitation literature (Dominguez-Olmedo et al. 2023, Wang et al. 2024).

### Implementation
- New script: `eurollm/extract_logprobs_robust.py`
- For each question, generate K random permutations of option order (K=8 for
  ≤5 options, K=12 for likert10). For small option counts (2-3), use all
  possible permutations instead of sampling.
- For each permutation: construct prompt, extract logprobs, renormalize, remap
  to semantic values.
- Average the K remapped distributions → one debiased distribution per question.
- Output same parquet schema as before plus `n_permutations` column.
- Compare new results against old forward+reversed results.

### Compute Cost
~4-6x more model calls per question (K=8 vs K=2). For HPLT 2.15B models this
is still fast (~2min per language). For EuroLLM-22B, ~10min per language.
Total: ~45min HPLT (22 langs) + ~4hr EuroLLM (22 langs). Very doable in one
SLURM submission.

## Phase 2: Prompt Template Ensemble

### What
Run each question through M different prompt templates and average the
resulting distributions. Templates vary the structural formatting while keeping
the semantic content identical.

### Why
The rephrase experiment showed JSD ~0.08 from wording changes alone. A single
template is a single point estimate. Averaging across templates reduces
template-specific bias, analogous to ensemble methods in ML.

### Templates to test (M=3-4)
1. **Current format** (baseline):
   ```
   {question text}
   1. {label}
   2. {label}
   ...
   Answer:
   ```

2. **Integrated options** (from rephrase experiment):
   ```
   {question text} — {label1}, {label2}, ..., or {labelN}?
   1. {label}
   2. {label}
   ...
   Answer:
   ```

3. **Parenthetical format**:
   ```
   {question text}
   (1) {label}
   (2) {label}
   ...
   Answer:
   ```

4. **Inline format** (no numbered list):
   ```
   {question text} ({label1}=1, {label2}=2, ..., {labelN}=N)
   Answer:
   ```

### Implementation
- Modify `prompt_templates.py` to accept a `template_style` parameter.
- `extract_logprobs_robust.py` iterates over both permutations AND templates.
- Final distribution = average over K permutations × M templates.

### Compute Cost
M=3 templates × K=8 permutations = 24 calls per question vs current 2. Still
manageable: ~15min per HPLT language, ~45min per EuroLLM language.

## Phase 3: Re-analysis with Robust Distributions

### What
Re-run the full analysis pipeline (JSD matrix, PCA, t-SNE, quality report)
using the robustly-debiased distributions. Compare to original results.

### Key Questions
1. Does the HPLT-vs-EuroLLM split shrink when position bias is better
   controlled? (If yes → the architecture difference was partly a bias
   artifact.)
2. Do same-language pairs (HPLT-X, EuroLLM-X) become more similar? (If yes →
   cultural signal was being masked by template/position noise.)
3. Do cultural clusters (Nordic, Western, etc.) become more visible in
   PCA/t-SNE?
4. Does cross-language JSD decrease within model type? (More consistent
   cultural signal with less noise.)

### Figures to Generate
- Side-by-side PCA/t-SNE: old vs robust
- JSD heatmap comparison (old vs robust, or delta heatmap)
- P_valid improvement from multi-permutation
- Position bias magnitude: old (2-sample) vs robust (K-sample)

## Phase 4: Calibration Against Human Survey Data (stretch)

### What
Compare LLM response distributions to actual EVS human survey results for the
same countries. This is the ultimate test of whether the models capture real
cultural variation.

### Why
Even if cross-language LLM distributions differ, they could be reflecting
training corpus artifacts (topic frequency, writing style) rather than genuine
cultural values. Correlation with human survey data would validate the method.

### Implementation
- Download EVS wave 5 integrated dataset from GESIS
- Compute country-level response distributions for each question
- Map languages to countries (e.g., deu→Germany, fra→France)
- Compute JSD(LLM distribution, human distribution) per question per country
- Scatter plot: LLM expected value vs human expected value, per question

## Implementation Order

1. **Phase 1** first — it's the highest-impact change (better position
   debiasing) with moderate implementation effort.
2. **Phase 2** next — template ensembling builds on Phase 1 infrastructure.
3. **Phase 3** follows naturally — just re-running analysis.
4. **Phase 4** is a stretch goal requiring external data download.

## Files to Create/Modify

### New Files
- `eurollm/extract_logprobs_robust.py` — Multi-permutation + multi-template extraction
- `eurollm/run_robust.sh` — SLURM launcher for all 44 model-lang pairs
- `eurollm/compare_robustness.py` — Side-by-side comparison of old vs robust

### Modified Files
- `eurollm/prompt_templates.py` — Add `template_style` parameter to format_prompt()
- `eurollm/analyze.py` — Add `compare` subcommand for old-vs-robust analysis
