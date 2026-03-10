# Gemma SAE — Culture vs Language Feature Isolation

## Background: Gemma Scope 2

Gemma Scope 2 (Dec 2025) is a comprehensive open interpretability suite for all **Gemma 3** models (270M, 1B, 4B, 12B, 27B), with SAEs and transcoders on every layer of both pretrained and instruction-tuned variants.

Key improvements over Gemma Scope 1:
- **Matryoshka training** — SAEs detect more useful concepts, resolves flaws from Gemma Scope 1
- **Transcoders & skip-transcoders** — decode multi-step computations across layers
- **Cross-layer transcoders** — for 270M and 1B models
- **Partial residual stream crosscoders** — for all base models

Resources:
- Technical paper: https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/Gemma_Scope_2_Technical_Paper.pdf
- Blog: https://deepmind.google/blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/
- Weights: https://huggingface.co/google/gemma-scope-2
- Interactive explorer: https://www.neuronpedia.org/gemma-scope-2
- Gemma Scope 1 paper (background): https://arxiv.org/abs/2408.05147

### Available Gemma Scope 2 Repos (Gemma 3)

| Repo | Model | Notes |
|------|-------|-------|
| `google/gemma-scope-2-27b-pt` | 27B base | Primary target |
| `google/gemma-scope-2-27b-it` | 27B instruct | |
| `google/gemma-scope-2-12b-pt` | 12B base | |
| `google/gemma-scope-2-12b-it` | 12B instruct | |
| `google/gemma-scope-2-4b-pt` | 4B base | Size-matched comparison |
| `google/gemma-scope-2-4b-it` | 4B instruct | |
| `google/gemma-scope-2-1b-pt` | 1B base | Size-matched comparison |
| `google/gemma-scope-2-1b-it` | 1B instruct | |

**Layer coverage options** (for 27B):
- `resid_post`, `attn_out`, `mlp_out` — SAEs at 4 layers (25%, 50%, 65%, 85% depth), multiple widths & L0
- `resid_post_all`, `attn_out_all`, `mlp_out_all`, `transcoder_all` — all layers, fewer width/L0 combos

**SAE widths**: 16k, 64k, 256k, 1M
**Target L0**: "small" (10-20), "medium" (30-60), "large" (60-150)

## Existing Work Directly Relevant to Our Intuition

### 1. Language-Specific SAE Features (ACL 2025)
**"Unveiling Language-Specific Features in LLMs via Sparse Autoencoders"** — Deng et al.
https://arxiv.org/abs/2505.05111

Key findings:
- Defined a **monolinguality metric** ν: for feature s and language L, compute ν = μ_s^L - γ_s^L (mean activation in L minus mean across all other languages)
- Found features strongly tied to specific languages (high ν scores)
- **Ablating** language-specific features (directional ablation: x' = x - d̂d̂ᵀx) degrades only that language's performance, leaving others intact
- Some languages have **synergistic features** — ablating them together has greater effect than individually
- Used language-specific feature activations to **gate steering vectors** for controlled language switching

This is exactly the "language signal" we need to subtract.

### 2. Cultural Steering Vectors (April 2025)
**"Localized Cultural Knowledge is Conserved and Controllable in LLMs"**
https://arxiv.org/abs/2504.10191

Key findings:
- Used **Contrastive Activation Addition (CAA)**: v = mean(h_cultural) - mean(h_control)
- Found cultural localization mechanism concentrated in **layers 19-28** (for their model)
- Activation patching shows implicit and explicit cultural localization spike/drop at the same layers (23 and 30)
- Cultural steering vectors are **task-agnostic** (trained on names, transfers to cities and CulturalBench)
- Cultural steering vectors are **language-agnostic** (English-derived vectors work for Russian, Turkish, French, Bengali)
- Achieved 65-78% localization rate vs 85-91% with explicit prompting

This paper does culture but does NOT disentangle it from language. They acknowledge this limitation.

### 3. LinguaLens (EMNLP 2025)
**"LinguaLens: Towards Interpreting Linguistic Mechanisms of LLMs via Sparse Auto-Encoder"**
https://arxiv.org/abs/2502.20344 | https://github.com/THU-KEG/LinguaLens

- Uses SAEs to extract linguistic features (morphology, syntax, semantics, pragmatics)
- Built counterfactual sentence pairs for 100+ linguistic phenomena in English and Chinese
- Shows cross-layer and cross-lingual distribution patterns
- Demonstrates causal control via feature intervention

### 4. Multilingual != Multicultural (Feb 2025)
https://arxiv.org/abs/2502.16534
- Compared LLM responses against WVS data across 4 languages
- Found **no consistent relationship** between language capability and cultural alignment
- Language ability does not imply cultural representation

### 5. Prompt Language vs Cultural Framing (Nov 2025)
https://arxiv.org/abs/2511.03980
- Explicit cultural framing is more effective than prompt language for eliciting cultural values
- Combining both is no more effective than cultural framing in English alone
- Models have systematic bias toward a few countries (NL, DE, US, JP)

## The Core Intuition: Culture - Language = Transferable Cultural Style

### Formalization

Let h(x) be the activation vector for input x at some layer. The SAE decomposes this into sparse features:

```
h(x) ≈ Σ_i a_i(x) · d_i    (where a_i are sparse activations, d_i are decoder directions)
```

We want to partition features into three sets:
- **L_features**: language-specific (activate for language L regardless of content)
- **C_features**: culture-specific (activate for culture C regardless of language)
- **Shared features**: neither language nor culture specific

The hypothesis:
```
culture_vector(C) = mean_over_cultural_items(Σ_{i ∈ C_features} a_i · d_i)
language_vector(L) = mean_over_language_items(Σ_{i ∈ L_features} a_i · d_i)
```

To generate text with culture C in language L':
```
h_steered = h_base + α·culture_vector(C) + β·language_vector(L') - β·language_vector(L_original)
```

---

# Implementation Plan: "I Am" Cultural Completion Comparison

## Overview

A systematic comparison framework that measures how different models complete culturally diagnostic sentence stems (starting with "I am") across languages, then compares the resulting cultural profiles against each other and against WVS/EVS human data.

The framework serves two purposes:
1. **Standalone**: Compare cultural self-concept across models and languages
2. **Validation harness for SAE steering**: Once culture vectors are extracted (Phase 3), test whether SAE-steered completions match naturally-cultured model behavior

## Models

| Model | Size | Languages | Role |
|-------|------|-----------|------|
| HPLT monolingual (`HPLT/hplt2c_{lang}`) | 2.15B | 22 European | Monolingual cultural baseline |
| EuroLLM-22B (`utter-project/EuroLLM-22B-2512`) | 22B | 22 European | Multilingual European baseline |
| Gemma 3 27B PT (`google/gemma-3-27b-pt`) | 27B | 22 European + 5 expanded | Primary multilingual model |
| Gemma 3 4B PT (`google/gemma-3-4b-pt`) | 4B | 22 European + 5 expanded | Size-matched comparison to HPLT |
| Gemma 3 1B PT (`google/gemma-3-1b-pt`) | 1B | 22 European + 5 expanded | Size-matched comparison to HPLT |
| Gemma 3 27B PT + SAE steering | 27B | 22 European + 5 expanded | Culture-steered (Phase 3) |

**Explicit model-size comparison axis**: By including Gemma 3 at 1B, 4B, and 27B, we can distinguish cultural signal from model capability effects. HPLT 2.15B sits between 1B and 4B.

## Languages

### European (22) — available for all models
bul, ces, dan, deu, ell, eng, est, fin, fra, hrv, hun, ita, lit, lvs, nld, pol, por, ron, slk, slv, spa, swe

### Expanded (5) — Gemma 3 models only
zho (Chinese), jpn (Japanese), ara (Arabic), hin (Hindi), tur (Turkish)

Rationale: The strongest cultural contrasts in the TST literature are Western vs East Asian / Middle Eastern. All 22 European languages are relatively culturally similar. Adding these 5 provides much stronger signal for the individualism-collectivism and traditional-secular axes. HPLT monolingual models only cover European languages, so the expanded set runs on Gemma 3 only — this is itself a useful comparison (European-only vs global language set).

### Cultural Clusters

```
European:
  Nordic:        dan, fin, swe
  Western:       deu, fra, nld, eng
  Mediterranean: ita, spa, por, ell
  Central:       ces, hun, pol, slk, slv
  Baltic:        est, lit, lvs
  Southeast:     bul, hrv, ron

Expanded:
  East Asian:    zho, jpn
  South Asian:   hin
  Middle Eastern: ara, tur
```

## Prompt Design

### Phase 1 stem: "I am"

The prompt is the translation of "I am" in each language, formatted as a bare completion prompt (no system prompt, no instructions):

```
English:  "I am"
German:   "Ich bin"
French:   "Je suis"
Chinese:  "我是"
Japanese: "私は"
Arabic:   "أنا"
Hindi:    "मैं"
Turkish:  "Ben"
...
```

The model completes the sentence. We measure what it produces.

**Note on grammar**: Some languages require different forms. "I am" may be "I am a ___" in languages with articles. The translations should be natural sentence-start fragments that elicit self-description — not word-for-word translations. Specifically:
- Languages with copula + indefinite article (English, German, French): "I am a" / "Ich bin ein" / "Je suis un"
- Languages without articles (Chinese, Japanese, Turkish, most Slavic): "I am" / "我是" / "私は" / "Ben"
- Consider both "I am" and "I am a" variants for languages where both are natural

### Future stems (not in scope for Phase 1, but reserve the infrastructure)
- "The most important thing in life is"
- "A good person"
- "Children should"
- "When I am old"
- "My country"

## Method: Logprob Extraction Over Category Tokens

### Approach

For each (model, language, prompt), compute the probability of culturally diagnostic continuation sequences using **autoregressive logprob extraction**:

```
P("a father" | "I am") = P("a" | "I am") × P("father" | "I am a")
```

This requires multiple forward passes per category item (one per token in the continuation), but is exact and deterministic.

### Implementation

For each category item C = [t_1, t_2, ..., t_k] (tokenized continuation):
1. Start with prompt tokens
2. For each token t_i in C:
   - Run forward pass on prompt + [t_1, ..., t_{i-1}]
   - Extract logprob of t_i at the next-token position
3. P(C | prompt) = exp(Σ logprobs)

With KV caching, this is efficient: one initial forward pass for the prompt, then incremental passes for each continuation token. Batch across category items that share prefixes (e.g., "a father" and "a friend" share the "a" prefix).

### Comprehensive category taxonomy

Aim for **comprehensive coverage** — include many items per category even if some seem unlikely. The logprob approach is cheap per item (single token lookup with KV cache), so we can afford breadth.

```
IDENTITY CATEGORIES:

family_role:
  father, mother, son, daughter, husband, wife, brother, sister,
  parent, child, grandparent, grandmother, grandfather, uncle, aunt

occupation:
  teacher, doctor, engineer, student, worker, professor, lawyer,
  farmer, soldier, artist, scientist, businessman, nurse, driver,
  cook, writer, musician, politician, priest, craftsman

personality_trait:
  happy, sad, strong, kind, honest, brave, intelligent, creative,
  patient, generous, proud, humble, curious, careful, hardworking,
  friendly, quiet, independent, loyal, optimistic

religious:
  Christian, Muslim, Catholic, Orthodox, believer, faithful,
  Buddhist, Hindu, Jewish, religious, spiritual, devout

national_ethnic:
  [per-language: demonym for the language's primary country]
  e.g., English→"American"/"British", German→"German", Chinese→"Chinese"

relational:
  friend, neighbor, citizen, member, colleague, companion,
  teammate, partner, volunteer, leader, follower

abstract_universal:
  human, person, man, woman, alive, free, mortal, individual,
  being, soul, nobody, somebody, part of something

material_physical:
  rich, poor, young, old, healthy, sick, tall, short,
  tired, hungry, alone, lucky, unlucky
```

Each of these must be translated per language. Translation approach:
- Use the existing translation infrastructure from Track 1 (Gemini Flash extraction) or batch-translate via an LLM
- Manual review for high-value items (family_role, occupation, personality_trait)
- Store as `data/i_am_categories/{lang}.json`

### Normalization

After extracting P(item | prompt) for all items:
1. **Within-category normalization**: P(father | family_role) = P(father) / Σ_{i ∈ family_role} P(i)
2. **Cross-category distribution**: P(family_role) = Σ_{i ∈ family_role} P(i) / Σ_{all items} P(all)
3. The cross-category distribution is the primary cultural signal (e.g., 40% family vs 25% occupation vs 15% trait...)

### Fallback: Temperature Sampling

> **NOTE**: If logprob extraction does not yield clear cultural differentiation (e.g., all models produce similar distributions, or the probability mass is too spread out), fall back to **temperature sampling**:
> - Generate N=200 completions per (model, language) at temperature 0.7-1.0
> - Truncate at first sentence boundary
> - Classify each completion into the category taxonomy (via LLM classifier or embedding similarity)
> - Build empirical distributions
>
> This is slower and noisier but captures the full generative distribution rather than only pre-specified items. The two methods should be compared if both are run.

## Human Comparison: WVS/EVS Proxy

No open cross-cultural dataset exists for the "I am" completion task specifically. Instead, use WVS/EVS data as **proxy ground truth** for the individualism-collectivism axis:

### Relevant WVS/EVS items

Map WVS questions to the "I am" category dimensions:

| WVS item | Maps to | Dimension |
|----------|---------|-----------|
| Importance of family (V4) | family_role weight | Traditional |
| Importance of work (V5) | occupation weight | Secular |
| Importance of religion (V6) | religious weight | Traditional |
| Importance of friends (V7) | relational weight | Self-expression |
| Child qualities: independence (V12) | personality_trait weight | Secular |
| Child qualities: obedience (V16) | family_role weight | Traditional |
| National pride (V211) | national_ethnic weight | Traditional |

For each country/language, the WVS human distribution over these items gives an expected cultural profile. Compare against the model's cross-category distribution from "I am" completions.

**Metric**: Rank correlation between the model's category weights and the WVS-derived expected weights per language. This is not an exact comparison (different instruments) but tests whether the same cultural axes emerge.

### Inglehart-Welzel positioning

The cross-category distributions can also be projected onto Inglehart-Welzel axes:
- **Traditional–Secular**: (family_role + religious + national_ethnic) vs (occupation + personality_trait + abstract)
- **Survival–Self-expression**: (material_physical + occupation) vs (personality_trait + relational + abstract)

This produces a 2D cultural position per (model, language) that can be plotted alongside WVS country positions.

## Analysis Plan

### Primary comparisons

1. **Within-language, across-model**: For each language L, compare category distributions from HPLT_L vs EuroLLM_L vs Gemma3_27B_L vs Gemma3_4B_L vs Gemma3_1B_L
   - JSD between distribution pairs
   - Does model size or architecture matter more than language?

2. **Within-model, across-language**: For each multilingual model, compare category distributions across languages
   - Do languages cluster by cultural cluster?
   - UMAP of category distributions colored by cluster

3. **European vs expanded languages**: Do the 5 expanded languages (zho, jpn, ara, hin, tur) show stronger cultural differentiation than intra-European variation?

4. **Model size effect**: For Gemma 3 at 1B/4B/27B, does cultural differentiation increase with size? (Hypothesis: larger models have richer cultural representations)

5. **Monolingual vs multilingual**: Do HPLT monolingual models show stronger cultural signal than multilingual models prompted in the same language?

6. **Human alignment**: Rank correlation of model category profiles with WVS-derived profiles per country

### Visualizations

- Heatmap: models × languages, colored by dominant category
- UMAP/PCA of category distributions (one point per model-language pair)
- Inglehart-Welzel plot with model positions alongside WVS country positions
- Bar charts: category distribution per language for a single model (e.g., Gemma 3 27B)
- Model-size scaling plot: cultural differentiation metric vs parameter count

## Phased Implementation

### Phase 1: Base infrastructure & "I am" completions (no SAE)

**Goal**: Build the comparison framework and run all non-SAE models.

1. **Translate "I am" prompts** for all 27 languages
   - Store in `data/i_am_prompts.json` (lang → prompt string)
   - Include both "I am" and "I am a" variants where applicable

2. **Build category taxonomy** with per-language translations
   - Store in `data/i_am_categories/{lang}.json`
   - ~100-150 items per language across all categories
   - Translate via LLM, manually review family/occupation/trait categories

3. **Implement logprob extraction** for open-ended stems
   - Extend or adapt `eurollm/inference/extract_logprobs.py`
   - Input: prompt + list of candidate continuations
   - Output: P(continuation | prompt) for each continuation
   - Handle multi-token continuations via autoregressive logprob accumulation
   - KV-cache optimization for shared prefixes

4. **Run on all models**
   - 22 HPLT models × 22 European languages (= 22 runs, one lang per model)
   - EuroLLM-22B × 22 European languages (= 22 runs)
   - Gemma 3 27B × 27 languages (= 27 runs)
   - Gemma 3 4B × 27 languages (= 27 runs)
   - Gemma 3 1B × 27 languages (= 27 runs)
   - SLURM job scripts for cluster execution

5. **Build analysis pipeline**
   - Category distribution computation & normalization
   - JSD matrices, UMAP, Inglehart-Welzel projection
   - WVS/EVS human comparison

### Phase 2: Analysis & iteration

**Goal**: Analyze Phase 1 results, identify whether the signal is there.

- If logprob extraction gives flat/indistinguishable distributions → try temperature sampling fallback
- If European languages are too similar → the expanded languages become the primary analysis
- If model size dominates cultural signal → focus on same-size comparisons
- Refine category taxonomy based on which items carry the most cross-cultural variance

### Phase 3: SAE culture vector extraction & steering

**Goal**: Use Gemma Scope 2 SAEs to extract culture vectors and add SAE-steered Gemma 3 27B to the comparison.

> This phase builds on the RESEARCH.md background and the method from Deng et al. (ACL 2025) and the CAA cultural steering work. It requires the Phase 1 infrastructure to be in place as the evaluation harness.

1. **Load Gemma 3 27B + Gemma Scope 2 residual SAEs**
   - Focus on layers at 50-85% depth (where cultural localization concentrates per prior work)
   - Start with `resid_post` SAEs, width 64k or 256k

2. **Identify language-specific features**
   - Replicate Deng et al. monolinguality metric ν_s^L on parallel text (Flores or similar)
   - Feed same content in all 27 languages, compute per-feature activation stats
   - Threshold for language-specific features

3. **Identify culture-specific features**
   - Feed culturally-loaded text (WVS completions + Phase 1 "I am" completions from HPLT models as proxy for "natural culture")
   - Compute monoculturality metric ν_s^C grouped by cultural cluster
   - Features with high ν for any culture cluster → culture features

4. **Build culture vectors via subtraction**
   - culture_only(C) = mean activations for culture C − projection onto language features
   - Validate: culture vectors from different languages for same cultural cluster should be similar

5. **Steer Gemma 3 27B and evaluate**
   - For each cultural cluster C and language L:
     - h_steered = h_base + α·culture_vector(C)
     - Run "I am" completion with steering
     - Compare against HPLT model for cluster C's languages
   - **Key test**: steer toward culture C while prompting in language L' ≠ L
     - e.g., Bulgarian culture vector + English prompt → do completions shift toward collectivist?

6. **Dimensionality reduction of culture vectors**
   - PCA/UMAP on culture-only vectors for all clusters
   - Compare against Inglehart-Welzel map
   - Compare against Phase 1 UMAP of category distributions

## Output Data Format

### Per-run output: `results/{model_type}_{lang}.parquet`

| Column | Type | Description |
|--------|------|-------------|
| prompt_variant | str | "i_am" or "i_am_a" |
| category | str | e.g., "family_role", "occupation" |
| item | str | e.g., "father", "teacher" |
| item_tokens | str | Tokenized form (for reproducibility) |
| n_tokens | int | Number of tokens in continuation |
| logprob_sum | float | Σ log P(t_i | prefix) |
| prob | float | exp(logprob_sum) = joint probability |

### Aggregated output: `results/category_distributions.parquet`

| Column | Type | Description |
|--------|------|-------------|
| model_type | str | e.g., "hplt2c", "gemma3_27b_pt" |
| lang | str | Language code |
| category | str | Category name |
| prob_category | float | Normalized probability mass for this category |
| top_item | str | Highest-probability item in category |
| entropy | float | Entropy of within-category distribution |

## Hardware Requirements

- **Gemma 3 27B**: ~54GB bf16, ~27GB int8 → A100 80GB or H100
- **Gemma 3 4B**: ~8GB bf16 → any modern GPU
- **Gemma 3 1B**: ~2GB bf16 → any GPU
- **HPLT 2.15B**: ~5GB bf16 → any GPU
- **EuroLLM-22B**: ~45GB bf16 → A100 80GB or H100
- **SAE inference**: minimal overhead (single matmul per layer per token)
- **Per-model run time**: ~100-150 category items × 27 languages × ~3 tokens avg = ~12K forward passes (incremental with KV cache) — minutes per model on GPU

## Key Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Logprob distributions too flat / uninformative | Fall back to temperature sampling (Phase 2) |
| European languages too culturally similar | Expanded language set provides stronger contrast |
| Model size confounds cultural signal | Explicit size comparison (1B/4B/27B) |
| Category taxonomy misses important dimensions | Start broad, refine based on variance analysis |
| "I am" stem too short / ambiguous | Reserve infrastructure for additional stems |
| SAE features don't cleanly separate culture/language | This is a genuine open question — negative result is publishable |
| No human TST data for direct comparison | WVS proxy comparison on matched dimensions |
