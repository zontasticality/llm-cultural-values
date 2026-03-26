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

## Comprehensive SAE Literature Review (March 2026)

This section synthesizes the current state of SAE research, drawing on papers and technical reports from 2024-2026, to inform whether an SAE-based approach to isolating cultural features is feasible.

### 1. What Do SAE Features Actually Capture?

#### Types of features found

SAEs decompose model activations into sparse, high-dimensional representations where individual latent dimensions often have interpretable activation patterns. Anthropic's "Scaling Monosemanticity" paper (Templeton et al., 2024) trained SAEs at three sizes (1M, 4M, 34M features) on the **middle residual stream** of Claude 3 Sonnet. They found features ranging from highly concrete (Golden Gate Bridge, specific programming error types) to highly abstract (sycophantic praise, inner conflict, logical inconsistency). Many features are **multilingual and multimodal** — a single feature for a concept activates across languages and modalities.

Key examples of abstract features found in Claude 3 Sonnet:
- A "sycophantic praise" feature that activates on inputs containing compliments and, when artificially activated, causes the model to respond with flowery deception
- Features for "inner conflict" surrounded by related features for relationship breakups, conflicting allegiances, and the phrase "catch-22"
- Safety-relevant features for deception, bias, and dangerous content

**Citation**: [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/) (Templeton et al., 2024)

#### Human interpretability rates

Anthropic reported **~70% of features as genuinely interpretable** by human evaluators — substantially better than neuron-level interpretation. However, this headline figure requires important caveats.

OpenAI's parallel work on GPT-4 SAEs (Gao et al., 2024) trained a 16M-feature SAE on GPT-4 activations. With their TopK architecture plus AuxK loss, they achieved only **7% dead features** at the 16M scale, compared to Anthropic's **~65% dead features** in their 34M SAE (only 12M of 34M features were alive). Dead features increase dramatically with SAE size under standard training.

**Citation**: [Scaling and evaluating sparse autoencoders](https://arxiv.org/abs/2406.04093) (Gao et al., 2024)

The Gemma Scope 2 technical paper (McDougall et al., 2025) evaluated interpretability using automated classification and found that **lower-frequency features tend to be more interpretable** (see Figure 3 in the paper), while higher-frequency features are slightly less interpretable. This is consistent across model sizes and depths.

**Citation**: [Gemma Scope 2 Technical Paper](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/Gemma_Scope_2_Technical_Paper.pdf) (McDougall et al., 2025)

#### Features vs circuits

SAE features are more like **individual directions in activation space** than circuits. Transcoders (Section 1.5 below) bridge this gap by decomposing MLP computations into interpretable input-output feature relationships, enabling circuit-level analysis. SAEs on the residual stream give you individual concept detectors; transcoders give you computational steps; crosscoders and cross-layer transcoders (CLTs) give you features that span multiple layers.

Anthropic's circuit tracing work (Lindsey et al., 2025) built "replacement models" using cross-layer transcoders that can substitute for a model's MLPs while matching outputs in ~50% of cases. Attribution graphs then trace feature-to-feature interactions across layers, revealing computational circuits. This approach is substantially more powerful than per-layer SAE analysis for understanding how computations unfold.

**Citation**: [Circuit Tracing: Revealing Computational Graphs](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) (Lindsey et al., 2025)

#### Known failure modes

**Feature absorption**: A parent feature only partially splits — specific instances of a general feature get absorbed by more specialized latents, leaving "holes" in the representation (e.g., a feature that activates on all tokens starting with "E" except the word "Elephant"). This is caused by sparsity regularization.

**Feature composition/hedging**: When features naturally co-occur, the sparsity objective incentivizes learning single latents that capture specific combinations rather than representing independent features separately. Feature hedging (Chanin et al., 2025) demonstrates that narrower SAEs suffer more from this — they merge components of correlated features, destroying monosemanticity. **This is particularly relevant for cultural features**, which are likely correlated with language features.

**Feature splitting**: With large capacity, sparsity penalties push the SAE to replace general concepts with narrowly specialized features without retaining the high-level category.

**Spurious features**: A February 2026 paper (Korznikov et al.) introduced "sanity checks" showing that random baselines match fully-trained SAEs on interpretability (0.87 vs 0.90), sparse probing (0.69 vs 0.72), and causal editing (0.73 vs 0.72). On synthetic setups with known ground-truth features, SAEs recovered **only 9% of true features** despite achieving 71% explained variance.

**Feature sensitivity**: Tian (2025) showed that **many interpretable features have poor sensitivity** — they activate on their concept but also fail to activate on many instances of that concept. Wider SAEs consistently show worse sensitivity.

**False discovery rate**: Approximately **25% of highly active SAE features** from a single layer genuinely encode task-relevant information, while 75% represent noise or spurious correlations (Model-X knockoffs analysis, 2025).

**Citations**:
- [Feature Hedging](https://arxiv.org/abs/2505.11756) (Chanin et al., 2025)
- [Sanity Checks for SAEs](https://arxiv.org/abs/2602.14111) (Korznikov et al., 2026)
- [Measuring SAE Feature Sensitivity](https://arxiv.org/abs/2509.23717) (Tian, 2025)
- [Model-X Knockoffs for FDR Control](https://arxiv.org/abs/2511.11711) (2025)

#### The DeepMind negative results

Google DeepMind's mech interp team published a critical finding (March 2025): SAEs **underperform linear probes** on out-of-distribution detection of harmful user intent. 1-sparse SAE probes failed to fit training data; k-sparse probes (k~20) could fit training data but showed "distinctly worse performance on the OOD set." Linear probes achieved "nearly perfect" performance including on OOD data.

As a result, DeepMind **deprioritized fundamental SAE research**, though they continue to use SAEs as one tool among others. Their conclusion: SAEs are useful for **discovering unknown concepts** (exploratory analysis) but not for **acting on known concepts** (where simpler baselines win).

This distinction is captured well by a June 2025 paper: "Use Sparse Autoencoders to Discover Unknown Concepts, Not to Act on Known Concepts." SAEs excel at hypothesis generation (e.g., discovering that LLMs represent "words rhyming with 'it'" during poetry completion) but underperform prompting and fine-tuning for steering toward known concepts.

**Citations**:
- [Negative Results for SAEs on Downstream Tasks](https://www.lesswrong.com/posts/4uXCAJNuPKtKBsi28/negative-results-for-saes-on-downstream-tasks) (DeepMind, 2025)
- [Use SAEs to Discover Unknown Concepts](https://arxiv.org/abs/2506.23845) (2025)

#### The 2026 field status

MIT Technology Review named mechanistic interpretability a "breakthrough technology for 2026." However, the field is fundamentally split. Neel Nanda publicly updated his views in September 2025: "I've become more pessimistic about the high-risk, high-reward approach... The most ambitious vision of mechanistic interpretability I once dreamed of is probably dead. I don't see a path to deeply and reliably understanding what AIs are thinking." Computational costs remain prohibitive: Gemma 2 SAEs alone required 20 petabytes of storage and GPT-3-level compute. Circuit analysis of even short prompts takes hours of human effort.

**Implication for our project**: SAEs are best for **exploratory discovery** of cultural features, not for reliable downstream steering. We should plan for the possibility that SAE-identified features do not cleanly separate culture from language, and that simpler methods (CAA, linear probes) may outperform SAE-based steering.

**Citation**: [Mechanistic interpretability: 2026 status report](https://gist.github.com/bigsnarfdude/629f19f635981999c51a8bd44c6e2a54)

---

### 2. SAE Features Across Layers

#### The layer hierarchy

There is strong evidence for a progression from concrete/structural features in early layers to abstract/semantic features in later layers:

- **LinguaLens** (EMNLP 2025) identified four stages in Llama-3.1-8B:
  - Stage I (layers 0-2): primarily morphology and basic syntax
  - Stage II (layers 3-8): complex syntactic phenomena, early pragmatics
  - Stage III (layers 9-16): discourse markers, pragmatic functions
  - Stage IV (layers 17-31): deep semantics and rhetorical structures
  - Cross-lingual overlap is **higher in the first 16 layers** and lower in later layers, suggesting upper layers process language less universally

- **Evolution of SAE Features** (Balcells et al., 2024) on GPT-2-small found:
  - >50% of features appear or disappear at each layer, increasing to ~80% in later layers
  - Features specialize: a general "evidence" detector at layer 2 becomes a "court evidence" detector at layer 4
  - Some features act as AND/OR gates combining upstream features (e.g., "Baltic regions" AND "Estonia-related" → "mentions of Estonia")

- **"Dense SAE Latents Are Features"** found a shift from structural features in early layers → semantic features in middle layers → output-oriented signals in the last layers

- The Gemma Scope 2 paper notes that multi-layer training "should allocate activations non-uniformly across depth (empirically we see allocations rise through the network and drop near the end)"

**Citations**:
- [LinguaLens](https://arxiv.org/abs/2502.20344) (EMNLP 2025)
- [Evolution of SAE Features Across Layers](https://arxiv.org/abs/2410.08869) (Balcells et al., 2024)
- [Dense SAE Latents Are Features](https://arxiv.org/abs/2506.15679) (2025)

#### Where do cultural/value/identity features appear?

No one has directly studied where SAE features for cultural values appear. However, converging evidence points to **mid-to-late layers (50-85% depth)**:

- **Cultural steering** (Veselovsky et al., 2025): Activation patching on Gemma2-9b-it showed cultural localization spiking at layers 23 and 30 (out of 42 total, i.e., 55-71% depth). CAA steering was most effective at layers 15-30.

- **Language-specific features** (Deng et al., 2025): Language-specific SAE features show increasing activation values at deeper layers (see Figures 3-4 in the paper), with strongest effects from layer ~10 onward in the 26-layer Gemma 2 2B.

- **Causal language control** (2025): Language steering via SAE features on Gemma-2-2B and 9B was most effective in layers 29-36 (out of 42 in the 9B model, i.e., 69-86% depth).

- **Moral indifference** (March 2026): SAEs on Qwen3-8B were used to isolate mono-semantic moral features and reconstruct moral topological relationships, suggesting value-laden features do exist but require targeted extraction.

**Practical recommendation**: For cultural feature identification on Gemma 3 27B (62 layers), focus on layers 31-53 (50-85% depth). Start with the 4 layers at expanded hyperparameter sweeps: layers 16, 31, 40, 53. Layer 31 (50% depth) is where abstract semantics begin; layer 53 (85% depth) is where output-oriented features dominate.

**Citations**:
- [Localized Cultural Knowledge](https://arxiv.org/abs/2504.10191) (Veselovsky et al., 2025)
- [Causal Language Control](https://arxiv.org/abs/2507.13410) (2025)
- [Mechanistic Origin of Moral Indifference](https://arxiv.org/abs/2603.15615) (2026)

---

### 3. SAE Features Across Model Sizes

#### General trends

Larger models have more features, and those features tend to be more abstract. Anthropic's scaling monosemanticity work found features for "famous people, countries and cities, code type signatures" — concepts that become more diverse and abstract at larger scales.

OpenAI's scaling laws (Gao et al., 2024) showed that larger language models require proportionally more latents for equivalent reconstruction quality. The scaling exponent is substantially worse across model sizes than within a model — **feature proliferation accompanies model capability**.

#### The 1B question

For very small models (1B parameters), the question is whether they even have separable cultural features. Evidence is mixed:

- **Cross-linguistic disparities** (2025) on Gemma-2-2B found systematic activation gaps between high-resource and low-resource languages (up to 26.27% lower activations for low-resource languages), suggesting the 2B model has some language-specific structure but it is weaker than in larger models.

- **LinguaLens** on Llama-3.1-8B found clear linguistic feature hierarchies. At 8B, all four stages (morphology → syntax → semantics → pragmatics) are present.

- The Gemma Scope 2 paper shows sparsity-fidelity tradeoff curves for Gemma 3 1B (Figure 7). Autointerp scores are consistently 0.7-0.9 across L0 values and widths, suggesting 1B models do have interpretable features. However, interpretability scores are lower than for larger models at the same layer positions.

**Practical expectation**: Gemma 3 1B likely has language-specific features but may lack separable cultural features (which are more abstract). Cultural differentiation probably requires at least 4B parameters. The 1B model is best used as a **negative control** — if it shows cultural features comparable to the 27B model, something is wrong with the methodology.

**Citation**: [Uncovering Cross-Linguistic Disparities using SAEs](https://arxiv.org/abs/2507.18918) (2025)

#### Gemma Scope 2 across sizes

Gemma Scope 2 covers 270M, 1B, 4B, 12B, and 27B. The paper does not directly compare feature quality across sizes, but the infrastructure enables such comparison. The layer percentiles (25%, 50%, 65%, 85%) are calibrated per model size, making cross-size comparisons at equivalent relative depths straightforward.

---

### 4. SAE Width (Dictionary Size) Effects

#### Available configurations

From the Gemma Scope 2 technical paper (Table 1), the available widths and their deployment vary by artifact type:

| SAE Type | Widths | L0 Targets | Notes |
|----------|--------|------------|-------|
| Residual SAE (all layers) | {16k, 256k} | {10, 100} | Full layer coverage |
| Residual SAE (4 depth layers) | {16k, 64k, 256k, 1m} | {10, 50, 150} | Expanded sweep |
| Transcoder (all layers) | {16k, 256k} | {10, 100} | Full layer coverage |
| Transcoder (4 depth layers) | {16k, 64k, 256k} | {10, 50, 150} | Expanded sweep |
| Crosscoder (4 depth layers) | {64k, 256k, 512k, 1m} | {50, 150} | Multi-layer |
| CLT (all layers) | {256k, 512k} | {50, 150} | Cross-layer transcoders |

#### Width-sparsity-fidelity tradeoff

From the Gemma Scope 2 paper (Figure 7), for Gemma 3 1B residual SAEs:
- Higher L0s and wider SAEs lead to better reconstruction (lower FVU, lower delta LM loss)
- **Autointerp scores are relatively flat across widths and L0** (0.7-0.9 range), meaning wider SAEs do not significantly hurt interpretability
- But wider SAEs have **worse feature sensitivity** (Tian, 2025) — features fire on concept instances but also fail to fire on many valid instances

OpenAI found clean **power-law scaling**: reconstruction quality improves predictably with width, with TopK SAEs empirically outperforming ReLU SAEs on the sparsity-reconstruction frontier.

#### What width for abstract cultural features?

The Gemma Scope 2 announcement recommends:
- **262k (256k) width with medium L0 (30-60)** for most use cases
- **16k width** for circuit-style analysis where computational cost matters
- For **behavioral steering**: residual stream models at 262k width, medium L0

For cultural features specifically:
- **Start with 64k-256k**. Cultural concepts are relatively abstract (similar to "sycophancy" or "inner conflict" in the Anthropic work), so they likely require moderate width to disentangle from related concepts.
- **Avoid 1M width** unless exploring fine-grained cultural distinctions — the sensitivity problems at large widths would create more noise than signal for abstract concepts.
- **16k may be too narrow** for disentangling culture from language, since feature hedging would merge these correlated features.

#### L0 (sparsity) and its interaction with width

L0 measures the average number of active features per token. Higher L0 = less sparse = better reconstruction but harder interpretation. SAEBench (Karvonen et al., 2025) found that optimal sparsity is highly task-dependent, with best performance for concept detection typically in the L0 20-200 range.

The Gemma Scope 2 paper uses a **quadratic L0 penalty** to target specific sparsity levels. Their three tiers:
- Small: L0 = 10-20 (very sparse, potentially missing cultural features)
- Medium: L0 = 30-60 (recommended starting point)
- Large: L0 = 90-150 (better reconstruction, more feature co-activation)

For cultural analysis, **medium L0 (30-60) at 256k width** is the recommended starting point. This gives enough capacity to separate culture from language without the noise problems of very wide SAEs.

**Citations**:
- [SAEBench](https://arxiv.org/abs/2503.09532) (Karvonen et al., 2025)
- [The "Sparsity vs Reconstruction Tradeoff" Illusion](https://www.lesswrong.com/posts/RwWrkGnncSCqryrSZ/the-sparsity-vs-reconstruction-tradeoff-illusion)

---

### 5. Practical Considerations for Cultural Feature Identification

#### Deng et al. (ACL 2025) — detailed methodology

**Paper**: "Unveiling Language-Specific Features in LLMs via Sparse Autoencoders"
**Code**: https://github.com/Aatrox103/multilingual-llm-features

**Models and SAEs used**:
- Gemma 2 2B and Gemma 2 9B with **Gemma Scope SAEs** (residual stream at each layer)
- Llama-3.1-8B with **Llama Scope SAEs**
- Selected the SAE with the second-smallest L0 value for each layer (they did not specify exact width, but Gemma Scope 1 standard widths were 16k and 64k for residual stream)

**Monolinguality metric** (Equation 3 in the paper):
```
mu_s^L = (1/|D_L|) * sum_{x in D_L} f_s(x)        # mean activation of feature s on language L
gamma_s^L = (1/|D\{D_L}|) * sum_{D_l in D\{D_L}} (1/|D_l|) * sum_{x in D_l} f_s(x)   # mean activation on all other languages
nu_s^L = mu_s^L - gamma_s^L                          # monolinguality score
```
Higher nu means stronger language specificity. They rank features by nu for each language.

**Languages tested**: 10 languages from Flores-200: English (en), Spanish (es), French (fr), Japanese (ja), Korean (ko), Portuguese (pt), Thai (th), Vietnamese (vi), Chinese (zh), Arabic (ar).

**Features per language**: The paper focuses on **top-4 features** per language. In most languages, the top-1 feature has a significantly larger nu than others. For Layer 20 of Gemma 2 2B, the top-1 feature nu values range from ~20 (Japanese, Korean) to ~70 (Spanish, French) — these are very strong monolinguality signals. A random feature (feature 2000) has nu close to zero.

**Key quantitative results**:
- Ablating the top-2 language-specific features increases CE loss by 3-12 nats for the target language (varies by layer), while CE loss for other languages changes by <1 nat
- **Synergistic effect**: Ablating top-2 features together produces greater CE loss increase than the sum of individual ablations — observed specifically in layers 7, 9, 10, 11, 14, and 15 for French features
- Language features extend beyond tokens: adding a Spanish prefix to French/Korean nouns **increases Spanish feature activation** for those nouns, suggesting features capture linguistic context, not just token identity

**Steering results** (Table 1 in the paper): SAE-enhanced steering vectors (applied to 3 consecutive layers) achieved:
- Spanish: 95.8/4.2 success rate/CE loss (vs 92.1/4.7 for baseline SV)
- French: 96.2/3.4 (vs 94.6/2.9)
- Thai: 90.7/5.0 (vs 85.7/5.3)
- Cross all 9 non-English languages and 3 models, SAE-enhanced method won on both metrics in most cases

**Limitations the authors note**: Does not cover low-resource languages; SAEs were not trained on curated multilingual data; some cases where baseline SV outperforms SAE method.

#### Causal Language Control via SAE Steering (2025)

**Paper**: "Causal Language Control in Multilingual Transformers via Sparse Feature Steering"
Used **Gemma Scope 16k SAEs** on Gemma-2-2B and Gemma-2-9B residual streams.

Identified language-specific features via contrastive analysis on 1,000 parallel sentence pairs per language from Tatoeba Project. Selected **top-3 features** per language based on largest absolute activation differences. With just 3 features per language:
- Chinese: 97.8% success (FastText accuracy)
- Japanese: 93.8%
- Spanish: 88.8%
- French: 85.2%

Most effective layers: **29-36 in the 42-layer 9B model** (69-86% depth). Layer 29 showed a sharp increase in effectiveness; earlier layers had minimal effect. Attention Head 12 at Layer 29 was disproportionately associated with language-sensitive features across all four languages.

**Key implication for cultural features**: If just 3 SAE features per language can achieve 85-98% language steering with 16k-width SAEs, the language signal is very concentrated. This makes it plausible that cultural features exist separately — but also raises the risk that cultural features are **entangled with** these few dominant language features.

**Citation**: [Causal Language Control](https://arxiv.org/abs/2507.13410) (2025)

#### Cross-linguistic SAE disparities

On Gemma-2-2B with Gemma Scope 16k SAEs, systematic activation gaps exist across languages:
- Medium-to-low resource languages receive up to **26.27% lower activations** in early layers
- Gap persists at **19.89% in deeper layers**
- Strong correlation between activation differences and benchmark performance (r = -0.95)
- Fine-tuning dramatically helps: Malayalam showed 87.69% activation gain

This means for our 22 European languages, **activation levels will vary by language**, which must be controlled for when computing culture-specific features.

**Citation**: [Uncovering Cross-Linguistic Disparities](https://arxiv.org/abs/2507.18918) (2025)

#### Would SAEs give finer-grained cultural features than CAA?

CAA (Contrastive Activation Addition) computes a single dense steering vector per concept. SAEs decompose this into potentially hundreds of sparse features. In principle, SAEs should give finer granularity — you could identify separate features for "collectivism," "respect for elders," "religious authority," etc., rather than a single "culture X" direction.

However, the DeepMind negative results and SAEBench data suggest that in practice, **CAA may be more reliable for steering** while SAEs are better for **understanding what the cultural direction contains**. The optimal approach may be hybrid: use SAEs to discover and interpret cultural features, then use CAA or FGAA (Feature-Guided Activation Additions) for actual steering.

FGAA (Kharlapenko et al., 2025) combines both: it uses SAE features to construct precise, interpretable steering vectors via optimization in the SAE latent space. It achieved 2-3x improvement over raw SAE steering and became competitive with supervised LoRA methods.

**Citations**:
- [Feature Guided Activation Additions](https://arxiv.org/abs/2501.09929) (2025)
- [SAEs Are Good for Steering — If You Select the Right Features](https://arxiv.org/abs/2505.20063) (2025)

#### Has anyone used SAEs for cross-cultural feature identification?

**No one has directly used SAEs for cross-cultural (as opposed to cross-lingual) feature identification.** The closest work is:

1. **Deng et al.** (2025) — language-specific features only, no cultural dimension
2. **Veselovsky et al.** (2025) — cultural steering via CAA, not SAEs
3. **LinguaLens** (2025) — linguistic (not cultural) features, including pragmatics like politeness (FIC score = 46.9)
4. **Moral indifference paper** (March 2026) — used SAEs on Qwen3-8B to isolate moral features, found LLMs exhibit "categorical indifference" (fail to separate opposed moral categories) and "gradient indifference" (fail to preserve intensity gradients). Targeted SAE feature reconstruction achieved 75% pairwise win-rate on the Flames benchmark.

This means our project would be the **first to attempt SAE-based cross-cultural feature identification**, making it novel regardless of whether the result is positive or negative.

---

### 6. Transcoders, Crosscoders, and Cross-Layer Transcoders

These are available in Gemma Scope 2 and may be more useful than standard residual SAEs for cultural feature work.

#### Transcoders

Unlike SAEs (which reconstruct activations at a single point), transcoders learn a sparse mapping from MLP input to MLP output. This makes them particularly useful for **circuit analysis**: freezing attention layers makes all connections between transcoder features linear, enabling clean upstream/downstream attribution.

Dunefsky et al. (2024, NeurIPS) showed transcoder features are **significantly more interpretable than SAE features** on detection and fuzzing metrics. Skip transcoders add an affine skip connection (y_skip = W_dec f + b_dec + W_skip x), which captures the linear component of MLP computation and was "strictly beneficial" per the Gemma Scope 2 team.

**Citation**: [Transcoders Find Interpretable LLM Feature Circuits](https://arxiv.org/abs/2406.11944) (Dunefsky et al., 2024)

#### Crosscoders (weakly causal)

Crosscoders (Lindsey et al., 2024) read from and write to multiple layers simultaneously. They resolve **cross-layer superposition** — when a single feature is distributed across latents in several layers. Benefits:
- Track persistent features through the residual stream
- Remove "duplicate features" from circuit analysis
- Enable **model diffing** (shared features across base vs. instruct, or across different model sizes)

Gemma Scope 2 provides crosscoders trained on 4 concatenated layers at 25/50/65/85% depth, with widths {64k, 256k, 512k, 1m} and L0 {50, 150}.

**Potential for cultural work**: Crosscoders could identify features that persist from mid-layers (where cultural knowledge consolidates) through to output layers. A cultural feature that exists at layer 31 but gets overwritten by layer 53 would be invisible to per-layer SAEs but visible to crosscoders.

**Citation**: [Sparse Crosscoders for Cross-Layer Features](https://transformer-circuits.pub/2024/crosscoders/index.html) (Lindsey et al., 2024)

#### Cross-layer transcoders (CLTs)

CLTs generalize transcoders to multiple layers: each feature reads from the residual stream at one layer and contributes to MLP outputs at all subsequent layers. Anthropic's circuit tracing work showed CLTs can generate attribution graphs with **higher sparsity** (fewer nodes/edges needed for the same total influence) than single-layer transcoders. Affine skip connections further improve both transcoders and CLTs.

In Gemma Scope 2, CLTs are available at {256k, 512k} width with L0 {50, 150} on all layers for all model sizes.

**Potential for cultural work**: CLTs could reveal how cultural knowledge is computed — whether it emerges from a few key MLP computations at specific layers or is distributed across many layers. For steering, CLT features might provide more precise intervention points than residual stream features.

---

### 7. Feature Steering with SAEs

#### SAE steering vs CAA comparison

| Aspect | CAA | SAE Steering | FGAA (hybrid) |
|--------|-----|-------------|---------------|
| Interpretability | Low (dense vector) | High (individual features) | High |
| Theoretical optimality | Optimal under squared error for linear steering | Suboptimal for known concepts | Combines both |
| Practical performance | Dominant on MCQ and open-ended generation | 2-3x improvement with proper feature selection | Competitive with LoRA |
| Coherence at high scaling | Degrades quickly | Degrades at high clamp values | Better maintenance |
| Multi-layer robustness | Degrades with multiple layers | SAE-gated application is more robust | Designed for multi-layer |

**Key insight**: "SAEs Are Good for Steering — If You Select the Right Features" (2025) discovered that **input features and output features are distinct**:
- **Input features** (early layers, high "input score"): activate on concept-relevant tokens, but do NOT effectively steer generation
- **Output features** (later layers, high "output score"): minimal activation overlap with input patterns, but powerfully shape generation
- These roles "rarely co-occur" — high input and output scores showed **near-zero correlation**
- Using output score filtering improved steering from 0.191 to 0.546 on AxBench (Gemma-2 2B/9B)

**Practical recommendation for cultural steering**: Focus on features at **66-100% depth** for steering (where output scores are high). Features at 50-65% depth may be useful for detecting/classifying cultural content but not for steering it.

#### Reliability of SAE steering

Steering reliability depends heavily on feature selection and clamp value tuning:
- Higher clamp values increase steering strength but model performance drops quickly
- Multi-concept steering shows widespread **feature interference** that correlational analyses miss
- The clamp value is a hyperparameter requiring per-feature tuning

#### Practical workflow

1. **Identify candidate features**: Use Deng et al.'s monolinguality metric (adapted for monoculturality) to rank features by cultural specificity
2. **Validate via ablation**: Directional ablation of candidate features — does ablating "collectivism" features reduce collectivist completions without affecting other dimensions?
3. **Select steering features**: Focus on features with high output scores at later layers (66-100% depth)
4. **Steer**: Apply via clamping/scaling at specific layers, tune clamp values on validation set
5. **Measure**: Use behavioral completion pipeline (Phase 2) to quantify cultural shift
6. **Compare against CAA baseline**: Run the same evaluation with simple CAA steering vectors

---

### 8. Gemma Scope 2 Specifics

#### Full configuration (from Technical Paper Table 1)

For Gemma 3 27B (both PT and IT):

| Type | Layers | Widths | L0 |
|------|--------|--------|----|
| SAE (all layers) | All 62 | {16k, 256k} | {10, 100} |
| SAE (4 depth layers) | {16, 31, 40, 53} | {16k, 64k, 256k, 1m} | {10, 50, 150} |
| Transcoder (all layers) | All | {16k, 256k} | {10, 100} |
| Transcoder (4 depth layers) | {16, 31, 40, 53} | {16k, 64k, 256k} | {10, 50, 150} |
| Crosscoder | {16, 31, 40, 53} | {64k, 256k, 512k, 1m} | {50, 150} |
| CLT | All | {256k, 512k} | {50, 150} |

All SAEs and transcoders are **JumpReLU** with quadratic L0 penalty and direct frequency penalization. Training used end-to-end finetuning with KL divergence regularization for select models.

IT variants are **initialized from PT SAEs** and finetuned on model rollout data (from OpenAssistant and LMSYS-Chat-1M prompts), not trained from scratch.

**Three sites per layer**: attention head outputs (pre-W_O, pre-RMSNorm), MLP outputs (post-RMSNorm), post-MLP residual stream.

#### Training details

- Learning rate: 7e-5 with cosine schedule
- Batch size: 4,096
- Optimizer: Adam (beta_1=0, beta_2=0.999, epsilon=1e-8)
- Activation normalization: fixed scalar to unit mean squared norm
- Matryoshka loss: applied during training, features with smaller indices tend to be more important and fire more frequently (Figure 8)
- Multi-layer initialization: novel strategy that iterates through single-layer SAEs choosing non-redundant features, avoiding duplicate concepts across layers

#### Quality vs Gemma Scope 1

Gemma Scope 2 is built on Gemma 3 (vs Gemma 2 for Scope 1). Key improvements:
- **Matryoshka training** reduces feature absorption (absorption rate ~0.03 vs ~0.29 for BatchTopK at larger sizes)
- **JumpReLU with quadratic L0** provides more stable training around target sparsity
- **Direct frequency penalization** eliminates high-frequency latents
- **End-to-end finetuning** (KL-regularized) produces functionally important features
- **Skip transcoders and CLTs** enable circuit-level analysis not possible with Scope 1
- **IT SAE transfer** (PT→IT finetuning) preserves feature alignment between base and chat models

No published analyses using Gemma Scope 2 specifically for multilingual or cultural features exist yet.

#### Recommended configuration for cultural feature work

Based on all the evidence reviewed:

| Purpose | SAE Type | Width | L0 | Layers |
|---------|----------|-------|----|--------|
| Cultural feature discovery | Residual SAE | 256k | 50 | All layers (focused analysis at 31, 40, 53) |
| Language feature identification | Residual SAE | 64k-256k | 50 | All layers (strongest at 40-53) |
| Circuit analysis | Skip transcoder | 16k-256k | 50-100 | 31-53 |
| Cross-layer cultural persistence | Crosscoder | 256k-512k | 50 | {16, 31, 40, 53} |
| Steering | Residual SAE (output features) | 256k | 50 | 40-53 (66-85% depth) |

---

### 9. Summary Assessment: Is SAE-Based Cultural Feature Isolation Feasible?

#### Evidence FOR feasibility:
1. **Language features are real and concentrated**: Just 1-4 SAE features per language capture strong monolinguality signal (Deng et al.). This is well-established across multiple models and SAE suites.
2. **Cultural knowledge is localized**: CAA work shows cultural localization concentrates in specific layers (50-70% depth) and can be steered with language-agnostic vectors (Veselovsky et al.).
3. **Features are hierarchical and abstract**: SAEs do find highly abstract features (sycophancy, moral concepts) at the scale of Claude 3 Sonnet and GPT-4.
4. **Infrastructure exists**: Gemma Scope 2 provides comprehensive SAEs across all layers, widths, and model sizes, plus transcoders and crosscoders for deeper analysis.
5. **Novel contribution**: No one has attempted SAE-based cultural feature identification — the project is publishable regardless of outcome.

#### Evidence AGAINST feasibility:
1. **Culture and language are deeply correlated**: Feature hedging theory predicts that SAEs will merge correlated features. Cultural features may be inseparable from language features without very wide SAEs (which have worse sensitivity).
2. **SAEs underperform baselines on downstream tasks**: DeepMind's negative results and the sanity checks paper suggest SAE features may not capture task-relevant information better than simpler methods.
3. **Only 25% of active features are genuinely task-relevant**: The false discovery rate is high. Naive analysis of cultural SAE features will be dominated by spurious correlations.
4. **Moral indifference**: LLMs exhibit systematic failure to represent moral/value distinctions (March 2026), suggesting that even if SAE features for cultural values exist, they may be degenerate or compressed.
5. **Computational cost**: Running 256k-width SAEs on 62-layer 27B models across 27 languages is non-trivial.

#### Realistic expectations:
- **Language feature subtraction will work** (high confidence): The methodology for identifying and ablating language-specific features is well-established.
- **Some cultural features will be identifiable** (moderate confidence): Features for broad cultural concepts (e.g., religious vs. secular text) likely exist, especially in later layers of the 27B model.
- **Clean culture-language disentanglement is unlikely** (low confidence): The partition into pure language features, pure culture features, and shared features will be fuzzy. Expect significant overlap.
- **SAE steering for cultural transfer will be weak** (moderate concern): Based on the input/output feature distinction and DeepMind negative results, SAE-based steering may produce detectable but small effects. CAA or FGAA may be more effective for actual steering.
- **The 1B model will show minimal cultural features** (high confidence): 1B is likely too small for abstract cultural representations; it serves as a negative control.
- **A negative result is valuable and publishable**: The first systematic attempt to disentangle culture from language using SAEs, with well-characterized failure modes, would be a significant contribution.

---

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

**Key assumptions and caveats:**
- This assumes culture and language are approximately linearly separable in SAE feature space. They are likely correlated (cultural text IS linguistic data), so the partition may be fuzzy.
- The CAA cultural steering paper provides the strongest evidence for feasibility: culture vectors are language-agnostic and task-agnostic, suggesting some disentanglement already exists in the model.
- A negative result (features don't cleanly partition) is publishable and informative.

---

# Implementation Plan: Cultural Completion Comparison

## Motivation: Why Sampling Over Logprobs

Track 1 (WVS survey elicitation) demonstrated that logprob extraction over constrained digit vocabularies **does not capture within-question cultural signal** from base models. The headline finding: raw Pearson r = 0.67–0.72 between LLM and human expected values drops to **r ≈ 0.00** after z-scoring with human reference statistics. The entire apparent correlation was driven by cross-question scale structure (both humans and LLMs agree that 1–10 scales produce higher means than 1–3 scales), not genuine cultural differentiation.

The core problem is **entropy compression**: base models are high-entropy by nature, producing near-uniform distributions over response options. They capture the direction of broad consensus on binary questions (median JSD ≈ 0.23) but fail on Likert scales where cultural nuance lives.

This motivates a fundamentally different approach for Phases 1–2:
- **Temperature sampling** captures the model's full generative distribution, not just a pre-specified slice
- **LLM classification** maps free-form completions onto cultural dimensions without assuming a fixed taxonomy
- **Multiple prompt templates** control for grammatical confounds by averaging over prompt-specific variation
- **Empirical distributions** from N=200+ samples give confidence intervals, enabling statistical testing

## Overview

A systematic comparison framework that measures how different models complete culturally diagnostic sentence stems across languages, using temperature sampling and external LLM classification to build cultural profiles.

The framework serves two purposes:
1. **Standalone behavioral result**: Compare cultural expression across models, languages, and model sizes
2. **Phase 3 validation harness**: SAE-steered completions should produce behavioral changes consistent with the cultural profiles observed here

## Models

| Model | Size | Languages | Role |
|-------|------|-----------|------|
| HPLT monolingual (`HPLT/hplt2c_{lang}`) | 2.15B | 22 European | Monolingual cultural baseline |
| EuroLLM-22B (`utter-project/EuroLLM-22B-2512`) | 22B | 22 European | Multilingual European baseline |
| Gemma 3 12B PT (`google/gemma-3-12b-pt`) | 12B | 22 European + 5 expanded | Size comparison (pilot confirmed: 12B lacks cultural signal, 27B has it) |
| Gemma 3 27B PT (`google/gemma-3-27b-pt`) | 27B | 22 European + 5 expanded | Primary multilingual model |
| Gemma 3 27B PT + SAE steering | 27B | 22 European + 5 expanded | Culture-steered (Phase 3) |

**Explicit model-size comparison axis**: Gemma 3 12B vs 27B establishes the size threshold for cultural encoding. The pilot confirmed 12B does not reach significance on cross-cluster chi-square (p=0.11), while 27B does (p=1.2×10⁻⁶). HPLT 2.15B monolingual models provide a lower bound.

## Languages

### European (22) — available for all models
bul, ces, dan, deu, ell, eng, est, fin, fra, hrv, hun, ita, lit, lvs, nld, pol, por, ron, slk, slv, spa, swe

### Expanded (5) — Gemma 3 models only
zho (Chinese), jpn (Japanese), ara (Arabic), hin (Hindi), tur (Turkish)

Rationale: The strongest cultural contrasts in the cross-cultural psychology literature are Western vs East Asian / Middle Eastern. All 22 European languages are relatively culturally similar (Track 1 confirmed this — even with 149 ordinal questions, LLM distributions for European languages cluster tightly). Adding these 5 provides much stronger signal for the individualism-collectivism and traditional-secular axes.

### Cultural Clusters (Inglehart-Welzel classification)

```
European:
  Protestant Europe:  dan, fin, swe, nld, deu
  Catholic Europe:    fra, ita, spa, por, ces, hun, pol, slk, slv, hrv
  English-speaking:   eng
  Orthodox:           bul, ron, ell
  Baltic:             est, lit, lvs

Expanded:
  East Asian:         zho, jpn
  South Asian:        hin
  Middle Eastern:     ara, tur
```

## Prompt Design

### Multiple culturally diagnostic templates

Using a single stem ("I am") risks conflating grammatical structure with cultural signal. Instead, use 6–8 prompt templates that each target a different cultural dimension. Cross-prompt stability of the cultural signal serves as a built-in grammatical confound control.

```
PROMPT TEMPLATES:

1. self_concept:     "I am" / "I am a"
2. values:           "The most important thing in life is"
3. moral:            "A good person is someone who"
4. childrearing:     "Children should learn to"
5. family:           "In my family, the most important thing is"
6. success:          "Success means"
7. decision:         "When making a difficult decision, I"
8. belief:           "I believe that"
```

Each template is translated naturally per language (not word-for-word). For languages where the natural phrasing differs structurally, use the idiomatic form. Where a template has multiple natural renderings (e.g., "I am" vs "I am a" in article languages), include both variants.

### Grammatical control variants

For a subset of languages (eng, deu, fra, zho, tur — spanning major grammar families), create 2–3 rephrasings of each template that preserve semantic intent but alter syntax. If the cultural signal is stable across rephrasings, it's not a grammatical artifact.

Example for `values` template in English:
- "The most important thing in life is"
- "What matters most in life is"
- "Above all else, I value"

### Translation approach

- Batch-translate via LLM (Claude or Gemini), with instructions to produce natural sentence-start fragments
- Native speaker review for a core subset (eng, deu, fra, zho, jpn, ara, tur, pol, fin, ron — covering all cluster types)
- Store in `data/prompt_templates/{lang}.json`

## Method: Temperature Sampling + LLM Classification

### Sampling

For each (model, language, prompt_template):

1. Generate **N=200 completions** at temperature 0.8, top-p 0.95
2. Truncate at first sentence boundary or max 50 tokens (whichever comes first)
3. Store raw completions for reproducibility

With 6 prompt templates × 27 languages × 5 models = 810 configurations × 200 samples = 162,000 total completions. At ~0.5s per completion on GPU, this is ~22 GPU-hours for the full matrix (parallelizable across models).

### Classification

Use a strong instruction-tuned model (Gemma 3 27B IT or Claude API) to classify each completion along established cross-cultural psychology dimensions:

**Content category** (what the completion is about — one label per completion):
- `family_social`: family roles, relationships, social bonds
- `occupation_achievement`: work, career, accomplishment, education
- `personality_trait`: character qualities, self-description
- `material_physical`: wealth, health, material conditions
- `abstract_philosophical`: existential, philosophical, meaning-of-life
- `religious_spiritual`: faith, God, spiritual practice
- `national_civic`: nationality, patriotism, civic identity
- `other`: doesn't fit above categories

**Cultural dimension scores** (Likert 1–5 per completion):
- **Individualist ↔ Collectivist**: Does this completion emphasize the self or the group/family/community?
- **Traditional ↔ Secular-Rational**: Does this completion reflect traditional authority/religion/norms or secular/rational values?
- **Survival ↔ Self-Expression**: Does this completion emphasize material security/basic needs or self-actualization/quality of life?

These three dimensions correspond to established frameworks (Hofstede, Inglehart-Welzel) and can be directly compared to human WVS/EVS data.

### Classifier validation

Before deployment, validate the classifier on a hand-labeled subset:
- Label 200 completions manually across 5 languages
- Measure inter-rater agreement (Cohen's κ) and classifier-vs-human agreement
- Iterate on classifier prompt until κ > 0.7 on content categories and dimension scores

### Statistical framework

Temperature sampling gives empirical distributions with natural variance, enabling:
- **Bootstrap confidence intervals** (95% CI) on all metrics (mean dimension scores, category proportions)
- **Permutation tests** for cross-language differences (is Finnish really more secular than Romanian, or could the difference arise by chance from sampling noise?)
- **Effect sizes** (Cohen's d) for cultural dimension comparisons
- **Cross-prompt stability coefficient**: for each (model, language), compute pairwise correlation of cultural dimension scores across prompt templates. High stability → cultural signal. Low stability → grammatical/prompt-specific artifact.

## Human Comparison

### Direct comparison via Inglehart-Welzel dimensions

Track 1 already computes IW composite scores for human EVS data using established methodology (10 EVS questions, 5 per dimension). Use these directly:

- For each language/country, compute human IW position: (Traditional→Secular, Survival→Self-Expression)
- For each (model, language), compute LLM IW position from the classifier's cultural dimension scores:
  - Traditional→Secular score = mean of `traditional_secular` dimension across all completions
  - Survival→Self-Expression score = mean of `survival_selfexpr` dimension across all completions
- Rank correlation (Spearman ρ) across languages between human and LLM positions on each axis

### Known-group validation

The strongest test of whether the behavioral profiles capture culture:
- **Nordic vs Orthodox**: Finnish/Swedish/Danish completions should score higher on secular-rational and self-expression than Bulgarian/Romanian completions. Human EVS data shows this contrast clearly.
- **East Asian vs Western**: Chinese/Japanese completions should score higher on collectivism than English/German/Dutch.
- **Protestant vs Catholic Europe**: German/Dutch/Swedish should score slightly more secular than Polish/Italian/Spanish.

If the sampling approach can't detect the Nordic–Orthodox contrast (the largest effect in European IW data), the method lacks sensitivity and should be abandoned.

### Complementary validation via Track 1

Track 1's WVS logprob extraction works well for binary/categorical questions (median JSD ≈ 0.23). Use these as an independent check:
- Do models that show more secular behavioral profiles in completions also produce more secular WVS binary responses?
- Convergent validity across two independent methodologies strengthens both.

## Analysis Plan

### Primary comparisons

1. **Within-model, across-language**: For each model, compare cultural dimension score distributions across languages. Do languages cluster by IW cultural cluster? UMAP of cultural profiles colored by cluster.

2. **Known-group differentiation**: Can the method reliably detect the Nordic–Orthodox and East Asian–Western contrasts? Effect sizes (Cohen's d) for each known contrast on each dimension.

3. **Cross-prompt stability**: For each (model, language), how consistent are cultural profiles across the 6–8 prompt templates? Report stability coefficients. Prompt-specific outliers indicate grammatical confounds.

4. **Model size effect**: Gemma 3 12B vs 27B — does cultural differentiation require scale? The pilot already showed 12B fails (χ² p=0.11) while 27B succeeds (p=1.2×10⁻⁶). The full run confirms this across all 27 languages.

5. **Monolingual vs multilingual**: Do HPLT monolingual models show stronger cultural signal than multilingual models prompted in the same language? This tests whether monolingual training data produces stronger cultural encoding.

6. **Human alignment**: Rank correlation of LLM cultural dimension scores with human IW positions per country. How does this compare to Track 1's null result (r ≈ 0.00 on z-scored logprobs)?

### Visualizations

- **Cultural dimension scatter**: 2D scatter (Traditional→Secular vs Survival→Self-Expression) with LLM points alongside human IW positions
- **UMAP of behavioral profiles**: one point per (model, language), colored by IW cluster
- **Known-group effect size plot**: bar chart of Cohen's d for each contrast × dimension
- **Cross-prompt stability heatmap**: languages × prompt templates, colored by consistency
- **Model-size scaling plot**: cultural differentiation metric for 12B vs 27B (+ HPLT 2.15B monolingual)
- **Content category distributions**: stacked bar charts per language for a single model

## Phased Implementation

### Phase 1: Sampling infrastructure & pilot (2–3 weeks)

**Goal**: Build sampling pipeline, validate classifier, run pilot on 3–5 languages.

1. **Translate prompt templates** for pilot languages (eng, fin, pol, ron, zho)
   - Store in `data/prompt_templates/{lang}.json`
   - Include grammatical control variants for these languages

2. **Implement sampling pipeline**
   - Input: model, language, prompt template, N samples
   - Output: raw completions + metadata (tokens generated, truncation point)
   - Handle sentence boundary detection across languages
   - SLURM job scripts for cluster execution

3. **Build and validate LLM classifier**
   - Design classifier prompt (content category + 3 cultural dimensions)
   - Hand-label 200 completions across pilot languages
   - Iterate until κ > 0.7 on content categories
   - Batch classification pipeline (can use API or local IT model)

4. **Run pilot**: Gemma 3 27B PT × 5 pilot languages × 6 templates × 200 samples
   - Assess: Does the Nordic–Orthodox contrast appear? Is cross-prompt stability high?
   - If the pilot shows flat/indistinguishable distributions → investigate prompts, expand templates, or revise approach before scaling

5. **Build analysis pipeline**
   - Cultural dimension aggregation with bootstrap CIs
   - Known-group effect size computation
   - Cross-prompt stability metrics
   - IW comparison plots

### Phase 2: Full-scale behavioral comparison (2–3 weeks)

**Goal**: Run all models across all languages, produce behavioral cultural profiles.

1. **Complete translations** for all 27 languages
   - Native speaker review for core subset (10 languages)

2. **Run full matrix**
   - 22 HPLT models × 22 European languages (22 runs, one per model)
   - EuroLLM-22B × 22 European languages
   - Gemma 3 12B PT × 27 languages (size comparison — expected null)
   - Gemma 3 27B PT × 27 languages
   - Total: ~93 model-language configurations × 8 templates × 200 samples

3. **Classify all completions** (batch job, can run in parallel with sampling)

4. **Full analysis**
   - All primary comparisons (see Analysis Plan)
   - Statistical tests with multiple-comparison correction (Bonferroni or FDR)
   - Compile behavioral cultural profiles per (model, language) for Phase 3 validation

### Phase 3: SAE culture vector extraction & steering (3–4 weeks)

**Goal**: Use Gemma Scope 2 SAEs to extract culture vectors and validate via behavioral comparison.

> This phase requires the Phase 1–2 infrastructure as the evaluation harness. Culture vectors are validated by showing that steering produces behavioral changes consistent with observed cross-linguistic differences.

1. **Load Gemma 3 27B + Gemma Scope 2 residual SAEs**
   - Focus on layers at 50–85% depth (where cultural localization concentrates per prior work)
   - Start with `resid_post` SAEs, width 64k or 256k

2. **Identify language-specific features**
   - Replicate Deng et al. monolinguality metric ν_s^L on parallel text (Flores or similar)
   - Feed same content in all 27 languages, compute per-feature activation stats
   - Threshold for language-specific features

3. **Identify culture-specific features**
   - Feed curated culturally-loaded text through Gemma 3 27B:
     - WVS questions (translated) in each language — measure which SAE features activate differentially across languages
     - Wikipedia articles about culturally specific topics (family structure, religious practice, national holidays) written in the same language (English) but about different cultures — controls for language while varying culture
   - Compute monoculturality metric ν_s^C grouped by IW cultural cluster
   - Features with high ν for any culture cluster → culture feature candidates
   - Cross-validate: do features identified from WVS text agree with features identified from Wikipedia text?

4. **Build culture vectors**
   - culture_only(C) = mean activations for culture C − projection onto language feature subspace
   - Validate internal consistency: culture vectors from different languages for same IW cluster should be similar
   - Validate language invariance: a "Protestant Europe" culture feature should activate for text about Nordic values regardless of whether the text is in English, Finnish, or Chinese

5. **Steer Gemma 3 27B and evaluate**
   - For each cultural cluster C and language L:
     - h_steered = h_base + α·culture_vector(C)
     - Run the full sampling pipeline (all prompt templates, N=200 samples) with steering active
     - Classify steered completions using the same LLM classifier
     - Compare steered cultural profiles against Phase 2 profiles for cluster C's languages
   - **Key test — cross-language culture transfer**:
     - Steer toward Bulgarian (Orthodox) culture while prompting in English
     - Do completions shift toward more traditional/survival-oriented profiles?
     - Compare against English completions from the Bulgarian HPLT monolingual model
   - **Complementary validation via Track 1**:
     - Run WVS binary/categorical questions on steered model
     - Does steering toward culture C shift WVS responses toward country C's human EVS data?

6. **Dimensionality reduction of culture vectors**
   - PCA/UMAP on culture-only feature vectors for all clusters
   - Compare against IW cultural map positions
   - Compare against Phase 2 behavioral UMAP
   - Three-way convergence (SAE geometry, behavioral profiles, human IW positions) is the strongest possible evidence

## Output Data Format

### Raw completions: `results/completions/{model_type}_{lang}_{template}.jsonl`

Each line is a JSON object:
```json
{
  "model_type": "gemma3_27b_pt",
  "lang": "fin",
  "template": "values",
  "prompt": "Elämässä tärkeintä on",
  "completion": "perhe ja rakkaus...",
  "n_tokens": 23,
  "temperature": 0.8,
  "sample_idx": 42
}
```

### Classified completions: `results/classified/{model_type}_{lang}_{template}.parquet`

| Column | Type | Description |
|--------|------|-------------|
| model_type | str | e.g., "gemma3_27b_pt" |
| lang | str | Language code |
| template | str | Prompt template name |
| sample_idx | int | Sample index (0–199) |
| completion | str | Raw completion text |
| content_category | str | Classified content category |
| dim_indiv_collect | int | 1 (individualist) to 5 (collectivist) |
| dim_trad_secular | int | 1 (traditional) to 5 (secular-rational) |
| dim_surv_selfexpr | int | 1 (survival) to 5 (self-expression) |

### Aggregated profiles: `results/cultural_profiles.parquet`

| Column | Type | Description |
|--------|------|-------------|
| model_type | str | Model identifier |
| lang | str | Language code |
| template | str | Prompt template (or "all" for aggregate) |
| mean_indiv_collect | float | Mean individualism-collectivism score |
| ci_lower_ic | float | 95% CI lower bound |
| ci_upper_ic | float | 95% CI upper bound |
| mean_trad_secular | float | Mean traditional-secular score |
| mean_surv_selfexpr | float | Mean survival-self-expression score |
| cross_prompt_stability | float | Cross-template correlation (NaN for per-template rows) |
| n_samples | int | Number of classified completions |

## Hardware Requirements

- **Gemma 3 27B**: ~54GB bf16, ~27GB int8 → A100 80GB or H100
- **Gemma 3 12B**: ~24GB bf16 → A100 or L40S
- **HPLT 2.15B**: ~5GB bf16 → any GPU
- **EuroLLM-22B**: ~45GB bf16 → A100 80GB or H100
- **SAE inference**: minimal overhead (single matmul per layer per token)
- **Sampling throughput**: ~200 completions × 50 tokens × 8 templates ≈ 80K tokens per (model, language). At ~500 tok/s on A100, each config takes ~3 min. Full matrix of ~93 configs ≈ 5 GPU-hours.
- **Classification**: ~150K completions × ~200 classifier tokens ≈ 30M tokens. Via API or batched local inference.

## Key Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Sampling distributions too similar across languages (same entropy problem as logprobs) | Multiple prompt templates increase signal; expanded language set provides stronger contrast; pilot on 5 languages first to check |
| LLM classifier imposes its own cultural biases | Validate on hand-labeled data; use multiple classifier models; report raw completions alongside classifications |
| European languages too culturally similar | Expanded language set (zho, jpn, ara, hin, tur) provides strong contrasts; if intra-European variation is undetectable, focus on expanded set |
| Grammatical structure dominates cultural signal | Cross-prompt stability metric explicitly measures this; multiple rephrasings per template; if stability is low, the signal is grammatical |
| Model size confounds cultural signal | Explicit size comparison (12B/27B) controls for this; pilot already confirmed the threshold |
| Temperature sampling not deterministic | Use fixed seeds per configuration for reproducibility; N=200 samples gives sufficient statistical power |
| SAE features don't cleanly separate culture/language | This is a genuine open question — negative result is publishable |
| Phase 3 chicken-and-egg (need cultural signal to identify cultural features) | Use two independent source texts (WVS translations + English Wikipedia about different cultures) to break circularity |
