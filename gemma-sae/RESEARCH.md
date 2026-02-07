# Gemma SAE — Culture vs Language Feature Isolation

## Background: What Gemma Scope Is

Gemma Scope is a suite of >400 pretrained JumpReLU Sparse Autoencoders (SAEs) covering every layer and sublayer of Gemma 2 (2B, 9B, 27B). SAEs decompose model activations into sparse, overcomplete feature sets — each feature ideally captures a single interpretable concept (monosemantic).

- Paper: https://arxiv.org/abs/2408.05147
- Weights: https://huggingface.co/google/gemma-scope
- Interactive explorer: https://neuronpedia.org/gemma-scope
- Tutorial notebook: https://colab.research.google.com/drive/17dQFYUYnuKnP6OwQPH9v_GSYUW5aj-Rp

### Available SAE Weight Repos

| Repo | Model | Component |
|------|-------|-----------|
| `google/gemma-scope-2b-pt-res` | 2B base | Residual stream |
| `google/gemma-scope-2b-pt-mlp` | 2B base | MLP |
| `google/gemma-scope-2b-pt-att` | 2B base | Attention |
| `google/gemma-scope-9b-pt-res` | 9B base | Residual stream |
| `google/gemma-scope-9b-pt-mlp` | 9B base | MLP |
| `google/gemma-scope-9b-pt-att` | 9B base | Attention |
| `google/gemma-scope-27b-pt-res` | 27B base | Residual stream |
| `google/gemma-scope-9b-it-res` | 9B instruct | Residual stream |

Weight path format: `layer_{N}/width_{W}/average_l0_{sparsity}/params.npz`
Width options: 16k, 32k, 65k, 131k, 262k, 524k, 1M

### Loading SAE Weights (Python)

```python
import torch
import numpy as np
from huggingface_hub import hf_hub_download

# Download SAE params
path = hf_hub_download(
    repo_id="google/gemma-scope-9b-pt-res",
    filename="layer_20/width_16k/average_l0_71/params.npz",
    force_download=False,
)
params = np.load(path)
# params contains: W_enc, W_dec, b_enc, b_dec, threshold

# JumpReLU SAE
class JumpReLUSAE(torch.nn.Module):
    def __init__(self, d_model, d_sae):
        super().__init__()
        self.W_enc = torch.nn.Parameter(torch.zeros(d_model, d_sae))
        self.W_dec = torch.nn.Parameter(torch.zeros(d_sae, d_model))
        self.threshold = torch.nn.Parameter(torch.zeros(d_sae))
        self.b_enc = torch.nn.Parameter(torch.zeros(d_sae))
        self.b_dec = torch.nn.Parameter(torch.zeros(d_model))

    def encode(self, input_acts):
        pre_acts = input_acts @ self.W_enc + self.b_enc
        mask = (pre_acts > self.threshold)
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts

    def decode(self, acts):
        return acts @ self.W_dec + self.b_dec

    def forward(self, acts):
        acts = self.encode(acts)
        recon = self.decode(acts)
        return recon

# Load weights
pt_params = {k: torch.from_numpy(v) for k, v in params.items()}
sae = JumpReLUSAE(pt_params['W_enc'].shape[0], pt_params['W_enc'].shape[1])
sae.load_state_dict(pt_params)
```

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

### Proposed Method

**Step 1: Identify language-specific features**
- Use the monolinguality metric from Deng et al. (ACL 2025)
- Feed parallel text in N languages through Gemma 2
- For each SAE feature, compute ν_s^L = μ_s^L - γ_s^L
- Features with high ν for any language → language features

**Step 2: Identify culture-specific features**
- Feed culturally-loaded text (WVS question completions, cultural narratives, proverbs, value statements) in multiple languages
- For each SAE feature, compute a "monoculturality" metric analogous to monolinguality:
  - Group inputs by culture (not language)
  - ν_s^C = μ_s^C - γ_s^C (mean activation for culture C minus mean across other cultures)
- Key challenge: need same-culture content in multiple languages to disentangle
  - Use WVS response patterns as culture labels
  - Use translated cultural proverbs/sayings
  - Use culturally-specific but translatable scenarios

**Step 3: Averaging for robust signals**
- Your intuition about averaging is sound and supported by the literature
- CAA (Contrastive Activation Addition) already does this: v = mean(h_positive) - mean(h_negative)
- The more diverse items you average over for a given culture, the more the language/topic noise cancels out, leaving the "pure culture" signal
- This is analogous to how fMRI researchers average over many trials to isolate neural responses

**Step 4: Subtraction and transfer**
- Given culture_vector(Japanese) computed from Japanese-language text:
  - Subtract language_vector(Japanese) → culture-only vector
  - Add language_vector(English) → should produce "Japanese cultural style in English"
- Validate by checking if generated text reflects Japanese values on WVS items

**Step 5: Dimensionality reduction of culture vectors**
- Collect culture-only vectors for N cultures
- Stack into matrix R^{N × d_sae}
- Apply PCA, t-SNE, or UMAP
- Hypothesis: the resulting map should resemble the Inglehart-Welzel cultural map
  - Dimension 1 ≈ Traditional vs Secular-rational values
  - Dimension 2 ≈ Survival vs Self-expression values
- If this works, it would be strong evidence that LLMs have learned a structured cultural representation

## Feasibility Assessment

### What makes this tractable:
- Gemma Scope SAEs already exist and are free to download (CC-BY-4.0)
- Gemma 2 9B is runnable on a single A100 or even 2x A6000
- The language-feature identification method is proven (ACL 2025)
- WVS provides standardized cultural measurement across 80+ countries
- Cultural steering vectors are proven to be conserved across languages (April 2025 paper)

### Key challenges:
- **Data**: Need culturally-loaded parallel corpora. WVS translations help but are limited to survey items. May need to supplement with translated cultural texts.
- **Confounds**: Culture and language are deeply entangled. "Japanese culture" items in English may not activate the same features as in Japanese. The averaging-and-subtracting approach partially addresses this but may not fully separate them.
- **Granularity**: SAE features may not cleanly decompose into language/culture. Some features may be "Japanese formal register" — both language and culture simultaneously.
- **Validation**: Hard to ground-truth "culture-only" vectors without circular reasoning.

### Recommended starting point:
1. Load Gemma 2 9B + Gemma Scope residual SAEs (layers 19-28, where cultural localization concentrates)
2. Replicate the language-feature identification from Deng et al. using Flores-10 parallel data
3. Feed WVS questions in 5+ languages, collect SAE activations
4. Compute culture-specific features using the monoculturality metric
5. Build culture-only vectors via subtraction
6. PCA on the culture-only vectors — see if Inglehart-Welzel structure emerges

## Hardware Requirements
- Gemma 2 9B: ~18GB in fp16, ~9GB in int8
- SAE inference adds minimal overhead (single matrix multiply)
- Can run on 1x A100 40GB or 2x consumer GPUs with model parallelism
- Batch processing WVS items is cheap (250 questions × N languages × N runs)
