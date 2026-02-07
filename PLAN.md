# LLM Cultural Values Project — Plan

## Goal
Investigate how pretrained (non-finetuned) monolingual and multilingual LLMs encode and express cultural values, using the World Values Survey (WVS) Wave 8 questionnaire as a standardized measurement instrument.

## Two Research Tracks

### Track 1: EuroLLM — Survey Response Elicitation (`eurollm/`)

**Objective**: Get base (non-instruct) LLMs to "answer" WVS questions and compare response distributions across languages/models.

**Recommended approach**:

1. **Extract questions programmatically** from the WVS-8 PDF (Q1-Q250) into a structured JSON/CSV format with: question text, response options, response scale type (Likert-4, Likert-5, Likert-10, binary, categorical, ranked).

2. **Prompt design** — use bare question + response options with no framing:
   - No system prompt or minimal completion-style prompt (`"Complete the following."`)
   - Do NOT mention "World Values Survey" (avoids the model reciting known WVS findings)
   - Do NOT use persona prompts in the baseline condition
   - Present response options as the original numbered codes

3. **Models to test**:
   - Monolingual base models (e.g., Chinese-only, Arabic-only, Russian-only)
   - Multilingual base models (e.g., Llama-3 base, Gemma 2 base, EuroLLM)
   - Compare instruct-tuned vs base to measure RLHF alignment bias

4. **Primary method: logprob extraction**
   - For each question, extract log-probabilities over valid response tokens at the answer position
   - Normalize to get a probability distribution over responses
   - This avoids generation artifacts (hedging, refusals, meta-commentary)
   - Run on base models only (instruct models refuse or moralize)

5. **Secondary method: sampling distributions**
   - Run each question N=100+ times at temperature 0.7-1.0
   - Build empirical response distributions
   - Useful for models where logprobs aren't accessible

6. **Language-as-culture probe** (for multilingual models):
   - Present identical questions in different languages (using official WVS translations where available)
   - The language itself is the independent variable
   - Hypothesis: response distributions shift by language because training corpora are culturally distinct

7. **Controls**:
   - Randomize question order across runs
   - Test sensitivity to prompt formatting (numbering, spacing, newlines)
   - Compare against actual WVS country-level response data as ground truth

8. **Analysis**:
   - Compare model response distributions to real WVS country-level data
   - Compute Inglehart-Welzel cultural map coordinates from model responses
   - Measure effect of language vs explicit demographic persona on cultural alignment

### Track 2: Gemma SAE — Mechanistic Interpretability of Culture (`gemma-sae/`)

See `gemma-sae/RESEARCH.md` for detailed investigation notes.

**Objective**: Use Sparse Autoencoders (SAEs) from Gemma Scope to isolate culture-specific vs language-specific features in Gemma 2's internal representations, and explore whether culture can be disentangled from language mechanistically.

**Core idea**: If we can identify SAE features that activate for "culture X" independently of language, we could in theory:
- Add culture-X features + language-Y features to generate text with culture X's values in language Y
- Dimensionality-reduce the space of culture-only vectors to find a low-dimensional cultural map emerging from model internals

## Key References

- WVS-8 Questionnaire: `eurollm/WVS-8_QUESTIONNAIRE_V11_FINAL_Jan_2024.pdf`
- Gemma Scope paper: https://arxiv.org/abs/2408.05147
- "Localized Cultural Knowledge is Conserved and Controllable in LLMs" (2025): https://arxiv.org/abs/2504.10191
- "Unveiling Language-Specific Features in LLMs via SAEs" (ACL 2025): https://arxiv.org/abs/2505.05111
- "LinguaLens" (EMNLP 2025): https://arxiv.org/abs/2502.20344
- "Multilingual != Multicultural" (2025): https://arxiv.org/abs/2502.16534
