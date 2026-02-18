# Session State — Saved 2026-02-07

## Ultimate Goal
Run Track 1 of the LLM cultural values project: elicit WVS/EVS survey responses from 22 HPLT monolingual 2.15B models + EuroLLM-22B multilingual, compare response distributions across languages, and plot on an Inglehart-Welzel cultural map.

Full plan: `eurollm/TRACK_1_PLAN.md` (fully updated and refined)

## Current Status: Step 2 COMPLETE — Ready for Step 3

### Step 2: Gemini Flash Question Extraction — DONE (22/22)

**Script**: `eurollm/extract_questions.py`
- Uses `google.genai` SDK (venv at `eurollm/.venv/`)
- Supports `--lang-hint` for bilingual PDF filtering (added during re-extraction)
- max_rounds=12 (increased from 8)
- Deduplicates by (id, item) key

**All 22 extractions in `eurollm/translations/`:**
bul:211, ces:229, dan:196, deu:204, ell:189, eng:202, est:301, fin:303,
fra:297, hrv:202, hun:201, ita:207, lit:204, lvs:208, nld:193, pol:216,
por:213, ron:213, slk:291, slv:189, spa:195, swe:225

**Verification findings (from detailed QA agent):**
- Bilingual PDFs (slk, lvs, est) were re-extracted with language filter hints — no more Russian/Hungarian contamination
- Truncated extractions (deu, swe, nld, fin) were re-extracted with "don't stop early" hints — all now have full coverage
- Some languages have overcounting (slk:291, est:301, fin:303, fra:297) due to continuation loop duplicates — will be cleaned in alignment
- Systematic issues to fix in Step 3:
  1. **ID normalization**: IDs are completely inconsistent across languages (Q-codes, v-codes, P-codes, numeric, Cyrillic). Need semantic/positional matching.
  2. **DK/NA option filtering**: 18/22 languages have "don't know" codes (88, 99) in options
  3. **Empty likert10 options**: Polish has 40/41 likert10 Qs with 0 options; fixable programmatically
  4. **Response type normalization**: Minor (`action` vs `categorical`, `binary` vs `categorical`)
  5. **Null text fields**: bul(2), dan(3), fin(15), slk(2) — minor

## What Comes Next (Implementation Order)

1. ~~Download EVS PDFs~~ DONE
2. ~~Extract questions via Gemini Flash~~ DONE (22/22)
3. **Align and validate** — cross-language question alignment, build master `questions.json` ← NEXT
4. **Tokenizer verification** — confirm digit tokenization for Gemma-3 and EuroLLM tokenizers
5. **Write prompt template code** → `prompt_templates.py`
6. **Write logprob extraction** → `extract_logprobs.py`
7. **Validate on English HPLT model locally**
8. **Write SLURM scripts** → `run_slurm.sh`, `launch_all.sh`
9. **Run on cluster** → submit 44 jobs
10. **Write analysis** → `analyze.py`
11. **Generate figures** → `figures/`

## Key Issues Already Resolved

1. **ZIP files**: 5 GESIS downloads were ZIP archives — extracted to PDFs
2. **Gemini output truncation**: continuation prompts with full JSON schema
3. **Gemini file size**: uses file upload API, not inline bytes
4. **Bilingual PDFs**: slk (Slovak/Hungarian), lvs (Latvian/Russian), est (Estonian/Russian) — re-extracted with `--lang-hint` to filter target language only
5. **Premature extraction completion**: deu, swe, nld, fin — re-extracted with "extract ALL questions" hints

## Key Design Decisions (Already Made)
- Two-step conditional logprob for "10" (preserves original 1-10 WVS scales)
- EuroLLM-22B-2512 (not 9B) as the multilingual model
- HPLT monolingual models kept (core research value: isolating per-language cultural signal)
- Forward + reversed option ordering to debias position bias
- P_valid quality metric (total prob mass on valid answer tokens before renormalization)
- Leading-space token variants summed (" 1" + "1")
- EVS 2017 questionnaires (human translations, not machine translated)
- Training dynamics bonus section dropped (overcomplicated)

## Environment
- Python venv: `eurollm/.venv/` (has google-genai, google-generativeai packages)
- Working dir: `/scratch4/workspace/zevwilson_umass_edu-culture-llm/llm-cultural-values`
- GEMINI_API_KEY in .env
- Memory file: see `/home/zevwilson_umass_edu/.claude/projects/-scratch4-workspace-zevwilson-umass-edu-culture-llm-llm-cultural-values/memory/MEMORY.md`
