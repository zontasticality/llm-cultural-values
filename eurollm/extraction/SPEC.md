# Extraction Module

## Purpose
Extract structured survey questions from EVS questionnaire PDFs and align them across 22 languages into a canonical question set.

## Inputs
- `../questionnaires/*.pdf` — EVS national questionnaire PDFs (22 languages + master English)
- Gemini Flash API (via `.env` key)

## Outputs
- `../data/translations/*.json` — Per-language extracted questions (one JSON per language)
- `../data/questions.json` — Aligned canonical question set (187 questions, 22 languages)

## Key Functions / Public API

### `extract_questions.py`
- CLI: `python extraction/extract_questions.py --pdf <path> --lang <code> --output <path>`
- Uses Gemini Flash to extract structured questions from PDFs
- Handles output truncation via continuation prompts

### `align_questions.py`
- CLI: `python extraction/align_questions.py`
- Three-pass matching: v-code regex, Q-code position, Gemini semantic
- Outputs canonical `questions.json` with per-language translations

## Dependencies
- External: `google-genai` (Gemini API)
- No cross-module imports
