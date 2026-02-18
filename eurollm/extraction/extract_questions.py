#!/usr/bin/env python3
"""Extract survey questions from EVS 2017 field questionnaire PDFs using Gemini Flash.

Handles output truncation by making continuation requests until all questions
are extracted.
"""

from google import genai
from google.genai import types
import json
import re
import sys
import os
import time
from pathlib import Path

EXTRACTION_PROMPT = """Extract all opinion/values survey questions from this questionnaire PDF into JSON.

RULES:
- Extract ONLY opinion and values questions (attitudes, beliefs, importance ratings, justifiability scales, trust, etc.)
- SKIP demographic questions (age, sex, education, income, marital status, household size, etc.)
- SKIP interviewer instructions, routing instructions, and show card references
- SKIP questions about factual behavior (e.g., "how many hours do you work?") unless they are about frequency of activities like volunteering, political participation, etc.
- For battery questions (a shared stem with multiple sub-items), create a SEPARATE entry for each sub-item
- Preserve ALL text in the ORIGINAL LANGUAGE of the questionnaire
- Use the question numbering exactly as it appears in the PDF (e.g., Q1, Q2, v1, v2, etc.)

For each question, output a JSON object with these fields:
{
  "id": "Q1",           // Question number as it appears in the PDF
  "text": "...",         // Full question text in the original language. For battery items, include both the stem and the specific item.
  "stem": "...",         // Shared stem if this is part of a battery (null otherwise)
  "item": "...",         // Sub-item label if part of a battery (null otherwise)
  "response_type": "likert4|likert5|likert10|binary|categorical|frequency|action|confidence",
  "options": [
    {"value": 1, "label": "..."},
    {"value": 2, "label": "..."}
  ],
  "answer_cue": "..."   // The word for "Answer" in this language (e.g., "Answer" for English, "Antwort" for German, "Réponse" for French, etc.)
}

RESPONSE TYPE GUIDE:
- likert4: 4-point scales (importance, agreement, confidence, favorability)
- likert5: 5-point scales (agreement, frequency)
- likert10: 1-10 scales with labeled endpoints
- binary: yes/no, mentioned/not mentioned
- categorical: pick one from a list of distinct categories
- frequency: how often (never/sometimes/often/always type scales)
- action: have done / might do / would never do
- confidence: confidence in institutions (a great deal / quite a lot / not very much / none at all)

Output ONLY a valid JSON array. No markdown formatting, no code blocks, no explanation.
"""

CONTINUATION_PROMPT = """Continue extracting opinion/values survey questions from this questionnaire PDF.
You already extracted questions up to and including {last_id} ("{last_text}").

START from the NEXT question after {last_id}. Do NOT re-extract any questions you already extracted.

Use the EXACT SAME JSON schema for each question:
{{
  "id": "...",
  "text": "...",
  "stem": "...",
  "item": "...",
  "response_type": "likert4|likert5|likert10|binary|categorical|frequency|action|confidence",
  "options": [{{"value": 1, "label": "..."}}, ...],
  "answer_cue": "..."
}}

Same rules apply:
- Extract ONLY opinion/values questions, SKIP demographics and interviewer instructions
- For battery questions, create SEPARATE entries per sub-item
- Preserve original language

Output ONLY a valid JSON array. No markdown, no code blocks, no explanation.
"""

GEN_CONFIG = types.GenerateContentConfig(
    temperature=0.0,
    max_output_tokens=65536,
    response_mime_type="application/json",
)


def _upload_pdf(client, pdf_path):
    """Upload a PDF file and wait for it to be ready."""
    uploaded = client.files.upload(file=pdf_path)
    # Wait for processing
    while uploaded.state.name == "PROCESSING":
        time.sleep(2)
        uploaded = client.files.get(name=uploaded.name)
    if uploaded.state.name == "FAILED":
        raise RuntimeError(f"File upload failed for {pdf_path}")
    return uploaded


def _call_gemini(client, uploaded_file, prompt_text):
    """Make a single Gemini API call with an uploaded PDF and text prompt."""
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_uri(file_uri=uploaded_file.uri, mime_type="application/pdf"),
                    types.Part.from_text(text=prompt_text),
                ],
            )
        ],
        config=GEN_CONFIG,
    )
    return response.text.strip()


def _parse_json_response(text: str) -> tuple[list, bool]:
    """Parse JSON response, handling truncation.

    Returns:
        (questions, was_truncated)
    """
    try:
        return json.loads(text), False
    except json.JSONDecodeError:
        pass

    # Response was truncated. Find last complete question object.
    matches = list(re.finditer(r'\n  \}', text))
    if not matches:
        return [], True

    for match in reversed(matches):
        truncated = text[:match.end()]
        truncated = truncated.rstrip().rstrip(",") + "\n]"
        try:
            questions = json.loads(truncated)
            return questions, True
        except json.JSONDecodeError:
            continue

    return [], True


def extract_questions_from_pdf(pdf_path: str, client: genai.Client, lang_hint: str = "") -> list:
    """Extract questions from a single PDF using Gemini Flash.

    Automatically makes continuation requests if the response is truncated.
    lang_hint: optional extra instructions prepended to the prompt (e.g., language filter).
    """
    # Upload file once, reuse for all rounds
    uploaded_file = _upload_pdf(client, pdf_path)

    all_questions = []
    max_rounds = 12  # Safety limit

    try:
        for round_num in range(max_rounds):
            if round_num == 0:
                prompt = (lang_hint + "\n\n" + EXTRACTION_PROMPT) if lang_hint else EXTRACTION_PROMPT
            else:
                last_q = all_questions[-1]
                last_id = last_q.get("id", "?") or "?"
                last_text = (last_q.get("text", "") or "")[:80]
                cont = CONTINUATION_PROMPT.format(last_id=last_id, last_text=last_text)
                prompt = (lang_hint + "\n\n" + cont) if lang_hint else cont

            text = _call_gemini(client, uploaded_file, prompt)
            questions, was_truncated = _parse_json_response(text)

            if not questions:
                if round_num == 0:
                    raise RuntimeError(f"Failed to extract any questions from {pdf_path}")
                break

            all_questions.extend(questions)
            count = len(questions)

            # Show ID range for this round
            ids = [q.get("id", "?") for q in questions]
            first_id, last_id_round = ids[0], ids[-1]

            if was_truncated:
                print(f"    Round {round_num + 1}: {count} questions [{first_id}..{last_id_round}] (truncated, continuing...)")
                time.sleep(2)
            else:
                print(f"    Round {round_num + 1}: {count} questions [{first_id}..{last_id_round}] (complete)")
                break

    finally:
        # Clean up uploaded file
        try:
            client.files.delete(name=uploaded_file.name)
        except Exception:
            pass

    # Deduplicate by ID (continuation might re-extract a few)
    # Use (id, item) as the key since battery questions share the same id
    seen = set()
    deduped = []
    for q in all_questions:
        qid = q.get("id", "")
        item = q.get("item", "")
        key = (qid, item)
        if not qid:
            continue  # Skip malformed entries
        if key not in seen:
            seen.add(key)
            deduped.append(q)

    return deduped


def main():
    if len(sys.argv) < 3:
        print("Usage: extract_questions.py <pdf_path_or_dir> <output_path_or_dir> [--all]")
        print("  Single file: extract_questions.py eng.pdf eng.json")
        print("  All files:   extract_questions.py evs_questionnaires/ translations/ --all")
        sys.exit(1)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: Set GEMINI_API_KEY environment variable")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    process_all = "--all" in sys.argv

    if process_all:
        input_dir = input_path
        output_dir = output_path
        output_dir.mkdir(parents=True, exist_ok=True)

        pdf_files = sorted(input_dir.glob("*.pdf"))
        pdf_files = [f for f in pdf_files if f.stem != "evs_master_en"]

        total = len(pdf_files)
        for i, pdf_file in enumerate(pdf_files):
            lang_code = pdf_file.stem
            out_file = output_dir / f"{lang_code}.json"

            if out_file.exists():
                print(f"[{i+1}/{total}] Skipping {lang_code} (already exists)")
                continue

            print(f"[{i+1}/{total}] Processing {lang_code}...")
            try:
                questions = extract_questions_from_pdf(str(pdf_file), client)
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(questions, f, ensure_ascii=False, indent=2)
                print(f"  -> Total: {len(questions)} questions -> {out_file}")
            except Exception as e:
                print(f"  -> ERROR: {e}")

            # Rate limiting between PDFs
            if i < total - 1:
                time.sleep(3)

        print(f"\nDone. Processed {total} PDFs.")
    else:
        # Parse --lang-hint "..."
        lang_hint = ""
        for i, arg in enumerate(sys.argv):
            if arg == "--lang-hint" and i + 1 < len(sys.argv):
                lang_hint = sys.argv[i + 1]
                break

        print(f"Processing {input_path}...")
        if lang_hint:
            print(f"  Language hint: {lang_hint[:80]}...")
        questions = extract_questions_from_pdf(str(input_path), client, lang_hint=lang_hint)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(questions, f, ensure_ascii=False, indent=2)
        print(f"Total: {len(questions)} questions -> {output_path}")


if __name__ == "__main__":
    main()
