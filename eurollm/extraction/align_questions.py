#!/usr/bin/env python3
"""Align EVS survey questions across 22 languages into a canonical questions.json.

Phases:
1. Extract v-codes from all questions via regex
2. Build Q-to-v mapping from Greek (Rosetta Stone) + Hungarian reference
3. Build canonical question list from English
4. Auto-match all languages via v-codes
5. Match remaining Q-code questions by Q-code + battery position
6. Gemini Flash semantic matching for DAN, FIN, NLD + gaps
7. Clean options (DK/NA, likert10, response types, answer_cue, v-code prefixes)
8. Validate and output questions.json
"""

import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRANSLATIONS_DIR = PROJECT_ROOT / "data" / "translations"
OUTPUT_FILE = PROJECT_ROOT / "data" / "questions.json"
CACHE_DIR = PROJECT_ROOT / ".alignment_cache"

ALL_LANGS = [
    "bul", "ces", "dan", "deu", "ell", "eng", "est", "fin", "fra",
    "hrv", "hun", "ita", "lit", "lvs", "nld", "pol", "por", "ron",
    "slk", "slv", "spa", "swe",
]

# Hardcoded answer cues per language (for prompt construction)
ANSWER_CUES = {
    "bul": "Отговор",
    "ces": "Odpověď",
    "dan": "Svar",
    "deu": "Antwort",
    "ell": "Απάντηση",
    "eng": "Answer",
    "est": "Vastus",
    "fin": "Vastaus",
    "fra": "Réponse",
    "hrv": "Odgovor",
    "hun": "Válasz",
    "ita": "Risposta",
    "lit": "Atsakymas",
    "lvs": "Atbilde",
    "nld": "Antwoord",
    "pol": "Odpowiedź",
    "por": "Resposta",
    "ron": "Răspuns",
    "slk": "Odpoveď",
    "slv": "Odgovor",
    "spa": "Respuesta",
    "swe": "Svar",
}

# Response type normalization map
RESPONSE_TYPE_NORM = {
    "action": "categorical",
    "binary": "categorical",
    "confidence": "likert4",
}


# ── Phase 1: Extract v-codes from all questions via regex ──────────────

def extract_vcode(id_str: str, item_str: str) -> str | None:
    """Extract v-code from question ID or item field using cascading regex."""
    id_str = id_str or ""
    item_str = item_str or ""

    # 1. Exact vNN in ID (hun, ron, spa, swe, etc.)
    if m := re.match(r"^v(\d+)$", id_str):
        return f"v{m.group(1)}"

    # 2. vNNa variant in ID (v45a → v45)
    if m := re.match(r"^v(\d+)[a-z]$", id_str):
        return f"v{m.group(1)}"

    # 3. [QNN][vNN] format (Greek)
    if m := re.match(r"^\[Q\d+[A-Za-z]?\]\[v(\d+)\]$", id_str):
        return f"v{m.group(1)}"

    # 4. QNN_vNN format (deu, hrv, slv)
    if m := re.match(r"^Q\d+_v(\d+)$", id_str, re.IGNORECASE):
        return f"v{m.group(1)}"

    # 5. qNvN format (Latvian)
    if m := re.match(r"^q\d+v(\d+)$", id_str):
        return f"v{m.group(1)}"

    # 6. vNN at start of item field (pol, hrv, slv: "v1 praca")
    if m := re.match(r"^v(\d+)\s", item_str):
        return f"v{m.group(1)}"

    # 7. vNN as entire item (fra, lit)
    if m := re.match(r"^v(\d+)$", item_str):
        return f"v{m.group(1)}"

    return None


def extract_qcode(id_str: str) -> str | None:
    """Extract base Q-code (without letter suffix) from question ID."""
    id_str = id_str or ""

    # [QNN][vNN] → QNN
    if m := re.match(r"^\[(Q\d+)[A-Za-z]?\]\[v\d+\]$", id_str):
        return m.group(1)

    # QNN_vNN → QNN
    if m := re.match(r"^(Q\d+)_v\d+$", id_str, re.IGNORECASE):
        return m.group(1).upper()

    # qNvN → QN
    if m := re.match(r"^q(\d+)v\d+$", id_str):
        return f"Q{m.group(1)}"

    # Plain QNN or qNN (including letter suffixes like Q23a)
    if m := re.match(r"^[Qq](\d+)[a-zA-Z]?$", id_str):
        return f"Q{m.group(1)}"

    # qN_N format (Latvian: q38_1)
    if m := re.match(r"^q(\d+)_\d+$", id_str):
        return f"Q{m.group(1)}"

    # PNN (Polish)
    if m := re.match(r"^P(\d+)$", id_str):
        return f"Q{m.group(1)}"

    # Cyrillic В (Bulgarian) → Q
    if m := re.match(r"^В(\d+)$", id_str):
        return f"Q{m.group(1)}"

    return None


# ── Phase 2: Build Q-to-v mapping from Greek ───────────────────────────

def build_q_to_v_map(ell_questions: list) -> dict:
    """Build Q-code+position → v-code mapping from Greek file.

    Greek uses [QNNx][vNN] format, giving us an authoritative mapping.
    Returns dict like {"Q1": ["v1", "v2", "v3", "v4", "v5", "v6"], "Q2": ["v7"], ...}
    """
    q_to_v = defaultdict(list)
    for q in ell_questions:
        id_str = q.get("id", "")
        m = re.match(r"^\[(Q\d+)[A-Za-z]?\]\[v(\d+)\]$", id_str)
        if m:
            qcode = m.group(1)
            vcode = f"v{m.group(2)}"
            q_to_v[qcode].append(vcode)
    # Sort each battery by v-code number to maintain position order
    for qcode in q_to_v:
        q_to_v[qcode].sort(key=lambda v: int(v[1:]))
    return dict(q_to_v)


# ── Phase 3: Build canonical question list from English ────────────────

def build_canonical_questions(eng_questions: list, q_to_v: dict) -> list:
    """Build canonical question list from English extraction.

    Each canonical question gets a v-code (from regex or Q-to-v mapping).
    """
    canonical = []
    seen_vcodes = set()

    # Group English questions by (id, position) to handle batteries
    battery_counts = defaultdict(int)

    for q in eng_questions:
        qid = q.get("id", "")

        # Try regex extraction first
        vcode = extract_vcode(qid, q.get("item", ""))

        if not vcode:
            # Use Q-to-v mapping from Greek
            qcode = extract_qcode(qid)
            if qcode and qcode in q_to_v:
                position = battery_counts[qcode]
                vcodes_for_q = q_to_v[qcode]
                if position < len(vcodes_for_q):
                    vcode = vcodes_for_q[position]

        battery_counts[qid] += 1

        if not vcode:
            continue

        if vcode in seen_vcodes:
            continue
        seen_vcodes.add(vcode)

        canonical.append({
            "canonical_id": vcode,
            "response_type": q.get("response_type", ""),
            "option_count": len(q.get("options", [])),
            "eng_text": q.get("text", ""),
            "eng_stem": q.get("stem"),
            "eng_item": q.get("item"),
            "eng_options": q.get("options", []),
            "eng_original_id": qid,
        })

    canonical.sort(key=lambda c: int(c["canonical_id"][1:]))
    return canonical


# ── Phase 4 & 5: Match all languages ──────────────────────────────────

def _response_type_compatible(q_rtype: str, canonical_rtype: str) -> bool:
    """Check if a question's response type is compatible with the canonical one."""
    # Normalize both
    q_norm = RESPONSE_TYPE_NORM.get(q_rtype, q_rtype)
    c_norm = RESPONSE_TYPE_NORM.get(canonical_rtype, canonical_rtype)

    if q_norm == c_norm:
        return True

    # likert4 and likert5 are close enough (some languages use 5-point for what
    # others use 4-point)
    if {q_norm, c_norm} <= {"likert4", "likert5"}:
        return True

    return False


def match_language(lang: str, questions: list, canonical: dict,
                   q_to_v: dict) -> dict:
    """Match a language's questions to canonical v-codes.

    Returns dict: vcode → {question_data, match_method}
    """
    matched = {}
    battery_counts = defaultdict(int)

    for q in questions:
        qid = q.get("id", "")
        item = q.get("item", "")

        # Phase 4: Try v-code regex
        vcode = extract_vcode(qid, item)
        if vcode and vcode in canonical:
            if vcode not in matched:
                matched[vcode] = (q, "vcode")
            battery_counts[qid] += 1
            continue

        # Phase 5: Try Q-code + battery position
        qcode = extract_qcode(qid)
        if qcode and qcode in q_to_v:
            position = battery_counts.get(qid, 0)
            vcodes_for_q = q_to_v[qcode]
            if position < len(vcodes_for_q):
                vcode = vcodes_for_q[position]
                if vcode in canonical and vcode not in matched:
                    # Validate response type compatibility
                    canon_rtype = canonical[vcode]["response_type"]
                    q_rtype = q.get("response_type", "")
                    if _response_type_compatible(q_rtype, canon_rtype):
                        matched[vcode] = (q, "position")
                    # else: skip - response type mismatch indicates wrong match

        battery_counts[qid] += 1

    return matched


# ── Phase 6: Gemini semantic matching ──────────────────────────────────

def gemini_match_batch(unmatched: list, canonical_list: list,
                       lang: str, client) -> dict:
    """Use Gemini to semantically match unmatched questions to canonical list.

    Returns dict: vcode → question_data
    """
    from google.genai import types

    if not unmatched:
        return {}

    # Build canonical reference text with full context
    canonical_ref = []
    for c in canonical_list:
        item = c.get("eng_item") or ""
        stem = c.get("eng_stem") or ""
        text = c.get("eng_text", "")[:100]
        if item and stem:
            ref = f"{stem[:60]} → {item}"
        elif item:
            ref = f"{text[:80]} → {item}"
        else:
            ref = text[:100]
        canonical_ref.append(f'{c["canonical_id"]}: {ref}')
    canonical_text = "\n".join(canonical_ref)

    # Build unmatched questions text with full context
    unmatched_items = []
    for i, q in enumerate(unmatched):
        item = q.get("item") or ""
        stem = q.get("stem") or ""
        text = q.get("text", "")[:100]
        rtype = q.get("response_type", "")
        if item and stem:
            desc = f"{stem[:50]} → {item} [{rtype}]"
        elif item:
            desc = f"{text[:60]} → {item} [{rtype}]"
        else:
            desc = f"{text[:80]} [{rtype}]"
        unmatched_items.append(f"  {i}: [{q.get('id', '?')}] {desc}")
    unmatched_text = "\n".join(unmatched_items)

    prompt = f"""Match these {lang.upper()} EVS survey questions to their English canonical equivalents.
These are from the European Values Study 2017 questionnaire.

CANONICAL QUESTIONS (English) with v-codes:
{canonical_text}

UNMATCHED QUESTIONS ({lang.upper()}):
{unmatched_text}

RULES:
- Match ONLY if the questions ask about THE SAME THING (same topic, same response scale)
- A question about "happiness" is NOT the same as "life importance"
- A question about "prayer frequency" is NOT the same as "God importance"
- A question about "EU expansion" is NOT the same as "national pride"
- If unsure, output "NONE" - false matches are worse than missing matches
- Battery sub-items must match the specific sub-item, not just the battery stem

Output ONLY a JSON object mapping index (as string) to v-code or "NONE":
{{"0": "v1", "1": "NONE", "2": "v39", ...}}
"""

    config = types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=8192,
        response_mime_type="application/json",
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=config,
        )
    except Exception as e:
        print(f"    WARNING: Gemini API call failed for {lang}: {e}")
        return {}

    try:
        resp_text = response.text
    except Exception:
        resp_text = None
    if not resp_text:
        print(f"    WARNING: Empty Gemini response for {lang}")
        return {}

    try:
        result = json.loads(resp_text.strip())
    except json.JSONDecodeError:
        print(f"    WARNING: Failed to parse Gemini response for {lang}")
        return {}

    matches = {}
    for idx_str, vcode in result.items():
        if vcode == "NONE" or not vcode:
            continue
        idx = int(idx_str)
        if 0 <= idx < len(unmatched) and vcode in {c["canonical_id"] for c in canonical_list}:
            matches[vcode] = unmatched[idx]

    return matches


def load_gemini_cache(lang: str) -> dict | None:
    """Load cached Gemini matches for a language."""
    cache_file = CACHE_DIR / f"{lang}_gemini.json"
    if not cache_file.exists():
        return None
    with open(cache_file, "r", encoding="utf-8") as f:
        return json.load(f)


def run_gemini_matching(all_questions: dict, matched_per_lang: dict,
                        canonical_dict: dict, canonical_list: list) -> dict:
    """Run Gemini matching for languages with significant gaps.

    Sends ALL unmatched questions (including Q-coded ones that didn't match
    via position) to Gemini for semantic matching.

    Returns updated matched_per_lang.
    """
    api_key = os.environ.get("GEMINI_API_KEY")

    for lang in ALL_LANGS:
        if lang == "eng" or lang not in all_questions:
            continue

        already_matched_vcodes = set(matched_per_lang.get(lang, {}).keys())
        total_canonical = len(canonical_dict)
        coverage = len(already_matched_vcodes) / total_canonical if total_canonical else 0

        # Only use Gemini if coverage is below 85%
        if coverage >= 0.85:
            continue

        # Check cache first
        cached = load_gemini_cache(lang)
        if cached:
            if lang not in matched_per_lang:
                matched_per_lang[lang] = {}
            loaded = 0
            # Rebuild matches from cache by finding the original question data
            for vcode, cache_entry in cached.items():
                if vcode in matched_per_lang[lang] or vcode not in canonical_dict:
                    continue
                # Find the original question by id
                orig_q = None
                for q in all_questions[lang]:
                    if q.get("id") == cache_entry.get("id"):
                        item_match = (
                            not cache_entry.get("item")
                            or q.get("item") == cache_entry.get("item")
                        )
                        if item_match:
                            orig_q = q
                            break
                if orig_q:
                    matched_per_lang[lang][vcode] = (orig_q, "gemini")
                    already_matched_vcodes.add(vcode)
                    loaded += 1
            if loaded:
                new_coverage = len(already_matched_vcodes) / total_canonical * 100
                print(f"  {lang}: loaded {loaded} from cache ({new_coverage:.1f}% coverage)")
                continue

        if not api_key:
            print(f"  {lang}: needs Gemini ({coverage*100:.1f}% coverage) but GEMINI_API_KEY not set")
            continue

        from google import genai
        client = genai.Client(api_key=api_key)

        questions = all_questions[lang]

        # Collect ALL unmatched questions (including Q-coded ones that didn't match)
        unmatched = []
        matched_q_ids = set()
        for vcode, (q, _) in matched_per_lang.get(lang, {}).items():
            matched_q_ids.add((q.get("id", ""), q.get("item", "")))

        for q in questions:
            key = (q.get("id", ""), q.get("item", ""))
            if key in matched_q_ids:
                continue
            unmatched.append(q)

        if not unmatched:
            continue

        # Unmatched canonical questions (ones we still need)
        needed_vcodes = set(canonical_dict.keys()) - already_matched_vcodes
        if not needed_vcodes:
            continue

        print(f"  {lang}: {len(unmatched)} unmatched questions, "
              f"{len(needed_vcodes)} canonical gaps, sending to Gemini...")

        # Build needed canonical subset
        needed_canonical = [c for c in canonical_list if c["canonical_id"] in needed_vcodes]

        # Batch in groups of 50
        gemini_matches = {}
        batch_num = 0
        for batch_start in range(0, len(unmatched), 50):
            batch = unmatched[batch_start:batch_start + 50]
            # Reduce canonical list as we find matches
            remaining_canonical = [
                c for c in needed_canonical
                if c["canonical_id"] not in gemini_matches
            ]
            if not remaining_canonical:
                break

            batch_num += 1
            # Retry up to 2 times on failure
            for attempt in range(2):
                try:
                    batch_matches = gemini_match_batch(
                        batch, remaining_canonical, lang, client
                    )
                    gemini_matches.update(batch_matches)
                    print(f"    Batch {batch_num}: "
                          f"matched {len(batch_matches)}/{len(batch)}")
                    break
                except Exception as e:
                    if attempt == 0:
                        print(f"    Batch {batch_num}: failed ({e}), retrying...")
                        time.sleep(5)
                    else:
                        print(f"    Batch {batch_num}: failed again ({e}), skipping")

            time.sleep(2)

        # Merge Gemini matches
        if lang not in matched_per_lang:
            matched_per_lang[lang] = {}
        for vcode, q in gemini_matches.items():
            if vcode not in matched_per_lang[lang]:
                matched_per_lang[lang][vcode] = (q, "gemini")

        final_count = len(matched_per_lang[lang])
        final_pct = final_count / total_canonical * 100
        print(f"    Total: {final_count}/{total_canonical} ({final_pct:.1f}%)")

        # Cache results
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = CACHE_DIR / f"{lang}_gemini.json"
        cache_data = {
            vcode: {
                "id": q.get("id"),
                "item": q.get("item"),
                "text": (q.get("text", "") or "")[:100],
            }
            for vcode, q in gemini_matches.items()
        }
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)

    return matched_per_lang


# ── Phase 7: Clean options ─────────────────────────────────────────────

def clean_options(options: list, response_type: str,
                  expected_count: int = 0) -> list:
    """Clean options: remove DK/NA, reconstruct empty likert10.

    Args:
        options: Raw options from extraction
        response_type: Question response type
        expected_count: Expected number of valid options (from English canonical)
    """
    if not options:
        if response_type == "likert10":
            return [{"value": i, "label": ""} for i in range(1, 11)]
        return []

    # Filter DK/NA options by label patterns
    dk_na_patterns = re.compile(
        r"(?i)(don.?t know|weet niet|vet ikke|ei osaa|weiß nicht|"
        r"no s[eé]|non so|nie wiem|nu [sș]tiu|neviem|ne vem|"
        r"no answer|geen antwoord|intet svar|ei vastausta|"
        r"keine antwort|sin respuesta|senza risposta|"
        r"bez odpovědi|bez odgovora|nav atbildes|neatsakyta|"
        r"nincs válasz|fără răspuns|bez odpovede|"
        r"refuse|refus|odmítá|odbija|atteikās|atsisako|"
        r"megtagadja|refuză|odmieta|zavrača|vägrar|"
        r"не знам|не знаю|δεν [ξγ]νωρίζω|"
        r"not applicable|ej tillämpligt|ei koske|"
        r"nežinau|nezinu|не знаe|δεν απαντ|"
        r"nemám názor|brez mnenja|ohne meinung|"
        r"neturiu nuomonės|nav viedokļa)"
    )

    cleaned = []
    for opt in options:
        val = opt.get("value")
        label = opt.get("label", "")

        # Skip DK/NA by high value (88, 98, 99 are standard EVS DK/NA codes)
        if isinstance(val, int) and val >= 88:
            continue

        # Skip value 0 if it looks like DK/NA (only when there are enough options)
        if val == 0 and len(options) > 3:
            if not label or dk_na_patterns.search(label):
                continue

        # Skip DK/NA by label
        if label and dk_na_patterns.search(label):
            continue

        cleaned.append(opt)

    # For likert10, if we have too few options, reconstruct 1-10
    if response_type == "likert10" and len(cleaned) < 3:
        return [{"value": i, "label": ""} for i in range(1, 11)]

    # If expected_count is set and we still have too many options,
    # trim extras from the end (likely DK/NA not caught by patterns)
    if expected_count > 0 and len(cleaned) > expected_count:
        # Only trim if the extra options have higher values
        max_expected_val = expected_count
        if response_type == "likert10":
            max_expected_val = 10
        trimmed = [o for o in cleaned if o.get("value", 0) <= max_expected_val]
        if len(trimmed) >= expected_count:
            cleaned = trimmed

    return cleaned


def clean_item_text(item: str) -> str:
    """Strip v-code prefix from item text ('v1 praca' → 'praca')."""
    if not item:
        return item
    return re.sub(r"^v\d+\s+", "", item)


def normalize_response_type(rtype: str) -> str:
    """Normalize response types."""
    return RESPONSE_TYPE_NORM.get(rtype, rtype)


# ── Phase 8: Build output ─────────────────────────────────────────────

def build_output(canonical_list: list, matched_per_lang: dict) -> dict:
    """Build the final questions.json output."""
    questions = []
    lang_coverage = defaultdict(int)

    for canon in canonical_list:
        vid = canon["canonical_id"]
        rtype = normalize_response_type(canon["response_type"])
        option_count = canon["option_count"]

        translations = {}

        # English entry
        eng_options = clean_options(canon["eng_options"], rtype)
        eng_item = canon["eng_item"]
        translations["eng"] = {
            "text": canon["eng_text"],
            "stem": canon["eng_stem"],
            "item": eng_item,
            "options": eng_options,
            "answer_cue": ANSWER_CUES["eng"],
            "original_id": canon["eng_original_id"],
            "match_method": "canonical",
        }
        lang_coverage["eng"] += 1

        # Update option_count from cleaned English options
        if eng_options:
            option_count = len(eng_options)

        # Other languages
        for lang in ALL_LANGS:
            if lang == "eng":
                continue
            lang_matches = matched_per_lang.get(lang, {})
            if vid not in lang_matches:
                continue

            q, method = lang_matches[vid]
            opts = clean_options(q.get("options", []), rtype, option_count)
            item = clean_item_text(q.get("item"))
            text = q.get("text", "")
            stem = q.get("stem")

            # If text is just a Q-code/v-code reference, use stem or item instead
            if text and re.match(r"^[QqPpVv]\d+[A-Za-z]?$", text.strip()):
                text = stem or item or ""

            # Strip v-code prefix from text too
            if text:
                text = re.sub(r"^v\d+\s+", "", text)

            # If text is empty but we have stem+item, reconstruct
            if not text and stem and item:
                text = f"{stem}\n{item}"
            elif not text and item:
                text = item
            elif not text and stem:
                text = stem

            # Skip if text is empty/null
            if not text:
                continue

            translations[lang] = {
                "text": text,
                "stem": stem,
                "item": item,
                "options": opts,
                "answer_cue": ANSWER_CUES.get(lang, q.get("answer_cue")),
                "original_id": q.get("id", ""),
                "match_method": method,
            }
            lang_coverage[lang] += 1

        questions.append({
            "canonical_id": vid,
            "response_type": rtype,
            "option_count": option_count,
            "translations": translations,
        })

    # Build metadata
    alignment_stats = {
        "total_canonical": len(canonical_list),
        "coverage_per_language": dict(sorted(lang_coverage.items())),
        "match_methods": {},
    }

    # Count match methods
    method_counts = defaultdict(lambda: defaultdict(int))
    for lang in ALL_LANGS:
        if lang == "eng":
            continue
        for vid, (q, method) in matched_per_lang.get(lang, {}).items():
            method_counts[lang][method] += 1
    alignment_stats["match_methods"] = {
        lang: dict(counts) for lang, counts in sorted(method_counts.items())
    }

    return {
        "metadata": {
            "languages": ALL_LANGS,
            "total_questions": len(questions),
            "alignment_stats": alignment_stats,
        },
        "questions": questions,
    }


# ── Validation ─────────────────────────────────────────────────────────

INGLEHART_WELZEL_CORE = {
    "v63": "God importance",
    "v39": "Life satisfaction",
    "v7": "Happiness",
    "v31": "Trust",
    "v198": "National pride",
    "v95": "Obedience (child quality)",
    "v98": "Political action: signing petition",
    "v99": "Political action: joining boycotts",
    "v100": "Political action: attending demonstrations",
    "v101": "Political action: joining strikes",
}


def validate_output(output: dict):
    """Validate the output and print a report."""
    questions = output["questions"]
    stats = output["metadata"]["alignment_stats"]

    print("\n" + "=" * 60)
    print("ALIGNMENT REPORT")
    print("=" * 60)

    print(f"\nTotal canonical questions: {len(questions)}")

    # Coverage per language
    print("\nCoverage per language:")
    print(f"  {'Lang':<6} {'Matched':>8} {'Coverage':>10}")
    print(f"  {'-'*6} {'-'*8} {'-'*10}")
    for lang in ALL_LANGS:
        count = stats["coverage_per_language"].get(lang, 0)
        pct = count / len(questions) * 100 if questions else 0
        flag = " *** LOW ***" if pct < 75 else ""
        print(f"  {lang:<6} {count:>8} {pct:>9.1f}%{flag}")

    # Match methods
    print("\nMatch methods per language:")
    for lang, methods in sorted(stats.get("match_methods", {}).items()):
        parts = [f"{method}={count}" for method, count in sorted(methods.items())]
        print(f"  {lang}: {', '.join(parts)}")

    # Inglehart-Welzel core questions
    print("\nInglehart-Welzel core questions:")
    canonical_ids = {q["canonical_id"] for q in questions}
    for vid, desc in sorted(INGLEHART_WELZEL_CORE.items()):
        present = vid in canonical_ids
        if present:
            q = next(q for q in questions if q["canonical_id"] == vid)
            langs_with = len(q["translations"])
            print(f"  {vid:<6} {desc:<40} ✓ ({langs_with} languages)")
        else:
            print(f"  {vid:<6} {desc:<40} ✗ MISSING")

    # Cross-language option count consistency
    print("\nOption count consistency (questions with >1 different count across langs):")
    inconsistent = 0
    for q in questions:
        counts = set()
        for lang, t in q["translations"].items():
            if t["options"]:
                counts.add(len(t["options"]))
        if len(counts) > 1:
            inconsistent += 1
            if inconsistent <= 5:
                print(f"  {q['canonical_id']}: option counts = {sorted(counts)}")
    if inconsistent > 5:
        print(f"  ... and {inconsistent - 5} more")
    print(f"  Total inconsistent: {inconsistent}/{len(questions)}")

    # Spot check a few questions
    spot_checks = [
        ("v63", "God importance"),
        ("v39", "Life satisfaction"),
        ("v31", "Trust"),
    ]
    spot_langs = ["eng", "deu", "fra", "spa", "pol"]
    for vid, desc in spot_checks:
        print(f"\nSpot check - {vid} ({desc}):")
        for q in questions:
            if q["canonical_id"] == vid:
                for lang in spot_langs:
                    if lang in q["translations"]:
                        t = q["translations"][lang]
                        text = t.get("text", "")[:70]
                        method = t.get("match_method", "?")
                        nopts = len(t.get("options", []))
                        print(f"  {lang}: [{method}] ({nopts} opts) {text}")
                break


# ── Main ───────────────────────────────────────────────────────────────

def main():
    print("Step 3: Cross-Language Question Alignment")
    print("=" * 60)

    # Load all translations
    print("\nLoading translations...")
    all_questions = {}
    for lang in ALL_LANGS:
        path = TRANSLATIONS_DIR / f"{lang}.json"
        if not path.exists():
            print(f"  WARNING: {lang}.json not found, skipping")
            continue
        with open(path, "r", encoding="utf-8") as f:
            all_questions[lang] = json.load(f)
        print(f"  {lang}: {len(all_questions[lang])} questions")

    # Phase 1: Extract v-codes
    print("\nPhase 1: Extracting v-codes via regex...")
    vcode_counts = {}
    for lang, questions in all_questions.items():
        count = 0
        for q in questions:
            vcode = extract_vcode(q.get("id", ""), q.get("item", ""))
            if vcode:
                count += 1
        vcode_counts[lang] = count
        total = len(questions)
        pct = count / total * 100 if total else 0
        print(f"  {lang}: {count}/{total} ({pct:.1f}%)")

    # Phase 2: Build Q-to-v mapping from Greek
    print("\nPhase 2: Building Q-to-v mapping from Greek...")
    q_to_v = build_q_to_v_map(all_questions["ell"])
    print(f"  {len(q_to_v)} Q-codes mapped, covering "
          f"{sum(len(v) for v in q_to_v.values())} v-codes")

    # Phase 3: Build canonical question list from English
    print("\nPhase 3: Building canonical question list from English...")
    canonical_list = build_canonical_questions(all_questions["eng"], q_to_v)
    canonical_dict = {c["canonical_id"]: c for c in canonical_list}
    print(f"  {len(canonical_list)} canonical questions")

    # Phase 4 & 5: Match all languages
    print("\nPhase 4-5: Matching all languages via v-code + Q-code position...")
    matched_per_lang = {}
    for lang in ALL_LANGS:
        if lang == "eng" or lang not in all_questions:
            continue
        matched = match_language(
            lang, all_questions[lang], canonical_dict, q_to_v
        )
        matched_per_lang[lang] = matched
        total = len(canonical_dict)
        count = len(matched)
        pct = count / total * 100 if total else 0
        print(f"  {lang}: {count}/{total} ({pct:.1f}%)")

    # Phase 6: Gemini semantic matching for gaps
    print("\nPhase 6: Gemini semantic matching for remaining gaps...")
    matched_per_lang = run_gemini_matching(
        all_questions, matched_per_lang, canonical_dict, canonical_list
    )

    # Phase 7 & 8: Build and validate output
    print("\nPhase 7-8: Building output and validating...")
    output = build_output(canonical_list, matched_per_lang)

    # Write output
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nOutput written to {OUTPUT_FILE}")

    # Validate
    validate_output(output)


if __name__ == "__main__":
    main()
