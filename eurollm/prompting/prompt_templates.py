"""Prompt formatting for EVS survey questions.

Formats questions from questions.json into completion prompts for base LMs.
Supports forward and reversed option orderings for position-bias debiasing.
"""

import json
import re
from pathlib import Path

# Patterns to strip from question text
# Native-language "Answer with a number from 1 to {n}:" translations
# Used by scale_hint="cue" mode to replace the answer cue line
ANSWER_NUMBER_CUES = {
    "bul": "Отговорете с число от 1 до {n}",
    "ces": "Odpovězte číslem od 1 do {n}",
    "dan": "Svar med et tal fra 1 til {n}",
    "deu": "Antworten Sie mit einer Zahl von 1 bis {n}",
    "ell": "Απαντήστε με έναν αριθμό από 1 έως {n}",
    "eng": "Answer with a number from 1 to {n}",
    "est": "Vastake arvuga 1 kuni {n}",
    "fin": "Vastaa numerolla 1–{n}",
    "fra": "Répondez avec un nombre de 1 à {n}",
    "hrv": "Odgovorite brojem od 1 do {n}",
    "hun": "Válaszoljon egy számmal 1-től {n}-ig",
    "ita": "Rispondi con un numero da 1 a {n}",
    "lit": "Atsakykite skaičiumi nuo 1 iki {n}",
    "lvs": "Atbildiet ar skaitli no 1 līdz {n}",
    "nld": "Antwoord met een getal van 1 tot {n}",
    "pol": "Odpowiedz liczbą od 1 do {n}",
    "por": "Responda com um número de 1 a {n}",
    "ron": "Răspundeți cu un număr de la 1 la {n}",
    "slk": "Odpovedzte číslom od 1 do {n}",
    "slv": "Odgovorite s številko od 1 do {n}",
    "spa": "Responda con un número del 1 al {n}",
    "swe": "Svara med ett tal från 1 till {n}",
}

# Patterns to strip from question text
_INTERVIEWER_PATTERNS = [
    r"\bREAD\s*OUT\b[^.]*\.?",
    r"\bSHOWCARD\b[^.]*\.?",
    r"\bSHOW\s*CARD\b[^.]*\.?",
    r"\bUSE\s*CARD\b[^.]*\.?",
    r"Please use this card[^.]*\.?",
    r"Моля, посочете отговора си на тази карта\.?",  # Bulgarian "use this card"
    r"Моля, използвайте скалата[^.]*\.?",  # Bulgarian "use the scale"
]
_INTERVIEWER_RE = re.compile(
    "|".join(f"(?:{p})" for p in _INTERVIEWER_PATTERNS),
    re.IGNORECASE,
)

# Leading number prefix pattern: "1 - ", "1. ", "01 - ", etc.
_NUMBER_PREFIX_RE = re.compile(r"^\d+\s*[-–.]\s*")


def clean_label(label: str | None) -> str:
    """Clean an option label: None→'', strip leading number prefixes."""
    if label is None:
        return ""
    label = label.strip()
    label = _NUMBER_PREFIX_RE.sub("", label)
    # Strip trailing question mark (artifact from some extractions)
    label = label.rstrip("?")
    return label.strip()


def clean_text(text: str) -> str:
    """Strip interviewer instructions and embedded option lists from question text."""
    text = _INTERVIEWER_RE.sub("", text)
    # Strip embedded numbered option lists (extraction artifact):
    # matches "\n1 - Label\n2 - Label..." or "\n1. Label\n2. Label..." at end of text
    text = re.sub(r"(?:\n\d+\s*[-–.]\s*[^\n]*){2,}$", "", text)
    # Collapse multiple spaces/newlines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def _is_anchor_label(label: str) -> bool:
    """Check if a label is a meaningful anchor (not empty or digit-only)."""
    cleaned = clean_label(label)
    if not cleaned:
        return False
    if cleaned.isdigit():
        return False
    return True


def format_prompt(question: dict, lang: str, reverse: bool = False,
                  text_override: str | None = None) -> dict:
    """Format a question from questions.json into a completion prompt.

    Args:
        question: A question dict from questions.json with canonical_id,
                  response_type, translations, etc.
        lang: Language code (e.g. "eng", "deu").
        reverse: If True, reverse the option ordering for position-bias control.
        text_override: If provided, use this text instead of the question's
                       translation text (for rephrase experiments).

    Returns:
        dict with keys:
            prompt: str — the full prompt string
            valid_values: list[str] — valid response value strings
            value_map: dict — position→semantic value mapping
            is_likert10: bool — whether two-step "10" resolution is needed
    """
    trans = question["translations"][lang]
    text = clean_text(text_override) if text_override is not None else clean_text(trans["text"])
    answer_cue = trans.get("answer_cue", "Answer")
    options = trans["options"]
    is_likert10 = question["response_type"] == "likert10"
    n = len(options)

    if is_likert10:
        return _format_likert10(text, answer_cue, options, reverse)
    else:
        return _format_standard(text, answer_cue, options, reverse, n)


def _format_standard(
    text: str,
    answer_cue: str,
    options: list[dict],
    reverse: bool,
    n: int,
) -> dict:
    """Format a non-likert10 question with numbered options."""
    if reverse:
        # Reverse label order: position 1 gets label from value N, etc.
        reversed_options = list(reversed(options))
        value_map = {}
        lines = []
        for i, opt in enumerate(reversed_options, 1):
            label = clean_label(opt["label"])
            original_value = str(opt["value"])
            value_map[str(i)] = original_value
            if label:
                lines.append(f"{i}. {label}")
            else:
                lines.append(f"{i}.")
        valid_values = [str(i) for i in range(1, n + 1)]
    else:
        value_map = {}
        lines = []
        for opt in options:
            pos = opt["value"]
            label = clean_label(opt["label"])
            value_map[str(pos)] = str(pos)
            if label:
                lines.append(f"{pos}. {label}")
            else:
                lines.append(f"{pos}.")
        valid_values = [str(opt["value"]) for opt in options]

    # Trailing space after colon: model expects "Answer: " not "Answer:"
    # (without space, >50% of probability goes to \n or space tokens)
    prompt = f"{text}\n" + "\n".join(lines) + f"\n{answer_cue}: "

    return {
        "prompt": prompt,
        "valid_values": valid_values,
        "value_map": value_map,
        "is_likert10": False,
    }


def _format_likert10(
    text: str,
    answer_cue: str,
    options: list[dict],
    reverse: bool,
) -> dict:
    """Format a 1-10 scale question with anchors at endpoints."""
    # Extract anchors from value=1 and value=10 option labels
    opts_by_value = {opt["value"]: opt for opt in options}
    left_anchor = ""
    right_anchor = ""
    if 1 in opts_by_value and _is_anchor_label(opts_by_value[1].get("label", "")):
        left_anchor = clean_label(opts_by_value[1]["label"])
    if 10 in opts_by_value and _is_anchor_label(opts_by_value[10].get("label", "")):
        right_anchor = clean_label(opts_by_value[10]["label"])

    if reverse:
        # Swap anchors
        left_anchor, right_anchor = right_anchor, left_anchor
        value_map = {str(i): str(11 - i) for i in range(1, 11)}
    else:
        value_map = {str(i): str(i) for i in range(1, 11)}

    lines = []
    for i in range(1, 11):
        if i == 1 and left_anchor:
            lines.append(f"1. {left_anchor}")
        elif i == 10 and right_anchor:
            lines.append(f"10. {right_anchor}")
        else:
            lines.append(f"{i}.")

    # Trailing space after colon: model expects "Answer: " not "Answer:"
    prompt = f"{text}\n" + "\n".join(lines) + f"\n{answer_cue}: "

    valid_values = [str(i) for i in range(1, 11)]

    return {
        "prompt": prompt,
        "valid_values": valid_values,
        "value_map": value_map,
        "is_likert10": True,
    }


def format_prompt_permuted(question: dict, lang: str, permutation: list[int]) -> dict:
    """Format prompt with options in the given permutation order.

    Args:
        question: A question dict from questions.json.
        lang: Language code (e.g. "eng", "deu").
        permutation: list of 0-indexed positions into the original options list.
                     e.g., [2, 0, 3, 1] means show option[2] first, option[0] second, etc.
                     For likert10, indices are into values 1-10 (0=value 1, 9=value 10).

    Returns same dict as format_prompt: {prompt, valid_values, value_map, is_likert10}
    """
    trans = question["translations"][lang]
    text = clean_text(trans["text"])
    answer_cue = trans.get("answer_cue", "Answer")
    options = trans["options"]
    is_likert10 = question["response_type"] == "likert10"

    if is_likert10:
        return _format_likert10_permuted(text, answer_cue, options, permutation)
    else:
        return _format_standard_permuted(text, answer_cue, options, permutation)


def _format_standard_permuted(
    text: str,
    answer_cue: str,
    options: list[dict],
    permutation: list[int],
) -> dict:
    """Format a standard question with permuted option ordering."""
    n = len(options)
    permuted_options = [options[i] for i in permutation]

    value_map = {}
    lines = []
    for pos, opt in enumerate(permuted_options, 1):
        label = clean_label(opt["label"])
        original_value = str(opt["value"])
        value_map[str(pos)] = original_value
        if label:
            lines.append(f"{pos}. {label}")
        else:
            lines.append(f"{pos}.")

    valid_values = [str(i) for i in range(1, n + 1)]
    prompt = f"{text}\n" + "\n".join(lines) + f"\n{answer_cue}: "

    return {
        "prompt": prompt,
        "valid_values": valid_values,
        "value_map": value_map,
        "is_likert10": False,
    }


def _format_likert10_permuted(
    text: str,
    answer_cue: str,
    options: list[dict],
    permutation: list[int],
) -> dict:
    """Format a likert10 question with permuted value ordering."""
    opts_by_value = {opt["value"]: opt for opt in options}

    value_map = {}
    lines = []
    for pos_idx, orig_idx in enumerate(permutation):
        pos = pos_idx + 1
        orig_value = orig_idx + 1
        value_map[str(pos)] = str(orig_value)

        opt = opts_by_value.get(orig_value, {})
        label = opt.get("label", "")
        if _is_anchor_label(label):
            lines.append(f"{pos}. {clean_label(label)}")
        else:
            lines.append(f"{pos}.")

    valid_values = [str(i) for i in range(1, 11)]
    prompt = f"{text}\n" + "\n".join(lines) + f"\n{answer_cue}: "

    return {
        "prompt": prompt,
        "valid_values": valid_values,
        "value_map": value_map,
        "is_likert10": True,
    }


def _format_option_line(pos: int, label: str, opt_format: str) -> str:
    """Format a single option line according to the given format style."""
    if opt_format == "numbered_paren":
        return f"{pos}) {label}" if label else f"{pos})"
    elif opt_format == "bullet":
        return f"- {label}" if label else f"- {pos}"
    else:  # "numbered_dot" (default)
        return f"{pos}. {label}" if label else f"{pos}."


def _build_prompt(
    text: str,
    opt_lines: list[str],
    cue_style: str,
    answer_cue: str,
    scale_hint: str,
    embed_style: str,
    n: int,
    lang: str = "eng",
) -> str:
    """Assemble prompt from components with configurable formatting.

    Args:
        scale_hint: "english" (prepend "On a scale of 1 to N:"),
                    "cue" (native-language answer cue replaces answer_cue),
                    "none" (no hint).
    """
    parts = []

    if scale_hint == "english":
        parts.append(f"On a scale of 1 to {n}:")

    if embed_style == "inline":
        opts_str = ", ".join(opt_lines)
        parts.append(f"{text} ({opts_str})")
    else:  # "separate" (default)
        parts.append(text)
        parts.extend(opt_lines)

    body = "\n".join(parts)

    if scale_hint == "cue":
        # Native-language answer cue with number range
        cue_text = ANSWER_NUMBER_CUES.get(lang, ANSWER_NUMBER_CUES["eng"])
        return body + f"\n{cue_text.format(n=n)}: "
    elif cue_style == "answer":
        return body + "\nAnswer: "
    elif cue_style == "none":
        return body + "\n"
    else:  # "lang" (default)
        return body + f"\n{answer_cue}: "


def format_prompt_custom(
    question: dict,
    lang: str,
    permutation: list[int],
    cue_style: str = "lang",
    opt_format: str = "numbered_dot",
    scale_hint: str = "none",
    embed_style: str = "separate",
) -> dict:
    """Format prompt with configurable format dimensions for optimization.

    Args:
        question: A question dict from questions.json.
        lang: Language code.
        permutation: 0-indexed option ordering.
        cue_style: "answer" ("Answer: "), "none" (newline only), "lang" (translated cue).
        opt_format: "numbered_dot" ("1. Label"), "numbered_paren" ("1) Label"),
                    "bullet" ("- Label").
        scale_hint: "english" (prepend "On a scale of 1 to N:"),
                    "cue" (native-language answer cue), "none" (no hint).
        embed_style: "separate" (options on own lines) or "inline" (in parentheses).

    Returns same dict as format_prompt: {prompt, valid_values, value_map, is_likert10}
    """
    trans = question["translations"][lang]
    text = clean_text(trans["text"])
    answer_cue = trans.get("answer_cue", "Answer")
    options = trans["options"]
    is_likert10 = question["response_type"] == "likert10"
    n = 10 if is_likert10 else len(options)

    if is_likert10:
        opts_by_value = {opt["value"]: opt for opt in options}
        value_map = {}
        opt_lines = []
        for pos_idx, orig_idx in enumerate(permutation):
            pos = pos_idx + 1
            orig_value = orig_idx + 1
            value_map[str(pos)] = str(orig_value)
            opt = opts_by_value.get(orig_value, {})
            label = opt.get("label", "")
            anchor = clean_label(label) if _is_anchor_label(label) else ""
            opt_lines.append(_format_option_line(pos, anchor, opt_format))
    else:
        permuted_options = [options[i] for i in permutation]
        value_map = {}
        opt_lines = []
        for pos, opt in enumerate(permuted_options, 1):
            label = clean_label(opt["label"])
            value_map[str(pos)] = str(opt["value"])
            opt_lines.append(_format_option_line(pos, label, opt_format))

    valid_values = [str(i) for i in range(1, n + 1)]
    prompt = _build_prompt(text, opt_lines, cue_style, answer_cue,
                           scale_hint, embed_style, n, lang=lang)

    return {
        "prompt": prompt,
        "valid_values": valid_values,
        "value_map": value_map,
        "is_likert10": is_likert10,
    }


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    questions_path = PROJECT_ROOT / "data" / "questions.json"
    with open(questions_path) as f:
        data = json.load(f)

    questions = data["questions"]

    # Find one example per response type
    type_examples = {}
    for q in questions:
        rt = q["response_type"]
        if rt not in type_examples and "eng" in q["translations"]:
            type_examples[rt] = q
        if len(type_examples) >= 6:
            break

    print(f"Found {len(type_examples)} response types: {list(type_examples.keys())}\n")
    print("=" * 70)

    for rt, q in type_examples.items():
        print(f"\n{'='*70}")
        print(f"Response type: {rt} | Question: {q['canonical_id']}")
        print(f"Option count: {q['option_count']}")
        print(f"{'='*70}")

        # Forward
        fwd = format_prompt(q, "eng", reverse=False)
        print(f"\n--- FORWARD ---")
        print(fwd["prompt"])
        print(f"\nvalid_values: {fwd['valid_values']}")
        print(f"value_map: {fwd['value_map']}")
        print(f"is_likert10: {fwd['is_likert10']}")

        # Reversed
        rev = format_prompt(q, "eng", reverse=True)
        print(f"\n--- REVERSED ---")
        print(rev["prompt"])
        print(f"\nvalid_values: {rev['valid_values']}")
        print(f"value_map: {rev['value_map']}")
        print(f"is_likert10: {rev['is_likert10']}")

    # Also show a non-English example with anchors
    print(f"\n{'='*70}")
    print("NON-ENGLISH EXAMPLE: Bulgarian likert10 with anchors (v102)")
    print(f"{'='*70}")
    for q in questions:
        if q["canonical_id"] == "v102" and "bul" in q["translations"]:
            fwd = format_prompt(q, "bul", reverse=False)
            print(f"\n--- FORWARD (Bulgarian) ---")
            print(fwd["prompt"])
            rev = format_prompt(q, "bul", reverse=True)
            print(f"\n--- REVERSED (Bulgarian) ---")
            print(rev["prompt"])
            break
