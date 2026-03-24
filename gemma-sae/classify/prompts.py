"""Classifier prompt template and JSON response parsing."""

import json

from analysis.constants import CONTENT_CATEGORIES

CLASSIFIER_SYSTEM = """You are a cultural content classifier. Given a short text completion, classify it on the following dimensions. Respond ONLY with valid JSON.

Content categories (pick exactly one):
- family_social: references to family, relationships, community, social bonds
- occupation_achievement: references to work, career, accomplishments, status
- personality_trait: references to personal qualities, character, traits
- physical_attribute: references to appearance, age, health, body
- spiritual_religious: references to religion, spirituality, faith, God
- emotional_state: references to feelings, emotions, mood, happiness
- material_practical: references to money, possessions, practical needs
- other: does not fit any of the above

Cultural dimensions (rate each 1-5):
- dim_indiv_collect: 1 = individualist (personal autonomy, uniqueness, self-reliance) to 5 = collectivist (group harmony, duty, interdependence)
- dim_trad_secular: 1 = traditional (religion, authority, family norms) to 5 = secular-rational (questioning authority, rational-legal norms)
- dim_surv_selfexpr: 1 = survival (economic/physical security focus) to 5 = self-expression (quality of life, tolerance, participation)

Respond with this exact JSON schema:
{"content_category": "<category>", "dim_indiv_collect": <1-5>, "dim_trad_secular": <1-5>, "dim_surv_selfexpr": <1-5>}"""


def make_classifier_prompt(completion_text: str, lang: str, template_id: str) -> str:
    """Build the user message for classification."""
    return (
        f"Language: {lang}\n"
        f"Prompt type: {template_id}\n"
        f"Completion: {completion_text}\n\n"
        f"Classify this completion."
    )


# ── JSON Schema for structured output (OpenAI / Anthropic) ───────

CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "content_category": {
            "type": "string",
            "enum": CONTENT_CATEGORIES,
        },
        "dim_indiv_collect": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
        },
        "dim_trad_secular": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
        },
        "dim_surv_selfexpr": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
        },
    },
    "required": [
        "content_category",
        "dim_indiv_collect",
        "dim_trad_secular",
        "dim_surv_selfexpr",
    ],
    "additionalProperties": False,
}


def parse_classification(raw_response: str) -> dict | None:
    """Parse classifier JSON output. Returns None on failure."""
    try:
        # Strip markdown code fences if present
        text = raw_response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        data = json.loads(text)

        cat = data.get("content_category", "")
        if cat not in CONTENT_CATEGORIES:
            return None

        dims = {}
        for key in ("dim_indiv_collect", "dim_trad_secular", "dim_surv_selfexpr"):
            val = int(data.get(key, 0))
            if not 1 <= val <= 5:
                return None
            dims[key] = val

        return {"content_category": cat, **dims}

    except (json.JSONDecodeError, TypeError, ValueError):
        return None
