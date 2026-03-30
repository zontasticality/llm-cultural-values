"""Classifier prompt template and JSON response parsing."""

import json

from analysis.constants import CONTENT_CATEGORIES

CLASSIFIER_SYSTEM = """You are a cultural content classifier. Given a short text completion produced by a language model, classify it on the following dimensions. Respond ONLY with valid JSON.

Content categories (pick exactly one):
- family_social: references to family, relationships, community, social bonds
- occupation_achievement: references to work, career, accomplishments, professional status
- personality_trait: references to personal qualities, character, dispositions, values, life philosophy
- physical_attribute: references to the person's own appearance, age, or bodily characteristics (NOT valuing health in the abstract — that is material_practical)
- spiritual_religious: references to religion, spirituality, faith, God, prayer (NOT nature appreciation or secular philosophy)
- emotional_state: references to feelings, emotions, mood — the completion is primarily ABOUT an emotional experience
- material_practical: references to money, possessions, practical needs, health as a priority, safety, security, technical tasks
- other: does not fit any of the above, OR the completion is off-topic noise (e.g., encyclopedia entries, product descriptions, code, text in the wrong language that has no cultural self-description content)

IMPORTANT classification guidelines:
- Classify based on the VALUES OR STANCE the speaker expresses, not just what the text mentions. A completion about family conflict where the speaker rejects family obligations is individualist, not collectivist.
- If the completion is clearly off-topic, noise, or not a meaningful self-description (e.g., Wikipedia articles, technical troubleshooting, product listings, text in the wrong language with no cultural content), classify as "other" and set ALL dimension scores to 3.
- "physical_attribute" means bodily self-description ("I am tall", "I have brown eyes"), NOT abstract statements about valuing health or safety — those are "material_practical".
- "spiritual_religious" requires explicit religious or spiritual content. Nature appreciation, environmentalism, or secular philosophy should be "personality_trait" or "other", not "spiritual_religious".
- Corporate advertising, product descriptions, celebrity gossip, and web-scraped boilerplate should be "other" — they are not genuine cultural self-descriptions.

Cultural dimensions (rate each 1-5):
- dim_indiv_collect: 1 = individualist, 5 = collectivist. Reserve 1 for EXPLICIT rejection of group obligation or strong autonomy assertions ("I don't need anyone", "I make my own rules"). Reserve 5 for EXPLICIT group-first framing ("our unity gives us strength", "family above all"). Use 2 for individual-focused content (personal achievement, self-description). Use 4 for group-oriented content (family care, community, "we" framing). Use 3 when neither axis is clearly expressed. Do NOT score 1 simply because the text uses "I" — first-person narration is the default, not a signal of individualism.
- dim_trad_secular: 1 = traditional, 5 = secular-rational. Traditional signals include: religion, prayer, God, divine authority, family duty/obligation, folk proverbs, deference to elders, agrarian/rural nostalgia, traditional gender roles. Secular signals include: questioning authority, gender equality, scientific reasoning, rational self-improvement, progressive social values. Score 3 ONLY when the content is genuinely neutral (e.g., factual statements, technical content, generic descriptions). If the content contains ANY traditional or secular signal, move off 3.
- dim_surv_selfexpr: 1 = survival (economic/physical security focus, material needs) to 5 = self-expression (quality of life, creativity, tolerance, self-actualization). Reserve 5 for strongly self-actualizing content; use 4 for moderately self-expressive content.

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
