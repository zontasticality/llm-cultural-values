"""Shared constants: languages, cultural clusters, model metadata."""

# ── Languages ────────────────────────────────────────────────────

LANG_NAMES = {
    # European (22)
    "bul": "Bulgarian", "ces": "Czech", "dan": "Danish", "deu": "German",
    "ell": "Greek", "eng": "English", "est": "Estonian", "fin": "Finnish",
    "fra": "French", "hrv": "Croatian", "hun": "Hungarian", "ita": "Italian",
    "lit": "Lithuanian", "lvs": "Latvian", "nld": "Dutch", "pol": "Polish",
    "por": "Portuguese", "ron": "Romanian", "slk": "Slovak", "slv": "Slovenian",
    "spa": "Spanish", "swe": "Swedish",
    # Expanded (5) -- Gemma 3 only
    "zho": "Chinese", "jpn": "Japanese", "ara": "Arabic",
    "hin": "Hindi", "tur": "Turkish",
}

EU_LANGS = [
    "bul", "ces", "dan", "deu", "ell", "eng", "est", "fin",
    "fra", "hrv", "hun", "ita", "lit", "lvs", "nld", "pol",
    "por", "ron", "slk", "slv", "spa", "swe",
]

EXPANDED_LANGS = ["zho", "jpn", "ara", "hin", "tur"]

ALL_LANGS = EU_LANGS + EXPANDED_LANGS

PILOT_LANGS = ["eng", "fin", "pol", "ron", "zho"]
PILOT_TEMPLATES = ["self_concept", "values"]

# ── Cultural Clusters (Inglehart-Welzel) ─────────────────────────

CULTURAL_CLUSTERS = {
    "Protestant Europe": ["dan", "fin", "swe", "nld", "deu"],
    "Catholic Europe": ["fra", "ita", "spa", "por", "ces", "hun", "pol", "slk", "slv", "hrv"],
    "English-speaking": ["eng"],
    "Orthodox": ["bul", "ron", "ell"],
    "Baltic": ["est", "lit", "lvs"],
    "East Asian": ["zho", "jpn"],
    "South Asian": ["hin"],
    "Middle Eastern": ["ara", "tur"],
}

LANG_TO_CLUSTER = {}
for _cluster, _langs in CULTURAL_CLUSTERS.items():
    for _lang in _langs:
        LANG_TO_CLUSTER[_lang] = _cluster

CLUSTER_COLORS = {
    "Protestant Europe": "#1f77b4",
    "Catholic Europe": "#2ca02c",
    "English-speaking": "#ff7f0e",
    "Orthodox": "#d62728",
    "Baltic": "#9467bd",
    "East Asian": "#8c564b",
    "South Asian": "#e377c2",
    "Middle Eastern": "#7f7f7f",
}

# ── Models ───────────────────────────────────────────────────────

MODELS = {
    "gemma3_27b_pt": {
        "hf_id": "google/gemma-3-27b-pt",
        "family": "gemma3",
        "multilingual": True,
        "langs": ALL_LANGS,
        "temperature": 1.0,
    },
    "gemma3_12b_pt": {
        "hf_id": "google/gemma-3-12b-pt",
        "family": "gemma3",
        "multilingual": True,
        "langs": ALL_LANGS,
        "temperature": 1.0,
    },
    "eurollm22b": {
        "hf_id": "utter-project/EuroLLM-22B-2512",
        "family": "eurollm",
        "multilingual": True,
        "langs": EU_LANGS,
        "temperature": 1.0,
    },
}

# HPLT monolingual models -- one per EU language
for _lang in EU_LANGS:
    MODELS[f"hplt2c_{_lang}"] = {
        "hf_id": f"HPLT/hplt2c_{_lang}_checkpoints",
        "family": "hplt2c",
        "multilingual": False,
        "langs": [_lang],
        "temperature": 0.8,
    }

MODEL_COLORS = {
    "hplt2c": "#1f77b4",
    "eurollm": "#ff7f0e",
    "gemma3": "#e377c2",
}

MODEL_LABELS = {
    "gemma3_27b_pt": "Gemma-3-27B",
    "gemma3_12b_pt": "Gemma-3-12B",
    "eurollm22b": "EuroLLM-22B",
    "hplt2c": "HPLT-2.15B",
}

MODEL_MARKERS = {
    "hplt2c": "o",
    "eurollm": "^",
    "gemma3": "D",
}

# ── Trimming ────────────────────────────────────────────────────

TRIMMED_VARIANT_MIN = 100  # variant_idx threshold for trimmed (no trailing space) prompts

# ── EVS Country Names (for scatter plot labels) ─────────────────

EVS_COUNTRY_NAMES = {
    "bul": "Bulgaria", "ces": "Czechia", "dan": "Denmark", "deu": "Germany",
    "ell": "Greece", "eng": "England", "est": "Estonia", "fin": "Finland",
    "fra": "France", "hrv": "Croatia", "hun": "Hungary", "ita": "Italy",
    "lit": "Lithuania", "lvs": "Latvia", "nld": "Netherlands", "pol": "Poland",
    "por": "Portugal", "ron": "Romania", "slk": "Slovakia", "slv": "Slovenia",
    "spa": "Spain", "swe": "Sweden",
}

# ── Inglehart-Welzel Dimension Weights ─────────────────────────

IW_TRADITIONAL_SECULAR = {
    "v63":  {"flip": True,  "max_val": 11, "label": "Importance of God"},
    "v6":   {"flip": False, "max_val": None, "label": "Religion importance"},
    "v154": {"flip": False, "max_val": None, "label": "Abortion justifiable"},
    "v95":  {"flip": False, "max_val": None, "label": "Obedience in children"},
    "v86":  {"flip": True,  "max_val": 3,   "label": "Independence in children"},
}

IW_SURVIVAL_SELFEXPR = {
    "v39":  {"flip": False, "max_val": None, "label": "Life satisfaction"},
    "v31":  {"flip": True,  "max_val": 3,   "label": "Interpersonal trust"},
    "v82":  {"flip": True,  "max_val": 6,   "label": "Gay couples as parents"},
    "v153": {"flip": False, "max_val": None, "label": "Homosexuality justifiable"},
    "v98":  {"flip": True,  "max_val": 4,   "label": "Petition signing"},
}

# ── Logprob Column Mapping ─────────────────────────────────────

LOGPROB_DIMS = {
    "dim_indiv_collect": "dim_ic_probs",
    "dim_trad_secular":  "dim_ts_probs",
    "dim_surv_selfexpr": "dim_ss_probs",
}

# ── Content Categories ───────────────────────────────────────────

CONTENT_CATEGORIES = [
    "family_social",
    "occupation_achievement",
    "personality_trait",
    "physical_attribute",
    "spiritual_religious",
    "emotional_state",
    "material_practical",
    "other",
]

# ── Cultural Dimensions ─────────────────────────────────────────

CULTURAL_DIMENSIONS = [
    "dim_indiv_collect",   # 1=individualist, 5=collectivist
    "dim_trad_secular",    # 1=traditional, 5=secular-rational
    "dim_surv_selfexpr",   # 1=survival, 5=self-expression
]
