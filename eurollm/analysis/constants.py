"""Shared constants for analysis modules."""

LANG_NAMES = {
    "bul": "Bulgarian", "ces": "Czech", "dan": "Danish", "deu": "German",
    "ell": "Greek", "eng": "English", "est": "Estonian", "fin": "Finnish",
    "fra": "French", "hrv": "Croatian", "hun": "Hungarian", "ita": "Italian",
    "lit": "Lithuanian", "lvs": "Latvian", "nld": "Dutch", "pol": "Polish",
    "por": "Portuguese", "ron": "Romanian", "slk": "Slovak", "slv": "Slovenian",
    "spa": "Spanish", "swe": "Swedish",
}

CULTURAL_CLUSTERS = {
    "Nordic": ["dan", "fin", "swe"],
    "Western": ["deu", "fra", "nld", "eng"],
    "Mediterranean": ["ita", "spa", "por", "ell"],
    "Central": ["ces", "hun", "pol", "slk", "slv"],
    "Baltic": ["est", "lit", "lvs"],
    "Southeast": ["bul", "hrv", "ron"],
}

LANG_TO_CLUSTER = {}
for _cluster, _langs in CULTURAL_CLUSTERS.items():
    for _lang in _langs:
        LANG_TO_CLUSTER[_lang] = _cluster

CLUSTER_COLORS = {
    "Nordic": "#1f77b4",
    "Western": "#ff7f0e",
    "Mediterranean": "#2ca02c",
    "Central": "#d62728",
    "Baltic": "#9467bd",
    "Southeast": "#8c564b",
}

MODEL_COLORS = {
    "hplt2c": "#1f77b4",
    "eurollm22b": "#ff7f0e",
    "gemma3_27b_pt": "#e377c2",
    "gemma3_27b_it": "#bcbd22",
    "qwen3235b": "#17becf",
}

MODEL_LABELS = {
    "hplt2c": "HPLT-2.15B",
    "eurollm22b": "EuroLLM-22B",
    "gemma3_27b_pt": "Gemma-3-27B",
    "gemma3_27b_it": "Gemma-3-27B-IT",
    "qwen3235b": "Qwen3-235B",
}

MODEL_MARKERS = {
    "hplt2c": "o",
    "eurollm22b": "^",
    "gemma3_27b_pt": "D",
    "gemma3_27b_it": "p",
    "qwen3235b": "h",
}

MODEL_SIZES = {
    "hplt2c": 80,
    "eurollm22b": 100,
    "gemma3_27b_pt": 90,
    "gemma3_27b_it": 90,
    "qwen3235b": 100,
}

# Smaller sizes for the combined UMAP plot (human stars are larger)
MODEL_SIZES_SMALL = {
    "hplt2c": 60,
    "eurollm22b": 70,
    "gemma3_27b_pt": 65,
    "gemma3_27b_it": 65,
    "qwen3235b": 70,
}

ORDINAL_TYPES = {"likert3", "likert4", "likert5", "likert10", "frequency"}

MODELS_EXCLUDE = {"gemma3_27b_it"}

# Inglehart-Welzel dimensions: question dicts for composite score computation
# flip=True means higher raw value → more traditional/survival, so we flip
# max_val is the scale maximum (needed for flipping); None = use data max
IW_TRADITIONAL_SECULAR = {
    "v63": {"flip": True,  "max_val": 11, "label": "Importance of God"},
    "v6":  {"flip": False, "max_val": None, "label": "Religion importance"},
    "v154":{"flip": False, "max_val": None, "label": "Abortion justifiable"},
    "v95": {"flip": False, "max_val": None, "label": "Obedience in children"},
    "v86": {"flip": True,  "max_val": 3,   "label": "Independence in children"},
}

IW_SURVIVAL_SELFEXPR = {
    "v39": {"flip": False, "max_val": None, "label": "Life satisfaction"},
    "v31": {"flip": True,  "max_val": 3,   "label": "Interpersonal trust"},
    "v82": {"flip": True,  "max_val": 6,   "label": "Gay couples as parents"},
    "v153":{"flip": False, "max_val": None, "label": "Homosexuality justifiable"},
    "v98": {"flip": True,  "max_val": 4,   "label": "Petition signing"},
}

DEEPDIVE_IW_QUESTIONS = [
    ("v63",  "How important is God in your life?"),
    ("v153", "Can homosexuality be justified?"),
    ("v82",  "Gay couples can be as good parents as others"),
]
DEEPDIVE_LANGS = ["eng", "fin", "pol", "ron"]  # Western, Nordic, Central, Southeast
