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
    "qwen2572b": "#2ca02c",
    "gemma3_27b_pt": "#e377c2",
    "gemma3_27b_it": "#bcbd22",
    "qwen3235b": "#17becf",
}

MODEL_LABELS = {
    "hplt2c": "HPLT-2.15B",
    "eurollm22b": "EuroLLM-22B",
    "qwen2572b": "Qwen2.5-72B",
    "gemma3_27b_pt": "Gemma-3-27B",
    "gemma3_27b_it": "Gemma-3-27B-IT",
    "qwen3235b": "Qwen3-235B",
}

MODEL_MARKERS = {
    "hplt2c": "o",
    "eurollm22b": "^",
    "qwen2572b": "s",
    "gemma3_27b_pt": "D",
    "gemma3_27b_it": "p",
    "qwen3235b": "h",
}

MODEL_SIZES = {
    "hplt2c": 80,
    "eurollm22b": 100,
    "qwen2572b": 90,
    "gemma3_27b_pt": 90,
    "gemma3_27b_it": 90,
    "qwen3235b": 100,
}

# Smaller sizes for the combined UMAP plot (human stars are larger)
MODEL_SIZES_SMALL = {
    "hplt2c": 60,
    "eurollm22b": 70,
    "qwen2572b": 65,
    "gemma3_27b_pt": 65,
    "gemma3_27b_it": 65,
    "qwen3235b": 70,
}

ORDINAL_TYPES = {"likert3", "likert4", "likert5", "likert10", "frequency"}
