#!/usr/bin/env python3
"""Analysis pipeline for LLM cultural values extraction results.

Usage:
    python eurollm/analyze.py quality     # Quality report + P_valid heatmap
    python eurollm/analyze.py bias        # Position bias analysis + figure
    python eurollm/analyze.py summary     # Per-question summary stats
    python eurollm/analyze.py distance    # 44×44 JSD matrix + clustered heatmap
    python eurollm/analyze.py pca         # PCA cultural map
    python eurollm/analyze.py examples    # Example distribution bar charts
    python eurollm/analyze.py rephrase    # Prompt sensitivity experiment analysis
    python eurollm/analyze.py all         # Run everything (except rephrase)
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import jensenshannon, squareform
from scipy.stats import entropy
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE


# ── Constants ────────────────────────────────────────────────────────────────

CULTURAL_CLUSTERS = {
    "Nordic": ["dan", "fin", "swe"],
    "Western": ["deu", "fra", "nld", "eng"],
    "Mediterranean": ["ita", "spa", "por", "ell"],
    "Central": ["ces", "hun", "pol", "slk", "slv"],
    "Baltic": ["est", "lit", "lvs"],
    "Southeast": ["bul", "hrv", "ron"],
}

LANG_TO_CLUSTER = {}
for cluster, langs in CULTURAL_CLUSTERS.items():
    for lang in langs:
        LANG_TO_CLUSTER[lang] = cluster

LANG_NAMES = {
    "bul": "Bulgarian", "ces": "Czech", "dan": "Danish", "deu": "German",
    "ell": "Greek", "eng": "English", "est": "Estonian", "fin": "Finnish",
    "fra": "French", "hrv": "Croatian", "hun": "Hungarian", "ita": "Italian",
    "lit": "Lithuanian", "lvs": "Latvian", "nld": "Dutch", "pol": "Polish",
    "por": "Portuguese", "ron": "Romanian", "slk": "Slovak", "slv": "Slovenian",
    "spa": "Spanish", "swe": "Swedish",
}

CLUSTER_COLORS = {
    "Nordic": "#1f77b4",
    "Western": "#ff7f0e",
    "Mediterranean": "#2ca02c",
    "Central": "#d62728",
    "Baltic": "#9467bd",
    "Southeast": "#8c564b",
}

EXAMPLE_QUESTIONS = [
    ("v198", "EU enlargement: further or too far?"),
    ("v184", "Impact of immigrants on development"),
    ("v206", "Should gov't monitor emails?"),
    ("v102", "Left-right political self-placement"),
    ("v64", "How often do you pray?"),
    ("v68", "Household chores for successful marriage"),
]
EXAMPLE_LANGS = ["eng", "deu", "fra", "ita", "pol", "fin"]

ISSUE_TYPE_COLORS = {
    "battery_stem": "#e41a1c",
    "card_reference": "#377eb8",
    "read_out": "#4daf4a",
    "control_clean": "#984ea3",
}


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_all_results(results_dir: Path) -> pd.DataFrame:
    """Load all parquet files from results_dir, adding model_type and lang columns."""
    frames = []
    for path in sorted(results_dir.glob("*.parquet")):
        if "rephrase" in path.stem:
            continue  # skip rephrase experiment files
        stem = path.stem  # e.g. "hplt2c_eng" or "eurollm22b_eng"
        parts = stem.split("_", 1)
        if len(parts) != 2:
            print(f"  Skipping {path.name}: unexpected filename format")
            continue
        model_type, lang = parts
        df = pd.read_parquet(path)
        df["model_type"] = model_type
        df["lang"] = lang
        frames.append(df)
    if not frames:
        print(f"ERROR: No parquet files found in {results_dir}")
        sys.exit(1)
    combined = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(combined)} rows from {len(frames)} files")
    return combined


def load_question_metadata(questions_path: Path) -> dict:
    """Load questions.json, return dict keyed by canonical_id."""
    with open(questions_path) as f:
        data = json.load(f)
    meta = {}
    for q in data["questions"]:
        qid = q["canonical_id"]
        meta[qid] = {
            "response_type": q["response_type"],
            "option_count": q["option_count"],
        }
    return meta


def apply_quality_filter(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter low-quality questions and flag low-quality models.

    Returns (filtered_df, report_df with flagged info).
    """
    # Per-question P_valid stats across model-lang pairs
    p_valid_avg = df.groupby(["model_type", "lang", "question_id"]).agg(
        p_valid=("p_valid_forward", "mean")
    ).reset_index()

    # Questions where P_valid < 0.10 for >50% of model-lang pairs
    q_stats = p_valid_avg.groupby("question_id").agg(
        frac_low=("p_valid", lambda x: (x < 0.10).mean()),
        median_pvalid=("p_valid", "median"),
    )
    bad_questions = set(q_stats[q_stats["frac_low"] > 0.50].index)

    # Models with median P_valid < 0.20
    model_stats = p_valid_avg.groupby(["model_type", "lang"]).agg(
        median_pvalid=("p_valid", "median")
    ).reset_index()
    flagged_models = model_stats[model_stats["median_pvalid"] < 0.20]

    if bad_questions:
        print(f"  Filtered {len(bad_questions)} low-quality questions: {bad_questions}")
        filtered = df[~df["question_id"].isin(bad_questions)]
    else:
        print("  No questions filtered (all have sufficient P_valid)")
        filtered = df

    if len(flagged_models) > 0:
        print(f"  WARNING: {len(flagged_models)} models with median P_valid < 0.20:")
        for _, row in flagged_models.iterrows():
            print(f"    {row['model_type']}_{row['lang']}: {row['median_pvalid']:.3f}")
    else:
        print("  No models flagged for low quality")

    report = pd.DataFrame({
        "bad_questions": [list(bad_questions)],
        "n_flagged_models": [len(flagged_models)],
    })
    return filtered, report


# ── Analysis Functions ───────────────────────────────────────────────────────

def compute_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per (model_type, lang) quality summary."""
    # Average P_valid per question first, then per model-lang
    q_pvalid = df.groupby(["model_type", "lang", "question_id"]).agg(
        pv_fwd=("p_valid_forward", "first"),
        pv_rev=("p_valid_reversed", "first"),
    ).reset_index()
    q_pvalid["p_valid"] = (q_pvalid["pv_fwd"] + q_pvalid["pv_rev"]) / 2

    report = q_pvalid.groupby(["model_type", "lang"]).agg(
        mean_pvalid=("p_valid", "mean"),
        median_pvalid=("p_valid", "median"),
        frac_below_010=("p_valid", lambda x: (x < 0.10).mean()),
        n_questions=("question_id", "nunique"),
    ).reset_index()

    print("\n=== Quality Report ===")
    print(f"{'Model':<14} {'Lang':<5} {'Mean P_v':>8} {'Med P_v':>8} "
          f"{'%<0.10':>7} {'N_Q':>5}")
    print("-" * 52)
    for _, r in report.sort_values(["model_type", "lang"]).iterrows():
        print(f"{r['model_type']:<14} {r['lang']:<5} {r['mean_pvalid']:>8.3f} "
              f"{r['median_pvalid']:>8.3f} {r['frac_below_010']:>6.1%} "
              f"{r['n_questions']:>5}")
    return report


def compute_bias_report(df: pd.DataFrame, questions: dict) -> pd.DataFrame:
    """Position bias analysis per question."""
    # One row per (model_type, lang, question_id) with bias magnitude
    bias = df.groupby(["model_type", "lang", "question_id"]).agg(
        position_bias=("position_bias_magnitude", "first"),
    ).reset_index()

    # Add response_type
    bias["response_type"] = bias["question_id"].map(
        lambda qid: questions.get(qid, {}).get("response_type", "unknown")
    )

    print("\n=== Position Bias Report ===")
    for mt in sorted(bias["model_type"].unique()):
        sub = bias[bias["model_type"] == mt]
        print(f"\n{mt}:")
        print(f"  Mean bias:   {sub['position_bias'].mean():.3f}")
        print(f"  Median bias: {sub['position_bias'].median():.3f}")
        print(f"  Std bias:    {sub['position_bias'].std():.3f}")

    print("\nBy response type:")
    for rt in sorted(bias["response_type"].unique()):
        sub = bias[bias["response_type"] == rt]
        print(f"  {rt:<14} mean={sub['position_bias'].mean():.3f}  "
              f"median={sub['position_bias'].median():.3f}")

    # Statistical test: EuroLLM vs HPLT
    euro = bias[bias["model_type"] == "eurollm22b"]["position_bias"]
    hplt = bias[bias["model_type"] == "hplt2c"]["position_bias"]
    if len(euro) > 0 and len(hplt) > 0:
        from scipy.stats import mannwhitneyu
        stat, pval = mannwhitneyu(euro, hplt, alternative="two-sided")
        print(f"\nMann-Whitney U test (EuroLLM vs HPLT): U={stat:.0f}, p={pval:.2e}")
        if euro.median() > hplt.median():
            print("  → EuroLLM-22B shows MORE position bias than HPLT-2.15B")
        else:
            print("  → HPLT-2.15B shows MORE position bias than EuroLLM-22B")

    return bias


def compute_summary_stats(df: pd.DataFrame, questions: dict) -> pd.DataFrame:
    """Per-question summary: expected value, entropy, mode, concentration."""
    rows = []
    for (mt, lang, qid), grp in df.groupby(["model_type", "lang", "question_id"]):
        probs = grp["prob_averaged"].values
        values = grp["response_value"].values

        # Normalize to distribution
        total = probs.sum()
        if total > 0:
            probs_norm = probs / total
        else:
            probs_norm = np.ones_like(probs) / len(probs)

        # Entropy (in nats)
        h = entropy(probs_norm)

        # Mode
        mode_val = values[np.argmax(probs_norm)]

        # Concentration (max prob)
        concentration = probs_norm.max()

        # Expected value (only for ordinal types)
        rtype = questions.get(qid, {}).get("response_type", "unknown")
        ordinal_types = {"likert3", "likert4", "likert5", "likert10", "frequency"}
        if rtype in ordinal_types:
            expected = np.dot(values, probs_norm)
        else:
            expected = np.nan

        rows.append({
            "model_type": mt, "lang": lang, "question_id": qid,
            "response_type": rtype,
            "expected_value": expected, "entropy": h,
            "mode": mode_val, "concentration": concentration,
        })

    summary = pd.DataFrame(rows)

    print("\n=== Summary Statistics ===")
    print(f"Total entries: {len(summary)}")
    for mt in sorted(summary["model_type"].unique()):
        sub = summary[summary["model_type"] == mt]
        print(f"\n{mt}:")
        print(f"  Mean entropy:       {sub['entropy'].mean():.3f}")
        print(f"  Mean concentration: {sub['concentration'].mean():.3f}")
        ordinal = sub[sub["response_type"].isin(
            {"likert3", "likert4", "likert5", "likert10", "frequency"})]
        if len(ordinal) > 0:
            print(f"  Mean E[X] (ordinal): {ordinal['expected_value'].mean():.3f}")

    return summary


def compute_jsd_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Compute 44×44 JSD distance matrix between all model-lang pairs."""
    # Build per-(model_type, lang, question_id) distributions
    pairs = sorted(df.groupby(["model_type", "lang"]).groups.keys())
    labels = [f"{mt}_{lang}" for mt, lang in pairs]
    n = len(labels)

    # Pre-compute distributions per pair
    distributions = {}
    for mt, lang in pairs:
        key = (mt, lang)
        sub = df[(df["model_type"] == mt) & (df["lang"] == lang)]
        distributions[key] = {}
        for qid, grp in sub.groupby("question_id"):
            probs = grp.sort_values("response_value")["prob_averaged"].values
            total = probs.sum()
            if total > 0:
                distributions[key][qid] = probs / total
            else:
                distributions[key][qid] = np.ones(len(probs)) / len(probs)

    # Compute pairwise JSD
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            key_i = pairs[i]
            key_j = pairs[j]
            shared_qs = set(distributions[key_i].keys()) & set(distributions[key_j].keys())
            if not shared_qs:
                matrix[i, j] = matrix[j, i] = np.nan
                continue
            jsds = []
            for qid in shared_qs:
                p = distributions[key_i][qid]
                q = distributions[key_j][qid]
                if len(p) == len(q):
                    jsds.append(jensenshannon(p, q))
            if jsds:
                matrix[i, j] = matrix[j, i] = np.nanmean(jsds)
            else:
                matrix[i, j] = matrix[j, i] = np.nan

    print(f"\n=== JSD Matrix ===")
    print(f"Shape: {n}×{n}, {n} model-lang pairs")
    valid = matrix[np.triu_indices(n, k=1)]
    valid = valid[~np.isnan(valid)]
    print(f"Mean JSD: {valid.mean():.4f}, Median: {np.median(valid):.4f}, "
          f"Range: [{valid.min():.4f}, {valid.max():.4f}]")

    return matrix, labels


def compute_pca_map(summary_df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """PCA cultural map from ordinal expected values."""
    ordinal = summary_df[summary_df["response_type"].isin(
        {"likert3", "likert4", "likert5", "likert10", "frequency"}
    )].copy()

    # Pivot: rows = model-lang pairs, columns = question_ids
    ordinal["pair"] = ordinal["model_type"] + "_" + ordinal["lang"]
    pivot = ordinal.pivot_table(
        index="pair", columns="question_id", values="expected_value", aggfunc="first"
    )

    labels = list(pivot.index)
    X = pivot.values

    # Impute missing with column means
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    # PCA
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_imputed)

    print(f"\n=== PCA Cultural Map ===")
    print(f"Features: {X_imputed.shape[1]} ordinal questions")
    print(f"Explained variance: PC1={pca.explained_variance_ratio_[0]:.1%}, "
          f"PC2={pca.explained_variance_ratio_[1]:.1%}")

    return coords, labels


# ── Visualization Functions ──────────────────────────────────────────────────

def plot_pvalid_heatmap(report: pd.DataFrame, figures_dir: Path):
    """Heatmap of mean P_valid: model types × languages."""
    pivot = report.pivot_table(
        index="model_type", columns="lang", values="mean_pvalid"
    )
    # Sort languages alphabetically and rename to full names
    pivot = pivot[sorted(pivot.columns)]
    pivot.columns = [LANG_NAMES.get(c, c) for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(18, 4))
    sns.heatmap(
        pivot, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0.5, vmax=1.0,
        ax=ax, linewidths=0.5, cbar_kws={"label": "Mean P_valid"}
    )
    ax.set_title("P_valid by Model Type and Language")
    ax.set_xlabel("Language")
    ax.set_ylabel("Model")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    out = figures_dir / "pvalid_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


def plot_bias_distribution(bias: pd.DataFrame, figures_dir: Path):
    """Violin plots of position bias by model type and response type."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) By model_type
    sns.violinplot(data=bias, x="model_type", y="position_bias",
                   ax=axes[0], inner="box", cut=0)
    axes[0].set_title("Position Bias by Model Type")
    axes[0].set_xlabel("Model Type")
    axes[0].set_ylabel("Position Bias Magnitude")

    # (b) By response_type
    order = sorted(bias["response_type"].unique())
    sns.violinplot(data=bias, x="response_type", y="position_bias",
                   ax=axes[1], inner="box", order=order, cut=0)
    axes[1].set_title("Position Bias by Response Type")
    axes[1].set_xlabel("Response Type")
    axes[1].set_ylabel("Position Bias Magnitude")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    out = figures_dir / "position_bias.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


def plot_jsd_heatmap(matrix: np.ndarray, labels: list[str], figures_dir: Path):
    """Clustered heatmap of JSD distances."""
    # Replace NaN with max for clustering
    mat_filled = np.nan_to_num(matrix, nan=np.nanmax(matrix))
    np.fill_diagonal(mat_filled, 0)

    # Color labels by model type
    model_types = [l.split("_")[0] for l in labels]
    model_colors = {"hplt2c": "#1f77b4", "eurollm22b": "#ff7f0e", "qwen2572b": "#2ca02c"}
    row_colors = [model_colors.get(mt, "#999999") for mt in model_types]

    # Short labels with full language names
    model_prefixes = {"hplt2c": "H", "eurollm22b": "E", "qwen2572b": "Q"}

    def _jsd_label(l):
        parts = l.split("_", 1)
        mt, lang = parts[0], parts[1]
        prefix = model_prefixes.get(mt, mt[0].upper())
        name = LANG_NAMES.get(lang, lang)
        return f"{prefix}:{name}"
    short_labels = [_jsd_label(l) for l in labels]

    df_matrix = pd.DataFrame(mat_filled, index=short_labels, columns=short_labels)

    # Pre-compute linkage from condensed distance to avoid seaborn warning
    condensed = squareform(mat_filled)
    row_linkage = linkage(condensed, method="average")
    col_linkage = row_linkage

    g = sns.clustermap(
        df_matrix, cmap="viridis_r", figsize=(14, 12),
        row_colors=row_colors, col_colors=row_colors,
        row_linkage=row_linkage, col_linkage=col_linkage,
        linewidths=0, xticklabels=True, yticklabels=True,
        cbar_kws={"label": "Jensen-Shannon Distance"},
        dendrogram_ratio=(0.12, 0.12),
    )
    g.ax_heatmap.set_title("JSD Distance Matrix (Clustered)", pad=60)

    # Legend for model types
    handles = [mpatches.Patch(color=c, label=mt)
               for mt, c in model_colors.items()]
    g.ax_heatmap.legend(handles=handles, loc="upper left",
                        bbox_to_anchor=(1.05, 1.15), frameon=False)

    out = figures_dir / "jsd_heatmap.png"
    g.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


def plot_pca_cultural_map(coords: np.ndarray, labels: list[str], figures_dir: Path):
    """Scatter plot of PC1 vs PC2, colored by cultural cluster, shaped by model type."""
    fig, ax = plt.subplots(figsize=(12, 9))

    model_markers = {"hplt2c": "o", "eurollm22b": "^", "qwen2572b": "s"}
    model_sizes = {"hplt2c": 80, "eurollm22b": 100, "qwen2572b": 90}
    model_labels = {"hplt2c": "HPLT-2.15B", "eurollm22b": "EuroLLM-22B", "qwen2572b": "Qwen2.5-72B"}

    for i, label in enumerate(labels):
        parts = label.split("_", 1)
        mt, lang = parts[0], parts[1]
        cluster = LANG_TO_CLUSTER.get(lang, "Other")
        color = CLUSTER_COLORS.get(cluster, "#999999")
        marker = model_markers.get(mt, "D")
        size = model_sizes.get(mt, 80)

        ax.scatter(coords[i, 0], coords[i, 1], c=color, marker=marker,
                   s=size, edgecolors="black", linewidths=0.5, zorder=3)
        display_name = LANG_NAMES.get(lang, lang)
        ax.annotate(display_name, (coords[i, 0], coords[i, 1]),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=7, alpha=0.8)

    # Cluster legend
    cluster_handles = [mpatches.Patch(color=c, label=cl)
                       for cl, c in CLUSTER_COLORS.items()]
    # Model type legend — dynamic based on models present in data
    present_models = sorted(set(l.split("_", 1)[0] for l in labels))
    model_handles = [
        plt.Line2D([0], [0], marker=model_markers.get(mt, "D"), color="w",
                   markerfacecolor="gray", markersize=8,
                   label=model_labels.get(mt, mt))
        for mt in present_models
    ]
    leg1 = ax.legend(handles=cluster_handles, title="Cultural Cluster",
                     loc="upper left", framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=model_handles, title="Model Type",
              loc="lower left", framealpha=0.9)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA Cultural Map of LLM Value Distributions")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = figures_dir / "pca_cultural_map.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


def compute_tsne_map(summary_df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """t-SNE cultural map from ordinal expected values."""
    ordinal = summary_df[summary_df["response_type"].isin(
        {"likert3", "likert4", "likert5", "likert10", "frequency"}
    )].copy()

    ordinal["pair"] = ordinal["model_type"] + "_" + ordinal["lang"]
    pivot = ordinal.pivot_table(
        index="pair", columns="question_id", values="expected_value", aggfunc="first"
    )

    labels = list(pivot.index)
    X = pivot.values

    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    tsne = TSNE(n_components=2, perplexity=min(15, len(labels) - 1),
                random_state=42, init="pca", learning_rate="auto")
    coords = tsne.fit_transform(X_imputed)

    print(f"\n=== t-SNE Cultural Map ===")
    print(f"Features: {X_imputed.shape[1]} ordinal questions")
    print(f"Perplexity: {tsne.perplexity}, KL divergence: {tsne.kl_divergence_:.4f}")

    return coords, labels


def plot_tsne_cultural_map(coords: np.ndarray, labels: list[str], figures_dir: Path):
    """Scatter plot of t-SNE dim1 vs dim2, colored by cultural cluster, shaped by model type."""
    model_markers = {"hplt2c": "o", "eurollm22b": "^", "qwen2572b": "s"}
    model_sizes = {"hplt2c": 80, "eurollm22b": 100, "qwen2572b": 90}
    model_labels = {"hplt2c": "HPLT-2.15B", "eurollm22b": "EuroLLM-22B", "qwen2572b": "Qwen2.5-72B"}

    fig, ax = plt.subplots(figsize=(12, 9))

    for i, label in enumerate(labels):
        parts = label.split("_", 1)
        mt, lang = parts[0], parts[1]
        cluster = LANG_TO_CLUSTER.get(lang, "Other")
        color = CLUSTER_COLORS.get(cluster, "#999999")
        marker = model_markers.get(mt, "D")
        size = model_sizes.get(mt, 80)

        ax.scatter(coords[i, 0], coords[i, 1], c=color, marker=marker,
                   s=size, edgecolors="black", linewidths=0.5, zorder=3)
        display_name = LANG_NAMES.get(lang, lang)
        ax.annotate(display_name, (coords[i, 0], coords[i, 1]),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=7, alpha=0.8)

    cluster_handles = [mpatches.Patch(color=c, label=cl)
                       for cl, c in CLUSTER_COLORS.items()]
    present_models = sorted(set(l.split("_", 1)[0] for l in labels))
    model_handles = [
        plt.Line2D([0], [0], marker=model_markers.get(mt, "D"), color="w",
                   markerfacecolor="gray", markersize=8,
                   label=model_labels.get(mt, mt))
        for mt in present_models
    ]
    leg1 = ax.legend(handles=cluster_handles, title="Cultural Cluster",
                     loc="upper left", framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=model_handles, title="Model Type",
              loc="lower left", framealpha=0.9)

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("t-SNE Cultural Map of LLM Value Distributions")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = figures_dir / "tsne_cultural_map.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


def plot_example_distributions(df: pd.DataFrame, questions: dict,
                               figures_dir: Path):
    """2×3 grid of bar charts for hand-picked questions."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()

    for idx, (qid, summary) in enumerate(EXAMPLE_QUESTIONS):
        ax = axes[idx]
        qmeta = questions.get(qid, {})
        rtype = qmeta.get("response_type", "unknown")

        all_model_types = sorted(df["model_type"].unique())
        for lang in EXAMPLE_LANGS:
            for mt in all_model_types:
                sub = df[(df["question_id"] == qid) &
                         (df["lang"] == lang) &
                         (df["model_type"] == mt)]
                if sub.empty:
                    continue
                sub = sub.sort_values("response_value")
                probs = sub["prob_averaged"].values
                total = probs.sum()
                if total > 0:
                    probs = probs / total
                values = sub["response_value"].values

                model_linestyles = {"hplt2c": "-", "eurollm22b": "--", "qwen2572b": ":"}
                model_short = {"hplt2c": "H", "eurollm22b": "E", "qwen2572b": "Q"}
                linestyle = model_linestyles.get(mt, "-.")
                display_name = LANG_NAMES.get(lang, lang)
                label = f"{display_name} ({model_short.get(mt, mt[0].upper())})"
                ax.plot(values, probs, marker="o", markersize=3,
                        linestyle=linestyle, label=label, alpha=0.7)

        ax.set_title(f"{qid}: {summary}\n({rtype}, {qmeta.get('option_count', '?')} options)",
                     fontsize=10)
        ax.set_xlabel("Response Value")
        ax.set_ylabel("Probability")
        if idx == 0:
            ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Example Question Distributions — Most Divergent Questions",
                 fontsize=14)
    plt.tight_layout()
    out = figures_dir / "example_distributions.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Rephrase Experiment Analysis ──────────────────────────────────────────────

def load_rephrase_results(results_dir: Path) -> pd.DataFrame:
    """Load rephrase_test_*.parquet files from results_dir."""
    frames = []
    for path in sorted(results_dir.glob("rephrase_test_*.parquet")):
        stem = path.stem  # e.g. "rephrase_test_hplt2c_eng"
        parts = stem.replace("rephrase_test_", "").split("_", 1)
        if len(parts) != 2:
            print(f"  Skipping {path.name}: unexpected filename format")
            continue
        model_type, lang = parts
        df = pd.read_parquet(path)
        df["model_type"] = model_type
        frames.append(df)
    if not frames:
        print(f"ERROR: No rephrase_test_*.parquet files found in {results_dir}")
        sys.exit(1)
    combined = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(combined)} rows from {len(frames)} rephrase result files")
    return combined


def load_rephrasings_metadata(rephrasings_path: Path) -> dict:
    """Load rephrasings.json, return lookup dicts."""
    with open(rephrasings_path) as f:
        entries = json.load(f)

    # Map variant_id → issue_type, question_id → issue_type
    variant_issue = {}
    question_issue = {}
    variant_labels = {}  # variant_id → short text label
    for entry in entries:
        qid = entry["canonical_id"]
        issue = entry["issue_type"]
        question_issue[qid] = issue
        variant_issue[f"{qid}_original"] = issue
        variant_labels[f"{qid}_original"] = "original"
        for var in entry["variants"]:
            variant_issue[var["id"]] = issue
            variant_labels[var["id"]] = var["id"].split("_", 1)[1]  # e.g. "r1"

    return {
        "entries": entries,
        "variant_issue": variant_issue,
        "question_issue": question_issue,
        "variant_labels": variant_labels,
    }


def compute_rephrase_metrics(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Compute per-(question, variant, model) JSD, delta P_valid, delta entropy, delta bias.

    Compares each variant to the original for the same question.
    """
    rows = []

    for model_type in df["model_type"].unique():
        mdf = df[df["model_type"] == model_type]

        for qid in mdf["question_id"].unique():
            qdf = mdf[mdf["question_id"] == qid]
            orig_id = f"{qid}_original"
            orig = qdf[qdf["variant_id"] == orig_id].sort_values("response_value")

            if orig.empty:
                continue

            orig_probs = orig["prob_averaged"].values
            orig_total = orig_probs.sum()
            if orig_total > 0:
                orig_dist = orig_probs / orig_total
            else:
                orig_dist = np.ones(len(orig_probs)) / len(orig_probs)

            orig_pvalid = orig["p_valid_forward"].iloc[0]
            orig_entropy_val = entropy(orig_dist) if len(orig_dist) > 0 else 0.0
            orig_bias = orig["position_bias_magnitude"].iloc[0]
            issue = meta["question_issue"].get(qid, "unknown")

            for vid in qdf["variant_id"].unique():
                if vid == orig_id:
                    continue  # skip original-vs-original

                var = qdf[qdf["variant_id"] == vid].sort_values("response_value")
                var_probs = var["prob_averaged"].values
                var_total = var_probs.sum()
                if var_total > 0:
                    var_dist = var_probs / var_total
                else:
                    var_dist = np.ones(len(var_probs)) / len(var_probs)

                # Align distributions (same response values)
                orig_vals = set(orig["response_value"].values)
                var_vals = set(var["response_value"].values)
                all_vals = sorted(orig_vals | var_vals)

                orig_aligned = np.zeros(len(all_vals))
                var_aligned = np.zeros(len(all_vals))
                orig_lookup = dict(zip(orig["response_value"].values, orig_dist))
                var_lookup = dict(zip(var["response_value"].values, var_dist))
                for i, v in enumerate(all_vals):
                    orig_aligned[i] = orig_lookup.get(v, 0.0)
                    var_aligned[i] = var_lookup.get(v, 0.0)

                # Re-normalize after alignment
                if orig_aligned.sum() > 0:
                    orig_aligned = orig_aligned / orig_aligned.sum()
                if var_aligned.sum() > 0:
                    var_aligned = var_aligned / var_aligned.sum()

                jsd_val = jensenshannon(orig_aligned, var_aligned)
                var_pvalid = var["p_valid_forward"].iloc[0]
                var_entropy_val = entropy(var_aligned) if len(var_aligned) > 0 else 0.0
                var_bias = var["position_bias_magnitude"].iloc[0]

                # Get variant_type from the parquet data if available
                vtype = var["variant_type"].iloc[0] if "variant_type" in var.columns else "unknown"

                rows.append({
                    "model_type": model_type,
                    "question_id": qid,
                    "variant_id": vid,
                    "variant_type": vtype,
                    "issue_type": issue,
                    "jsd": jsd_val,
                    "p_valid_original": orig_pvalid,
                    "p_valid_variant": var_pvalid,
                    "delta_p_valid": var_pvalid - orig_pvalid,
                    "entropy_original": orig_entropy_val,
                    "entropy_variant": var_entropy_val,
                    "delta_entropy": var_entropy_val - orig_entropy_val,
                    "bias_original": orig_bias,
                    "bias_variant": var_bias,
                    "delta_bias": var_bias - orig_bias,
                })

    return pd.DataFrame(rows)


VARIANT_TYPE_COLORS = {
    "integrated": "#1b9e77",
    "rephrased": "#d95f02",
    "both": "#7570b3",
}


def plot_rephrase_jsd(metrics: pd.DataFrame, figures_dir: Path):
    """Grouped bar chart: JSD by question, grouped by variant_type, colored by issue type."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # (a) By question, colored by issue type (averaged across variant_types)
    ax = axes[0]
    q_avg = metrics.groupby(["question_id", "issue_type"]).agg(
        mean_jsd=("jsd", "mean"),
        std_jsd=("jsd", "std"),
    ).reset_index().sort_values("mean_jsd", ascending=False)

    x = range(len(q_avg))
    colors = [ISSUE_TYPE_COLORS.get(it, "#999999") for it in q_avg["issue_type"]]
    ax.bar(x, q_avg["mean_jsd"], color=colors, edgecolor="black", linewidth=0.5)
    ax.errorbar(x, q_avg["mean_jsd"], yerr=q_avg["std_jsd"],
                fmt="none", ecolor="black", capsize=3, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(q_avg["question_id"], rotation=45, ha="right")
    ax.set_ylabel("JSD (original vs variant)")
    ax.set_title("(a) Mean JSD by Question (colored by issue type)")
    ax.grid(True, alpha=0.3, axis="y")
    handles = [mpatches.Patch(color=c, label=it)
               for it, c in ISSUE_TYPE_COLORS.items()]
    ax.legend(handles=handles, title="Issue Type", loc="upper right", fontsize=8)

    # (b) By variant_type × issue_type — the 2×2 view
    ax = axes[1]
    vtypes = sorted(metrics["variant_type"].unique())
    issues = sorted(metrics["issue_type"].unique())
    n_vt = len(vtypes)
    n_is = len(issues)
    bar_width = 0.8 / n_vt
    x_pos = np.arange(n_is)

    for i, vt in enumerate(vtypes):
        means = []
        stds = []
        for iss in issues:
            sub = metrics[(metrics["variant_type"] == vt) & (metrics["issue_type"] == iss)]
            means.append(sub["jsd"].mean() if len(sub) > 0 else 0)
            stds.append(sub["jsd"].std() if len(sub) > 1 else 0)
        offset = (i - (n_vt - 1) / 2) * bar_width
        color = VARIANT_TYPE_COLORS.get(vt, "#999999")
        ax.bar(x_pos + offset, means, bar_width * 0.9, label=vt,
               color=color, edgecolor="black", linewidth=0.5)
        ax.errorbar(x_pos + offset, means, yerr=stds,
                    fmt="none", ecolor="black", capsize=2, linewidth=0.6)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(issues, rotation=30, ha="right")
    ax.set_ylabel("JSD (original vs variant)")
    ax.set_title("(b) Mean JSD by Variant Type × Issue Type")
    ax.legend(title="Variant Type", loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = figures_dir / "rephrase_jsd.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


def plot_rephrase_pvalid(metrics: pd.DataFrame, figures_dir: Path):
    """Scatter: original P_valid vs variant P_valid, colored by issue type."""
    fig, ax = plt.subplots(figsize=(8, 8))

    for issue, color in ISSUE_TYPE_COLORS.items():
        sub = metrics[metrics["issue_type"] == issue]
        if sub.empty:
            continue
        ax.scatter(sub["p_valid_original"], sub["p_valid_variant"],
                   c=color, label=issue, alpha=0.7, edgecolors="black",
                   linewidths=0.5, s=50)

    # Diagonal reference line
    lims = [0, max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1)

    ax.set_xlabel("P_valid (original)")
    ax.set_ylabel("P_valid (variant)")
    ax.set_title("P_valid: Original vs Rephrased")
    ax.legend(title="Issue Type")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    out = figures_dir / "rephrase_pvalid.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


def print_rephrase_summary(metrics: pd.DataFrame):
    """Print summary table of rephrase experiment results."""
    print("\n=== Rephrase Experiment Summary ===\n")

    # Per variant type (the key new dimension)
    print("── By Variant Type ──")
    print(f"{'Variant Type':<14} {'Mean JSD':>9} {'Med JSD':>9} {'Δ P_valid':>10} "
          f"{'Δ Entropy':>10} {'Δ Bias':>8} {'N':>4}")
    print("-" * 70)
    for vt in sorted(metrics["variant_type"].unique()):
        sub = metrics[metrics["variant_type"] == vt]
        print(f"{vt:<14} {sub['jsd'].mean():>9.4f} {sub['jsd'].median():>9.4f} "
              f"{sub['delta_p_valid'].mean():>+10.4f} "
              f"{sub['delta_entropy'].mean():>+10.4f} "
              f"{sub['delta_bias'].mean():>+8.4f} {len(sub):>4}")

    # Per issue type
    print("\n── By Issue Type ──")
    print(f"{'Issue Type':<16} {'Mean JSD':>9} {'Med JSD':>9} {'Δ P_valid':>10} "
          f"{'Δ Entropy':>10} {'Δ Bias':>8} {'N':>4}")
    print("-" * 70)
    for issue in sorted(metrics["issue_type"].unique()):
        sub = metrics[metrics["issue_type"] == issue]
        print(f"{issue:<16} {sub['jsd'].mean():>9.4f} {sub['jsd'].median():>9.4f} "
              f"{sub['delta_p_valid'].mean():>+10.4f} "
              f"{sub['delta_entropy'].mean():>+10.4f} "
              f"{sub['delta_bias'].mean():>+8.4f} {len(sub):>4}")

    # 2×2 decomposition: variant_type × issue_type
    print("\n── 2×2 Decomposition: Variant Type × Issue Type (Mean JSD) ──")
    vtypes = sorted(metrics["variant_type"].unique())
    issues = sorted(metrics["issue_type"].unique())
    header = f"{'':>16}" + "".join(f"{vt:>12}" for vt in vtypes)
    print(header)
    print("-" * (16 + 12 * len(vtypes)))
    for iss in issues:
        row = f"{iss:>16}"
        for vt in vtypes:
            sub = metrics[(metrics["variant_type"] == vt) & (metrics["issue_type"] == iss)]
            if len(sub) > 0:
                row += f"{sub['jsd'].mean():>12.4f}"
            else:
                row += f"{'—':>12}"
        print(row)

    # Per question detail
    print(f"\n── Per Question ──")
    print(f"{'Question':<8} {'Issue':<16} {'Var Type':<12} {'JSD':>7} {'Δ P_v':>8} "
          f"{'Δ Ent':>8}")
    print("-" * 65)
    for _, r in metrics.sort_values(["question_id", "variant_type"]).iterrows():
        print(f"{r['question_id']:<8} {r['issue_type']:<16} {r['variant_type']:<12} "
              f"{r['jsd']:>7.4f} {r['delta_p_valid']:>+8.4f} "
              f"{r['delta_entropy']:>+8.4f}")

    # Key hypothesis tests
    print("\n── Hypothesis Tests ──")

    # H1: problematic phrasing → larger JSD than clean controls
    print("\nH1: Problematic phrasing shows larger JSD than clean controls")
    problematic = metrics[metrics["issue_type"].isin(["battery_stem", "card_reference", "read_out"])]
    clean = metrics[metrics["issue_type"] == "control_clean"]
    if len(problematic) > 0 and len(clean) > 0:
        print(f"  Problematic: {problematic['jsd'].mean():.4f}  Clean: {clean['jsd'].mean():.4f}")
        from scipy.stats import mannwhitneyu
        stat, pval = mannwhitneyu(problematic["jsd"], clean["jsd"], alternative="greater")
        print(f"  Mann-Whitney U (one-sided): U={stat:.0f}, p={pval:.3e}")
    else:
        print("  Insufficient data")

    # H2: integration helps (lower JSD than rephrasing alone? or higher P_valid?)
    print("\nH2: Option integration improves P_valid")
    has_int = metrics[metrics["variant_type"].isin(["integrated", "both"])]
    no_int = metrics[metrics["variant_type"] == "rephrased"]
    if len(has_int) > 0 and len(no_int) > 0:
        print(f"  With integration:    mean Δ P_valid = {has_int['delta_p_valid'].mean():+.4f}")
        print(f"  Without integration: mean Δ P_valid = {no_int['delta_p_valid'].mean():+.4f}")
        stat, pval = mannwhitneyu(has_int["delta_p_valid"], no_int["delta_p_valid"], alternative="greater")
        print(f"  Mann-Whitney U (one-sided): U={stat:.0f}, p={pval:.3e}")
    else:
        print("  Insufficient data")

    # H3: rephrasing causes more divergence than integration alone
    print("\nH3: Semantic rephrasing causes more divergence than integration alone")
    has_reph = metrics[metrics["variant_type"].isin(["rephrased", "both"])]
    no_reph = metrics[metrics["variant_type"] == "integrated"]
    if len(has_reph) > 0 and len(no_reph) > 0:
        print(f"  With rephrasing: mean JSD = {has_reph['jsd'].mean():.4f}")
        print(f"  Integration only: mean JSD = {no_reph['jsd'].mean():.4f}")
        stat, pval = mannwhitneyu(has_reph["jsd"], no_reph["jsd"], alternative="greater")
        print(f"  Mann-Whitney U (one-sided): U={stat:.0f}, p={pval:.3e}")
    else:
        print("  Insufficient data")


# ── Subcommands ──────────────────────────────────────────────────────────────

def cmd_quality(df, questions, figures_dir):
    print("\n── Quality Analysis ──")
    df, _ = apply_quality_filter(df)
    report = compute_quality_report(df)
    plot_pvalid_heatmap(report, figures_dir)
    return df


def cmd_bias(df, questions, figures_dir):
    print("\n── Position Bias Analysis ──")
    bias = compute_bias_report(df, questions)
    plot_bias_distribution(bias, figures_dir)


def cmd_summary(df, questions, figures_dir):
    print("\n── Summary Statistics ──")
    summary = compute_summary_stats(df, questions)
    out = figures_dir.parent / "summary_stats.parquet"
    summary.to_parquet(out, index=False)
    print(f"  Saved {out}")
    return summary


def cmd_distance(df, questions, figures_dir):
    print("\n── JSD Distance Analysis ──")
    matrix, labels = compute_jsd_matrix(df)
    plot_jsd_heatmap(matrix, labels, figures_dir)
    # Save matrix
    out = figures_dir.parent / "jsd_matrix.npz"
    np.savez(out, matrix=matrix, labels=labels)
    print(f"  Saved {out}")


def cmd_pca(df, questions, figures_dir):
    print("\n── PCA Cultural Map ──")
    summary = compute_summary_stats(df, questions)
    coords, labels = compute_pca_map(summary)
    plot_pca_cultural_map(coords, labels, figures_dir)


def cmd_tsne(df, questions, figures_dir):
    print("\n── t-SNE Cultural Map ──")
    summary = compute_summary_stats(df, questions)
    coords, labels = compute_tsne_map(summary)
    plot_tsne_cultural_map(coords, labels, figures_dir)


def cmd_examples(df, questions, figures_dir):
    print("\n── Example Distributions ──")
    plot_example_distributions(df, questions, figures_dir)


def cmd_rephrase(results_dir, figures_dir, rephrasings_path):
    """Analyze rephrasing experiment results (independent of main results)."""
    print("\n── Rephrase Experiment Analysis ──")
    rdf = load_rephrase_results(results_dir)
    meta = load_rephrasings_metadata(rephrasings_path)
    metrics = compute_rephrase_metrics(rdf, meta)

    if metrics.empty:
        print("ERROR: No metrics computed — check that rephrase result files exist")
        return

    print_rephrase_summary(metrics)
    plot_rephrase_jsd(metrics, figures_dir)
    plot_rephrase_pvalid(metrics, figures_dir)

    # Save metrics
    out = results_dir / "rephrase_metrics.parquet"
    metrics.to_parquet(out, index=False)
    print(f"  Saved metrics to {out}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM Cultural Values Analysis")
    parser.add_argument("command", choices=[
        "quality", "bias", "summary", "distance", "pca", "tsne", "examples",
        "rephrase", "all",
    ])
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    parser.add_argument("--results-dir", type=Path,
                        default=PROJECT_ROOT / "results")
    parser.add_argument("--questions", type=Path,
                        default=PROJECT_ROOT / "data" / "questions.json")
    parser.add_argument("--figures-dir", type=Path,
                        default=PROJECT_ROOT / "figures")
    parser.add_argument("--rephrasings", type=Path,
                        default=PROJECT_ROOT / "data" / "rephrasings.json")
    args = parser.parse_args()

    args.figures_dir.mkdir(parents=True, exist_ok=True)

    # The rephrase command has its own data loading path
    if args.command == "rephrase":
        cmd_rephrase(args.results_dir, args.figures_dir, args.rephrasings)
        print("\nDone!")
        return

    print(f"Loading data from {args.results_dir}...")
    df = load_all_results(args.results_dir)
    questions = load_question_metadata(args.questions)
    print(f"Loaded metadata for {len(questions)} questions")

    commands = {
        "quality": cmd_quality,
        "bias": cmd_bias,
        "summary": cmd_summary,
        "distance": cmd_distance,
        "pca": cmd_pca,
        "tsne": cmd_tsne,
        "examples": cmd_examples,
    }

    if args.command == "all":
        df = cmd_quality(df, questions, args.figures_dir)
        cmd_bias(df, questions, args.figures_dir)
        cmd_summary(df, questions, args.figures_dir)
        cmd_distance(df, questions, args.figures_dir)
        cmd_pca(df, questions, args.figures_dir)
        cmd_tsne(df, questions, args.figures_dir)
        cmd_examples(df, questions, args.figures_dir)
    else:
        result = commands[args.command](df, questions, args.figures_dir)
        if args.command == "quality" and result is not None:
            df = result

    print("\nDone!")


if __name__ == "__main__":
    main()
