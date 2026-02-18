"""Compare LLM survey distributions to real EVS human survey data.

Core analysis: do LLMs trained on language X produce value distributions
that resemble country X's actual survey responses?

Generates 5 figures:
    1. human_scatter_ev.png     — LLM expected value vs human expected value
    2. human_jsd_by_qtype.png   — Mean JSD to human by question type
    3. human_model_comparison.png — model comparison: JSD to human per language
    4. human_jsd_heatmap.png    — JSD(LLM, human) heatmap by model × language
    5. human_best_worst.png     — Example distributions for best/worst aligned Qs

Usage:
    python eurollm/analysis/compare_to_human.py \
        --results-dir eurollm/results \
        --human eurollm/human_data/data/human_distributions.parquet \
        --questions eurollm/data/questions.json \
        --figures-dir eurollm/figures
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
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr, spearmanr


# ── Constants ────────────────────────────────────────────────────────────────

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
for cluster, langs in CULTURAL_CLUSTERS.items():
    for lang in langs:
        LANG_TO_CLUSTER[lang] = cluster

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
}

MODEL_LABELS = {
    "hplt2c": "HPLT-2.15B",
    "eurollm22b": "EuroLLM-22B",
    "qwen2572b": "Qwen2.5-72B",
}

ORDINAL_TYPES = {"likert3", "likert4", "likert5", "likert10", "frequency"}


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_llm_results(results_dir: Path) -> pd.DataFrame:
    """Load all LLM parquet results, adding model_type and lang columns."""
    frames = []
    for path in sorted(results_dir.glob("*.parquet")):
        if "rephrase" in path.stem or "validate" in path.stem:
            continue
        stem = path.stem
        parts = stem.split("_", 1)
        if len(parts) != 2:
            continue
        model_type, lang = parts
        df = pd.read_parquet(path)
        df["model_type"] = model_type
        df["lang"] = lang
        frames.append(df)
    if not frames:
        print(f"ERROR: No LLM result parquets found in {results_dir}")
        sys.exit(1)
    combined = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(combined)} LLM rows from {len(frames)} files")
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


def build_distributions(df: pd.DataFrame, prob_col: str, group_cols: list[str]) -> dict:
    """Build {(group_key): {response_value: prob}} from a DataFrame.

    Normalizes probabilities to sum to 1 per group.
    """
    distributions = {}
    for group_key, grp in df.groupby(group_cols):
        grp_sorted = grp.sort_values("response_value")
        values = grp_sorted["response_value"].values
        probs = grp_sorted[prob_col].values
        total = probs.sum()
        if total > 0:
            probs = probs / total
        else:
            probs = np.ones(len(probs)) / len(probs)
        distributions[group_key] = dict(zip(values, probs))
    return distributions


def align_distributions(dist_a: dict, dist_b: dict) -> tuple[np.ndarray, np.ndarray]:
    """Align two {value: prob} dicts to arrays over shared values."""
    all_values = sorted(set(dist_a.keys()) | set(dist_b.keys()))
    a = np.array([dist_a.get(v, 0.0) for v in all_values])
    b = np.array([dist_b.get(v, 0.0) for v in all_values])
    # Re-normalize
    if a.sum() > 0:
        a = a / a.sum()
    if b.sum() > 0:
        b = b / b.sum()
    return a, b


# ── Analysis Functions ───────────────────────────────────────────────────────

def compute_jsd_per_question(
    llm_dists: dict,
    human_dists: dict,
    questions: dict,
) -> pd.DataFrame:
    """Compute JSD(LLM, human) for each (model_type, lang, question_id).

    Returns DataFrame with columns:
        model_type, lang, question_id, response_type, jsd, expected_llm, expected_human
    """
    rows = []

    for (mt, lang, qid), llm_dist in llm_dists.items():
        human_key = (lang, qid)
        if human_key not in human_dists:
            continue

        human_dist = human_dists[human_key]
        a, b = align_distributions(llm_dist, human_dist)
        all_values = sorted(set(llm_dist.keys()) | set(human_dist.keys()))

        jsd = jensenshannon(a, b)

        # Expected values for ordinal questions
        rtype = questions.get(qid, {}).get("response_type", "unknown")
        values_arr = np.array(all_values, dtype=float)
        if rtype in ORDINAL_TYPES:
            ev_llm = np.dot(values_arr, a)
            ev_human = np.dot(values_arr, b)
        else:
            ev_llm = np.nan
            ev_human = np.nan

        rows.append({
            "model_type": mt,
            "lang": lang,
            "question_id": qid,
            "response_type": rtype,
            "jsd": jsd,
            "expected_llm": ev_llm,
            "expected_human": ev_human,
        })

    return pd.DataFrame(rows)


def print_summary(jsd_df: pd.DataFrame):
    """Print overall summary statistics."""
    print(f"\n{'='*70}")
    print("LLM vs HUMAN COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"Total comparisons: {len(jsd_df)}")
    print(f"Model types: {sorted(jsd_df['model_type'].unique())}")
    print(f"Languages: {jsd_df['lang'].nunique()}")
    print(f"Questions: {jsd_df['question_id'].nunique()}")

    # Overall JSD
    print(f"\n--- Overall JSD ---")
    for mt in sorted(jsd_df["model_type"].unique()):
        sub = jsd_df[jsd_df["model_type"] == mt]
        print(f"  {mt}: mean={sub['jsd'].mean():.4f}, "
              f"median={sub['jsd'].median():.4f}, "
              f"std={sub['jsd'].std():.4f}")

    # Correlation for ordinal questions
    ordinal = jsd_df[jsd_df["response_type"].isin(ORDINAL_TYPES)].dropna(
        subset=["expected_llm", "expected_human"]
    )
    if len(ordinal) > 2:
        print(f"\n--- Expected Value Correlation (ordinal questions, n={len(ordinal)}) ---")
        for mt in sorted(ordinal["model_type"].unique()):
            sub = ordinal[ordinal["model_type"] == mt]
            if len(sub) > 2:
                r_p, p_p = pearsonr(sub["expected_llm"], sub["expected_human"])
                r_s, p_s = spearmanr(sub["expected_llm"], sub["expected_human"])
                print(f"  {mt}: Pearson r={r_p:.3f} (p={p_p:.2e}), "
                      f"Spearman rho={r_s:.3f} (p={p_s:.2e})")

    # JSD by response type
    print(f"\n--- JSD by Response Type ---")
    for rt in sorted(jsd_df["response_type"].unique()):
        sub = jsd_df[jsd_df["response_type"] == rt]
        print(f"  {rt:<14} mean={sub['jsd'].mean():.4f}  "
              f"median={sub['jsd'].median():.4f}  n={len(sub)}")

    # Per-model per-language comparison
    model_types = sorted(jsd_df["model_type"].unique())
    print(f"\n--- Per-Model Mean JSD to Human by Language ---")
    header = f"  {'lang':<5}" + "".join(f"  {MODEL_LABELS.get(mt, mt):>14}" for mt in model_types) + "  closest"
    print(header)
    for lang in sorted(jsd_df["lang"].unique()):
        vals = {}
        parts = f"  {lang:<5}"
        for mt in model_types:
            sub = jsd_df[(jsd_df["model_type"] == mt) & (jsd_df["lang"] == lang)]
            if len(sub) > 0:
                vals[mt] = sub["jsd"].mean()
                parts += f"  {vals[mt]:>14.4f}"
            else:
                parts += f"  {'—':>14}"
        if vals:
            winner = min(vals, key=vals.get)
            parts += f"  {MODEL_LABELS.get(winner, winner)}"
        print(parts)


# ── Figure 1: Expected Value Scatter ─────────────────────────────────────────

def plot_scatter_ev(jsd_df: pd.DataFrame, figures_dir: Path):
    """Scatter: LLM expected value vs human expected value, per model type."""
    ordinal = jsd_df[jsd_df["response_type"].isin(ORDINAL_TYPES)].dropna(
        subset=["expected_llm", "expected_human"]
    )
    if ordinal.empty:
        print("  No ordinal data for scatter plot")
        return

    model_types = sorted(ordinal["model_type"].unique())
    n_models = len(model_types)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 6), squeeze=False)

    for idx, mt in enumerate(model_types):
        ax = axes[0, idx]
        sub = ordinal[ordinal["model_type"] == mt]

        for lang in sorted(sub["lang"].unique()):
            lsub = sub[sub["lang"] == lang]
            cluster = LANG_TO_CLUSTER.get(lang, "Other")
            color = CLUSTER_COLORS.get(cluster, "#999999")
            ax.scatter(lsub["expected_human"], lsub["expected_llm"],
                       c=color, alpha=0.3, s=15, edgecolors="none")

        # Correlation line and stats
        r_p, p_p = pearsonr(sub["expected_human"], sub["expected_llm"])
        r_s, _ = spearmanr(sub["expected_human"], sub["expected_llm"])

        # Regression line
        z = np.polyfit(sub["expected_human"], sub["expected_llm"], 1)
        x_line = np.linspace(sub["expected_human"].min(), sub["expected_human"].max(), 100)
        ax.plot(x_line, np.polyval(z, x_line), "k--", alpha=0.5, linewidth=1)

        # Identity line
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
        ax.plot(lims, lims, "k:", alpha=0.3, linewidth=0.5)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        mt_label = MODEL_LABELS.get(mt, mt)
        ax.set_title(f"{mt_label}\nr={r_p:.3f}, rho={r_s:.3f} (n={len(sub)})")
        ax.set_xlabel("Human Expected Value")
        ax.set_ylabel("LLM Expected Value")
        ax.grid(True, alpha=0.3)

    # Cluster legend
    handles = [mpatches.Patch(color=c, label=cl)
               for cl, c in CLUSTER_COLORS.items()]
    axes[0, -1].legend(handles=handles, title="Cultural Cluster",
                       loc="upper left", fontsize=7, framealpha=0.9)

    plt.suptitle("LLM vs Human Expected Values (Ordinal Questions)", fontsize=14)
    plt.tight_layout()
    out = figures_dir / "human_scatter_ev.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 2: JSD by Question Type ──────────────────────────────────────────

def plot_jsd_by_qtype(jsd_df: pd.DataFrame, figures_dir: Path):
    """Box/violin plots of JSD to human by response type and model type."""
    fig, ax = plt.subplots(figsize=(10, 6))

    order = sorted(jsd_df["response_type"].unique())
    sns.boxplot(
        data=jsd_df, x="response_type", y="jsd", hue="model_type",
        order=order, ax=ax, fliersize=2, linewidth=0.8,
    )
    ax.set_xlabel("Response Type")
    ax.set_ylabel("JSD (LLM vs Human)")
    ax.set_title("JSD to Human Data by Question Type")
    ax.legend(title="Model Type")
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    out = figures_dir / "human_jsd_by_qtype.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 3: Monolingual vs Multilingual ────────────────────────────────────

def plot_model_comparison(jsd_df: pd.DataFrame, figures_dir: Path):
    """Grouped bar chart comparing all model types' mean JSD to human per language."""
    # Compute per-language mean JSD for each model type
    lang_jsd = jsd_df.groupby(["model_type", "lang"]).agg(
        mean_jsd=("jsd", "mean"),
    ).reset_index()

    model_types = sorted(lang_jsd["model_type"].unique())
    # Find languages present in all model types
    per_model_langs = [set(lang_jsd[lang_jsd["model_type"] == mt]["lang"]) for mt in model_types]
    shared_langs = sorted(set.intersection(*per_model_langs)) if per_model_langs else []

    if not shared_langs:
        print("  No shared languages for model comparison")
        return

    # Build per-model series
    model_series = {}
    for mt in model_types:
        sub = lang_jsd[lang_jsd["model_type"] == mt].set_index("lang")["mean_jsd"]
        model_series[mt] = sub

    n_models = len(model_types)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # (a) Grouped bar chart
    ax = axes[0]
    x = np.arange(len(shared_langs))
    width = 0.8 / n_models
    for i, mt in enumerate(model_types):
        offset = (i - (n_models - 1) / 2) * width
        vals = [model_series[mt][l] for l in shared_langs]
        ax.bar(x + offset, vals, width * 0.9,
               label=MODEL_LABELS.get(mt, mt),
               color=MODEL_COLORS.get(mt, "#999999"),
               edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([LANG_NAMES.get(l, l) for l in shared_langs],
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean JSD to Human Data")
    ax.set_title("(a) Mean JSD to Human by Language")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # (b) Pairwise scatter — if exactly 2 models, show classic scatter;
    #     if 3+, show first two on axes with third as color intensity
    if n_models == 2:
        mt_x, mt_y = model_types[0], model_types[1]
        ax = axes[1]
        for lang in shared_langs:
            cluster = LANG_TO_CLUSTER.get(lang, "Other")
            color = CLUSTER_COLORS.get(cluster, "#999999")
            ax.scatter(model_series[mt_x][lang], model_series[mt_y][lang],
                       c=color, s=60, edgecolors="black", linewidths=0.5, zorder=3)
            ax.annotate(LANG_NAMES.get(lang, lang),
                        (model_series[mt_x][lang], model_series[mt_y][lang]),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=7, alpha=0.8)
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
        ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(f"{MODEL_LABELS.get(mt_x, mt_x)} Mean JSD")
        ax.set_ylabel(f"{MODEL_LABELS.get(mt_y, mt_y)} Mean JSD")
        ax.set_title("(b) Model Comparison")
        ax.grid(True, alpha=0.3)
    else:
        # 3+ models: show per-language ranking — which model is closest to human?
        ax = axes[1]
        win_counts = {mt: 0 for mt in model_types}
        for lang in shared_langs:
            best = min(model_types, key=lambda mt: model_series[mt][lang])
            win_counts[best] += 1
        labels_bar = [MODEL_LABELS.get(mt, mt) for mt in model_types]
        counts = [win_counts[mt] for mt in model_types]
        colors = [MODEL_COLORS.get(mt, "#999999") for mt in model_types]
        ax.bar(labels_bar, counts, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_ylabel("Number of Languages (closest to human)")
        ax.set_title(f"(b) Model Wins across {len(shared_langs)} Languages")
        ax.grid(True, alpha=0.3, axis="y")
        for i, (lbl, cnt) in enumerate(zip(labels_bar, counts)):
            ax.text(i, cnt + 0.3, str(cnt), ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    out = figures_dir / "human_model_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 4: JSD Heatmap ───────────────────────────────────────────────────

def plot_jsd_heatmap(jsd_df: pd.DataFrame, figures_dir: Path):
    """Heatmap: mean JSD(LLM, human) by model_type × language."""
    pivot = jsd_df.groupby(["model_type", "lang"]).agg(
        mean_jsd=("jsd", "mean"),
    ).reset_index().pivot_table(
        index="model_type", columns="lang", values="mean_jsd"
    )

    # Sort columns alphabetically, rename to full names
    pivot = pivot[sorted(pivot.columns)]
    pivot.columns = [LANG_NAMES.get(c, c) for c in pivot.columns]
    pivot.index = [MODEL_LABELS.get(i, i) for i in pivot.index]

    fig, ax = plt.subplots(figsize=(18, 4))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="RdYlGn_r",
        ax=ax, linewidths=0.5,
        cbar_kws={"label": "Mean JSD to Human"},
    )
    ax.set_title("Mean JSD (LLM vs Human) by Model and Language")
    ax.set_xlabel("Language")
    ax.set_ylabel("Model")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    out = figures_dir / "human_jsd_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 5: Best and Worst Aligned Questions ──────────────────────────────

def plot_best_worst(
    jsd_df: pd.DataFrame,
    llm_dists: dict,
    human_dists: dict,
    figures_dir: Path,
):
    """Show distribution comparisons for the 3 best and 3 worst aligned questions."""
    # Average JSD across all (model_type, lang) for each question
    q_jsd = jsd_df.groupby("question_id").agg(mean_jsd=("jsd", "mean")).reset_index()
    q_jsd = q_jsd.sort_values("mean_jsd")

    best_3 = list(q_jsd.head(3)["question_id"])
    worst_3 = list(q_jsd.tail(3)["question_id"])
    examples = best_3 + worst_3

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    # Pick a representative language and model for visualization
    example_configs = [
        ("eng", "hplt2c", "HPLT English"),
        ("deu", "hplt2c", "HPLT German"),
    ]

    for idx, qid in enumerate(examples):
        ax = axes[idx]
        is_best = idx < 3
        mean_jsd = q_jsd[q_jsd["question_id"] == qid]["mean_jsd"].iloc[0]

        for lang, mt, label in example_configs:
            llm_key = (mt, lang, qid)
            human_key = (lang, qid)

            if llm_key in llm_dists and human_key in human_dists:
                llm_d = llm_dists[llm_key]
                human_d = human_dists[human_key]
                all_vals = sorted(set(llm_d.keys()) | set(human_d.keys()))

                llm_arr = np.array([llm_d.get(v, 0) for v in all_vals])
                human_arr = np.array([human_d.get(v, 0) for v in all_vals])
                if llm_arr.sum() > 0:
                    llm_arr = llm_arr / llm_arr.sum()
                if human_arr.sum() > 0:
                    human_arr = human_arr / human_arr.sum()

                width = 0.35
                x = np.arange(len(all_vals))
                lang_name = LANG_NAMES.get(lang, lang)
                ax.bar(x - width / 2, human_arr, width,
                       label=f"Human ({lang_name})", alpha=0.7)
                ax.bar(x + width / 2, llm_arr, width,
                       label=f"LLM ({label})", alpha=0.7)
                ax.set_xticks(x)
                ax.set_xticklabels(all_vals, fontsize=8)
                break  # One comparison per subplot

        category = "BEST" if is_best else "WORST"
        ax.set_title(f"{category}: {qid} (mean JSD={mean_jsd:.3f})", fontsize=10)
        ax.set_xlabel("Response Value")
        ax.set_ylabel("Probability")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Best and Worst Aligned Questions (LLM vs Human)", fontsize=14)
    plt.tight_layout()
    out = figures_dir / "human_best_worst.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Compare LLM cultural value distributions to human survey data"
    )
    parser.add_argument("--results-dir", type=Path,
                        default=PROJECT_ROOT / "results")
    parser.add_argument("--human", type=Path,
                        default=PROJECT_ROOT / "human_data" / "data" / "human_distributions.parquet")
    parser.add_argument("--questions", type=Path,
                        default=PROJECT_ROOT / "data" / "questions.json")
    parser.add_argument("--figures-dir", type=Path,
                        default=PROJECT_ROOT / "figures")
    args = parser.parse_args()

    args.figures_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    llm_df = load_llm_results(args.results_dir)
    human_df = pd.read_parquet(args.human)
    questions = load_question_metadata(args.questions)
    print(f"Human data: {len(human_df)} rows, "
          f"{human_df['lang'].nunique()} languages, "
          f"{human_df['question_id'].nunique()} questions")

    # Build distribution lookups
    print("\nBuilding distributions...")
    llm_dists = build_distributions(
        llm_df, "prob_averaged", ["model_type", "lang", "question_id"]
    )
    human_dists = build_distributions(
        human_df, "prob_human", ["lang", "question_id"]
    )
    print(f"  LLM distributions: {len(llm_dists)}")
    print(f"  Human distributions: {len(human_dists)}")

    # Compute JSD for all (model_type, lang, question_id) triples
    print("\nComputing JSD...")
    jsd_df = compute_jsd_per_question(llm_dists, human_dists, questions)
    print(f"  Computed {len(jsd_df)} JSD values")

    if jsd_df.empty:
        print("ERROR: No overlapping (model_type, lang, question_id) between LLM and human data")
        sys.exit(1)

    # Print summary
    print_summary(jsd_df)

    # Generate figures
    print(f"\nGenerating figures in {args.figures_dir}/...")
    plot_scatter_ev(jsd_df, args.figures_dir)
    plot_jsd_by_qtype(jsd_df, args.figures_dir)
    plot_model_comparison(jsd_df, args.figures_dir)
    plot_jsd_heatmap(jsd_df, args.figures_dir)
    plot_best_worst(jsd_df, llm_dists, human_dists, args.figures_dir)

    # Save JSD data for further analysis
    out = args.figures_dir.parent / "human_comparison.parquet"
    jsd_df.to_parquet(out, index=False)
    print(f"\nSaved comparison data to {out}")

    print("\nDone!")


if __name__ == "__main__":
    main()
