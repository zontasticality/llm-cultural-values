"""Compare LLM survey distributions to real EVS human survey data.

Core analysis: do LLMs trained on language X produce value distributions
that resemble country X's actual survey responses?

Main figures (F1-F5):
    F1. F1_human_fitted_umap.png  — Human-fitted UMAP with LLM projections
    F2. F2_inglehart_welzel.png   — Inglehart-Welzel composite scatter
    F3. F3_ev_scatter.png         — EV scatter with z-scored row
    F4. F4_jsd_heatmap.png        — JSD heatmap with mean±std annotations
    F5. F5_deepdive.png           — Per-question deep dive (3×3 grid)

Usage:
    python eurollm/analysis/compare_to_human.py \
        --db eurollm/data/survey.db \
        --figures-dir eurollm/figures
    python eurollm/analysis/compare_to_human.py \
        --db eurollm/data/survey.db --figure umap
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
from sklearn.impute import SimpleImputer

from analysis.constants import (
    LANG_NAMES, CULTURAL_CLUSTERS, LANG_TO_CLUSTER, CLUSTER_COLORS,
    MODEL_COLORS, MODEL_LABELS, MODEL_MARKERS, ORDINAL_TYPES,
    MODELS_EXCLUDE, IW_TRADITIONAL_SECULAR, IW_SURVIVAL_SELFEXPR,
    DEEPDIVE_IW_QUESTIONS, DEEPDIVE_LANGS,
)


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_llm_results(results_dir: Path, db_path: str | None = None) -> pd.DataFrame:
    """Load LLM results, from DB if db_path given, else from parquet files."""
    if db_path:
        from db.load import load_results
        df = load_results(db_path)
        print(f"Loaded {len(df)} LLM rows from {db_path}")
        return df

    frames = []
    for path in sorted(results_dir.glob("*.parquet")):
        if "rephrase" in path.stem or "validate" in path.stem:
            continue
        stem = path.stem
        parts = stem.rsplit("_", 1)
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


# ── F1: Human-Fitted UMAP ────────────────────────────────────────────────────

def _build_ev_matrix(dists: dict, group_type: str, questions: dict):
    """Build expected-value matrix from distribution dicts.

    group_type: "human" → keys are (lang, qid), "llm" → keys are (mt, lang, qid)
    Returns (matrix DataFrame indexed by row label, column=question_id).
    """
    rows = []
    for key, dist in dists.items():
        if group_type == "human":
            lang, qid = key
            row_label = lang
        else:
            mt, lang, qid = key
            row_label = f"{mt}_{lang}"

        rtype = questions.get(qid, {}).get("response_type", "unknown")
        if rtype not in ORDINAL_TYPES:
            continue

        all_vals = sorted(dist.keys())
        probs = np.array([dist[v] for v in all_vals])
        total = probs.sum()
        if total > 0:
            probs = probs / total
        vals = np.array(all_vals, dtype=float)
        ev = np.dot(vals, probs)
        rows.append({"row_label": row_label, "question_id": qid, "ev": ev})

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index="row_label", columns="question_id",
                           values="ev", aggfunc="first")
    return pivot


def compute_human_fitted_umap(
    llm_dists: dict,
    human_dists: dict,
    questions: dict,
):
    """Fit UMAP on human data, project LLM data into the same space.

    Returns (human_coords, llm_coords, human_labels, llm_labels).
    """
    from umap import UMAP

    human_pivot = _build_ev_matrix(human_dists, "human", questions)
    llm_pivot = _build_ev_matrix(llm_dists, "llm", questions)

    # Filter excluded models from LLM data
    llm_pivot = llm_pivot[
        ~llm_pivot.index.str.startswith(tuple(f"{m}_" for m in MODELS_EXCLUDE))
    ]

    # Align columns
    shared_qs = sorted(set(human_pivot.columns) & set(llm_pivot.columns))
    human_aligned = human_pivot[shared_qs]
    llm_aligned = llm_pivot[shared_qs]

    # Impute: fit on human, transform both
    imputer = SimpleImputer(strategy="mean")
    X_human = imputer.fit_transform(human_aligned.values)
    X_llm = imputer.transform(llm_aligned.values)

    # Fit UMAP on human data only
    reducer = UMAP(n_components=2, n_neighbors=10, min_dist=0.3, random_state=42)
    human_coords = reducer.fit_transform(X_human)

    # Project LLM data
    llm_coords = reducer.transform(X_llm)

    print(f"  Human-fitted UMAP: {len(human_aligned)} human, "
          f"{len(llm_aligned)} LLM, {len(shared_qs)} shared questions")

    return human_coords, llm_coords, list(human_aligned.index), list(llm_aligned.index)


def plot_human_fitted_umap(
    human_coords: np.ndarray,
    llm_coords: np.ndarray,
    human_labels: list[str],
    llm_labels: list[str],
    figures_dir: Path,
):
    """F1: 1×3 faceted UMAP — one panel per model family."""
    # Determine model families present
    model_families = {}
    for i, label in enumerate(llm_labels):
        mt = label.rsplit("_", 1)[0]
        model_families.setdefault(mt, []).append(i)

    family_order = sorted(model_families.keys())
    n_panels = len(family_order)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 7), squeeze=False)

    # Build human_coords lookup by lang
    human_by_lang = {}
    for i, lang in enumerate(human_labels):
        human_by_lang[lang] = human_coords[i]

    for panel_idx, mt in enumerate(family_order):
        ax = axes[0, panel_idx]

        # Gray lines LLM→human
        for i in model_families[mt]:
            lang = llm_labels[i].rsplit("_", 1)[1]
            if lang in human_by_lang:
                ax.plot(
                    [llm_coords[i, 0], human_by_lang[lang][0]],
                    [llm_coords[i, 1], human_by_lang[lang][1]],
                    color="#cccccc", linewidth=0.8, alpha=0.5, zorder=1,
                )

        # Human points: large circles
        for i, lang in enumerate(human_labels):
            cluster = LANG_TO_CLUSTER.get(lang, "Other")
            color = CLUSTER_COLORS.get(cluster, "#999999")
            ax.scatter(human_coords[i, 0], human_coords[i, 1], c=color,
                       s=200, edgecolors="black", linewidths=1.2, zorder=5,
                       marker="o")
            ax.annotate(LANG_NAMES.get(lang, lang),
                        (human_coords[i, 0], human_coords[i, 1]),
                        textcoords="offset points", xytext=(7, 7),
                        fontsize=8, fontweight="bold", alpha=0.9)

        # LLM points: smaller markers
        marker = MODEL_MARKERS.get(mt, "D")
        for i in model_families[mt]:
            lang = llm_labels[i].rsplit("_", 1)[1]
            cluster = LANG_TO_CLUSTER.get(lang, "Other")
            color = CLUSTER_COLORS.get(cluster, "#999999")
            ax.scatter(llm_coords[i, 0], llm_coords[i, 1], c=color,
                       s=80, marker=marker, edgecolors="black",
                       linewidths=0.5, alpha=0.8, zorder=3)

        ax.set_title(MODEL_LABELS.get(mt, mt), fontsize=13, fontweight="bold")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.grid(True, alpha=0.2)

    # Shared legend
    cluster_handles = [mpatches.Patch(color=c, label=cl)
                       for cl, c in CLUSTER_COLORS.items()]
    source_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                   markersize=11, markeredgecolor="black", label="Human (EVS)"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="gray",
                   markersize=7, markeredgecolor="black", label="LLM"),
    ]
    fig.legend(handles=cluster_handles + source_handles,
               loc="lower center", ncol=len(CLUSTER_COLORS) + 2,
               fontsize=8, framealpha=0.9)

    plt.suptitle("F1: Human-Fitted UMAP — LLM Projected into Human Cultural Space",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    out = figures_dir / "F1_human_fitted_umap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── F2: Inglehart-Welzel Composite ──────────────────────────────────────────

def compute_iw_composites(
    llm_dists: dict,
    human_dists: dict,
    questions: dict,
) -> pd.DataFrame:
    """Compute Inglehart-Welzel composite scores for human and LLM sources.

    Returns DataFrame: source, lang, trad_secular, surv_selfexpr
    """
    all_iw_qs = {**IW_TRADITIONAL_SECULAR, **IW_SURVIVAL_SELFEXPR}

    # Collect expected values per (source, lang, qid)
    ev_rows = []

    # Human EVs
    for (lang, qid), dist in human_dists.items():
        if qid not in all_iw_qs:
            continue
        vals = sorted(dist.keys())
        probs = np.array([dist[v] for v in vals])
        total = probs.sum()
        if total > 0:
            probs = probs / total
        ev = np.dot(np.array(vals, dtype=float), probs)
        ev_rows.append({"source": "human", "lang": lang, "qid": qid, "ev": ev})

    # LLM EVs
    for (mt, lang, qid), dist in llm_dists.items():
        if qid not in all_iw_qs or mt in MODELS_EXCLUDE:
            continue
        vals = sorted(dist.keys())
        probs = np.array([dist[v] for v in vals])
        total = probs.sum()
        if total > 0:
            probs = probs / total
        ev = np.dot(np.array(vals, dtype=float), probs)
        ev_rows.append({"source": mt, "lang": lang, "qid": qid, "ev": ev})

    ev_df = pd.DataFrame(ev_rows)
    if ev_df.empty:
        return pd.DataFrame()

    # Apply polarity flips
    def flip_ev(row):
        info = all_iw_qs.get(row["qid"], {})
        if info.get("flip", False):
            max_val = info.get("max_val")
            if max_val is not None:
                return max_val - row["ev"]
            # Fallback: use data max for this question
            q_max = ev_df[ev_df["qid"] == row["qid"]]["ev"].max()
            return q_max - row["ev"]
        return row["ev"]

    ev_df["ev_flipped"] = ev_df.apply(flip_ev, axis=1)

    # Z-score normalize per question across ALL (source, lang) jointly
    for qid in ev_df["qid"].unique():
        mask = ev_df["qid"] == qid
        vals = ev_df.loc[mask, "ev_flipped"]
        mu, sigma = vals.mean(), vals.std()
        if sigma > 0:
            ev_df.loc[mask, "ev_z"] = (vals - mu) / sigma
        else:
            ev_df.loc[mask, "ev_z"] = 0.0

    # Average z-scores within each dimension
    result_rows = []
    for (source, lang), grp in ev_df.groupby(["source", "lang"]):
        # Traditional-Secular
        ts_qs = grp[grp["qid"].isin(IW_TRADITIONAL_SECULAR)]
        if len(ts_qs) >= 3:
            trad_secular = ts_qs["ev_z"].mean()
        else:
            trad_secular = np.nan

        # Survival-SelfExpression
        se_qs = grp[grp["qid"].isin(IW_SURVIVAL_SELFEXPR)]
        if len(se_qs) >= 3:
            surv_selfexpr = se_qs["ev_z"].mean()
        else:
            surv_selfexpr = np.nan

        result_rows.append({
            "source": source,
            "lang": lang,
            "trad_secular": trad_secular,
            "surv_selfexpr": surv_selfexpr,
        })

    return pd.DataFrame(result_rows).dropna()


def plot_iw_composite(iw_df: pd.DataFrame, figures_dir: Path):
    """F2: Inglehart-Welzel composite scatter plot."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Build human coords lookup
    human = iw_df[iw_df["source"] == "human"]
    human_by_lang = {
        row["lang"]: (row["surv_selfexpr"], row["trad_secular"])
        for _, row in human.iterrows()
    }

    llm = iw_df[iw_df["source"] != "human"]

    # Gray lines LLM→human
    for _, row in llm.iterrows():
        lang = row["lang"]
        if lang in human_by_lang:
            hx, hy = human_by_lang[lang]
            ax.plot([row["surv_selfexpr"], hx], [row["trad_secular"], hy],
                    color="#cccccc", linewidth=0.8, alpha=0.5, zorder=1)

    # LLM points
    for _, row in llm.iterrows():
        cluster = LANG_TO_CLUSTER.get(row["lang"], "Other")
        color = CLUSTER_COLORS.get(cluster, "#999999")
        marker = MODEL_MARKERS.get(row["source"], "D")
        ax.scatter(row["surv_selfexpr"], row["trad_secular"],
                   c=color, marker=marker, s=80, edgecolors="black",
                   linewidths=0.5, alpha=0.8, zorder=3)

    # Human points (large circles with labels)
    for _, row in human.iterrows():
        cluster = LANG_TO_CLUSTER.get(row["lang"], "Other")
        color = CLUSTER_COLORS.get(cluster, "#999999")
        ax.scatter(row["surv_selfexpr"], row["trad_secular"],
                   c=color, s=200, edgecolors="black", linewidths=1.2,
                   zorder=5, marker="o")
        ax.annotate(LANG_NAMES.get(row["lang"], row["lang"]),
                    (row["surv_selfexpr"], row["trad_secular"]),
                    textcoords="offset points", xytext=(7, 7),
                    fontsize=8, fontweight="bold", alpha=0.9)

    # Legends
    cluster_handles = [mpatches.Patch(color=c, label=cl)
                       for cl, c in CLUSTER_COLORS.items()]
    present_models = sorted(llm["source"].unique())
    model_handles = [
        plt.Line2D([0], [0], marker=MODEL_MARKERS.get(mt, "D"), color="w",
                   markerfacecolor="gray", markersize=7,
                   markeredgecolor="black",
                   label=MODEL_LABELS.get(mt, mt))
        for mt in present_models
    ]
    model_handles.append(
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                   markersize=11, markeredgecolor="black", label="Human (EVS)")
    )
    leg1 = ax.legend(handles=cluster_handles, title="Cultural Cluster",
                     loc="upper left", framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=model_handles, title="Source",
              loc="lower left", framealpha=0.9)

    ax.set_xlabel("Survival → Self-Expression", fontsize=12)
    ax.set_ylabel("Traditional → Secular-Rational", fontsize=12)
    ax.set_title("F2: Inglehart-Welzel Cultural Map (Composite Scores)",
                 fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out = figures_dir / "F2_inglehart_welzel.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── F3: EV Scatter + Z-Score ────────────────────────────────────────────────

def plot_scatter_ev(jsd_df: pd.DataFrame, figures_dir: Path):
    """F3: 2-row scatter — raw EV (top) and z-scored EV (bottom)."""
    ordinal = jsd_df[
        jsd_df["response_type"].isin(ORDINAL_TYPES)
        & ~jsd_df["model_type"].isin(MODELS_EXCLUDE)
    ].dropna(subset=["expected_llm", "expected_human"])
    if ordinal.empty:
        print("  No ordinal data for scatter plot")
        return

    model_types = sorted(ordinal["model_type"].unique())
    n_models = len(model_types)
    fig, axes = plt.subplots(2, n_models, figsize=(7 * n_models, 11), squeeze=False)

    # Compute z-scores per question (jointly across all sources)
    ordinal = ordinal.copy()
    for qid in ordinal["question_id"].unique():
        mask = ordinal["question_id"] == qid
        # Pool human and LLM values for z-scoring
        all_vals = pd.concat([
            ordinal.loc[mask, "expected_human"],
            ordinal.loc[mask, "expected_llm"],
        ])
        mu, sigma = all_vals.mean(), all_vals.std()
        if sigma > 0:
            ordinal.loc[mask, "z_human"] = (ordinal.loc[mask, "expected_human"] - mu) / sigma
            ordinal.loc[mask, "z_llm"] = (ordinal.loc[mask, "expected_llm"] - mu) / sigma
        else:
            ordinal.loc[mask, "z_human"] = 0.0
            ordinal.loc[mask, "z_llm"] = 0.0

    for row_idx, (x_col, y_col, row_title) in enumerate([
        ("expected_human", "expected_llm", "Raw Expected Value"),
        ("z_human", "z_llm", "Z-Scored Expected Value"),
    ]):
        for idx, mt in enumerate(model_types):
            ax = axes[row_idx, idx]
            sub = ordinal[ordinal["model_type"] == mt]

            for lang in sorted(sub["lang"].unique()):
                lsub = sub[sub["lang"] == lang]
                cluster = LANG_TO_CLUSTER.get(lang, "Other")
                color = CLUSTER_COLORS.get(cluster, "#999999")
                ax.scatter(lsub[x_col], lsub[y_col],
                           c=color, alpha=0.3, s=15, edgecolors="none")

            r_p, _ = pearsonr(sub[x_col], sub[y_col])
            r_s, _ = spearmanr(sub[x_col], sub[y_col])

            # Regression line
            z = np.polyfit(sub[x_col], sub[y_col], 1)
            x_line = np.linspace(sub[x_col].min(), sub[x_col].max(), 100)
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
            ax.set_title(f"{mt_label}\nr={r_p:.3f}, ρ={r_s:.3f} (n={len(sub)})")
            ax.set_xlabel(f"Human {row_title}")
            ax.set_ylabel(f"LLM {row_title}")
            ax.grid(True, alpha=0.3)

    # Cluster legend
    handles = [mpatches.Patch(color=c, label=cl)
               for cl, c in CLUSTER_COLORS.items()]
    axes[0, -1].legend(handles=handles, title="Cultural Cluster",
                       loc="upper left", fontsize=7, framealpha=0.9)

    plt.suptitle("F3: LLM vs Human Expected Values — Raw (top) and Z-Scored (bottom)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = figures_dir / "F3_ev_scatter.png"
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


# ── F4: JSD Heatmap (revised) ────────────────────────────────────────────────

def plot_jsd_heatmap(jsd_df: pd.DataFrame, figures_dir: Path):
    """F4: Heatmap of mean JSD(LLM, human) by model_type × language, with std annotations."""
    filtered = jsd_df[~jsd_df["model_type"].isin(MODELS_EXCLUDE)]

    stats = filtered.groupby(["model_type", "lang"]).agg(
        mean_jsd=("jsd", "mean"),
        std_jsd=("jsd", "std"),
    ).reset_index()

    pivot_mean = stats.pivot_table(
        index="model_type", columns="lang", values="mean_jsd"
    )
    pivot_std = stats.pivot_table(
        index="model_type", columns="lang", values="std_jsd"
    )

    # Sort columns alphabetically, rename
    cols = sorted(pivot_mean.columns)
    pivot_mean = pivot_mean[cols]
    pivot_std = pivot_std[cols]
    pivot_mean.columns = [LANG_NAMES.get(c, c) for c in pivot_mean.columns]
    pivot_std.columns = [LANG_NAMES.get(c, c) for c in pivot_std.columns]
    pivot_mean.index = [MODEL_LABELS.get(i, i) for i in pivot_mean.index]
    pivot_std.index = [MODEL_LABELS.get(i, i) for i in pivot_std.index]

    # Build annotation strings: mean ± std
    annot = pivot_mean.copy().astype(str)
    for r in annot.index:
        for c in annot.columns:
            m = pivot_mean.loc[r, c]
            s = pivot_std.loc[r, c]
            if pd.notna(m):
                annot.loc[r, c] = f"{m:.3f}\n±{s:.3f}"
            else:
                annot.loc[r, c] = ""

    fig, ax = plt.subplots(figsize=(18, max(4, len(pivot_mean) * 1.2)))
    sns.heatmap(
        pivot_mean, annot=annot, fmt="", cmap="RdYlGn_r",
        ax=ax, linewidths=0.5,
        cbar_kws={"label": "Mean JSD to Human"},
        annot_kws={"size": 8},
    )
    ax.set_title("F4: Mean JSD (LLM vs Human) by Model and Language",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Language")
    ax.set_ylabel("Model")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    out = figures_dir / "F4_jsd_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── F5: Per-Question Deep Dive ───────────────────────────────────────────────

def plot_deepdive(
    jsd_df: pd.DataFrame,
    llm_dists: dict,
    human_dists: dict,
    figures_dir: Path,
):
    """F5: 3×3 grid — 3 IW questions + 3 best JSD + 3 worst JSD."""
    filtered = jsd_df[~jsd_df["model_type"].isin(MODELS_EXCLUDE)]

    # Average JSD across all (model_type, lang) for each question
    q_jsd = filtered.groupby("question_id").agg(mean_jsd=("jsd", "mean")).reset_index()
    q_jsd = q_jsd.sort_values("mean_jsd")

    iw_qids = [qid for qid, _ in DEEPDIVE_IW_QUESTIONS]
    best_candidates = [qid for qid in q_jsd.head(10)["question_id"] if qid not in iw_qids]
    worst_candidates = [qid for qid in q_jsd.tail(10)["question_id"].iloc[::-1] if qid not in iw_qids]

    best_3 = best_candidates[:3]
    worst_3 = worst_candidates[:3]
    all_questions = iw_qids + best_3 + worst_3

    # Find best-performing model per question (lowest mean JSD excluding MODELS_EXCLUDE)
    best_model_per_q = {}
    for qid in all_questions:
        q_sub = filtered[filtered["question_id"] == qid]
        if q_sub.empty:
            continue
        mt_jsd = q_sub.groupby("model_type")["jsd"].mean()
        best_model_per_q[qid] = mt_jsd.idxmin()

    # IW question labels
    iw_labels = {qid: text for qid, text in DEEPDIVE_IW_QUESTIONS}

    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    axes = axes.flatten()

    category_colors = {}
    for qid in iw_qids:
        category_colors[qid] = "#3366cc"  # blue for IW
    for qid in best_3:
        category_colors[qid] = "#33aa33"  # green for best
    for qid in worst_3:
        category_colors[qid] = "#cc3333"  # red for worst

    for idx, qid in enumerate(all_questions):
        ax = axes[idx]
        best_mt = best_model_per_q.get(qid, "hplt2c")
        mean_jsd_val = q_jsd[q_jsd["question_id"] == qid]["mean_jsd"]
        mean_jsd_val = mean_jsd_val.iloc[0] if len(mean_jsd_val) > 0 else np.nan

        n_langs_plotted = 0
        for lang_idx, lang in enumerate(DEEPDIVE_LANGS):
            human_key = (lang, qid)
            llm_key = (best_mt, lang, qid)

            if human_key not in human_dists:
                continue
            human_d = human_dists[human_key]
            all_vals = sorted(human_d.keys())

            human_arr = np.array([human_d.get(v, 0) for v in all_vals])
            if human_arr.sum() > 0:
                human_arr = human_arr / human_arr.sum()

            x = np.arange(len(all_vals))
            width = 0.8 / (2 * len(DEEPDIVE_LANGS))
            offset = (lang_idx - (len(DEEPDIVE_LANGS) - 1) / 2) * width * 2

            lang_name = LANG_NAMES.get(lang, lang)

            # Human bars: thick outlined bars
            ax.bar(x + offset - width / 2, human_arr, width,
                   facecolor="none", edgecolor=CLUSTER_COLORS.get(
                       LANG_TO_CLUSTER.get(lang, "Other"), "#999999"),
                   linewidth=2.0, label=f"Human {lang_name}" if idx == 0 else None)

            # LLM bars: filled thinner bars
            if llm_key in llm_dists:
                llm_d = llm_dists[llm_key]
                llm_arr = np.array([llm_d.get(v, 0) for v in all_vals])
                if llm_arr.sum() > 0:
                    llm_arr = llm_arr / llm_arr.sum()
                ax.bar(x + offset + width / 2, llm_arr, width,
                       color=CLUSTER_COLORS.get(
                           LANG_TO_CLUSTER.get(lang, "Other"), "#999999"),
                       alpha=0.6,
                       label=f"LLM {lang_name}" if idx == 0 else None)

            n_langs_plotted += 1

        ax.set_xticks(np.arange(len(all_vals)))
        ax.set_xticklabels(all_vals, fontsize=7)
        ax.set_ylabel("Probability", fontsize=8)
        ax.grid(True, alpha=0.2, axis="y")

        # Title with color coding
        if qid in iw_labels:
            subtitle = iw_labels[qid]
            category = "IW"
        elif qid in best_3:
            subtitle = f"Best aligned (JSD={mean_jsd_val:.3f})"
            category = "BEST"
        else:
            subtitle = f"Worst aligned (JSD={mean_jsd_val:.3f})"
            category = "WORST"

        title_color = category_colors.get(qid, "black")
        ax.set_title(f"{qid}: {subtitle}\n[{category}] model={MODEL_LABELS.get(best_mt, best_mt)}",
                     fontsize=9, color=title_color, fontweight="bold")

    # Add shared legend
    if len(all_questions) > 0:
        legend_elements = []
        for lang in DEEPDIVE_LANGS:
            color = CLUSTER_COLORS.get(LANG_TO_CLUSTER.get(lang, "Other"), "#999999")
            lang_name = LANG_NAMES.get(lang, lang)
            legend_elements.append(
                mpatches.Patch(facecolor="none", edgecolor=color,
                               linewidth=2, label=f"Human {lang_name}"))
            legend_elements.append(
                mpatches.Patch(facecolor=color, alpha=0.6,
                               label=f"LLM {lang_name}"))
        fig.legend(handles=legend_elements, loc="lower center",
                   ncol=len(DEEPDIVE_LANGS) * 2, fontsize=7, framealpha=0.9)

    plt.suptitle("F5: Per-Question Deep Dive — IW (blue), Best (green), Worst (red)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    out = figures_dir / "F5_deepdive.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Legacy figures (kept for supplementary) ──────────────────────────────────

def plot_jsd_by_qtype(jsd_df: pd.DataFrame, figures_dir: Path):
    """Box/violin plots of JSD to human by response type and model type."""
    filtered = jsd_df[~jsd_df["model_type"].isin(MODELS_EXCLUDE)]
    fig, ax = plt.subplots(figsize=(10, 6))

    order = sorted(filtered["response_type"].unique())
    sns.boxplot(
        data=filtered, x="response_type", y="jsd", hue="model_type",
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


def plot_model_comparison(jsd_df: pd.DataFrame, figures_dir: Path):
    """Grouped bar chart comparing all model types' mean JSD to human per language."""
    filtered = jsd_df[~jsd_df["model_type"].isin(MODELS_EXCLUDE)]
    lang_jsd = filtered.groupby(["model_type", "lang"]).agg(
        mean_jsd=("jsd", "mean"),
    ).reset_index()

    model_types = sorted(lang_jsd["model_type"].unique())
    per_model_langs = [set(lang_jsd[lang_jsd["model_type"] == mt]["lang"]) for mt in model_types]
    shared_langs = sorted(set.intersection(*per_model_langs)) if per_model_langs else []

    if not shared_langs:
        print("  No shared languages for model comparison")
        return

    model_series = {}
    for mt in model_types:
        sub = lang_jsd[lang_jsd["model_type"] == mt].set_index("lang")["mean_jsd"]
        model_series[mt] = sub

    n_models = len(model_types)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

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

    # Win count bar chart
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


# ── CLI ──────────────────────────────────────────────────────────────────────

def run_figure(
    figure: str,
    jsd_df: pd.DataFrame,
    llm_dists: dict,
    human_dists: dict,
    questions: dict,
    figures_dir: Path,
):
    """Dispatch to individual figure generators."""
    if figure in ("all", "umap"):
        print("\n  Generating F1: Human-Fitted UMAP...")
        try:
            h_coords, l_coords, h_labels, l_labels = compute_human_fitted_umap(
                llm_dists, human_dists, questions
            )
            plot_human_fitted_umap(h_coords, l_coords, h_labels, l_labels, figures_dir)
        except ImportError:
            print("  Skipping F1: umap-learn not installed")

    if figure in ("all", "iw"):
        print("\n  Generating F2: Inglehart-Welzel Composite...")
        iw_df = compute_iw_composites(llm_dists, human_dists, questions)
        if not iw_df.empty:
            plot_iw_composite(iw_df, figures_dir)
        else:
            print("  Skipping F2: insufficient IW question data")

    if figure in ("all", "scatter"):
        print("\n  Generating F3: EV Scatter + Z-Score...")
        plot_scatter_ev(jsd_df, figures_dir)

    if figure in ("all", "heatmap"):
        print("\n  Generating F4: JSD Heatmap...")
        plot_jsd_heatmap(jsd_df, figures_dir)

    if figure in ("all", "deepdive"):
        print("\n  Generating F5: Deep Dive...")
        plot_deepdive(jsd_df, llm_dists, human_dists, figures_dir)

    # Also generate legacy supplementary figures when running all
    if figure == "all":
        print("\n  Generating supplementary figures...")
        plot_jsd_by_qtype(jsd_df, figures_dir)
        plot_model_comparison(jsd_df, figures_dir)


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
    parser.add_argument("--db", type=str, default=None,
                        help="Path to survey.db (loads LLM + human data from DB)")
    parser.add_argument("--figure", type=str, default="all",
                        choices=["all", "umap", "iw", "scatter", "heatmap", "deepdive"],
                        help="Which figure to generate (default: all)")
    args = parser.parse_args()

    args.figures_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    llm_df = load_llm_results(args.results_dir, db_path=args.db)
    if args.db:
        from db.load import load_human_distributions
        human_df = load_human_distributions(args.db)
    else:
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
    run_figure(args.figure, jsd_df, llm_dists, human_dists, questions, args.figures_dir)

    # Save JSD data for further analysis
    out = args.figures_dir.parent / "human_comparison.parquet"
    jsd_df.to_parquet(out, index=False)
    print(f"\nSaved comparison data to {out}")

    print("\nDone!")


if __name__ == "__main__":
    main()
