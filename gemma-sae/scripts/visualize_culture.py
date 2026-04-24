"""Visualize cultural signal in LLM completions.

Three figures:
  1. IW Cultural Map (LLM vs Human) — side-by-side TS×SE scatter
  2. Language-similarity dendrogram — hierarchical clustering from category JSD
  3. Per-question scatter grid — 8 templates × 10 EVS questions, mini-scatters

Usage:
    PYTHONPATH=. python scripts/visualize_culture.py \
        --llm-db data/culture.db \
        --human-db ../eurollm/data/survey.db \
        --classifier gemma3_27b_it_v2 \
        [--lang-match] \
        [--outdir figures/trimmed]
"""
import argparse
import json
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

from analysis.constants import (
    CONTENT_CATEGORIES, EVS_COUNTRY_NAMES, IW_TRADITIONAL_SECULAR,
    IW_SURVIVAL_SELFEXPR, LANG_NAMES, LANG_TO_CLUSTER, TRIMMED_VARIANT_MIN,
)

CLUSTER_COLORS = {
    "Protestant Europe": "#1f77b4",
    "Catholic Europe": "#2ca02c",
    "English-speaking": "#ff7f0e",
    "Orthodox": "#d62728",
    "Baltic": "#9467bd",
    "South Asian": "#8c564b",
    "East Asian": "#e377c2",
    "Middle Eastern": "#bcbd22",
}

MULTILINGUAL_MODELS = ["gemma3_27b_pt", "gemma3_12b_pt", "eurollm22b"]
MODEL_LABELS = {
    "gemma3_27b_pt": "Gemma-3-27B",
    "gemma3_12b_pt": "Gemma-3-12B",
    "eurollm22b": "EuroLLM-22B",
}


# ── Shared data loading ──────────────────────────────────────────

def load_llm_data(db_path, classifier, lang_match=False):
    db = sqlite3.connect(db_path)
    lf = "AND (c.detected_lang IS NULL OR c.detected_lang = p.lang)" if lang_match else ""
    rows = db.execute(f"""
        SELECT c.model_id, p.lang, p.template_id, p.variant_idx,
               cl.content_category,
               cl.dim_trad_secular, cl.dim_surv_selfexpr,
               cl.dim_indiv_collect
        FROM classifications cl
        JOIN completions c ON cl.completion_id = c.completion_id
        JOIN prompts p ON c.prompt_id = p.prompt_id
        WHERE cl.classifier_model = ?
          AND p.variant_idx >= {TRIMMED_VARIANT_MIN}
          AND c.filter_status = 'ok'
          {lf}
    """, (classifier,)).fetchall()
    db.close()
    cols = ["model_id", "lang", "template_id", "variant_idx",
            "content_category", "dim_ts", "dim_ss", "dim_ic"]
    return pd.DataFrame(rows, columns=cols)


def load_human_iw(human_db):
    db = sqlite3.connect(human_db)
    all_qs = {**IW_TRADITIONAL_SECULAR, **IW_SURVIVAL_SELFEXPR}
    ev_rows = []
    for qid, info in all_qs.items():
        for lang in EVS_COUNTRY_NAMES.keys():
            rows = db.execute(
                "SELECT response_value, prob_human FROM human_distributions "
                "WHERE lang = ? AND question_id = ? ORDER BY response_value",
                (lang, qid),
            ).fetchall()
            if not rows:
                continue
            vals = np.array([r[0] for r in rows], dtype=float)
            probs = np.array([r[1] for r in rows])
            total = probs.sum()
            if total > 0:
                probs /= total
            ev = np.dot(vals, probs)
            if info.get("flip") and info.get("max_val") is not None:
                ev = info["max_val"] - ev
            dim = "TS" if qid in IW_TRADITIONAL_SECULAR else "SE"
            ev_rows.append({"lang": lang, "qid": qid, "ev": ev, "dimension": dim})
    db.close()
    ev_df = pd.DataFrame(ev_rows)
    if ev_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    ts_qs = [q for q in IW_TRADITIONAL_SECULAR]
    se_qs = [q for q in IW_SURVIVAL_SELFEXPR]

    composites = []
    for lang in ev_df["lang"].unique():
        ldf = ev_df[ev_df["lang"] == lang]
        ts_vals = ldf[ldf["qid"].isin(ts_qs)]["ev"]
        se_vals = ldf[ldf["qid"].isin(se_qs)]["ev"]
        if len(ts_vals) >= 3 and len(se_vals) >= 3:
            composites.append({
                "lang": lang,
                "human_ts": (ts_vals - ts_vals.mean()).values.mean(),
                "human_se": (se_vals - se_vals.mean()).values.mean(),
            })
    # z-score per question then average
    z_rows = []
    for dim_qs, dim_name in [(ts_qs, "TS"), (se_qs, "SE")]:
        for qid in dim_qs:
            qdf = ev_df[ev_df["qid"] == qid]
            if len(qdf) < 2:
                continue
            mu, sd = qdf["ev"].mean(), qdf["ev"].std()
            for _, row in qdf.iterrows():
                z = (row["ev"] - mu) / sd if sd > 0 else 0
                z_rows.append({"lang": row["lang"], "qid": qid, "ev_z": z, "dimension": dim_name})
    z_df = pd.DataFrame(z_rows)
    comp_df = z_df.groupby(["lang", "dimension"])["ev_z"].mean().unstack(fill_value=0).reset_index()
    comp_df.columns = ["lang", "human_se", "human_ts"]

    per_q = ev_df.copy()
    for qid in per_q["qid"].unique():
        mask = per_q["qid"] == qid
        mu, sd = per_q.loc[mask, "ev"].mean(), per_q.loc[mask, "ev"].std()
        per_q.loc[mask, "ev_z"] = (per_q.loc[mask, "ev"] - mu) / sd if sd > 0 else 0

    return comp_df, per_q


# ── Figure 1: IW Cultural Map ────────────────────────────────────

def fig_iw_map(df, human, outdir):
    fig, axes = plt.subplots(1, 4, figsize=(28, 7))

    # Human map
    ax = axes[0]
    ax.set_title("Human (EVS 2017)", fontsize=14, fontweight="bold")
    for _, row in human.iterrows():
        lang = row["lang"]
        cluster = LANG_TO_CLUSTER.get(lang, "Other")
        color = CLUSTER_COLORS.get(cluster, "gray")
        ax.scatter(row["human_ts"], row["human_se"], c=color, s=80, zorder=3)
        ax.annotate(lang.upper(), (row["human_ts"], row["human_se"]),
                    fontsize=7, ha="center", va="bottom", xytext=(0, 4),
                    textcoords="offset points")
    ax.set_xlabel("Traditional → Secular (z-scored)")
    ax.set_ylabel("Survival → Self-Expression (z-scored)")
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.axvline(0, color="gray", lw=0.5, ls="--")
    ax.grid(True, alpha=0.2)

    # LLM maps
    for i, model in enumerate(MULTILINGUAL_MODELS):
        ax = axes[i + 1]
        mdf = df[df["model_id"] == model]
        lang_means = mdf.groupby("lang")[["dim_ts", "dim_ss"]].mean()
        # z-score the LLM means to match human scale
        for col in ["dim_ts", "dim_ss"]:
            mu, sd = lang_means[col].mean(), lang_means[col].std()
            if sd > 0:
                lang_means[col + "_z"] = (lang_means[col] - mu) / sd
            else:
                lang_means[col + "_z"] = 0
        ax.set_title(MODEL_LABELS.get(model, model), fontsize=14, fontweight="bold")
        # Only plot EU langs that appear in human
        eu_langs = set(human["lang"])
        for lang in lang_means.index:
            if lang not in eu_langs:
                continue
            cluster = LANG_TO_CLUSTER.get(lang, "Other")
            color = CLUSTER_COLORS.get(cluster, "gray")
            ax.scatter(lang_means.loc[lang, "dim_ts_z"],
                       lang_means.loc[lang, "dim_ss_z"],
                       c=color, s=80, zorder=3)
            ax.annotate(lang.upper(), (lang_means.loc[lang, "dim_ts_z"],
                                       lang_means.loc[lang, "dim_ss_z"]),
                        fontsize=7, ha="center", va="bottom", xytext=(0, 4),
                        textcoords="offset points")
        ax.set_xlabel("Traditional → Secular (z-scored)")
        if i == 0:
            ax.set_ylabel("Survival → Self-Expression (z-scored)")
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.axvline(0, color="gray", lw=0.5, ls="--")
        ax.grid(True, alpha=0.2)

        # Spearman correlation annotation
        merged = human.merge(
            lang_means[["dim_ts_z", "dim_ss_z"]].reset_index(),
            on="lang", how="inner")
        if len(merged) >= 5:
            rho_ts, p_ts = spearmanr(merged["human_ts"], merged["dim_ts_z"])
            rho_se, p_se = spearmanr(merged["human_se"], merged["dim_ss_z"])
            ax.text(0.03, 0.97,
                    f"TS: ρ={rho_ts:.2f} {'*' if p_ts<0.05 else ''}\n"
                    f"SE: ρ={rho_se:.2f} {'**' if p_se<0.01 else '*' if p_se<0.05 else ''}",
                    transform=ax.transAxes, fontsize=9, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Legend
    patches = [mpatches.Patch(color=c, label=cl) for cl, c in CLUSTER_COLORS.items()]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Inglehart–Welzel Cultural Map: Human vs LLM", fontsize=16, y=1.02)
    plt.tight_layout()
    path = outdir / "iw_cultural_map.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Figure 2: Language-similarity dendrogram ─────────────────────

def fig_dendrogram(df, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    for ax, model in zip(axes, MULTILINGUAL_MODELS):
        mdf = df[df["model_id"] == model]
        # Build category distribution per language
        ct = pd.crosstab(mdf["lang"], mdf["content_category"], normalize="index")
        cats = [c for c in CONTENT_CATEGORIES if c in ct.columns]
        ct = ct.reindex(columns=cats, fill_value=0)

        if len(ct) < 3:
            ax.text(0.5, 0.5, "Not enough languages", ha="center", va="center")
            continue

        # JSD distance matrix
        from scipy.spatial.distance import jensenshannon
        langs = ct.index.tolist()
        n = len(langs)
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = jensenshannon(ct.iloc[i].values, ct.iloc[j].values)
                dist[i, j] = dist[j, i] = d

        condensed = squareform(dist)
        Z = linkage(condensed, method="ward")

        # Color leaves by cluster
        lang_colors = {l: CLUSTER_COLORS.get(LANG_TO_CLUSTER.get(l, ""), "gray") for l in langs}
        dendro = dendrogram(Z, labels=[l.upper() for l in langs], ax=ax,
                           leaf_rotation=90, leaf_font_size=9)
        # Color x-tick labels
        for lbl in ax.get_xticklabels():
            lang_code = lbl.get_text().lower()
            lbl.set_color(lang_colors.get(lang_code, "black"))
            lbl.set_fontweight("bold")

        ax.set_title(MODEL_LABELS.get(model, model), fontsize=14, fontweight="bold")
        ax.set_ylabel("Ward distance (JSD)")

    patches = [mpatches.Patch(color=c, label=cl) for cl, c in CLUSTER_COLORS.items()]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Language Clustering by Content-Category Distribution (JSD + Ward)",
                 fontsize=16, y=1.02)
    plt.tight_layout()
    path = outdir / "lang_dendrogram.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Figure 3: Per-question scatter grid ──────────────────────────

def fig_question_grid(df, human_per_q, outdir):
    templates = sorted(df["template_id"].unique())
    all_qs = {**IW_TRADITIONAL_SECULAR, **IW_SURVIVAL_SELFEXPR}
    q_order = list(IW_TRADITIONAL_SECULAR.keys()) + list(IW_SURVIVAL_SELFEXPR.keys())
    q_labels = {qid: info.get("label", qid) for qid, info in all_qs.items()}

    # Use EuroLLM only (strongest signal)
    edf = df[df["model_id"] == "eurollm22b"]

    n_tmpl = len(templates)
    n_q = len(q_order)
    fig, axes = plt.subplots(n_tmpl, n_q, figsize=(n_q * 2.5, n_tmpl * 2.5),
                              squeeze=False)

    eu_langs = set(human_per_q["lang"].unique())

    for row_i, tmpl in enumerate(templates):
        tdf = edf[edf["template_id"] == tmpl]
        # Mean dim scores per language for this template
        lang_ts = tdf.groupby("lang")["dim_ts"].mean()
        lang_se = tdf.groupby("lang")["dim_ss"].mean()

        for col_j, qid in enumerate(q_order):
            ax = axes[row_i, col_j]
            # Human z-scored values for this question
            hq = human_per_q[human_per_q["qid"] == qid][["lang", "ev_z"]].copy()
            dim = "TS" if qid in IW_TRADITIONAL_SECULAR else "SE"
            llm_series = lang_ts if dim == "TS" else lang_se

            merged = hq.merge(
                llm_series.reset_index().rename(columns={llm_series.name: "llm"}),
                on="lang", how="inner"
            )
            merged = merged[merged["lang"].isin(eu_langs)]

            if len(merged) >= 5:
                rho, p = spearmanr(merged["ev_z"], merged["llm"])
                # Color by significance
                if p < 0.01:
                    bg = "#d4edda"
                elif p < 0.05:
                    bg = "#fff3cd"
                else:
                    bg = "#f8f9fa"
                ax.set_facecolor(bg)

                for _, r in merged.iterrows():
                    cluster = LANG_TO_CLUSTER.get(r["lang"], "")
                    color = CLUSTER_COLORS.get(cluster, "gray")
                    ax.scatter(r["ev_z"], r["llm"], c=color, s=15, alpha=0.7)

                ax.text(0.05, 0.95, f"ρ={rho:.2f}", transform=ax.transAxes,
                        fontsize=7, va="top",
                        fontweight="bold" if p < 0.05 else "normal")
            else:
                ax.set_facecolor("#f0f0f0")
                ax.text(0.5, 0.5, "n<5", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7, color="gray")

            ax.tick_params(labelsize=5)
            if row_i == 0:
                label = q_labels.get(qid, qid)
                if len(label) > 15:
                    label = label[:14] + "…"
                ax.set_title(label, fontsize=7, rotation=30, ha="left")
            if col_j == 0:
                ax.set_ylabel(tmpl, fontsize=8)
            if row_i < n_tmpl - 1:
                ax.set_xticklabels([])
            if col_j > 0:
                ax.set_yticklabels([])

    # Divider between TS and SE questions
    n_ts = len(IW_TRADITIONAL_SECULAR)
    for row_i in range(n_tmpl):
        axes[row_i, n_ts - 1].axvline(
            axes[row_i, n_ts - 1].get_xlim()[1], color="black", lw=2)

    fig.suptitle("EuroLLM-22B: Template × EVS Question Correlations\n"
                 "(green=p<0.01, yellow=p<0.05, gray=n.s.)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    path = outdir / "question_scatter_grid.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Main ─────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-db", required=True)
    ap.add_argument("--human-db", required=True)
    ap.add_argument("--classifier", default="gemma3_27b_it_v2")
    ap.add_argument("--lang-match", action="store_true")
    ap.add_argument("--outdir", default="figures/trimmed")
    args = ap.parse_args()

    from pathlib import Path
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading LLM data...", flush=True)
    df = load_llm_data(args.llm_db, args.classifier, lang_match=args.lang_match)
    print(f"  {len(df)} rows, {df['model_id'].nunique()} models, {df['lang'].nunique()} langs")

    print("Loading human EVS data...", flush=True)
    human, human_per_q = load_human_iw(args.human_db)
    print(f"  {len(human)} languages")

    print("\n1/3: IW Cultural Map...")
    fig_iw_map(df, human, outdir)

    print("\n2/3: Language dendrogram...")
    fig_dendrogram(df, outdir)

    print("\n3/3: Per-question scatter grid...")
    fig_question_grid(df, human_per_q, outdir)

    print("\nDone.")


if __name__ == "__main__":
    main()
