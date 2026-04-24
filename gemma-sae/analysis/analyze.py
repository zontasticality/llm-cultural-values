#!/usr/bin/env python3
"""Analysis pipeline for cultural completion comparison.

Usage:
    PYTHONPATH=. python -m analysis.analyze categories --db data/culture.db
    PYTHONPATH=. python -m analysis.analyze known_groups --db data/culture.db
    PYTHONPATH=. python -m analysis.analyze all --db data/culture.db
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from analysis.constants import (
    LANG_NAMES, CULTURAL_CLUSTERS, LANG_TO_CLUSTER, CLUSTER_COLORS,
    MODEL_LABELS, CONTENT_CATEGORIES, CULTURAL_DIMENSIONS, PILOT_LANGS,
    TRIMMED_VARIANT_MIN,
)
from db.load import load_results
from db.schema import get_connection


# ── Data loading ─────────────────────────────────────────────────

def load_pilot_data(db_path: str, classifier: str = "gpt-4.1-mini") -> pd.DataFrame:
    """Load classified pilot results into a DataFrame."""
    conn = get_connection(db_path)
    rows = load_results(conn, classifier_model=classifier)
    conn.close()
    df = pd.DataFrame(rows)
    # Add cluster info
    df["cluster"] = df["lang"].map(LANG_TO_CLUSTER)
    # Friendly model labels
    df["model_label"] = df["model_id"].map(
        lambda m: MODEL_LABELS.get(m, MODEL_LABELS.get(m.rsplit("_", 1)[0], m))
    )
    print(f"Loaded {len(df)} classified completions")
    print(f"  Models: {sorted(df['model_id'].unique())}")
    print(f"  Languages: {sorted(df['lang'].unique())}")
    print(f"  Templates: {sorted(df['template_id'].unique())}")
    return df


# ── Categories analysis ──────────────────────────────────────────

def analyze_categories(df: pd.DataFrame, fig_dir: Path):
    """Content category distributions per language, chi-square tests."""
    print("\n" + "="*70)
    print("CONTENT CATEGORY ANALYSIS")
    print("="*70)

    # ── Per-language category distributions (primary models only) ──
    for model_id in sorted(df["model_id"].unique()):
        mdf = df[df["model_id"] == model_id]
        if mdf.empty:
            continue

        print(f"\n--- {model_id} ---")

        # Category proportions per language
        ct = pd.crosstab(mdf["lang"], mdf["content_category"], normalize="index")
        # Reorder columns to match CONTENT_CATEGORIES
        cols = [c for c in CONTENT_CATEGORIES if c in ct.columns]
        ct = ct.reindex(columns=cols, fill_value=0)
        print(ct.round(3).to_string())

        # Stacked bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        ct.plot(kind="bar", stacked=True, ax=ax, colormap="Set2")
        ax.set_title(f"Content Categories — {model_id}")
        ax.set_xlabel("Language")
        ax.set_ylabel("Proportion")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        ax.set_xticklabels([LANG_NAMES.get(l, l) for l in ct.index], rotation=45, ha="right")
        plt.tight_layout()
        fig.savefig(fig_dir / f"categories_{model_id}.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: {fig_dir}/categories_{model_id}.png")

    # ── Chi-square: between IW clusters ──
    print(f"\n--- Chi-square tests (between IW clusters) ---")

    # Use 27B as the primary model for cross-cluster tests
    for model_id in ["gemma3_27b_pt", "gemma3_12b_pt", "eurollm22b"]:
        mdf = df[(df["model_id"] == model_id) & (df["cluster"].notna())]
        if mdf.empty:
            continue

        clusters_present = sorted(mdf["cluster"].unique())
        if len(clusters_present) < 2:
            continue

        print(f"\n  {model_id} (clusters: {clusters_present})")

        # Overall chi-square across all clusters
        ct_raw = pd.crosstab(mdf["cluster"], mdf["content_category"])
        chi2, p, dof, expected = chi2_contingency(ct_raw)
        print(f"  Overall χ²={chi2:.1f}, df={dof}, p={p:.2e}")
        if p < 0.01:
            print(f"  *** GO/NO-GO GATE 2 PASSES: p < 0.01 ***")

        # Pairwise chi-square between clusters
        for i, c1 in enumerate(clusters_present):
            for c2 in clusters_present[i+1:]:
                pair_df = mdf[mdf["cluster"].isin([c1, c2])]
                ct_pair = pd.crosstab(pair_df["cluster"], pair_df["content_category"])
                if ct_pair.shape[0] < 2:
                    continue
                chi2_p, p_p, dof_p, _ = chi2_contingency(ct_pair)
                sig = "***" if p_p < 0.01 else "**" if p_p < 0.05 else ""
                print(f"    {c1} vs {c2}: χ²={chi2_p:.1f}, p={p_p:.2e} {sig}")

    # ── Cross-model comparison for same language ──
    print(f"\n--- Cross-model comparison (same language) ---")
    for lang in PILOT_LANGS:
        lang_df = df[df["lang"] == lang]
        models = sorted(lang_df["model_id"].unique())
        if len(models) < 2:
            continue
        ct_model = pd.crosstab(lang_df["model_id"], lang_df["content_category"])
        if ct_model.shape[0] < 2:
            continue
        chi2, p, dof, _ = chi2_contingency(ct_model)
        sig = "***" if p < 0.01 else "**" if p < 0.05 else ""
        print(f"  {LANG_NAMES.get(lang, lang)}: χ²={chi2:.1f}, p={p:.2e} {sig} (models: {models})")


# ── Known groups analysis ────────────────────────────────────────

def cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return float("nan")
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return float("nan")
    return (m1 - m2) / pooled_std


def analyze_known_groups(df: pd.DataFrame, fig_dir: Path):
    """Cohen's d for expected cultural contrasts."""
    print("\n" + "="*70)
    print("KNOWN-GROUP CONTRASTS")
    print("="*70)

    contrasts = [
        ("Finnish", "Romanian", "fin", "ron",
         "Protestant vs Orthodox — expect fin more secular"),
        ("Finnish", "Chinese", "fin", "zho",
         "Nordic vs East Asian — expect cultural distance"),
        ("English", "Chinese", "eng", "zho",
         "English-speaking vs East Asian"),
        ("English", "Polish", "eng", "pol",
         "English-speaking vs Catholic Europe"),
        ("Polish", "Romanian", "pol", "ron",
         "Catholic vs Orthodox"),
    ]

    dim_labels = {
        "dim_indiv_collect": "Individualist↔Collectivist",
        "dim_trad_secular": "Traditional↔Secular",
        "dim_surv_selfexpr": "Survival↔Self-Expression",
    }

    for model_id in ["gemma3_27b_pt", "gemma3_12b_pt", "eurollm22b"]:
        mdf = df[df["model_id"] == model_id]
        if mdf.empty:
            continue

        print(f"\n--- {model_id} ---")
        print(f"{'Contrast':45s} {'IC d':>8s} {'TS d':>8s} {'SS d':>8s}")
        print("-" * 75)

        results = []
        for name1, name2, lang1, lang2, desc in contrasts:
            g1 = mdf[mdf["lang"] == lang1]
            g2 = mdf[mdf["lang"] == lang2]
            if g1.empty or g2.empty:
                continue

            ds = {}
            for dim in CULTURAL_DIMENSIONS:
                d = cohens_d(g1[dim].values, g2[dim].values)
                ds[dim] = d

            flag = ""
            if dim == "dim_trad_secular" and lang1 == "fin" and lang2 == "ron":
                if abs(ds.get("dim_trad_secular", 0)) > 0.3:
                    flag = " *** GO/NO-GO GATE 1 PASSES ***"

            print(f"  {name1} vs {name2:10s} ({desc:30s})"
                  f" {ds.get('dim_indiv_collect', float('nan')):8.3f}"
                  f" {ds.get('dim_trad_secular', float('nan')):8.3f}"
                  f" {ds.get('dim_surv_selfexpr', float('nan')):8.3f}"
                  f"{flag}")

            results.append({
                "model": model_id, "lang1": lang1, "lang2": lang2,
                **{f"d_{k.split('_', 1)[1]}": v for k, v in ds.items()},
            })

        # Mean dimension scores per language
        print(f"\n  Mean dimension scores per language:")
        print(f"  {'Lang':5s} {'IC':>6s} {'TS':>6s} {'SS':>6s} {'n':>5s}")
        for lang in sorted(mdf["lang"].unique()):
            ldf = mdf[mdf["lang"] == lang]
            print(f"  {lang:5s}"
                  f" {ldf['dim_indiv_collect'].mean():6.2f}"
                  f" {ldf['dim_trad_secular'].mean():6.2f}"
                  f" {ldf['dim_surv_selfexpr'].mean():6.2f}"
                  f" {len(ldf):5d}")

    # ── Effect size bar chart ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for model_id, ax in zip(["gemma3_27b_pt", "gemma3_12b_pt", "eurollm22b"], axes):
        mdf = df[df["model_id"] == model_id]
        if mdf.empty:
            ax.set_visible(False)
            continue

        contrast_labels = []
        d_values = {dim: [] for dim in CULTURAL_DIMENSIONS}

        for name1, name2, lang1, lang2, desc in contrasts:
            g1 = mdf[mdf["lang"] == lang1]
            g2 = mdf[mdf["lang"] == lang2]
            if g1.empty or g2.empty:
                continue
            contrast_labels.append(f"{lang1}v{lang2}")
            for dim in CULTURAL_DIMENSIONS:
                d_values[dim].append(cohens_d(g1[dim].values, g2[dim].values))

        if not contrast_labels:
            ax.set_visible(False)
            continue

        x = np.arange(len(contrast_labels))
        width = 0.25
        for i, dim in enumerate(CULTURAL_DIMENSIONS):
            short = dim.split("_", 1)[1][:2].upper()
            ax.bar(x + i * width, d_values[dim], width, label=short)

        ax.axhline(y=0.3, color="red", linestyle="--", alpha=0.5, label="d=0.3 threshold")
        ax.axhline(y=-0.3, color="red", linestyle="--", alpha=0.5)
        ax.set_xlabel("Contrast")
        ax.set_ylabel("Cohen's d")
        ax.set_title(model_id)
        ax.set_xticks(x + width)
        ax.set_xticklabels(contrast_labels, rotation=45, ha="right")
        ax.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(fig_dir / "known_groups_cohens_d.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved: {fig_dir}/known_groups_cohens_d.png")


# ── Main ─────────────────────────────────────────────────────────

SUBCOMMANDS = {
    "categories": analyze_categories,
    "known_groups": analyze_known_groups,
}


def main():
    parser = argparse.ArgumentParser(description="Cultural completion analysis")
    parser.add_argument("subcommand", choices=[*SUBCOMMANDS.keys(), "all"])
    parser.add_argument("--db", required=True)
    parser.add_argument("--classifier", default="gpt-4.1-mini")
    parser.add_argument("--figure-dir", default="figures/diagnostic")
    parser.add_argument("--trimmed-only", action="store_true",
                        help=f"Only include trimmed prompts (variant_idx >= {TRIMMED_VARIANT_MIN})")
    parser.add_argument("--lang-match", action="store_true",
                        help="Drop completions whose detected_lang != prompts.lang")
    args = parser.parse_args()

    fig_dir = Path(args.figure_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = load_pilot_data(args.db, classifier=args.classifier)

    if args.trimmed_only:
        df = df[df["variant_idx"] >= TRIMMED_VARIANT_MIN].copy()
        print(f"Filtered to trimmed prompts: {len(df)} rows")

    if args.lang_match:
        before = len(df)
        mask = df["detected_lang"].isna() | (df["detected_lang"] == df["lang"])
        df = df[mask].copy()
        print(f"Lang-match filter: {before} → {len(df)} rows ({before-len(df)} dropped)")

    if args.subcommand == "all":
        for name, fn in SUBCOMMANDS.items():
            fn(df, fig_dir)
    else:
        SUBCOMMANDS[args.subcommand](df, fig_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
