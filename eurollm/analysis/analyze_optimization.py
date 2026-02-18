"""Analyze prompt optimization grid search results.

Reads grid search parquets from optimize_prompts.py and produces:
    1. optimization_main_effects.png  — Mean JSD reduction per format dimension
    2. optimization_best_by_qtype.png — Best format per response type
    3. optimization_heatmap.png       — Full grid JSD heatmap

Usage:
    python eurollm/analysis/analyze_optimization.py \
        --results-dir eurollm/results/optimization \
        --figures-dir eurollm/figures
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ── Format dimension labels ───────────────────────────────────────────────────

DIMENSION_NAMES = {
    "cue_style": "Answer Cue",
    "opt_format": "Option Format",
    "scale_hint": "Scale Hint",
    "embed_style": "Embed Style",
    "n_perms": "Permutations",
}

CONFIG_COLS = ["cue_style", "opt_format", "scale_hint", "embed_style", "n_perms"]


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_grid_results(results_dir: Path) -> pd.DataFrame:
    """Load all grid search parquets from optimization directory."""
    frames = []
    for path in sorted(results_dir.glob("*.parquet")):
        df = pd.read_parquet(path)
        # Extract model and lang from filename: e.g. hplt2c_eng_grid.parquet
        stem = path.stem
        parts = stem.rsplit("_grid", 1)
        if len(parts) == 2:
            model_lang = parts[0]
            ml_parts = model_lang.split("_", 1)
            if len(ml_parts) == 2:
                df["model_type"] = ml_parts[0]
                df["lang"] = ml_parts[1]
        frames.append(df)

    if not frames:
        print(f"ERROR: No grid search parquets found in {results_dir}")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(combined)} rows from {len(frames)} grid search files")
    return combined


def aggregate_to_config_level(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate from per-value rows to one row per (config, question, model, lang)."""
    group_cols = CONFIG_COLS + ["question_id", "response_type"]
    if "model_type" in df.columns:
        group_cols += ["model_type", "lang"]

    agg = df.groupby(group_cols).agg(
        jsd=("jsd_to_human", "first"),
        p_valid=("p_valid", "first"),
        position_bias=("position_bias", "first"),
    ).reset_index()

    return agg


# ── Analysis 1: Main Effects ─────────────────────────────────────────────────

def analyze_main_effects(agg: pd.DataFrame) -> pd.DataFrame:
    """Compute mean JSD for each level of each format dimension."""
    rows = []
    for dim in CONFIG_COLS:
        for level, grp in agg.groupby(dim):
            rows.append({
                "dimension": DIMENSION_NAMES.get(dim, dim),
                "level": str(level),
                "mean_jsd": grp["jsd"].mean(),
                "std_jsd": grp["jsd"].std(),
                "mean_pvalid": grp["p_valid"].mean(),
                "n": len(grp),
            })
    return pd.DataFrame(rows)


def plot_main_effects(effects: pd.DataFrame, figures_dir: Path):
    """Bar chart: mean JSD for each level of each format dimension."""
    dimensions = effects["dimension"].unique()
    n_dims = len(dimensions)

    fig, axes = plt.subplots(1, n_dims, figsize=(4 * n_dims, 5), squeeze=False)

    for idx, dim in enumerate(dimensions):
        ax = axes[0, idx]
        sub = effects[effects["dimension"] == dim].sort_values("mean_jsd")

        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sub)))
        bars = ax.bar(range(len(sub)), sub["mean_jsd"], color=colors,
                       edgecolor="black", linewidth=0.5)
        ax.errorbar(range(len(sub)), sub["mean_jsd"],
                     yerr=sub["std_jsd"] / np.sqrt(sub["n"]),
                     fmt="none", color="black", capsize=3)

        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(sub["level"], rotation=30, ha="right", fontsize=8)
        ax.set_title(dim, fontsize=11)
        ax.set_ylabel("Mean JSD to Human" if idx == 0 else "")
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Main Effect of Each Format Dimension on JSD", fontsize=13)
    plt.tight_layout()
    out = figures_dir / "optimization_main_effects.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Analysis 2: Best Config by Question Type ──────────────────────────────────

def find_best_per_qtype(agg: pd.DataFrame) -> pd.DataFrame:
    """Find the best format configuration for each response type."""
    results = []
    for rtype, grp in agg.groupby("response_type"):
        config_means = grp.groupby(CONFIG_COLS).agg(
            mean_jsd=("jsd", "mean"),
            mean_pvalid=("p_valid", "mean"),
        ).reset_index()
        best = config_means.loc[config_means["mean_jsd"].idxmin()]
        results.append({
            "response_type": rtype,
            **{c: best[c] for c in CONFIG_COLS},
            "mean_jsd": best["mean_jsd"],
            "mean_pvalid": best["mean_pvalid"],
        })
    return pd.DataFrame(results)


def plot_best_by_qtype(best_df: pd.DataFrame, agg: pd.DataFrame, figures_dir: Path):
    """Show best format per response type vs baseline (default config)."""
    # Baseline: numbered_dot, lang cue, no hint, separate, 2 perms
    baseline_mask = (
        (agg["cue_style"] == "lang") &
        (agg["opt_format"] == "numbered_dot") &
        (~agg["scale_hint"]) &
        (agg["embed_style"] == "separate") &
        (agg["n_perms"] == 2)
    )
    baseline = agg[baseline_mask].groupby("response_type").agg(
        baseline_jsd=("jsd", "mean"),
    ).reset_index()

    merged = best_df.merge(baseline, on="response_type", how="left")

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(merged))
    width = 0.35

    ax.bar(x - width / 2, merged["baseline_jsd"], width, label="Baseline (current)",
           color="#d62728", edgecolor="black", linewidth=0.5, alpha=0.8)
    ax.bar(x + width / 2, merged["mean_jsd"], width, label="Best optimized",
           color="#2ca02c", edgecolor="black", linewidth=0.5, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(merged["response_type"], rotation=30, ha="right")
    ax.set_ylabel("Mean JSD to Human")
    ax.set_title("Baseline vs Best Optimized Format by Response Type")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate best configs
    for i, row in merged.iterrows():
        config_str = (f"cue={row['cue_style']}\n"
                      f"opt={row['opt_format']}\n"
                      f"K={row['n_perms']}")
        ax.annotate(config_str, (i + width / 2, row["mean_jsd"]),
                    textcoords="offset points", xytext=(0, 5),
                    fontsize=6, ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="wheat", alpha=0.7))

    plt.tight_layout()
    out = figures_dir / "optimization_best_by_qtype.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Analysis 3: Full Grid Heatmap ────────────────────────────────────────────

def plot_heatmap(agg: pd.DataFrame, figures_dir: Path):
    """Heatmap of mean JSD across format configurations."""
    # Create a combined config label
    config_jsd = agg.groupby(CONFIG_COLS).agg(
        mean_jsd=("jsd", "mean"),
    ).reset_index()

    # Pivot: rows = (cue_style, n_perms), cols = (opt_format, embed_style, scale_hint)
    config_jsd["row_label"] = (config_jsd["cue_style"] + " / K=" +
                                config_jsd["n_perms"].astype(str))
    config_jsd["col_label"] = (config_jsd["opt_format"] + " / " +
                                config_jsd["embed_style"] +
                                config_jsd["scale_hint"].map({True: " +hint", False: ""}))

    pivot = config_jsd.pivot_table(
        index="row_label", columns="col_label", values="mean_jsd"
    )

    fig, ax = plt.subplots(figsize=(16, 6))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="RdYlGn_r",
        ax=ax, linewidths=0.5,
        cbar_kws={"label": "Mean JSD to Human"},
    )
    ax.set_title("Prompt Format Grid: Mean JSD to Human Data")
    ax.set_xlabel("Option Format / Embed Style / Scale Hint")
    ax.set_ylabel("Cue Style / Permutations")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    out = figures_dir / "optimization_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Analysis 4: P_valid Impact ────────────────────────────────────────────────

def print_pvalid_summary(agg: pd.DataFrame):
    """Print P_valid statistics across format variants."""
    print(f"\n--- P_valid Impact ---")
    print(f"Overall mean P_valid: {agg['p_valid'].mean():.4f}")
    print(f"Overall min P_valid:  {agg['p_valid'].min():.4f}")

    # Configs where P_valid drops below 0.5
    low_pvalid = agg.groupby(CONFIG_COLS).agg(
        mean_pvalid=("p_valid", "mean"),
    ).reset_index()
    low = low_pvalid[low_pvalid["mean_pvalid"] < 0.50]
    if not low.empty:
        print(f"\nConfigs with mean P_valid < 0.50 ({len(low)}):")
        for _, row in low.iterrows():
            print(f"  P_valid={row['mean_pvalid']:.4f} | "
                  f"cue={row['cue_style']} opt={row['opt_format']} "
                  f"hint={row['scale_hint']} embed={row['embed_style']} K={row['n_perms']}")
    else:
        print("\nNo configs have mean P_valid < 0.50")


# ── Analysis 5: ANOVA-style Variance Decomposition ───────────────────────────

def variance_decomposition(agg: pd.DataFrame):
    """Decompose JSD variance by format dimension (eta-squared)."""
    print(f"\n--- Variance Decomposition (eta-squared) ---")
    total_var = agg["jsd"].var()

    for dim in CONFIG_COLS:
        groups = agg.groupby(dim)["jsd"]
        grand_mean = agg["jsd"].mean()
        ss_between = sum(
            len(grp) * (grp.mean() - grand_mean) ** 2
            for _, grp in groups
        )
        ss_total = ((agg["jsd"] - grand_mean) ** 2).sum()
        eta_sq = ss_between / ss_total if ss_total > 0 else 0
        print(f"  {DIMENSION_NAMES.get(dim, dim):<20} eta^2 = {eta_sq:.4f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Analyze prompt optimization grid search results"
    )
    parser.add_argument("--results-dir", type=Path,
                        default=PROJECT_ROOT / "results" / "optimization")
    parser.add_argument("--figures-dir", type=Path,
                        default=PROJECT_ROOT / "figures")
    args = parser.parse_args()

    args.figures_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading grid search results...")
    df = load_grid_results(args.results_dir)

    # Drop rows with NaN JSD (no human data available)
    has_jsd = df.dropna(subset=["jsd_to_human"])
    print(f"Rows with JSD data: {len(has_jsd)}/{len(df)}")

    if has_jsd.empty:
        print("ERROR: No JSD data available. Run with --human flag in optimize_prompts.py")
        sys.exit(1)

    # Aggregate to config level
    agg = aggregate_to_config_level(has_jsd)
    print(f"Aggregated to {len(agg)} (config, question) combinations")

    # Analysis 1: Main effects
    print("\n--- Main Effects ---")
    effects = analyze_main_effects(agg)
    for _, row in effects.iterrows():
        print(f"  {row['dimension']:<20} {row['level']:<16} "
              f"JSD={row['mean_jsd']:.4f} +/- {row['std_jsd']:.4f}  "
              f"P_valid={row['mean_pvalid']:.4f}")

    # Analysis 2: Best per question type
    print("\n--- Best Config per Response Type ---")
    best_df = find_best_per_qtype(agg)
    for _, row in best_df.iterrows():
        print(f"  {row['response_type']:<14} JSD={row['mean_jsd']:.4f} | "
              f"cue={row['cue_style']} opt={row['opt_format']} "
              f"hint={row['scale_hint']} embed={row['embed_style']} K={row['n_perms']}")

    # Analysis 3: Best overall config
    print("\n--- Best Overall Config ---")
    overall = agg.groupby(CONFIG_COLS).agg(
        mean_jsd=("jsd", "mean"),
        mean_pvalid=("p_valid", "mean"),
    ).reset_index().sort_values("mean_jsd")
    best = overall.iloc[0]
    print(f"  JSD={best['mean_jsd']:.4f} P_valid={best['mean_pvalid']:.4f}")
    for c in CONFIG_COLS:
        print(f"    {c}: {best[c]}")

    # Analysis 4: P_valid impact
    print_pvalid_summary(agg)

    # Analysis 5: Variance decomposition
    variance_decomposition(agg)

    # Generate figures
    print(f"\nGenerating figures in {args.figures_dir}/...")
    plot_main_effects(effects, args.figures_dir)
    plot_best_by_qtype(best_df, agg, args.figures_dir)
    plot_heatmap(agg, args.figures_dir)

    # Save best config recommendation
    recommendation = {
        "default": {c: best[c] for c in CONFIG_COLS},
        "overrides_by_response_type": {},
    }
    for _, row in best_df.iterrows():
        rtype = row["response_type"]
        override = {}
        for c in CONFIG_COLS:
            if row[c] != best[c]:
                override[c] = row[c]
        if override:
            recommendation["overrides_by_response_type"][rtype] = override

    rec_path = args.figures_dir.parent / "data" / "prompt_config_recommended.json"
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(rec_path, "w") as f:
        json.dump(recommendation, f, indent=2, default=str)
    print(f"\nSaved recommended config to {rec_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
