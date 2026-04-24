"""Compare LLM cultural dimension scores to human EVS/WVS data.

Computes Inglehart-Welzel composite scores from EVS human survey data
and compares them to LLM classifier dimension scores, producing a
scatter plot and rank correlations.

Usage:
    PYTHONPATH=. python scripts/compare_to_wvs.py \
        --llm-db data/culture.db \
        --human-db ../eurollm/data/survey.db \
        --classifier gemma3_27b_it \
        --trimmed-only \
        --output figures/trimmed/iw_comparison.png
"""
import argparse
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis.constants import (
    EVS_COUNTRY_NAMES, LANG_TO_CLUSTER, CLUSTER_COLORS,
    IW_TRADITIONAL_SECULAR, IW_SURVIVAL_SELFEXPR, TRIMMED_VARIANT_MIN,
    MODEL_LABELS,
)


def compute_human_iw(human_db_path: str) -> pd.DataFrame:
    """Compute IW composite scores from EVS human distributions."""
    db = sqlite3.connect(human_db_path)
    all_iw_qs = {**IW_TRADITIONAL_SECULAR, **IW_SURVIVAL_SELFEXPR}

    ev_rows = []
    for (lang, qid), info in [
        ((lang, qid), info)
        for qid, info in all_iw_qs.items()
        for lang in EVS_COUNTRY_NAMES.keys()
    ]:
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
            probs = probs / total
        ev = np.dot(vals, probs)

        # Apply flip
        if info.get("flip", False):
            max_val = info.get("max_val")
            if max_val is not None:
                ev = max_val - ev
        ev_rows.append({"lang": lang, "qid": qid, "ev": ev})

    db.close()
    ev_df = pd.DataFrame(ev_rows)

    # Per-question z-score normalization
    for qid in ev_df["qid"].unique():
        mask = ev_df["qid"] == qid
        vals = ev_df.loc[mask, "ev"]
        ev_df.loc[mask, "ev_z"] = (vals - vals.mean()) / vals.std() if vals.std() > 0 else 0

    # Average z-scores per dimension
    result = []
    for lang in EVS_COUNTRY_NAMES.keys():
        ldf = ev_df[ev_df["lang"] == lang]

        ts_qs = ldf[ldf["qid"].isin(IW_TRADITIONAL_SECULAR)]
        ts = ts_qs["ev_z"].mean() if len(ts_qs) >= 3 else np.nan

        se_qs = ldf[ldf["qid"].isin(IW_SURVIVAL_SELFEXPR)]
        se = se_qs["ev_z"].mean() if len(se_qs) >= 3 else np.nan

        result.append({"lang": lang, "human_ts": ts, "human_se": se})

    return pd.DataFrame(result).dropna()


def compute_llm_iw(llm_db_path: str, classifier: str, trimmed_only: bool,
                    model_id: str = "gemma3_27b_pt", lang_match: bool = False) -> pd.DataFrame:
    """Compute mean LLM dimension scores per language."""
    db = sqlite3.connect(llm_db_path)

    where = "cl.classifier_model = ? AND c.model_id = ?"
    params = [classifier, model_id]
    if trimmed_only:
        where += f" AND p.variant_idx >= {TRIMMED_VARIANT_MIN}"
    if lang_match:
        where += " AND (c.detected_lang IS NULL OR c.detected_lang = p.lang)"

    rows = db.execute(f"""
        SELECT p.lang,
               AVG(cl.dim_trad_secular) AS llm_ts,
               AVG(cl.dim_surv_selfexpr) AS llm_se,
               COUNT(*) AS n
        FROM classifications cl
        JOIN completions c ON cl.completion_id = c.completion_id
        JOIN prompts p ON c.prompt_id = p.prompt_id
        WHERE {where}
        GROUP BY p.lang
    """, params).fetchall()
    db.close()

    return pd.DataFrame(rows, columns=["lang", "llm_ts", "llm_se", "n"])


def main():
    parser = argparse.ArgumentParser(description="Compare LLM vs human IW scores")
    parser.add_argument("--llm-db", required=True)
    parser.add_argument("--human-db", required=True)
    parser.add_argument("--classifier", default="gemma3_27b_it")
    parser.add_argument("--trimmed-only", action="store_true")
    parser.add_argument("--lang-match", action="store_true",
                        help="Drop completions whose detected_lang != prompts.lang")
    parser.add_argument("--output", default="figures/trimmed/iw_comparison.png")
    args = parser.parse_args()

    print("Computing human IW composites from EVS data...")
    human = compute_human_iw(args.human_db)
    print(f"  {len(human)} languages with human IW scores")

    models = ["gemma3_27b_pt", "gemma3_12b_pt", "eurollm22b"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    for col, model_id in enumerate(models):
        print(f"\nComputing LLM IW scores for {model_id}...")
        llm = compute_llm_iw(args.llm_db, args.classifier, args.trimmed_only, model_id,
                             lang_match=args.lang_match)

        # Merge on lang (only EU languages with human data)
        merged = human.merge(llm, on="lang", how="inner")
        print(f"  {len(merged)} languages matched")

        if merged.empty:
            continue

        # Traditional-Secular comparison
        ax_ts = axes[0, col]
        for _, row in merged.iterrows():
            cluster = LANG_TO_CLUSTER.get(row["lang"], "Other")
            color = CLUSTER_COLORS.get(cluster, "gray")
            ax_ts.scatter(row["human_ts"], row["llm_ts"], c=color, s=60, zorder=5)
            ax_ts.annotate(row["lang"], (row["human_ts"], row["llm_ts"]),
                          fontsize=7, ha="center", va="bottom", xytext=(0, 4),
                          textcoords="offset points")

        rho_ts, p_ts = spearmanr(merged["human_ts"], merged["llm_ts"])
        ax_ts.set_xlabel("Human EVS (z-scored)")
        ax_ts.set_ylabel("LLM classifier (mean 1-5)")
        ax_ts.set_title(f"{MODEL_LABELS.get(model_id, model_id)}\nTraditional ↔ Secular\n"
                       f"ρ = {rho_ts:.3f} (p = {p_ts:.3e})")

        # Fit line
        if len(merged) > 2:
            z = np.polyfit(merged["human_ts"], merged["llm_ts"], 1)
            x_line = np.linspace(merged["human_ts"].min(), merged["human_ts"].max(), 50)
            ax_ts.plot(x_line, np.polyval(z, x_line), "k--", alpha=0.3)

        # Survival-Self-Expression comparison
        ax_se = axes[1, col]
        for _, row in merged.iterrows():
            cluster = LANG_TO_CLUSTER.get(row["lang"], "Other")
            color = CLUSTER_COLORS.get(cluster, "gray")
            ax_se.scatter(row["human_se"], row["llm_se"], c=color, s=60, zorder=5)
            ax_se.annotate(row["lang"], (row["human_se"], row["llm_se"]),
                          fontsize=7, ha="center", va="bottom", xytext=(0, 4),
                          textcoords="offset points")

        rho_se, p_se = spearmanr(merged["human_se"], merged["llm_se"])
        ax_se.set_xlabel("Human EVS (z-scored)")
        ax_se.set_ylabel("LLM classifier (mean 1-5)")
        ax_se.set_title(f"Survival ↔ Self-Expression\n"
                       f"ρ = {rho_se:.3f} (p = {p_se:.3e})")

        if len(merged) > 2:
            z = np.polyfit(merged["human_se"], merged["llm_se"], 1)
            x_line = np.linspace(merged["human_se"].min(), merged["human_se"].max(), 50)
            ax_se.plot(x_line, np.polyval(z, x_line), "k--", alpha=0.3)

        print(f"  TS: ρ={rho_ts:.3f} p={p_ts:.3e}")
        print(f"  SE: ρ={rho_se:.3f} p={p_se:.3e}")

    # Legend
    for cluster, color in CLUSTER_COLORS.items():
        axes[0, 0].scatter([], [], c=color, s=60, label=cluster)
    axes[0, 0].legend(loc="upper left", fontsize=7)

    plt.suptitle("LLM Cultural Dimensions vs Human EVS Data (21 European Languages)",
                fontsize=14, y=1.02)
    plt.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
