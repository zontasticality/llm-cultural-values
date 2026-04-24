"""Ablation diagnostics: decompose where cultural signal lives and identify biases.

Six analyses using existing classified data (no re-inference needed):
  B1. logprob_validation  — verify argmax(probs) == greedy integer
  B2. logprob_ev          — logprob expected values vs greedy, compared to EVS
  B3. per_prompt_rho      — which prompts carry cultural signal?
  B4. variance_decomposition — eta-squared for language/template/model factors
  B5. category_evs        — content category proportions vs EVS composites
  B6. template_bias       — does the classifier just map template names to categories?

Usage:
    PYTHONPATH=. python scripts/ablation_analysis.py {analysis|all} \\
        --llm-db data/culture.db \\
        --human-db ../eurollm/data/survey.db \\
        --classifier gemma3_27b_it
"""
import argparse
import json
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
    MODEL_LABELS, CONTENT_CATEGORIES,
    IW_TRADITIONAL_SECULAR, IW_SURVIVAL_SELFEXPR,
    TRIMMED_VARIANT_MIN, LOGPROB_DIMS,
)


FIG_DIR = Path("figures/trimmed/ablation")
MULTILINGUAL_MODELS = ["gemma3_27b_pt", "gemma3_12b_pt", "eurollm22b"]
PRIMARY_MODEL = "gemma3_27b_pt"


# ── Data loading ──────────────────────────────────────────────────

def load_classified_with_logprobs(llm_db: str, classifier: str, lang_match: bool = False) -> pd.DataFrame:
    """Load trimmed classified completions with logprob data.

    lang_match: drop rows where detected_lang exists and doesn't match prompts.lang.
    """
    db = sqlite3.connect(llm_db)
    db.row_factory = sqlite3.Row
    lang_filter = "AND (c.detected_lang IS NULL OR c.detected_lang = p.lang)" if lang_match else ""
    rows = db.execute(f"""
        SELECT c.completion_id, c.model_id,
               p.template_id, p.lang, p.variant_idx,
               cl.content_category,
               cl.dim_indiv_collect, cl.dim_trad_secular, cl.dim_surv_selfexpr,
               cl.dim_ic_probs, cl.dim_ts_probs, cl.dim_ss_probs, cl.cat_probs
        FROM classifications cl
        JOIN completions c ON cl.completion_id = c.completion_id
        JOIN prompts p ON c.prompt_id = p.prompt_id
        WHERE cl.classifier_model = ?
          AND p.variant_idx >= {TRIMMED_VARIANT_MIN}
          AND c.filter_status = 'ok'
          {lang_filter}
        ORDER BY c.completion_id
    """, (classifier,)).fetchall()
    db.close()

    df = pd.DataFrame([dict(r) for r in rows])
    if df.empty:
        return df

    # Parse JSON logprob columns
    for col in ["dim_ic_probs", "dim_ts_probs", "dim_ss_probs", "cat_probs"]:
        df[col] = df[col].apply(lambda x: json.loads(x) if x else None)

    # Add best-available measure columns (EV where logprobs exist, greedy otherwise)
    for dim_col, prob_col in LOGPROB_DIMS.items():
        measure_col = dim_col + "_ev"
        df[measure_col] = df[dim_col].astype(float)
        ev_mask = df[prob_col].notna()
        if ev_mask.any():
            df.loc[ev_mask, measure_col] = df.loc[ev_mask, prob_col].apply(
                lambda p: sum((i + 1) * p[i] for i in range(len(p)))
            )

    return df


def compute_human_iw(human_db_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute IW composite scores from EVS human distributions.

    Returns:
        (composites_df, per_question_df)
        composites_df: columns [lang, human_ts, human_se]
        per_question_df: columns [lang, qid, ev_z, dimension]
    """
    db = sqlite3.connect(human_db_path)
    all_iw_qs = {**IW_TRADITIONAL_SECULAR, **IW_SURVIVAL_SELFEXPR}

    ev_rows = []
    for qid, info in all_iw_qs.items():
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
                probs = probs / total
            ev = np.dot(vals, probs)

            if info.get("flip", False) and info.get("max_val") is not None:
                ev = info["max_val"] - ev
            dim = "TS" if qid in IW_TRADITIONAL_SECULAR else "SE"
            ev_rows.append({"lang": lang, "qid": qid, "ev": ev, "dimension": dim})

    db.close()
    ev_df = pd.DataFrame(ev_rows)

    for qid in ev_df["qid"].unique():
        mask = ev_df["qid"] == qid
        vals = ev_df.loc[mask, "ev"]
        ev_df.loc[mask, "ev_z"] = (vals - vals.mean()) / vals.std() if vals.std() > 0 else 0

    # Per-question z-scores (for B3b)
    per_q = ev_df[["lang", "qid", "ev_z", "dimension"]].copy()

    # Composites
    result = []
    for lang in EVS_COUNTRY_NAMES.keys():
        ldf = ev_df[ev_df["lang"] == lang]
        ts_qs = ldf[ldf["qid"].isin(IW_TRADITIONAL_SECULAR)]
        ts = ts_qs["ev_z"].mean() if len(ts_qs) >= 3 else np.nan
        se_qs = ldf[ldf["qid"].isin(IW_SURVIVAL_SELFEXPR)]
        se = se_qs["ev_z"].mean() if len(se_qs) >= 3 else np.nan
        result.append({"lang": lang, "human_ts": ts, "human_se": se})

    return pd.DataFrame(result).dropna(), per_q


# ── B1: Logprob validation ────────────────────────────────────────

def logprob_validation(df: pd.DataFrame, **_):
    """Verify argmax(dim_*_probs) == dim_* (greedy integer)."""
    print("\n" + "=" * 70)
    print("B1: LOGPROB VALIDATION")
    print("=" * 70)

    for dim_col, prob_col in LOGPROB_DIMS.items():
        valid = df[df[prob_col].notna()]
        if valid.empty:
            print(f"\n  {dim_col}: no logprob data available")
            continue

        greedy = valid[dim_col].values
        argmax = np.array([np.argmax(p) + 1 for p in valid[prob_col]])
        agree = int((greedy == argmax).sum())
        total = len(valid)
        pct = agree / total * 100

        print(f"\n  {dim_col}:")
        print(f"    Agreement: {agree}/{total} ({pct:.2f}%)")
        print(f"    Mismatch:  {total - agree} ({100 - pct:.2f}%)")

        if (100 - pct) > 1:
            print(f"    *** WARNING: >1% mismatch — logprobs may be unreliable ***")
            mismatch_mask = greedy != argmax
            for _, row in valid[mismatch_mask].head(5).iterrows():
                probs_str = [f"{p:.3f}" for p in row[prob_col]]
                print(f"      greedy={row[dim_col]}, argmax={np.argmax(row[prob_col])+1}, "
                      f"probs=[{', '.join(probs_str)}]")


# ── B2: Logprob EVs vs greedy ────────────────────────────────────

def logprob_ev(df: pd.DataFrame, human: pd.DataFrame, **_):
    """Compare greedy integer vs logprob expected value against EVS."""
    print("\n" + "=" * 70)
    print("B2: LOGPROB EXPECTED VALUES vs GREEDY — EVS COMPARISON")
    print("=" * 70)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    eu_langs = set(EVS_COUNTRY_NAMES.keys())
    human_indexed = human.set_index("lang")

    dim_pairs = [
        ("dim_trad_secular", "dim_trad_secular_ev", "human_ts", "Traditional ↔ Secular"),
        ("dim_surv_selfexpr", "dim_surv_selfexpr_ev", "human_se", "Survival ↔ Self-Expression"),
    ]

    header = f"  {'Model':<20s} {'Dimension':<25s} {'Greedy ρ':>10s} {'EV ρ':>10s} {'Greedy range':>14s} {'EV range':>14s}"
    print(f"\n{header}")
    print("  " + "-" * 95)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    for col_idx, model_id in enumerate(MULTILINGUAL_MODELS):
        mdf = df[df["model_id"] == model_id]
        if mdf.empty:
            continue

        for row_idx, (greedy_col, ev_col, human_col, label) in enumerate(dim_pairs):
            # Per-language means
            lang_greedy = mdf.groupby("lang")[greedy_col].mean()
            lang_ev = mdf.groupby("lang")[ev_col].mean()

            # Inner join with human on EU langs only
            common_greedy = human_indexed[[human_col]].join(
                lang_greedy.rename("llm"), how="inner"
            )
            common_greedy = common_greedy[common_greedy.index.isin(eu_langs)]

            common_ev = human_indexed[[human_col]].join(
                lang_ev.rename("llm"), how="inner"
            )
            common_ev = common_ev[common_ev.index.isin(eu_langs)]

            if len(common_greedy) < 5 or len(common_ev) < 5:
                continue

            rho_greedy, _ = spearmanr(common_greedy[human_col], common_greedy["llm"])
            rho_ev, _ = spearmanr(common_ev[human_col], common_ev["llm"])

            greedy_range = f"{lang_greedy.min():.3f}–{lang_greedy.max():.3f}"
            ev_range = f"{lang_ev.min():.3f}–{lang_ev.max():.3f}"

            print(f"  {MODEL_LABELS.get(model_id, model_id):<20s} {label:<25s} "
                  f"{rho_greedy:>10.3f} {rho_ev:>10.3f} {greedy_range:>14s} {ev_range:>14s}")

            # Scatter plot using logprob EVs
            ax = axes[row_idx, col_idx]
            for lang in common_ev.index:
                cluster = LANG_TO_CLUSTER.get(lang, "Other")
                color = CLUSTER_COLORS.get(cluster, "gray")
                ax.scatter(common_ev.loc[lang, human_col], common_ev.loc[lang, "llm"],
                           c=color, s=60, zorder=5)
                ax.annotate(lang, (common_ev.loc[lang, human_col], common_ev.loc[lang, "llm"]),
                            fontsize=7, ha="center", va="bottom", xytext=(0, 4),
                            textcoords="offset points")

            ax.set_xlabel("Human EVS (z-scored)")
            ax.set_ylabel("LLM logprob EV")
            ax.set_title(f"{MODEL_LABELS.get(model_id, model_id)}\n{label}\n"
                         f"ρ = {rho_ev:.3f}")

            if len(common_ev) > 2:
                z = np.polyfit(common_ev[human_col].values, common_ev["llm"].values, 1)
                x_line = np.linspace(common_ev[human_col].min(), common_ev[human_col].max(), 50)
                ax.plot(x_line, np.polyval(z, x_line), "k--", alpha=0.3)

    # Legend
    for cluster, color in CLUSTER_COLORS.items():
        axes[0, 0].scatter([], [], c=color, s=60, label=cluster)
    axes[0, 0].legend(loc="upper left", fontsize=7)

    plt.suptitle("LLM Logprob Expected Values vs Human EVS", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "iw_comparison_logprob_ev.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {FIG_DIR}/iw_comparison_logprob_ev.png")


# ── B3: Per-prompt explained variance ────────────────────────────

def per_prompt_rho(df: pd.DataFrame, human: pd.DataFrame, **_):
    """Per-(template, variant) Spearman rho against EVS composites."""
    print("\n" + "=" * 70)
    print("B3: PER-PROMPT EXPLAINED VARIANCE")
    print("=" * 70)

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    primary = df[df["model_id"] == PRIMARY_MODEL]
    if primary.empty:
        print(f"  No data for {PRIMARY_MODEL}")
        return

    eu_langs = set(EVS_COUNTRY_NAMES.keys())
    human_indexed = human.set_index("lang")

    dim_pairs = [
        ("dim_trad_secular_ev", "human_ts", "Trad-Secular"),
        ("dim_surv_selfexpr_ev", "human_se", "Surv-SelfExpr"),
    ]

    results = []
    for measure_col, human_col, label in dim_pairs:
        for (tmpl, vidx), grp in primary.groupby(["template_id", "variant_idx"]):
            lang_means = grp.groupby("lang")[measure_col].mean()

            common = human_indexed[[human_col]].join(
                lang_means.rename("llm"), how="inner"
            )
            common = common[common.index.isin(eu_langs)]

            if len(common) < 5:
                continue

            rho, p_val = spearmanr(common[human_col], common["llm"])
            results.append({
                "dimension": label,
                "template_id": tmpl,
                "variant_idx": int(vidx),
                "rho": rho,
                "p": p_val,
                "n_langs": len(common),
            })

    if not results:
        print("  No prompt-level results computed")
        return

    rdf = pd.DataFrame(results).sort_values("rho", key=abs, ascending=False)

    print(f"\n  {'Dimension':<15s} {'Template':<20s} {'Var':>4s} "
          f"{'ρ':>8s} {'p':>10s} {'Langs':>5s}")
    print("  " + "-" * 68)
    for _, row in rdf.iterrows():
        sig = "***" if row["p"] < 0.01 else "**" if row["p"] < 0.05 else "*" if row["p"] < 0.1 else ""
        print(f"  {row['dimension']:<15s} {row['template_id']:<20s} {row['variant_idx']:>4d} "
              f"{row['rho']:>8.3f} {row['p']:>10.3e} {row['n_langs']:>5d} {sig}")

    # Bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(rdf) * 0.15)))
    for ax, dim_label in zip(axes, ["Trad-Secular", "Surv-SelfExpr"]):
        ddf = rdf[rdf["dimension"] == dim_label].sort_values("rho")
        if ddf.empty:
            ax.set_visible(False)
            continue
        labels = [f"{r['template_id']}:{r['variant_idx']}" for _, r in ddf.iterrows()]
        colors = ["#2ca02c" if r["p"] < 0.05 else "#d62728" if r["p"] > 0.1 else "#ff7f0e"
                  for _, r in ddf.iterrows()]
        ax.barh(range(len(ddf)), ddf["rho"].values, color=colors)
        ax.set_yticks(range(len(ddf)))
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel("Spearman ρ vs EVS")
        ax.set_title(f"{dim_label}: Per-prompt ρ")
        ax.axvline(x=0, color="k", linewidth=0.5)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "per_prompt_rho.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {FIG_DIR}/per_prompt_rho.png")


# ── B3b: Per-prompt × per-question correlation matrix ─────────────

def per_prompt_question_rho(df: pd.DataFrame, human_per_q: pd.DataFrame, **_):
    """Prompt × WVS question Spearman rho matrix.

    Instead of correlating each prompt against the IW composite,
    correlate against each individual WVS question's per-language z-score.
    This reveals which prompts map to which specific survey questions.
    """
    print("\n" + "=" * 70)
    print("B3b: PER-PROMPT × PER-QUESTION CORRELATION MATRIX")
    print("=" * 70)

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    primary = df[df["model_id"] == PRIMARY_MODEL]
    if primary.empty:
        print(f"  No data for {PRIMARY_MODEL}")
        return

    eu_langs = set(EVS_COUNTRY_NAMES.keys())

    # Pivot human per-question data: rows=lang, columns=qid, values=ev_z
    q_pivot = human_per_q.pivot_table(index="lang", columns="qid", values="ev_z")
    q_pivot = q_pivot[q_pivot.index.isin(eu_langs)]

    all_iw_qs = {**IW_TRADITIONAL_SECULAR, **IW_SURVIVAL_SELFEXPR}

    # LLM dimension to correlate against each question's dimension
    dim_for_q = {}
    for qid in IW_TRADITIONAL_SECULAR:
        dim_for_q[qid] = "dim_trad_secular_ev"
    for qid in IW_SURVIVAL_SELFEXPR:
        dim_for_q[qid] = "dim_surv_selfexpr_ev"

    templates = sorted(primary["template_id"].unique())
    qids = list(IW_TRADITIONAL_SECULAR.keys()) + list(IW_SURVIVAL_SELFEXPR.keys())

    # Build correlation matrix: templates × questions
    rho_matrix = np.full((len(templates), len(qids)), np.nan)
    p_matrix = np.full((len(templates), len(qids)), np.nan)

    for i, tmpl in enumerate(templates):
        tdf = primary[primary["template_id"] == tmpl]
        for j, qid in enumerate(qids):
            if qid not in q_pivot.columns:
                continue
            measure_col = dim_for_q[qid]
            lang_means = tdf.groupby("lang")[measure_col].mean()

            # Align languages
            common_langs = lang_means.index.intersection(q_pivot.index)
            if len(common_langs) < 5:
                continue

            rho, p_val = spearmanr(
                q_pivot.loc[common_langs, qid].values,
                lang_means.loc[common_langs].values,
            )
            rho_matrix[i, j] = rho
            p_matrix[i, j] = p_val

    # Print table
    q_labels = [f"{qid} ({all_iw_qs[qid]['label'][:15]})" for qid in qids]
    q_short = [qid for qid in qids]

    print(f"\n  {'Template':<18s}", end="")
    for ql in q_short:
        print(f" {ql:>8s}", end="")
    print()
    print("  " + "-" * (18 + 9 * len(qids)))

    for i, tmpl in enumerate(templates):
        print(f"  {tmpl:<18s}", end="")
        for j in range(len(qids)):
            rho = rho_matrix[i, j]
            if np.isnan(rho):
                print(f" {'---':>8s}", end="")
            else:
                sig = "*" if p_matrix[i, j] < 0.05 else ""
                print(f" {rho:>7.3f}{sig}", end="")
        print()

    # TS/SE separator label
    ts_desc = ", ".join(f"{q} ({all_iw_qs[q]['label']})" for q in IW_TRADITIONAL_SECULAR)
    se_desc = ", ".join(f"{q} ({all_iw_qs[q]['label']})" for q in IW_SURVIVAL_SELFEXPR)
    print(f"\n  TS questions: {ts_desc}")
    print(f"  SE questions: {se_desc}")

    # Highlight strongest prompt-question pairs
    print(f"\n  Strongest prompt–question associations (|ρ| > 0.3):")
    pairs = []
    for i, tmpl in enumerate(templates):
        for j, qid in enumerate(qids):
            rho = rho_matrix[i, j]
            if not np.isnan(rho) and abs(rho) > 0.3:
                pairs.append((abs(rho), tmpl, qid, rho, p_matrix[i, j]))
    pairs.sort(reverse=True)
    if pairs:
        for _, tmpl, qid, rho, p_val in pairs:
            sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
            print(f"    {tmpl:<18s} × {qid:<5s} ({all_iw_qs[qid]['label']:<25s}): "
                  f"ρ = {rho:>7.3f}  p = {p_val:.3e} {sig}")
    else:
        print("    (none)")

    # Heatmap
    fig, ax = plt.subplots(figsize=(14, max(4, len(templates) * 0.6)))
    im = ax.imshow(rho_matrix, cmap="RdBu_r", vmin=-0.7, vmax=0.7, aspect="auto")

    ax.set_xticks(range(len(qids)))
    ax.set_xticklabels(q_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(templates)))
    ax.set_yticklabels(templates, fontsize=9)

    # Annotate cells
    for i in range(len(templates)):
        for j in range(len(qids)):
            val = rho_matrix[i, j]
            if np.isnan(val):
                continue
            sig = "*" if p_matrix[i, j] < 0.05 else ""
            ax.text(j, i, f"{val:.2f}{sig}", ha="center", va="center", fontsize=7,
                    color="white" if abs(val) > 0.4 else "black")

    # Draw divider between TS and SE questions
    n_ts = len(IW_TRADITIONAL_SECULAR)
    ax.axvline(x=n_ts - 0.5, color="black", linewidth=2)
    ax.text(n_ts / 2 - 0.5, -0.8, "Traditional–Secular", ha="center", fontsize=8, fontweight="bold")
    ax.text(n_ts + len(IW_SURVIVAL_SELFEXPR) / 2 - 0.5, -0.8, "Survival–SelfExpr",
            ha="center", fontsize=8, fontweight="bold")

    plt.colorbar(im, label="Spearman ρ")
    ax.set_title(f"Prompt Template × WVS Question Correlation ({PRIMARY_MODEL})")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "prompt_question_rho_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {FIG_DIR}/prompt_question_rho_matrix.png")


# ── B4: Variance decomposition ───────────────────────────────────

def _eta_squared(data: pd.Series, groups: pd.Series) -> float:
    """Compute eta-squared = SS_between / SS_total."""
    grand_mean = data.mean()
    ss_total = ((data - grand_mean) ** 2).sum()
    if ss_total == 0:
        return 0.0
    ss_between = 0.0
    for _, grp in data.groupby(groups):
        ss_between += len(grp) * (grp.mean() - grand_mean) ** 2
    return ss_between / ss_total


def variance_decomposition(df: pd.DataFrame, **_):
    """Eta-squared for language, template, model factors."""
    print("\n" + "=" * 70)
    print("B4: VARIANCE DECOMPOSITION (multilingual models only)")
    print("=" * 70)

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    multi = df[df["model_id"].isin(MULTILINGUAL_MODELS)].copy()
    if multi.empty:
        print("  No multilingual model data")
        return

    response_vars = [
        ("dim_trad_secular_ev", "Trad-Secular"),
        ("dim_surv_selfexpr_ev", "Surv-SelfExpr"),
    ]

    factors = ["lang", "template_id", "model_id"]

    print(f"\n  {'Response Variable':<25s}", end="")
    for f in factors:
        print(f" {f:>14s}", end="")
    print(f" {'lang×template':>14s}")
    print("  " + "-" * 85)

    eta_results = {}
    for resp_col, resp_label in response_vars:
        valid = multi[multi[resp_col].notna()]
        if valid.empty:
            continue

        etas = {}
        for factor in factors:
            etas[factor] = _eta_squared(valid[resp_col], valid[factor])

        # Interaction: lang x template
        interaction = valid["lang"].astype(str) + ":" + valid["template_id"].astype(str)
        etas["lang×template"] = _eta_squared(valid[resp_col], interaction)

        print(f"  {resp_label:<25s}", end="")
        for f in factors:
            print(f" {etas[f]:>13.4f}", end="")
        print(f" {etas['lang×template']:>13.4f}")

        eta_results[resp_label] = etas

    if not eta_results:
        return

    # Stacked bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = list(eta_results.keys())
    all_factors = factors + ["lang×template"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    bottom = np.zeros(len(labels))

    for i, factor in enumerate(all_factors):
        vals = [eta_results[l].get(factor, 0) for l in labels]
        ax.bar(labels, vals, bottom=bottom, label=factor, color=colors[i])
        bottom += vals

    ax.set_ylabel("η² (proportion of variance)")
    ax.set_title("Variance Decomposition")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIG_DIR / "variance_decomposition.png", dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {FIG_DIR}/variance_decomposition.png")


# ── B5: Content categories vs EVS ────────────────────────────────

def category_evs(df: pd.DataFrame, human: pd.DataFrame, **_):
    """Correlate per-language category proportions with EVS composites."""
    print("\n" + "=" * 70)
    print("B5: CONTENT CATEGORIES vs EVS")
    print("=" * 70)

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    primary = df[df["model_id"] == PRIMARY_MODEL]
    if primary.empty:
        print(f"  No data for {PRIMARY_MODEL}")
        return

    eu_langs = set(EVS_COUNTRY_NAMES.keys())
    primary_eu = primary[primary["lang"].isin(eu_langs)]

    # Per-language category proportions
    ct = pd.crosstab(primary_eu["lang"], primary_eu["content_category"], normalize="index")

    human_indexed = human.set_index("lang")
    merged = ct.join(human_indexed, how="inner")

    if len(merged) < 5:
        print(f"  Only {len(merged)} languages matched — too few")
        return

    print(f"\n  {'Category':<25s} {'ρ vs TS':>10s} {'p(TS)':>10s}     "
          f"{'ρ vs SE':>10s} {'p(SE)':>10s}")
    print("  " + "-" * 75)

    cat_results = []
    for cat in CONTENT_CATEGORIES:
        if cat not in merged.columns:
            continue
        rho_ts, p_ts = spearmanr(merged[cat], merged["human_ts"])
        rho_se, p_se = spearmanr(merged[cat], merged["human_se"])
        sig_ts = "***" if p_ts < 0.01 else "**" if p_ts < 0.05 else "*" if p_ts < 0.1 else ""
        sig_se = "***" if p_se < 0.01 else "**" if p_se < 0.05 else "*" if p_se < 0.1 else ""
        print(f"  {cat:<25s} {rho_ts:>10.3f} {p_ts:>10.3e} {sig_ts:>3s} "
              f"{rho_se:>10.3f} {p_se:>10.3e} {sig_se:>3s}")
        cat_results.append({
            "cat": cat, "rho_ts": rho_ts, "rho_se": rho_se,
            "p_ts": p_ts, "p_se": p_se,
        })

    if not cat_results:
        return

    # Scatter plots for top 4 correlated categories
    cdf = pd.DataFrame(cat_results)
    cdf["max_abs_rho"] = cdf[["rho_ts", "rho_se"]].abs().max(axis=1)
    top = cdf.nlargest(4, "max_abs_rho")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (_, row) in zip(axes.flat, top.iterrows()):
        cat = row["cat"]
        if abs(row["rho_ts"]) >= abs(row["rho_se"]):
            human_col, dim_label, rho = "human_ts", "Trad-Secular", row["rho_ts"]
        else:
            human_col, dim_label, rho = "human_se", "Surv-SelfExpr", row["rho_se"]

        for lang in merged.index:
            cluster = LANG_TO_CLUSTER.get(lang, "Other")
            color = CLUSTER_COLORS.get(cluster, "gray")
            ax.scatter(merged.loc[lang, human_col], merged.loc[lang, cat],
                       c=color, s=60, zorder=5)
            ax.annotate(lang, (merged.loc[lang, human_col], merged.loc[lang, cat]),
                        fontsize=7, ha="center", va="bottom", xytext=(0, 4),
                        textcoords="offset points")

        ax.set_xlabel(f"Human EVS {dim_label}")
        ax.set_ylabel(f"{cat} proportion")
        ax.set_title(f"{cat}\nρ = {rho:.3f} vs {dim_label}")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "category_evs_correlation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {FIG_DIR}/category_evs_correlation.png")


# ── B6: Template label bias ──────────────────────────────────────

def template_bias(df: pd.DataFrame, **_):
    """Test whether classifier maps template names to categories."""
    print("\n" + "=" * 70)
    print("B6: TEMPLATE LABEL BIAS DIAGNOSTIC")
    print("=" * 70)

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    primary = df[df["model_id"] == PRIMARY_MODEL]
    if primary.empty:
        print(f"  No data for {PRIMARY_MODEL}")
        return

    templates = sorted(primary["template_id"].unique())
    cats = [c for c in CONTENT_CATEGORIES if c in primary["content_category"].unique()]

    print(f"\n  Template dominance = var(template means) / var(language means)")
    print(f"  If >> 1 for a category, the template label may be biasing the classifier\n")
    print(f"  {'Category':<25s} {'Between-tmpl var':>16s} {'Between-lang var':>16s} {'Dominance':>10s}")
    print("  " + "-" * 70)

    dominance_matrix = np.zeros((len(templates), len(cats)))

    for j, cat in enumerate(cats):
        tmpl_props = []
        for i, tmpl in enumerate(templates):
            tdf = primary[primary["template_id"] == tmpl]
            prop = (tdf["content_category"] == cat).mean() if not tdf.empty else 0.0
            tmpl_props.append(prop)
            dominance_matrix[i, j] = prop

        lang_props = []
        for lang in sorted(primary["lang"].unique()):
            ldf = primary[primary["lang"] == lang]
            lang_props.append((ldf["content_category"] == cat).mean() if not ldf.empty else 0.0)

        var_tmpl = np.var(tmpl_props) if len(tmpl_props) > 1 else 0
        var_lang = np.var(lang_props) if len(lang_props) > 1 else 0
        if var_lang > 0:
            dominance = var_tmpl / var_lang
        elif var_tmpl > 0:
            dominance = float("inf")
        else:
            dominance = 0.0

        flag = " *** BIAS?" if dominance > 3 else ""
        print(f"  {cat:<25s} {var_tmpl:>16.6f} {var_lang:>16.6f} {dominance:>10.2f}{flag}")

    # Heatmap: templates x categories
    fig, ax = plt.subplots(figsize=(12, max(4, len(templates) * 0.6)))
    im = ax.imshow(dominance_matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(templates)))
    ax.set_yticklabels(templates, fontsize=8)

    for i in range(len(templates)):
        for j in range(len(cats)):
            val = dominance_matrix[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if val > 0.3 else "black")

    plt.colorbar(im, label="Category proportion")
    ax.set_title(f"Category Distribution by Template ({PRIMARY_MODEL})")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "template_category_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {FIG_DIR}/template_category_heatmap.png")


# ── Main ──────────────────────────────────────────────────────────

ANALYSES = {
    "logprob_validation": logprob_validation,
    "logprob_ev": logprob_ev,
    "per_prompt_rho": per_prompt_rho,
    "per_prompt_question_rho": per_prompt_question_rho,
    "variance_decomposition": variance_decomposition,
    "category_evs": category_evs,
    "template_bias": template_bias,
}


def main():
    parser = argparse.ArgumentParser(
        description="Ablation diagnostics for cultural completion analysis",
    )
    parser.add_argument("analysis", choices=[*ANALYSES.keys(), "all"])
    parser.add_argument("--llm-db", required=True)
    parser.add_argument("--human-db", required=True)
    parser.add_argument("--classifier", default="gemma3_27b_it")
    parser.add_argument("--lang-match", action="store_true",
                        help="Drop completions whose detected_lang doesn't match prompts.lang "
                             "(requires scripts/detect_lang.py to have been run)")
    args = parser.parse_args()

    print(f"Loading classified completions with logprobs... (lang_match={args.lang_match})")
    df = load_classified_with_logprobs(args.llm_db, args.classifier, lang_match=args.lang_match)
    print(f"  {len(df)} rows loaded")

    if df.empty:
        print("  ERROR: No data found. Check --llm-db and --classifier.")
        return

    print(f"  Models: {sorted(df['model_id'].unique())}")
    print(f"  Languages: {sorted(df['lang'].unique())}")

    for prob_col in ["dim_ic_probs", "dim_ts_probs", "dim_ss_probs"]:
        n_with = df[prob_col].notna().sum()
        print(f"  {prob_col}: {n_with}/{len(df)} ({n_with/len(df)*100:.1f}%) have logprobs")

    print("\nLoading human EVS composites...")
    human, human_per_q = compute_human_iw(args.human_db)
    print(f"  {len(human)} languages with human IW scores")
    print(f"  {human_per_q['qid'].nunique()} individual WVS questions")

    kwargs = {"df": df, "human": human, "human_per_q": human_per_q}

    if args.analysis == "all":
        for name, fn in ANALYSES.items():
            fn(**kwargs)
    else:
        ANALYSES[args.analysis](**kwargs)

    print("\nDone.")


if __name__ == "__main__":
    main()
