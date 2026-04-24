"""Generate completion-quality visualizations for the typst report.

Three figures, all computed from the classification-level 'other' signal
on trimmed prompts (variant_idx >= TRIMMED_VARIANT_MIN):

  1. quality_by_model.png       — per-model overall noise rate, broken out by template
  2. quality_lang_template.png  — heatmap of prompt quality (lang × template)
  3. quality_model_lang_self_concept.png
                                 — heatmap of %other (model × lang) for the self_concept
                                   template, which carries the most noise

Usage:
    PYTHONPATH=. python scripts/quality_figures.py \
        --db data/culture.db \
        --classifier gemma3_27b_it \
        --outdir figures/trimmed/quality
"""
import argparse
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.constants import (
    ALL_LANGS, EU_LANGS, LANG_NAMES, MODEL_LABELS, TRIMMED_VARIANT_MIN,
)


TEMPLATE_ORDER = [
    "self_concept", "belief", "success", "childrearing",
    "values", "family", "decision", "moral",
]


def load_classification_df(conn: sqlite3.Connection, classifier: str) -> pd.DataFrame:
    """One row per classified trimmed completion."""
    return pd.read_sql_query(
        """
        SELECT p.template_id, p.lang, p.prompt_id,
               c.model_id, cl.content_category
        FROM prompts p
        JOIN completions c ON c.prompt_id = p.prompt_id
        JOIN classifications cl ON cl.completion_id = c.completion_id
                               AND cl.classifier_model = ?
        WHERE p.variant_idx >= ? AND c.filter_status = 'ok'
        """,
        conn,
        params=(classifier, TRIMMED_VARIANT_MIN),
    )


def pretty_model(m: str) -> str:
    """Short label for a model_id."""
    if m.startswith("hplt2c_"):
        lang = m.replace("hplt2c_", "")
        return f"HPLT-{lang}"
    return MODEL_LABELS.get(m, m)


# ─── Figure 1: per-model noise, broken out by template ──────────

def fig_quality_by_model(df: pd.DataFrame, outpath: Path):
    df = df.copy()
    df["is_other"] = (df["content_category"] == "other").astype(int)

    # Overall per-model
    overall = (df.groupby("model_id")["is_other"]
                 .agg(["mean", "count"])
                 .rename(columns={"mean": "frac_other", "count": "n"})
                 .reset_index())

    # Separate multilingual vs HPLT for readability
    multi_ids = ["gemma3_27b_pt", "gemma3_12b_pt", "eurollm22b"]
    multi = overall[overall["model_id"].isin(multi_ids)].copy()
    multi["order"] = multi["model_id"].map({m: i for i, m in enumerate(multi_ids)})
    multi = multi.sort_values("order")

    hplt = (overall[overall["model_id"].str.startswith("hplt2c_")]
            .copy().sort_values("frac_other", ascending=False))

    # Per (model, template) for stacked display
    per_tmpl = (df.groupby(["model_id", "template_id"])["is_other"]
                  .mean().unstack(fill_value=0.0))
    per_tmpl = per_tmpl.reindex(columns=TEMPLATE_ORDER)

    fig, (ax_multi, ax_hplt) = plt.subplots(
        1, 2, figsize=(14, 5.5),
        gridspec_kw={"width_ratios": [1.3, 3]},
    )

    # Left: multilingual models, grouped bars by template
    x_multi = np.arange(len(multi))
    width = 0.10
    palette = plt.cm.tab10(np.linspace(0, 1, len(TEMPLATE_ORDER)))
    for i, tmpl in enumerate(TEMPLATE_ORDER):
        vals = [per_tmpl.loc[m, tmpl] if m in per_tmpl.index else 0
                for m in multi["model_id"]]
        ax_multi.bar(x_multi + (i - len(TEMPLATE_ORDER) / 2) * width + width / 2,
                     vals, width, label=tmpl, color=palette[i])
    ax_multi.plot(x_multi, multi["frac_other"], "k_", markersize=22, mew=2,
                  label="overall")
    ax_multi.set_xticks(x_multi)
    ax_multi.set_xticklabels([pretty_model(m) for m in multi["model_id"]],
                             rotation=20, ha="right")
    ax_multi.set_ylabel("fraction classified as 'other'")
    ax_multi.set_title("Multilingual models (per-template)")
    ax_multi.set_ylim(0, max(0.45, multi["frac_other"].max() * 1.2))
    ax_multi.axhline(0.1, color="gray", ls=":", lw=0.7)
    ax_multi.legend(fontsize=7, ncol=3, loc="upper left")
    ax_multi.grid(axis="y", alpha=0.2)

    # Right: HPLT monolingual (one bar per language, overall only)
    x_hplt = np.arange(len(hplt))
    ax_hplt.bar(x_hplt, hplt["frac_other"], color="#4c72b0")
    ax_hplt.set_xticks(x_hplt)
    ax_hplt.set_xticklabels(
        [m.replace("hplt2c_", "") for m in hplt["model_id"]],
        rotation=0,
    )
    ax_hplt.set_ylabel("fraction 'other'")
    ax_hplt.set_title("HPLT monolingual models (overall)")
    ax_hplt.axhline(0.1, color="gray", ls=":", lw=0.7, label="10% reference")
    ax_hplt.legend(loc="upper right", fontsize=8)
    ax_hplt.grid(axis="y", alpha=0.2)

    fig.suptitle("Completion quality by model — fraction of completions classified 'other' "
                 f"(trimmed prompts, classifier={df['model_id'].iloc[0] and 'gemma3_27b_it'})",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath}")


# ─── Figure 2: lang × template quality heatmap ──────────────────

def fig_quality_lang_template(df: pd.DataFrame, outpath: Path):
    """quality_score = 1 - frac_other, aggregated across all models."""
    df = df.copy()
    df["is_other"] = (df["content_category"] == "other").astype(int)

    q = (df.groupby(["lang", "template_id"])["is_other"]
           .mean().unstack(fill_value=np.nan))
    q = q.reindex(columns=TEMPLATE_ORDER)
    # Languages: EU first (alphabetical) then expanded
    expanded = ["zho", "jpn", "ara", "hin", "tur"]
    lang_order = sorted(EU_LANGS) + [l for l in expanded if l in q.index]
    q = q.reindex(index=[l for l in lang_order if l in q.index])

    score = 1 - q  # quality score

    fig, ax = plt.subplots(figsize=(9, 10))
    im = ax.imshow(score.values, aspect="auto",
                   cmap="RdYlGn", vmin=0.4, vmax=1.0)
    ax.set_xticks(np.arange(len(score.columns)))
    ax.set_xticklabels(score.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(score.index)))
    ax.set_yticklabels([f"{l} ({LANG_NAMES.get(l, l)})" for l in score.index],
                       fontsize=9)

    # Cell annotations
    for i in range(score.shape[0]):
        for j in range(score.shape[1]):
            v = score.values[i, j]
            if np.isnan(v):
                continue
            color = "white" if v < 0.55 or v > 0.95 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("quality score (1 − fraction 'other')")

    ax.set_title("Prompt quality by language × template\n"
                 "(aggregated across all 25 models)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath}")


# ─── Figure 3: model × lang heatmap for self_concept ────────────

def fig_model_lang_self_concept(df: pd.DataFrame, outpath: Path):
    """Which (model, lang) combinations are noisy on self_concept specifically?"""
    df = df[df["template_id"] == "self_concept"].copy()
    df["is_other"] = (df["content_category"] == "other").astype(int)

    mat = (df.groupby(["model_id", "lang"])["is_other"]
             .mean().unstack(fill_value=np.nan))

    # Lang order
    expanded = ["zho", "jpn", "ara", "hin", "tur"]
    lang_order = [l for l in sorted(EU_LANGS) if l in mat.columns]
    lang_order += [l for l in expanded if l in mat.columns]
    mat = mat.reindex(columns=lang_order)

    # Model order: multilingual first, then HPLT alphabetical
    multi = ["gemma3_27b_pt", "gemma3_12b_pt", "eurollm22b"]
    hplt = sorted(m for m in mat.index if m.startswith("hplt2c_"))
    mat = mat.reindex(index=[m for m in multi + hplt if m in mat.index])

    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(mat.values, aspect="auto", cmap="Reds", vmin=0, vmax=0.9)
    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=0, fontsize=9)
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels([pretty_model(m) for m in mat.index], fontsize=9)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=7, color="gray")
                continue
            color = "white" if v > 0.5 else "black"
            ax.text(j, i, f"{int(round(v*100))}", ha="center", va="center",
                    fontsize=7, color=color)

    # Horizontal rule between multilingual and HPLT
    ax.axhline(2.5, color="black", lw=1)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("fraction 'other' (%)")

    ax.set_title("self_concept: noise rate by model × language\n"
                 "(values are % of completions classified as 'other')")
    fig.tight_layout()
    fig.savefig(outpath, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath}")


# ─── Figure 4: per-template quality score distribution ──────────

def fig_template_distribution(df: pd.DataFrame, outpath: Path):
    df = df.copy()
    df["is_other"] = (df["content_category"] == "other").astype(int)
    per_prompt = (df.groupby(["prompt_id", "template_id", "lang"])["is_other"]
                    .mean().reset_index())
    per_prompt["quality"] = 1 - per_prompt["is_other"]

    fig, ax = plt.subplots(figsize=(9, 5))
    data = [per_prompt[per_prompt["template_id"] == t]["quality"].values
            for t in TEMPLATE_ORDER]
    bp = ax.boxplot(data, labels=TEMPLATE_ORDER, vert=True, showmeans=True,
                    patch_artist=True, widths=0.6)
    for patch, color in zip(bp["boxes"], plt.cm.tab10(np.linspace(0, 1, len(TEMPLATE_ORDER)))):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    # Jittered points
    for i, d in enumerate(data):
        x = np.random.normal(i + 1, 0.05, size=len(d))
        ax.scatter(x, d, alpha=0.5, s=14, color="black", zorder=3)

    ax.set_ylabel("quality score (1 − fraction 'other')")
    ax.set_title("Per-prompt quality distribution by template\n"
                 "(one point per language; aggregated across all 25 models)")
    ax.axhline(0.5, color="red", ls="--", lw=0.8, label="unsalvageable threshold")
    ax.axhline(0.75, color="orange", ls="--", lw=0.8, label="degraded threshold")
    ax.axhline(0.9, color="green", ls="--", lw=0.8, label="clean threshold")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0.3, 1.02)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(outpath, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/culture.db")
    ap.add_argument("--classifier", default="gemma3_27b_it")
    ap.add_argument("--outdir", default="figures/trimmed/quality")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(args.db)
    print(f"Loading classified completions (trimmed, classifier={args.classifier})...")
    df = load_classification_df(conn, args.classifier)
    conn.close()
    print(f"  {len(df):,} completions across {df['model_id'].nunique()} models, "
          f"{df['lang'].nunique()} langs, {df['template_id'].nunique()} templates")

    print("\nGenerating figures...")
    fig_quality_by_model(df, outdir / "quality_by_model.png")
    fig_quality_lang_template(df, outdir / "quality_lang_template.png")
    fig_model_lang_self_concept(df, outdir / "quality_model_lang_self_concept.png")
    fig_template_distribution(df, outdir / "quality_template_distribution.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
