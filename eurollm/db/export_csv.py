"""Export debiased evaluation results from survey.db to CSV.

Aggregates per-permutation logprob data into mean/variance per
(question, language, model) and pivots response values into wide columns.

Usage:
    # Export all configs:
    PYTHONPATH=eurollm eurollm/.venv/bin/python -m db.export_csv

    # Filter to specific config(s) or models:
    PYTHONPATH=eurollm eurollm/.venv/bin/python -m db.export_csv \
        --config optimized_v1 cue_hint \
        --models hplt2c_eng eurollm22b
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from db.schema import get_connection
from analysis.constants import LANG_NAMES


def _build_human_rows(db_path: str) -> list[dict]:
    """Build CSV rows from human_distributions, matching the LLM row schema.

    Variance uses the multinomial sampling formula: Var(p̂) = p(1-p)/n.
    """
    conn = get_connection(db_path)
    human = pd.read_sql_query(
        """SELECT h.*, q.response_type
           FROM human_distributions h
           JOIN questions q ON h.question_id = q.question_id""",
        conn,
    )
    conn.close()

    if human.empty:
        return []

    rows = []
    for (lang, qid), grp in human.groupby(["lang", "question_id"]):
        lang_name = LANG_NAMES.get(lang, lang)
        n_valid = grp["n_valid"].iloc[0]
        n_respondents = grp["n_respondents"].iloc[0]

        response_type = grp["response_type"].iloc[0]

        row = {
            "question_id": qid,
            "response_type": response_type,
            "lang": lang_name,
            "lang_code": lang,
            "config": "human_evs",
            "model_id": "human_evs",
            "model_family": "human",
            "n_permutations": n_valid,  # sample size (meaningful analog)
            "p_valid_mean": 1.0,
            "p_valid_var": 0.0,
            "prompt_text_perm0": f"EVS survey, n_respondents={n_respondents}, n_valid={n_valid}",
        }

        for i in range(1, 11):
            match = grp[grp.response_value == i]
            if len(match) > 0:
                p = float(match["prob_human"].iloc[0])
                row[f"opt{i}_val"] = i
                row[f"opt{i}_prob_mean"] = p
                row[f"opt{i}_prob_var"] = p * (1 - p) / n_valid if n_valid > 0 else 0.0
            else:
                row[f"opt{i}_val"] = None
                row[f"opt{i}_prob_mean"] = None
                row[f"opt{i}_prob_var"] = None

        rows.append(row)

    return rows


def export_debiased_csv(
    db_path: str,
    output_path: str,
    configs: list[str] | None = None,
    model_ids: list[str] | None = None,
):
    """Export aggregated evaluation results to CSV.

    Args:
        db_path: Path to survey.db.
        output_path: Output CSV path.
        configs: Prompt configs to export (default: all with evaluations).
        model_ids: Filter to these models (default: all with evaluations).
    """
    conn = get_connection(db_path)

    query = """
        SELECT
            p.question_id, q.response_type, p.lang, p.config,
            e.model_id, m.model_family,
            p.permutation_idx, e.response_value, e.prob, e.p_valid,
            p.prompt_text
        FROM evaluations e
        JOIN prompts p ON e.prompt_id = p.prompt_id
        JOIN models m ON e.model_id = m.model_id
        JOIN questions q ON p.question_id = q.question_id
        WHERE p.variant_id IS NULL
    """
    params: list = []

    if configs:
        placeholders = ",".join("?" * len(configs))
        query += f" AND p.config IN ({placeholders})"
        params.extend(configs)

    if model_ids:
        placeholders = ",".join("?" * len(model_ids))
        query += f" AND e.model_id IN ({placeholders})"
        params.extend(model_ids)

    raw = pd.read_sql_query(query, conn, params=params)
    conn.close()

    if raw.empty:
        print("No evaluations found for the given filters")
        return

    print(f"Raw rows: {len(raw):,}  |  "
          f"Configs: {sorted(raw.config.unique())}  |  "
          f"Models: {raw.model_id.nunique()}  |  "
          f"Langs: {raw.lang.nunique()}")

    # Aggregate across permutations per (question, lang, model, config)
    rows = []
    group_cols = ["question_id", "response_type", "lang", "config", "model_id", "model_family"]

    for group_key, grp in raw.groupby(group_cols):
        qid, response_type, lang, config, model_id, model_family = group_key
        n_perms = grp["permutation_idx"].nunique()

        # Prompt text from identity permutation
        perm0 = grp[grp.permutation_idx == 0]
        prompt_text = perm0["prompt_text"].iloc[0] if len(perm0) > 0 else ""

        # p_valid: mean and variance across permutations
        pv_per_perm = grp.drop_duplicates(["permutation_idx"])[["permutation_idx", "p_valid"]]
        p_valid_mean = pv_per_perm["p_valid"].mean()
        p_valid_var = pv_per_perm["p_valid"].var() if len(pv_per_perm) > 1 else 0.0

        # Per response value: mean and variance of prob across permutations
        val_stats = {}
        for rv, rv_grp in grp.groupby("response_value"):
            probs = rv_grp.groupby("permutation_idx")["prob"].first()
            val_stats[rv] = {
                "prob_mean": probs.mean(),
                "prob_var": probs.var() if len(probs) > 1 else 0.0,
            }

        lang_name = LANG_NAMES.get(lang, lang)

        row = {
            "question_id": qid,
            "response_type": response_type,
            "lang": lang_name,
            "lang_code": lang,
            "config": config,
            "model_id": model_id,
            "model_family": model_family,
            "n_permutations": n_perms,
            "p_valid_mean": p_valid_mean,
            "p_valid_var": p_valid_var,
            "prompt_text_perm0": prompt_text,
        }

        # Pivot response values into opt1..opt10 wide columns
        for i in range(1, 11):
            if i in val_stats:
                row[f"opt{i}_val"] = i
                row[f"opt{i}_prob_mean"] = val_stats[i]["prob_mean"]
                row[f"opt{i}_prob_var"] = val_stats[i]["prob_var"]
            else:
                row[f"opt{i}_val"] = None
                row[f"opt{i}_prob_mean"] = None
                row[f"opt{i}_prob_var"] = None

        rows.append(row)

    df = pd.DataFrame(rows)

    # Append human survey data as a pseudo-model
    human_rows = _build_human_rows(db_path)
    if human_rows:
        df = pd.concat([df, pd.DataFrame(human_rows)], ignore_index=True)
        print(f"  + {len(human_rows):,} human survey rows")

    df = df.sort_values(["lang", "response_type", "question_id", "config", "model_id"]).reset_index(drop=True)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Exported {len(df):,} rows to {out}")


def main():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Export debiased evaluation results from survey.db to CSV"
    )
    parser.add_argument(
        "--db", type=Path,
        default=PROJECT_ROOT / "data" / "survey.db",
        help="Path to survey.db",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "data" / "exports" / "debiased_prompts.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--config", nargs="+", default=None,
        help="Prompt config(s) to export (default: all with evaluations)",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Filter to specific model IDs",
    )
    args = parser.parse_args()

    export_debiased_csv(
        db_path=str(args.db),
        output_path=str(args.output),
        configs=args.config,
        model_ids=args.models,
    )


if __name__ == "__main__":
    main()
