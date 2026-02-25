"""Query the survey database and return pandas DataFrames.

Provides backward-compatible column names for the analysis pipeline.
Aggregation (averaging across permutations, computing position bias)
happens here at load time rather than being stored in the DB.
"""

import json
import sqlite3

import numpy as np
import pandas as pd

from db.schema import get_connection


def load_results(
    db_path: str,
    model_ids: list[str] | None = None,
    langs: list[str] | None = None,
    config: str | None = None,
) -> pd.DataFrame:
    """Load evaluation results aggregated across permutations.

    Returns DataFrame with columns matching the existing analysis pipeline:
        model_type, lang, question_id, response_type, response_value,
        prob_forward, prob_reversed, prob_averaged,
        p_valid_forward, p_valid_reversed,
        position_bias_magnitude, n_permutations
    """
    conn = get_connection(db_path)

    query = """
        SELECT
            m.model_family AS model_type,
            p.lang,
            p.question_id,
            q.response_type,
            e.response_value,
            p.permutation_idx,
            e.prob,
            e.p_valid,
            p.prompt_text,
            p.question_text,
            p.value_map
        FROM evaluations e
        JOIN prompts p ON e.prompt_id = p.prompt_id
        JOIN models m ON e.model_id = m.model_id
        JOIN questions q ON p.question_id = q.question_id
        WHERE p.variant_id IS NULL
    """
    params = []

    if config is not None:
        query += " AND p.config = ?"
        params.append(config)

    if model_ids:
        placeholders = ",".join("?" * len(model_ids))
        query += f" AND e.model_id IN ({placeholders})"
        params.extend(model_ids)

    if langs:
        placeholders = ",".join("?" * len(langs))
        query += f" AND p.lang IN ({placeholders})"
        params.extend(langs)

    raw = pd.read_sql_query(query, conn, params=params)
    conn.close()

    if raw.empty:
        return pd.DataFrame(columns=[
            "model_type", "lang", "question_id", "response_type",
            "response_value", "prob_forward", "prob_reversed", "prob_averaged",
            "p_valid_forward", "p_valid_reversed",
            "position_bias_magnitude", "n_permutations",
        ])

    return _aggregate_permutations(raw)


def _aggregate_permutations(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-permutation data into forward/reversed/averaged format."""
    rows = []
    group_cols = ["model_type", "lang", "question_id", "response_type"]

    for group_key, grp in raw.groupby(group_cols):
        model_type, lang, qid, rtype = group_key
        n_perms = grp["permutation_idx"].nunique()

        # Get per-permutation data for each response value
        perm_data = {}  # response_value -> {perm_idx: (prob, p_valid)}
        for _, row in grp.iterrows():
            rv = row["response_value"]
            pi = row["permutation_idx"]
            if rv not in perm_data:
                perm_data[rv] = {}
            perm_data[rv][pi] = (row["prob"], row["p_valid"])

        # Forward = permutation_idx 0, reversed = permutation_idx 1
        all_values = sorted(perm_data.keys())

        # Position bias: max across values of (max - min prob across permutations)
        pos_bias = 0.0
        for rv in all_values:
            probs_for_v = [perm_data[rv][pi][0] for pi in perm_data[rv]]
            if len(probs_for_v) > 1:
                pos_bias = max(pos_bias, max(probs_for_v) - min(probs_for_v))

        # Get prompt and question text from forward permutation
        fwd_rows = grp[grp["permutation_idx"] == 0]
        prompt_text = fwd_rows["prompt_text"].iloc[0] if len(fwd_rows) > 0 else ""
        question_text = fwd_rows["question_text"].iloc[0] if len(fwd_rows) > 0 else ""

        for rv in all_values:
            probs_by_perm = perm_data[rv]
            prob_fwd = probs_by_perm.get(0, (0.0, 0.0))[0]
            prob_rev = probs_by_perm.get(1, (0.0, 0.0))[0]
            p_valid_fwd = probs_by_perm.get(0, (0.0, 0.0))[1]
            p_valid_rev = probs_by_perm.get(1, (0.0, 0.0))[1]
            avg = np.mean([p[0] for p in probs_by_perm.values()])

            rows.append({
                "model_type": model_type,
                "lang": lang,
                "question_id": qid,
                "response_type": rtype,
                "response_value": rv,
                "prob_forward": prob_fwd,
                "prob_reversed": prob_rev,
                "prob_averaged": avg,
                "p_valid_forward": p_valid_fwd,
                "p_valid_reversed": p_valid_rev,
                "position_bias_magnitude": pos_bias,
                "n_permutations": n_perms,
                "question_text": question_text,
                "prompt": prompt_text,
            })

    return pd.DataFrame(rows)


def load_results_raw(
    db_path: str,
    model_ids: list[str] | None = None,
    langs: list[str] | None = None,
) -> pd.DataFrame:
    """Load per-permutation evaluation data without aggregation."""
    conn = get_connection(db_path)

    query = """
        SELECT
            m.model_family AS model_type,
            e.model_id,
            p.lang,
            p.question_id,
            q.response_type,
            e.response_value,
            p.permutation_idx,
            e.prob,
            e.p_valid,
            p.prompt_text,
            p.question_text,
            p.value_map,
            p.config,
            p.variant_id,
            p.variant_type
        FROM evaluations e
        JOIN prompts p ON e.prompt_id = p.prompt_id
        JOIN models m ON e.model_id = m.model_id
        JOIN questions q ON p.question_id = q.question_id
        WHERE 1=1
    """
    params = []

    if model_ids:
        placeholders = ",".join("?" * len(model_ids))
        query += f" AND e.model_id IN ({placeholders})"
        params.extend(model_ids)

    if langs:
        placeholders = ",".join("?" * len(langs))
        query += f" AND p.lang IN ({placeholders})"
        params.extend(langs)

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def load_human_distributions(db_path: str) -> pd.DataFrame:
    """Load human distribution data from the database."""
    conn = get_connection(db_path)
    df = pd.read_sql_query(
        "SELECT lang, question_id, response_value, prob_human, "
        "n_respondents, n_valid FROM human_distributions",
        conn,
    )
    conn.close()
    return df


def load_unevaluated_prompts(
    db_path: str,
    model_id: str,
    lang: str | None = None,
    chat_template: bool | None = None,
    config: str | None = None,
) -> pd.DataFrame:
    """Load prompts not yet evaluated by the given model.

    Core query for DB-based inference: returns prompts that need processing.
    """
    conn = get_connection(db_path)

    query = """
        SELECT
            p.prompt_id,
            p.question_id,
            p.lang,
            p.config,
            p.permutation_idx,
            p.prompt_text,
            p.question_text,
            p.valid_values,
            p.value_map,
            p.is_likert10,
            p.chat_template,
            p.variant_id,
            p.variant_type
        FROM prompts p
        WHERE p.prompt_id NOT IN (
            SELECT e.prompt_id FROM evaluations e WHERE e.model_id = ?
        )
    """
    params = [model_id]

    if chat_template is not None:
        query += " AND p.chat_template = ?"
        params.append(int(chat_template))

    if lang:
        query += " AND p.lang = ?"
        params.append(lang)

    if config:
        query += " AND p.config = ?"
        params.append(config)

    query += " ORDER BY p.prompt_id"

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def load_rephrase_results(db_path: str) -> pd.DataFrame:
    """Load rephrase variant evaluation results, compatible with existing analysis.

    Returns DataFrame with columns:
        model_type, question_id, variant_id, variant_type,
        response_value, prob_forward, prob_reversed, prob_averaged,
        p_valid_forward, p_valid_reversed, position_bias_magnitude
    """
    conn = get_connection(db_path)

    query = """
        SELECT
            m.model_family AS model_type,
            p.question_id,
            p.variant_id,
            p.variant_type,
            e.response_value,
            p.permutation_idx,
            e.prob,
            e.p_valid
        FROM evaluations e
        JOIN prompts p ON e.prompt_id = p.prompt_id
        JOIN models m ON e.model_id = m.model_id
        WHERE p.config = 'rephrase' AND p.variant_id IS NOT NULL
    """

    raw = pd.read_sql_query(query, conn)
    conn.close()

    if raw.empty:
        return pd.DataFrame()

    # Aggregate: forward (perm 0) + reversed (perm 1)
    rows = []
    group_cols = ["model_type", "question_id", "variant_id", "variant_type"]
    for group_key, grp in raw.groupby(group_cols):
        model_type, qid, vid, vtype = group_key

        perm_data = {}
        for _, row in grp.iterrows():
            rv = row["response_value"]
            pi = row["permutation_idx"]
            if rv not in perm_data:
                perm_data[rv] = {}
            perm_data[rv][pi] = (row["prob"], row["p_valid"])

        all_values = sorted(perm_data.keys())
        pos_bias = 0.0
        for rv in all_values:
            probs_for_v = [perm_data[rv][pi][0] for pi in perm_data[rv]]
            if len(probs_for_v) > 1:
                pos_bias = max(pos_bias, max(probs_for_v) - min(probs_for_v))

        for rv in all_values:
            probs_by_perm = perm_data[rv]
            prob_fwd = probs_by_perm.get(0, (0.0, 0.0))[0]
            prob_rev = probs_by_perm.get(1, (0.0, 0.0))[0]
            p_valid_fwd = probs_by_perm.get(0, (0.0, 0.0))[1]
            p_valid_rev = probs_by_perm.get(1, (0.0, 0.0))[1]
            avg = np.mean([p[0] for p in probs_by_perm.values()])

            rows.append({
                "model_type": model_type,
                "question_id": qid,
                "variant_id": vid,
                "variant_type": vtype,
                "response_value": rv,
                "prob_forward": prob_fwd,
                "prob_reversed": prob_rev,
                "prob_averaged": avg,
                "p_valid_forward": p_valid_fwd,
                "p_valid_reversed": p_valid_rev,
                "position_bias_magnitude": pos_bias,
            })

    return pd.DataFrame(rows)


def get_evaluation_progress(db_path: str) -> pd.DataFrame:
    """Summary of evaluation progress per model."""
    conn = get_connection(db_path)
    df = pd.read_sql_query(
        """
        SELECT
            e.model_id,
            COUNT(DISTINCT e.prompt_id) AS prompts_evaluated,
            COUNT(*) AS total_rows
        FROM evaluations e
        GROUP BY e.model_id
        """,
        conn,
    )

    # Also get total prompts for context
    total_prompts = conn.execute("SELECT COUNT(*) FROM prompts").fetchone()[0]
    conn.close()

    df["total_prompts"] = total_prompts
    df["pct_complete"] = (df["prompts_evaluated"] / total_prompts * 100).round(1)
    return df
