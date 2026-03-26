"""Query helpers for the culture database.

All functions take a sqlite3.Connection and return lists of dicts or Row objects.
"""

import sqlite3


def load_unsampled_prompts(
    conn: sqlite3.Connection,
    model_id: str,
    n_samples: int = 200,
    steering_config: str = "none",
    lang: str | None = None,
    template_id: str | None = None,
) -> list[dict]:
    """Return prompts with fewer than n_samples completions for this model.

    Each row: {prompt_id, template_id, lang, prompt_text, existing_count, needed}
    """
    sql = """
        SELECT p.prompt_id, p.template_id, p.lang, p.prompt_text,
               COALESCE(cnt.n, 0) AS existing_count,
               ? - COALESCE(cnt.n, 0) AS needed
        FROM prompts p
        LEFT JOIN (
            SELECT prompt_id, COUNT(*) AS n
            FROM completions
            WHERE model_id = ? AND steering_config = ?
            GROUP BY prompt_id
        ) cnt ON p.prompt_id = cnt.prompt_id
        WHERE COALESCE(cnt.n, 0) < ?
    """
    params: list = [n_samples, model_id, steering_config, n_samples]
    if lang:
        sql += " AND p.lang = ?"
        params.append(lang)
    if template_id:
        sql += " AND p.template_id = ?"
        params.append(template_id)
    sql += " ORDER BY p.prompt_id"

    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, params).fetchall()
    conn.row_factory = None
    return [dict(r) for r in rows]


def load_unclassified_completions(
    conn: sqlite3.Connection,
    classifier_model: str,
    limit: int | None = None,
) -> list[dict]:
    """Return ok-filtered completions lacking a classification for this classifier.

    Each row: {completion_id, completion_text, template_id, lang, model_id}
    """
    sql = """
        SELECT c.completion_id, c.completion_text,
               p.template_id, p.lang, c.model_id
        FROM completions c
        JOIN prompts p ON c.prompt_id = p.prompt_id
        LEFT JOIN classifications cl
            ON c.completion_id = cl.completion_id
            AND cl.classifier_model = ?
        WHERE c.filter_status = 'ok'
          AND c.completion_text IS NOT NULL
          AND cl.completion_id IS NULL
        ORDER BY c.completion_id
    """
    params: list = [classifier_model]
    if limit:
        sql += " LIMIT ?"
        params.append(limit)

    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, params).fetchall()
    conn.row_factory = None
    return [dict(r) for r in rows]


def load_results(
    conn: sqlite3.Connection,
    model_ids: list[str] | None = None,
    langs: list[str] | None = None,
    template_ids: list[str] | None = None,
    classifier_model: str | None = None,
) -> list[dict]:
    """Load classified completions joined with prompt/template metadata."""
    sql = """
        SELECT c.completion_id, c.model_id, c.completion_text, c.filter_status,
               c.temperature, c.steering_config,
               p.template_id, p.lang, p.prompt_text, p.variant_idx,
               cl.classifier_model, cl.content_category,
               cl.dim_indiv_collect, cl.dim_trad_secular, cl.dim_surv_selfexpr
        FROM completions c
        JOIN prompts p ON c.prompt_id = p.prompt_id
        JOIN classifications cl ON c.completion_id = cl.completion_id
        WHERE 1=1
    """
    params: list = []
    if model_ids:
        placeholders = ",".join("?" * len(model_ids))
        sql += f" AND c.model_id IN ({placeholders})"
        params.extend(model_ids)
    if langs:
        placeholders = ",".join("?" * len(langs))
        sql += f" AND p.lang IN ({placeholders})"
        params.extend(langs)
    if template_ids:
        placeholders = ",".join("?" * len(template_ids))
        sql += f" AND p.template_id IN ({placeholders})"
        params.extend(template_ids)
    if classifier_model:
        sql += " AND cl.classifier_model = ?"
        params.append(classifier_model)

    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, params).fetchall()
    conn.row_factory = None
    return [dict(r) for r in rows]


def get_sampling_progress(conn: sqlite3.Connection) -> list[dict]:
    """Per-(model, lang, template) completion counts and filter rates."""
    sql = """
        SELECT c.model_id, p.lang, p.template_id,
               COUNT(*) AS total,
               SUM(CASE WHEN c.filter_status = 'ok' THEN 1 ELSE 0 END) AS ok,
               ROUND(1.0 - (1.0 * SUM(CASE WHEN c.filter_status = 'ok' THEN 1 ELSE 0 END) / COUNT(*)), 3) AS filter_rate
        FROM completions c
        JOIN prompts p ON c.prompt_id = p.prompt_id
        GROUP BY c.model_id, p.lang, p.template_id
        ORDER BY c.model_id, p.lang, p.template_id
    """
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql).fetchall()
    conn.row_factory = None
    return [dict(r) for r in rows]
