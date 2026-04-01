"""Merge per-job SQLite databases into a single culture.db.

Each SLURM job writes to its own DB on node-local storage, then copies it
back to gemma-sae/data/job_dbs/{model_id}_{job_id}.db. This script merges
all completions from job DBs into the main culture.db.

Usage:
    PYTHONPATH=gemma-sae python -m db.merge \
        --target data/culture.db \
        --job-dir data/job_dbs \
        [--dry-run]
"""

import argparse
import sqlite3
from pathlib import Path


def merge_prompt_metrics(target_conn: sqlite3.Connection, job_conn: sqlite3.Connection) -> int:
    """Merge prompt_metrics rows from a job DB. Returns count added."""
    # Check if source has the table
    has_table = job_conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='prompt_metrics'"
    ).fetchone()
    if not has_table:
        return 0

    rows = job_conn.execute("""
        SELECT prompt_id, model_id, prompt_ppl, prompt_logprob,
               prompt_n_tokens, next_token_entropy
        FROM prompt_metrics
    """).fetchall()

    count = 0
    for row in rows:
        try:
            target_conn.execute(
                "INSERT OR IGNORE INTO prompt_metrics "
                "(prompt_id, model_id, prompt_ppl, prompt_logprob, "
                " prompt_n_tokens, next_token_entropy) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                tuple(row),
            )
            count += 1
        except sqlite3.IntegrityError:
            pass

    return count


def merge_job_db(target_conn: sqlite3.Connection, job_path: Path, dry_run: bool = False) -> tuple[int, int, int]:
    """Merge completions, classifications, and prompt_metrics from a job DB.
    Returns (completions_added, classifications_added, metrics_added)."""
    job_conn = sqlite3.connect(str(job_path))
    job_conn.row_factory = sqlite3.Row

    comp_rows = job_conn.execute("""
        SELECT prompt_id, model_id, sample_idx, completion_raw, completion_text,
               n_tokens_raw, n_tokens, filter_status, temperature, top_p, seed,
               steering_config
        FROM completions
    """).fetchall()

    # Check if classifications table has the new prob columns
    job_cols = {row[1] for row in job_conn.execute("PRAGMA table_info(classifications)")}
    has_probs = "dim_ic_probs" in job_cols

    if has_probs:
        cls_rows = job_conn.execute("""
            SELECT completion_id, classifier_model, content_category,
                   dim_indiv_collect, dim_trad_secular, dim_surv_selfexpr,
                   raw_response, dim_ic_probs, dim_ts_probs, dim_ss_probs, cat_probs
            FROM classifications
        """).fetchall()
    else:
        cls_rows = job_conn.execute("""
            SELECT completion_id, classifier_model, content_category,
                   dim_indiv_collect, dim_trad_secular, dim_surv_selfexpr,
                   raw_response
            FROM classifications
        """).fetchall()

    if not comp_rows and not cls_rows:
        # Still check for prompt_metrics even if no completions/classifications
        met_count = 0
        if not dry_run:
            met_count = merge_prompt_metrics(target_conn, job_conn)
            if met_count:
                target_conn.commit()
        job_conn.close()
        return 0, 0, met_count

    if dry_run:
        job_conn.close()
        return len(comp_rows), len(cls_rows), 0

    comp_inserted = 0
    for row in comp_rows:
        try:
            target_conn.execute(
                "INSERT OR IGNORE INTO completions "
                "(prompt_id, model_id, sample_idx, completion_raw, completion_text, "
                " n_tokens_raw, n_tokens, filter_status, temperature, top_p, seed, "
                " steering_config) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                tuple(row),
            )
            comp_inserted += 1
        except sqlite3.IntegrityError:
            pass

    cls_inserted = 0
    for row in cls_rows:
        try:
            if has_probs:
                target_conn.execute(
                    "INSERT OR IGNORE INTO classifications "
                    "(completion_id, classifier_model, content_category, "
                    " dim_indiv_collect, dim_trad_secular, dim_surv_selfexpr, "
                    " raw_response, dim_ic_probs, dim_ts_probs, dim_ss_probs, cat_probs) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    tuple(row),
                )
            else:
                target_conn.execute(
                    "INSERT OR IGNORE INTO classifications "
                    "(completion_id, classifier_model, content_category, "
                    " dim_indiv_collect, dim_trad_secular, dim_surv_selfexpr, raw_response) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    tuple(row),
                )
            cls_inserted += 1
        except sqlite3.IntegrityError:
            pass

    met_count = merge_prompt_metrics(target_conn, job_conn)

    target_conn.commit()
    job_conn.close()
    return comp_inserted, cls_inserted, met_count


def main():
    parser = argparse.ArgumentParser(description="Merge per-job DBs into culture.db")
    parser.add_argument("--target", required=True, help="Path to target culture.db")
    parser.add_argument("--job-dir", required=True, help="Directory containing job DBs")
    parser.add_argument("--dry-run", action="store_true", help="Count rows without merging")
    args = parser.parse_args()

    job_dir = Path(args.job_dir)
    job_dbs = sorted(job_dir.glob("*.db"))

    if not job_dbs:
        print(f"No job DBs found in {job_dir}")
        return

    print(f"Found {len(job_dbs)} job DBs in {job_dir}")

    target_conn = sqlite3.connect(args.target)
    target_conn.execute("PRAGMA journal_mode=DELETE")
    target_conn.execute("PRAGMA busy_timeout=10000")

    # Ensure target has new prob columns
    from db.schema import migrate_classifications_probs
    migrate_classifications_probs(target_conn)

    before_comp = target_conn.execute("SELECT COUNT(*) FROM completions").fetchone()[0]
    before_cls = target_conn.execute("SELECT COUNT(*) FROM classifications").fetchone()[0]
    print(f"Target DB has {before_comp:,} completions, {before_cls:,} classifications before merge")

    total_comp = 0
    total_cls = 0
    total_met = 0
    for db_path in job_dbs:
        nc, ncl, nm = merge_job_db(target_conn, db_path, dry_run=args.dry_run)
        status = "would add" if args.dry_run else "added"
        parts = []
        if nc > 0:
            parts.append(f"{nc:,} completions")
        if ncl > 0:
            parts.append(f"{ncl:,} classifications")
        if nm > 0:
            parts.append(f"{nm:,} prompt_metrics")
        if parts:
            print(f"  {db_path.name}: {status} {', '.join(parts)}")
        total_comp += nc
        total_cls += ncl
        total_met += nm

    after_comp = target_conn.execute("SELECT COUNT(*) FROM completions").fetchone()[0]
    after_cls = target_conn.execute("SELECT COUNT(*) FROM classifications").fetchone()[0]
    verb = "Would merge" if args.dry_run else "Merged"
    print(f"\n{verb}: {total_comp:,} completion rows, {total_cls:,} classification rows"
          + (f", {total_met:,} prompt_metrics rows" if total_met > 0 else ""))
    if not args.dry_run:
        print(f"Target DB now has {after_comp:,} completions ({after_comp - before_comp:,} new), "
              f"{after_cls:,} classifications ({after_cls - before_cls:,} new)")

    target_conn.close()


if __name__ == "__main__":
    main()
