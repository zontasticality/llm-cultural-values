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


def merge_job_db(target_conn: sqlite3.Connection, job_path: Path, dry_run: bool = False) -> int:
    """Merge completions from a job DB into the target DB. Returns rows added."""
    job_conn = sqlite3.connect(str(job_path))
    job_conn.row_factory = sqlite3.Row

    rows = job_conn.execute("""
        SELECT prompt_id, model_id, sample_idx, completion_raw, completion_text,
               n_tokens_raw, n_tokens, filter_status, temperature, top_p, seed,
               steering_config
        FROM completions
    """).fetchall()

    if not rows:
        job_conn.close()
        return 0

    if dry_run:
        job_conn.close()
        return len(rows)

    inserted = 0
    for row in rows:
        try:
            target_conn.execute(
                "INSERT OR IGNORE INTO completions "
                "(prompt_id, model_id, sample_idx, completion_raw, completion_text, "
                " n_tokens_raw, n_tokens, filter_status, temperature, top_p, seed, "
                " steering_config) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                tuple(row),
            )
            inserted += 1
        except sqlite3.IntegrityError:
            pass  # duplicate, skip

    target_conn.commit()
    job_conn.close()
    return inserted


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

    before = target_conn.execute("SELECT COUNT(*) FROM completions").fetchone()[0]
    print(f"Target DB has {before:,} completions before merge")

    total_added = 0
    for db_path in job_dbs:
        n = merge_job_db(target_conn, db_path, dry_run=args.dry_run)
        status = "would add" if args.dry_run else "added"
        if n > 0:
            print(f"  {db_path.name}: {status} {n:,} rows")
        total_added += n

    after = target_conn.execute("SELECT COUNT(*) FROM completions").fetchone()[0]
    print(f"\n{'Would merge' if args.dry_run else 'Merged'}: {total_added:,} rows")
    if not args.dry_run:
        print(f"Target DB now has {after:,} completions ({after - before:,} new)")

    target_conn.close()


if __name__ == "__main__":
    main()
