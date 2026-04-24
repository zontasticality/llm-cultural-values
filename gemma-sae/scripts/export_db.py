"""Checkpoint WAL and export culture.db to a clean copy or CSV.

Usage:
    # DB copy (checkpoint WAL first):
    python scripts/export_db.py culture.db culture_clean.db

    # CSV export (all completions + classifications):
    python scripts/export_db.py culture.db completions.csv --csv

    # CSV export (trimmed prompts only):
    python scripts/export_db.py culture.db completions.csv --csv --trimmed-only

    # CSV export (trimmed + spaceless CJK originals):
    python scripts/export_db.py culture.db completions.csv --csv --clean
"""
import argparse
import csv
import sqlite3
import shutil

from analysis.constants import TRIMMED_VARIANT_MIN


def export_db_copy(src: str, dst: str):
    conn = sqlite3.connect(src)
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM completions").fetchone()[0]
    print(f"Source: {src}, journal_mode={mode}, completions={total}")

    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.close()

    shutil.copy2(src, dst)

    conn2 = sqlite3.connect(dst)
    total2 = conn2.execute("SELECT COUNT(*) FROM completions").fetchone()[0]
    print(f"Export: {dst}, completions={total2}")
    conn2.close()


def export_csv(src: str, dst: str, trimmed_only: bool = False, clean: bool = False, langs: list[str] | None = None):
    conn = sqlite3.connect(src)
    conn.row_factory = sqlite3.Row

    sql = """
        SELECT m.model_hf_id   AS model,
               m.model_family,
               p.lang,
               t.template_name,
               t.cultural_target,
               p.variant_idx,
               c.sample_idx,
               c.filter_status,
               p.prompt_text    AS prompt,
               c.completion_text AS completion,
               c.n_tokens,
               cl.classifier_model,
               cl.content_category,
               cl.dim_indiv_collect,
               cl.dim_trad_secular,
               cl.dim_surv_selfexpr
        FROM completions c
        JOIN prompts   p ON c.prompt_id  = p.prompt_id
        JOIN templates t ON p.template_id = t.template_id
        JOIN models    m ON c.model_id   = m.model_id
        LEFT JOIN classifications cl ON c.completion_id = cl.completion_id
        WHERE 1=1
    """
    params: list = []
    if trimmed_only:
        sql += f" AND p.variant_idx >= {TRIMMED_VARIANT_MIN}"
    if clean:
        # Trimmed prompts (variant_idx >= TRIMMED_VARIANT_MIN) for languages that had the
        # trailing-space tokenization issue, plus originals (variant_idx < TRIMMED_VARIANT_MIN)
        # for languages whose prompts never had trailing spaces (jpn, zho).
        spaceless_langs = [r[0] for r in conn.execute(f"""
            SELECT DISTINCT lang FROM prompts
            WHERE variant_idx < {TRIMMED_VARIANT_MIN}
            GROUP BY lang
            HAVING SUM(prompt_text LIKE '%% ') = 0
        """).fetchall()]
        if spaceless_langs:
            ph = ",".join("?" * len(spaceless_langs))
            sql += f" AND ((p.variant_idx >= {TRIMMED_VARIANT_MIN}) OR (p.variant_idx < {TRIMMED_VARIANT_MIN} AND p.lang IN ({ph})))"
            params.extend(spaceless_langs)
        else:
            sql += f" AND p.variant_idx >= {TRIMMED_VARIANT_MIN}"
    if langs:
        placeholders = ",".join("?" * len(langs))
        sql += f" AND p.lang IN ({placeholders})"
        params.extend(langs)
    sql += " ORDER BY m.model_hf_id, p.lang, t.template_name, p.variant_idx, c.sample_idx"

    rows = conn.execute(sql, params).fetchall()
    columns = rows[0].keys() if rows else []

    with open(dst, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)

    conn.close()
    label = "clean " if clean else "trimmed-only " if trimmed_only else ""
    print(f"Exported {len(rows)} {label}rows to {dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export culture.db to DB copy or CSV")
    parser.add_argument("src", help="Source database path")
    parser.add_argument("dst", help="Destination path (.db or .csv)")
    parser.add_argument("--csv", action="store_true", help="Export as CSV instead of DB copy")
    parser.add_argument("--trimmed-only", action="store_true",
                        help=f"Only include trimmed prompts (variant_idx >= {TRIMMED_VARIANT_MIN})")
    parser.add_argument("--clean", action="store_true",
                        help="Trimmed prompts + originals for spaceless languages (jpn, zho)")
    parser.add_argument("--langs", nargs="+", default=None,
                        help="Only include these language codes (e.g. eng spa hin)")
    args = parser.parse_args()

    if args.csv:
        export_csv(args.src, args.dst, trimmed_only=args.trimmed_only, clean=args.clean, langs=args.langs)
    else:
        export_db_copy(args.src, args.dst)
