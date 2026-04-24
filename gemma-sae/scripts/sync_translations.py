"""Sync prompts table ⇄ data/translations/{lang}.json.

Two operations, run in order:

  1. INSERT alternatives.
     Read alternative prompts from prompts.quality_notes (JSON with key
     "alternatives") and insert them as new rows in prompts with
     variant_idx >= ALT_VARIANT_MIN (default 200), one variant per
     alternative. INSERT OR IGNORE — re-running is a no-op.

  2. DUMP prompts → JSONs.
     For each language, write all prompt rows (variant_idx >= 0) into
     data/translations/{lang}.json, overwriting the file. After this, the
     JSONs are authoritative again — db.populate on a fresh DB will
     reproduce the current prompt set, including v100 trimmed prompts and
     v200 alternatives.

Usage:
    PYTHONPATH=. python scripts/sync_translations.py \
        --db data/culture.db [--insert-only] [--dump-only]
"""
import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

from analysis.constants import LANG_NAMES

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRANS_DIR = DATA_DIR / "translations"
ALT_VARIANT_MIN = 200


def insert_alternatives(conn: sqlite3.Connection) -> int:
    """Insert prompts.quality_notes['alternatives'] as new prompt rows.

    Idempotent: an alternative is identified by its prompt_text, so re-running
    will not insert duplicates even though variant_idx is auto-assigned.
    Variant_idx is allocated as max(existing, ALT_VARIANT_MIN-1) + 1 per
    (template_id, lang).
    """
    rows = conn.execute(
        "SELECT prompt_id, lang, template_id, prompt_text, quality_notes "
        "FROM prompts WHERE quality_notes IS NOT NULL ORDER BY prompt_id"
    ).fetchall()

    # Existing (template, lang) -> set of prompt_texts already in DB
    existing_texts: dict[tuple[str, str], set[str]] = defaultdict(set)
    # Current max variant_idx per (template, lang) for allocation
    max_idx: dict[tuple[str, str], int] = {}
    for tmpl, lang, vidx, text in conn.execute(
        "SELECT template_id, lang, variant_idx, prompt_text FROM prompts"
    ):
        existing_texts[(tmpl, lang)].add(text)
        cur = max_idx.get((tmpl, lang), -1)
        if vidx > cur:
            max_idx[(tmpl, lang)] = vidx

    inserted = 0
    skipped = 0
    for pid, lang, tmpl, text, notes_raw in rows:
        try:
            notes = json.loads(notes_raw)
        except json.JSONDecodeError:
            print(f"  [{pid}] {lang}/{tmpl}: malformed notes JSON, skipping")
            continue
        alternatives = notes.get("alternatives", [])
        if not alternatives:
            continue

        for alt_text in alternatives:
            key = (tmpl, lang)
            if alt_text in existing_texts[key]:
                skipped += 1
                continue
            base = max(max_idx.get(key, -1), ALT_VARIANT_MIN - 1)
            new_vidx = base + 1
            conn.execute(
                "INSERT INTO prompts "
                "(template_id, lang, variant_idx, prompt_text) VALUES (?, ?, ?, ?)",
                (tmpl, lang, new_vidx, alt_text),
            )
            existing_texts[key].add(alt_text)
            max_idx[key] = new_vidx
            inserted += 1
            print(f"  + [{lang}/{tmpl}] vidx={new_vidx}: {alt_text!r}")

    conn.commit()
    if skipped:
        print(f"  ({skipped} alternatives already in DB, skipped)")
    return inserted


def dump_to_json(conn: sqlite3.Connection) -> int:
    """Write all prompts back to data/translations/{lang}.json, one file per lang.

    Replaces the file contents fully — JSONs become authoritative.
    """
    by_lang: dict[str, list[dict]] = defaultdict(list)
    for tmpl, lang, vidx, text in conn.execute(
        "SELECT template_id, lang, variant_idx, prompt_text "
        "FROM prompts ORDER BY lang, template_id, variant_idx"
    ):
        by_lang[lang].append({
            "template_id": tmpl,
            "variant_idx": vidx,
            "prompt_text": text,
        })

    TRANS_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for lang, prompts in sorted(by_lang.items()):
        path = TRANS_DIR / f"{lang}.json"
        payload = {"lang": lang, "prompts": prompts}
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        v_counts = defaultdict(int)
        for p in prompts:
            v_counts[(p["variant_idx"] // 100) * 100] += 1
        bucket_str = ", ".join(f"v{k}+={v}" for k, v in sorted(v_counts.items()))
        print(f"  {lang} ({LANG_NAMES.get(lang, '?')}): {len(prompts):>3d} prompts  [{bucket_str}]")
        written += 1
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DATA_DIR / "culture.db"))
    ap.add_argument("--insert-only", action="store_true",
                    help="Only insert alternatives, do not rewrite JSONs")
    ap.add_argument("--dump-only", action="store_true",
                    help="Only rewrite JSONs from current DB state")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)

    if not args.dump_only:
        print("=== Inserting alternatives from quality_notes ===")
        n = insert_alternatives(conn)
        print(f"  → inserted {n} new prompt rows")

    if not args.insert_only:
        print("\n=== Dumping prompts to translation JSONs ===")
        n = dump_to_json(conn)
        print(f"  → wrote {n} JSON files")

    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
