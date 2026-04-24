"""Language-detect every completion with lingua.

Populates completions.detected_lang (ISO 639-3) and detected_lang_conf.

Lingua returns iso_639_3 directly; one remap: Latvian comes out as 'lav'
in lingua but our schema uses 'lvs'. No other remaps needed.

Completions shorter than MIN_CHARS are left with NULL detected_lang — too
short to classify reliably. The languages parameter restricts detection to
prompt langs + common drift targets (English, Indonesian, Russian, etc.)
so that short mixed text is not matched against all 75 lingua languages.

Run:
    PYTHONPATH=. .venv/bin/python scripts/detect_lang.py --db data/culture.db
"""
import argparse
import sqlite3
import time

from lingua import Language, LanguageDetectorBuilder

from db.schema import migrate_completions_lang

MIN_CHARS = 10

LANG_NAMES = [
    "ENGLISH", "GERMAN", "FRENCH", "SPANISH", "ITALIAN", "PORTUGUESE",
    "DUTCH", "SWEDISH", "DANISH", "FINNISH", "NORWEGIAN", "BOKMAL", "NYNORSK",
    "CZECH", "SLOVAK", "SLOVENE", "POLISH", "HUNGARIAN",
    "GREEK", "BULGARIAN", "CROATIAN", "SERBIAN", "ROMANIAN",
    "ESTONIAN", "LITHUANIAN", "LATVIAN",
    "TURKISH", "ARABIC", "HINDI", "CHINESE", "JAPANESE",
    "RUSSIAN", "UKRAINIAN", "BELARUSIAN",
    "INDONESIAN", "MALAY", "KOREAN", "VIETNAMESE",
]

# lingua iso639_3 → our schema iso code (only needed where they differ)
ISO_REMAP = {"lav": "lvs"}


def build_detector():
    langs = []
    for name in LANG_NAMES:
        if hasattr(Language, name):
            langs.append(getattr(Language, name))
    print(f"Detector: {len(langs)} languages", flush=True)
    return LanguageDetectorBuilder.from_languages(*langs).with_preloaded_language_models().build()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/culture.db")
    ap.add_argument("--batch", type=int, default=5000,
                    help="Commit after each batch of N detections")
    ap.add_argument("--only-missing", action="store_true", default=True,
                    help="Skip rows where detected_lang is already set")
    ap.add_argument("--report-only", action="store_true",
                    help="Print per-pair lang-mismatch report from existing detections; do not re-run detection")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    migrate_completions_lang(conn)

    if args.report_only:
        report(conn)
        return

    det = build_detector()

    where = "completion_text IS NOT NULL"
    if args.only_missing:
        where += " AND detected_lang IS NULL"
    rows = conn.execute(
        f"SELECT completion_id, completion_text FROM completions WHERE {where}"
    ).fetchall()
    print(f"To process: {len(rows):,} completions", flush=True)
    if not rows:
        report(conn)
        return

    t0 = time.time()
    updates = []
    n_too_short = 0
    for i, (cid, text) in enumerate(rows):
        if len(text) < MIN_CHARS:
            n_too_short += 1
            updates.append((None, None, cid))
        else:
            res = det.detect_language_of(text)
            if res is None:
                updates.append((None, None, cid))
            else:
                iso = res.iso_code_639_3.name.lower()
                iso = ISO_REMAP.get(iso, iso)
                conf = det.compute_language_confidence(text, res)
                updates.append((iso, conf, cid))

        if len(updates) >= args.batch:
            conn.executemany(
                "UPDATE completions SET detected_lang=?, detected_lang_conf=? WHERE completion_id=?",
                updates,
            )
            conn.commit()
            updates.clear()
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(rows) - i - 1) / rate
            print(f"  {i+1:,}/{len(rows):,}  {rate:.0f}/s  eta {eta/60:.1f}min", flush=True)

    if updates:
        conn.executemany(
            "UPDATE completions SET detected_lang=?, detected_lang_conf=? WHERE completion_id=?",
            updates,
        )
        conn.commit()

    print(f"\nDone in {(time.time()-t0)/60:.1f}min. Too-short skipped: {n_too_short:,}")
    report(conn)


def report(conn: sqlite3.Connection):
    print("\n" + "=" * 72)
    print("LANGUAGE DETECTION REPORT")
    print("=" * 72)

    total, detected, match, mismatch, too_short = conn.execute("""
        SELECT COUNT(*),
               SUM(detected_lang IS NOT NULL),
               SUM(detected_lang IS NOT NULL AND detected_lang = p.lang),
               SUM(detected_lang IS NOT NULL AND detected_lang != p.lang),
               SUM(detected_lang IS NULL)
        FROM completions c JOIN prompts p ON c.prompt_id = p.prompt_id
        WHERE c.completion_text IS NOT NULL
    """).fetchone()
    print(f"  Total completions:      {total:,}")
    print(f"  Detected:               {detected:,}  ({100*detected/total:.1f}%)")
    print(f"    → matches prompt lang:  {match:,}  ({100*match/total:.1f}%)")
    print(f"    → mismatched:           {mismatch:,}  ({100*mismatch/total:.1f}%)")
    print(f"  Too short / NULL:       {too_short:,}  ({100*too_short/total:.1f}%)")

    print("\n--- Mismatch rate by model ---")
    rows = conn.execute("""
        SELECT c.model_id,
               COUNT(*) AS n,
               SUM(detected_lang IS NOT NULL AND detected_lang != p.lang) AS mism,
               SUM(detected_lang IS NOT NULL AND detected_lang = p.lang) AS match_
        FROM completions c JOIN prompts p ON c.prompt_id = p.prompt_id
        WHERE c.completion_text IS NOT NULL
        GROUP BY c.model_id
        ORDER BY mism * 1.0 / n DESC
    """).fetchall()
    print(f"  {'model':<22}  {'n':>7}  {'mism':>6}  {'match':>6}  {'mism%':>6}")
    for model, n, mism, match_ in rows[:15]:
        pct = 100 * mism / n if n else 0
        print(f"  {model:<22}  {n:>7,}  {mism:>6,}  {match_:>6,}  {pct:>5.1f}%")

    print("\n--- Top mismatched (model, prompt_lang) cells ---")
    rows = conn.execute("""
        SELECT c.model_id, p.lang,
               SUM(detected_lang IS NOT NULL AND detected_lang != p.lang) AS mism,
               COUNT(*) AS n
        FROM completions c JOIN prompts p ON c.prompt_id = p.prompt_id
        WHERE c.completion_text IS NOT NULL
        GROUP BY c.model_id, p.lang
        HAVING n >= 100
        ORDER BY mism * 1.0 / n DESC
        LIMIT 20
    """).fetchall()
    print(f"  {'model':<22}  {'lang':<5}  {'n':>6}  {'mism':>5}  {'mism%':>6}")
    for model, lang, mism, n in rows:
        pct = 100 * mism / n if n else 0
        print(f"  {model:<22}  {lang:<5}  {n:>6,}  {mism:>5,}  {pct:>5.1f}%")

    print("\n--- Top drift destinations (where mismatches land) ---")
    rows = conn.execute("""
        SELECT detected_lang, COUNT(*) AS n
        FROM completions c JOIN prompts p ON c.prompt_id = p.prompt_id
        WHERE detected_lang IS NOT NULL AND detected_lang != p.lang
        GROUP BY detected_lang
        ORDER BY n DESC
        LIMIT 15
    """).fetchall()
    for lang, n in rows:
        print(f"  {lang}: {n:,}")


if __name__ == "__main__":
    main()
