"""Populate the culture.db with templates, prompts, and model registrations.

Usage:
    PYTHONPATH=gemma-sae python -m db.populate --db data/culture.db [--pilot]
"""

import argparse
import json
import sqlite3
from pathlib import Path

from analysis.constants import MODELS, PILOT_LANGS, PILOT_TEMPLATES
from db.schema import init_db

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def upsert_templates(conn: sqlite3.Connection):
    """Load templates from prompt_templates.json."""
    templates = json.loads((DATA_DIR / "prompt_templates.json").read_text())
    conn.executemany(
        "INSERT OR IGNORE INTO templates (template_id, template_name, cultural_target) "
        "VALUES (:template_id, :template_name, :cultural_target)",
        templates,
    )
    conn.commit()
    print(f"  Templates: {len(templates)}")


def upsert_prompts(conn: sqlite3.Connection, pilot: bool = False):
    """Load prompts from per-language translation files."""
    trans_dir = DATA_DIR / "translations"
    count = 0
    for path in sorted(trans_dir.glob("*.json")):
        data = json.loads(path.read_text())
        lang = data["lang"]
        if pilot and lang not in PILOT_LANGS:
            continue
        for p in data["prompts"]:
            if pilot and p["template_id"] not in PILOT_TEMPLATES:
                continue
            conn.execute(
                "INSERT OR IGNORE INTO prompts "
                "(template_id, lang, variant_idx, prompt_text) "
                "VALUES (?, ?, ?, ?)",
                (p["template_id"], lang, p["variant_idx"],
                 p["prompt_text"]),
            )
            count += 1
    conn.commit()
    print(f"  Prompts: {count}")


def upsert_models(conn: sqlite3.Connection):
    """Register all models from constants."""
    count = 0
    for model_id, info in MODELS.items():
        conn.execute(
            "INSERT OR IGNORE INTO models "
            "(model_id, model_hf_id, model_family, is_multilingual, dtype) "
            "VALUES (?, ?, ?, ?, ?)",
            (model_id, info["hf_id"], info["family"],
             int(info["multilingual"]), "bf16"),
        )
        count += 1
    conn.commit()
    print(f"  Models: {count}")


def main():
    parser = argparse.ArgumentParser(description="Populate culture.db")
    parser.add_argument("--db", type=str, default=str(DATA_DIR / "culture.db"))
    parser.add_argument("--pilot", action="store_true",
                        help="Only load pilot langs/templates")
    args = parser.parse_args()

    print(f"Initializing DB: {args.db}")
    conn = init_db(args.db)

    upsert_templates(conn)
    upsert_prompts(conn, pilot=args.pilot)
    upsert_models(conn)

    # Summary
    cur = conn.execute("SELECT COUNT(*) FROM templates")
    print(f"\nDB summary:")
    print(f"  templates:  {cur.fetchone()[0]}")
    cur = conn.execute("SELECT COUNT(*) FROM prompts")
    print(f"  prompts:    {cur.fetchone()[0]}")
    cur = conn.execute("SELECT COUNT(*) FROM models")
    print(f"  models:     {cur.fetchone()[0]}")

    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
