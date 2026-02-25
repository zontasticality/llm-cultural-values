"""Populate the survey database with questions, prompts, models, and human data.

Generates prompt permutations from questions.json and inserts them into the
prompts table. Registers models and imports human distribution data.

Usage:
    PYTHONPATH=eurollm eurollm/.venv/bin/python -m db.populate \
        --questions eurollm/data/questions.json \
        --db eurollm/data/survey.db

    # With prompt config, human data, and rephrasings:
    PYTHONPATH=eurollm eurollm/.venv/bin/python -m db.populate \
        --questions eurollm/data/questions.json \
        --prompt-config eurollm/data/prompt_config.json \
        --human-data eurollm/human_data/data/human_distributions.parquet \
        --rephrasings eurollm/data/rephrasings.json \
        --db eurollm/data/survey.db
"""

import argparse
import json
import sqlite3
from pathlib import Path

from db.schema import init_db
from prompting.prompt_templates import (
    clean_text,
    format_prompt,
    format_prompt_custom,
    format_prompt_permuted,
)
from inference.extract_logprobs import generate_permutations, get_format_kwargs


# All 22 languages in the project
ALL_LANGS = [
    "bul", "ces", "dan", "deu", "ell", "eng", "est", "fin", "fra", "hrv",
    "hun", "ita", "lit", "lvs", "nld", "pol", "por", "ron", "slk", "slv",
    "spa", "swe",
]

# Known models for initial registration
KNOWN_MODELS = [
    ("hplt2c_bul", "HPLT/hplt2c_bul_checkpoints", "hplt2c", False),
    ("hplt2c_ces", "HPLT/hplt2c_ces_checkpoints", "hplt2c", False),
    ("hplt2c_dan", "HPLT/hplt2c_dan_checkpoints", "hplt2c", False),
    ("hplt2c_deu", "HPLT/hplt2c_deu_checkpoints", "hplt2c", False),
    ("hplt2c_ell", "HPLT/hplt2c_ell_checkpoints", "hplt2c", False),
    ("hplt2c_eng", "HPLT/hplt2c_eng_checkpoints", "hplt2c", False),
    ("hplt2c_est", "HPLT/hplt2c_est_checkpoints", "hplt2c", False),
    ("hplt2c_fin", "HPLT/hplt2c_fin_checkpoints", "hplt2c", False),
    ("hplt2c_fra", "HPLT/hplt2c_fra_checkpoints", "hplt2c", False),
    ("hplt2c_hrv", "HPLT/hplt2c_hrv_checkpoints", "hplt2c", False),
    ("hplt2c_hun", "HPLT/hplt2c_hun_checkpoints", "hplt2c", False),
    ("hplt2c_ita", "HPLT/hplt2c_ita_checkpoints", "hplt2c", False),
    ("hplt2c_lit", "HPLT/hplt2c_lit_checkpoints", "hplt2c", False),
    ("hplt2c_lvs", "HPLT/hplt2c_lvs_checkpoints", "hplt2c", False),
    ("hplt2c_nld", "HPLT/hplt2c_nld_checkpoints", "hplt2c", False),
    ("hplt2c_pol", "HPLT/hplt2c_pol_checkpoints", "hplt2c", False),
    ("hplt2c_por", "HPLT/hplt2c_por_checkpoints", "hplt2c", False),
    ("hplt2c_ron", "HPLT/hplt2c_ron_checkpoints", "hplt2c", False),
    ("hplt2c_slk", "HPLT/hplt2c_slk_checkpoints", "hplt2c", False),
    ("hplt2c_slv", "HPLT/hplt2c_slv_checkpoints", "hplt2c", False),
    ("hplt2c_spa", "HPLT/hplt2c_spa_checkpoints", "hplt2c", False),
    ("hplt2c_swe", "HPLT/hplt2c_swe_checkpoints", "hplt2c", False),
    ("eurollm22b", "utter-project/EuroLLM-22B-2512", "eurollm22b", True),
    ("gemma3_27b_pt", "google/gemma-3-27b-pt", "gemma3_27b_pt", True),
    ("gemma3_27b_it", "google/gemma-3-27b-it", "gemma3_27b_it", True),
    ("qwen3235b", "Qwen/Qwen3-235B-A22B", "qwen3235b", True),
]


def populate_questions(conn: sqlite3.Connection, questions_path: str | Path):
    """Insert question metadata from questions.json."""
    with open(questions_path) as f:
        data = json.load(f)

    rows = [(q["canonical_id"], q["response_type"]) for q in data["questions"]]
    conn.executemany(
        "INSERT OR IGNORE INTO questions (question_id, response_type) VALUES (?, ?)",
        rows,
    )
    conn.commit()
    print(f"  Inserted/skipped {len(rows)} questions")


def populate_prompts(
    conn: sqlite3.Connection,
    questions_path: str | Path,
    config_name: str = "legacy",
    prompt_config_path: str | Path | None = None,
    n_permutations: int = 6,
    langs: list[str] | None = None,
    chat_template: bool = False,
):
    """Generate prompt permutations and insert into prompts table.

    Args:
        conn: Database connection.
        questions_path: Path to questions.json.
        config_name: Config name label (e.g. "legacy", "optimized_v1").
        prompt_config_path: Path to prompt_config.json (None for legacy format).
        n_permutations: Number of option-order permutations per question.
        langs: Languages to generate for (default: all 22).
        chat_template: Whether prompts use chat template wrapping.
    """
    with open(questions_path) as f:
        data = json.load(f)
    questions = data["questions"]

    prompt_config = None
    if prompt_config_path:
        with open(prompt_config_path) as f:
            prompt_config = json.load(f)
        # Override n_perms from config if present
        config_perms = prompt_config.get("default", {}).get("n_perms")
        if config_perms:
            n_permutations = config_perms

    if langs is None:
        langs = ALL_LANGS

    chat_template_int = 1 if chat_template else 0
    inserted = 0
    skipped = 0

    for question in questions:
        qid = question["canonical_id"]
        rtype = question["response_type"]

        fmt_kwargs = get_format_kwargs(prompt_config, rtype) if prompt_config else None

        for lang in langs:
            if lang not in question["translations"]:
                continue

            trans = question["translations"][lang]
            question_text = clean_text(trans["text"])

            # Use actual options count for this translation (may differ from canonical)
            n_options = 10 if rtype == "likert10" else len(trans["options"])
            if n_options == 0:
                continue
            perms = generate_permutations(n_options, n_permutations, qid)

            for perm_idx, perm in enumerate(perms):
                if prompt_config:
                    formatted = format_prompt_custom(question, lang, perm, **fmt_kwargs)
                elif perm_idx == 0:
                    formatted = format_prompt(question, lang, reverse=False)
                elif perm_idx == 1 and n_permutations >= 2:
                    formatted = format_prompt(question, lang, reverse=True)
                else:
                    formatted = format_prompt_permuted(question, lang, perm)

                valid_values_json = json.dumps(formatted["valid_values"])
                value_map_json = json.dumps(formatted["value_map"])
                is_likert10_int = 1 if formatted["is_likert10"] else 0

                try:
                    conn.execute(
                        """INSERT INTO prompts
                           (question_id, lang, config, permutation_idx, prompt_text,
                            question_text, valid_values, value_map, is_likert10,
                            chat_template, variant_id, variant_type)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)""",
                        (qid, lang, config_name, perm_idx, formatted["prompt"],
                         question_text, valid_values_json, value_map_json,
                         is_likert10_int, chat_template_int),
                    )
                    inserted += 1
                except sqlite3.IntegrityError:
                    skipped += 1

    conn.commit()
    print(f"  Prompts: {inserted} inserted, {skipped} skipped (duplicates)")


def populate_rephrase_prompts(
    conn: sqlite3.Connection,
    questions_path: str | Path,
    rephrasings_path: str | Path,
):
    """Generate rephrase variant prompts and insert into prompts table.

    Uses forward + reversed (K=2) matching the existing rephrase_test.py behavior,
    since format_prompt with text_override only supports forward/reversed.
    """
    with open(questions_path) as f:
        data = json.load(f)
    questions_by_id = {q["canonical_id"]: q for q in data["questions"]}

    with open(rephrasings_path) as f:
        entries = json.load(f)

    inserted = 0
    skipped = 0

    def _insert_variant(qid, question, text, variant_id, variant_type):
        nonlocal inserted, skipped
        lang = "eng"
        question_text = clean_text(text)

        for perm_idx, reverse in enumerate([False, True]):
            formatted = format_prompt(question, lang, reverse=reverse,
                                      text_override=text)
            valid_values_json = json.dumps(formatted["valid_values"])
            value_map_json = json.dumps(formatted["value_map"])
            is_likert10_int = 1 if formatted["is_likert10"] else 0

            try:
                conn.execute(
                    """INSERT INTO prompts
                       (question_id, lang, config, permutation_idx, prompt_text,
                        question_text, valid_values, value_map, is_likert10,
                        chat_template, variant_id, variant_type)
                       VALUES (?, ?, 'rephrase', ?, ?, ?, ?, ?, ?, 0, ?, ?)""",
                    (qid, lang, perm_idx, formatted["prompt"],
                     question_text, valid_values_json, value_map_json,
                     is_likert10_int, variant_id, variant_type),
                )
                inserted += 1
            except sqlite3.IntegrityError:
                skipped += 1

    for entry in entries:
        qid = entry["canonical_id"]
        question = questions_by_id.get(qid)
        if question is None:
            continue
        if "eng" not in question["translations"]:
            continue

        # Original text
        _insert_variant(qid, question, entry["original_text"],
                        f"{qid}_original", "original")

        # Each variant
        for variant in entry["variants"]:
            _insert_variant(qid, question, variant["text"],
                            variant["id"], variant["variant_type"])

    conn.commit()
    print(f"  Rephrase prompts: {inserted} inserted, {skipped} skipped")


def register_model(
    conn: sqlite3.Connection,
    model_id: str,
    model_hf_id: str,
    model_family: str,
    is_multilingual: bool,
    dtype: str = "bf16",
    chat_template: bool = False,
    notes: str | None = None,
):
    """Insert or update a model registration."""
    conn.execute(
        """INSERT INTO models
           (model_id, model_hf_id, model_family, is_multilingual, dtype,
            chat_template, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(model_id) DO UPDATE SET
               model_hf_id=excluded.model_hf_id,
               model_family=excluded.model_family,
               is_multilingual=excluded.is_multilingual,
               dtype=excluded.dtype,
               chat_template=excluded.chat_template,
               notes=excluded.notes""",
        (model_id, model_hf_id, model_family, int(is_multilingual),
         dtype, int(chat_template), notes),
    )
    conn.commit()


def register_known_models(conn: sqlite3.Connection):
    """Register all known models."""
    for model_id, hf_id, family, multilingual in KNOWN_MODELS:
        is_chat = "it" in model_id.split("_")
        register_model(conn, model_id, hf_id, family, multilingual,
                        chat_template=is_chat)
    print(f"  Registered {len(KNOWN_MODELS)} models")


def populate_human_data(
    conn: sqlite3.Connection,
    human_parquet_path: str | Path,
):
    """Import human_distributions.parquet into human_distributions table."""
    import pandas as pd

    df = pd.read_parquet(human_parquet_path)
    inserted = 0
    skipped = 0

    for _, row in df.iterrows():
        try:
            conn.execute(
                """INSERT INTO human_distributions
                   (lang, question_id, response_value, prob_human, n_respondents, n_valid)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (row["lang"], row["question_id"], int(row["response_value"]),
                 float(row["prob_human"]),
                 int(row["n_respondents"]) if "n_respondents" in row else None,
                 int(row["n_valid"]) if "n_valid" in row else None),
            )
            inserted += 1
        except sqlite3.IntegrityError:
            skipped += 1

    conn.commit()
    print(f"  Human data: {inserted} inserted, {skipped} skipped")


def main():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Populate the survey database with questions, prompts, and models"
    )
    parser.add_argument(
        "--questions", type=Path,
        default=PROJECT_ROOT / "data" / "questions.json",
        help="Path to questions.json",
    )
    parser.add_argument(
        "--prompt-config", type=Path, default=None,
        help="Path to prompt_config.json (omit for legacy format)",
    )
    parser.add_argument(
        "--config-name", default="legacy",
        help="Config name label (default: legacy; use 'optimized_v1' with --prompt-config)",
    )
    parser.add_argument(
        "--human-data", type=Path, default=None,
        help="Path to human_distributions.parquet",
    )
    parser.add_argument(
        "--rephrasings", type=Path, default=None,
        help="Path to rephrasings.json",
    )
    parser.add_argument(
        "--db", type=Path,
        default=PROJECT_ROOT / "data" / "survey.db",
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--permutations", type=int, default=6,
        help="Number of permutations per question (default: 6)",
    )
    parser.add_argument(
        "--chat-template", action="store_true",
        help="Mark prompts as requiring chat template wrapping",
    )
    parser.add_argument(
        "--langs", nargs="+", default=None,
        help="Languages to generate (default: all 22)",
    )
    args = parser.parse_args()

    # Auto-detect config name
    config_name = args.config_name
    if args.prompt_config and config_name == "legacy":
        config_name = "optimized_v1"

    print(f"Initializing database: {args.db}")
    conn = init_db(args.db)

    print("Populating questions...")
    populate_questions(conn, args.questions)

    print(f"Generating prompts (config={config_name}, perms={args.permutations})...")
    populate_prompts(
        conn, args.questions,
        config_name=config_name,
        prompt_config_path=args.prompt_config,
        n_permutations=args.permutations,
        langs=args.langs,
        chat_template=args.chat_template,
    )

    if args.rephrasings and args.rephrasings.exists():
        print("Generating rephrase prompts...")
        populate_rephrase_prompts(conn, args.questions, args.rephrasings)

    print("Registering models...")
    register_known_models(conn)

    if args.human_data and args.human_data.exists():
        print("Importing human data...")
        populate_human_data(conn, args.human_data)

    # Print summary
    counts = {}
    for table in ["questions", "prompts", "models", "evaluations", "human_distributions"]:
        row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        counts[table] = row[0]
    conn.close()

    print(f"\nDatabase summary ({args.db}):")
    for table, count in counts.items():
        print(f"  {table}: {count}")


if __name__ == "__main__":
    main()
