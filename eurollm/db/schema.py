"""Database schema and connection helpers for the survey results database.

Provides DDL for the centralized SQLite database storing questions, prompts,
model registrations, evaluation results, and human survey distributions.
"""

import sqlite3
from pathlib import Path

DDL = """
CREATE TABLE IF NOT EXISTS questions (
    question_id   TEXT PRIMARY KEY,
    response_type TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS prompts (
    prompt_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id     TEXT NOT NULL REFERENCES questions(question_id),
    lang            TEXT NOT NULL,
    config          TEXT,
    permutation_idx INTEGER NOT NULL,
    prompt_text     TEXT NOT NULL,
    question_text   TEXT NOT NULL,
    valid_values    TEXT NOT NULL,
    value_map       TEXT NOT NULL,
    is_likert10     INTEGER NOT NULL DEFAULT 0,
    chat_template   INTEGER NOT NULL DEFAULT 0,
    variant_id      TEXT,
    variant_type    TEXT,
    UNIQUE(question_id, lang, permutation_idx, chat_template, config, variant_id)
);

CREATE TABLE IF NOT EXISTS models (
    model_id        TEXT PRIMARY KEY,
    model_hf_id     TEXT NOT NULL,
    model_family    TEXT NOT NULL,
    is_multilingual INTEGER NOT NULL,
    dtype           TEXT DEFAULT 'bf16',
    chat_template   INTEGER NOT NULL DEFAULT 0,
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS evaluations (
    prompt_id       INTEGER NOT NULL REFERENCES prompts(prompt_id),
    model_id        TEXT NOT NULL REFERENCES models(model_id),
    response_value  INTEGER NOT NULL,
    prob            REAL NOT NULL,
    p_valid         REAL NOT NULL,
    UNIQUE(prompt_id, model_id, response_value)
);

CREATE TABLE IF NOT EXISTS human_distributions (
    lang            TEXT NOT NULL,
    question_id     TEXT NOT NULL REFERENCES questions(question_id),
    response_value  INTEGER NOT NULL,
    prob_human      REAL NOT NULL,
    n_respondents   INTEGER,
    n_valid         INTEGER,
    PRIMARY KEY (lang, question_id, response_value)
);

CREATE INDEX IF NOT EXISTS idx_eval_model_prompt ON evaluations(model_id, prompt_id);
CREATE INDEX IF NOT EXISTS idx_prompt_question_lang ON prompts(question_id, lang);
CREATE INDEX IF NOT EXISTS idx_prompt_lang ON prompts(lang);
"""


def get_connection(db_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite connection with WAL mode, foreign keys, and busy timeout."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=120000")
    return conn


def create_tables(conn: sqlite3.Connection):
    """Execute DDL to create all tables and indexes."""
    conn.executescript(DDL)
    conn.commit()


def init_db(db_path: str | Path) -> sqlite3.Connection:
    """Create database file and tables (idempotent)."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = get_connection(db_path)
    create_tables(conn)
    return conn
