"""Database schema and connection helpers for the cultural completions database."""

import sqlite3
from pathlib import Path

DDL = """
CREATE TABLE IF NOT EXISTS templates (
    template_id     TEXT PRIMARY KEY,
    template_name   TEXT NOT NULL,
    cultural_target TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS prompts (
    prompt_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    template_id     TEXT NOT NULL REFERENCES templates(template_id),
    lang            TEXT NOT NULL,
    variant_idx     INTEGER NOT NULL DEFAULT 0,
    prompt_text     TEXT NOT NULL,
    UNIQUE(template_id, lang, variant_idx)
);

CREATE TABLE IF NOT EXISTS models (
    model_id        TEXT PRIMARY KEY,
    model_hf_id     TEXT NOT NULL,
    model_family    TEXT NOT NULL,
    is_multilingual INTEGER NOT NULL,
    dtype           TEXT DEFAULT 'bf16',
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS completions (
    completion_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt_id       INTEGER NOT NULL REFERENCES prompts(prompt_id),
    model_id        TEXT NOT NULL REFERENCES models(model_id),
    sample_idx      INTEGER NOT NULL,
    completion_raw  TEXT NOT NULL,
    completion_text TEXT,
    n_tokens_raw    INTEGER NOT NULL,
    n_tokens        INTEGER,
    filter_status   TEXT NOT NULL DEFAULT 'ok',
    temperature     REAL NOT NULL DEFAULT 1.0,
    top_p           REAL NOT NULL DEFAULT 0.95,
    seed            INTEGER,
    steering_config TEXT NOT NULL DEFAULT 'none',
    UNIQUE(prompt_id, model_id, sample_idx, steering_config)
);

CREATE TABLE IF NOT EXISTS classifications (
    completion_id       INTEGER NOT NULL REFERENCES completions(completion_id),
    classifier_model    TEXT NOT NULL,
    content_category    TEXT NOT NULL,
    dim_indiv_collect   INTEGER NOT NULL,
    dim_trad_secular    INTEGER NOT NULL,
    dim_surv_selfexpr   INTEGER NOT NULL,
    raw_response        TEXT,
    dim_ic_probs        TEXT,           -- JSON [p1,p2,p3,p4,p5] from logprobs (NULL for API classifiers)
    dim_ts_probs        TEXT,           -- JSON [p1,p2,p3,p4,p5] from logprobs
    dim_ss_probs        TEXT,           -- JSON [p1,p2,p3,p4,p5] from logprobs
    cat_probs           TEXT,           -- JSON {"category": prob, ...} from logprobs
    UNIQUE(completion_id, classifier_model)
);

CREATE TABLE IF NOT EXISTS prompt_metrics (
    prompt_id       INTEGER NOT NULL REFERENCES prompts(prompt_id),
    model_id        TEXT NOT NULL REFERENCES models(model_id),
    prompt_ppl      REAL NOT NULL,
    prompt_logprob  REAL NOT NULL,
    prompt_n_tokens INTEGER NOT NULL,
    next_token_entropy REAL,
    UNIQUE(prompt_id, model_id)
);

CREATE INDEX IF NOT EXISTS idx_comp_model_prompt ON completions(model_id, prompt_id);
CREATE INDEX IF NOT EXISTS idx_comp_filter ON completions(filter_status);
CREATE INDEX IF NOT EXISTS idx_prompt_template_lang ON prompts(template_id, lang);
"""


def get_connection(db_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite connection with foreign keys and busy timeout.

    WAL mode is set once during init_db(), not on every connection.
    Re-issuing PRAGMA journal_mode=WAL from multiple processes simultaneously
    on a network filesystem can corrupt the DB.
    """
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=120000")
    return conn


def create_tables(conn: sqlite3.Connection):
    """Execute DDL to create all tables and indexes."""
    conn.executescript(DDL)
    conn.commit()


def migrate_classifications_probs(conn: sqlite3.Connection):
    """Add logprob probability columns if they don't exist yet (idempotent)."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(classifications)")}
    for col in ["dim_ic_probs", "dim_ts_probs", "dim_ss_probs", "cat_probs"]:
        if col not in existing:
            conn.execute(f"ALTER TABLE classifications ADD COLUMN {col} TEXT")
    conn.commit()


def init_db(db_path: str | Path) -> sqlite3.Connection:
    """Create database file and tables (idempotent).

    Uses DELETE journal mode (not WAL) for NFS compatibility.
    Each SLURM job works on a local copy; results are merged afterward.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = get_connection(db_path)
    conn.execute("PRAGMA journal_mode=DELETE")
    create_tables(conn)
    migrate_classifications_probs(conn)
    return conn
