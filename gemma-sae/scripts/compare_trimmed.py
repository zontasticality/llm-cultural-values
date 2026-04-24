"""Compare trimmed (no trailing space) vs original (trailing space) completions."""
import sqlite3
import sys

from analysis.constants import TRIMMED_VARIANT_MIN

db_path = sys.argv[1] if len(sys.argv) > 1 else "gemma-sae/data/culture.db"
db = sqlite3.connect(db_path)

print("=" * 80)
print(f"TRIMMED (variant_idx >= {TRIMMED_VARIANT_MIN}) vs TRAILING SPACE (variant_idx < {TRIMMED_VARIANT_MIN})")
print("=" * 80)

rows = db.execute(f"""
    SELECT c.completion_text, c.model_id, c.filter_status,
           p.lang, p.template_id, p.variant_idx, p.prompt_text
    FROM completions c
    JOIN prompts p ON c.prompt_id = p.prompt_id
    WHERE p.variant_idx >= {TRIMMED_VARIANT_MIN} AND p.lang = 'eng'
    ORDER BY RANDOM()
    LIMIT 15
""").fetchall()

for i, (text, model, status, lang, tmpl, vidx, prompt) in enumerate(rows):
    print(f"\n--- [{i+1}] {model} | {lang}:{tmpl} ---")
    print(f"  TRIMMED prompt: {prompt!r}")
    print(f"  [{status}] {text!r}")

    orig_vidx = vidx - TRIMMED_VARIANT_MIN
    match = db.execute("""
        SELECT c.completion_text, c.filter_status, p.prompt_text
        FROM completions c
        JOIN prompts p ON c.prompt_id = p.prompt_id
        WHERE p.template_id = ? AND p.lang = ? AND p.variant_idx = ?
          AND c.model_id = ?
        ORDER BY RANDOM() LIMIT 1
    """, (tmpl, lang, orig_vidx, model)).fetchone()
    if match:
        print(f"  SPACE prompt:   {match[2]!r}")
        print(f"  [{match[1]}] {match[0]!r}")

print("\n" + "=" * 80)
print("FILTER RATE COMPARISON")
print("=" * 80)

for label, cond in [(f"Trailing space (v<{TRIMMED_VARIANT_MIN})", f"p.variant_idx < {TRIMMED_VARIANT_MIN}"),
                     (f"Trimmed (v>={TRIMMED_VARIANT_MIN})", f"p.variant_idx >= {TRIMMED_VARIANT_MIN}")]:
    total = db.execute(
        f"SELECT COUNT(*) FROM completions c JOIN prompts p ON c.prompt_id=p.prompt_id WHERE {cond}"
    ).fetchone()[0]
    if total == 0:
        print(f"  {label}: no data")
        continue
    ok = db.execute(
        f"SELECT COUNT(*) FROM completions c JOIN prompts p ON c.prompt_id=p.prompt_id WHERE {cond} AND c.filter_status='ok'"
    ).fetchone()[0]
    degen = db.execute(
        f"SELECT COUNT(*) FROM completions c JOIN prompts p ON c.prompt_id=p.prompt_id WHERE {cond} AND c.filter_status='degenerate'"
    ).fetchone()[0]
    print(f"  {label}: {total} total, {ok} ok ({ok/total*100:.1f}%), {degen} degenerate ({degen/total*100:.1f}%)")

print("\nEnglish filter rates by model:")
for label, cond in [("Trailing space", f"p.variant_idx < {TRIMMED_VARIANT_MIN}"), ("Trimmed", f"p.variant_idx >= {TRIMMED_VARIANT_MIN}")]:
    print(f"  {label}:")
    for row in db.execute(f"""
        SELECT c.model_id, COUNT(*) as total,
               SUM(CASE WHEN c.filter_status='ok' THEN 1 ELSE 0 END) as ok
        FROM completions c JOIN prompts p ON c.prompt_id=p.prompt_id
        WHERE {cond} AND p.lang='eng'
        GROUP BY c.model_id
    """).fetchall():
        m, total, ok = row
        print(f"    {m}: {ok}/{total} ok ({ok/total*100:.1f}%)")

# Check if any trimmed completions have classifications yet
cl_count = db.execute(f"""
    SELECT COUNT(*) FROM classifications cl
    JOIN completions c ON cl.completion_id = c.completion_id
    JOIN prompts p ON c.prompt_id = p.prompt_id
    WHERE p.variant_idx >= {TRIMMED_VARIANT_MIN}
""").fetchone()[0]
print(f"\nTrimmed completions classified so far: {cl_count}")

db.close()
