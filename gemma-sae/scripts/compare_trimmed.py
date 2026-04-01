"""Compare trimmed (no trailing space) vs original (trailing space) completions."""
import sqlite3
import sys

db_path = sys.argv[1] if len(sys.argv) > 1 else "gemma-sae/data/culture.db"
db = sqlite3.connect(db_path)

print("=" * 80)
print("TRIMMED (variant_idx >= 100) vs TRAILING SPACE (variant_idx < 100)")
print("=" * 80)

rows = db.execute("""
    SELECT c.completion_text, c.model_id, c.filter_status,
           p.lang, p.template_id, p.variant_idx, p.prompt_text
    FROM completions c
    JOIN prompts p ON c.prompt_id = p.prompt_id
    WHERE p.variant_idx >= 100 AND p.lang = 'eng'
    ORDER BY RANDOM()
    LIMIT 15
""").fetchall()

for i, (text, model, status, lang, tmpl, vidx, prompt) in enumerate(rows):
    print(f"\n--- [{i+1}] {model} | {lang}:{tmpl} ---")
    print(f"  TRIMMED prompt: {prompt!r}")
    print(f"  [{status}] {text!r}")

    orig_vidx = vidx - 100
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

for label, cond in [("Trailing space (v<100)", "p.variant_idx < 100"),
                     ("Trimmed (v>=100)", "p.variant_idx >= 100")]:
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
for label, cond in [("Trailing space", "p.variant_idx < 100"), ("Trimmed", "p.variant_idx >= 100")]:
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
cl_count = db.execute("""
    SELECT COUNT(*) FROM classifications cl
    JOIN completions c ON cl.completion_id = c.completion_id
    JOIN prompts p ON c.prompt_id = p.prompt_id
    WHERE p.variant_idx >= 100
""").fetchone()[0]
print(f"\nTrimmed completions classified so far: {cl_count}")

db.close()
