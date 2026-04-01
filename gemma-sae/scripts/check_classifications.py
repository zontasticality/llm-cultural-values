"""Sample random classifications and print them for review."""
import sqlite3
import json
import sys

db_path = sys.argv[1] if len(sys.argv) > 1 else "gemma-sae/data/culture.db"
db = sqlite3.connect(db_path)

rows = db.execute("""
    SELECT c.completion_text, c.model_id, p.lang, p.template_id, p.prompt_text,
           cl.content_category, cl.dim_indiv_collect, cl.dim_trad_secular,
           cl.dim_surv_selfexpr, cl.cat_probs, cl.dim_ic_probs
    FROM classifications cl
    JOIN completions c ON cl.completion_id = c.completion_id
    JOIN prompts p ON c.prompt_id = p.prompt_id
    WHERE c.filter_status = 'ok'
    ORDER BY RANDOM()
    LIMIT 20
""").fetchall()

for i, row in enumerate(rows):
    (text, model, lang, template, prompt,
     cat, ic, ts, ss, cat_probs_raw, ic_probs_raw) = row

    print(f"--- [{i+1}] {model} | {lang}:{template} ---")
    print(f"  Prompt: {prompt!r}")
    print(f"  Completion: {text!r}")
    print(f"  Cat: {cat}  IC={ic} TS={ts} SS={ss}")

    if cat_probs_raw:
        cp = json.loads(cat_probs_raw)
        top3 = sorted(cp.items(), key=lambda x: -x[1])[:3]
        print("  Cat probs: " + ", ".join(f"{c}={p:.2f}" for c, p in top3))

    if ic_probs_raw:
        ic_p = json.loads(ic_probs_raw)
        ev = sum((j + 1) * p for j, p in enumerate(ic_p))
        print(f"  IC probs: {[round(p, 2) for p in ic_p]} E={ev:.2f}")

    print()

# Summary
total_cl = db.execute("SELECT COUNT(*) FROM classifications").fetchone()[0]
total_comp = db.execute("SELECT COUNT(*) FROM completions").fetchone()[0]
unclassified = db.execute("""
    SELECT COUNT(*) FROM completions c
    LEFT JOIN classifications cl ON c.completion_id = cl.completion_id
    WHERE c.filter_status = 'ok' AND cl.completion_id IS NULL
""").fetchone()[0]
print(f"=== Summary: {total_cl} classified / {total_comp} completions, {unclassified} unclassified ===")

db.close()
