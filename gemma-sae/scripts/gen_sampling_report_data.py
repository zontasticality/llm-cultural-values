"""Generate data/sampling_report_data.json for the typst report."""

import json
import sqlite3
from analysis.constants import LANG_NAMES

DB = "data/culture.db"
OUT = "data/sampling_report_data.json"

def main():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # 1. Get all templates
    c.execute("SELECT template_id, cultural_target FROM templates ORDER BY template_id")
    templates = c.fetchall()

    result = {"templates": []}

    for tmpl in templates:
        tid = tmpl["template_id"]
        ct = tmpl["cultural_target"]
        print(f"Processing template: {tid}")

        # 2. Get one prompt per language (lowest variant_idx >= 100)
        c.execute("""
            SELECT p.prompt_id, p.lang, p.prompt_text, p.variant_idx
            FROM prompts p
            WHERE p.template_id = ? AND p.variant_idx >= 100
            AND p.variant_idx = (
                SELECT MIN(p2.variant_idx)
                FROM prompts p2
                WHERE p2.template_id = p.template_id
                  AND p2.lang = p.lang
                  AND p2.variant_idx >= 100
            )
            ORDER BY p.lang
        """, (tid,))
        prompts = c.fetchall()

        langs_list = []
        for pr in prompts:
            pid = pr["prompt_id"]
            lang = pr["lang"]
            lang_name = LANG_NAMES.get(lang, lang)

            # 3. Get 3 completions with v2 classifications, diverse models
            # Try to get from different models by ordering by model then random
            c.execute("""
                SELECT c.model_id, c.completion_text,
                       cl.content_category, cl.dim_indiv_collect, cl.dim_trad_secular, cl.dim_surv_selfexpr
                FROM completions c
                JOIN classifications cl ON cl.completion_id = c.completion_id
                WHERE c.prompt_id = ?
                  AND cl.classifier_model = 'gemma3_27b_it_v2'
                  AND c.filter_status = 'ok'
                  AND (c.detected_lang IS NULL OR c.detected_lang = ?)
                ORDER BY RANDOM()
                LIMIT 9
            """, (pid, lang))
            rows = c.fetchall()

            # Pick up to 3, preferring different models
            seen_models = set()
            completions = []
            # First pass: one per model
            for r in rows:
                if r["model_id"] not in seen_models:
                    seen_models.add(r["model_id"])
                    completions.append(r)
                    if len(completions) == 3:
                        break
            # Second pass: fill remaining
            if len(completions) < 3:
                for r in rows:
                    if r not in completions:
                        completions.append(r)
                        if len(completions) == 3:
                            break

            comp_list = []
            for comp in completions:
                text = comp["completion_text"] or ""
                text = text.replace("\n", "\\n")
                if len(text) > 300:
                    text = text[:297] + "..."
                comp_list.append({
                    "model_id": comp["model_id"],
                    "completion_text": text,
                    "content_category": comp["content_category"],
                    "dim_ic": comp["dim_indiv_collect"],
                    "dim_ts": comp["dim_trad_secular"],
                    "dim_ss": comp["dim_surv_selfexpr"],
                })

            langs_list.append({
                "lang": lang,
                "lang_name": lang_name,
                "prompt_text": pr["prompt_text"],
                "variant_idx": pr["variant_idx"],
                "completions": comp_list,
            })

        result["templates"].append({
            "template_id": tid,
            "cultural_target": ct,
            "languages": langs_list,
        })

    conn.close()

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Summary
    n_templates = len(result["templates"])
    n_langs = sum(len(t["languages"]) for t in result["templates"])
    n_comps = sum(len(l["completions"]) for t in result["templates"] for l in t["languages"])
    print(f"\nWrote {OUT}: {n_templates} templates, {n_langs} language entries, {n_comps} completions")

if __name__ == "__main__":
    main()
