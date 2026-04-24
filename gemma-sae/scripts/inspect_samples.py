"""Stratified random inspection of completions + v2 classifications.

Samples ~N completions covering as many (model, lang) pairs as possible so a
human can eyeball classifier agreement. Prints prompt, completion, and the
v2 classification (category + IC/TS/SS) for each.
"""
import argparse
import random
import sqlite3
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/culture.db")
    ap.add_argument("--classifier", default="gemma3_27b_it_v2")
    ap.add_argument("--n-per-pair", type=int, default=1,
                    help="Samples per (model, lang) pair")
    ap.add_argument("--filter-category", default=None,
                    help="Only sample completions classified as this category")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trimmed-only", action="store_true", default=True)
    args = ap.parse_args()

    random.seed(args.seed)
    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    cat_filter = ""
    params: list = [args.classifier]
    if args.filter_category:
        cat_filter = "AND cl.content_category = ?"
        params.append(args.filter_category)

    variant_filter = "AND p.variant_idx >= 100" if args.trimmed_only else ""

    sql = f"""
        SELECT c.model_id, p.lang, p.template_id, p.variant_idx, p.prompt_text,
               c.completion_text, c.filter_status,
               cl.content_category, cl.dim_indiv_collect,
               cl.dim_trad_secular, cl.dim_surv_selfexpr
        FROM completions c
        JOIN prompts p ON c.prompt_id = p.prompt_id
        JOIN classifications cl ON cl.completion_id = c.completion_id
        WHERE cl.classifier_model = ?
          {cat_filter}
          {variant_filter}
    """
    rows = conn.execute(sql, params).fetchall()

    by_pair: dict[tuple[str, str], list] = defaultdict(list)
    for r in rows:
        by_pair[(r["model_id"], r["lang"])].append(r)

    print(f"Total (model, lang) pairs: {len(by_pair)}")
    print(f"Sampling {args.n_per_pair} per pair → {len(by_pair) * args.n_per_pair} completions")
    print()

    pairs = sorted(by_pair.keys())
    for (model, lang) in pairs:
        pool = by_pair[(model, lang)]
        picks = random.sample(pool, min(args.n_per_pair, len(pool)))
        for r in picks:
            print("─" * 92)
            print(f"[{model:<18} | {lang}] tmpl={r['template_id']:<12} v{r['variant_idx']}  status={r['filter_status']}")
            print(f"  prompt:     {r['prompt_text']!r}")
            print(f"  completion: {r['completion_text']!r}")
            dims = f"IC={r['dim_indiv_collect']} TS={r['dim_trad_secular']} SS={r['dim_surv_selfexpr']}"
            print(f"  → category={r['content_category']:<22s}  {dims}")

    print("─" * 92)
    print(f"Total printed: {len(pairs) * args.n_per_pair}")


if __name__ == "__main__":
    main()
