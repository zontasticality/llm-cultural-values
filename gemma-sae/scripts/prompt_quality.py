"""Score per-prompt quality for trimmed prompts (variant_idx >= TRIMMED_VARIANT_MIN).

For each trimmed prompt, computes:
  - fraction of completions classified as "other" (primary noise signal)
  - fraction of very short completions (<20 chars)
  - per-model %other breakdown (spot model-specific noise)
  - 20 sampled completions stratified across models, with classification

Writes prompts.quality_score = 1 - frac_other and dumps a review CSV.

Usage:
    PYTHONPATH=. python scripts/prompt_quality.py \
        --db data/culture.db \
        --classifier gemma3_27b_it \
        --out data/prompt_quality_report.csv \
        [--samples-per-prompt 20] [--write]
"""
import argparse
import csv
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

from analysis.constants import TRIMMED_VARIANT_MIN


SHORT_LEN = 20


def fetch_prompt_stats(conn: sqlite3.Connection, classifier: str) -> list[dict]:
    """Aggregate per-prompt noise stats across all models."""
    rows = conn.execute(
        """
        SELECT p.prompt_id, p.template_id, p.lang, p.variant_idx, p.prompt_text,
               c.model_id,
               cl.content_category,
               LENGTH(c.completion_text) AS len
        FROM prompts p
        JOIN completions c ON c.prompt_id = p.prompt_id
        JOIN classifications cl ON cl.completion_id = c.completion_id
                               AND cl.classifier_model = ?
        WHERE p.variant_idx >= ?
          AND c.filter_status = 'ok'
        """,
        (classifier, TRIMMED_VARIANT_MIN),
    ).fetchall()

    # Aggregate in Python so we can produce per-model breakdown in one pass
    acc: dict[int, dict] = {}
    for (pid, tmpl, lang, vidx, text, model, cat, clen) in rows:
        p = acc.setdefault(pid, {
            "prompt_id": pid,
            "template_id": tmpl,
            "lang": lang,
            "variant_idx": vidx,
            "prompt_text": text,
            "total": 0,
            "n_other": 0,
            "n_short": 0,
            "per_model": defaultdict(lambda: [0, 0]),  # model -> [total, n_other]
        })
        p["total"] += 1
        if cat == "other":
            p["n_other"] += 1
        if clen is not None and clen < SHORT_LEN:
            p["n_short"] += 1
        m = p["per_model"][model]
        m[0] += 1
        if cat == "other":
            m[1] += 1

    for p in acc.values():
        p["frac_other"] = p["n_other"] / max(p["total"], 1)
        p["frac_short"] = p["n_short"] / max(p["total"], 1)
        p["quality_score"] = 1.0 - p["frac_other"]
        # collapse per_model defaultdict to plain dict
        p["per_model"] = {m: {"n": n, "n_other": no, "frac_other": no / max(n, 1)}
                          for m, (n, no) in p["per_model"].items()}

    return sorted(acc.values(), key=lambda r: r["frac_other"], reverse=True)


def sample_completions(conn: sqlite3.Connection, prompt_id: int, classifier: str,
                       k: int) -> list[dict]:
    """Sample k completions for a prompt, stratified across models where possible."""
    # Determine distinct models with completions for this prompt
    models = [r[0] for r in conn.execute(
        "SELECT DISTINCT model_id FROM completions "
        "WHERE prompt_id=? AND filter_status='ok'",
        (prompt_id,),
    )]
    if not models:
        return []

    per_model = max(1, k // len(models))
    samples = []
    for model in models:
        rows = conn.execute(
            """
            SELECT c.completion_text, c.model_id, cl.content_category,
                   cl.dim_indiv_collect, cl.dim_trad_secular, cl.dim_surv_selfexpr
            FROM completions c
            LEFT JOIN classifications cl ON cl.completion_id = c.completion_id
                                        AND cl.classifier_model = ?
            WHERE c.prompt_id=? AND c.filter_status='ok'
            ORDER BY RANDOM()
            LIMIT ?
            """,
            (classifier, prompt_id, per_model),
        ).fetchall()
        for (text, m, cat, ic, ts, ss) in rows:
            samples.append({
                "model": m,
                "category": cat or "<unclassified>",
                "ic": ic, "ts": ts, "ss": ss,
                "text": text,
            })
    return samples[:k]


def write_quality_to_db(conn: sqlite3.Connection, stats: list[dict]):
    """Write quality_score back to prompts table."""
    for p in stats:
        conn.execute(
            "UPDATE prompts SET quality_score=? WHERE prompt_id=?",
            (round(p["quality_score"], 4), p["prompt_id"]),
        )
    conn.commit()


def write_report(stats: list[dict], samples_by_pid: dict[int, list[dict]],
                 out_path: Path):
    """Emit a CSV with one row per prompt: stats + 20 sampled completions flattened."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        header = [
            "prompt_id", "lang", "template_id", "variant_idx", "prompt_text",
            "total", "quality_score", "frac_other", "frac_short",
            "per_model_frac_other", "samples_json",
        ]
        w.writerow(header)
        for p in stats:
            samples = samples_by_pid.get(p["prompt_id"], [])
            per_model = {m: round(v["frac_other"], 3) for m, v in p["per_model"].items()}
            w.writerow([
                p["prompt_id"], p["lang"], p["template_id"], p["variant_idx"],
                p["prompt_text"], p["total"],
                round(p["quality_score"], 4), round(p["frac_other"], 4),
                round(p["frac_short"], 4),
                json.dumps(per_model, ensure_ascii=False),
                json.dumps(samples, ensure_ascii=False),
            ])


def main():
    ap = argparse.ArgumentParser(description="Per-prompt quality scoring")
    ap.add_argument("--db", default="data/culture.db")
    ap.add_argument("--classifier", default="gemma3_27b_it")
    ap.add_argument("--out", default="data/prompt_quality_report.csv")
    ap.add_argument("--samples-per-prompt", type=int, default=20)
    ap.add_argument("--write", action="store_true",
                    help="Write quality_score back to prompts table")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)

    print("Aggregating stats...")
    stats = fetch_prompt_stats(conn, args.classifier)
    print(f"  {len(stats)} trimmed prompts")

    print(f"Sampling {args.samples_per_prompt} completions per prompt...")
    samples_by_pid = {}
    for p in stats:
        samples_by_pid[p["prompt_id"]] = sample_completions(
            conn, p["prompt_id"], args.classifier, args.samples_per_prompt,
        )

    write_report(stats, samples_by_pid, Path(args.out))
    print(f"Report: {args.out}")

    # Distribution summary
    buckets = [(0.9, "clean"), (0.75, "usable"), (0.5, "degraded"), (0.0, "unsalvageable")]
    counts = defaultdict(int)
    for p in stats:
        for thresh, label in buckets:
            if p["quality_score"] >= thresh:
                counts[label] += 1
                break
    print("\nQuality distribution:")
    for _, label in buckets:
        print(f"  {label:<15s}: {counts[label]:>3d}")

    print("\nWorst 15:")
    print(f"  {'pid':>4s} {'lang':<4s} {'template':<14s} {'score':>6s} {'text':s}")
    for p in stats[:15]:
        print(f"  {p['prompt_id']:>4d} {p['lang']:<4s} {p['template_id']:<14s} "
              f"{p['quality_score']:>6.2f} {p['prompt_text'][:60]}")

    if args.write:
        print("\nWriting quality_score to prompts table...")
        write_quality_to_db(conn, stats)
        print(f"  Updated {len(stats)} rows")

    conn.close()


if __name__ == "__main__":
    main()
