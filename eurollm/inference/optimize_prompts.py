"""Grid search over prompt format dimensions to optimize LLM-to-human agreement.

Tests combinations of answer cue, option format, scale hint, option embedding,
and permutation count on a subset of questions, comparing to human survey data.

Grid dimensions (3 x 3 x 2 x 2 x 2 = 72 variants per question):
    A. cue_style:   "answer" | "none" | "lang"
    B. opt_format:  "numbered_dot" | "numbered_paren" | "bullet"
    C. scale_hint:  False | True
    D. embed_style: "separate" | "inline"
    E. n_perms:     2 | 6

Usage:
    python eurollm/inference/optimize_prompts.py \
        --model_id HPLT/hplt2c_eng_checkpoints \
        --lang eng \
        --questions eurollm/data/questions.json \
        --human eurollm/human_data/data/human_distributions.parquet \
        --output eurollm/results/optimization/hplt2c_eng_grid.parquet
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import jensenshannon

from prompting.prompt_templates import format_prompt_custom
from inference.extract_logprobs import (
    TokenMap,
    extract_logprobs_for_question,
    renormalize,
    remap_reversed,
    generate_permutations,
    load_model,
)


# ── Grid Configuration ────────────────────────────────────────────────────────

GRID = {
    "cue_style": ["answer", "none", "lang"],
    "opt_format": ["numbered_dot", "numbered_paren", "bullet"],
    "scale_hint": [False, True],
    "embed_style": ["separate", "inline"],
    "n_perms": [2, 6],
}

N_PER_TYPE = 6  # questions per response type


# ── Question Selection ────────────────────────────────────────────────────────

def select_questions(
    questions: list[dict],
    lang: str,
    human_df: pd.DataFrame | None = None,
    n_per_type: int = N_PER_TYPE,
) -> list[dict]:
    """Select a balanced subset of questions for optimization.

    Picks n_per_type questions per response type, preferring those with human
    data available. If human data is provided, selects a mix of best and worst
    aligned questions.
    """
    # Filter to questions available in this language
    available = [q for q in questions if lang in q["translations"]]

    # Group by response type
    by_type: dict[str, list[dict]] = {}
    for q in available:
        rt = q["response_type"]
        if rt not in by_type:
            by_type[rt] = []
        by_type[rt].append(q)

    selected = []
    for rt, qs in sorted(by_type.items()):
        if human_df is not None and not human_df.empty:
            human_qids = set(human_df[human_df["lang"] == lang]["question_id"].unique())
            with_human = [q for q in qs if q["canonical_id"] in human_qids]
            without = [q for q in qs if q["canonical_id"] not in human_qids]
            pool = with_human + without
        else:
            pool = qs

        selected.extend(pool[:n_per_type])

    return selected


def build_human_dists(human_df: pd.DataFrame, lang: str) -> dict:
    """Build {question_id: {response_value: prob}} from human data for one language."""
    dists = {}
    sub = human_df[human_df["lang"] == lang]
    for qid, grp in sub.groupby("question_id"):
        grp_sorted = grp.sort_values("response_value")
        values = grp_sorted["response_value"].values
        probs = grp_sorted["prob_human"].values
        total = probs.sum()
        if total > 0:
            probs = probs / total
        dists[qid] = dict(zip(values, probs))
    return dists


# ── Grid Search ───────────────────────────────────────────────────────────────

def compute_jsd(llm_dist: dict, human_dist: dict) -> float:
    """Compute JSD between two distributions."""
    all_values = sorted(set(llm_dist.keys()) | set(human_dist.keys()))
    a = np.array([llm_dist.get(v, 0.0) for v in all_values])
    b = np.array([human_dist.get(v, 0.0) for v in all_values])
    if a.sum() > 0:
        a = a / a.sum()
    if b.sum() > 0:
        b = b / b.sum()
    return float(jensenshannon(a, b))


def run_grid_search(
    model,
    tokenizer,
    token_map: TokenMap,
    questions: list[dict],
    lang: str,
    human_dists: dict,
) -> pd.DataFrame:
    """Run full grid search over format dimensions."""
    configs = list(itertools.product(
        GRID["cue_style"],
        GRID["opt_format"],
        GRID["scale_hint"],
        GRID["embed_style"],
        GRID["n_perms"],
    ))

    total_evals = len(configs) * len(questions)
    print(f"Grid: {len(configs)} configs x {len(questions)} questions = {total_evals} evaluations")

    all_rows = []
    eval_count = 0
    t_start = time.time()

    for qi, q in enumerate(questions):
        qid = q["canonical_id"]
        rtype = q["response_type"]
        n_options = (10 if rtype == "likert10"
                     else len(q["translations"][lang]["options"]))

        for cue, opt_fmt, hint, embed, n_perms in configs:
            perms = generate_permutations(n_options, n_perms, qid)

            perm_probs = []
            perm_p_valids = []

            for perm in perms:
                formatted = format_prompt_custom(
                    q, lang, perm,
                    cue_style=cue,
                    opt_format=opt_fmt,
                    scale_hint=hint,
                    embed_style=embed,
                )
                logprobs, p_valid = extract_logprobs_for_question(
                    model, tokenizer, token_map, formatted
                )
                probs_raw = renormalize(logprobs)
                probs = remap_reversed(probs_raw, formatted["value_map"])
                perm_probs.append(probs)
                perm_p_valids.append(p_valid)

            # Average across permutations
            all_values = sorted(
                set(v for p in perm_probs for v in p.keys()),
                key=lambda x: int(x),
            )

            avg_probs = {}
            for v in all_values:
                avg_probs[v] = float(np.mean([p.get(v, 0.0) for p in perm_probs]))

            # Position bias
            pos_bias = 0.0
            for v in all_values:
                vals = [p.get(v, 0.0) for p in perm_probs]
                pos_bias = max(pos_bias, max(vals) - min(vals))

            # JSD to human
            jsd_val = np.nan
            if qid in human_dists:
                # Map semantic values to ints for comparison
                llm_int = {int(k): v for k, v in avg_probs.items()}
                jsd_val = compute_jsd(llm_int, human_dists[qid])

            p_valid_mean = float(np.mean(perm_p_valids))

            for v in all_values:
                all_rows.append({
                    "question_id": qid,
                    "response_type": rtype,
                    "cue_style": cue,
                    "opt_format": opt_fmt,
                    "scale_hint": hint,
                    "embed_style": embed,
                    "n_perms": n_perms,
                    "response_value": int(v),
                    "prob_averaged": avg_probs[v],
                    "p_valid": p_valid_mean,
                    "position_bias": pos_bias,
                    "jsd_to_human": jsd_val,
                })

            eval_count += 1

        if qi == 0:
            t_first = time.time() - t_start
            print(f"  First question ({len(configs)} configs) took {t_first:.1f}s")
        if (qi + 1) % 5 == 0:
            elapsed = time.time() - t_start
            rate = (qi + 1) / elapsed
            eta = (len(questions) - qi - 1) / rate
            pct = eval_count / total_evals * 100
            print(f"  Processed {qi + 1}/{len(questions)} questions ({pct:.0f}%) "
                  f"[{elapsed:.0f}s elapsed, ETA {eta:.0f}s]")

    return pd.DataFrame(all_rows)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Grid search over prompt format dimensions"
    )
    parser.add_argument(
        "--model_id", required=True,
        help="HuggingFace model ID"
    )
    parser.add_argument(
        "--lang", required=True,
        help="Language code (e.g. eng, deu)"
    )
    parser.add_argument(
        "--questions", required=True,
        help="Path to questions.json"
    )
    parser.add_argument(
        "--human", default=None,
        help="Path to human_distributions.parquet (optional)"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output parquet file path"
    )
    parser.add_argument(
        "--dtype", default="bf16", choices=["bf16", "int8", "fp8"],
        help="Model precision (default: bf16)"
    )
    parser.add_argument(
        "--n-per-type", type=int, default=N_PER_TYPE,
        help=f"Questions per response type (default: {N_PER_TYPE})"
    )
    args = parser.parse_args()

    # Load questions
    with open(args.questions) as f:
        data = json.load(f)
    questions = data["questions"]

    # Load human data
    human_df = None
    if args.human and Path(args.human).exists():
        human_df = pd.read_parquet(args.human)
        print(f"Loaded human data: {len(human_df)} rows")

    # Select questions
    selected = select_questions(questions, args.lang, human_df, args.n_per_type)
    print(f"Selected {len(selected)} questions for optimization")
    by_type = {}
    for q in selected:
        rt = q["response_type"]
        by_type[rt] = by_type.get(rt, 0) + 1
    print(f"  By type: {by_type}")

    # Build human distributions
    human_dists = {}
    if human_df is not None:
        human_dists = build_human_dists(human_df, args.lang)
        print(f"  Human distributions available: {len(human_dists)} questions")

    # Load model
    model, tokenizer = load_model(args.model_id, args.dtype)
    t0 = time.time()
    token_map = TokenMap.from_tokenizer(tokenizer)
    print(f"Token map built in {time.time() - t0:.1f}s")

    # Run grid search
    print("\nStarting grid search...")
    df = run_grid_search(model, tokenizer, token_map, selected, args.lang, human_dists)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"\nSaved {len(df)} rows to {output_path}")

    # Print quick summary
    # Aggregate to one row per (question, config) — take first row's jsd
    config_cols = ["cue_style", "opt_format", "scale_hint", "embed_style", "n_perms"]
    summary = df.groupby(config_cols + ["question_id"]).agg(
        jsd=("jsd_to_human", "first"),
        p_valid=("p_valid", "first"),
    ).reset_index()

    # Best config by mean JSD
    config_jsd = summary.groupby(config_cols).agg(
        mean_jsd=("jsd", "mean"),
        mean_pvalid=("p_valid", "mean"),
    ).reset_index().sort_values("mean_jsd")

    print("\nTop 5 configurations by mean JSD:")
    for _, row in config_jsd.head(5).iterrows():
        print(f"  JSD={row['mean_jsd']:.4f} P_valid={row['mean_pvalid']:.4f} | "
              f"cue={row['cue_style']} opt={row['opt_format']} "
              f"hint={row['scale_hint']} embed={row['embed_style']} K={row['n_perms']}")

    print("\nBottom 5 configurations:")
    for _, row in config_jsd.tail(5).iterrows():
        print(f"  JSD={row['mean_jsd']:.4f} P_valid={row['mean_pvalid']:.4f} | "
              f"cue={row['cue_style']} opt={row['opt_format']} "
              f"hint={row['scale_hint']} embed={row['embed_style']} K={row['n_perms']}")


if __name__ == "__main__":
    main()
