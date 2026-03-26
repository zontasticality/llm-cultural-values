"""LLM classification pipeline -- API-based (OpenAI / Anthropic batch).

Usage:
    # Interactive (for pilot/debugging)
    PYTHONPATH=gemma-sae python -m classify.classify \
        --db data/culture.db --classifier gpt-4.1-mini [--limit 100] [--validate]

    # Batch mode (submit async batch job)
    PYTHONPATH=gemma-sae python -m classify.classify \
        --db data/culture.db --classifier gpt-4.1-mini --batch-mode

    # Check batch status / retrieve results
    PYTHONPATH=gemma-sae python -m classify.classify \
        --db data/culture.db --classifier gpt-4.1-mini --batch-retrieve BATCH_ID
"""

import argparse
import json
import sys
import time
from pathlib import Path

from classify.prompts import (
    CLASSIFICATION_SCHEMA,
    CLASSIFIER_SYSTEM,
    make_classifier_prompt,
    parse_classification,
)
from db.load import load_unclassified_completions
from db.schema import get_connection

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ── Classifier dispatch ──────────────────────────────────────────

SUPPORTED_CLASSIFIERS = {
    # Direct APIs
    "gpt-4.1-mini": {"provider": "openai", "model": "gpt-4.1-mini"},
    "gpt-4.1-nano": {"provider": "openai", "model": "gpt-4.1-nano"},
    "claude-haiku-4.5": {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
    # OpenRouter (OpenAI-compatible, uses OPENROUTER_API_KEY)
    "or-gpt-4.1-mini": {"provider": "openrouter", "model": "openai/gpt-4.1-mini"},
    "or-haiku-4.5": {"provider": "openrouter", "model": "anthropic/claude-haiku-4-5"},
    "or-gemma-27b-it": {"provider": "openrouter", "model": "google/gemma-3-27b-it:free"},
}


def _make_openai_client(provider: str):
    """Create an OpenAI-compatible client for direct or OpenRouter APIs."""
    import os
    from openai import OpenAI

    if provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")
        return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    else:
        return OpenAI()


def classify_openai_interactive(
    completions: list[dict],
    model: str,
    batch_size: int = 20,
    provider: str = "openai",
    concurrency: int = 20,
    conn=None,
    classifier_name: str | None = None,
) -> list[dict]:
    """Classify completions concurrently using OpenAI-compatible API.

    Writes to DB every batch_size completions if conn and classifier_name provided.
    """
    import asyncio
    from openai import AsyncOpenAI
    import os

    if provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")
        client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    else:
        client = AsyncOpenAI()

    use_strict = provider == "openai"
    all_results = []
    failed = 0

    async def classify_one(comp: dict, semaphore: asyncio.Semaphore) -> dict | None:
        user_msg = make_classifier_prompt(
            comp["completion_text"], comp["lang"], comp["template_id"],
        )
        kwargs = dict(
            model=model,
            messages=[
                {"role": "system", "content": CLASSIFIER_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
        )
        if use_strict:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "classification",
                    "strict": True,
                    "schema": CLASSIFICATION_SCHEMA,
                },
            }
        else:
            kwargs["response_format"] = {"type": "json_object"}

        async with semaphore:
            try:
                response = await client.chat.completions.create(**kwargs)
                raw = response.choices[0].message.content
                parsed = parse_classification(raw)
                if parsed:
                    return {"completion_id": comp["completion_id"], "raw_response": raw, **parsed}
                else:
                    return None
            except Exception as e:
                print(f"  ERROR: {e} for completion {comp['completion_id']}")
                return None

    async def run_batches():
        nonlocal failed
        semaphore = asyncio.Semaphore(concurrency)
        t0 = time.time()

        for batch_start in range(0, len(completions), batch_size):
            batch = completions[batch_start:batch_start + batch_size]
            tasks = [classify_one(comp, semaphore) for comp in batch]
            results = await asyncio.gather(*tasks)

            batch_results = [r for r in results if r is not None]
            failed += sum(1 for r in results if r is None)
            all_results.extend(batch_results)

            # Write to DB incrementally
            if conn and classifier_name and batch_results:
                insert_classifications(conn, classifier_name, batch_results)

            elapsed = time.time() - t0
            done = batch_start + len(batch)
            rate = done / elapsed if elapsed > 0 else 0
            eta = (len(completions) - done) / rate if rate > 0 else 0
            print(f"  {done}/{len(completions)} classified "
                  f"({len(all_results)} ok, {failed} failed, "
                  f"{rate:.1f}/s, ETA {eta:.0f}s)")

    asyncio.run(run_batches())
    return all_results


def classify_anthropic_interactive(
    completions: list[dict],
    model: str,
) -> list[dict]:
    """Classify completions one-by-one using Anthropic API."""
    import anthropic
    client = anthropic.Anthropic()
    results = []

    for i, comp in enumerate(completions):
        user_msg = make_classifier_prompt(
            comp["completion_text"], comp["lang"], comp["template_id"],
        )
        try:
            response = client.messages.create(
                model=model,
                max_tokens=200,
                system=CLASSIFIER_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
                temperature=0,
            )
            raw = response.content[0].text
            parsed = parse_classification(raw)
            if parsed:
                results.append({
                    "completion_id": comp["completion_id"],
                    "raw_response": raw,
                    **parsed,
                })
            else:
                print(f"  WARN: failed to parse response for completion {comp['completion_id']}")

        except Exception as e:
            print(f"  ERROR: {e} for completion {comp['completion_id']}")

        if (i + 1) % 50 == 0:
            print(f"  Classified {i+1}/{len(completions)}")

    return results


# ── Batch mode (OpenAI) ──────────────────────────────────────────

def submit_openai_batch(completions: list[dict], model: str, db_path: str) -> str:
    """Create and submit an OpenAI Batch API request. Returns batch ID."""
    from openai import OpenAI
    client = OpenAI()

    # Build JSONL batch file
    batch_file = DATA_DIR / "batch_requests.jsonl"
    with open(batch_file, "w") as f:
        for comp in completions:
            user_msg = make_classifier_prompt(
                comp["completion_text"], comp["lang"], comp["template_id"],
            )
            request = {
                "custom_id": str(comp["completion_id"]),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": CLASSIFIER_SYSTEM},
                        {"role": "user", "content": user_msg},
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "classification",
                            "strict": True,
                            "schema": CLASSIFICATION_SCHEMA,
                        },
                    },
                    "temperature": 0,
                },
            }
            f.write(json.dumps(request) + "\n")

    # Upload file
    with open(batch_file, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")

    # Create batch
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"db_path": db_path, "n_completions": str(len(completions))},
    )

    print(f"Batch submitted: {batch.id}")
    print(f"  Status: {batch.status}")
    print(f"  Completions: {len(completions)}")
    print(f"  File: {file_obj.id}")
    return batch.id


def retrieve_openai_batch(batch_id: str) -> list[dict] | None:
    """Check batch status and retrieve results if complete."""
    from openai import OpenAI
    client = OpenAI()

    batch = client.batches.retrieve(batch_id)
    print(f"Batch {batch_id}: {batch.status}")
    print(f"  Completed: {batch.request_counts.completed}/{batch.request_counts.total}")
    print(f"  Failed: {batch.request_counts.failed}")

    if batch.status != "completed":
        return None

    # Download results
    content = client.files.content(batch.output_file_id)
    results = []
    for line in content.text.strip().split("\n"):
        item = json.loads(line)
        completion_id = int(item["custom_id"])
        if item["response"]["status_code"] == 200:
            raw = item["response"]["body"]["choices"][0]["message"]["content"]
            parsed = parse_classification(raw)
            if parsed:
                results.append({
                    "completion_id": completion_id,
                    "raw_response": raw,
                    **parsed,
                })

    print(f"  Parsed: {len(results)} results")
    return results


# ── Batch mode (Anthropic) ───────────────────────────────────────

def submit_anthropic_batch(completions: list[dict], model: str, db_path: str) -> str:
    """Create and submit an Anthropic Message Batches request."""
    import anthropic
    client = anthropic.Anthropic()

    requests = []
    for comp in completions:
        user_msg = make_classifier_prompt(
            comp["completion_text"], comp["lang"], comp["template_id"],
        )
        requests.append({
            "custom_id": str(comp["completion_id"]),
            "params": {
                "model": model,
                "max_tokens": 200,
                "system": CLASSIFIER_SYSTEM,
                "messages": [{"role": "user", "content": user_msg}],
                "temperature": 0,
            },
        })

    batch = client.messages.batches.create(requests=requests)
    print(f"Batch submitted: {batch.id}")
    print(f"  Completions: {len(completions)}")
    return batch.id


def retrieve_anthropic_batch(batch_id: str) -> list[dict] | None:
    """Check batch status and retrieve results if complete."""
    import anthropic
    client = anthropic.Anthropic()

    batch = client.messages.batches.retrieve(batch_id)
    print(f"Batch {batch_id}: {batch.processing_status}")

    if batch.processing_status != "ended":
        return None

    results = []
    for result in client.messages.batches.results(batch_id):
        completion_id = int(result.custom_id)
        if result.result.type == "succeeded":
            raw = result.result.message.content[0].text
            parsed = parse_classification(raw)
            if parsed:
                results.append({
                    "completion_id": completion_id,
                    "raw_response": raw,
                    **parsed,
                })

    print(f"  Parsed: {len(results)} results")
    return results


# ── DB insertion ─────────────────────────────────────────────────

def insert_classifications(conn, classifier_model: str, results: list[dict]):
    """Batch insert classification results."""
    conn.executemany(
        "INSERT OR IGNORE INTO classifications "
        "(completion_id, classifier_model, content_category, "
        " dim_indiv_collect, dim_trad_secular, dim_surv_selfexpr, raw_response) "
        "VALUES (:completion_id, :classifier_model, :content_category, "
        " :dim_indiv_collect, :dim_trad_secular, :dim_surv_selfexpr, :raw_response)",
        [{"classifier_model": classifier_model, **r} for r in results],
    )
    conn.commit()


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM classification pipeline")
    parser.add_argument("--db", required=True)
    parser.add_argument("--classifier", required=True, choices=list(SUPPORTED_CLASSIFIERS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--validate", action="store_true",
                        help="Classify 10 completions, print results, don't save")
    parser.add_argument("--batch-mode", action="store_true",
                        help="Submit as async batch job")
    parser.add_argument("--batch-retrieve", type=str, default=None,
                        help="Retrieve results from a batch job ID")
    args = parser.parse_args()

    config = SUPPORTED_CLASSIFIERS[args.classifier]
    conn = get_connection(args.db)

    # Retrieve mode
    if args.batch_retrieve:
        if config["provider"] == "openai":
            results = retrieve_openai_batch(args.batch_retrieve)
        else:
            results = retrieve_anthropic_batch(args.batch_retrieve)

        if results:
            insert_classifications(conn, args.classifier, results)
            print(f"Inserted {len(results)} classifications into DB.")
        else:
            print("Batch not yet complete. Try again later.")
        conn.close()
        return

    # Load unclassified completions
    completions = load_unclassified_completions(
        conn, args.classifier, limit=args.limit,
    )

    if not completions:
        print("No unclassified completions. Nothing to do.")
        conn.close()
        return

    print(f"Found {len(completions)} unclassified completions for {args.classifier}")

    if args.validate:
        completions = completions[:10]
        print(f"Validate mode: classifying {len(completions)} samples")

    # Batch mode: submit and exit (not supported for OpenRouter)
    if args.batch_mode:
        if config["provider"] == "openrouter":
            print("ERROR: Batch mode not supported for OpenRouter. Use interactive mode.")
            conn.close()
            return
        if config["provider"] == "openai":
            batch_id = submit_openai_batch(completions, config["model"], args.db)
        else:
            batch_id = submit_anthropic_batch(completions, config["model"], args.db)
        print(f"\nTo retrieve results:\n  python -m classify.classify "
              f"--db {args.db} --classifier {args.classifier} "
              f"--batch-retrieve {batch_id}")
        conn.close()
        return

    # Interactive mode
    t0 = time.time()
    # Pass conn for incremental DB writes (not in validate mode)
    db_conn = None if args.validate else conn
    if config["provider"] in ("openai", "openrouter"):
        results = classify_openai_interactive(
            completions, config["model"], provider=config["provider"],
            conn=db_conn, classifier_name=args.classifier)
    else:
        results = classify_anthropic_interactive(completions, config["model"])
        if not args.validate:
            insert_classifications(conn, args.classifier, results)

    elapsed = time.time() - t0

    if args.validate:
        print(f"\n{'='*60}")
        print(f"Validation results ({elapsed:.1f}s):")
        for r in results:
            comp = next(c for c in completions if c["completion_id"] == r["completion_id"])
            print(f"  [{comp['lang']}:{comp['template_id']}] {comp['completion_text']!r}")
            print(f"    → {r['content_category']} | IC={r['dim_indiv_collect']} "
                  f"TS={r['dim_trad_secular']} SS={r['dim_surv_selfexpr']}")
    else:
        print(f"\nDone. Classified {len(results)}/{len(completions)} in {elapsed:.1f}s")
        print(f"  Success rate: {len(results)/len(completions)*100:.1f}%")

    conn.close()


if __name__ == "__main__":
    main()
