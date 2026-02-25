"""Process EVS 2017 survey data into country-level response distributions.

Reads the EVS Integrated Dataset (SPSS format or .sav.zip), filters by country,
drops missing/DK/NA values, and computes weighted frequency distributions
for each of our 187 canonical survey questions across 22 countries.

Requirements:
    pip install pyreadstat pandas numpy pyarrow

Usage:
    python eurollm/human_data/process_evs.py \
        --input eurollm/human_data/data/ZA7500_v5-0-0.sav.zip \
        --questions eurollm/data/questions.json \
        --output eurollm/human_data/data/human_distributions.parquet
"""

import argparse
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat


# Country code (ISO numeric) → our language code
COUNTRY_TO_LANG = {
    100: "bul",   # Bulgaria
    203: "ces",   # Czechia
    208: "dan",   # Denmark
    276: "deu",   # Germany
    300: "ell",   # Greece
    826: "eng",   # Great Britain
    233: "est",   # Estonia
    246: "fin",   # Finland
    250: "fra",   # France
    191: "hrv",   # Croatia
    348: "hun",   # Hungary
    380: "ita",   # Italy
    440: "lit",   # Lithuania
    428: "lvs",   # Latvia
    528: "nld",   # Netherlands
    616: "pol",   # Poland
    620: "por",   # Portugal
    642: "ron",   # Romania
    703: "slk",   # Slovakia
    705: "slv",   # Slovenia
    724: "spa",   # Spain
    752: "swe",   # Sweden
}


def find_country_variable(columns: list[str]) -> str:
    """Find the country variable in the dataset."""
    for candidate in ["country", "c_aession", "S003"]:
        if candidate in columns:
            return candidate
    # Fall back to any column containing "country" (case-insensitive)
    for col in columns:
        if "country" in col.lower():
            return col
    raise ValueError(
        f"Could not find country variable. Available columns: {columns[:20]}..."
    )


def find_weight_variable(columns: list[str]) -> str | None:
    """Find the recommended weight variable. Returns None if no weights found."""
    for candidate in ["gweight", "dweight", "w_EVS5"]:
        if candidate in columns:
            return candidate
    return None


def get_valid_range(question: dict) -> set[int]:
    """Get the set of valid response values from a question's options.

    Valid responses are the numbered option values (typically 1-4, 1-5, or 1-10).
    Anything outside this range (negative codes, 77, 88, 99) is treated as missing.
    """
    values = set()
    # Use the canonical options (from the first available translation)
    for lang_data in question["translations"].values():
        for opt in lang_data["options"]:
            values.add(opt["value"])
        break  # Only need one translation for the value range
    return values


def process_evs(
    input_path: str,
    questions_path: str,
    output_path: str,
):
    """Main processing pipeline."""
    # Load questions
    with open(questions_path) as f:
        data = json.load(f)
    questions = data["questions"]
    questions_by_id = {q["canonical_id"]: q for q in questions}
    print(f"Loaded {len(questions)} canonical questions")

    # Load EVS SPSS file (supports .sav or .sav.zip)
    input_p = Path(input_path)
    if input_p.suffix == ".zip" or input_p.name.endswith(".sav.zip"):
        print(f"Extracting from zip: {input_path}...")
        with zipfile.ZipFile(input_path) as zf:
            sav_names = [n for n in zf.namelist() if n.endswith(".sav")]
            if not sav_names:
                raise FileNotFoundError(f"No .sav file found in {input_path}")
            sav_name = sav_names[0]
            print(f"  Found: {sav_name}")
            with tempfile.TemporaryDirectory() as tmpdir:
                zf.extract(sav_name, tmpdir)
                extracted = Path(tmpdir) / sav_name
                print(f"  Loading {extracted}...")
                df, meta = pyreadstat.read_sav(str(extracted))
    else:
        print(f"Loading {input_path}...")
        df, meta = pyreadstat.read_sav(input_path)
    print(f"  {len(df)} respondents, {len(df.columns)} variables")

    # Find key variables
    country_var = find_country_variable(list(df.columns))
    weight_var = find_weight_variable(list(df.columns))
    print(f"  Country variable: {country_var}")
    print(f"  Weight variable: {weight_var or '(none — using unweighted)'}")

    # Map countries to our languages
    available_countries = set(df[country_var].dropna().unique())
    mapped = {}
    for code, lang in COUNTRY_TO_LANG.items():
        if code in available_countries:
            mapped[code] = lang
        elif float(code) in available_countries:
            mapped[float(code)] = lang
    print(f"  Mapped {len(mapped)}/{len(COUNTRY_TO_LANG)} countries")

    if not mapped:
        # Try matching by country name if numeric codes don't work
        print("  WARNING: No numeric country codes matched. Trying value labels...")
        if country_var in meta.variable_value_labels:
            labels = meta.variable_value_labels[country_var]
            print(f"  Available country labels: {list(labels.values())[:10]}...")

    # Identify which v-code variables exist in the dataset
    available_vars = set(df.columns)
    matched_questions = {}
    for qid, question in questions_by_id.items():
        # canonical_id is a v-code like "v1", "v39", etc.
        if qid in available_vars:
            matched_questions[qid] = question
    print(f"  Matched {len(matched_questions)}/{len(questions)} questions to dataset variables")

    # Process each (country, question) pair
    rows = []
    n_skipped = 0

    for country_code, lang in sorted(mapped.items(), key=lambda x: x[1]):
        country_mask = df[country_var] == country_code
        country_df = df[country_mask]
        n_respondents = len(country_df)

        if n_respondents == 0:
            continue

        # Get weights for this country
        if weight_var and weight_var in country_df.columns:
            weights = country_df[weight_var].values
            # Replace NaN weights with 1.0
            weights = np.where(np.isnan(weights), 1.0, weights)
        else:
            weights = np.ones(n_respondents)

        for qid, question in matched_questions.items():
            valid_values = get_valid_range(question)
            if not valid_values:
                n_skipped += 1
                continue

            responses = country_df[qid].values

            # Filter to valid responses (in our option range, not NaN)
            valid_mask = np.isin(responses, list(valid_values)) & ~np.isnan(responses)
            valid_responses = responses[valid_mask].astype(int)
            valid_weights = weights[valid_mask]
            n_valid = int(valid_mask.sum())

            if n_valid == 0:
                n_skipped += 1
                continue

            # Compute weighted frequency distribution
            total_weight = valid_weights.sum()
            if total_weight == 0:
                n_skipped += 1
                continue

            for val in sorted(valid_values):
                val_mask = valid_responses == val
                weighted_count = valid_weights[val_mask].sum()
                prob = weighted_count / total_weight

                rows.append({
                    "lang": lang,
                    "question_id": qid,
                    "response_value": val,
                    "prob_human": prob,
                    "n_respondents": n_respondents,
                    "n_valid": n_valid,
                })

    print(f"\nProcessed {len(rows)} distribution entries")
    if n_skipped > 0:
        print(f"  Skipped {n_skipped} (country, question) pairs (no valid responses)")

    # Build output DataFrame
    result = pd.DataFrame(rows)

    # Summary
    n_langs = result["lang"].nunique()
    n_questions = result["question_id"].nunique()
    print(f"\nOutput summary:")
    print(f"  Languages: {n_langs}")
    print(f"  Questions: {n_questions}")
    print(f"  Total rows: {len(result)}")

    # Per-language coverage
    print(f"\n  Per-language question counts:")
    for lang in sorted(result["lang"].unique()):
        nq = result[result["lang"] == lang]["question_id"].nunique()
        print(f"    {lang}: {nq} questions")

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    print(f"\nSaved to {output_path}")


def main():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Process EVS 2017 survey data into country-level distributions"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to EVS SPSS file (ZA7500.sav)"
    )
    parser.add_argument(
        "--questions",
        default=str(PROJECT_ROOT / "data" / "questions.json"),
        help="Path to questions.json"
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "human_data" / "data" / "human_distributions.parquet"),
        help="Output parquet path"
    )
    parser.add_argument(
        "--db", default=None,
        help="Path to survey.db (also write human data to DB)"
    )
    args = parser.parse_args()

    process_evs(args.input, args.questions, args.output)

    if args.db:
        from db.populate import populate_human_data
        from db.schema import get_connection
        print(f"\nImporting to database: {args.db}")
        conn = get_connection(args.db)
        populate_human_data(conn, args.output)
        conn.close()


if __name__ == "__main__":
    main()
