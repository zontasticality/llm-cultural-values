"""Download Flores+ parallel sentences for our 27 target languages.

Requires: pip install datasets huggingface_hub
Requires: HuggingFace login (huggingface-cli login) with access to
          openlanguagedata/flores_plus

Usage:
    python -m steering.download_flores --output data/probes/flores_200

Outputs one JSONL file per language: {output_dir}/{lang}.jsonl
Each line: {"id": "...", "text": "...", "lang": "...", "topic": "..."}
"""

import argparse
import json
from pathlib import Path

# Map our ISO 639-3 codes to Flores+ config names (iso_639_3 + "_" + iso_15924)
LANG_TO_FLORES = {
    "bul": "bul_Cyrl",
    "ces": "ces_Latn",
    "dan": "dan_Latn",
    "deu": "deu_Latn",
    "ell": "ell_Grek",
    "eng": "eng_Latn",
    "est": "ekk_Latn",  # Estonian is 'ekk' in Flores+
    "fin": "fin_Latn",
    "fra": "fra_Latn",
    "hrv": "hrv_Latn",
    "hun": "hun_Latn",
    "ita": "ita_Latn",
    "lit": "lit_Latn",
    "lvs": "lvs_Latn",
    "nld": "nld_Latn",
    "pol": "pol_Latn",
    "por": "por_Latn",
    "ron": "ron_Latn",
    "slk": "slk_Latn",
    "slv": "slv_Latn",
    "spa": "spa_Latn",
    "swe": "swe_Latn",
    "zho": "cmn_Hans",  # Simplified Chinese (Mandarin)
    "jpn": "jpn_Jpan",
    "ara": "arb_Arab",  # Modern Standard Arabic
    "hin": "hin_Deva",
    "tur": "tur_Latn",
}


def download_lang(lang: str, flores_config: str, output_dir: Path, split: str = "devtest"):
    """Download one language from Flores+ and save as JSONL."""
    import shutil
    from huggingface_hub import hf_hub_download

    # Download the JSONL file directly from the repo
    src_path = hf_hub_download(
        repo_id="openlanguagedata/flores_plus",
        filename=f"{split}/{flores_config}.jsonl",
        repo_type="dataset",
    )

    out_path = output_dir / f"{lang}.jsonl"

    # Read source, re-emit with our lang code and consistent fields
    n = 0
    with open(src_path, encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line)
            record = {
                "id": row.get("id", n),
                "text": row["text"],
                "lang": lang,
                "topic": row.get("topic", ""),
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1

    print(f"  {lang} ({flores_config}): {n} sentences -> {out_path}")
    return n


def main():
    parser = argparse.ArgumentParser(description="Download Flores+ for target languages")
    parser.add_argument("--output", default="data/probes/flores_200", help="Output directory")
    parser.add_argument("--split", default="devtest", choices=["dev", "devtest"])
    parser.add_argument("--langs", nargs="*", help="Subset of languages (default: all 27)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    langs = args.langs or list(LANG_TO_FLORES.keys())
    print(f"Downloading Flores+ {args.split} for {len(langs)} languages -> {output_dir}/")

    total = 0
    for lang in langs:
        flores_config = LANG_TO_FLORES.get(lang)
        if not flores_config:
            print(f"  {lang}: SKIPPED (no Flores+ mapping)")
            continue
        try:
            n = download_lang(lang, flores_config, output_dir, args.split)
            total += n
        except Exception as e:
            print(f"  {lang}: FAILED ({e})")

    print(f"\nDone: {total} sentences across {len(langs)} languages.")


if __name__ == "__main__":
    main()
