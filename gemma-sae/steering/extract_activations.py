"""Extract residual stream activations from Gemma 3 27B on Flores parallel text.

For each language, runs Flores sentences through the model and saves
residual stream activations at specified layers. These activations are
then encoded by SAEs in score_features.py.

Usage:
    PYTHONPATH=. python -m steering.extract_activations \
        --model google/gemma-3-27b-pt \
        --flores-dir data/probes/flores_200 \
        --output-dir data/activations \
        --layers 31 40 53 \
        --batch-size 8 \
        [--langs eng fin pol] \
        [--pool last] \
        [--max-sentences 0]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_flores(flores_dir: Path, lang: str) -> list[str]:
    """Load Flores sentences for one language."""
    path = flores_dir / f"{lang}.jsonl"
    texts = []
    with open(path) as f:
        for line in f:
            texts.append(json.loads(line)["text"])
    return texts


def extract_for_lang(
    model,
    tokenizer,
    texts: list[str],
    layers: list[int],
    batch_size: int,
    pool: str,
    device: torch.device,
) -> dict[int, np.ndarray]:
    """Extract residual stream activations for a list of texts.

    Returns dict mapping layer_idx -> np.ndarray of shape (n_texts, hidden_size).
    """
    # Register hooks to capture residual stream (output of each layer)
    activations: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # output is (hidden_states, ...) — first element is the residual stream
            hidden = output[0]  # (batch, seq_len, hidden_size)
            activations[layer_idx].append(hidden.detach().cpu())
        return hook_fn

    for layer_idx in layers:
        h = model.model.language_model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

    try:
        for batch_start in range(0, len(texts), batch_size):
            batch_texts = texts[batch_start:batch_start + batch_size]

            encoded = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).to(device)

            with torch.no_grad():
                model(**encoded)

            # Pool activations per sentence
            for layer_idx in layers:
                hidden = activations[layer_idx][-1]  # (batch, seq_len, hidden_size)
                mask = encoded["attention_mask"].cpu()

                if pool == "last":
                    # Last real token position per sentence
                    lengths = mask.sum(dim=1) - 1  # 0-indexed
                    pooled = torch.stack([
                        hidden[i, lengths[i]] for i in range(len(batch_texts))
                    ])
                elif pool == "mean":
                    # Mean over non-padding positions
                    mask_expanded = mask.unsqueeze(-1).float()
                    pooled = (hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
                else:
                    raise ValueError(f"Unknown pool method: {pool}")

                # Replace the raw hidden states with pooled version
                activations[layer_idx][-1] = pooled

    finally:
        for h in hooks:
            h.remove()

    # Concatenate batches
    result = {}
    for layer_idx in layers:
        result[layer_idx] = torch.cat(activations[layer_idx], dim=0).float().numpy()

    return result


def main():
    parser = argparse.ArgumentParser(description="Extract residual stream activations")
    parser.add_argument("--model", default="google/gemma-3-27b-pt")
    parser.add_argument("--flores-dir", default="data/probes/flores_200")
    parser.add_argument("--output-dir", default="data/activations")
    parser.add_argument("--layers", nargs="+", type=int, default=[31, 40, 53])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--pool", default="last", choices=["last", "mean"])
    parser.add_argument("--langs", nargs="*", help="Subset of languages (default: all in flores-dir)")
    parser.add_argument("--max-sentences", type=int, default=0,
                        help="Limit sentences per language (0 = all)")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "int8"])
    args = parser.parse_args()

    flores_dir = Path(args.flores_dir)
    output_dir = Path(args.output_dir)

    # Discover languages
    if args.langs:
        langs = args.langs
    else:
        langs = sorted(p.stem for p in flores_dir.glob("*.jsonl"))

    print(f"Model: {args.model}")
    print(f"Layers: {args.layers}")
    print(f"Pool: {args.pool}")
    print(f"Languages: {len(langs)}")
    print(f"Batch size: {args.batch_size}")

    # Load model
    print(f"\nLoading model...")
    t0 = time.time()

    if args.dtype == "bf16":
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto",
        )
    else:
        from transformers import BitsAndBytesConfig
        model = AutoModelForCausalLM.from_pretrained(
            args.model, device_map="auto",
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        )

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    device = next(model.parameters()).device
    print(f"Model loaded on {device} in {time.time() - t0:.1f}s")

    # Verify layer access
    n_layers = len(model.model.language_model.layers)
    print(f"Model has {n_layers} layers")
    for l in args.layers:
        assert l < n_layers, f"Layer {l} >= {n_layers}"

    # Process each language
    for lang in langs:
        if not (flores_dir / f"{lang}.jsonl").exists():
            print(f"\n  {lang}: SKIP (no Flores data)")
            continue

        texts = load_flores(flores_dir, lang)
        if args.max_sentences > 0:
            texts = texts[:args.max_sentences]

        print(f"\n  {lang}: {len(texts)} sentences")
        t1 = time.time()

        acts = extract_for_lang(
            model, tokenizer, texts, args.layers, args.batch_size, args.pool, device,
        )

        for layer_idx, arr in acts.items():
            layer_dir = output_dir / f"layer_{layer_idx}" / "gemma3_27b_pt"
            layer_dir.mkdir(parents=True, exist_ok=True)
            out_path = layer_dir / f"{lang}.npz"
            np.savez_compressed(out_path, activations=arr)
            print(f"    layer {layer_idx}: {arr.shape} -> {out_path}")

        elapsed = time.time() - t1
        print(f"    {elapsed:.1f}s ({len(texts) / elapsed:.1f} sent/s)")

    print(f"\nDone. Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
