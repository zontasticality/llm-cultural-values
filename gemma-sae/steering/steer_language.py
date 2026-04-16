"""Language steering via SAE encode-modify-decode at consecutive layers.

Unlike the additive steering vector approach, this does the full loop:
  1. Hook residual stream at target layers
  2. At each hook: SAE.encode → modify features → SAE.decode → replace
  3. Modify ALL token positions (not just last)

This replicates the approach from "Causal Language Control in Multilingual
Transformers via Sparse Feature Steering" (2025) which achieved 85-98%
language switching success with just 3 features per language.

Usage:
    PYTHONPATH=. python -m steering.steer_language \
        --model google/gemma-3-27b-pt \
        --layers 43 44 45 \
        --sae-release gemma-scope-2-27b-pt-res-all \
        --sae-width 16k --sae-l0 big \
        --feature-scores data/feature_scores/layers_43_44_45_16k/monolinguality.csv \
        --source-lang eng --target-langs fin zho ara \
        --top-k 3 --clamp-value 50 \
        --prompt "The most important thing in life is" \
        --n-samples 3

    # Or with --scan to auto-detect features from Flores data first:
    PYTHONPATH=. python -m steering.steer_language \
        --model google/gemma-3-27b-pt \
        --layers 43 44 45 \
        --sae-release gemma-scope-2-27b-pt-res-all \
        --sae-width 16k --sae-l0 big \
        --scan --flores-dir data/probes/flores_200 --scan-sentences 200 \
        --source-lang eng --target-langs fin zho ara \
        --top-k 3 --clamp-value 50 \
        --prompt "The most important thing in life is" \
        --n-samples 3
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


WIDTH_MAP = {"16k": "16k", "64k": "65k", "256k": "262k"}


def load_sae(release: str, layer: int, width: str, l0: str, device: str = "cpu"):
    """Load a per-layer SAE via sae_lens."""
    from sae_lens import SAE
    sae_id = f"layer_{layer}_width_{WIDTH_MAP[width]}_l0_{l0}"
    print(f"    Loading SAE: {release} / {sae_id}")
    sae = SAE.from_pretrained(release=release, sae_id=sae_id)
    sae = sae.to(device)
    sae.eval()
    return sae


class EncodeModifyDecodeHook:
    """Hook that does full SAE encode → modify features → decode → replace."""

    def __init__(self, sae, source_features: list[int], target_features: list[int],
                 clamp_value: float):
        self.sae = sae
        self.source_features = source_features
        self.target_features = target_features
        self.clamp_value = clamp_value
        self.enabled = False

    def __call__(self, module, input, output):
        if not self.enabled:
            return output

        hidden = output[0]  # (batch, seq_len, hidden_size)
        rest = output[1:]

        with torch.no_grad():
            # Move to CPU for SAE encode/decode (SAE stays on CPU to save GPU memory)
            hidden_cpu = hidden.float().cpu()

            original_features = self.sae.encode(hidden_cpu)
            modified_features = original_features.clone()
            for f_idx in self.source_features:
                modified_features[:, :, f_idx] = 0.0
            for f_idx in self.target_features:
                modified_features[:, :, f_idx] = self.clamp_value

            delta = self.sae.decode(modified_features) - self.sae.decode(original_features)
            result = hidden_cpu + delta

        return (result.to(hidden.device, hidden.dtype),) + rest


def scan_features(model, tokenizer, saes: dict[int, object], flores_dir: Path,
                  langs: list[str], n_sentences: int, device: torch.device) -> pd.DataFrame:
    """Quick monolinguality scan: run Flores through model, encode with SAEs,
    compute nu scores. Returns DataFrame with top features per language per layer."""

    print(f"\n  Scanning {n_sentences} Flores sentences × {len(langs)} languages...")

    # Collect activations via hooks
    layer_indices = sorted(saes.keys())
    captured: dict[int, list] = {l: [] for l in layer_indices}
    hooks = []

    def make_capture_hook(layer_idx):
        def hook_fn(module, input, output):
            captured[layer_idx].append(output[0].detach().cpu())
        return hook_fn

    for layer_idx in layer_indices:
        h = model.model.language_model.layers[layer_idx].register_forward_hook(
            make_capture_hook(layer_idx))
        hooks.append(h)

    # Per-language, per-layer feature activations
    all_feature_acts: dict[int, dict[str, list]] = {l: {} for l in layer_indices}

    try:
        for lang in langs:
            path = flores_dir / f"{lang}.jsonl"
            texts = []
            with open(path) as f:
                for i, line in enumerate(f):
                    if i >= n_sentences:
                        break
                    texts.append(json.loads(line)["text"])

            # Reset captures
            for l in layer_indices:
                captured[l] = []

            # Forward pass in batches
            for batch_start in range(0, len(texts), 8):
                batch = texts[batch_start:batch_start + 8]
                encoded = tokenizer(batch, return_tensors="pt", padding=True,
                                    truncation=True, max_length=256).to(device)
                with torch.no_grad():
                    model(**encoded)

                # Encode through SAEs and extract last-token features
                mask = encoded["attention_mask"].cpu()
                lengths = mask.sum(dim=1) - 1

                for layer_idx in layer_indices:
                    hidden = captured[layer_idx][-1]  # (batch, seq, hidden)
                    # Last token per sentence
                    last_hidden = torch.stack([hidden[i, lengths[i]] for i in range(len(batch))])
                    feats = saes[layer_idx].encode(last_hidden.float().to(
                        next(saes[layer_idx].parameters()).device))
                    if lang not in all_feature_acts[layer_idx]:
                        all_feature_acts[layer_idx][lang] = []
                    all_feature_acts[layer_idx][lang].append(feats.detach().cpu().numpy())

            print(f"    {lang}: done")

    finally:
        for h in hooks:
            h.remove()

    # Compute monolinguality per layer
    results = []
    for layer_idx in layer_indices:
        feature_acts = {}
        for lang in langs:
            feature_acts[lang] = np.concatenate(all_feature_acts[layer_idx][lang], axis=0)

        means = {lang: feature_acts[lang].mean(axis=0) for lang in langs}
        mean_matrix = np.stack([means[lang] for lang in langs])

        for i, lang in enumerate(langs):
            mu = mean_matrix[i]
            other_mask = np.ones(len(langs), dtype=bool)
            other_mask[i] = False
            gamma = mean_matrix[other_mask].mean(axis=0)
            nu = mu - gamma
            top_indices = np.argsort(nu)[::-1][:20]
            for rank, feat_idx in enumerate(top_indices):
                results.append({
                    "layer": layer_idx,
                    "lang": lang,
                    "feature_idx": int(feat_idx),
                    "nu": float(nu[feat_idx]),
                    "rank": rank + 1,
                })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Language steering via SAE encode-modify-decode")
    parser.add_argument("--model", default="google/gemma-3-27b-pt")
    parser.add_argument("--layers", nargs="+", type=int, default=[43, 44, 45])
    parser.add_argument("--sae-release", default="gemma-scope-2-27b-pt-res-all")
    parser.add_argument("--sae-width", default="16k")
    parser.add_argument("--sae-l0", default="big")

    # Feature source: either pre-computed CSV or live scan
    parser.add_argument("--feature-scores", help="Pre-computed monolinguality CSV")
    parser.add_argument("--scan", action="store_true", help="Scan Flores to find features")
    parser.add_argument("--flores-dir", default="data/probes/flores_200")
    parser.add_argument("--scan-sentences", type=int, default=200)

    parser.add_argument("--source-lang", default="eng")
    parser.add_argument("--target-lang", default="fin")
    parser.add_argument("--target-langs", nargs="*")
    parser.add_argument("--prompt", default="The most important thing in life is")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--clamp-value", type=float, default=50.0)
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--n-samples", type=int, default=3)
    args = parser.parse_args()

    target_langs = args.target_langs or [args.target_lang]
    all_langs_needed = list(set([args.source_lang] + target_langs))

    # Load model
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    device = next(model.parameters()).device
    print(f"Model on {device}")

    # Load SAEs for all target layers
    print(f"\nLoading SAEs for layers {args.layers}...")
    saes = {}
    for layer in args.layers:
        saes[layer] = load_sae(args.sae_release, layer, args.sae_width, args.sae_l0,
                                device="cpu")  # keep on CPU to avoid OOM with large model
    n_features = next(iter(saes.values())).cfg.d_sae
    print(f"  {n_features} features per SAE")

    # Get feature indices
    if args.scan:
        print(f"\nScanning for language features...")
        # Scan needs all 27 languages for proper monolinguality computation
        flores_dir = Path(args.flores_dir)
        scan_langs = sorted(p.stem for p in flores_dir.glob("*.jsonl"))
        scores = scan_features(model, tokenizer, saes, flores_dir,
                               scan_langs, args.scan_sentences, device)

        # Save scan results
        scan_path = Path("data/feature_scores") / f"scan_{'_'.join(map(str, args.layers))}_{args.sae_width}"
        scan_path.mkdir(parents=True, exist_ok=True)
        scores.to_csv(scan_path / "monolinguality.csv", index=False)
        print(f"  Saved scan results to {scan_path}/monolinguality.csv")
    elif args.feature_scores:
        scores = pd.read_csv(args.feature_scores)
    else:
        parser.error("Either --feature-scores or --scan is required")

    # Print top features for relevant languages
    for lang in all_langs_needed:
        for layer in args.layers:
            if "layer" in scores.columns:
                layer_scores = scores[(scores["layer"] == layer) & (scores["lang"] == lang)]
            else:
                layer_scores = scores[scores["lang"] == lang]
            top = layer_scores.nsmallest(args.top_k, "rank")
            feats = top["feature_idx"].tolist()
            nus = top["nu"].tolist()
            print(f"  {lang} @ layer {layer}: features={feats}, nu={[f'{n:.1f}' for n in nus]}")

    # Run steering tests
    for target_lang in target_langs:
        print(f"\n{'='*60}")
        print(f"Steering: {args.source_lang} -> {target_lang}")
        print(f"Layers: {args.layers}, Top-k: {args.top_k}, Clamp: {args.clamp_value}")
        print(f"{'='*60}")

        # Set up encode-modify-decode hooks at all layers
        hook_objs = []
        hook_handles = []
        for layer in args.layers:
            if "layer" in scores.columns:
                src_scores = scores[(scores["layer"] == layer) & (scores["lang"] == args.source_lang)]
                tgt_scores = scores[(scores["layer"] == layer) & (scores["lang"] == target_lang)]
            else:
                src_scores = scores[scores["lang"] == args.source_lang]
                tgt_scores = scores[scores["lang"] == target_lang]

            src_feats = src_scores.nsmallest(args.top_k, "rank")["feature_idx"].tolist()
            tgt_feats = tgt_scores.nsmallest(args.top_k, "rank")["feature_idx"].tolist()

            # Remove overlapping features (can't ablate and clamp the same one)
            overlap = set(src_feats) & set(tgt_feats)
            if overlap:
                print(f"    WARNING: overlapping features {overlap} — excluding from ablation")
                src_feats = [f for f in src_feats if f not in overlap]

            hook = EncodeModifyDecodeHook(saes[layer], src_feats, tgt_feats, args.clamp_value)
            handle = model.model.language_model.layers[layer].register_forward_hook(hook)
            hook_objs.append(hook)
            hook_handles.append(handle)

            print(f"  Layer {layer}: ablate {src_feats}, clamp {tgt_feats}")

        try:
            input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)

            for i in range(args.n_samples):
                # Baseline
                for h in hook_objs:
                    h.enabled = False
                with torch.no_grad():
                    base_ids = model.generate(
                        input_ids, max_new_tokens=args.max_tokens,
                        do_sample=True, temperature=0.8, top_p=0.95,
                    )
                baseline = tokenizer.decode(base_ids[0][input_ids.shape[1]:],
                                            skip_special_tokens=True)

                # Steered
                for h in hook_objs:
                    h.enabled = True
                with torch.no_grad():
                    steer_ids = model.generate(
                        input_ids, max_new_tokens=args.max_tokens,
                        do_sample=True, temperature=0.8, top_p=0.95,
                    )
                steered = tokenizer.decode(steer_ids[0][input_ids.shape[1]:],
                                           skip_special_tokens=True)

                print(f"\n  Sample {i+1}:")
                print(f"  Baseline: {baseline[:200]}")
                print(f"  Steered:  {steered[:200]}")

        finally:
            for h in hook_handles:
                h.remove()

    print(f"\nDone.")


if __name__ == "__main__":
    main()
