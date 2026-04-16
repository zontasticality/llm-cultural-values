"""Sweep monolinguality across ALL 62 layers with 262k SAEs.

Produces a (language × layer) heatmap of where language features live.
Two phases:
  Phase 1: Forward pass to extract activations at all layers (GPU)
  Phase 2: Encode through SAEs layer-by-layer, compute monolinguality (CPU)

Usage:
    # Phase 1: extract (GPU, ~15 min)
    PYTHONPATH=. python -m steering.sweep_all_layers extract \
        --model google/gemma-3-27b-pt \
        --flores-dir data/probes/flores_200 \
        --output-dir data/activations_all \
        --max-sentences 200 --batch-size 4

    # Phase 2: score (CPU, ~2 hours for 62 layers)
    PYTHONPATH=. python -m steering.sweep_all_layers score \
        --activations-dir data/activations_all \
        --output-dir data/feature_scores/all_layers_262k \
        --sae-release gemma-scope-2-27b-pt-res-all \
        --sae-width 262k --sae-l0 big

    # Or both in one go (GPU job):
    PYTHONPATH=. python -m steering.sweep_all_layers all \
        --model google/gemma-3-27b-pt \
        --flores-dir data/probes/flores_200 \
        --output-dir data/activations_all \
        --max-sentences 200
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


WIDTH_MAP = {"16k": "16k", "262k": "262k"}
N_LAYERS = 62


# ── Phase 1: Extract activations at all layers ────────────────────

def extract_all_layers(model_name: str, flores_dir: Path, output_dir: Path,
                       max_sentences: int, batch_size: int, pool: str):
    """Forward pass with hooks at all 62 layers, save per-language activations."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    device = next(model.parameters()).device
    n_layers = len(model.model.language_model.layers)
    print(f"Model on {device}, {n_layers} layers")

    langs = sorted(p.stem for p in flores_dir.glob("*.jsonl"))
    print(f"Languages: {len(langs)}, max sentences: {max_sentences}")

    for lang in langs:
        # Load texts
        texts = []
        with open(flores_dir / f"{lang}.jsonl") as f:
            for i, line in enumerate(f):
                if i >= max_sentences:
                    break
                texts.append(json.loads(line)["text"])

        print(f"\n  {lang}: {len(texts)} sentences")
        t0 = time.time()

        # Set up hooks for ALL layers
        activations = {l: [] for l in range(n_layers)}
        hooks = []

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                hidden = output[0]  # (batch, seq, hidden)
                activations[layer_idx].append(hidden.detach().cpu())
            return hook_fn

        for l in range(n_layers):
            h = model.model.language_model.layers[l].register_forward_hook(make_hook(l))
            hooks.append(h)

        # Forward pass in batches
        for batch_start in range(0, len(texts), batch_size):
            batch = texts[batch_start:batch_start + batch_size]
            encoded = tokenizer(batch, return_tensors="pt", padding=True,
                                truncation=True, max_length=256).to(device)

            with torch.no_grad():
                model(**encoded)

            # Pool to last token per sentence
            mask = encoded["attention_mask"].cpu()
            lengths = mask.sum(dim=1) - 1

            for l in range(n_layers):
                hidden = activations[l][-1]
                if pool == "last":
                    pooled = torch.stack([hidden[i, lengths[i]] for i in range(len(batch))])
                else:
                    mask_exp = mask.unsqueeze(-1).float()
                    pooled = (hidden * mask_exp).sum(dim=1) / mask_exp.sum(dim=1)
                activations[l][-1] = pooled

        # Remove hooks
        for h in hooks:
            h.remove()

        # Save per-layer
        for l in range(n_layers):
            arr = torch.cat(activations[l], dim=0).float().numpy()
            layer_dir = output_dir / f"layer_{l}"
            layer_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(layer_dir / f"{lang}.npz", activations=arr)

        elapsed = time.time() - t0
        print(f"    {elapsed:.1f}s ({len(texts) / elapsed:.1f} sent/s)")

        # Free memory
        del activations
        torch.cuda.empty_cache()

    print(f"\nExtraction complete.")


# ── Phase 2: Score monolinguality across all layers ────────────────

def score_all_layers(activations_dir: Path, output_dir: Path,
                     sae_release: str, sae_width: str, sae_l0: str,
                     top_k: int):
    """Load SAEs one layer at a time, compute monolinguality."""
    from sae_lens import SAE

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find which layers have activations
    layer_dirs = sorted(activations_dir.glob("layer_*"))
    layers = [int(d.name.split("_")[1]) for d in layer_dirs]
    print(f"Found activations for {len(layers)} layers: {layers[0]}-{layers[-1]}")

    # Find languages
    first_dir = layer_dirs[0]
    langs = sorted(p.stem for p in first_dir.glob("*.npz"))
    print(f"Languages: {len(langs)}")

    all_results = []
    # Per-layer: top-1 nu for summary heatmap
    heatmap_data = []

    for layer in layers:
        print(f"\n  Layer {layer}/{layers[-1]}")
        t0 = time.time()

        # Load activations for this layer
        acts = {}
        for lang in langs:
            path = activations_dir / f"layer_{layer}" / f"{lang}.npz"
            acts[lang] = np.load(path)["activations"]

        # Load SAE
        sae_id = f"layer_{layer}_width_{WIDTH_MAP[sae_width]}_l0_{sae_l0}"
        try:
            sae = SAE.from_pretrained(release=sae_release, sae_id=sae_id)
        except Exception as e:
            print(f"    SKIP: {e}")
            continue
        sae.eval()
        n_features = sae.cfg.d_sae

        # Encode activations
        feature_means = {}
        for lang in langs:
            tensor = torch.tensor(acts[lang], dtype=torch.float32)
            with torch.no_grad():
                features = sae.encode(tensor)
            feature_means[lang] = features.detach().cpu().numpy().mean(axis=0)

        # Compute monolinguality
        mean_matrix = np.stack([feature_means[lang] for lang in langs])

        for i, lang in enumerate(langs):
            mu = mean_matrix[i]
            other_mask = np.ones(len(langs), dtype=bool)
            other_mask[i] = False
            gamma = mean_matrix[other_mask].mean(axis=0)
            nu = mu - gamma

            top_indices = np.argsort(nu)[::-1][:top_k]
            for rank, feat_idx in enumerate(top_indices):
                all_results.append({
                    "layer": layer,
                    "lang": lang,
                    "feature_idx": int(feat_idx),
                    "nu": float(nu[feat_idx]),
                    "mu": float(mu[feat_idx]),
                    "gamma": float(gamma[feat_idx]),
                    "rank": rank + 1,
                })

            # Top-1 nu for heatmap
            heatmap_data.append({
                "layer": layer,
                "lang": lang,
                "top1_nu": float(nu[top_indices[0]]),
                "top1_feature": int(top_indices[0]),
            })

        # Free SAE
        del sae, feature_means
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        elapsed = time.time() - t0
        # Print compact summary
        top1_by_lang = {r["lang"]: r["top1_nu"]
                        for r in heatmap_data if r["layer"] == layer}
        max_lang = max(top1_by_lang, key=top1_by_lang.get)
        min_lang = min(top1_by_lang, key=top1_by_lang.get)
        print(f"    {elapsed:.1f}s | strongest: {max_lang}={top1_by_lang[max_lang]:.0f}, "
              f"weakest: {min_lang}={top1_by_lang[min_lang]:.0f}")

    # Save full results
    df = pd.DataFrame(all_results)
    df.to_csv(output_dir / "monolinguality_all_layers.csv", index=False)
    print(f"\nSaved: {output_dir}/monolinguality_all_layers.csv")

    # Save heatmap data
    hdf = pd.DataFrame(heatmap_data)
    hdf.to_csv(output_dir / "heatmap_top1_nu.csv", index=False)
    print(f"Saved: {output_dir}/heatmap_top1_nu.csv")

    # Print summary: layer with strongest language differentiation
    layer_strength = hdf.groupby("layer")["top1_nu"].mean().sort_values(ascending=False)
    print(f"\nTop 10 layers by mean top-1 monolinguality:")
    for layer, mean_nu in layer_strength.head(10).items():
        print(f"  layer {layer:>2d}: mean nu = {mean_nu:.0f}")

    # Generate heatmap figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pivot = hdf.pivot(index="lang", columns="layer", values="top1_nu")
        fig, ax = plt.subplots(figsize=(20, 8))
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=6, rotation=90)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Language")
        ax.set_title("Top-1 Monolinguality Score (nu) by Language × Layer\n"
                      f"SAE: {sae_release}, {sae_width}, L0={sae_l0}")
        plt.colorbar(im, label="nu (monolinguality)")
        plt.tight_layout()
        fig.savefig(output_dir / "monolinguality_heatmap.png", dpi=150)
        plt.close(fig)
        print(f"Saved: {output_dir}/monolinguality_heatmap.png")
    except ImportError:
        print("  (matplotlib not available, skipping heatmap figure)")


# ── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sweep monolinguality across all layers")
    parser.add_argument("phase", choices=["extract", "score", "all"])
    parser.add_argument("--model", default="google/gemma-3-27b-pt")
    parser.add_argument("--flores-dir", default="data/probes/flores_200")
    parser.add_argument("--output-dir", default="data/activations_all",
                        help="For extract: where to save activations. For score: ignored.")
    parser.add_argument("--activations-dir", default="data/activations_all",
                        help="For score: where activations are.")
    parser.add_argument("--scores-dir", default="data/feature_scores/all_layers_262k",
                        help="For score: where to save results.")
    parser.add_argument("--max-sentences", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--pool", default="last", choices=["last", "mean"])
    parser.add_argument("--sae-release", default="gemma-scope-2-27b-pt-res-all")
    parser.add_argument("--sae-width", default="262k", choices=list(WIDTH_MAP.keys()))
    parser.add_argument("--sae-l0", default="big", choices=["small", "big"])
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    if args.phase in ("extract", "all"):
        extract_all_layers(
            args.model, Path(args.flores_dir), Path(args.output_dir),
            args.max_sentences, args.batch_size, args.pool,
        )

    if args.phase in ("score", "all"):
        # In "all" mode, activations were saved to --output-dir
        act_dir = Path(args.output_dir) if args.phase == "all" else Path(args.activations_dir)
        score_all_layers(
            act_dir, Path(args.scores_dir),
            args.sae_release, args.sae_width, args.sae_l0, args.top_k,
        )


if __name__ == "__main__":
    main()
