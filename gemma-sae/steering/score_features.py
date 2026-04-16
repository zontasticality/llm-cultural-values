"""Score SAE features for language specificity (monolinguality metric).

Loads pre-extracted activations, encodes them through Gemma Scope 2 SAEs,
and computes the Deng et al. (ACL 2025) monolinguality metric:

    nu_s^L = mu_s^L - gamma_s^L

where mu_s^L is the mean activation of feature s on language L, and
gamma_s^L is the mean activation across all other languages.

Usage:
    PYTHONPATH=. python -m steering.score_features \
        --activations-dir data/activations \
        --output-dir data/feature_scores \
        --layers 31 40 53 \
        --sae-width 256k \
        --sae-l0 medium \
        [--top-k 20]
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# Map human-readable width to sae_lens naming
# Note: Gemma Scope 2 uses "262k" and "65k" (not "256k"/"64k")
WIDTH_MAP = {"16k": "16k", "64k": "65k", "256k": "262k", "1m": "1m"}


def load_activations(activations_dir: Path, layer: int) -> dict[str, np.ndarray]:
    """Load all language activations for a given layer.

    Returns: {lang: np.ndarray of shape (n_sentences, hidden_size)}
    """
    layer_dir = activations_dir / f"layer_{layer}" / "gemma3_27b_pt"
    if not layer_dir.exists():
        raise FileNotFoundError(f"No activations at {layer_dir}")

    acts = {}
    for path in sorted(layer_dir.glob("*.npz")):
        lang = path.stem
        data = np.load(path)
        acts[lang] = data["activations"]

    return acts


def load_sae(layer: int, width: str, l0: str):
    """Load a Gemma Scope 2 SAE via sae_lens.

    Returns (sae, cfg_dict) where sae can encode activations.
    """
    from sae_lens import SAE

    release = "gemma-scope-2-27b-pt-res"
    sae_id = f"layer_{layer}_width_{WIDTH_MAP[width]}_l0_{l0}"

    print(f"    Loading SAE: {release} / {sae_id}")
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=release,
        sae_id=sae_id,
    )
    sae.eval()
    return sae, cfg_dict


def encode_activations(sae, activations: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """Encode activations through SAE, return feature activations.

    Returns: np.ndarray of shape (n_sentences, n_features)
    """
    device = next(sae.parameters()).device
    all_features = []

    for i in range(0, len(activations), batch_size):
        batch = torch.tensor(activations[i:i + batch_size], dtype=torch.float32).to(device)
        with torch.no_grad():
            features = sae.encode(batch)
        all_features.append(features.cpu().numpy())

    return np.concatenate(all_features, axis=0)


def compute_monolinguality(
    feature_acts: dict[str, np.ndarray],
    top_k: int = 20,
) -> pd.DataFrame:
    """Compute Deng et al. monolinguality metric for all features.

    Args:
        feature_acts: {lang: np.ndarray of shape (n_sentences, n_features)}
        top_k: number of top features to report per language

    Returns: DataFrame with columns [lang, feature_idx, nu, mu, gamma, rank]
    """
    langs = sorted(feature_acts.keys())
    n_features = next(iter(feature_acts.values())).shape[1]

    # Compute mean activation per feature per language
    means = {}  # {lang: np.ndarray of shape (n_features,)}
    for lang in langs:
        means[lang] = feature_acts[lang].mean(axis=0)

    # Stack for vectorized computation
    mean_matrix = np.stack([means[lang] for lang in langs])  # (n_langs, n_features)

    results = []
    for i, lang in enumerate(langs):
        mu = mean_matrix[i]  # (n_features,)

        # gamma = mean of all other languages' means
        other_mask = np.ones(len(langs), dtype=bool)
        other_mask[i] = False
        gamma = mean_matrix[other_mask].mean(axis=0)  # (n_features,)

        nu = mu - gamma  # (n_features,)

        # Top-k features by nu
        top_indices = np.argsort(nu)[::-1][:top_k]
        for rank, feat_idx in enumerate(top_indices):
            results.append({
                "lang": lang,
                "feature_idx": int(feat_idx),
                "nu": float(nu[feat_idx]),
                "mu": float(mu[feat_idx]),
                "gamma": float(gamma[feat_idx]),
                "rank": rank + 1,
            })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Score SAE features for language specificity")
    parser.add_argument("--activations-dir", default="data/activations")
    parser.add_argument("--output-dir", default="data/feature_scores")
    parser.add_argument("--layers", nargs="+", type=int, default=[31, 40, 53])
    parser.add_argument("--sae-width", default="256k", choices=list(WIDTH_MAP.keys()))
    parser.add_argument("--sae-l0", default="medium", choices=["small", "medium", "large"])
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--sae-device", default="cpu", help="Device for SAE encoding")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    activations_dir = Path(args.activations_dir)
    output_dir = Path(args.output_dir)

    for layer in args.layers:
        print(f"\n{'='*60}")
        print(f"Layer {layer}")
        print(f"{'='*60}")

        # Load activations
        print(f"  Loading activations...")
        t0 = time.time()
        acts = load_activations(activations_dir, layer)
        langs = sorted(acts.keys())
        n_sentences = sum(a.shape[0] for a in acts.values())
        hidden_size = next(iter(acts.values())).shape[1]
        print(f"  {len(langs)} languages, {n_sentences} total sentences, hidden_size={hidden_size}")
        print(f"  Loaded in {time.time() - t0:.1f}s")

        # Load SAE
        print(f"  Loading SAE...")
        t1 = time.time()
        sae, cfg_dict = load_sae(layer, args.sae_width, args.sae_l0)
        if args.sae_device != "cpu":
            sae = sae.to(args.sae_device)
        n_features = sae.W_enc.shape[1] if hasattr(sae, 'W_enc') else sae.cfg.d_sae
        print(f"  SAE: {n_features} features, loaded in {time.time() - t1:.1f}s")

        # Encode activations through SAE
        print(f"  Encoding activations through SAE...")
        t2 = time.time()
        feature_acts = {}
        for lang in langs:
            feature_acts[lang] = encode_activations(sae, acts[lang], args.batch_size)
            print(f"    {lang}: {feature_acts[lang].shape}")
        print(f"  Encoded in {time.time() - t2:.1f}s")

        # Compute monolinguality scores
        print(f"  Computing monolinguality scores...")
        scores = compute_monolinguality(feature_acts, top_k=args.top_k)

        # Save
        layer_out = output_dir / f"layer_{layer}"
        layer_out.mkdir(parents=True, exist_ok=True)
        csv_path = layer_out / "monolinguality.csv"
        scores.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")

        # Print summary: top-5 per language
        print(f"\n  Top-5 language-specific features:")
        print(f"  {'Lang':<6s} {'#1':>8s} {'#2':>8s} {'#3':>8s} {'#4':>8s} {'#5':>8s}")
        print(f"  {'-'*50}")
        for lang in langs:
            lang_scores = scores[(scores["lang"] == lang) & (scores["rank"] <= 5)]
            nu_vals = lang_scores.sort_values("rank")["nu"].values
            feat_ids = lang_scores.sort_values("rank")["feature_idx"].values
            line = f"  {lang:<6s}"
            for j in range(min(5, len(nu_vals))):
                line += f" {nu_vals[j]:>7.2f}"
            print(line)

        # Free SAE memory before next layer
        del sae, feature_acts
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"\nDone.")


if __name__ == "__main__":
    main()
