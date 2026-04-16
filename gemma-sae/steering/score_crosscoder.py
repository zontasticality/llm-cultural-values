"""Score crosscoder features for language specificity.

Crosscoders encode activations from multiple layers simultaneously,
finding features that persist across the network. This should identify
language features that are both detectable AND steerable.

The crosscoder has per-layer encoder/decoder weights but a shared
latent space. Feature activation = JumpReLU(sum of per-layer encodings).

Usage:
    PYTHONPATH=. python -m steering.score_crosscoder \
        --activations-dir data/activations \
        --output-dir data/feature_scores \
        --width 262k --l0 medium \
        --top-k 20
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from huggingface_hub import hf_hub_download


CROSSCODER_LAYERS = [16, 31, 40, 53]
REPO_ID = "google/gemma-scope-2-27b-pt"
WIDTH_MAP = {"262k": "262k", "524k": "524k"}


class Crosscoder:
    """Manual crosscoder implementation from Gemma Scope 2 safetensors."""

    def __init__(self, W_enc: list[torch.Tensor], b_enc: torch.Tensor,
                 threshold: torch.Tensor, W_dec: list[torch.Tensor],
                 b_dec: list[torch.Tensor]):
        """
        W_enc: list of (hidden_size, n_features) per layer — encoder weights
        b_enc: (n_features,) — shared encoder bias
        threshold: (n_features,) — JumpReLU threshold
        W_dec: list of (n_features, hidden_size) per layer — decoder weights
        b_dec: list of (hidden_size,) per layer — decoder bias
        """
        self.W_enc = W_enc
        self.b_enc = b_enc
        self.threshold = threshold
        self.W_dec = W_dec
        self.b_dec = b_dec
        self.n_features = b_enc.shape[0]
        self.n_layers = len(W_enc)

    @classmethod
    def from_pretrained(cls, width: str, l0: str, device: str = "cpu"):
        """Load crosscoder weights from HuggingFace."""
        layers_str = "_".join(str(l) for l in CROSSCODER_LAYERS)
        prefix = f"crosscoder/layer_{layers_str}_width_{width}_l0_{l0}"

        print(f"  Downloading crosscoder: {prefix}")

        # Load config
        cfg_path = hf_hub_download(REPO_ID, f"{prefix}/config.json")
        with open(cfg_path) as f:
            config = json.load(f)
        print(f"  Config: width={config['width']}, l0={config['l0']}, arch={config['architecture']}")

        W_enc_list = []
        W_dec_list = []
        b_dec_list = []
        b_enc = None
        threshold = None

        for i in range(len(CROSSCODER_LAYERS)):
            path = hf_hub_download(REPO_ID, f"{prefix}/params_layer_{i}.safetensors")
            with safe_open(path, framework="pt", device=device) as f:
                keys = list(f.keys())
                print(f"  layer_{i} keys: {keys}")

                # Keys are lowercase in Gemma Scope 2 safetensors
                W_enc_list.append(f.get_tensor("w_enc").float())
                W_dec_list.append(f.get_tensor("w_dec").float())

                if "b_dec" in keys:
                    b_dec_list.append(f.get_tensor("b_dec").float())

                # Shared encoder bias and threshold (same across layers, load once)
                if b_enc is None and "b_enc" in keys:
                    b_enc = f.get_tensor("b_enc").float()
                if threshold is None and "threshold" in keys:
                    threshold = f.get_tensor("threshold").float()

        if b_enc is None:
            # Try loading from layer 0 if not found
            raise ValueError("Could not find b_enc in any layer file")

        for tensor_list, name in [(W_enc_list, "W_enc"), (W_dec_list, "W_dec")]:
            shapes = [t.shape for t in tensor_list]
            print(f"  {name} shapes: {shapes}")

        print(f"  b_enc: {b_enc.shape}, threshold: {threshold.shape if threshold is not None else 'None'}")
        if b_dec_list:
            print(f"  b_dec shapes: {[t.shape for t in b_dec_list]}")

        return cls(W_enc_list, b_enc, threshold, W_dec_list, b_dec_list)

    def encode(self, activations_per_layer: list[torch.Tensor]) -> torch.Tensor:
        """Encode activations from all layers into shared feature space.

        Args:
            activations_per_layer: list of (batch, hidden_size) tensors, one per layer

        Returns: (batch, n_features) sparse feature activations
        """
        # Sum encoder contributions from all layers
        pre_act = self.b_enc.unsqueeze(0)  # (1, n_features)
        for i, (acts, W) in enumerate(zip(activations_per_layer, self.W_enc)):
            pre_act = pre_act + acts @ W  # (batch, n_features)

        # JumpReLU: activate if above threshold
        if self.threshold is not None:
            mask = pre_act > self.threshold.unsqueeze(0)
            features = pre_act * mask.float()
        else:
            features = torch.relu(pre_act)

        return features


def load_activations_multilayer(activations_dir: Path) -> dict[str, list[np.ndarray]]:
    """Load activations for all languages across crosscoder layers.

    Returns: {lang: [np.ndarray(n_sent, hidden) for each layer in CROSSCODER_LAYERS]}
    """
    # Check which languages have data for all layers
    first_layer_dir = activations_dir / f"layer_{CROSSCODER_LAYERS[0]}" / "gemma3_27b_pt"
    langs = sorted(p.stem for p in first_layer_dir.glob("*.npz"))

    result = {}
    for lang in langs:
        layer_acts = []
        for layer in CROSSCODER_LAYERS:
            path = activations_dir / f"layer_{layer}" / "gemma3_27b_pt" / f"{lang}.npz"
            if not path.exists():
                break
            layer_acts.append(np.load(path)["activations"])
        else:
            result[lang] = layer_acts

    return result


def compute_monolinguality(feature_acts: dict[str, np.ndarray], top_k: int) -> pd.DataFrame:
    """Compute Deng et al. monolinguality metric (same as score_features.py)."""
    langs = sorted(feature_acts.keys())
    means = {lang: feature_acts[lang].mean(axis=0) for lang in langs}
    mean_matrix = np.stack([means[lang] for lang in langs])

    results = []
    for i, lang in enumerate(langs):
        mu = mean_matrix[i]
        other_mask = np.ones(len(langs), dtype=bool)
        other_mask[i] = False
        gamma = mean_matrix[other_mask].mean(axis=0)
        nu = mu - gamma

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
    parser = argparse.ArgumentParser(description="Score crosscoder features for language specificity")
    parser.add_argument("--activations-dir", default="data/activations")
    parser.add_argument("--output-dir", default="data/feature_scores")
    parser.add_argument("--width", default="262k", choices=list(WIDTH_MAP.keys()))
    parser.add_argument("--l0", default="medium", choices=["small", "medium", "big"])
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    activations_dir = Path(args.activations_dir)
    output_dir = Path(args.output_dir)

    # We need layer 16 activations too — check if they exist
    layer_16_dir = activations_dir / "layer_16" / "gemma3_27b_pt"
    if not layer_16_dir.exists():
        print(f"ERROR: Missing layer 16 activations at {layer_16_dir}")
        print(f"Re-run extract_activations.py with --layers 16 31 40 53")
        return

    print("Loading activations for all 4 crosscoder layers...")
    t0 = time.time()
    all_acts = load_activations_multilayer(activations_dir)
    langs = sorted(all_acts.keys())
    n_sentences = sum(acts[0].shape[0] for acts in all_acts.values())
    print(f"  {len(langs)} languages, {n_sentences} total sentences")
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # Load crosscoder
    print(f"\nLoading crosscoder...")
    t1 = time.time()
    cc = Crosscoder.from_pretrained(WIDTH_MAP[args.width], args.l0, device=args.device)
    print(f"  {cc.n_features} features, {cc.n_layers} layers")
    print(f"  Loaded in {time.time() - t1:.1f}s")

    # Encode all activations
    print(f"\nEncoding activations through crosscoder...")
    t2 = time.time()
    feature_acts = {}
    for lang in langs:
        layer_tensors = [torch.tensor(a, dtype=torch.float32).to(args.device)
                         for a in all_acts[lang]]
        n = layer_tensors[0].shape[0]

        # Process in batches
        all_features = []
        for start in range(0, n, args.batch_size):
            batch = [t[start:start + args.batch_size] for t in layer_tensors]
            with torch.no_grad():
                feats = cc.encode(batch)
            all_features.append(feats.cpu().numpy())

        feature_acts[lang] = np.concatenate(all_features, axis=0)
        print(f"  {lang}: {feature_acts[lang].shape}")

    print(f"  Encoded in {time.time() - t2:.1f}s")

    # Compute monolinguality
    print(f"\nComputing monolinguality scores...")
    scores = compute_monolinguality(feature_acts, top_k=args.top_k)

    out_path = output_dir / "crosscoder_16_31_40_53"
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_path / "monolinguality.csv"
    scores.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Print summary
    print(f"\nTop-5 language-specific crosscoder features:")
    print(f"  {'Lang':<6s} {'#1':>8s} {'#2':>8s} {'#3':>8s} {'#4':>8s} {'#5':>8s}")
    print(f"  {'-'*50}")
    for lang in langs:
        lang_scores = scores[(scores["lang"] == lang) & (scores["rank"] <= 5)]
        nu_vals = lang_scores.sort_values("rank")["nu"].values
        line = f"  {lang:<6s}"
        for j in range(min(5, len(nu_vals))):
            line += f" {nu_vals[j]:>7.2f}"
        print(line)

    print(f"\nDone.")


if __name__ == "__main__":
    main()
