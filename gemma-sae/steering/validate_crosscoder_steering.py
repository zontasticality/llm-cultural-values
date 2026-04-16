"""Validate language steering via crosscoder features.

Unlike per-layer SAE steering (which only intervenes at one layer),
crosscoder features have decoder directions at ALL 4 layers (16, 31, 40, 53).
We apply the steering delta simultaneously at all layers during generation.

Approach:
  1. From monolinguality scores, get top-k features per language
  2. From crosscoder decoder weights, get each feature's direction at each layer
  3. During generation, at each crosscoder layer:
     - Add (scale * target_feature_decoder_direction)
     - Subtract (scale * source_feature_decoder_direction)

Usage:
    PYTHONPATH=. python -m steering.validate_crosscoder_steering \
        --model google/gemma-3-27b-pt \
        --feature-scores data/feature_scores/crosscoder_16_31_40_53/monolinguality.csv \
        --source-lang eng --target-langs fin zho ara \
        --top-k 2 --scale 3.0 \
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
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer


CROSSCODER_LAYERS = [16, 31, 40, 53]
REPO_ID = "google/gemma-scope-2-27b-pt"


def load_crosscoder_decoder(width: str = "262k", l0: str = "medium",
                             device: str = "cpu") -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Load crosscoder decoder weights.

    Returns:
        W_dec: (n_features, n_output_layers, hidden_size) — decoder directions
        b_dec: list of (hidden_size,) per layer — decoder biases
    """
    layers_str = "_".join(str(l) for l in CROSSCODER_LAYERS)
    prefix = f"crosscoder/layer_{layers_str}_width_{width}_l0_{l0}"

    # Load W_dec from first layer file (it's the full decoder, same in all files)
    path = hf_hub_download(REPO_ID, f"{prefix}/params_layer_0.safetensors")
    with safe_open(path, framework="pt", device=device) as f:
        W_dec = f.get_tensor("w_dec").float()  # (n_features, n_output_layers, hidden_size)

    b_dec = []
    for i in range(len(CROSSCODER_LAYERS)):
        path = hf_hub_download(REPO_ID, f"{prefix}/params_layer_{i}.safetensors")
        with safe_open(path, framework="pt", device=device) as f:
            b_dec.append(f.get_tensor("b_dec").float())

    print(f"  W_dec: {W_dec.shape}")
    print(f"  b_dec: {[b.shape for b in b_dec]}")
    return W_dec, b_dec


def get_top_features(scores_path: str, lang: str, top_k: int) -> list[int]:
    """Get top-k feature indices for a language."""
    df = pd.read_csv(scores_path)
    lang_df = df[(df["lang"] == lang) & (df["rank"] <= top_k)]
    return lang_df.sort_values("rank")["feature_idx"].tolist()


def compute_steering_vectors(
    W_dec: torch.Tensor,
    source_features: list[int],
    target_features: list[int],
    scale: float,
) -> dict[int, torch.Tensor]:
    """Compute per-layer steering vectors from crosscoder decoder directions.

    Returns: {layer_idx: steering_vector of shape (hidden_size,)}
    """
    vectors = {}
    for layer_pos, layer_idx in enumerate(CROSSCODER_LAYERS):
        # Sum target feature directions, subtract source feature directions
        delta = torch.zeros(W_dec.shape[2], device=W_dec.device)

        for f_idx in target_features:
            delta += scale * W_dec[f_idx, layer_pos, :]

        for f_idx in source_features:
            delta -= scale * W_dec[f_idx, layer_pos, :]

        vectors[layer_idx] = delta

    return vectors


class MultiLayerSteeringHook:
    """Hook that adds a pre-computed steering vector to the residual stream."""

    def __init__(self, steering_vector: torch.Tensor):
        self.steering_vector = steering_vector
        self.enabled = False

    def __call__(self, module, input, output):
        if not self.enabled:
            return output

        hidden = output[0]
        rest = output[1:]

        # Add steering vector to all positions (or just last for efficiency)
        modified = hidden.clone()
        modified[:, -1:, :] += self.steering_vector.to(hidden.device, hidden.dtype).unsqueeze(0).unsqueeze(0)

        return (modified,) + rest


def main():
    parser = argparse.ArgumentParser(description="Validate crosscoder language steering")
    parser.add_argument("--model", default="google/gemma-3-27b-pt")
    parser.add_argument("--feature-scores", required=True)
    parser.add_argument("--cc-width", default="262k")
    parser.add_argument("--cc-l0", default="medium")
    parser.add_argument("--source-lang", default="eng")
    parser.add_argument("--target-lang", default="fin")
    parser.add_argument("--target-langs", nargs="*",
                        help="Multiple target languages (overrides --target-lang)")
    parser.add_argument("--prompt", default="The most important thing in life is")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--scale", type=float, default=3.0,
                        help="Multiplier for steering vectors (start low, increase)")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--n-samples", type=int, default=3)
    args = parser.parse_args()

    # Load model
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    device = next(model.parameters()).device
    print(f"Model on {device}")

    # Load crosscoder decoder
    print(f"\nLoading crosscoder decoder...")
    W_dec, b_dec = load_crosscoder_decoder(args.cc_width, args.cc_l0, device="cpu")

    # Get source features
    source_feats = get_top_features(args.feature_scores, args.source_lang, args.top_k)
    print(f"\nSource ({args.source_lang}) features: {source_feats}")

    target_langs = args.target_langs or [args.target_lang]

    for target_lang in target_langs:
        target_feats = get_top_features(args.feature_scores, target_lang, args.top_k)

        # Compute steering vectors for all 4 layers
        steer_vecs = compute_steering_vectors(W_dec, source_feats, target_feats, args.scale)

        print(f"\n{'='*60}")
        print(f"Steering: {args.source_lang} -> {target_lang}")
        print(f"Target features: {target_feats}")
        print(f"Scale: {args.scale}, Top-k: {args.top_k}")
        print(f"Steering vector norms per layer:")
        for layer_idx, vec in steer_vecs.items():
            print(f"  layer {layer_idx}: ||v|| = {vec.norm():.1f}")
        print(f"Prompt: {args.prompt!r}")
        print(f"{'='*60}")

        # Register hooks at all 4 crosscoder layers
        hooks_list = []
        hook_handles = []
        for layer_idx in CROSSCODER_LAYERS:
            hook = MultiLayerSteeringHook(steer_vecs[layer_idx])
            handle = model.model.language_model.layers[layer_idx].register_forward_hook(hook)
            hooks_list.append(hook)
            hook_handles.append(handle)

        try:
            input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)

            for i in range(args.n_samples):
                # Baseline
                for h in hooks_list:
                    h.enabled = False
                with torch.no_grad():
                    base_ids = model.generate(
                        input_ids, max_new_tokens=args.max_tokens,
                        do_sample=True, temperature=0.8, top_p=0.95,
                    )
                baseline = tokenizer.decode(base_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

                # Steered
                for h in hooks_list:
                    h.enabled = True
                with torch.no_grad():
                    steer_ids = model.generate(
                        input_ids, max_new_tokens=args.max_tokens,
                        do_sample=True, temperature=0.8, top_p=0.95,
                    )
                steered = tokenizer.decode(steer_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

                print(f"\n  Sample {i+1}:")
                print(f"  Baseline: {baseline[:200]}")
                print(f"  Steered:  {steered[:200]}")

        finally:
            for h in hook_handles:
                h.remove()

    print(f"\nDone.")


if __name__ == "__main__":
    main()
