"""Validate that language-specific SAE features are causally controllable.

Test: give an English prompt, clamp target language features ON and
source language features OFF at a specific layer, check if generation
switches language.

Usage:
    PYTHONPATH=. python -m steering.validate_language_steering \
        --model google/gemma-3-27b-pt \
        --layer 40 \
        --sae-width 256k --sae-l0 medium \
        --feature-scores data/feature_scores/layer_40/monolinguality.csv \
        --source-lang eng --target-lang fin \
        --prompt "The most important thing in life is" \
        --top-k 2 --clamp-value 500 \
        --max-tokens 50
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_sae(layer: int, width: str, l0: str):
    """Load SAE via sae_lens."""
    from sae_lens import SAE
    width_map = {"16k": "16k", "64k": "65k", "256k": "262k", "1m": "1m"}
    release = "gemma-scope-2-27b-pt-res"
    sae_id = f"layer_{layer}_width_{width_map[width]}_l0_{l0}"
    print(f"Loading SAE: {release} / {sae_id}")
    sae = SAE.from_pretrained(release=release, sae_id=sae_id)
    sae.eval()
    return sae


def get_top_features(scores_path: str, lang: str, top_k: int) -> list[int]:
    """Get top-k feature indices for a language from monolinguality scores."""
    df = pd.read_csv(scores_path)
    lang_df = df[(df["lang"] == lang) & (df["rank"] <= top_k)]
    return lang_df.sort_values("rank")["feature_idx"].tolist()


class LanguageSteeringHook:
    """Hook that modifies SAE feature activations during forward pass."""

    def __init__(self, sae, source_features: list[int], target_features: list[int],
                 clamp_value: float, device: torch.device):
        self.sae = sae.to(device)
        self.source_features = source_features
        self.target_features = target_features
        self.clamp_value = clamp_value
        self.device = device
        self.enabled = False

    def __call__(self, module, input, output):
        if not self.enabled:
            return output

        hidden = output[0]  # (batch, seq_len, hidden_size)
        rest = output[1:]

        # Only modify the last token position (during generation)
        original_shape = hidden.shape
        last_hidden = hidden[:, -1:, :]  # (batch, 1, hidden_size)

        # Encode through SAE
        flat = last_hidden.reshape(-1, last_hidden.shape[-1]).float()
        with torch.no_grad():
            features = self.sae.encode(flat)

            # Ablate source language features (set to 0)
            for f_idx in self.source_features:
                features[:, f_idx] = 0.0

            # Clamp target language features to high value
            for f_idx in self.target_features:
                features[:, f_idx] = self.clamp_value

            # Decode back
            reconstructed = self.sae.decode(features)

        # Replace last token's hidden state
        modified = hidden.clone()
        modified[:, -1:, :] = reconstructed.reshape(1, 1, -1).to(hidden.dtype)

        return (modified,) + rest


def generate_with_steering(model, tokenizer, prompt: str, hook: LanguageSteeringHook,
                           max_tokens: int, device: torch.device) -> tuple[str, str]:
    """Generate text with and without steering, return both."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Baseline (no steering)
    hook.enabled = False
    with torch.no_grad():
        baseline_ids = model.generate(
            input_ids, max_new_tokens=max_tokens, do_sample=True,
            temperature=0.8, top_p=0.95,
        )
    baseline_text = tokenizer.decode(baseline_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

    # Steered
    hook.enabled = True
    with torch.no_grad():
        steered_ids = model.generate(
            input_ids, max_new_tokens=max_tokens, do_sample=True,
            temperature=0.8, top_p=0.95,
        )
    steered_text = tokenizer.decode(steered_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    hook.enabled = False

    return baseline_text, steered_text


def main():
    parser = argparse.ArgumentParser(description="Validate language steering via SAE features")
    parser.add_argument("--model", default="google/gemma-3-27b-pt")
    parser.add_argument("--layer", type=int, default=40)
    parser.add_argument("--sae-width", default="256k")
    parser.add_argument("--sae-l0", default="medium")
    parser.add_argument("--feature-scores", required=True)
    parser.add_argument("--source-lang", default="eng")
    parser.add_argument("--target-lang", default="fin")
    parser.add_argument("--prompt", default="The most important thing in life is")
    parser.add_argument("--top-k", type=int, default=2, help="Number of features to steer per language")
    parser.add_argument("--clamp-value", type=float, default=500.0)
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--n-samples", type=int, default=3, help="Number of generations per condition")
    parser.add_argument("--target-langs", nargs="*",
                        help="Test multiple target languages (overrides --target-lang)")
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

    # Load SAE
    sae = load_sae(args.layer, args.sae_width, args.sae_l0)

    # Get feature indices
    source_feats = get_top_features(args.feature_scores, args.source_lang, args.top_k)
    print(f"\nSource ({args.source_lang}) features: {source_feats}")

    target_langs = args.target_langs or [args.target_lang]

    for target_lang in target_langs:
        target_feats = get_top_features(args.feature_scores, target_lang, args.top_k)
        print(f"\n{'='*60}")
        print(f"Steering: {args.source_lang} -> {target_lang}")
        print(f"Target features: {target_feats}")
        print(f"Clamp value: {args.clamp_value}, Layer: {args.layer}")
        print(f"Prompt: {args.prompt!r}")
        print(f"{'='*60}")

        # Set up hook
        hook = LanguageSteeringHook(sae, source_feats, target_feats, args.clamp_value, device)
        handle = model.model.language_model.layers[args.layer].register_forward_hook(hook)

        try:
            for i in range(args.n_samples):
                baseline, steered = generate_with_steering(
                    model, tokenizer, args.prompt, hook, args.max_tokens, device,
                )
                print(f"\n  Sample {i+1}:")
                print(f"  Baseline:  {baseline[:200]}")
                print(f"  Steered:   {steered[:200]}")
        finally:
            handle.remove()

    print(f"\nDone.")


if __name__ == "__main__":
    main()
