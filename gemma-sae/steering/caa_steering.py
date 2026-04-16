"""Contrastive Activation Addition (CAA) for language steering.

Simplest possible activation steering — no SAE decomposition.
  1. Run source-language text, record mean residual stream at target layer(s)
  2. Run target-language text, record mean residual stream at same layer(s)
  3. Steering vector = mean(target) - mean(source)
  4. During generation, add scaled vector to residual stream

If this doesn't switch language, the problem is deeper than SAE decomposition.

Usage:
    PYTHONPATH=. python -m steering.caa_steering \
        --model google/gemma-3-27b-pt \
        --flores-dir data/probes/flores_200 \
        --source-lang eng --target-langs fin zho ara \
        --layers 25 30 35 40 45 50 55 60 \
        --scales 1 3 5 10 20 \
        --max-sentences 100 \
        --prompt "The most important thing in life is" \
        --n-samples 2 --max-tokens 50
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_flores_texts(flores_dir: Path, lang: str, max_sentences: int) -> list[str]:
    """Load sentences from Flores JSONL."""
    texts = []
    with open(flores_dir / f"{lang}.jsonl") as f:
        for i, line in enumerate(f):
            if i >= max_sentences:
                break
            texts.append(json.loads(line)["text"])
    return texts


def compute_mean_activations(
    model, tokenizer, texts: list[str], layers: list[int],
    device: torch.device, batch_size: int = 8,
) -> dict[int, torch.Tensor]:
    """Forward pass on texts, return mean residual stream activation per layer."""
    # Set up hooks
    captured: dict[int, list[torch.Tensor]] = {l: [] for l in layers}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # Pool to last non-padding token per sentence
            captured[layer_idx].append(output[0].detach().float().cpu())
        return hook_fn

    hooks = []
    for l in layers:
        h = model.model.language_model.layers[l].register_forward_hook(make_hook(l))
        hooks.append(h)

    try:
        for batch_start in range(0, len(texts), batch_size):
            batch = texts[batch_start:batch_start + batch_size]
            encoded = tokenizer(batch, return_tensors="pt", padding=True,
                                truncation=True, max_length=256).to(device)
            with torch.no_grad():
                model(**encoded)

            # Extract last-token activations
            mask = encoded["attention_mask"].cpu()
            lengths = mask.sum(dim=1) - 1

            for l in layers:
                hidden = captured[l][-1]  # (batch, seq, hidden)
                last_tok = torch.stack([hidden[i, lengths[i]] for i in range(len(batch))])
                captured[l][-1] = last_tok  # replace with pooled
    finally:
        for h in hooks:
            h.remove()

    # Compute means
    means = {}
    for l in layers:
        all_acts = torch.cat(captured[l], dim=0)  # (n_sentences, hidden)
        means[l] = all_acts.mean(dim=0)  # (hidden,)

    return means


class CAAHook:
    """Adds a steering vector to the residual stream during generation."""

    def __init__(self, steering_vector: torch.Tensor, scale: float):
        self.steering_vector = steering_vector
        self.scale = scale
        self.enabled = False

    def __call__(self, module, input, output):
        if not self.enabled:
            return output

        hidden = output[0]
        rest = output[1:]

        # Add to ALL token positions
        delta = self.steering_vector.to(hidden.device, hidden.dtype) * self.scale
        modified = hidden + delta.unsqueeze(0).unsqueeze(0)

        return (modified,) + rest


def main():
    parser = argparse.ArgumentParser(description="CAA language steering sanity check")
    parser.add_argument("--model", default="google/gemma-3-27b-pt")
    parser.add_argument("--flores-dir", default="data/probes/flores_200")
    parser.add_argument("--source-lang", default="eng")
    parser.add_argument("--target-lang", default="fin")
    parser.add_argument("--target-langs", nargs="*")
    parser.add_argument("--layers", nargs="+", type=int,
                        default=[25, 30, 35, 40, 45, 50, 55, 60])
    parser.add_argument("--scales", nargs="+", type=float,
                        default=[1, 3, 5, 10, 20])
    parser.add_argument("--max-sentences", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--prompt", default="The most important thing in life is")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--n-samples", type=int, default=2)
    args = parser.parse_args()

    target_langs = args.target_langs or [args.target_lang]
    flores_dir = Path(args.flores_dir)

    # Load model
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    device = next(model.parameters()).device
    print(f"Model on {device}")

    # Compute source language mean activations
    print(f"\nComputing mean activations for {args.source_lang}...")
    source_texts = load_flores_texts(flores_dir, args.source_lang, args.max_sentences)
    source_means = compute_mean_activations(
        model, tokenizer, source_texts, args.layers, device, args.batch_size)
    print(f"  {len(source_texts)} sentences, layers {args.layers}")

    for target_lang in target_langs:
        print(f"\nComputing mean activations for {target_lang}...")
        target_texts = load_flores_texts(flores_dir, target_lang, args.max_sentences)
        target_means = compute_mean_activations(
            model, tokenizer, target_texts, args.layers, device, args.batch_size)

        # Compute steering vectors
        steer_vecs = {}
        for l in args.layers:
            steer_vecs[l] = target_means[l] - source_means[l]
            norm = steer_vecs[l].norm().item()
            cos_sim = torch.nn.functional.cosine_similarity(
                target_means[l].unsqueeze(0), source_means[l].unsqueeze(0)).item()
            print(f"  Layer {l}: ||steer|| = {norm:.1f}, "
                  f"cos(src,tgt) = {cos_sim:.3f}")

        # Test each layer × scale combination
        input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)

        for layer in args.layers:
            for scale in args.scales:
                hook = CAAHook(steer_vecs[layer], scale)
                handle = model.model.language_model.layers[layer].register_forward_hook(hook)

                print(f"\n{'='*60}")
                print(f"CAA: {args.source_lang} -> {target_lang} | "
                      f"layer={layer}, scale={scale}")
                print(f"{'='*60}")

                try:
                    for i in range(args.n_samples):
                        # Baseline (only on first layer/scale combo)
                        if layer == args.layers[0] and scale == args.scales[0]:
                            hook.enabled = False
                            with torch.no_grad():
                                base_ids = model.generate(
                                    input_ids, max_new_tokens=args.max_tokens,
                                    do_sample=True, temperature=0.8, top_p=0.95,
                                )
                            baseline = tokenizer.decode(
                                base_ids[0][input_ids.shape[1]:],
                                skip_special_tokens=True)
                            print(f"  Baseline: {baseline[:200]}")

                        # Steered
                        hook.enabled = True
                        with torch.no_grad():
                            steer_ids = model.generate(
                                input_ids, max_new_tokens=args.max_tokens,
                                do_sample=True, temperature=0.8, top_p=0.95,
                            )
                        steered = tokenizer.decode(
                            steer_ids[0][input_ids.shape[1]:],
                            skip_special_tokens=True)
                        print(f"  Steered[{i+1}]: {steered[:200]}")

                finally:
                    handle.remove()

    print(f"\nDone.")


if __name__ == "__main__":
    main()
