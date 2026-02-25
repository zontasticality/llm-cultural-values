"""Quick test: can we load Qwen3-235B-A22B-GPTQ-Int4 on multi-GPU and get logits?"""
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-235B-A22B-GPTQ-Int4"

print(f"CUDA devices: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)} "
          f"({torch.cuda.get_device_properties(i).total_mem / 1e9:.1f} GB)")

print(f"\nLoading tokenizer: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print(f"Loading model: {MODEL_ID}")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
)
model.eval()
print(f"Model loaded in {time.time() - t0:.1f}s")
print(f"Device map summary: {set(model.hf_device_map.values())}")

# Test forward pass
prompt = "On a scale of 1 to 4: How important is work in your life?\n1. Very important\n2. Quite important\n3. Not very important\n4. Not at all important\nAnswer:"
inputs = tokenizer(prompt, return_tensors="pt")
# Move to first device
inputs = {k: v.to(model.device) for k, v in inputs.items()}

print(f"\nRunning forward pass (input length: {inputs['input_ids'].shape[1]} tokens)...")
t0 = time.time()
with torch.no_grad():
    outputs = model(**inputs)
logits = outputs.logits[0, -1, :]  # Last token logits
print(f"Forward pass took {time.time() - t0:.2f}s")
print(f"Logits shape: {logits.shape}, dtype: {logits.dtype}")

# Check logprobs for digit tokens
log_probs = torch.log_softmax(logits.float(), dim=-1)
for digit in ["1", "2", "3", "4", " 1", " 2", " 3", " 4"]:
    token_ids = tokenizer.encode(digit, add_special_tokens=False)
    for tid in token_ids:
        lp = log_probs[tid].item()
        print(f"  token '{digit}' (id={tid}): logprob={lp:.4f} prob={torch.exp(torch.tensor(lp)).item():.4f}")

print("\nSUCCESS: Model loads and produces logits on multi-GPU.")
