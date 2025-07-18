
import torch
from transformer_lens import HookedTransformer
import random
import numpy as np
import json
import time
import gc
import argparse
from datetime import datetime
from difflib import SequenceMatcher

# CLI setup
parser = argparse.ArgumentParser(description="Bumblebee - Transformer fuzzing toolkit")
parser.add_argument("--model", type=str, default="gpt2-small", help="Name of the model to load")
parser.add_argument("--fragment", action="store_true", help="Use Frontier Reasoning Attack on Generative Models via Entity Normalization Tampering (FRAGMENT)")
args = parser.parse_args()

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = HookedTransformer.from_pretrained(args.model, device=device)

# Prompts
if args.fragment:
    clean_prompt = "The doctor treated the Ρhοenιx with a"
    corrupted_prompt = "The butcher treated the Ρhοenιx with a"
else:
    clean_prompt = "The doctor treated the patient with a"
    corrupted_prompt = "The butcher treated the patient with a"

tokens_clean = model.to_tokens(clean_prompt, prepend_bos=True)
tokens_corrupt = model.to_tokens(corrupted_prompt, prepend_bos=True)

# Output setup
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f"buzzlens_results_{args.model.replace('/', '_')}_{timestamp}.json"
results = []

# Similarity function
def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

# Fuzzing functions
def gaussian_fuzz(activation, hook):
    noise_strength = 0.05
    noise = noise_strength * torch.randn_like(activation)
    return activation + noise

def bitflip_fuzz(activation, hook):
    flip_prob = 0.01
    mask = torch.rand_like(activation) < flip_prob
    return torch.where(mask, -activation, activation)

def dropout_fuzz(activation, hook):
    dropout_prob = 0.1
    return torch.nn.functional.dropout(activation, p=dropout_prob)

fuzz_methods = {
    "gaussian": gaussian_fuzz,
    "bitflip": bitflip_fuzz,
    "dropout": dropout_fuzz
}

hook_targets = ["hook_resid_pre", "hook_mlp_out", "hook_attn_out"]

# Run fuzzing
for layer in range(model.cfg.n_layers):
    for method_name, fuzz_fn in fuzz_methods.items():
        for hook_target in hook_targets:
            hook_name = f"blocks.{layer}.{hook_target}"
            model.reset_hooks()
            model.add_hook(hook_name, fuzz_fn)

            # Run inference
            try:
                logits_clean = model(tokens_clean)
                pred_clean = model.to_string(logits_clean.argmax(dim=-1)[0])
                logits_corrupt = model(tokens_corrupt)
                pred_corrupt = model.to_string(logits_corrupt.argmax(dim=-1)[0])
                sim_score = similarity(pred_clean, pred_corrupt)
            except RuntimeError as e:
                pred_clean = pred_corrupt = f"OOM: {str(e)}"
                sim_score = 0.0

            results.append({
                "layer": layer,
                "hook": hook_target,
                "method": method_name,
                "clean_prediction": pred_clean,
                "corrupted_prediction": pred_corrupt,
                "similarity_score": sim_score
            })

            model.reset_hooks()
            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(0.2)

# Save
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"Results saved to {output_path}")
