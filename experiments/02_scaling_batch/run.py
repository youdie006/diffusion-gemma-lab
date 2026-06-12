# EXP D: how the diffusion speed advantage changes with model size and batch size.
#
# EXP C (01_sampler_dynamics) measured a toy model's canvas forward at ~1.3x the cost of
# a 1-token decode. The vendor's "4-6x" claim is lower than our toy result (12-16x at
# 12-16 steps); two hypotheses:
#   (1) bigger models shift toward compute-bound, making the canvas forward pricier
#   (2) bigger batches make AR decode more compute-efficient, shrinking the advantage
# This experiment sweeps size/batch with the same DiffusionGemma code to test both.
#
# Run: .venv/bin/python experiments/02_scaling_batch/run.py

import json
import os
import time

import torch

import plot

from transformers import (
    DiffusionGemmaConfig,
    DiffusionGemmaForBlockDiffusion,
    DiffusionGemmaTextConfig,
    DynamicCache,
)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
DEVICE = "cuda"
# Hardware specs are intentionally not disclosed in any artifact.
RESULTS = {"device": "consumer GPU (model undisclosed)"}

CANVAS = 256
VOCAB = 32768


def build(hidden, layers=10, seed=0):
    torch.manual_seed(seed)
    # GQA constraint: kv heads must divide attention heads -> even heads, kv = heads/2
    heads = max(2, hidden // 128 // 2 * 2)
    text_cfg = DiffusionGemmaTextConfig(
        vocab_size=VOCAB, hidden_size=hidden, intermediate_size=hidden * 2,
        num_hidden_layers=layers, num_attention_heads=heads,
        num_key_value_heads=heads // 2, head_dim=64, global_head_dim=64,
        num_global_key_value_heads=heads // 2, sliding_window=128,
        max_position_embeddings=8192, num_experts=4, top_k_experts=2,
        # grouped_mm requires 16-byte-aligned strides -> expert dim multiple of 16
        moe_intermediate_size=max(32, hidden // 3 // 16 * 16),
    )
    vision_cfg = {"model_type": "gemma4_vision", "hidden_size": 32, "intermediate_size": 64,
                  "num_hidden_layers": 1, "num_attention_heads": 2, "image_size": 28, "patch_size": 14}
    cfg = DiffusionGemmaConfig(text_config=text_cfg, vision_config=vision_cfg,
                               image_token_id=VOCAB - 1, boi_token_id=VOCAB - 2, eoi_token_id=VOCAB - 3)
    return DiffusionGemmaForBlockDiffusion(cfg).to(DEVICE).eval()


def t_canvas_marginal(model, bs):
    """Two-point estimate of the marginal cost of one denoising step (256-token canvas forward)."""
    prompt = torch.randint(2, VOCAB - 10, (bs, 64), device=DEVICE)

    def gen_time(steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(prompt, max_new_tokens=CANVAS, max_denoising_steps=steps,
                           confidence_threshold=1e-9)
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    gen_time(4)  # warmup
    return (gen_time(24) - gen_time(8)) / 16


def t_ar_step(model, bs):
    """Measure one 1-token decode with the same-weight causal encoder (AR proxy)."""
    enc = model.model.encoder
    with torch.no_grad():
        cache = DynamicCache()
        ids = torch.randint(2, VOCAB - 10, (bs, 64), device=DEVICE)
        enc(input_ids=ids, past_key_values=cache, use_cache=True)
        tok = torch.randint(2, VOCAB - 10, (bs, 1), device=DEVICE)
        for _ in range(8):
            enc(input_ids=tok, past_key_values=cache, use_cache=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(64):
            enc(input_ids=tok, past_key_values=cache, use_cache=True)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / 64


def speedup(tc, ta, steps):
    return (CANVAS * ta) / (steps * tc)


def median3(fn, *args):
    """Median of 3 runs to reduce GPU clock jitter."""
    vals = sorted(fn(*args) for _ in range(3))
    return vals[1]


def sweep_size():
    print("[D1] model size sweep (bs=1)")
    rows = []
    for hidden in [256, 512, 768, 1152, 1536]:
        model = build(hidden)
        n = sum(p.numel() for p in model.parameters()) / 1e6
        tc, ta = median3(t_canvas_marginal, model, 1), median3(t_ar_step, model, 1)
        rows.append({"hidden": hidden, "params_m": round(n, 1),
                     "t_canvas_ms": round(tc * 1e3, 2), "t_ar_ms": round(ta * 1e3, 3),
                     "cost_ratio": round(tc / ta, 2),
                     "speedup_at_16": round(speedup(tc, ta, 16), 1)})
        print(f"  {rows[-1]}")
        del model
        torch.cuda.empty_cache()
    RESULTS["size_sweep"] = rows



def sweep_batch():
    print("[D2] batch size sweep (hidden=768)")
    model = build(768)
    rows = []
    for bs in [1, 2, 4, 8, 16]:
        tc, ta = median3(t_canvas_marginal, model, bs), median3(t_ar_step, model, bs)
        rows.append({"batch": bs, "t_canvas_ms": round(tc * 1e3, 2),
                     "t_ar_ms": round(ta * 1e3, 3), "cost_ratio": round(tc / ta, 2),
                     "speedup_at_16": round(speedup(tc, ta, 16), 1)})
        print(f"  {rows[-1]}")
    RESULTS["batch_sweep"] = rows



if __name__ == "__main__":
    sweep_size()
    sweep_batch()
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(RESULTS, f, indent=2, ensure_ascii=False)
    plot.render(RESULTS)
    print("done.")
