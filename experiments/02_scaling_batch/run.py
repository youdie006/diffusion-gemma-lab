# EXP D: 디퓨전 속도 이점은 모델 크기·배치 크기에 따라 어떻게 변하는가
#
# EXP C(01_sampler_dynamics)에서 토이 모델의 캔버스 forward가 1토큰 디코드의 ~1.3배
# 비용임을 측정했다. 벤더의 "4~6배" 주장은 우리 토이 실측(12~16스텝에서 12~16배)보다
# 낮은데, 가설은 두 가지다:
#   (1) 모델이 커지면 연산 비중이 커져 캔버스 forward가 상대적으로 비싸진다
#   (2) 배치가 커지면 AR 디코드도 연산 효율이 좋아져 이점이 줄어든다
# 이 실험은 같은 DiffusionGemma 코드로 크기/배치를 스윕해 두 가설을 검증한다.
#
# 실행: .venv/bin/python experiments/02_scaling_batch/run.py

import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

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
# 하드웨어 사양은 공개하지 않는다 - 구체적 모델명을 결과물에 남기지 말 것
RESULTS = {"device": "consumer GPU (model undisclosed)"}

CANVAS = 256
VOCAB = 32768


def build(hidden, layers=10, seed=0):
    torch.manual_seed(seed)
    # GQA 제약: kv heads가 attention heads를 나눠야 함 -> heads는 짝수, kv = heads/2
    heads = max(2, hidden // 128 // 2 * 2)
    text_cfg = DiffusionGemmaTextConfig(
        vocab_size=VOCAB, hidden_size=hidden, intermediate_size=hidden * 2,
        num_hidden_layers=layers, num_attention_heads=heads,
        num_key_value_heads=heads // 2, head_dim=64, global_head_dim=64,
        num_global_key_value_heads=heads // 2, sliding_window=128,
        max_position_embeddings=8192, num_experts=4, top_k_experts=2,
        # grouped_mm이 16바이트 정렬을 요구하므로 expert 차원은 16의 배수로
        moe_intermediate_size=max(32, hidden // 3 // 16 * 16),
    )
    vision_cfg = {"model_type": "gemma4_vision", "hidden_size": 32, "intermediate_size": 64,
                  "num_hidden_layers": 1, "num_attention_heads": 2, "image_size": 28, "patch_size": 14}
    cfg = DiffusionGemmaConfig(text_config=text_cfg, vision_config=vision_cfg,
                               image_token_id=VOCAB - 1, boi_token_id=VOCAB - 2, eoi_token_id=VOCAB - 3)
    return DiffusionGemmaForBlockDiffusion(cfg).to(DEVICE).eval()


def t_canvas_marginal(model, bs):
    """디노이징 스텝 1회(캔버스 256토큰 병렬 forward)의 한계비용을 2점법으로 측정."""
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
    """같은 가중치 인코더(causal)로 1토큰 디코드 1회 비용을 측정 (AR 프록시)."""
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
    """GPU 클럭 변동 노이즈를 줄이기 위해 3회 측정 중앙값 사용."""
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    xs = [r["params_m"] for r in rows]
    ax1.plot(xs, [r["cost_ratio"] for r in rows], "o-")
    ax1.set_xlabel("model size (M params)")
    ax1.set_ylabel("canvas forward / 1-token decode cost")
    ax1.set_title("cost ratio vs size: no clear trend at toy scale")
    ax2.plot(xs, [r["speedup_at_16"] for r in rows], "o-", color="tab:green")
    ax2.set_xlabel("model size (M params)")
    ax2.set_ylabel("speedup vs AR at 16 steps")
    ax2.set_title("speedup at 16 steps stays ~13-15x (17M-336M)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_d_size.png"), dpi=120)
    plt.close(fig)


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

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r["batch"] for r in rows], [r["speedup_at_16"] for r in rows], "o-")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("batch size")
    ax.set_ylabel("speedup vs AR at 16 steps")
    ax.set_title("EXP D: diffusion advantage vs batch size (toy scale)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_d_batch.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    sweep_size()
    sweep_batch()
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(RESULTS, f, indent=2, ensure_ascii=False)
    print("done.")
