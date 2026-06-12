# DiffusionGemma 샘플러 동역학 실험 (소비자급 GPU용)
#
# 실물 26B 모델은 양자화로도 18GB VRAM이 필요해 보유한 소비자급 GPU에서 구동 불가.
# 대신 Transformers의 "실제 DiffusionGemma 생성 코드"에 초소형 모델을 끼워
# 알고리즘 동역학을 계측한다. 텍스트 품질이 아니라 샘플러의 거동이 측정 대상.
#
# EXP A: 무학습(랜덤 가중치) 모델 + 실제 generate() - 확신 없는 모델에서의 거동
# EXP B: 실제 샘플러/정지 클래스 + 합성 logits - 확신이 차오를 때의 커밋 웨이브
# EXP C: 실제 모델 forward 실측 - 캔버스 병렬 forward vs AR 1토큰 디코드 속도 모델
#
# 실행: .venv/bin/python experiments/01_sampler_dynamics/run.py

import json
import math
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
)
from transformers.models.diffusion_gemma import generation_diffusion_gemma as gen_mod
from transformers.models.diffusion_gemma.generation_diffusion_gemma import (
    EntropyBoundSampler,
    EntropyBoundSamplerConfig,
    LinearTemperatureScheduleLogitsProcessor,
    StableAndConfidentStoppingCriteria,
)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 하드웨어 사양은 공개하지 않는다 - 구체적 모델명을 결과물에 남기지 말 것
RESULTS = {"device": "consumer GPU (model undisclosed)"}

CANVAS = 256
MAX_STEPS = 48


def tiny_model(vocab=1000, hidden=64, layers=2, seed=0):
    torch.manual_seed(seed)
    text_cfg = DiffusionGemmaTextConfig(
        vocab_size=vocab, hidden_size=hidden, intermediate_size=hidden * 2,
        num_hidden_layers=layers, num_attention_heads=2, num_key_value_heads=1,
        head_dim=32, global_head_dim=32, num_global_key_value_heads=1,
        sliding_window=128, max_position_embeddings=8192,
        num_experts=4, top_k_experts=2, moe_intermediate_size=32,
    )
    vision_cfg = {"model_type": "gemma4_vision", "hidden_size": 32, "intermediate_size": 64,
                  "num_hidden_layers": 1, "num_attention_heads": 2, "image_size": 28, "patch_size": 14}
    cfg = DiffusionGemmaConfig(text_config=text_cfg, vision_config=vision_cfg,
                               image_token_id=vocab - 1, boi_token_id=vocab - 2, eoi_token_id=vocab - 3)
    return DiffusionGemmaForBlockDiffusion(cfg).to(DEVICE).eval()


# ---------------------------------------------------------------------------
# EXP A: 무학습 모델을 실제 generate() 루프에 통과시키며 샘플러를 계측
# ---------------------------------------------------------------------------

def exp_a():
    print("[EXP A] untrained model through the real generate() loop")
    log = {"commits": [], "mean_entropy": []}

    orig_accept = EntropyBoundSampler.accept_canvas

    def patched_accept(self, current_canvas, denoiser_canvas, logits, cur_step):
        out = orig_accept(self, current_canvas, denoiser_canvas, logits, cur_step)
        ent = torch.distributions.Categorical(logits=logits).entropy()
        log["commits"].append(int(self.accepted_token_mask.sum()))
        log["mean_entropy"].append(float(ent.mean()))
        return out

    model = tiny_model()
    prompt = torch.randint(2, 900, (1, 16), device=DEVICE)
    EntropyBoundSampler.accept_canvas = patched_accept
    try:
        with torch.no_grad():
            out = model.generate(prompt, max_new_tokens=CANVAS, max_denoising_steps=MAX_STEPS,
                                 return_dict_in_generate=True)
    finally:
        EntropyBoundSampler.accept_canvas = orig_accept

    steps_used = len(log["commits"])
    res = {
        "steps_used": steps_used,
        "max_steps": MAX_STEPS,
        "early_stopped": steps_used < MAX_STEPS,
        "commits_per_step_mean": sum(log["commits"]) / steps_used,
        "mean_entropy_first": log["mean_entropy"][0],
        "mean_entropy_last": log["mean_entropy"][-1],
        "uniform_entropy": math.log(1000),
        "tokens_per_forward_reported": float(out.tokens_per_forward[0]),
    }
    RESULTS["exp_a"] = res
    print(f"  steps used: {steps_used}/{MAX_STEPS}, commits/step: {res['commits_per_step_mean']:.2f}, "
          f"entropy {res['mean_entropy_first']:.2f} -> {res['mean_entropy_last']:.2f} "
          f"(uniform={res['uniform_entropy']:.2f})")

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.bar(range(1, steps_used + 1), log["commits"], color="tab:blue", label="committed tokens")
    ax1.set_xlabel("denoising step")
    ax1.set_ylabel("tokens committed this step", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(range(1, steps_used + 1), log["mean_entropy"], color="tab:red", label="mean entropy")
    ax2.axhline(math.log(1000), color="tab:red", ls=":", lw=1)
    ax2.set_ylabel("mean token entropy (nats)", color="tab:red")
    ax1.set_title("EXP A: untrained model - commits per step (real generate loop)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_a_untrained.png"), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# EXP B: 합성 logits로 "확신이 차오르는 모델"을 흉내내며 실제 샘플러 클래스 구동
# ---------------------------------------------------------------------------

def synthetic_denoise(growth, entropy_bound=0.1, vocab=1000, seed=0,
                      t_min=0.4, t_max=0.8, conf_th=0.005, stab_th=1):
    """실제 EntropyBoundSampler / 온도 스케줄 / 정지 기준을 합성 logits로 구동한다.

    합성 모델: 위치별 난이도 d_i ~ U(0,1). 경과 스텝 e에서 목표 토큰의 logit 마진은
    margin_i(e) = growth * e - 8 * d_i  (마진이 클수록 확신, 음수면 사실상 노이즈)
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    target = torch.randint(0, vocab, (1, CANVAS), generator=g)
    difficulty = torch.rand(1, CANVAS, generator=g)

    sampler = EntropyBoundSampler(EntropyBoundSamplerConfig(entropy_bound=float(entropy_bound)),
                                  canvas_length=CANVAS, vocab_size=vocab, max_denoising_steps=MAX_STEPS)
    temp = LinearTemperatureScheduleLogitsProcessor(t_min=t_min, t_max=t_max, max_denoising_steps=MAX_STEPS)
    stop = StableAndConfidentStoppingCriteria(stability_threshold=stab_th, confidence_threshold=conf_th)

    canvas = sampler.initialize_canvas(1, torch.device("cpu"))
    commit_step = torch.full((CANVAS,), -1, dtype=torch.int)
    commits, entropies = [], []
    steps_used = MAX_STEPS

    for cur_step in reversed(range(1, MAX_STEPS + 1)):
        elapsed = MAX_STEPS - cur_step
        margin = growth * elapsed - 8.0 * difficulty  # (1, CANVAS)
        logits = torch.zeros(1, CANVAS, vocab)
        logits.scatter_(-1, target.unsqueeze(-1), margin.unsqueeze(-1).clamp(min=0.0))

        processed = temp(None, logits, cur_step)
        probs = torch.softmax(processed, dim=-1)
        denoiser_canvas = torch.multinomial(probs.view(-1, vocab), 1).view(1, CANVAS)
        canvas = sampler.accept_canvas(canvas, denoiser_canvas, processed, cur_step)

        newly = sampler.accepted_token_mask[0] & (commit_step == -1)
        commit_step[newly] = elapsed
        commits.append(int(sampler.accepted_token_mask.sum()))
        entropies.append(float(torch.distributions.Categorical(logits=processed).entropy().mean()))

        argmax_canvas = processed.argmax(-1)
        if bool(stop(argmax_canvas, processed)[0]):
            steps_used = elapsed + 1
            break
        canvas = sampler.renoise_canvas(canvas, cur_step)

    return {"steps_used": steps_used, "commits": commits, "entropies": entropies,
            "commit_step": commit_step, "difficulty": difficulty[0],
            "tokens_per_forward": CANVAS / steps_used}


def exp_b():
    print("[EXP B] synthetic confidence ramp through the real sampler classes")
    base = synthetic_denoise(growth=1.0)
    RESULTS["exp_b_base"] = {"growth": 1.0, "steps_used": base["steps_used"],
                             "tokens_per_forward": base["tokens_per_forward"]}
    print(f"  growth=1.0: stopped at step {base['steps_used']}, "
          f"tokens/forward={base['tokens_per_forward']:.1f}")

    # 캔버스 수렴 히트맵: 각 위치가 몇 스텝째에 커밋됐는지 (난이도 순 정렬)
    order = torch.argsort(base["difficulty"])
    cs = base["commit_step"][order]
    grid = torch.zeros(base["steps_used"], CANVAS)
    for e in range(base["steps_used"]):
        grid[e] = (cs <= e) & (cs >= 0)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.imshow(grid.numpy(), aspect="auto", cmap="Blues", interpolation="nearest")
    ax.set_xlabel("canvas position (sorted by difficulty)")
    ax.set_ylabel("elapsed denoising step")
    ax.set_title(f"EXP B: canvas convergence (growth=1.0, stop at step {base['steps_used']})")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_b_heatmap.png"), dpi=120)
    plt.close(fig)

    # 스윕 1: 확신 상승 속도 vs 수렴 스텝
    growths = [0.5, 1.0, 2.0, 4.0]
    steps = [synthetic_denoise(growth=gr)["steps_used"] for gr in growths]
    # 스윕 2: entropy_bound vs 스텝당 평균 커밋 수
    bounds = [0.01, 0.1, 1.0, 10.0]
    bound_runs = [synthetic_denoise(growth=1.0, entropy_bound=b) for b in bounds]
    mean_commits = [sum(r["commits"]) / len(r["commits"]) for r in bound_runs]
    RESULTS["exp_b_sweep"] = {
        "growth": dict(zip(map(str, growths), steps)),
        "entropy_bound_mean_commits": dict(zip(map(str, bounds), [round(c, 1) for c in mean_commits])),
        "entropy_bound_steps": dict(zip(map(str, bounds), [r["steps_used"] for r in bound_runs])),
    }
    print(f"  growth sweep {growths} -> steps {steps}")
    print(f"  entropy_bound sweep {bounds} -> commits/step {[round(c,1) for c in mean_commits]}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(growths, steps, "o-")
    ax1.set_xlabel("model confidence growth rate (synthetic)")
    ax1.set_ylabel("steps until adaptive stop")
    ax1.set_title("faster confidence -> fewer steps")
    ax2.semilogx(bounds, mean_commits, "o-")
    ax2.set_xlabel("entropy_bound")
    ax2.set_ylabel("mean tokens committed per step")
    ax2.set_title("entropy_bound is the aggressiveness knob")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_b_sweep.png"), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# EXP C: 캔버스 병렬 forward vs AR 1토큰 디코드 - 실측 기반 속도 모델
# ---------------------------------------------------------------------------

def timed_generate(model, prompt, steps):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        # confidence_threshold=0.0은 불가(validate가 float > 0 요구. 게다가 에러 메시지가
        # 존재하지 않는 self.entropy_bound를 참조해 AttributeError가 나는 upstream 버그 있음).
        # 1e-9로 사실상 조기 종료를 비활성화해 스텝 수를 고정한다.
        model.generate(prompt, max_new_tokens=CANVAS, max_denoising_steps=steps,
                       confidence_threshold=1e-9)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def exp_c():
    print("[EXP C] canvas-parallel forward vs sequential decode on this GPU")
    model = tiny_model(vocab=32768, hidden=768, layers=10, seed=0)
    n = sum(p.numel() for p in model.parameters())
    prompt = torch.randint(2, 30000, (1, 64), device=DEVICE)

    timed_generate(model, prompt, 4)  # warmup
    step_grid = [4, 8, 16, 32, 48]
    times = [timed_generate(model, prompt, s) for s in step_grid]
    # 선형 회귀로 스텝당 한계 비용(t_canvas)과 고정 비용 분리
    xs, ys = torch.tensor(step_grid, dtype=torch.float), torch.tensor(times)
    t_canvas = float(((xs - xs.mean()) * (ys - ys.mean())).sum() / ((xs - xs.mean()) ** 2).sum())
    t_fixed = float(ys.mean() - t_canvas * xs.mean())

    # AR 디코드 프록시: 같은 가중치의 인코더(causal)로 1토큰씩 256번 순차 forward
    from transformers import DynamicCache
    enc = model.model.encoder
    ar_time = None
    try:
        with torch.no_grad():
            cache = DynamicCache()
            ids = torch.randint(2, 30000, (1, 64), device=DEVICE)
            enc(input_ids=ids, past_key_values=cache, use_cache=True)  # prefill
            tok = torch.randint(2, 30000, (1, 1), device=DEVICE)
            for _ in range(8):  # warmup
                enc(input_ids=tok, past_key_values=cache, use_cache=True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(CANVAS):
                enc(input_ids=tok, past_key_values=cache, use_cache=True)
            torch.cuda.synchronize()
            ar_time = time.perf_counter() - t0
    except Exception as e:
        print(f"  AR proxy failed ({type(e).__name__}: {e}) - skipping speedup chart")

    res = {"params_m": round(n / 1e6, 1), "step_grid": step_grid,
           "gen_times_s": [round(t, 3) for t in times],
           "t_canvas_forward_ms": round(t_canvas * 1e3, 2), "t_fixed_ms": round(t_fixed * 1e3, 2)}
    if ar_time is not None:
        t_ar_step = ar_time / CANVAS
        res["t_ar_decode_step_ms"] = round(t_ar_step * 1e3, 3)
        res["canvas_vs_1tok_cost_ratio"] = round(t_canvas / t_ar_step, 1)
        res["speedup_at"] = {str(s): round((CANVAS * t_ar_step) / (t_fixed + s * t_canvas), 2)
                             for s in [12, 16, 48]}
    RESULTS["exp_c"] = res
    print(f"  {res['params_m']}M params: canvas forward {res['t_canvas_forward_ms']}ms"
          + (f", AR step {res['t_ar_decode_step_ms']}ms, speedup@12/16/48 = {res['speedup_at']}"
             if ar_time is not None else ""))

    if ar_time is not None:
        s_axis = list(range(2, 65, 2))
        speedup = [(CANVAS * t_ar_step) / (t_fixed + s * t_canvas) for s in s_axis]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(s_axis, speedup, "-")
        ax.axhline(1.0, color="gray", ls=":")
        ax.axvspan(12, 16, color="tab:green", alpha=0.15, label="vendor: typical 12-16 steps")
        ax.axvline(48, color="tab:red", ls="--", lw=1, label="max steps cap (48)")
        ax.set_xlabel("denoising steps per 256-token canvas")
        ax.set_ylabel("speedup vs AR decode (same weights, measured)")
        ax.set_title(f"EXP C: measured speed model, consumer GPU ({res['params_m']}M toy)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "fig_c_speedup.png"), dpi=120)
        plt.close(fig)


if __name__ == "__main__":
    exp_a()
    exp_b()
    exp_c()
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(RESULTS, f, indent=2, ensure_ascii=False)
    print("done. results.json + figures/ written.")
