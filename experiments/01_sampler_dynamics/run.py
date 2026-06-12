# DiffusionGemma sampler dynamics experiments (consumer-GPU friendly).
#
# The real 26B needs ~18GB VRAM even quantized, beyond the GPU used here. Instead we
# plug tiny models into the REAL Transformers generation code and instrument the
# algorithm's dynamics. The subject is sampler behavior, not text quality.
#
# EXP A: untrained (random-weight) model through the real generate() loop
# EXP B: real sampler/stopping classes driven by synthetic confidence-ramp logits
# EXP C: measured speed model - canvas-parallel forward vs AR 1-token decode
#
# All measured data is saved to results.json; figures are rendered by plot.py
# (so styling can be iterated without re-measuring).
#
# Run: .venv/bin/python experiments/01_sampler_dynamics/run.py

import json
import math
import os
import time

import torch

from transformers import (
    DiffusionGemmaConfig,
    DiffusionGemmaForBlockDiffusion,
    DiffusionGemmaTextConfig,
)
from transformers.models.diffusion_gemma.generation_diffusion_gemma import (
    EntropyBoundSampler,
    EntropyBoundSamplerConfig,
    LinearTemperatureScheduleLogitsProcessor,
    StableAndConfidentStoppingCriteria,
)

import plot

HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Hardware specs are intentionally not disclosed in any artifact.
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
# EXP A: instrument the sampler while an untrained model runs the real generate()
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
    RESULTS["exp_a"] = {
        "steps_used": steps_used,
        "max_steps": MAX_STEPS,
        "early_stopped": steps_used < MAX_STEPS,
        "commits_per_step_mean": sum(log["commits"]) / steps_used,
        "commits_per_step": log["commits"],
        "mean_entropy_per_step": [round(e, 4) for e in log["mean_entropy"]],
        "uniform_entropy": math.log(1000),
        "tokens_per_forward_reported": float(out.tokens_per_forward[0]),
    }
    r = RESULTS["exp_a"]
    print(f"  steps used: {steps_used}/{MAX_STEPS}, commits/step: {r['commits_per_step_mean']:.2f}, "
          f"entropy {r['mean_entropy_per_step'][0]:.2f} -> {r['mean_entropy_per_step'][-1]:.2f} "
          f"(uniform={r['uniform_entropy']:.2f})")


# ---------------------------------------------------------------------------
# EXP B: drive the real sampler classes with synthetic logits of growing confidence
# ---------------------------------------------------------------------------

def synthetic_denoise(growth, entropy_bound=0.1, vocab=1000, seed=0,
                      t_min=0.4, t_max=0.8, conf_th=0.005, stab_th=1):
    """Drive the real EntropyBoundSampler / temperature schedule / stopping criteria
    with synthetic logits.

    Synthetic model: per-position difficulty d_i ~ U(0,1). At elapsed step e the target
    token logit margin is margin_i(e) = growth * e - 8 * d_i (larger margin = more
    confident; negative is effectively noise).
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
    RESULTS["exp_b_base"] = {
        "growth": 1.0, "steps_used": base["steps_used"],
        "tokens_per_forward": base["tokens_per_forward"],
        "commit_step": base["commit_step"].tolist(),
        "difficulty": [round(d, 4) for d in base["difficulty"].tolist()],
    }
    print(f"  growth=1.0: stopped at step {base['steps_used']}, "
          f"tokens/forward={base['tokens_per_forward']:.1f}")

    growths = [0.5, 1.0, 2.0, 4.0]
    steps = [synthetic_denoise(growth=gr)["steps_used"] for gr in growths]
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


# ---------------------------------------------------------------------------
# EXP C: canvas-parallel forward vs AR 1-token decode - measured speed model
# ---------------------------------------------------------------------------

def timed_generate(model, prompt, steps):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        # confidence_threshold=0.0 is rejected (validate requires float > 0; worse, its
        # error message references a nonexistent self.entropy_bound, raising AttributeError
        # - an upstream bug). 1e-9 effectively disables early stopping to pin the step count.
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
    xs, ys = torch.tensor(step_grid, dtype=torch.float), torch.tensor(times)
    t_canvas = float(((xs - xs.mean()) * (ys - ys.mean())).sum() / ((xs - xs.mean()) ** 2).sum())
    t_fixed = float(ys.mean() - t_canvas * xs.mean())

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
        res["canvas_vs_1tok_cost_ratio"] = round(t_canvas / t_ar_step, 2)
        res["speedup_at"] = {str(s): round((CANVAS * t_ar_step) / (t_fixed + s * t_canvas), 2)
                             for s in [12, 16, 48]}
    RESULTS["exp_c"] = res
    print(f"  {res['params_m']}M params: canvas forward {res['t_canvas_forward_ms']}ms"
          + (f", AR step {res['t_ar_decode_step_ms']}ms, speedup@12/16/48 = {res['speedup_at']}"
             if ar_time is not None else ""))


if __name__ == "__main__":
    exp_a()
    exp_b()
    exp_c()
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(RESULTS, f, indent=2, ensure_ascii=False)
    plot.render(RESULTS)
    print("done. results.json + figures/ written.")
