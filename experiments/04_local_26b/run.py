# EXP F: the real 26B pair, run locally as far as consumer hardware allows.
#
# Neither 26B fits the GPU, but both fit system RAM as 4-bit GGUFs, so we run them
# CPU-only through llama.cpp:
#   - DiffusionGemma 26B-A4B, Q4_K_M GGUF (unsloth re-upload), via llama-diffusion-cli
#     from the (unmerged) llama.cpp PR #24423 that implements the diffusion-gemma
#     architecture and its entropy-bound decoder
#   - Gemma 4 26B-A4B, official QAT q4_0 GGUF, via llama-cli (the AR base model)
#
# Purpose: (1) prove the pair runs at all on consumer hardware, (2) anchor output quality
# against the real base model, (3) record CPU-only throughput with explicit caveats.
# CPU-only timings say nothing about the GPU speed claims (that remains cloud work).
#
# Run: .venv/bin/python experiments/04_local_26b/run.py

import json
import os
import re
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BIN_DIR = os.path.expanduser("~/builds/llama.cpp-dg/build/bin")
DIFF_CLI = os.path.join(BIN_DIR, "llama-diffusion-cli")
LLAMA_CLI = os.path.join(BIN_DIR, "llama-cli")

# Hardware specs are intentionally not disclosed in any artifact.
RESULTS = {
    "device": "consumer desktop, CPU-only llama.cpp build (specs undisclosed)",
    "llama_cpp": "PR #24423 head 10a2613 on top of master (depth-50 clone)",
    "runs": [],
}

PROMPTS = {
    "math": "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers "
            "per hour. How many kilometers can she run in 8 hours?",
    "korean": "대한민국의 수도는 어디인가요? 그 도시의 특징을 두 문장으로 설명해 주세요.",
    "code": "Write a Python function that checks whether a string is a palindrome.",
}


def gguf(repo, fn):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, fn)


def run_one(cmd, timeout=3600):
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    wall = time.perf_counter() - t0
    return p.stdout, p.stderr, wall


def parse_perf(stdout, stderr):
    """Extract throughput plus diffusion step stats from CLI output."""
    text = stdout + stderr
    perf = {}
    # diffusion-cli (entropy-bound mode):
    #   total time: Tms, time per step: Sms (K steps over B blocks, entropy-bound)
    #   throughput: X.X tok/s (N tok in Tms), in-step parallel Y tok/s ...
    m = re.search(r"throughput:\s*([\d.]+)\s*tok/s\s*\((\d+) tok", text)
    if m:
        perf["tok_per_s"], perf["tokens"] = float(m.group(1)), int(m.group(2))
    m = re.search(r"time per step:\s*([\d.]+)ms\s*\((\d+) steps over (\d+) blocks", text)
    if m:
        perf["ms_per_step"] = float(m.group(1))
        perf["steps_used"] = int(m.group(2))
        perf["blocks"] = int(m.group(3))
    m = re.search(r"in-step parallel\s*([\d.]+)\s*tok/s", text)
    if m:
        perf["in_step_parallel_tok_s"] = float(m.group(1))
    if perf:
        return perf
    # new interactive llama-cli prints "[ Prompt: X t/s | Generation: Y t/s ]" on stdout
    m = re.search(r"Generation:\s*([\d.]+)\s*t/s", text)
    if m:
        return {"tok_per_s": float(m.group(1))}
    # older llama-cli perf line on stderr
    hits = re.findall(r"eval time[^\n]*?/\s*(\d+)\s*(?:runs|tokens)[^\n]*?([\d.]+)\s*tokens per second",
                      stderr)
    if hits:
        n, ts = hits[-1]
        return {"tok_per_s": float(ts), "tokens": int(n)}
    return {}


def bench(label, cmd_builder, gen_tokens):
    for name, text in PROMPTS.items():
        cmd = cmd_builder(text)
        print(f"[{label}] {name} ...", flush=True)
        out, err, wall = run_one(cmd)
        perf = parse_perf(out, err)
        reply = out.strip()
        RESULTS["runs"].append({
            "model": label, "prompt": name, "wall_s": round(wall, 1),
            "gen_tokens_requested": gen_tokens, **perf, "output": reply[-1200:],
        })
        print(f"  wall {wall:.0f}s, perf {perf}", flush=True)


if __name__ == "__main__":
    dg = gguf("unsloth/diffusiongemma-26B-A4B-it-GGUF", "diffusiongemma-26B-A4B-it-Q4_K_M.gguf")
    g4 = gguf("google/gemma-4-26B-A4B-it-qat-q4_0-gguf", "gemma-4-26B_q4_0-it.gguf")

    bench("Gemma 4 26B-A4B QAT q4_0 (AR, llama-cli, CPU)",
          lambda p: [LLAMA_CLI, "-m", g4, "-p", p, "-n", "160", "--temp", "0",
                     "-t", "8", "--single-turn", "--no-display-prompt"],
          160)
    bench("DiffusionGemma 26B-A4B Q4_K_M (diffusion, llama-diffusion-cli, CPU)",
          lambda p: [DIFF_CLI, "-m", dg, "-p", p, "-n", "256", "--temp", "0", "-t", "8"],
          256)

    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(RESULTS, f, indent=2, ensure_ascii=False)
    print("done.")
