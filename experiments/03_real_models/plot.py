# EXP E 논문 스타일 렌더링 (results.json -> figures/)
# 측정 없이 재실행 가능: python experiments/03_real_models/plot.py

import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import paperstyle
from paperstyle import ACCENT, GRAY, INK

FIG_DIR = os.path.join(HERE, "figures")

ORDER = [
    ("LLaDA-MoE-7B-A1B (dLLM)", "diffusion steps=32"),
    ("LLaDA-MoE-7B-A1B (dLLM)", "diffusion steps=64"),
    ("LLaDA-MoE-7B-A1B (dLLM)", "diffusion steps=128"),
    ("Qwen2.5-1.5B (AR)", "AR greedy"),
    ("Qwen2.5-7B (AR)", "AR greedy"),
]
LABELS = [
    "LLaDA-MoE 7B-A1B\n(32 steps)",
    "LLaDA-MoE 7B-A1B\n(64 steps)",
    "LLaDA-MoE 7B-A1B\n(128 steps, official)",
    "Qwen2.5-1.5B\n(AR, active-param peer)",
    "Qwen2.5-7B\n(AR, total-param peer)",
]


def fig_e(res):
    groups = defaultdict(list)
    for r in res["runs"]:
        groups[(r["model"], r["mode"])].append(r["tok_per_s"])

    means, lo, hi, colors = [], [], [], []
    for key in ORDER:
        vals = groups[key]
        m = sum(vals) / len(vals)
        means.append(m)
        lo.append(max(0.0, m - min(vals)))
        hi.append(max(0.0, max(vals) - m))
        colors.append(ACCENT if "dLLM" in key[0] else GRAY)

    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    y = range(len(ORDER))
    ax.barh(y, means, xerr=[lo, hi], color=colors, height=0.62,
            error_kw={"elinewidth": 0.9, "capsize": 2.5, "ecolor": INK})
    for i, m in enumerate(means):
        ax.annotate(f"{m:.1f}", xy=(m + hi[i], i), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK)
    ax.set_yticks(list(y))
    ax.set_yticklabels(LABELS, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("generation throughput (tokens/s), batch 1, nf4 4-bit, 128 new tokens")
    fig.savefig(os.path.join(FIG_DIR, "fig_e_real_models.png"))
    plt.close(fig)


def render(res):
    paperstyle.apply()
    os.makedirs(FIG_DIR, exist_ok=True)
    fig_e(res)


if __name__ == "__main__":
    with open(os.path.join(HERE, "results.json")) as f:
        render(json.load(f))
    print("figures rendered.")
