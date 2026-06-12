# Paper-style rendering for EXP D (results.json -> figures/).
# Re-runnable without measuring: .venv/bin/python experiments/02_scaling_batch/plot.py

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import paperstyle
from paperstyle import ACCENT2, GRAY, INK

FIG_DIR = os.path.join(HERE, "figures")


def fig_size(res):
    rows = res["size_sweep"]
    xs = [r["params_m"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.6))
    ax1.plot(xs, [r["cost_ratio"] for r in rows], "o-", color=INK)
    ax1.set_xlabel("model size (M params)")
    ax1.set_ylabel("canvas / 1-token cost ratio")
    ax1.set_ylim(0, 2)
    paperstyle.panel_label(ax1, "(a)")
    ax2.plot(xs, [r["speedup_at_16"] for r in rows], "s-", color=INK)
    ax2.set_xlabel("model size (M params)")
    ax2.set_ylabel("speedup over AR at 16 steps")
    ax2.set_ylim(0, 20)
    paperstyle.panel_label(ax2, "(b)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_d_size.png"))
    plt.close(fig)


def fig_batch(res):
    rows = res["batch_sweep"]
    xs = [r["batch"] for r in rows]
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    ax.plot(xs, [r["speedup_at_16"] for r in rows], "o-", color=INK)
    ax.axhline(1.0, color=GRAY, ls=":", lw=0.9)
    ax.annotate("parity with AR", xy=(xs[1], 1.0), xytext=(0, 4),
                textcoords="offset points", color=GRAY, fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel("batch size")
    ax.set_ylabel("speedup over AR at 16 steps")
    fig.savefig(os.path.join(FIG_DIR, "fig_d_batch.png"))
    plt.close(fig)


def render(res):
    paperstyle.apply()
    os.makedirs(FIG_DIR, exist_ok=True)
    fig_size(res)
    fig_batch(res)


if __name__ == "__main__":
    with open(os.path.join(HERE, "results.json")) as f:
        render(json.load(f))
    print("figures rendered.")
