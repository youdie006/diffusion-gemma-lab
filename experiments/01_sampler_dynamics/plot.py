# Paper-style rendering for EXP A/B/C (results.json -> figures/).
# Re-runnable without measuring: .venv/bin/python experiments/01_sampler_dynamics/plot.py

import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import paperstyle
from paperstyle import ACCENT, ACCENT2, GRAY, INK, LIGHT

FIG_DIR = os.path.join(HERE, "figures")


def fig_a(res):
    a = res["exp_a"]
    commits, entropy = a["commits_per_step"], a["mean_entropy_per_step"]
    steps = range(1, len(commits) + 1)
    fig, ax1 = plt.subplots(figsize=(5.2, 2.8))
    ax1.bar(steps, commits, color=GRAY, width=0.8, label="tokens committed")
    ax1.set_xlabel("denoising step")
    ax1.set_ylabel("tokens committed per step")
    ax1.set_ylim(0, max(commits) * 2.5)
    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(steps, entropy, color=ACCENT2, lw=1.4, label="mean entropy")
    ax2.axhline(a["uniform_entropy"], color=ACCENT2, ls=":", lw=0.9)
    ax2.annotate("uniform entropy", xy=(len(commits) * 0.55, a["uniform_entropy"]),
                 xytext=(0, -9), textcoords="offset points", color=ACCENT2, fontsize=8)
    ax2.set_ylabel("mean token entropy (nats)", color=ACCENT2)
    ax2.tick_params(axis="y", colors=ACCENT2)
    ax2.set_ylim(0, a["uniform_entropy"] * 1.15)
    fig.savefig(os.path.join(FIG_DIR, "fig_a_untrained.png"))
    plt.close(fig)


def fig_b_heatmap(res):
    b = res["exp_b_base"]
    commit_step, difficulty = b["commit_step"], b["difficulty"]
    order = sorted(range(len(difficulty)), key=lambda i: difficulty[i])
    cs = [commit_step[i] for i in order]
    grid = [[1.0 if 0 <= cs[p] <= e else 0.0 for p in range(len(cs))]
            for e in range(b["steps_used"])]
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    ax.imshow(grid, aspect="auto", cmap="Greys", interpolation="nearest", vmin=0, vmax=1.4)
    ax.set_xlabel("canvas position (sorted by difficulty)")
    ax.set_ylabel("elapsed step")
    fig.savefig(os.path.join(FIG_DIR, "fig_b_heatmap.png"))
    plt.close(fig)


def fig_b_sweep(res):
    s = res["exp_b_sweep"]
    growths = [float(k) for k in s["growth"]]
    steps = list(s["growth"].values())
    bounds = [float(k) for k in s["entropy_bound_mean_commits"]]
    commits = list(s["entropy_bound_mean_commits"].values())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.6))
    ax1.plot(growths, steps, "o-", color=INK)
    ax1.set_xlabel("confidence growth rate (synthetic)")
    ax1.set_ylabel("steps until adaptive stop")
    paperstyle.panel_label(ax1, "(a)")
    ax2.semilogx(bounds, commits, "s-", color=INK)
    ax2.set_xlabel(r"entropy bound $\epsilon$")
    ax2.set_ylabel("mean tokens committed / step")
    paperstyle.panel_label(ax2, "(b)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_b_sweep.png"))
    plt.close(fig)


def fig_c(res):
    c = res["exp_c"]
    if "t_ar_decode_step_ms" not in c:
        return
    t_canvas, t_fixed, t_ar = c["t_canvas_forward_ms"], c["t_fixed_ms"], c["t_ar_decode_step_ms"]
    s_axis = list(range(2, 65, 2))
    speedup = [(256 * t_ar) / (t_fixed + s * t_canvas) for s in s_axis]
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    ax.axvspan(12, 16, color=LIGHT, label="vendor: typical 12-16 steps")
    ax.plot(s_axis, speedup, color=INK)
    ax.axhline(1.0, color=GRAY, ls=":", lw=0.9)
    ax.axvline(48, color=ACCENT2, ls="--", lw=1.0, label="max-step cap (48)")
    ax.set_xlabel("denoising steps per 256-token canvas")
    ax.set_ylabel("speedup over AR decode")
    ax.legend(loc="upper right")
    fig.savefig(os.path.join(FIG_DIR, "fig_c_speedup.png"))
    plt.close(fig)


def render(res):
    paperstyle.apply()
    os.makedirs(FIG_DIR, exist_ok=True)
    fig_a(res)
    fig_b_heatmap(res)
    fig_b_sweep(res)
    fig_c(res)


if __name__ == "__main__":
    with open(os.path.join(HERE, "results.json")) as f:
        render(json.load(f))
    print("figures rendered.")
