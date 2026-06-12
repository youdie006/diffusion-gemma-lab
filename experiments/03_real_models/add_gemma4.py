# Additive benchmark: append a Gemma 4 generation AR baseline to results.json.
#
# Gemma 4 E4B ("effective 4B", MatFormer-style) is the only Gemma 4 release that fits an
# 8GB-class GPU at 4-bit; the dense 12B/31B and the 26B-A4B MoE (DiffusionGemma's actual
# base) do not. Gemma 4 needs transformers >= 5.x, so this runs in the main .venv, not the
# .venv-llada used for LLaDA/Qwen/Gemma3 (the bench routine itself is version-agnostic).
#
# Run: LD_LIBRARY_PATH=<nvidia cu13 lib> .venv/bin/python experiments/03_real_models/add_gemma4.py

import json
import os

import run

HERE = os.path.dirname(os.path.abspath(__file__))
TARGETS = [
    ("google/gemma-4-E4B-it", "Gemma4-E4B (AR)", "~4B eff.", "8B raw"),
]

with open(os.path.join(HERE, "results.json")) as f:
    results = json.load(f)

labels = {t[1] for t in TARGETS}
results["runs"] = [r for r in results["runs"] if r["model"] not in labels]

run.RESULTS = results
run.free()
for mid, label, active, total in TARGETS:
    run.bench_ar(mid, label, active, total)

with open(os.path.join(HERE, "results.json"), "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("done: Gemma 4 rows merged.")
