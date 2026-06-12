# Additive benchmark: append Gemma-family AR baselines to an existing results.json
# without re-running the slow LLaDA sweep. Idempotent: existing rows for the same
# model label are replaced.
#
# Run: LD_LIBRARY_PATH=<nvidia cu13 lib> .venv-llada/bin/python experiments/03_real_models/add_gemma.py

import json
import os

import run

HERE = os.path.dirname(os.path.abspath(__file__))
TARGETS = [
    # Official google/gemma-3-* repos are gated; the unsloth re-uploads carry the
    # identical weights under the same Gemma license and are not gated.
    ("unsloth/gemma-3-1b-it", "Gemma3-1B (AR)", "1.0B", "1.0B"),
    ("unsloth/gemma-3-4b-it", "Gemma3-4B (AR)", "4.3B", "4.3B"),
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
print("done: Gemma rows merged.")
