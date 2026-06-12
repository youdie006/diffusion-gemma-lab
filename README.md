# DiffusionGemma Lab

Verifying Google DiffusionGemma's "up to 4x faster text generation with diffusion" claim
(released 2026-06-10) by reading the official inference code and running measurements on a
consumer-grade GPU.

TL;DR:

- Reading the code, the "diffusion" is really **entropy-guided iterative refinement**.
  There is no timestep embedding, no noise schedule, no noise prediction
- Where the speed comes from: replacing 256 memory-bound one-token AR decode steps with
  12-48 parallel forwards over a 256-token canvas
- Measured on a consumer GPU (69M toy model, same weights for both sides): one 256-token
  canvas forward costs only **1.26x** a single-token decode -> 12-16x speedup at the
  typical 12-16 steps, still 4.2x at the 48-step cap
- The speedup depends entirely on **model confidence (low entropy)**. An untrained model
  commits exactly 1 token per step and ends up slower than AR (measured)
- Batch size kills the advantage: measured speedup collapses from 14.4x (bs=1) to 1.4x
  (bs=16) -- diffusion decoding helps latency, not high-throughput serving
- Against a real 2025-generation diffusion LLM (LLaDA-MoE-7B-A1B) on the same GPU and
  quantization, see EXP E below

## 1. What is DiffusionGemma

- Google DeepMind's first open-weights text diffusion LLM. Apache 2.0
- Based on the Gemma 4 26B A4B MoE (26B total, ~4B active, top-8 of 128 experts)
- Instead of decoding token-by-token, it iteratively denoises a 256-token "canvas"
- Official claims: 1,008 tok/s on H100 FP8 (~5x over AR), typically converges in 12-16 steps
- Quality is consistently below same-size Gemma 4 (AIME 69.1 vs 88.3, GPQA 73.2 vs 82.3)
- No arXiv paper yet. Primary sources are the [model card], the [official blog post], and
  the Transformers implementation itself

[model card]: https://ai.google.dev/gemma/docs/diffusiongemma/model_card
[official blog post]: https://blog.google/innovation-and-ai/technology/developers-tools/diffusion-gemma-faster-text-generation/

Note: the official inference code is not a standalone repository -- it is **merged directly
into Hugging Face `transformers`** (`transformers>=5.8`,
`src/transformers/models/diffusion_gemma/`). All analysis and experiments here target that
code.

## 2. The generation algorithm, as written in the code

Distilled from a close reading of the implementation (~5,500 lines). Full analysis with
line references: [docs/01-generation-algorithm.md](docs/01-generation-algorithm.md)
(Korean).

```
prompt ──> encoder (causal attention, KV cache)      <- identical to AR prefill
              │
              v       repeat (<=48 steps per block, adaptive early stop at ~12-16)
  canvas init: 256 random tokens ──> decoder forward (bidirectional inside canvas)
              ^                            │
              │                            v
  re-randomize uncommitted        commit only low-entropy positions
  positions (renoise)             (entropy-bound acceptance rule)
              │                            │
              └────────────────────────────┘
              v
  append finished canvas to context, next block (block-level autoregression)
```

Key details (all verified in code):

1. **Canvas init is uniform random tokens from the vocab, not [MASK]**
2. **Acceptance rule**: sort positions by entropy ascending and commit the largest
   confident prefix satisfying `cumsum(H) - max(H) <= 0.1`. The rule comes from
   [arXiv:2505.24857](https://arxiv.org/abs/2505.24857) (entropy-bounded unmasking),
   cited in a code comment. Mathematically it always commits at least 1 token per step
3. **Early stop**: argmax canvas unchanged from the previous step (stable) AND mean
   entropy < 0.005 (confident) -- both required
4. **Temperature schedule**: linear 0.8 -> 0.4 across steps (distribution sharpens)
5. **Final output is the argmax canvas**. The multinomial sample is only used to decide
   which positions to commit
6. **Encoder and decoder share all weights** -- effectively one model with two attention
   modes (causal over context, bidirectional inside the canvas)
7. Short answers still pay for a full 256-token canvas; there is no early exit inside
   a canvas

Distance from the marketing: it is called a "diffusion model", but none of the continuous
diffusion machinery (timestep embedding, noise schedule, noise prediction network) exists.
Precisely speaking it is **temperature-scheduled, entropy-guided parallel iterative
refinement**.

## 3. Experiments on a consumer GPU that cannot run the real model

The real 26B needs ~18GB VRAM even at NVFP4, beyond the consumer GPU used here. Instead we
plug **tiny models into the real Transformers generation code** and instrument the
algorithm's dynamics. The subject is sampler behavior, not text quality.

Code: [experiments/01_sampler_dynamics/run.py](experiments/01_sampler_dynamics/run.py),
numbers: [results.json](experiments/01_sampler_dynamics/results.json)

### EXP A. With an unconfident model, diffusion degenerates (real generate loop)

We push an untrained (random-weight) 1M model through the real `generate()` and instrument
the sampler.

- Mean entropy stays at uniform (6.91 nats) -> early stop never fires, hits the 48-step cap
- **Tokens committed per step: exactly 1.00** -- only the rule's at-least-one guarantee fires
- `tokens_per_forward` (the official efficiency metric reported by `generate()`) = 1.19,
  i.e. effectively AR efficiency

![EXP A](experiments/01_sampler_dynamics/figures/fig_a_untrained.png)

Implication: the speed advantage comes from **trained confidence**, not from the
architecture. On inputs the model is unsure about (hard problems, out-of-distribution
text), step counts should rise and the advantage shrink.

### EXP B. Commit waves as confidence ramps (real sampler classes + synthetic logits)

We drive the real `EntropyBoundSampler`, temperature schedule, and stopping criteria with
synthetic logits whose confidence grows over steps, with per-position difficulty. Easy
positions commit first, in waves.

![EXP B heatmap](experiments/01_sampler_dynamics/figures/fig_b_heatmap.png)

- Scaling the confidence growth rate by 0.5/1/2/4 changes convergence to 31/18/10/6 steps --
  the "typically 12-16 steps" claim is equivalent to a claim about how fast the trained
  model becomes confident
- Raising `entropy_bound` from 0.01 to 10 raises commits per step from 29 to 90. It is the
  aggressiveness (speed) knob; the quality cost needs measuring on the real model

![EXP B sweep](experiments/01_sampler_dynamics/figures/fig_b_sweep.png)

### EXP C. Measured speed model: how cheap is a canvas forward

Measured with a 69M toy (identical weights for both modes):

| measurement | result |
|---|---|
| one 256-token canvas forward (marginal cost) | 51.7 ms |
| one AR 1-token decode (KV cache, same encoder weights) | 41.0 ms |
| cost ratio (canvas / 1-token) | **1.26x** |
| implied speedup at 12 steps | 16.3x |
| at 16 steps | 12.3x |
| at the 48-step cap | 4.2x |

![EXP C](experiments/01_sampler_dynamics/figures/fig_c_speedup.png)

Interpretation: a 1-token decode is bound by weight loading (memory bandwidth), not
compute, so pushing 256 tokens through at once costs only 1.26x. That is the hardware
essence of the diffusion speed claim. The vendor's "4-6x" is more conservative than our toy
measurement (12-16x at 12-16 steps), presumably because the real 26B has a much larger
compute share, making the canvas forward relatively pricier. Caveat: toy scale on a
consumer GPU -- this validates the structure, not absolute numbers.

### EXP D. Comparison 1: same code, varying size and batch (hypothesis tests)

Two hypotheses for why the vendor number is lower than our toy number, swept with the same
DiffusionGemma code. Median of 3 runs.
Code: [experiments/02_scaling_batch/run.py](experiments/02_scaling_batch/run.py),
numbers: [results.json](experiments/02_scaling_batch/results.json)

**Hypothesis 1 -- bigger model, smaller advantage: not observable at toy scale (null
result).** Across 17M-336M the canvas/1-token cost ratio stays flat at 1.1-1.2. All these
sizes are far below GPU compute saturation; the compute-share effect cannot be seen at toy
scale. Verifying it needs the real 26B on cloud hardware.

![EXP D size](experiments/02_scaling_batch/figures/fig_d_size.png)

**Hypothesis 2 -- bigger batch, vanishing advantage: strongly confirmed.** From batch 1 to
16 the cost ratio explodes from 1.1 to 11.2 and the 16-step speedup collapses from 14.4x to
1.4x. This confirms vLLM's "particularly attractive at low batch sizes" framing, and shows
the flip side: **at high-batch serving, the diffusion speed advantage effectively
disappears.**

| batch | cost ratio (canvas/1tok) | speedup at 16 steps |
|---|---|---|
| 1 | 1.11 | 14.4x |
| 2 | 1.51 | 10.6x |
| 4 | 1.61 | 10.0x |
| 8 | 2.59 | 6.2x |
| 16 | 11.22 | 1.4x |

![EXP D batch](experiments/02_scaling_batch/figures/fig_d_batch.png)

### Side finding: an upstream bug

Passing `confidence_threshold=0.0` makes the validator's error message reference a
nonexistent `self.entropy_bound`, raising AttributeError instead of the intended ValueError
(`generation_diffusion_gemma.py:177`, transformers 5.11.0).

## 4. Reproduce

```bash
python3 -m venv --system-site-packages .venv   # assuming torch+CUDA present
.venv/bin/pip install "transformers>=5.11" matplotlib
.venv/bin/python experiments/01_sampler_dynamics/run.py
.venv/bin/python experiments/02_scaling_batch/run.py
```

To fetch the official code/config locally:

```bash
# model config/tokenizer only (skip the 26B LFS weights)
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 \
  https://huggingface.co/google/diffusiongemma-26B-A4B-it reference/hf-model
# Transformers implementation (analysis pinned at commit 8014139)
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/huggingface/transformers reference/transformers
git -C reference/transformers sparse-checkout set src/transformers/models
```

## 5. Next steps

- [ ] Run the real 26B on a cloud GPU (H100-class), reproduce the official 1,008 tok/s
- [ ] Head-to-head vs Gemma 4 26B A4B (AR) on the same hardware -- batch 1/4/16 curves
- [ ] `max_denoising_steps` / `entropy_bound` vs output quality trade-off (real model)
- [ ] Korean output quality, block-boundary (multiples of 256) coherence, long-context
      degradation spot checks
- [ ] Find out what vLLM "native support" actually is (no diffusion code in vLLM main as
      of 2026-06-11)

## Environment

Single consumer-grade NVIDIA GPU (specs undisclosed) / torch 2.12 / transformers 5.11.0
