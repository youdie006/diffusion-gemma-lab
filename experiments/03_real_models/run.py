# EXP E: 실존 디퓨전 LLM vs AR LLM - 소비자급 GPU 실측 비교
#
# DiffusionGemma 26B는 로컬 구동이 불가하므로, 구조가 가장 가까운 실존 공개 dLLM인
# LLaDA-MoE-7B-A1B(총 7B, 활성 1.4B MoE - DiffusionGemma 26B-A4B와 같은 sparse MoE dLLM)를
# AR 모델과 같은 조건(nf4 4bit, batch 1, 128토큰 생성)에서 비교한다.
#
#   dLLM:            inclusionAI/LLaDA-MoE-7B-A1B-Instruct (steps 128/64/32 스윕)
#   AR 활성파라미터 동급: Qwen/Qwen2.5-1.5B-Instruct
#   AR 총파라미터 동급:   Qwen/Qwen2.5-7B-Instruct
#
# LLaDA 생성 함수는 모델 카드의 공식 구현을 그대로 사용 (mask_id=156895,
# low-confidence remasking). DiffusionGemma와 달리 KV 캐시가 없어 매 스텝
# 전체 시퀀스를 forward한다는 점에 유의 (2025년 세대 dLLM의 한계).
#
# 실행: LD_LIBRARY_PATH=<nvidia cu13 lib> .venv-llada/bin/python experiments/03_real_models/run.py

import gc
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda"
# 하드웨어 사양은 공개하지 않는다
RESULTS = {"device": "consumer GPU (model undisclosed)", "quantization": "bitsandbytes nf4 4bit", "runs": []}

GEN_LEN = 128
PROMPTS = {
    "math": "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers "
            "per hour. How many kilometers can she run in 8 hours?",
    "korean": "대한민국의 수도는 어디인가요? 그 도시의 특징을 두 문장으로 설명해 주세요.",
    "code": "Write a Python function that checks whether a string is a palindrome.",
}

BNB = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_quant_type="nf4")


# --- LLaDA 공식 생성 함수 (모델 카드 README에서 그대로, 타이밍만 추가) ---------------

def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device,
                                      dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1
    return num_transfer_tokens


@torch.no_grad()
def llada_generate(model, prompt, steps=128, gen_length=128, block_length=32,
                   temperature=0.0, remasking="low_confidence", mask_id=156895):
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, : prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_slice = slice(prompt.shape[1] + num_block * block_length,
                            prompt.shape[1] + (num_block + 1) * block_length)
        block_mask_index = x[:, block_slice] == mask_id
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = x == mask_id
            logits = model(x).logits
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == "low_confidence":
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


# --- 벤치마크 루틴 -------------------------------------------------------------

def free():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def bench_llada():
    mid = "inclusionAI/LLaDA-MoE-7B-A1B-Instruct"
    print(f"[llada] loading {mid}")
    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    model = AutoModel.from_pretrained(mid, trust_remote_code=True,
                                      quantization_config=BNB, device_map=DEVICE).eval()
    vram = torch.cuda.max_memory_allocated() / 2**30

    for steps in [128, 64, 32]:
        for name, text in PROMPTS.items():
            m = [{"role": "user", "content": text}]
            prompt = tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
            ids = torch.tensor(tok(prompt)["input_ids"]).unsqueeze(0).to(DEVICE)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = llada_generate(model, ids, steps=steps, gen_length=GEN_LEN, block_length=32)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            decoded = tok.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True)[0]
            RESULTS["runs"].append({
                "model": "LLaDA-MoE-7B-A1B (dLLM)", "active_params": "1.4B", "total_params": "7B",
                "mode": f"diffusion steps={steps}", "prompt": name,
                "gen_tokens": GEN_LEN, "time_s": round(dt, 2), "tok_per_s": round(GEN_LEN / dt, 1),
                "vram_gib": round(vram, 2), "output": decoded[:400],
            })
            print(f"  steps={steps} {name}: {GEN_LEN/dt:.1f} tok/s")
    del model
    free()


def bench_ar(mid, label, active, total):
    print(f"[ar] loading {mid}")
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModelForCausalLM.from_pretrained(mid, quantization_config=BNB,
                                                 device_map=DEVICE).eval()
    vram = torch.cuda.max_memory_allocated() / 2**30

    for name, text in PROMPTS.items():
        m = [{"role": "user", "content": text}]
        ids = tok.apply_chat_template(m, add_generation_prompt=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():  # warmup
            model.generate(ids, max_new_tokens=8, do_sample=False)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=GEN_LEN, min_new_tokens=GEN_LEN,
                                 do_sample=False)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        decoded = tok.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True)[0]
        RESULTS["runs"].append({
            "model": label, "active_params": active, "total_params": total,
            "mode": "AR greedy", "prompt": name,
            "gen_tokens": GEN_LEN, "time_s": round(dt, 2), "tok_per_s": round(GEN_LEN / dt, 1),
            "vram_gib": round(vram, 2), "output": decoded[:400],
        })
        print(f"  {name}: {GEN_LEN/dt:.1f} tok/s")
    del model
    free()


if __name__ == "__main__":
    free()
    bench_llada()
    bench_ar("Qwen/Qwen2.5-1.5B-Instruct", "Qwen2.5-1.5B (AR)", "1.5B", "1.5B")
    bench_ar("Qwen/Qwen2.5-7B-Instruct", "Qwen2.5-7B (AR)", "7.6B", "7.6B")
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(RESULTS, f, indent=2, ensure_ascii=False)
    print("done.")
