# DiffusionGemma Lab

Google DiffusionGemma(2026-06-10 공개)의 "디퓨전으로 최대 4배 빠른 텍스트 생성" 주장을,
공식 추론 코드 정독과 소비자급 GPU 실측으로 검증하는 실험 저장소.

요약:

- 코드를 읽어보면 "디퓨전"의 실체는 **엔트로피 기반 반복 정제**다. 타임스텝 임베딩,
  노이즈 스케줄, 노이즈 예측이 전부 없다
- 속도의 원천: AR이 256번 반복하는 메모리 바운드 1토큰 디코드를, 12~48번의
  256토큰 병렬 forward로 치환하는 것
- 소비자급 GPU 실측(69M 토이 모델, 동일 가중치 비교): 256토큰 캔버스 forward 1회 비용이
  1토큰 디코드의 **1.26배**에 불과 -> 12~16스텝 수렴 시 12~16배, 48스텝 캡에서도 4.2배
- 단, 이 속도는 전적으로 **모델의 확신(낮은 엔트로피)에 의존**한다. 무학습 모델은
  스텝당 정확히 1토큰만 커밋되어 AR보다 느려진다 (실측으로 확인)

## 1. DiffusionGemma가 무엇인가

- Google DeepMind의 첫 오픈 가중치 텍스트 디퓨전 LLM. Apache 2.0
- Gemma 4 26B A4B MoE 기반 (총 26B, 활성 ~4B, expert 128개 중 top-8)
- 토큰을 하나씩 뽑는 대신 256토큰 "캔버스"를 반복 디노이징으로 한 번에 생성
- 공식 주장: H100 FP8에서 1,008 tok/s (AR 대비 ~5배), 보통 12~16스텝에 수렴
- 품질은 같은 체급 Gemma 4보다 전반적으로 하락 (AIME 69.1 vs 88.3, GPQA 73.2 vs 82.3)
- 공식 arXiv 논문은 아직 없다. 1차 소스는 [모델 카드], [공식 블로그], 그리고
  Transformers 구현 코드 자체

[모델 카드]: https://ai.google.dev/gemma/docs/diffusiongemma/model_card
[공식 블로그]: https://blog.google/innovation-and-ai/technology/developers-tools/diffusion-gemma-faster-text-generation/

참고: 공식 추론 코드는 별도 저장소가 아니라 **Hugging Face `transformers` 라이브러리에
직접 머지**되어 있다 (`transformers>=5.8`, `src/transformers/models/diffusion_gemma/`).
이 저장소의 분석과 실험은 모두 그 코드를 대상으로 한다.

## 2. 원리: 코드가 말하는 생성 알고리즘

구현 코드(약 5,500줄)를 정독해 정리한 알고리즘. 상세 분석과 코드 라인 레퍼런스는
[docs/01-generation-algorithm.md](docs/01-generation-algorithm.md).

```
프롬프트 ──> 인코더 (causal attention, KV 캐시)        <- AR 모델의 prefill과 동일
                  │
                  v        반복 (블록당 최대 48스텝, 보통 12~16에서 조기 종료)
   캔버스 초기화: 256개 랜덤 토큰 ──> 디코더 forward (캔버스 내 양방향 attention)
                  ^                        │
                  │                        v
   미커밋 위치는 다시 랜덤 토큰    엔트로피 낮은 위치만 "커밋"
   (renoise)                       (entropy-bound 수락 규칙)
                  │                        │
                  └────────────────────────┘
                  v
   완성 캔버스를 컨텍스트에 append, 다음 블록으로 (블록 단위 autoregressive)
```

핵심 디테일 (전부 코드에서 확인):

1. **캔버스 초기화는 [MASK]가 아니라 vocab 전체에서 뽑은 uniform random 토큰**
2. **수락 규칙**: 위치별 엔트로피를 오름차순 정렬, `cumsum(H) - max(H) <= 0.1`을
   만족하는 가장 확신 있는 k개만 이번 스텝에 확정. 이 규칙은
   [arXiv:2505.24857](https://arxiv.org/abs/2505.24857) (entropy-bounded unmasking)에서 온 것
   (코드 주석에 명시). 수학적으로 스텝당 최소 1토큰은 항상 커밋된다
3. **조기 종료**: argmax 캔버스가 직전 스텝과 동일(stable) AND 평균 엔트로피 <
   0.005(confident) 둘 다 만족할 때
4. **온도 스케줄**: 스텝이 진행되며 0.8 -> 0.4로 선형 하강 (분포가 점점 샤프해짐)
5. **최종 출력은 argmax**. multinomial 샘플은 "어느 위치를 커밋할지" 판정에만 쓰인다
6. **인코더와 디코더는 모든 가중치를 공유**한다. 사실상 한 모델의 두 attention 모드
   (컨텍스트 = causal, 캔버스 내부 = 양방향)
7. 짧은 답변도 캔버스 256토큰 전체를 연산한다. 캔버스 내부 조기 종료는 없음

마케팅 문구와의 거리: "디퓨전 모델"이라 부르지만 연속 디퓨전의 구성요소(타임스텝
임베딩, 노이즈 스케줄, 노이즈 예측 네트워크)가 없다. 정확히 말하면 **온도 스케줄을 단
엔트로피 기반 병렬 반복 정제**다.

## 3. 실험: 실물을 못 돌리는 소비자급 GPU에서 무엇을 검증할 수 있나

실물 26B는 양자화(NVFP4)로도 ~18GB VRAM이 필요해서 보유한 소비자급 GPU로는 돌릴 수
없다. 대신 **실제 Transformers 생성 코드에 초소형 모델을 끼워** 알고리즘 동역학을
계측했다. 텍스트 품질이 아니라 샘플러의 거동이 측정 대상이다.

실험 코드: [experiments/01_sampler_dynamics/run.py](experiments/01_sampler_dynamics/run.py),
수치: [results.json](experiments/01_sampler_dynamics/results.json)

### EXP A. 확신 없는 모델에서 디퓨전은 퇴화한다 (실제 generate 루프)

무학습(랜덤 가중치) 1M 모델을 실제 `generate()`에 통과시키고 샘플러를 계측했다.

- 평균 엔트로피가 uniform(6.91 nats)에 붙어서 안 떨어짐 -> 조기 종료 미발동, 48스텝 캡 도달
- **스텝당 커밋 토큰 수: 정확히 1.00** -- 수락 규칙의 "최소 1개 보장"만 작동한 것
- `generate()`가 보고한 공식 지표 `tokens_per_forward` = 1.19 (AR과 사실상 동일한 효율)

![EXP A](experiments/01_sampler_dynamics/figures/fig_a_untrained.png)

시사점: 디퓨전의 속도 이점은 아키텍처가 아니라 **학습된 확신**에서 나온다. 모델이
확신하지 못하는 입력(어려운 문제, 분포 밖 입력)에서는 스텝 수가 늘어나 이점이
줄어들 것이라는 예측이 가능하다.

### EXP B. 확신이 차오를 때의 커밋 웨이브 (실제 샘플러 클래스 + 합성 logits)

실제 `EntropyBoundSampler`, 온도 스케줄, 정지 기준 클래스를 "확신이 점점 차오르는"
합성 logits로 구동했다. 위치별 난이도를 부여해 쉬운 위치부터 커밋되는 웨이브를 관찰.

![EXP B heatmap](experiments/01_sampler_dynamics/figures/fig_b_heatmap.png)

- 확신 상승 속도를 0.5/1/2/4배로 바꾸면 수렴 스텝이 31/18/10/6으로 변함 --
  "보통 12~16스텝" 주장은 학습된 모델의 확신 상승 속도에 대한 주장과 동치
- `entropy_bound`를 0.01 -> 10으로 키우면 스텝당 커밋이 29 -> 90개로 증가.
  공격성(속도) 노브가 맞다. 실모델에서는 품질과의 트레이드오프 측정 필요

![EXP B sweep](experiments/01_sampler_dynamics/figures/fig_b_sweep.png)

### EXP C. 속도 모델 실측: 캔버스 병렬 forward는 얼마나 싼가

69M 토이 모델(동일 가중치)로 보유 GPU에서 직접 측정:

| 측정값 | 결과 |
|---|---|
| 256토큰 캔버스 forward 1회 (한계비용) | 51.7 ms |
| AR 1토큰 디코드 1회 (KV 캐시, 같은 인코더 가중치) | 41.0 ms |
| 비용 비율 (canvas / 1-token) | **1.26배** |
| 함의: 12스텝 수렴 시 속도 이점 | 16.3배 |
| 16스텝 | 12.3배 |
| 48스텝 (캡) | 4.2배 |

![EXP C](experiments/01_sampler_dynamics/figures/fig_c_speedup.png)

해석: GPU에서 1토큰 디코드는 연산이 아니라 가중치 로딩(메모리 대역폭)에 묶여 있어,
256토큰을 한 번에 밀어도 비용이 1.26배밖에 안 늘어난다. 이것이 디퓨전 속도 주장의
하드웨어적 본질이다. 벤더의 "4~6배"는 우리 토이 실측(12~16스텝에서 12~16배)보다
보수적인데, 실물 26B는 토이보다 연산 비중이 커서 캔버스 forward가 상대적으로 더
비싸지기 때문으로 추정된다. 주의: 토이 스케일 + 소비자 GPU 측정이므로 절대값이 아닌
구조 확인용이다.

### 부수 발견: upstream 버그

`confidence_threshold=0.0`을 주면 validate의 에러 메시지가 존재하지 않는
`self.entropy_bound`를 참조해 ValueError 대신 AttributeError가 난다
(`generation_diffusion_gemma.py:177`, transformers 5.11.0).

## 4. 재현 방법

```bash
python3 -m venv --system-site-packages .venv   # torch+CUDA가 이미 있다면
.venv/bin/pip install "transformers>=5.11" matplotlib
.venv/bin/python experiments/01_sampler_dynamics/run.py
```

공식 코드/설정 원본을 로컬에 받으려면:

```bash
# 모델 config/토크나이저만 (26B 가중치 LFS 제외)
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 \
  https://huggingface.co/google/diffusiongemma-26B-A4B-it reference/hf-model
# Transformers 구현 (분석 기준 커밋 8014139)
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/huggingface/transformers reference/transformers
git -C reference/transformers sparse-checkout set src/transformers/models
```

## 5. 다음 단계

- [ ] 클라우드 GPU(H100급)에서 실물 26B 구동, 공식 속도 수치(1,008 tok/s) 재현
- [ ] 같은 하드웨어에서 Gemma 4 26B A4B(AR)와 직접 비교 -- batch size 1/4/16 곡선
- [ ] `max_denoising_steps` / `entropy_bound` vs 출력 품질 트레이드오프 (실모델)
- [ ] 한국어 품질, 블록 경계(256배수 지점) 일관성, 장문 컨텍스트 열화 스팟체크
- [ ] vLLM "네이티브 지원"의 실체 확인 (vLLM main에는 2026-06-11 기준 디퓨전 코드가 없음)

## 환경

단일 소비자급 NVIDIA GPU (사양 비공개) / torch 2.12 / transformers 5.11.0
