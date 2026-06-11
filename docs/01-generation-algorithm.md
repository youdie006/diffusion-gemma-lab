# DiffusionGemma 생성 알고리즘 분석 (코드 기준)

분석 대상: Hugging Face Transformers의 네이티브 구현 `src/transformers/models/diffusion_gemma/`
(transformers 커밋 `8014139`, 모델 config는 HF `google/diffusiongemma-26B-A4B-it` 커밋 `0f28bc4` 기준).
아래 줄 번호는 모두 이 커밋 기준이다. 클론 방법은 README 참고.

벤더 문서가 아니라 실제 코드를 읽고 정리한 것. 마케팅 문구와 다른 지점은 8절에 모았다.

## 0. 모델 골격 (config.json)

- 아키텍처: `DiffusionGemmaForBlockDiffusion`, 텍스트 30 레이어, hidden 2816
- MoE: expert 128개 중 top-8 활성 (`num_experts=128`, `top_k_experts=8`) -> 총 26B / 활성 ~4B
- Attention: sliding window 1024 + global 레이어 혼합 (`layer_types` 30개), RoPE는
  global 레이어만 partial rotary 0.25, theta 1e6
- vocab 262,144, max position 262,144 (256K 컨텍스트), 임베딩 타이드
- `canvas_length = 256` -- 디퓨전 생성의 블록 단위
- 멀티모달: SigLIP 계열 비전 인코더 포함 (텍스트+이미지+비디오 입력 -> 텍스트 출력)

## 1. 큰 그림: 가중치를 공유하는 인코더-디코더

구조상 인코더-디코더지만 두 스택이 **모든 가중치를 공유**한다
(`modeling_diffusion_gemma.py:1482-1491`, `_tied_weights_keys`). 역할 분담은:

- **인코더**: 프롬프트 + 지금까지 완성된 블록들을 causal attention으로 처리해 KV 캐시 생성.
  일반 AR 모델의 prefill과 같다 (`DiffusionGemmaEncoderTextAttention`, is_causal 기본 참)
- **디코더**: 256토큰 캔버스를 **양방향 attention**으로 처리 (`is_causal = False`,
  `modeling_diffusion_gemma.py:384`). 인코더 KV 캐시를 read-only로 이어붙여서
  컨텍스트를 본다 (`modeling_diffusion_gemma.py:446-452`, encoder KV를 canvas KV 앞에 concat)

즉 "캔버스 안은 양방향, 블록 사이는 causal"이 attention 구조로 구현돼 있다.

## 2. 바깥 루프: 블록-autoregressive

`generation_diffusion_gemma.py:639-818`

1. `max_new_canvases = ceil(max_new_tokens / 256)` 으로 블록 수 결정
2. 블록마다: 인코더가 새 컨텍스트를 증분 인코딩 (prefill 이후에는 직전 캔버스 256토큰만,
   `:943`) -> 디노이징 루프로 캔버스 완성 -> `input_ids`에 append (`:788`) -> 다음 블록
3. 모든 시퀀스가 EOS로 끝나면 바깥 루프 조기 종료 (`:804-805`)

## 3. 캔버스 초기화: 마스크 토큰이 아니라 진짜 랜덤 토큰

`generation_diffusion_gemma.py:389-399`

```python
torch.randint(low=0, high=vocab_size, size=(batch_size, canvas_length))
```

[MASK] 같은 특수 토큰이 없다. vocab 전체에서 uniform random으로 뽑은 "유효한 토큰
쓰레기"로 시작해서 매 스텝 다듬는다. 마케팅의 "노이즈 캔버스"는 문자 그대로 랜덤 토큰열.

## 4. 안쪽 루프: 디노이징 스텝 1회의 내부

`generation_diffusion_gemma.py:753` -- 스텝 카운터가 `max_denoising_steps(48) -> 1`로
**역방향** 진행. 각 스텝:

1. **Forward**: 디코더가 (현재 캔버스, self-conditioning logits, 인코더 KV)를 받아
   캔버스 전 위치의 logits 출력 (`:1025-1033`)
2. **온도 스케줄**: `temp = t_min + (t_max - t_min) * (cur_step / max_steps)`,
   기본 t_max=0.8, t_min=0.4 (`:312`). cur_step이 48->1로 줄어드니 온도는 0.8에서
   0.4로 선형 하강 -- 갈수록 분포가 샤프해짐
3. **샘플링 + argmax**: `multinomial(softmax(logits/temp))`로 샘플, 동시에 argmax
   캔버스도 유지 (`:1036-1043`). **최종 출력은 argmax 쪽이다** (샘플은 수락 판정용)
4. **엔트로피 바운드 수락** (`:401-443`): 위치별 엔트로피를 오름차순 정렬, 누적합이
   `cumsum(entropy) - max(entropy_1..k) <= entropy_bound(0.1)` 를 만족하는 가장 확신
   있는 k개 위치만 이번 스텝에 "커밋"
5. **리노이즈** (`:445-464`): 수락 안 된 위치는 다시 uniform random 토큰으로 교체

### 조기 종료 (adaptive stopping)

`StableAndConfidentStoppingCriteria` (`:479-536`) -- 둘 다 만족해야 멈춘다:

- **안정성**: argmax 캔버스가 직전 스텝(기본 stability_threshold=1)과 완전히 동일
- **확신**: 캔버스 평균 엔트로피 < confidence_threshold(0.005)

벤더 문서의 "보통 12~16스텝에 수렴"은 이 기준이 발동한 결과. 48은 하드 캡.

## 5. Self-conditioning

이전 스텝의 logits를 soft embedding으로 바꿔 디코더 입력에 **더한다**
(`modeling_diffusion_gemma.py:1257`). 캔버스 토큰 id만으로는 잃어버리는 "이전 스텝의
확신 분포"를 다음 스텝에 전달하는 장치.

## 6. EOS와 가변 길이 출력

`generation_diffusion_gemma.py:1075-1103`

- 캔버스는 항상 256토큰을 꽉 채워 생성된다. 캔버스 내부 조기 종료는 없음
- 완성된 캔버스에서 첫 EOS 이후의 모든 토큰을 pad로 치환
- 시퀀스가 finished로 마킹되면 다음 블록 생성에서 제외

짧은 답변(예: 20토큰)도 캔버스 1개(256토큰 분량의 연산)를 통째로 돌린다는 뜻.
짧은 출력에서는 AR 대비 속도 이점이 줄거나 역전될 수 있는 구조적 이유.

## 7. KV 캐시 관점의 속도 이점

- 인코더 KV 캐시는 AR 모델과 동일하게 쌓인다 (블록 단위 증분)
- 디코더는 스텝마다 256토큰을 병렬 forward -- AR이 256번 하던 메모리 바운드 디코드를
  12~48번의 컴퓨트 바운드 forward로 치환
- vLLM 블로그 표현대로 "메모리 대역폭 압력을 추가 연산으로 교환" -> 저배치에서 유리,
  고배치에서는 이점 감소 예상 (검증 항목)

## 8. 마케팅 문구 vs 실제 코드

| 마케팅 | 코드 |
|---|---|
| "디퓨전 모델" | 타임스텝 임베딩, 노이즈 스케줄, 노이즈 예측이 전부 없음. 온도 스케줄 + 엔트로피 기반 반복 정제(iterative refinement)에 가까움 |
| "노이즈에서 시작" | uniform random 토큰열에서 시작 (마스크 토큰 아님) |
| "인코더-디코더" | 두 스택이 가중치를 전부 공유. 사실상 한 모델의 두 attention 모드 |
| "샘플링" | 최종 토큰은 argmax. multinomial 샘플은 어느 위치를 커밋할지 판정에만 사용 |
| "4x faster" | 블록 1개당 forward 횟수(12~48)와 조기 종료 빈도에 전적으로 의존. entropy_bound/confidence_threshold가 실질적 속도 노브 |

## 9. 검증 실험으로 연결

이 분석에서 나온 측정 가능한 가설:

1. `max_denoising_steps`를 8/16/32/48로 캡하면 속도-품질 곡선이 나온다 (조기 종료
   통계도 함께 기록)
2. `entropy_bound`를 키우면 스텝당 커밋이 많아져 빨라지고 품질이 떨어질 것
3. 짧은 출력(<64토큰)에서는 256토큰 캔버스 고정 비용 때문에 AR 대비 이점이 사라질 것
4. 배치를 키우면 (컴퓨트 바운드 전환) AR 대비 배율이 1x로 수렴할 것
5. 블록 경계(256토큰 배수 지점)에서 문체/일관성 깨짐이 관측될 수 있다 -- 경계 전후
   텍스트 품질 스팟체크

## 미해결 질문

- vLLM "네이티브 지원"의 실체: vLLM main(커밋 9bbf42b, 2026-06-11)에는 diffusion 관련
  코드가 없다. 별도 플러그인 또는 미머지 브랜치로 추정. recipes.vllm.ai 레시피를 까보면
  확인 가능
- 공식 테크 리포트/arXiv 부재. 학습 방법(어떤 objective로 디노이징을 학습했는지)은
  코드만으로는 알 수 없음 -- 추론 코드만 공개된 상태
