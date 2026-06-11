# diffusion-gemma lab

Google DiffusionGemma (2026-06-10 공개, 첫 오픈 가중치 텍스트 디퓨전 LLM)의 주장을
직접 검증하는 실험 저장소. 위키 노트: `vault/trends/2026-06-diffusion-gemma.md` (별도 위키 저장소).

## 구조

```
docs/
  01-generation-algorithm.md   추론 코드 정독으로 정리한 생성 알고리즘 원리
reference/                     외부 코드 클론 (git 추적 제외, 아래 방법으로 재현)
```

## reference/ 재현 방법

```bash
# HF 모델 저장소 - config/토크나이저만, 26B 가중치(LFS)는 제외
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 \
  https://huggingface.co/google/diffusiongemma-26B-A4B-it reference/hf-model

# Transformers 구현 (분석 기준 커밋: 8014139)
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/huggingface/transformers reference/transformers
git -C reference/transformers sparse-checkout set src/transformers/models
```

분석 기준 커밋: transformers `8014139`, hf-model `0f28bc4`.

## 진행 상황

- [x] 추론 코드 입수 (Transformers 네이티브 `diffusion_gemma` 모듈)
- [x] 생성 알고리즘 원리 분석 (docs/01)
- [ ] 클라우드 GPU에서 vLLM/Transformers 구동 재현
- [ ] 속도 주장 실측 (vs Gemma 4 26B A4B, batch size별)
- [ ] 디노이징 스텝 캡 vs 품질 곡선
- [ ] 한국어 품질, 블록 경계 일관성, 장문 열화 스팟체크
