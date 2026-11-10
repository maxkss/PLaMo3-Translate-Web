# 이미지 번역 평가셋

공개 데이터셋에서 뽑은 샘플로 이미지 번역(OCR → 문단 묶기 → 번역 → 배치)을 평가합니다.
원본 대부분이 재배포를 금지하므로 **이미지와 정답은 저장소에 넣지 않고** `eval/data/`(git 제외)에 다시 만듭니다.

```bash
cd extension
node eval/build.mjs                       # 평가셋 만들기 (약 220 MB, 몇 분)
node eval/run.mjs 2b                      # 평가 (PROFILE=<모델이 설치된 프로필>, test/lib.mjs와 같은 로컬 모델 서버)
node eval/run.mjs 0.8b --sets=hiertext,openmantra --limit=20 --no-typography
```

결과는 `eval/results/<시각>-<모델>.json`(git 제외)에 남고, 세트별 요약이 화면에 출력됩니다.

## 구성

| 세트 | 원본 | 내용 | 정답 | 라이선스 |
|---|---|---|---|---|
| `cagui` (400) | [openbmb/CAGUI](https://huggingface.co/datasets/openbmb/CAGUI) grounding/OCR | 중국 앱 화면 | 화면마다 요소 1개의 위치와 이름 | CC BY-NC 4.0 |
| `hiertext` (80) | [HierText](https://github.com/google-research-datasets/hiertext) validation | 글자가 많은 사진·문서 (주로 영어, 여러 언어 섞임) | 문단 → 줄 위치와 글자, 세로쓰기 여부 | 주석 CC BY-SA 4.0, 이미지 Open Images (CC BY 2.0) |
| `openmantra` (60) | [OpenMantra](https://github.com/mantra-inc/open-mantra-dataset) | 일본 만화 5종 (시리즈별 고르게) | 말풍선 위치, 일본어 원문, 전문 번역가의 영어·중국어(번체) 번역 | CC BY-NC 4.0 |
| `vistra` (100) | [Vistra](https://github.com/esalesky/vistra-benchmark) | 영어 글자가 있는 사진 | 글자 묶음 전사(위치 없음)와 독일어·스페인어·러시아어·중국어(간체) 번역 | CC BY-SA 4.0 |
| `websight` (40) | [WebSight v0.1](https://huggingface.co/datasets/HuggingFaceM4/WebSight) HTML | 합성 영어 웹페이지를 이 스크립트가 1024px로 렌더링 | 글자 덩어리마다 위치, 크기, 굵기, 색, 보이는 정렬(블록 안 왼쪽·가운데·오른쪽) | CC BY 4.0 |

각 세트는 `images/`와 같은 형식의 `manifest.jsonl`을 가집니다.

```
{ id, set, image, width, height, lang, license, source,
  regions: [{ box: [x, y, w, h] | null, text, level, visible?, lines?, refs?: { en, zh, … } }],
  styles?: [{ box, text, fontSize, fontWeight, color, align, lines, key }] }   // websight
```

### 알려진 한계

- **CAGUI** 정답은 요소의 이름이라 화면에 안 보이는 경우가 많습니다(프로필 사진·뒤로 버튼의 접근성 이름 등). 글자처럼 생긴 박스이면서 흔한 아이콘 이름이 아닌 것(400개 중 약 158개)만 `visible: true`로 표시해 그것만 셉니다. 그래도 잡음이 있습니다.
- **OpenMantra** 말풍선 박스는 말풍선 안 글자 전체를 감싸서, 세로 줄 여러 개가 한 정답입니다.
- **Vistra**에는 위치 정답이 없어 글자를 읽었는지만 봅니다.
- **WebSight**는 합성 페이지라 실제 웹보다 단순합니다.
- 한국어 정답 번역은 어느 세트에도 없습니다. 번역 품질은 영어·중국어 정답으로 대신 잽니다.

## 지표 (`eval/run.mjs`)

| 지표 | 뜻 | 세트 |
|---|---|---|
| recall | 정답 영역 중 OCR 문단이 50% 이상 덮은 비율 | cagui, hiertext, openmantra, websight |
| precision | OCR 문단 중 50% 이상이 어떤 정답 영역 안에 있는 비율 | hiertext, openmantra, websight |
| split | 찾은 정답 문단·말풍선 중 OCR 문단 여러 개로 쪼개진 비율 | hiertext, openmantra |
| merge | OCR 문단 중 정답 문단·말풍선 여러 개에 걸친 비율 | hiertext, openmantra |
| cer | 찾은 영역의 글자 오류율 (NFKC, 공백 무시, 대소문자 무시, OCR 문단 안에서 가장 잘 맞는 구간 기준) | 위 넷 |
| textRecall | 정답 글자 묶음마다 1 − (OCR 글자와의 최소 CER) | vistra |
| chrF | 정답 원문을 PLaMo로 번역해 전문 번역과 비교 (글자 n-gram 1–6, β=2, sacreBLEU chrF와 같은 정의) | openmantra 일→영·일→중(번체), vistra 영→중(간체) |
| typography | 실제 페이지 오버레이로 한국어 번역 후: 같은 원본 스타일 안 글자 크기 편차, 보이는 정렬 기준 기준점 오차(em), 번역문끼리 겹침 | websight |

## 수동으로 더할 수 있는 데이터셋

자동으로 받을 수 없어 빠져 있습니다. 받은 뒤 같은 `manifest.jsonl` 형식으로 `eval/data/<세트>/`에 넣으면 `run.mjs --sets=<세트>`로 평가됩니다.

- [Manga109-s](https://huggingface.co/datasets/hal-utokyo/Manga109-s): 만화 87권, 글자 박스. Hugging Face 로그인·약관 동의 필요, 재배포 금지. 픽셀 단위 글자 마스크는 [Manga-Text-Segmentation](https://github.com/juvian/Manga-Text-Segmentation)(원문 지우기 마스크 평가용).
- [PopManga](https://huggingface.co/datasets/ragavsachdeva/popmanga_test): 영어판 만화, 글자·화자. 학술 연구용, 신청 필요.
- [COMICS Text+](https://github.com/gsoykan/comics_text_plus): 미국 고전 만화 글자 검출·인식. Google Drive.
- [SCUT-EnsText](https://github.com/HCIILAB/SCUT-EnsText): 글자를 지운 정답 배경이 있는 사진(원문 지우기 평가용).
- [CMGUI](https://huggingface.co/datasets/alibabagroup/CMGUI)(CC BY 4.0), [MobileViews](https://huggingface.co/datasets/mllmTeam/MobileViews)(MIT): 화면이 많지만 분할 압축이라 크기가 큽니다.

## 기준 결과 (2026-11-09, 2B, Apple M5)

| 세트 | 결과 |
|---|---|
| cagui | recall 0.29 (보이는 글자로 추정되는 158개 기준), CER 0.02 |
| hiertext | recall 0.81, precision 0.69, split 0.32, merge 0.14, CER 0.16 |
| openmantra | recall 0.74, precision 0.91, **split 0.50**, merge 0.01, CER 0.16 · chrF 일→영 35.4, 일→중(번체) 21.6 |
| vistra | textRecall 0.62 · chrF 영→중 39.7 |
| websight | recall 0.96, precision 0.82, split 0.08, merge 0.09, CER 0.08 · 배치: 크기 편차 0.16, 기준점 오차 평균 0.79em(p95 2.6em), 겹침 8건(40쪽) |

눈에 띄는 약점: 만화 말풍선의 절반이 문단 여러 개로 쪼개짐(세로쓰기 열 묶기), HierText의 문단 쪼개짐 32%,
CAGUI의 낮은 recall(정답 잡음과 작은 글자가 섞여 있어 원인 분리가 필요).

## 가벼운 기본 구성 (2026-11-10, 0.8B + OCR mobile + MI-GAN)

| 세트 | 결과 (괄호: OCR server로 바꿨을 때) |
|---|---|
| cagui | recall 0.29 (0.29), CER 0.035 (0.022) |
| hiertext | recall 0.79 (0.81), precision 0.70 (0.69), split 0.31 (0.32), CER 0.17 (0.16) |
| openmantra | recall 0.73 (0.74), precision 0.92 (0.91), split 0.52 (0.50), CER 0.18 (0.16) · chrF 일→영 33.7, 일→중 20.9 (2B: 35.4, 21.6) |
| vistra | textRecall 0.63 (0.62) · chrF 영→중 35.3 (2B: 39.7) |
| websight | recall 0.96 (0.96), CER 0.006 (0.081) · 크기 편차 0.17, 기준점 오차 0.74em, 겹침 7건 |

번역 시간은 2B의 약 3분의 1입니다(만화 471문장 19.8초, 2B 55.4초).
