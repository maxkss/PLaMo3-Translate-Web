# 이미지 번역 평가셋

공개 데이터셋에서 뽑은 샘플로 이미지 번역(OCR → 문단 묶기 → 번역 → 배치)을 평가합니다.
원본 대부분이 재배포를 금지하므로 **이미지와 정답은 저장소에 넣지 않고** `eval/data/`(git 제외)에 다시 만듭니다.

```bash
cd extension
node eval/build.mjs                       # 평가셋 만들기 (약 220 MB, 몇 분)
node eval/run.mjs 2b                      # 평가 (PROFILE=<모델이 설치된 프로필>, test/lib.mjs와 같은 로컬 모델 서버)
node eval/run.mjs 0.8b --sets=hiertext,openmantra --limit=20 --no-typography
node eval/run.mjs 0.8b --sets=crello --shots=80 --tag=mine   # 번역된 화면을 eval/results/shots-mine/에 저장
node eval/run.mjs 0.8b --overlay=30 --e2e --shots=30 --tag=mine   # 정답 스타일이 없는 세트도 30장씩 번역 화면까지
node eval/erase.mjs --save=20             # 원문 지우기 방식 비교 (crello, 모델 없이 OCR·지우기만)
node eval/build.mjs --holdout             # 홀드아웃: 같은 원본의 다른 이미지 760장을 eval/holdout/에
node eval/run.mjs 0.8b --data=holdout --overlay=30 --e2e   # (erase.mjs도 --data=holdout)
```

`CHROME=<chrome.exe>`로 다른 Chromium을, `DIST=<폴더>`로 다른 빌드를 지정할 수 있습니다(기준 빌드와 비교할 때).

결과는 `eval/results/<시각>-<모델>.json`(git 제외)에 남고, 세트별 요약이 화면에 출력됩니다.

## 구성

| 세트 | 원본 | 내용 | 정답 | 라이선스 |
|---|---|---|---|---|
| `cagui` (400) | [openbmb/CAGUI](https://huggingface.co/datasets/openbmb/CAGUI) grounding/OCR | 중국 앱 화면 | 화면마다 요소 1개의 위치와 이름 | CC BY-NC 4.0 |
| `hiertext` (80) | [HierText](https://github.com/google-research-datasets/hiertext) validation | 글자가 많은 사진·문서 (주로 영어, 여러 언어 섞임) | 문단 → 줄 위치와 글자, 세로쓰기 여부 | 주석 CC BY-SA 4.0, 이미지 Open Images (CC BY 2.0) |
| `openmantra` (60) | [OpenMantra](https://github.com/mantra-inc/open-mantra-dataset) | 일본 만화 5종 (시리즈별 고르게) | 말풍선 위치, 일본어 원문, 전문 번역가의 영어·중국어(번체) 번역 | CC BY-NC 4.0 |
| `vistra` (100) | [Vistra](https://github.com/esalesky/vistra-benchmark) | 영어 글자가 있는 사진 | 글자 묶음 전사(위치 없음)와 독일어·스페인어·러시아어·중국어(간체) 번역 | CC BY-SA 4.0 |
| `crello` (80) | [Crello](https://huggingface.co/datasets/cyberagent/crello) test | 포스터·전단·SNS 포스트·배너 등 디자인 템플릿 (형식별 최대 16장, 글자 요소 2개 이상, 회전 없는 글자) | 요소별 이미지를 합성해 만든 **글자 없는 배경**(`clean/`)과 **글자 픽셀 마스크**(`masks/`), 글자 요소마다 위치·글자 크기·굵기·색·정렬·줄 수·폰트 | CDLA-Permissive-2.0 |
| `websight` (40) | [WebSight v0.1](https://huggingface.co/datasets/HuggingFaceM4/WebSight) HTML | 합성 영어 웹페이지를 이 스크립트가 1024px로 렌더링 | 글자 덩어리마다 위치, 크기, 굵기, 색, 보이는 정렬(블록 안 왼쪽·가운데·오른쪽) | CC BY 4.0 |

각 세트는 `images/`와 같은 형식의 `manifest.jsonl`을 가집니다.

```
{ id, set, image, width, height, lang, license, source,
  regions: [{ box: [x, y, w, h] | null, text, level, visible?, lines?, refs?: { en, zh, … } }],
  styles?: [{ box, text, fontSize, fontWeight, color, align, lines, key }],    // websight, crello
  clean?, mask? }                                                             // crello: 글자 없는 배경, 글자 마스크
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
| font | 정답 글자 요소 하나와 OCR 문단 하나가 1:1로 맞는 경우: 추정 글자 크기 ÷ 실제 글자 크기(중앙값, ±10%·±20% 이내 비율), 굵기 판정 일치율 | websight, crello |
| typography | 실제 페이지 오버레이로 한국어 번역 후: 같은 원본 스타일 안 글자 크기 편차(sizeSpread), 정렬 기준점 오차(anchorErrEm), 정렬 판정 일치율(alignAcc), 번역문끼리 겹침(collisions), 번역 글자 크기 ÷ 원문 글자 크기(sizeRatio)와 목표 대비 ±15% 이내 비율(sizeOnTarget — 한글은 같은 px에서 라틴 대문자보다 크게 보이므로 라틴→한국어의 목표는 0.87), 줄 수 유지율(sameLines) | websight, crello |
| overlay (`--overlay=N`) | 정답 없이 재는 값, 세트마다 N장을 실제 오버레이로 번역: 번역문끼리 겹침(collisions), 이미지 밖으로 나감(outside), 원문에서 읽은 크기의 60% 미만으로 줄어든 문단(shrunk) | 전부 |
| e2e (`--e2e`) | 영어 정답 번역이 있는 세트를 영어로 번역해, 각 말풍선 위에 실제로 표시된 번역을 읽는 순서로 이어 정답과 비교한 chrF. 검출·문단 묶기·번역을 한꺼번에 반영합니다 | openmantra |

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

## 홀드아웃

`eval/data`의 이미지는 코드를 고치면서 계속 본 것이라, 거기서의 개선은 그 이미지에 맞춘 것일 수 있습니다.
`node eval/build.mjs --holdout`은 같은 원본에서 **겹치지 않는 다른 이미지**를 `eval/holdout/`에 만듭니다
(cagui 401–800번째, hiertext 81–160번째, openmantra·vistra는 사이사이 다른 쪽, websight 3000번째부터, crello는 validation 분할).
규칙을 넣거나 임계값을 고친 뒤에는 홀드아웃에서도 같은 방향인지 확인하고, 홀드아웃을 보며 조정하지 않습니다.

## 원문 지우기 비교 (`eval/erase.mjs`, 2026-11-12)

crello 80장(RTX 3070 WebGPU). 확장의 OCR·문단·마스크 코드를 그대로 돌리고, 채우는 방식만 바꿔 **글자 없는 정답 배경**과 비교합니다.
l1은 구멍과 실제 글자 픽셀 위의 평균 절대 오차(0–255), "악화"는 지운 결과가 안 지운 것보다 정답에서 먼 문단 수입니다.

| 방식 | 조정 세트 전체 l1 (사진·무늬) | 악화 | 홀드아웃 전체 l1 (사진·무늬) | 악화 | 문단당 시간 |
|---|---|---|---|---|---|
| 0.3.0: 사진은 박스 전체 + MI-GAN, 나머지는 확산 | 6.57 (21.8) | 33 | 6.28 (18.5) | 17 | 25 ms |
| 전부 확산 (같은 마스크) | 6.75 (22.5) | 34 | 6.64 (19.8) | 21 | 17 ms |
| 전부 MI-GAN (같은 마스크) | 9.18 (21.8) | 32 | 8.36 (18.8) | 13 | 70 ms (CPU wasm 8스레드: 770 ms) |
| **획 마스크 + 확산, 박스 전체일 때만 MI-GAN (현재)** | **5.23** (16.6) | 19 | **5.63** (16.1) | 13 | 17 ms |
| 정답 마스크 + 확산 (상한) | 2.02 (6.8) | 1 | 1.68 (4.7) | 0 | 11 ms |

- **채우는 방식보다 마스크가 훨씬 큰 변수입니다.** 같은 마스크에서는 확산과 MI-GAN의 오차가 비슷하고(두 세트 모두), 정답 마스크를 주면 3분의 1 이하로 떨어집니다.
- 확산은 모델이 없고 MI-GAN보다 WebGPU에서 4–5배, CPU에서 약 50배 빠르며 구멍이 크다고 포기하지 않습니다(MI-GAN은 문단의 12–14%를 거부).
- MI-GAN이 나은 곳은 무늬가 있는 배경의 큰 구멍뿐이라, 사진 위 글자는 먼저 획 마스크(줄 안 색 3-군집 → 박스 테두리와 이어지지 않은 획 모양 군집)를 시도하고 실패한 문단만 MI-GAN으로 보냅니다.
- **획 마스크의 효과는 홀드아웃에서 절반 이하입니다**(조정 세트 −20%, 홀드아웃 −10%). 조정 세트에서는 획 마스크가 잡힌 문단의 l1이 9.8인데 홀드아웃에서는 16.0이고, 잡힌 74개 중 9개는 글자의 3% 이상을 놓칩니다. 색 군집만으로는 한계가 있고, 정답 마스크와의 차이(5.6 대 1.7)가 다음 개선 지점입니다(글자 획 분할 모델).
- 홀드아웃에서 효과가 0이어서 뺀 규칙: 라벨 위 글자 예외(글자가 한 색에 둘러싸여 있으면 같은 색 배경과 이어져 있어도 인정), 띠 위 글자 판정(줄 안에 배경색이 거의 없으면 단색 배경에서도 획 마스크 사용). 조정 세트에서만 5.23 → 4.95였습니다.
- 마스크를 글자 밖으로 넓히는 폭은 0.08em입니다. 렌더링된 디자인만 보면 4px로 묶는 편이 조금 낫지만, 고해상도 사진의 흐린 글자 가장자리가 남아 표지판 사진(vistra)에서 원문이 비쳤습니다.

## 현재 결과 (2026-11-12, 0.8B + OCR mobile, RTX 3070 / Windows)

0.3.0 → 현재. 홀드아웃은 기준·현재 모두 Edge에서, 조정 세트는 Chromium 158에서 잰 값입니다(Windows의 글자 렌더링이 달라 위 Apple M5 표와는 직접 비교할 수 없습니다).
조정 세트의 지우기 관련 값은 위 두 규칙을 빼기 전 것이고, 문단·크기·정렬·번역 값은 그 규칙과 무관합니다.

| 세트 | 조정 세트 (`data`) | 홀드아웃 (`holdout`) |
|---|---|---|
| cagui | recall 0.285 → 0.285, CER 0.035 → 0.035 · 겹침 19 → 14 | recall 0.259 → 0.259, CER 0.077 → 0.077 · 겹침 11 → 14, 줄어든 문단 36 → 48 |
| hiertext | split 0.312 → 0.264, merge 0.135 → 0.148, CER 0.166 → 0.167 · 겹침 81 → 76 | split 0.335 → 0.301, merge 0.115 → 0.124, CER 0.184 → 0.182 · 겹침 76 → 66 |
| openmantra | **split 0.515 → 0.314**, merge 0.008 → 0.014, CER 0.183 → 0.128 · **e2e chrF 일→영 29.5 → 31.3** | **split 0.547 → 0.343**, merge 0.009 → 0.012, CER 0.190 → 0.148 · **e2e chrF 27.7 → 32.1**, 영어 오버레이 겹침 110 → 60 |
| vistra | textRecall 0.629 → 0.635 · 겹침 15 → 8 | textRecall 0.622 → 0.632 · 겹침 11 → 11, 밖으로 나감 1 → 3 |
| websight | split 0.068 → 0.003, merge 0.058 → 0.078 · 글자 크기 추정 ±10% 이내 0.53 → 0.90 · 크기 편차 0.154 → 0.040, 기준점 오차 0.60 → 0.40em, 정렬 일치 0.68 → 0.82, 겹침 13 → 2 | split 0.071 → 0.021, merge 0.088 → 0.099 · 글자 크기 추정 ±10% 이내 0.52 → 0.83 · 크기 편차 0.142 → 0.080, 기준점 오차 0.56 → 0.43em, 정렬 일치 0.71 → 0.79, 겹침 12 → 1 |
| crello | split 0.128 → 0.104 · 글자 크기 추정 ±20% 이내 0.76 → 0.85 · 기준점 오차 0.58 → 0.57em, 겹침 34 → 30, 크기 편차 0.115 → 0.132 | split 0.166 → 0.123 · 글자 크기 추정 ±20% 이내 0.72 → 0.86 · 기준점 오차 0.61 → 0.51em, 겹침 42 → 22, 크기 편차 0.150 → 0.072, coverage 0.70 → 0.65 |

홀드아웃에서도 같은 방향으로 재현된 것: 문단 묶기(모든 세트의 split), 글자 크기 추정, 정렬과 기준점, 번역문 겹침(웹·포스터·만화), 만화의 종단 번역 점수.
재현되지 않았거나 나빠진 것: 앱 화면(cagui)은 겹침과 줄어든 문단이 조금 늘었고, 표지판 사진(vistra)은 겹침이 그대로이며, 포스터의 coverage(정답 글자 요소 위에 번역이 놓인 비율)가 5%p 내려갔습니다. merge(다른 문단이 합쳐짐)는 모든 세트에서 조금씩 올랐습니다.

OCR 시간(연속 측정): 만화·웹페이지는 같고, 글자가 많은 사진(hiertext)은 25–35% 늘었습니다(획 마스크의 색 군집).

세트별로 화면에서 확인한 변화:

- **만화**: 한 말풍선의 세로줄이 한 문단으로 묶이고(검출 박스가 아니라 글자 잉크로 겹침을 판단, 말풍선 테두리가 닿은 줄도 같은 배경으로 인정), 번역문이 말풍선 여백을 양쪽으로 고르게 씁니다.
- **표지판 사진**: 번역문 색을 글자 획의 색에서 가져옵니다(0.3.0은 "흡연"이 배경색으로 그려져 보이지 않았음).
- **여러 줄로 나뉜 문구**(`shared/blocks.ts`): "NO / SMOKING", "Smart app / to / lose weight"처럼 줄마다 다른 문단으로 읽히는 문구를 한 번에 번역해 원래 줄들에 다시 나눠 놓습니다("아니요 / 흡연" → "금연"). 줄별 번역과 통번역이 같은 말이면 목록으로 보고 줄별 번역을 그대로 씁니다.
- **앱 화면**: 칩·버튼 글자 크기가 원문과 맞습니다(0.3.0은 지나치게 작거나 한 글자씩 줄바꿈).

남은 약점: 글자 색과 같은 색이 배경에도 있는 라벨(흰 페이지 위 검은 띠의 흰 글자, 노란 띠 위 글자) — 획 마스크가 실패해 박스 전체를 지우므로 띠가 뭉개지고 글자색을 띠 색으로 잡기도 함. 겹쳐 놓인 장식 글자의 검출. 굵기·필기체·세리프 같은 글꼴 모양. 사진 위 글자의 자간. 문구 묶음을 통번역해도 모델이 원문을 그대로 돌려주는 경우("NO BUSES"). 줄 수가 줄어든 번역은 원문보다 작게 보임.
