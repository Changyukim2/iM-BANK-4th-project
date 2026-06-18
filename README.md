# iM Bank Corporate DX Transition Prediction

iM뱅크 법인 익명 월별 패널 데이터를 활용해 **법인 고객의 온라인 뱅킹 활성화 가능성**을 예측하는 ML/DL 프로젝트입니다.

프로젝트의 핵심 주제는 다음과 같습니다.

>법인 고객이 다음 달 온라인 뱅킹 사용 비중을 높일 가능성 예측

온라인 뱅킹은 기본적으로 `인터넷뱅킹거래금액`과 `스마트뱅킹거래금액`을 기준으로 정의합니다.

## Project Goal

법인 고객의 과거 거래 패턴을 바탕으로 다음 달 온라인 채널 사용이 증가할 가능성이 높은 고객을 예측합니다. 최종적으로는 디지털 전환 캠페인의 타깃 고객 선정에 활용할 수 있는 모델을 만드는 것이 목표입니다.

주요 목표는 다음과 같습니다.

- 온라인 채널 활성화 가능성이 높은 법인 고객 예측
- 정형 데이터 기반 ML 모델과 월별 시계열 기반 DL 모델 비교
- 파생변수, 클러스터링, 튜닝이 성능에 미치는 영향 분석
- 모델 해석을 통해 온라인 뱅킹 전환에 영향을 주는 요인 도출

## Target Design

메인 타깃은 **다음 달 온라인 채널 점유율 상승 여부**입니다.

- Online amount = `인터넷뱅킹거래금액` + `스마트뱅킹거래금액`
- Total channel amount = `인터넷뱅킹거래금액` + `스마트뱅킹거래금액` + `창구거래금액` + `ATM거래금액` + `폰뱅킹거래금액`
- Online share = Online amount / Total channel amount
- Target = 다음 달 online share가 현재 월보다 5%p 이상 증가하면 1, 아니면 0

보조 타깃으로는 **다음 달 온라인 뱅킹 신규 활성화 여부**를 사용할 수 있습니다.

추가 실험용 Y1 타깃은 **t시점의 온라인 거래 금액 규모**입니다.

- `target_y1_online_amount_t` = 현재 월 `online_amount`
- `target_y1_log1p_online_amount_t` = `log(1 + target_y1_online_amount_t)`

Y1은 회귀 타깃입니다. 이 타깃으로 모델을 만들 때는 `online_amount`, `인터넷뱅킹거래금액`, `스마트뱅킹거래금액`, `online_share`처럼 현재 월 온라인 금액을 직접 포함하거나 계산에 사용하는 컬럼을 피처에서 제외해야 합니다.

## Design Flow

전체 설계 흐름은 다음 순서로 진행합니다.

1. **EDA**
   - 월별 법인 패널 구조 확인
   - 채널별 거래금액, 사용 비율, 고객 등급/업종/지역별 차이 분석
   - 온라인 채널과 오프라인 채널의 구성 변화 확인

2. **Target Creation**
   - 기준 월 `t`의 정보로 `t+1`월의 온라인 채널 점유율 상승 여부 생성
   - 마지막 월은 다음 달 정보가 없으므로 학습 타깃에서 제외
   - 미래 정보가 피처에 섞이지 않도록 처리

3. **Feature Engineering**
   - 최근 3개월/6개월 온라인 사용 패턴
   - 온라인 채널 점유율 및 변화량
   - 창구 채널 점유율 및 변화량
   - 거래 활동성, 예금/여신 규모, 고객 속성 변수
   - 선택적으로 채널 행동 기반 클러스터 라벨 추가

4. **ML Modeling**
   - Logistic Regression, Decision Tree, RandomForest로 baseline 구성
   - LightGBM, XGBoost, CatBoost 등 정형 데이터 모델 비교
   - AutoML은 직접 구현한 모델과 비교하는 용도로 사용

5. **DL Modeling**
   - 최근 6개월 또는 12개월 거래 시퀀스를 입력으로 사용
   - MLP, LSTM, GRU 등을 비교
   - 월별 순서 정보가 성능 향상에 도움이 되는지 확인

6. **Evaluation & Interpretation**
   - ROC-AUC, PR-AUC, F1-score, Top-K Recall, Lift@K 평가
   - 캠페인 타깃팅 관점에서는 Top-K Recall과 Lift@K를 우선 확인
   - Feature importance, SHAP, 상위 예측 고객 프로파일 분석 수행

## Data Split

데이터는 시간 순서를 유지해 분할합니다.

- Train: 2023-01 ~ 2024-12
- Validation: 2025-01 ~ 2025-06
- Test: 2025-07 ~ 2025-11

랜덤 분할은 같은 법인의 미래 정보가 학습에 섞일 수 있으므로 사용하지 않습니다.

## Preprocessing

전처리는 완성본 노트북인 `src/preprocessing/Preprocessing_DX.ipynb`에서 수행합니다.

```bash
/opt/anaconda3/envs/chatbot/bin/python -c "import json, os; os.chdir('src/preprocessing'); nb=json.load(open('Preprocessing_DX.ipynb', encoding='utf-8')); g={'__name__':'__main__', 'display': print}; [exec(compile(''.join(c.get('source', [])), f'Preprocessing_DX.ipynb:cell{i}', 'exec'), g) for i, c in enumerate(nb['cells']) if c.get('cell_type') == 'code' and ''.join(c.get('source', [])).strip()]"
```

생성되는 주요 산출물은 다음과 같습니다.

- `outputs/processed/dx_processed.csv`: 파생변수 포함 전처리 완료 데이터
- `outputs/processed/modeling_columns.json`: 노트북 기준 모델 피처 목록
- `outputs/processed/customer_type_stats.csv`: 고객 유형별 법인 수
- `outputs/processed/preprocessing_summary.png`: 전처리 요약 그래프

학습 스크립트는 `dx_processed.csv`를 읽은 뒤 다음 달이 실제로 존재하는 행만 사용하고, 프로젝트 기준에 맞춰 Train `2023-01~2024-12`, Validation `2025-01~2025-06`, Test `2025-07~2025-11`로 다시 분할합니다.

## Baseline Modeling

AutoML 전에 누수 방지 baseline 모델을 먼저 학습합니다.

```bash
python3 src/ml/train_baseline.py
```

baseline에서는 `outputs/processed/dx_processed.csv`와 `modeling_columns.json`의 `feature_cols`를 사용합니다. 추가로 코드에서 `next_*`, `target_*`, `기준년월`, `법인ID`, `split`, `eda_priority_score`를 학습 피처에서 제외합니다.

현재 비교한 모델은 다음과 같습니다.

- Logistic Regression
- Decision Tree
- RandomForest

검증 데이터 PR-AUC 기준 최고 baseline은 `random_forest_depth8`입니다.

현재 재실행 결과:

| run | best model | Valid PR-AUC | Test PR-AUC | Test Top 10% Recall | Test Top 10% Lift |
|---|---|---:|---:|---:|---:|
| baseline | `random_forest_depth8` | 0.6924 | 0.6930 | 0.5410 | 5.4091 |
| no-direct-online baseline | `random_forest_depth8` | 0.6608 | 0.6569 | 0.4978 | 4.9769 |
| no-direct-online tuned | `catboost_d8_lr003_l2_8` | 0.7211 | 0.7162 | 0.5519 | 5.5183 |

주요 산출물은 다음과 같습니다.

- `outputs/models/baseline/baseline_report.md`: baseline 비교 리포트
- `outputs/models/baseline/baseline_metrics.csv`: 모델별 성능표
- `outputs/models/baseline/baseline_metrics.json`: 상세 성능 수치
- `outputs/models/baseline/best_baseline_model.joblib`: 검증 PR-AUC 기준 최고 baseline 모델

온라인 직접 변수 의존도를 점검하려면 ablation baseline을 실행합니다.

```bash
python3 src/ml/train_baseline.py \
  --exclude-direct-online-features \
  --output-dir outputs/models/baseline_no_direct_online
```

데이터 누수 가능성과 중복 정보 의존도를 더 강하게 낮춘 1차 제거 실험은 다음 명령으로 실행합니다.

```bash
python3 src/ml/train_baseline.py \
  --exclude-direct-online-features \
  --exclude-leakage-duplicate-candidates \
  --output-dir outputs/models/baseline_leakage_duplicate_removed
```

이 설정은 미래/타깃/식별자 컬럼을 차단한 뒤, 온라인 직접 변수와 타깃 정의 또는 기존 원본 변수와 정보가 강하게 겹치는 파생변수를 함께 제외합니다. 제외 사유와 사용 여부는 `feature_audit.csv`에 저장됩니다.

온라인 직접 변수 제외 조건에서 성능을 개선하려면 CatBoost/XGBoost 튜닝 스크립트를 실행합니다.

```bash
python3 src/ml/tune_no_direct_online.py --models catboost
```

튜닝 모델에도 같은 1차 제거 기준을 적용하려면 다음 옵션을 추가합니다.

```bash
python3 src/ml/tune_no_direct_online.py \
  --models catboost \
  --exclude-leakage-duplicate-candidates \
  --output-dir outputs/models/tuned_leakage_duplicate_removed
```

현재 CatBoost 튜닝 결과 최고 모델은 `catboost_d8_lr003_l2_8`이며, valid PR-AUC `0.7211`을 기록했습니다.

주요 산출물은 다음과 같습니다.

- `outputs/models/tuned_no_direct_online/tuned_report.md`: 튜닝 결과 리포트
- `outputs/models/tuned_no_direct_online/tuned_metrics.csv`: 튜닝 모델별 성능표
- `outputs/models/tuned_no_direct_online/tuned_metrics.json`: 상세 성능 수치
- `outputs/models/tuned_no_direct_online/best_tuned_model.joblib`: valid PR-AUC 기준 최고 튜닝 모델

## SHAP Interpretation

모델 해석은 다음 단계에서 다시 실행합니다. 최고 baseline 모델의 SHAP 상위 변수는 다음 명령으로 확인합니다.

```bash
python3 src/ml/explain_shap.py --split test --sample-size 3000
```

주요 산출물은 다음과 같습니다.

- `outputs/models/baseline/shap/shap_top20_report.md`: SHAP 상위 20개 변수 리포트
- `outputs/models/baseline/shap/shap_top20_grouped.csv`: 원 변수 기준 SHAP 상위 20개
- `outputs/models/baseline/shap/shap_top20_encoded.csv`: one-hot 인코딩 컬럼 기준 SHAP 상위 20개
- `outputs/models/baseline/shap/shap_top20_grouped.png`: SHAP 상위 20개 그래프

온라인 직접 변수 제외 tuned CatBoost 모델의 SHAP은 다음 명령으로 확인합니다.

```bash
python3 src/ml/explain_tuned_no_direct_online_shap.py --split test --sample-size 3000
```

산출물은 `outputs/models/tuned_no_direct_online/shap/`에 저장됩니다.

## Files

- `src/preprocessing/Preprocessing_DX.ipynb`: DX 전환 예측용 전처리 완성본
- `src/ml/train_baseline.py`: AutoML 전 baseline 모델링 스크립트
- `src/ml/tune_no_direct_online.py`: 온라인 직접 변수 제외 조건 CatBoost/XGBoost 튜닝 스크립트
- `src/ml/explain_shap.py`: baseline 모델 SHAP 해석 스크립트
- `src/ml/explain_tuned_no_direct_online_shap.py`: tuned 모델 SHAP 해석 스크립트
- `src/dl/`: DL 모델링 코드를 둘 폴더
- `outputs/processed/`: 노트북 기반 전처리 데이터와 메타
- `outputs/models/baseline/`: baseline 모델 성능과 저장 모델
- `outputs/models/baseline_no_direct_online/`: 온라인 직접 변수 제외 baseline 결과
- `outputs/models/tuned_no_direct_online/`: 온라인 직접 변수 제외 tuned 모델 결과
- `AGENTS.md`: 프로젝트 작업 규칙과 모델링 기준

## Notes

- 원본 데이터는 Git에 커밋하지 않습니다.
- 비율 계산 시 분모가 0인 경우를 명시적으로 처리합니다.
- 설명과 결과 해석은 한국어로 작성합니다.
- 최종 모델링 코드는 재현 가능하도록 Python 파일로 분리하는 것을 권장합니다.
