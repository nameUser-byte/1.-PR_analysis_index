---
name: photo-process-ml-pipeline
description: >-
  End-to-end ML analysis pipeline for semiconductor photolithography (PR coating,
  exposure, development) process data. Covers VIF-based feature selection,
  PR-Tone-aware EDA, CD regression modeling (XGBoost+Optuna / Stacking), PASS/FAIL
  classification with class-imbalance correction, and automated HTML report generation.
  Use when given train.csv and A/B holdout_features.csv for a photo process challenge.
---

# Photo Process ML Pipeline Skill

## Overview

이 스킬은 반도체 **포토리소그래피(Photolithography)** 공정 데이터를 대상으로  
전처리 → EDA → 회귀(CD 예측) → 분류(PASS/FAIL) → HTML 리포트 생성까지의  
전체 파이프라인을 재현 가능하게 실행하기 위한 노하우를 담고 있습니다.

> [!IMPORTANT]
> 이 스킬은 `semiconductor-ai-workflow` 스킬을 **기반**으로 동작합니다.
> 범용 머신러닝 원칙(데이터 누수 방지, 재현성, 과학적 해석)은 해당 스킬을 참고하십시오.

---

## Dependencies

| 스킬 / 도구 | 역할 |
|---|---|
| `semiconductor-ai-workflow` | 범용 ML 원칙(기반 스킬) |
| Python ≥ 3.10 | 실행 환경 |
| `xgboost`, `lightgbm`, `scikit-learn` | 모델링 |
| `optuna` | 하이퍼파라미터 최적화 |
| `statsmodels` | VIF 다중공선성 검사 |
| `pandas`, `matplotlib`, `seaborn` | 데이터 처리 및 시각화 |

---

## Quick Start

아래 명령어 순서로 전체 파이프라인을 실행합니다.

```powershell
# 1. 데이터 전처리 (이상치 제거, 변수 정제)
python preprocess.py

# 2. 시각화 차트 생성 (PNG 파일들 생성)
python generate_all_charts.py

# 3. CD 회귀 모델 학습 및 평가
python ml_pipeline.py

# 4. Holdout 데이터 PASS/FAIL 분류 예측
python predict_holdout.py

# 5. HTML 보고서 조립 (analysis_index.html 생성)
python rebuild_html.py
```

---

## Workflow (단계별 상세 지침)

### Step 1. 데이터 파악 및 전처리

**입력 파일 확인:**
- `train.csv` — 정답(`spec_pass`, `measured_cd_nm`)이 포함된 학습 데이터
- `A_holdout_features.csv`, `B_holdout_features.csv` — 정답 없는 블라인드 테스트 데이터

**반드시 확인해야 할 사항:**
1. `pr_tone` 컬럼(POSITIVE/NEGATIVE)이 존재하는지 확인. 이 변수는 PR의 화학적 특성을 구분하는 핵심이므로 **절대 제거하지 않습니다.**
2. `normalized_dose_pct` 컬럼: 이 변수는 `exposure_dose`와 정보가 중복되어 다중공선성(VIF ~ 6.9)을 유발하므로 **학습 피처에서 제거**합니다.
3. 온도 변수(PEB, Softbake)들은 기본 VIF 계산시 높게 나오지만, `add_constant`를 적용하면 다중공선성이 없음(VIF ~ 1.01)을 확인했습니다.
4. 결측치는 행 단위 삭제 원칙(dropna)을 기본으로 적용합니다.

**학습/검증 분리:**
- 전체 데이터의 **70%를 학습, 30%를 검증**용으로 random_state=42 고정 분리합니다.
- 계층화(Stratified) 분리를 사용해 `spec_pass` 클래스 비율을 유지합니다.

---

### Step 2. EDA 및 시각화

> [!IMPORTANT]
> **POSITIVE PR과 NEGATIVE PR을 반드시 분리해서 상관관계 분석**을 수행합니다.
> 두 그룹은 광화학 반응 메커니즘이 정반대이므로, 합쳐서 분석하면 왜곡된 결론이 나옵니다.

생성해야 할 시각화 차트 목록:

| 차트 | 파일명 | 설명 |
|---|---|---|
| CD 분포 | `cd_dist.png` | PR Tone별 `measured_cd_nm` 히스토그램 |
| 상관관계 히트맵 | `corr_heatmap.png` | POSITIVE/NEGATIVE 각각 수치형 변수 상관계수 |
| 산점도 (Dose vs CD) | `dose_vs_cd.png` | 노광량과 CD의 관계, PR Tone별 색상 구분 |
| 공정 변수 박스플롯 | `process_boxplots.png` | PASS/FAIL 그룹별 공정 파라미터 분포 |
| 학습 곡선 | `learning_curve.png` | 과적합/과소적합 여부 진단 |
| 예측 vs 실제 산점도 | `pred_vs_actual.png` | 회귀 모델 성능 시각화 |
| 변수 중요도 | `feature_importance.png` | POSITIVE/NEGATIVE 모델 각각 상위 피처 |

---

### Step 3. CD 회귀 모델링 (선폭 예측)

> [!IMPORTANT]
> POSITIVE와 NEGATIVE PR의 CD 예측 모델을 **완전히 별도로** 학습해야 합니다.
> 동일한 전처리기(StandardScaler 등)를 두 그룹에 공유하면 **Domain Shift** 오류로
> R² 값이 -4 이하로 폭락합니다. 반드시 각 그룹별로 독립적인 `ColumnTransformer`를 생성하십시오.

**모델 선택 전략:**

```
POSITIVE PR → XGBoost + Optuna 하이퍼파라미터 최적화
              (시험 결과: R² ≈ 0.98~0.99)

NEGATIVE PR → Stacking Regressor (XGBoost + RF + LightGBM의 앙상블)
              (데이터 수가 적어 단일 모델보다 앙상블이 더 안정적)
```

**사용할 피처 목록:**
```python
num_features = [
    'nominal_cd_nm', 'exposure_dose_mj_cm2', 'focus_um',
    'coat_thickness_nm', 'softbake_temp_c', 'peb_temp_c',
    'develop_time_s', 'developer_concentration_pct',
    'field_x', 'field_y'
]
cat_features = ['tool_id', 'retained_pattern_source']
# 제거: 'normalized_dose_pct' (VIF > 1000)
```

**평가 지표:** R² (결정계수), RMSE (검증 세트 기준)

---

### Step 4. PASS/FAIL 분류 모델링 (불량 예측)

> [!IMPORTANT]
> 클래스 불균형을 반드시 먼저 확인합니다.
> 이 공정 데이터에서 FAIL 비율은 POSITIVE ≈ 67%, NEGATIVE ≈ 62%로
> **심각한 불균형**이 존재합니다. 단순 Accuracy만 보면 "가짜 정확도" 함정에 빠집니다.
> 반드시 **F1-Score**를 주 평가지표로 사용하십시오.

**모델 선택 전략 (5-Fold StratifiedKFold 교차 검증 기반):**

```
POSITIVE PR → RandomForestClassifier(class_weight='balanced')
              (진단 결과: Acc=79.95%, F1=0.688로 5가지 모델 중 최고)

NEGATIVE PR → XGBClassifier (baseline, 튜닝 없음)
              (진단 결과: 다른 모델/튜닝 적용 시 모두 성능 역효과.
               데이터 부족이 병목이므로 알고리즘 변경으로는 성능 한계 도달)
```

**교차 검증 설정:**
```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

**출력 파일:**
- `A_holdout_predictions.csv` (`sample_id`, `spec_pass` 두 컬럼만 포함)
- `B_holdout_predictions.csv`

---

### Step 5. HTML 보고서 자동 생성

`rebuild_html.py`를 실행하여 `analysis_index.html`을 조립합니다.

보고서에 포함되어야 하는 섹션 순서:
1. `#hero` — 프로젝트 개요 및 데이터 현황 KPI
2. `#preprocess` — 전처리 내역 (VIF, 결측치, 분리 전략)
3. `#data-preview` — 주요 데이터 요약 통계
4. `#visualization` — EDA 시각화 차트 (모달 팝업 확대 지원)
5. `#correlation` — PR Tone별 상관관계 분석
6. `#machine-learning` — CD 회귀 모델 결과 (R², 학습 곡선, 변수 중요도)
7. `#holdout` — Holdout A/B 분류 예측 결과
8. `#references` — 참고 문헌 및 향후 과제

---

## Common Mistakes (학습된 오류 목록)

> [!CAUTION]
> 아래 실수들은 이 프로젝트에서 실제로 발생했던 문제들입니다. 재현 시 반드시 주의하십시오.

### 1. 전처리기 공유로 인한 Domain Shift
**잘못된 방법:** POSITIVE PR로 학습한 `StandardScaler`를 NEGATIVE PR 데이터에 그대로 적용  
**증상:** NEGATIVE PR의 CD 예측 R² 값이 **-4.41**로 폭락  
**올바른 방법:** `ColumnTransformer`를 `make_preprocessor()` 함수로 감싸 매 그룹마다 **새 인스턴스**를 생성

```python
def make_preprocessor(num_features, cat_features):
    return ColumnTransformer([
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])
# POSITIVE와 NEGATIVE 각각 별도 호출
```

### 2. `normalized_dose_pct` 제거 누락
**잘못된 방법:** 해당 컬럼을 피처 목록에 포함한 채 학습**증상:** VIF 계산 시 온도 변수(PEB_temp 등)의 VIF가 1000 이상 폭발함.  
**원인:** statsmodels의 variance_inflation_factor 함수에 데이터 입력 시 `add_constant`를 생략하면, 비중심(uncentered) 분산이 계산되어 평균이 0이 아닌 변수는 극도로 높은 VIF를 출력함.  
**해결:** VIF 계산 전 `df = add_constant(df)`를 적용하여 상수항을 포함시켜야 진짜 다중공선성을 알 수 있음.시도
**잘못된 방법:** NEGATIVE 분류 성능이 낮다고 LightGBM, class_weight 튜닝 등 적극 시도  
**결과:** 모든 시도가 Acc 76.58% → 74.35%로 **역효과**  
**원인:** 데이터 부족(n=269)이 근본 원인이며, 알고리즘 변경으로 해결 불가  
**올바른 방법:** XGBoost baseline 유지. 성능 한계임을 명시하고 추가 데이터 수집 권장으로 결론 내릴 것.

### 3. matplotlib `boxplot` 파라미터 오류
**잘못된 방법:** `plt.boxplot(..., labels=[...])`  
**증상:** 최신 matplotlib에서 `labels` 파라미터 Deprecated 경고/오류  
**올바른 방법:** `plt.boxplot(..., tick_labels=[...])`

---

## 피처 의미 사전 (01_photo 공정)

| 피처명 | 한국어 명칭 | 단위 | 설명 |
|---|---|---|---|
| `nominal_cd_nm` | 목표 선폭 | nm | 설계 사양의 CD 목표값 |
| `exposure_dose_mj_cm2` | 노광량 | mJ/cm² | 단위 면적당 빛 에너지. POSITIVE PR에서 증가 시 CD 감소 |
| `focus_um` | 초점 거리 | µm | 0에서 멀어질수록 CD 불균일 |
| `coat_thickness_nm` | PR 도포 두께 | nm | 두꺼울수록 노광 에너지 더 필요 |
| `softbake_temp_c` | 소프트베이크 온도 | °C | PR 용매 제거를 위한 초기 열처리 |
| `peb_temp_c` | PEB 온도 | °C | 노광 후 열처리. 산 확산으로 CD에 큰 영향 |
| `develop_time_s` | 현상 시간 | s | 길수록 CD 증가 (POSITIVE 기준) |
| `developer_concentration_pct` | 현상액 농도 | % | 고농도일수록 현상 속도 증가 |
| `field_x`, `field_y` | 노광 필드 좌표 | a.u. | 웨이퍼 내 위치 정보 (가장자리 효과 반영) |
| `tool_id` | 장비 ID | 범주 | 노광 장비 고유 ID (장비 간 편차 모델링) |
| `retained_pattern_source` | 패턴 소스 | 범주 | 마스크 패턴의 기원 |
| `pr_tone` | PR 극성 | 범주 | POSITIVE(빛 맞은 부분 제거) / NEGATIVE(유지) |

---

## 참고 문헌

1. Mack, C. A. (2007). *Fundamental Principles of Optical Lithography*. Wiley.
2. Brunner, T. A. (2003). Why optical lithography will live forever. *Journal of Vacuum Science & Technology B*, 21(6), 2632–2637.
3. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD 2016*.
4. Prokhorenkova, L. et al. (2018). CatBoost: unbiased boosting with categorical features. *NeurIPS 2018*.
5. Akiba, T. et al. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. *KDD 2019*.
