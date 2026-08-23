# 반도체 포토 공정 (Photolithography) ML 분석 프로젝트

본 프로젝트는 반도체 포토 공정의 주요 파라미터(Dose, Focus, PEB Temp 등)를 기반으로 **임계 치수(CD, Critical Dimension)를 예측**하고, **공정 합격/불합격(PASS/FAIL) 여부를 분류**하는 머신러닝 파이프라인 구축 결과물입니다.

## 📌 주요 산출물 (Core Files)

- **최종 분석 리포트**: [`analysis_index.html`](analysis_index.html) (클릭하여 브라우저에서 열람)
- **제출용 예측 결과**: `A_holdout_predictions.csv`, `B_holdout_predictions.csv`
- **에이전트 자동화 스킬 (Agent Skills)**:
  - [Photo Process Pipeline 스킬](.agents/skills/photo-process-ml-pipeline/SKILL.md)
  - [범용 반도체 ML Pipeline 스킬](.agents/skills/semiconductor-process-ml-pipeline/SKILL.md)

---

## 🚀 파이프라인 핵심 요약

### 1. 도메인 지식 기반 전처리 (Domain-Aware Preprocessing)
- **PR Tone 분리**: 빛에 반응하는 화학적 메커니즘이 정반대인 Positive PR과 Negative PR을 완전히 분리하여 독립적인 모델을 구축했습니다.
- **다중공선성(VIF) 재검증**: 데이터에 상수항(Constant)을 포함하여 정확히 VIF를 재계산한 결과, 온도 변수들의 VIF는 1.01 수준으로 정상임이 확인되었습니다. 단, `exposure_dose`와 의미가 겹치는 `normalized_dose_pct`(VIF ~ 6.9) 변수는 피처 중요도 해석 왜곡을 방지하기 위해 제거했습니다.
- **Domain Shift 방지**: 그룹별 전처리 과정에서 `StandardScaler` 인스턴스가 섞여 R²가 폭락(-4.41)하던 현상을 독립된 `ColumnTransformer` 구축으로 해결했습니다.

### 2. CD 예측 회귀 모델 (Regression)
- **Positive PR**: `XGBoost + Optuna` 베이지안 튜닝 최적화 적용 (검증 **R²: 0.8158**)
- **Negative PR**: 데이터 크기 부족 한계를 극복하기 위해 `Stacking Ensemble (RF + XGB + LGBM)` 적용 (검증 **R²: 0.5752**)
- 변수 중요도(Feature Importance) 분석 결과, 두 그룹 모두 노광량(`exposure_dose_mj_cm2`)과 장비 편차(`tool_id`)가 CD에 가장 큰 영향을 미치는 것으로 확인되었습니다.

### 3. 공정 합불 분류 모델 (PASS/FAIL Classification)
- 공정 데이터 특유의 **심각한 불량 클래스 불균형**(FAIL 비율 약 67%)에 대응하기 위해 단순 정확도(Accuracy)가 아닌 **F1-Score**를 기준으로 모델을 평가했습니다.
- **Positive PR**: `RandomForestClassifier(class_weight='balanced')`를 사용하여 F1-Score **0.688** 달성.
- **Negative PR**: 과도한 튜닝이 오히려 역효과를 내는 소규모 데이터의 특성을 파악하여 `XGBoost` Baseline 모델을 채택.

---

## 🛠 실행 가이드 (How to Run)

모든 분석 코드는 Python 스크립트로 모듈화되어 있으며, 다음 순서로 실행하여 결과를 완벽하게 재현할 수 있습니다.

```bash
# 1. 데이터 파악 및 정제 (VIF 검증)
python preprocess.py

# 2. EDA 및 시각화 차트 생성 (cd_dist.png, corr_heatmap.png 등)
python generate_all_charts.py

# 3. CD 회귀 모델 학습 및 평가
python ml_pipeline.py

# 4. Holdout 블라인드 테스트 분류(PASS/FAIL) 예측
python predict_holdout.py

# 5. 종합 HTML 보고서 생성
python rebuild_html.py
```

## 📚 참고 자료
- 본 파이프라인은 반도체 공정 AI 분석의 표준 지침서인 `01_PHOTO_PR_PROCESS_GUIDE.md`를 바탕으로 설계되었습니다.
