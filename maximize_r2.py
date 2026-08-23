"""
R² 극대화 종합 탐색 스크립트
=================================
시도할 전략:
1. Random Forest (Baseline)
2. XGBoost (Baseline)
3. LightGBM (더 빠른 Gradient Boosting)
4. Stacking Ensemble (RF + XGB + LGBM 앙상블)
5. Optuna 기반 XGBoost 하이퍼파라미터 베이지안 탐색
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
import lightgbm as lgb
import optuna
import warnings
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

df = pd.read_csv('train_cleaned.csv')

# ── Features (normalized_dose_pct 제거, 공선성 통제 완료)
num_features = ['nominal_cd_nm', 'exposure_dose_mj_cm2', 'focus_um',
                'coat_thickness_nm', 'softbake_temp_c', 'peb_temp_c',
                'develop_time_s', 'developer_concentration_pct',
                'field_x', 'field_y']
cat_features = ['tool_id', 'retained_pattern_source']
target = 'resist_line_cd_nm'

all_results = {}

for tone in ['POSITIVE', 'NEGATIVE']:
    print(f"\n{'='*55}")
    print(f"  ▶ PR Tone: {tone}")
    print(f"{'='*55}")

    tone_df = df[df['pr_tone'] == tone].copy()
    X = tone_df[num_features + cat_features]
    y = tone_df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    # 전처리: 스케일러 + OHE (Poly는 Optuna/Stacking에서 별도 적용)
    base_preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])

    # 전처리 포함된 Poly 버전
    poly_preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2, include_bias=False, interaction_only=True))
        ]), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])

    tone_results = {}

    # ────────────────────────────────────
    # 1. Random Forest (base)
    # ────────────────────────────────────
    pipe_rf = Pipeline([('pre', base_preprocessor),
                        ('model', RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1))])
    pipe_rf.fit(X_train, y_train)
    r2_rf = r2_score(y_test, pipe_rf.predict(X_test))
    tone_results['Random Forest'] = r2_rf
    print(f"  Random Forest          R² = {r2_rf:.4f}")

    # ────────────────────────────────────
    # 2. XGBoost (base)
    # ────────────────────────────────────
    pipe_xgb = Pipeline([('pre', base_preprocessor),
                         ('model', XGBRegressor(n_estimators=300, max_depth=6,
                                                learning_rate=0.05, subsample=0.8,
                                                random_state=42))])
    pipe_xgb.fit(X_train, y_train)
    r2_xgb = r2_score(y_test, pipe_xgb.predict(X_test))
    tone_results['XGBoost'] = r2_xgb
    print(f"  XGBoost                R² = {r2_xgb:.4f}")

    # ────────────────────────────────────
    # 3. LightGBM
    # ────────────────────────────────────
    pipe_lgbm = Pipeline([('pre', base_preprocessor),
                          ('model', lgb.LGBMRegressor(n_estimators=300,
                                                      learning_rate=0.05,
                                                      num_leaves=63,
                                                      random_state=42,
                                                      verbose=-1))])
    pipe_lgbm.fit(X_train, y_train)
    r2_lgbm = r2_score(y_test, pipe_lgbm.predict(X_test))
    tone_results['LightGBM'] = r2_lgbm
    print(f"  LightGBM               R² = {r2_lgbm:.4f}")

    # ────────────────────────────────────
    # 4. Stacking Ensemble
    # ────────────────────────────────────
    estimators_stack = [
        ('rf',  RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
        ('xgb', XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42)),
        ('lgb', lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42, verbose=-1))
    ]
    stacking = StackingRegressor(estimators=estimators_stack,
                                 final_estimator=RidgeCV(),
                                 cv=3, n_jobs=-1)
    pipe_stack = Pipeline([('pre', base_preprocessor), ('model', stacking)])
    pipe_stack.fit(X_train, y_train)
    r2_stack = r2_score(y_test, pipe_stack.predict(X_test))
    tone_results['Stacking Ensemble'] = r2_stack
    print(f"  Stacking (RF+XGB+LGBM) R² = {r2_stack:.4f}")

    # ────────────────────────────────────
    # 5. Optuna Bayesian Tuning (XGBoost)
    # ────────────────────────────────────
    X_train_t = base_preprocessor.fit_transform(X_train)
    X_test_t  = base_preprocessor.transform(X_test)

    def objective(trial):
        params = {
            'n_estimators':    trial.suggest_int('n_estimators', 200, 1000),
            'max_depth':       trial.suggest_int('max_depth', 3, 10),
            'learning_rate':   trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample':       trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree':trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha':       trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
            'reg_lambda':      trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        }
        model = XGBRegressor(**params, random_state=42)
        score = cross_val_score(model, X_train_t, y_train, cv=3, scoring='r2').mean()
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, show_progress_bar=False)

    best_xgb = XGBRegressor(**study.best_params, random_state=42)
    best_xgb.fit(X_train_t, y_train)
    r2_optuna = r2_score(y_test, best_xgb.predict(X_test_t))
    tone_results['XGBoost + Optuna (베이지안 튜닝)'] = r2_optuna
    print(f"  XGBoost + Optuna       R² = {r2_optuna:.4f}")
    print(f"    └─ Best params: {study.best_params}")

    # -- 최종 승자
    best_name = max(tone_results, key=tone_results.get)
    print(f"\n  [Winner] {best_name}  (R2 = {tone_results[best_name]:.4f})")
    all_results[tone] = tone_results

print("\n\n-- 전체 요약 --")
for tone, res in all_results.items():
    print(f"\n[{tone}]")
    for model, r2 in sorted(res.items(), key=lambda x: -x[1]):
        print(f"  {model:40s} R2 = {r2:.4f}")
