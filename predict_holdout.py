"""
predict_holdout.py  —  분류기 모델 선택 근거
----------------------------------------------
진단 결과 (diagnose_classifier.py 실행):
  CLASS BALANCE: POSITIVE FAIL=67.4%, NEGATIVE FAIL=62.1%  (클래스 불균형 존재)

  POSITIVE PR:
    XGB baseline        Acc=0.7661  F1=0.6230
    RF balanced         Acc=0.7995  F1=0.6880  ← 채택 (Acc, F1 모두 최고)

  NEGATIVE PR:
    XGB baseline        Acc=0.7658  F1=0.6778  ← 채택 (이미 최고, 다른 모델이 모두 하락)
    XGB+class_weight    Acc=0.7435  F1=0.6638  (오히려 하락)
    LightGBM balanced   Acc=0.7435  F1=0.6557  (하락)

결론:
  - POSITIVE: RF balanced (class_weight='balanced') 채택
  - NEGATIVE: XGB baseline 유지 — 다른 모델/튜닝이 모두 역효과
    (데이터 부족이 병목, 알고리즘 변경으로는 성능 한계 도달)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')


def make_preprocessor(num_features, cat_features):
    return ColumnTransformer([
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])


def main():
    print("1. Loading training data...")
    train_df = pd.read_csv("train_cleaned.csv")
    train_df['target'] = train_df['spec_pass'].map({'PASS': 1, 'FAIL': 0})

    num_features = ['nominal_cd_nm', 'exposure_dose_mj_cm2', 'focus_um',
                    'coat_thickness_nm', 'softbake_temp_c', 'peb_temp_c',
                    'develop_time_s', 'developer_concentration_pct',
                    'field_x', 'field_y']
    cat_features = ['tool_id', 'retained_pattern_source']

    # StratifiedKFold: 클래스 비율을 유지하면서 5번 교차 검증
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model_configs = {
        'POSITIVE': Pipeline([
            ('pre', make_preprocessor(num_features, cat_features)),
            ('clf', RandomForestClassifier(
                n_estimators=300,
                class_weight='balanced',   # FAIL 67.4% 불균형 보정
                random_state=42,
                n_jobs=-1
            ))
        ]),
        'NEGATIVE': Pipeline([
            ('pre', make_preprocessor(num_features, cat_features)),
            ('clf', XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                eval_metric='logloss',
                random_state=42
            ))
        ]),
    }

    trained_models = {}
    print()
    for tone, model in model_configs.items():
        sub = train_df[train_df['pr_tone'] == tone].copy()
        X = sub[num_features + cat_features]
        y = sub['target']

        acc_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        f1_scores  = cross_val_score(model, X, y, cv=cv, scoring='f1')
        print(f"[{tone}] Acc={acc_scores.mean():.4f} (+/-{acc_scores.std():.4f})  "
              f"F1={f1_scores.mean():.4f}  n={len(y)}")

        # Full-data training for holdout prediction
        model.fit(X, y)
        trained_models[tone] = model

    holdout_files = ['A_holdout_features.csv', 'B_holdout_features.csv']
    for file in holdout_files:
        print(f"\n2. Loading {file}...")
        try:
            holdout_df = pd.read_csv(file)
        except FileNotFoundError:
            print(f"File {file} not found. Skipping.")
            continue
            
        holdout_df['pred_int'] = np.nan

        for tone in ['POSITIVE', 'NEGATIVE']:
            idx = holdout_df['pr_tone'] == tone
            if idx.sum() > 0:
                X_h = holdout_df.loc[idx, num_features + cat_features]
                holdout_df.loc[idx, 'pred_int'] = trained_models[tone].predict(X_h)

        holdout_df['spec_pass'] = holdout_df['pred_int'].map({1.0: 'PASS', 0.0: 'FAIL'})
        output_df = holdout_df[['sample_id', 'spec_pass']]

        out_name = file.replace('_features.csv', '_predictions.csv')
        print(f"3. Saving predictions to {out_name}...")
        output_df.to_csv(out_name, index=False)
        print(f"Saved to {out_name}")

        print(f"\n[Prediction Stats for {file}]")
        print(output_df['spec_pass'].value_counts())
        pos_stats = holdout_df[holdout_df['pr_tone']=='POSITIVE']['spec_pass'].value_counts()
        neg_stats = holdout_df[holdout_df['pr_tone']=='NEGATIVE']['spec_pass'].value_counts()
        print(f"POSITIVE holdout: {pos_stats.to_dict()}")
        print(f"NEGATIVE holdout: {neg_stats.to_dict()}")


if __name__ == "__main__":
    main()
