import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

train_df = pd.read_csv('train_cleaned.csv')
train_df['target'] = train_df['spec_pass'].map({'PASS': 1, 'FAIL': 0})

num_features = ['nominal_cd_nm','exposure_dose_mj_cm2','focus_um',
                'coat_thickness_nm','softbake_temp_c','peb_temp_c',
                'develop_time_s','developer_concentration_pct','field_x','field_y']
cat_features = ['tool_id','retained_pattern_source']

print('=== CLASS BALANCE ===')
for tone in ['POSITIVE','NEGATIVE']:
    sub = train_df[train_df['pr_tone']==tone]
    vc = sub['spec_pass'].value_counts()
    p = vc.get('PASS',0)
    f = vc.get('FAIL',0)
    print(f'{tone}: PASS={p}, FAIL={f}, FAIL ratio={f/len(sub):.1%}')

print()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for tone in ['POSITIVE','NEGATIVE']:
    sub = train_df[train_df['pr_tone']==tone].copy()
    X = sub[num_features+cat_features]
    y = sub['target']

    fail_count = int((y==0).sum())
    pass_count = int((y==1).sum())
    scale_pw = round(fail_count/pass_count, 3) if pass_count > 0 else 1

    def make_pre():
        return ColumnTransformer([
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ])

    models = {
        'XGB (baseline)':       Pipeline([('pre', make_pre()), ('clf', XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, eval_metric='logloss', random_state=42))]),
        'XGB+class_weight':     Pipeline([('pre', make_pre()), ('clf', XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, scale_pos_weight=scale_pw, eval_metric='logloss', random_state=42))]),
        'XGB deeper (depth=6)': Pipeline([('pre', make_pre()), ('clf', XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.03, eval_metric='logloss', random_state=42))]),
        'LightGBM balanced':    Pipeline([('pre', make_pre()), ('clf', lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, class_weight='balanced', random_state=42, verbose=-1))]),
        'RF balanced':          Pipeline([('pre', make_pre()), ('clf', RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42, n_jobs=-1))]),
    }

    print(f'--- {tone} PR (scale_pos_weight={scale_pw}) ---')
    for name, model in models.items():
        acc_sc = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        f1_sc  = cross_val_score(model, X, y, cv=cv, scoring='f1')
        print(f'  {name:35s} Acc={acc_sc.mean():.4f} (+/-{acc_sc.std():.4f})  F1={f1_sc.mean():.4f}')
    print()
