"""
보고서 품질 개선 - 시각화 일괄 생성 스크립트
Generates: heatmap, predicted_vs_actual, feature_boxplots, learning_curve, feature_importance_bar
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
import lightgbm as lgb
from sklearn.metrics import r2_score
import warnings, optuna
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 130

CYAN  = '#0a7a86'
AMBER = '#a3610a'
BG    = '#f3f4f0'

df = pd.read_csv('train_cleaned.csv')

num_features = ['nominal_cd_nm', 'exposure_dose_mj_cm2', 'focus_um',
                'coat_thickness_nm', 'softbake_temp_c', 'peb_temp_c',
                'develop_time_s', 'developer_concentration_pct',
                'field_x', 'field_y']
cat_features = ['tool_id', 'retained_pattern_source']
target = 'resist_line_cd_nm'

PROCESS_VARS = num_features  # 공정 조건 (모델 입력)
RESULT_VARS  = ['cdu_3sigma_nm', 'ler_nm', 'scum_probability',
                'pattern_collapse_probability', 'defect_probability']

# ── 모델 재학습 (예측 vs 실측, 학습 곡선용)
base_preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
])

trained_models = {}
split_data = {}
for tone in ['POSITIVE', 'NEGATIVE']:
    tone_df = df[df['pr_tone'] == tone].copy()
    X = tone_df[num_features + cat_features]
    y = tone_df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    split_data[tone] = (X_train, X_test, y_train, y_test)
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])
    
    if tone == 'POSITIVE':
        model = Pipeline([('pre', preprocessor),
                          ('model', XGBRegressor(n_estimators=950, max_depth=3,
                                                 learning_rate=0.0215752, subsample=0.60364,
                                                 colsample_bytree=0.74834, reg_alpha=0.03569,
                                                 reg_lambda=4.26036, random_state=42))])
    else:
        estimators = [
            ('rf',  RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
            ('xgb', XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42)),
            ('lgb', lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42, verbose=-1))
        ]
        model = Pipeline([('pre', preprocessor),
                          ('model', StackingRegressor(estimators=estimators,
                                                      final_estimator=RidgeCV(), cv=3, n_jobs=-1))])
    model.fit(X_train, y_train)
    trained_models[tone] = model


# ════════════════════════════════════════════
# B-3. Correlation Heatmap (공정 변수만)
# ════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, tone, color in zip(axes, ['POSITIVE', 'NEGATIVE'], [CYAN, AMBER]):
    tone_df = df[df['pr_tone'] == tone][PROCESS_VARS + [target]].copy()
    corr = tone_df.corr()
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask, k=0)] = True
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                annot=True, fmt='.2f', linewidths=.5, ax=ax,
                annot_kws={'size': 7}, square=True)
    ax.set_title(f'{tone} PR — Process Variable Correlation Heatmap',
                 fontsize=11, color=color, pad=10)
    ax.tick_params(axis='x', rotation=45, labelsize=7.5)
    ax.tick_params(axis='y', rotation=0,  labelsize=7.5)
plt.tight_layout(pad=2)
plt.savefig('corr_heatmap.png', bbox_inches='tight')
plt.close()
print('[B-3] corr_heatmap.png saved')


# ════════════════════════════════════════════
# B-4. Predicted vs Actual
# ════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, tone, color in zip(axes, ['POSITIVE', 'NEGATIVE'], [CYAN, AMBER]):
    _, X_test, _, y_test = split_data[tone]
    y_pred = trained_models[tone].predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mn = min(y_test.min(), y_pred.min()) - 0.5
    mx = max(y_test.max(), y_pred.max()) + 0.5
    ax.scatter(y_test, y_pred, alpha=0.55, s=25, color=color, edgecolors='white', linewidths=0.4)
    ax.plot([mn, mx], [mn, mx], 'k--', lw=1.2, alpha=0.5, label='Perfect fit')
    ax.set_xlabel('Actual CD (nm)', fontsize=10)
    ax.set_ylabel('Predicted CD (nm)', fontsize=10)
    ax.set_title(f'{tone} PR\nR² = {r2:.4f}', fontsize=11, color=color)
    ax.legend(fontsize=9)
    ax.set_xlim(mn, mx)
    ax.set_ylim(mn, mx)
plt.suptitle('Predicted vs Actual CD (nm)', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('pred_vs_actual.png', bbox_inches='tight')
plt.close()
print('[B-4] pred_vs_actual.png saved')


# ════════════════════════════════════════════
# B-5. Process Variable Boxplots by PR Tone
# ════════════════════════════════════════════
key_features = ['exposure_dose_mj_cm2', 'focus_um', 'peb_temp_c',
                'coat_thickness_nm', 'develop_time_s', 'developer_concentration_pct']
labels = ['Dose\n(mJ/cm²)', 'Focus\n(µm)', 'PEB Temp\n(°C)',
          'Coat Thick.\n(nm)', 'Develop\nTime (s)', 'Developer\nConc. (%)']

fig, axes = plt.subplots(2, 3, figsize=(13, 7))
axes = axes.flatten()
for ax, feat, label in zip(axes, key_features, labels):
    pos_vals = df[df['pr_tone'] == 'POSITIVE'][feat].dropna()
    neg_vals = df[df['pr_tone'] == 'NEGATIVE'][feat].dropna()
    bp = ax.boxplot([pos_vals, neg_vals], tick_labels=['POSITIVE', 'NEGATIVE'],
                    patch_artist=True, notch=False,
                    medianprops=dict(color='black', linewidth=1.5))
    bp['boxes'][0].set_facecolor(CYAN + '99')
    bp['boxes'][1].set_facecolor(AMBER + '99')
    ax.set_title(label, fontsize=10)
    ax.tick_params(labelsize=9)
plt.suptitle('Key Process Variables — Distribution by PR Tone', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig('process_boxplots.png', bbox_inches='tight')
plt.close()
print('[B-5] process_boxplots.png saved')


# ════════════════════════════════════════════
# B-6. Learning Curve
# ════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, tone, color in zip(axes, ['POSITIVE', 'NEGATIVE'], [CYAN, AMBER]):
    tone_df = df[df['pr_tone'] == tone].copy()
    X = tone_df[num_features + cat_features]
    y = tone_df[target]
    preprocessor_b6 = ColumnTransformer([
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])
    pipe = Pipeline([('pre', preprocessor_b6),
                     ('model', XGBRegressor(n_estimators=200, max_depth=3,
                                            learning_rate=0.05, random_state=42))])
    train_sizes, train_scores, val_scores = learning_curve(
        pipe, X, y, cv=3, scoring='r2',
        train_sizes=np.linspace(0.2, 1.0, 8), n_jobs=-1
    )
    tr_mean = train_scores.mean(axis=1)
    va_mean  = val_scores.mean(axis=1)
    tr_std   = train_scores.std(axis=1)
    va_std   = val_scores.std(axis=1)
    ax.plot(train_sizes, tr_mean, 'o-', color=color, label='Train R²')
    ax.fill_between(train_sizes, tr_mean-tr_std, tr_mean+tr_std, alpha=0.15, color=color)
    ax.plot(train_sizes, va_mean, 's--', color='gray', label='Validation R²')
    ax.fill_between(train_sizes, va_mean-va_std, va_mean+va_std, alpha=0.1, color='gray')
    ax.axhline(0, color='red', linewidth=0.8, linestyle=':')
    ax.set_xlabel('Training Set Size (samples)', fontsize=10)
    ax.set_ylabel('R² Score', fontsize=10)
    ax.set_title(f'{tone} PR — Learning Curve', fontsize=11, color=color)
    ax.legend(fontsize=9)
    ax.set_ylim(-0.1, 1.05)
plt.tight_layout()
plt.savefig('learning_curve.png', bbox_inches='tight')
plt.close()
print('[B-6] learning_curve.png saved')


# ════════════════════════════════════════════
# B-7. Feature Importance Bar Chart
# ════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, tone, color in zip(axes, ['POSITIVE', 'NEGATIVE'], [CYAN, AMBER]):
    X_train, X_test, y_train, _ = split_data[tone]
    # Use simple RF for interpretable importances
    preprocessor_b7 = ColumnTransformer([
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])
    pipe = Pipeline([('pre', preprocessor_b7),
                     ('model', RandomForestRegressor(n_estimators=300, random_state=42))])
    pipe.fit(X_train, y_train)
    cat_enc = pipe.named_steps['pre'].named_transformers_['cat']
    cat_names = list(cat_enc.get_feature_names_out(cat_features))
    all_names = num_features + cat_names
    importances = pd.Series(pipe.named_steps['model'].feature_importances_, index=all_names)
    top10 = importances.sort_values(ascending=True).tail(10)
    colors = [color if v > top10.values[-3] else '#d0d5dd' for v in top10.values]
    top10.plot(kind='barh', ax=ax, color=colors, edgecolor='white')
    ax.set_title(f'{tone} PR — Feature Importance (Top 10)', fontsize=11, color=color)
    ax.set_xlabel('Importance Score', fontsize=10)
    ax.tick_params(labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('feature_importance.png', bbox_inches='tight')
plt.close()
print('[B-7] feature_importance.png saved')

print('\nAll charts generated successfully.')
