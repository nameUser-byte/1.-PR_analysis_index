import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# 1. Load Data
df = pd.read_csv('train_cleaned.csv')

num_features = ['nominal_cd_nm', 'exposure_dose_mj_cm2', 'focus_um', 
                'coat_thickness_nm', 'softbake_temp_c', 'peb_temp_c', 'develop_time_s', 
                'developer_concentration_pct', 'field_x', 'field_y']
cat_features = ['tool_id', 'retained_pattern_source']
target = 'resist_line_cd_nm'

results_html = {}
corr_html = {}

for tone in ['POSITIVE', 'NEGATIVE']:
    tone_df = df[df['pr_tone'] == tone].copy()
    
    # 2. Correlation Analysis
    num_df_tone = tone_df.select_dtypes(include=['float64', 'int64'])
    num_df_tone = num_df_tone.loc[:, num_df_tone.std() > 0]
    if target in num_df_tone.columns:
        corr = num_df_tone.corr()[target].drop(target).sort_values(ascending=False)
        corr_html[tone] = corr
    
    # ML Preparation
    X = tone_df[num_features + cat_features]
    y = tone_df[target]
    
    # 70/30 Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Preprocessor with Polynomial Features
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2, include_bias=False, interaction_only=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ])
    
    # Base Models
    rf = RandomForestRegressor(random_state=42)
    xgb = XGBRegressor(random_state=42)
    
    pipelines = {
        'Random Forest': Pipeline([('preprocessor', preprocessor), ('model', rf)]),
        'XGBoost': Pipeline([('preprocessor', preprocessor), ('model', xgb)])
    }
    
    # GridSearchCV to maximize R2
    param_grids = {
        'Random Forest': {
            'model__n_estimators': [100, 200],
            'model__max_depth': [None, 10, 20],
            'model__min_samples_split': [2, 5]
        },
        'XGBoost': {
            'model__n_estimators': [100, 200],
            'model__max_depth': [3, 5, 7],
            'model__learning_rate': [0.05, 0.1, 0.2]
        }
    }
    
    best_model_name = ""
    best_r2 = -float('inf')
    best_rmse = 0
    best_pipeline = None
    
    for name in pipelines.keys():
        grid = GridSearchCV(pipelines[name], param_grids[name], cv=3, scoring='r2', n_jobs=-1)
        grid.fit(X_train, y_train)
        
        y_pred = grid.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        if r2 > best_r2:
            best_r2 = r2
            best_rmse = rmse
            best_model_name = name
            best_pipeline = grid.best_estimator_
            
    # Extract Feature Importances
    model_step = best_pipeline.named_steps['model']
    preprocessor_step = best_pipeline.named_steps['preprocessor']
    
    # Get numeric feature names after Polynomial expansion
    poly = preprocessor_step.named_transformers_['num'].named_steps['poly']
    num_feature_names = poly.get_feature_names_out(num_features)
    
    # Get categorical feature names
    cat_encoder = preprocessor_step.named_transformers_['cat']
    cat_feature_names = cat_encoder.get_feature_names_out(cat_features)
    
    all_feature_names = list(num_feature_names) + list(cat_feature_names)
    
    importances = model_step.feature_importances_
    feat_imp = pd.Series(importances, index=all_feature_names).sort_values(ascending=False).head(5)
    
    results_html[tone] = {
        'best_model': best_model_name,
        'r2': best_r2,
        'rmse': best_rmse,
        'importances': feat_imp
    }

# 3. Generate HTML
head_html = df.head(5).to_html(classes='data-table', index=False, border=0)

def render_corr_list(corr_series):
    html = '<ul class="check-list" style="margin-top:14px; max-height:220px; overflow-y:auto; padding-right:10px;">\n'
    for feat, val in corr_series.items():
        color = "var(--good)" if val > 0 else "var(--danger)"
        html += f'  <li><code style="flex:1;">{feat}</code> <b style="color:{color}">{val:+.4f}</b></li>\n'
    html += '</ul>'
    return html

def render_imp_list(imp_series):
    html = '<ul class="check-list" style="margin-top:14px;">\n'
    for feat, val in imp_series.items():
        html += f'  <li><code style="flex:1;">{feat}</code> <b>{val:.4f}</b></li>\n'
    html += '</ul>'
    return html

# Read existing HTML to replace only the Machine Learning section
with open('index.html', 'r', encoding='utf-8') as f:
    full_html = f.read()

import re

# We will generate a new ML section HTML and replace it in the original string using regex
ml_section = f"""<section id="machine-learning">
  <div class="wrap">
    <div class="section-head">
      <div class="eyebrow cyan">04 · MACHINE LEARNING PIPELINE</div>
      <h2 class="section-title">CD 예측 모델 <em>학습 결과</em></h2>
      <p class="section-lede">데이터 누수 방지를 위해 공정 조건 피처만 활용하였으며, 데이터를 Positive/Negative 그룹으로 나눈 후 70% 학습(Train), 30% 검증(Validation) 세트로 분할했습니다.<br><br>
      <strong style="color: var(--cyan);">💡 최적화 및 모델 선정 기준:</strong> 이전 모델의 정확도를 극대화하기 위해 다항 회귀(Polynomial Features) 피처 엔지니어링과 GridSearchCV(하이퍼파라미터 튜닝)를 추가했습니다. Random Forest와 XGBoost가 경쟁 학습하여 가장 성능(R²)이 높은 모델이 자동 채택되었습니다.</p>
    </div>
    <div class="grid-2">
      <div class="card border-l cy">
        <span class="tag cyan">POSITIVE PR 예측 모델</span>
        <h4>{results_html['POSITIVE']['best_model']}</h4>
        <div class="metric-box">
          R² Score: <strong>{results_html['POSITIVE']['r2']:.4f}</strong> <br>
          RMSE: {results_html['POSITIVE']['rmse']:.4f} nm
        </div>
        <h5 style="margin-top:20px; margin-bottom: 5px; font-size:13px; color:var(--text-dim);">Top 5 Feature Importances (중요도)</h5>
        {render_imp_list(results_html['POSITIVE']['importances'])}
      </div>
      <div class="card border-l" style="border-left-color:var(--amber);">
        <span class="tag" style="color:var(--amber);">NEGATIVE PR 예측 모델</span>
        <h4>{results_html['NEGATIVE']['best_model']}</h4>
        <div class="metric-box">
          R² Score: <strong style="color:var(--amber);">{results_html['NEGATIVE']['r2']:.4f}</strong> <br>
          RMSE: {results_html['NEGATIVE']['rmse']:.4f} nm
        </div>
        <h5 style="margin-top:20px; margin-bottom: 5px; font-size:13px; color:var(--text-dim);">Top 5 Feature Importances (중요도)</h5>
        {render_imp_list(results_html['NEGATIVE']['importances'])}
      </div>
    </div>
  </div>
</section>"""

new_html = re.sub(r'<section id="machine-learning">.*?</section>', ml_section, full_html, flags=re.DOTALL)

with open('index.html', 'w', encoding='utf-8') as f:
    f.write(new_html)

print(f"Positive R2: {results_html['POSITIVE']['r2']:.4f}")
print(f"Negative R2: {results_html['NEGATIVE']['r2']:.4f}")
