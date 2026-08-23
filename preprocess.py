import pandas as pd
import numpy as np

# 데이터 로드
df = pd.read_csv('train.csv')
print(f"Original shape: {df.shape}")

# 1. pr_tone / retained_pattern_source 조건부 대체
# 하나만 비어있는 경우 채우기
mask1 = df['pr_tone'].isna() & df['retained_pattern_source'].notna()
df.loc[mask1 & (df['retained_pattern_source'] == 'UNEXPOSED'), 'pr_tone'] = 'POSITIVE'
df.loc[mask1 & (df['retained_pattern_source'] == 'EXPOSED'), 'pr_tone'] = 'NEGATIVE'

mask2 = df['retained_pattern_source'].isna() & df['pr_tone'].notna()
df.loc[mask2 & (df['pr_tone'] == 'POSITIVE'), 'retained_pattern_source'] = 'UNEXPOSED'
df.loc[mask2 & (df['pr_tone'] == 'NEGATIVE'), 'retained_pattern_source'] = 'EXPOSED'

# 여전히 비어있는 경우 (둘 다 빈 경우) 최빈값으로 대체
df['pr_tone'] = df['pr_tone'].fillna(df['pr_tone'].mode()[0])
df['retained_pattern_source'] = df['retained_pattern_source'].fillna(df['retained_pattern_source'].mode()[0])

# 2. 공정 조건 수치형 변수 (dose, focus, temp, develop_time 등) 같은 lot_id 중앙값으로 대체
num_cols = ['exposure_dose_mj_cm2', 'normalized_dose_pct', 'focus_um', 'coat_thickness_nm', 
            'softbake_temp_c', 'peb_temp_c', 'develop_time_s', 'developer_concentration_pct']

for col in num_cols:
    df[col] = df.groupby('lot_id')[col].transform(lambda x: x.fillna(x.median()))
    # lot_id 전체가 비어있어서 median()이 NaN인 경우 대비하여 전체 median으로 한 번 더 채움
    df[col] = df[col].fillna(df[col].median())

# 3. nominal_cd_nm 결측치 및 이상값 대체
cd_mode = df['nominal_cd_nm'].mode()[0]
df['nominal_cd_nm'] = df['nominal_cd_nm'].fillna(cd_mode)
df.loc[df['nominal_cd_nm'] > 200, 'nominal_cd_nm'] = cd_mode

# 4. field_x, field_y 결측치 삭제
df = df.dropna(subset=['field_x', 'field_y'])

# 5. peb_temp_c 이상값 삭제 (예: 200 이상)
df = df[df['peb_temp_c'] < 200]

# 6. exposure_dose_mj_cm2 이상값(565.13) 역산 또는 수정
outlier_dose = df[df['exposure_dose_mj_cm2'] > 300]
print("\n[Outlier in exposure_dose_mj_cm2]")
print(outlier_dose[['exposure_dose_mj_cm2', 'normalized_dose_pct']])

# 역산 로직: 정상 데이터의 target_dose 추정 (target_dose = dose / (normalized_dose_pct / 100))
normal_df = df[df['exposure_dose_mj_cm2'] <= 300]
target_doses = normal_df['exposure_dose_mj_cm2'] / (normal_df['normalized_dose_pct'] / 100)
median_target_dose = target_doses.median()

# outlier 복원
for idx in outlier_dose.index:
    norm_pct = df.loc[idx, 'normalized_dose_pct']
    if pd.notna(norm_pct) and norm_pct < 200: # 정규화 노광량이 정상 범주라면
        df.loc[idx, 'exposure_dose_mj_cm2'] = (norm_pct / 100) * median_target_dose

# 7. developer_concentration_pct 이상값(12.21%) 수정
median_dev = normal_df['developer_concentration_pct'].median()
df.loc[df['developer_concentration_pct'] > 10, 'developer_concentration_pct'] = median_dev

# 8. Target 변수(CD) 계측 불량 및 PR Tone 오기입 이상치(Outlier) 제거
df = df[(df['resist_line_cd_nm'] >= 40) & (df['resist_line_cd_nm'] <= 60)]

print(f"\nFinal shape after cleaning: {df.shape}")
print("Missing values after cleaning:")
print(df.isnull().sum()[df.isnull().sum() > 0])
print("\nMax values after cleaning:")
print(df[num_cols + ['nominal_cd_nm']].max())

df.to_csv('train_cleaned.csv', index=False)
print("\nSaved to train_cleaned.csv")
