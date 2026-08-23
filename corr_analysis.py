import pandas as pd

df = pd.read_csv('train_cleaned.csv')

# 타겟 변수
target = 'resist_line_cd_nm'

for tone in ['POSITIVE', 'NEGATIVE']:
    # 해당 pr_tone 데이터 필터링
    tone_df = df[df['pr_tone'] == tone].select_dtypes(include=['float64', 'int64'])
    
    # 분산이 0인 컬럼 제거 (상관계수 계산 시 NaN 방지)
    tone_df = tone_df.loc[:, tone_df.std() > 0]
    
    # 타겟과의 상관계수 계산
    if target in tone_df.columns:
        corr = tone_df.corr()[target].drop(target)
        
        # 절대값 기준으로 정렬하여 상위 5개 추출
        top5 = corr.abs().sort_values(ascending=False).head(5)
        
        # 원본 상관계수 값과 함께 출력
        print(f"\n[{tone} PR] Top 5 Correlations with {target}:")
        for feature in top5.index:
            print(f"{feature}: {corr[feature]:.4f}")
