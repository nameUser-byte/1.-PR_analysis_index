import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Load Data
df = pd.read_csv('train_cleaned.csv')

# Features definition
num_features = ['nominal_cd_nm', 'exposure_dose_mj_cm2', 'normalized_dose_pct', 'focus_um', 
                'coat_thickness_nm', 'softbake_temp_c', 'peb_temp_c', 'develop_time_s', 
                'developer_concentration_pct', 'field_x', 'field_y']

for tone in ['POSITIVE', 'NEGATIVE']:
    print(f"--- VIF for {tone} PR ---")
    tone_df = df[df['pr_tone'] == tone][num_features].dropna()
    
    # Drop constant columns (std == 0)
    tone_df = tone_df.loc[:, tone_df.std() > 0]
    
    # Add constant for proper VIF calculation
    tone_df_with_const = add_constant(tone_df)
    
    # Calculate VIF
    vif_data = pd.DataFrame()
    vif_data["feature"] = tone_df_with_const.columns
    vif_data["VIF"] = [variance_inflation_factor(tone_df_with_const.values, i) for i in range(len(tone_df_with_const.columns))]
    
    # Exclude constant from printing
    vif_data = vif_data[vif_data['feature'] != 'const']
    
    print(vif_data.sort_values('VIF', ascending=False).to_string(index=False))
    print("\n")
