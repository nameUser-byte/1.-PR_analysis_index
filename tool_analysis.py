import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_tool_analysis():
    # Load data
    df = pd.read_csv('train_cleaned.csv')
    
    # Set style
    sns.set_theme(style="whitegrid", font="Malgun Gothic")
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. CD Distribution by Tool ID (Boxplot)
    sns.boxplot(data=df, x='tool_id', y='resist_line_cd_nm', hue='pr_tone', ax=axes[0], palette=['#0a7a86', '#a3610a'])
    axes[0].set_title('Tool ID 및 PR Tone별 CD(선폭) 분포', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('장비 (Tool ID)')
    axes[0].set_ylabel('CD (nm)')
    
    # 2. Defect (FAIL) Rate by Tool ID
    # Calculate FAIL rate
    fail_rates = df.groupby(['tool_id', 'pr_tone'])['spec_pass'].apply(lambda x: (x == 'FAIL').mean() * 100).reset_index()
    fail_rates.rename(columns={'spec_pass': 'fail_rate_pct'}, inplace=True)
    
    sns.barplot(data=fail_rates, x='tool_id', y='fail_rate_pct', hue='pr_tone', ax=axes[1], palette=['#0a7a86', '#a3610a'])
    axes[1].set_title('Tool ID 및 PR Tone별 불량(FAIL) 발생률', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('장비 (Tool ID)')
    axes[1].set_ylabel('FAIL 비율 (%)')
    
    for p in axes[1].patches:
        height = p.get_height()
        if pd.notnull(height) and height > 0:
            axes[1].annotate(f'{height:.1f}%', (p.get_x() + p.get_width() / 2., height),
                             ha='center', va='bottom', fontsize=10, fontweight='bold')
                             
    plt.tight_layout()
    plt.savefig('tool_analysis.png', dpi=150, bbox_inches='tight')
    print("tool_analysis.png generated successfully.")

if __name__ == "__main__":
    generate_tool_analysis()
