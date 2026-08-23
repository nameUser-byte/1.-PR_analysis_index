import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style for nice charts
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 120

df = pd.read_csv('train_cleaned.csv')

# 1. Target Distribution (Histogram)
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='resist_line_cd_nm', hue='pr_tone', kde=True, bins=30, palette=['#0a7a86', '#a3610a'])
plt.title('Distribution of Target CD (resist_line_cd_nm)')
plt.xlabel('CD (nm)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('cd_dist.png')
plt.close()

# 2. Boxplot CD by PR Tone
plt.figure(figsize=(6, 5))
sns.boxplot(data=df, x='pr_tone', y='resist_line_cd_nm', palette=['#0a7a86', '#a3610a'])
plt.title('Target CD Distribution by PR Tone')
plt.xlabel('PR Tone')
plt.ylabel('CD (nm)')
plt.tight_layout()
plt.savefig('cd_boxplot.png')
plt.close()

# 3. Scatter Plot: Dose vs CD
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='exposure_dose_mj_cm2', y='resist_line_cd_nm', hue='pr_tone', alpha=0.6, palette=['#0a7a86', '#a3610a'])
# Add trendlines manually for visualization
for tone, color in zip(['POSITIVE', 'NEGATIVE'], ['#0a7a86', '#a3610a']):
    subset = df[df['pr_tone'] == tone]
    if len(subset) > 0:
        sns.regplot(data=subset, x='exposure_dose_mj_cm2', y='resist_line_cd_nm', scatter=False, color=color, line_kws={"linestyle":"--"})
plt.title('Exposure Dose vs Line CD')
plt.xlabel('Exposure Dose (mJ/cm2)')
plt.ylabel('Line CD (nm)')
plt.legend(title='PR Tone')
plt.tight_layout()
plt.savefig('dose_vs_cd.png')
plt.close()

print("Charts generated successfully.")
