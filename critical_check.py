import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

df = pd.read_excel(
    r"C:\Users\chan\Documents\semiconductor-ai-project\20000_BGTTV.xlsx",
    sheet_name="통합_불량데이터_20000"
)
df["OOC"] = (df["SAW_SPC_Status"] == "Out_of_Control").astype(int)
df["BG_Defect"] = (df["BG_Crack_Count_After"] > 3) | (df["BG_Scratch_Count_After"] > 3)
ic_df  = df[df["OOC"] == 0]
ooc_df = df[df["OOC"] == 1]
SAW_PARAMS = ["SAW_Blade_Wear_pct", "SAW_Sawing_Speed_mm_s", "SAW_Coolant_Flow_Lmin"]

print("=" * 60)
print("CRITICAL CHECK 1: Final Test Fail vs OOC 상관관계")
print("=" * 60)
ft_all = df["Final_Test_Result"].eq("Fail").mean() * 100
ft_ic  = ic_df["Final_Test_Result"].eq("Fail").mean() * 100
ft_ooc = ooc_df["Final_Test_Result"].eq("Fail").mean() * 100
ct = pd.crosstab(df["OOC"], df["Final_Test_Result"])
from scipy.stats import chi2_contingency
chi2, p_ft, _, _ = chi2_contingency(ct)
print(f"전체 Fail율:   {ft_all:.2f}%")
print(f"IC  Fail율:    {ft_ic:.2f}%")
print(f"OOC Fail율:    {ft_ooc:.2f}%")
print(f"차이:          {ft_ooc - ft_ic:.2f}%p")
print(f"카이제곱: χ²={chi2:.4f}, p={p_ft:.4f}")
print(f"→ SAW OOC가 최종 불합격률에 미치는 영향: {'유의' if p_ft < 0.05 else '비유의'}")
print()

print("=" * 60)
print("CRITICAL CHECK 2: Cpk — 전체 vs IC군만 비교")
print("=" * 60)
def cpk(s, lsl, usl):
    mu, sigma = s.mean(), s.std(ddof=1)
    return min((usl - mu) / (3 * sigma), (mu - lsl) / (3 * sigma))

specs = {"SAW_Sawing_Speed_mm_s": (27, 34), "SAW_Coolant_Flow_Lmin": (1.5, 1.7)}
for col, (lsl, usl) in specs.items():
    ck_all = cpk(df[col], lsl, usl)
    ck_ic  = cpk(ic_df[col], lsl, usl)
    print(f"{col}:")
    print(f"  Cpk (전체, n=20000): {ck_all:.4f}")
    print(f"  Cpk (IC군만, n={len(ic_df)}): {ck_ic:.4f}")
    print(f"  → 차이: {abs(ck_all - ck_ic):.4f}")
print()

print("=" * 60)
print("CRITICAL CHECK 3: 시계열 CV — Random vs TimeSeriesSplit 비교")
print("=" * 60)
X = df[SAW_PARAMS + ["BG_Defect"]].copy()
X["BG_Defect"] = X["BG_Defect"].astype(int)
y = df["OOC"]
rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight="balanced", n_jobs=-1)
cv_rand = cross_val_score(rf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="roc_auc")
cv_ts   = cross_val_score(rf, X, y, cv=TimeSeriesSplit(n_splits=5), scoring="roc_auc")
print(f"Random 5-fold AUC:      {cv_rand.mean():.4f} ± {cv_rand.std():.4f}")
print(f"TimeSeriesSplit AUC:    {cv_ts.mean():.4f} ± {cv_ts.std():.4f}")
print(f"→ 시계열 CV와 랜덤 CV 차이: {abs(cv_rand.mean() - cv_ts.mean()):.4f}")
print()

print("=" * 60)
print("CRITICAL CHECK 4: Range-Only vs Xbar-Only 물리적 의미")
print("=" * 60)
xbar_cl = ic_df["SAW_Xbar_um"].mean(); xbar_std = ic_df["SAW_Xbar_um"].std()
range_cl = ic_df["SAW_Range_um"].mean(); range_std = ic_df["SAW_Range_um"].std()
UCL_xbar = xbar_cl + 3 * xbar_std; LCL_xbar = xbar_cl - 3 * xbar_std
UCL_range = range_cl + 3 * range_std; LCL_range = max(0, range_cl - 3 * range_std)

ooc_c = ooc_df.copy()
ooc_c["xbar_oos"]  = (ooc_c["SAW_Xbar_um"] > UCL_xbar) | (ooc_c["SAW_Xbar_um"] < LCL_xbar)
ooc_c["range_oos"] = ooc_c["SAW_Range_um"] > UCL_range
ro = ooc_c["range_oos"] & ~ooc_c["xbar_oos"]  # Range Only
xo = ooc_c["xbar_oos"] & ~ooc_c["range_oos"]  # Xbar Only

ro_df = ooc_c[ro]; xo_df = ooc_c[xo]
print(f"Range Only OOC {len(ro_df)}건 — 파라미터 평균:")
for p in SAW_PARAMS:
    print(f"  {p}: IC={ic_df[p].mean():.3f}, Range-OOC={ro_df[p].mean():.3f}")
print(f"Xbar Only OOC {len(xo_df)}건 — 파라미터 평균:")
for p in SAW_PARAMS:
    print(f"  {p}: IC={ic_df[p].mean():.3f}, Xbar-OOC={xo_df[p].mean():.3f}")
print()

print("=" * 60)
print("CRITICAL CHECK 5: BG 불량 포함 vs 제외 시 OOC율 변화")
print("=" * 60)
bg_ok  = df[~df["BG_Defect"]]
bg_bad = df[df["BG_Defect"]]
print(f"BG 불량 없는 LOT (n={len(bg_ok):,}): OOC율={bg_ok['OOC'].mean()*100:.2f}%")
print(f"BG 불량 있는 LOT (n={len(bg_bad):,}): OOC율={bg_bad['OOC'].mean()*100:.2f}%")
_, p_bg = stats.ttest_ind(bg_ok["OOC"], bg_bad["OOC"])
print(f"t-test p={p_bg:.4f} → {'유의 차이' if p_bg < 0.05 else '유의 차이 없음'}")
print()

print("=" * 60)
print("CRITICAL CHECK 6: BG_Defect 기준 재확인 (>3 vs >=3)")
print("=" * 60)
for col in ["BG_Crack_Count_After", "BG_Scratch_Count_After"]:
    vc = df[col].value_counts().sort_index()
    print(f"{col} 분포 (상위 10개값):")
    print(f"  {dict(list(vc.items())[:10])}")
    over3  = (df[col] > 3).sum()
    over3e = (df[col] >= 3).sum()
    print(f"  > 3 (초과): {over3:,}건 ({over3/len(df)*100:.2f}%)")
    print(f"  >=3 (이상): {over3e:,}건 ({over3e/len(df)*100:.2f}%)")
print()

print("=" * 60)
print("CRITICAL CHECK 7: 비선형 관계 — OOC와 파라미터 비선형 탐색")
print("=" * 60)
for p in SAW_PARAMS:
    # Spearman (순위 기반, 비선형 포착)
    sp_r, sp_p = stats.spearmanr(df[p], df["OOC"])
    pe_r, pe_p = stats.pearsonr(df[p], df["OOC"])
    print(f"{p}:")
    print(f"  Pearson r={pe_r:.4f} (p={pe_p:.4f}) | Spearman ρ={sp_r:.4f} (p={sp_p:.4f})")
    print(f"  Pearson vs Spearman 차이: {abs(sp_r - pe_r):.4f} {'← 비선형 가능성' if abs(sp_r - pe_r) > 0.02 else ''}")
print()

print("=" * 60)
print("CRITICAL CHECK 8: 미분석 변수들의 OOC 연관성")
print("=" * 60)
extra_cols = ["DA_Placement_Offset_um", "DA_Bonding_Pressure_N",
              "MOLD_Chip_Offset_um", "ALIGN_OVL_nm",
              "MOLD_Clamp_Pressure_kgf", "CLEAN_Residue_Count"]
for col in extra_cols:
    if col in df.columns:
        try:
            r, p = stats.spearmanr(df[col].fillna(df[col].median()), df["OOC"])
            print(f"  {col}: Spearman ρ={r:.4f} (p={p:.4f}) {'★ 주목' if abs(r) > 0.05 and p < 0.05 else ''}")
        except Exception as e:
            print(f"  {col}: error {e}")
print()

print("=" * 60)
print("CRITICAL CHECK 9: 이상치(Outlier) 영향")
print("=" * 60)
for col in SAW_PARAMS + ["SAW_Xbar_um", "SAW_Range_um"]:
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    out_n = ((df[col] < q1 - 3*iqr) | (df[col] > q3 + 3*iqr)).sum()
    print(f"  {col}: IQR 3배 이상치 {out_n}건 ({out_n/len(df)*100:.2f}%)")
print()

print("=" * 60)
print("CRITICAL CHECK 10: 고OOC LOT vs Primary_Defect_Cause 연계")
print("=" * 60)
pdc_ooc = ooc_df["Primary_Defect_Cause"].value_counts(normalize=True) * 100
pdc_ic  = ic_df["Primary_Defect_Cause"].value_counts(normalize=True) * 100
print("OOC군 Primary_Defect_Cause 분포:")
for k, v in pdc_ooc.items():
    ic_v = pdc_ic.get(k, 0)
    print(f"  {k}: OOC={v:.1f}% vs IC={ic_v:.1f}% (차이={v-ic_v:.1f}%p)")
