import pandas as pd

df = pd.read_csv('train_cleaned.csv')

# 상관관계 계산
target = 'resist_line_cd_nm'
corr_results = {}
for tone in ['POSITIVE', 'NEGATIVE']:
    tone_df = df[df['pr_tone'] == tone].select_dtypes(include=['float64', 'int64'])
    tone_df = tone_df.loc[:, tone_df.std() > 0]
    corr = tone_df.corr()[target].drop(target)
    top5 = corr.abs().sort_values(ascending=False).head(5)
    corr_results[tone] = [(f, corr[f]) for f in top5.index]

head_html = df.head(5).to_html(classes='data-table', index=False, border=0)

html_template = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>반도체 PR 공정 데이터 분석 리포트</title>
<meta name="description" content="PR Tone에 따른 데이터 샘플 및 상관관계 분석">
<meta name="theme-color" content="#f3f4f0">
<style>
:root{{
  --bg:#f3f4f0; --bg-soft:#eceee8; --surface:#ffffff; --surface-2:#eef0ea;
  --border:#d9dcd3; --border-soft:#e3e5de; --text:#12161f; --text-dim:#5b6270;
  --text-faint:#8a92a0; --amber:#a3610a; --amber-soft:#f7ead2;
  --cyan:#0a7a86; --cyan-soft:#dcf0f0; --danger:#b23c2c; --good:#2e7d52;
  --radius-s:6px; --radius-m:10px; --radius-l:16px; --maxw:1180px;
  --font-display:-apple-system,BlinkMacSystemFont,"Segoe UI","Pretendard",sans-serif;
  --font-body:-apple-system,BlinkMacSystemFont,"Segoe UI","Pretendard",sans-serif;
  --font-mono:"SF Mono",monospace;
}}
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{background:var(--bg);color:var(--text);font-family:var(--font-body);line-height:1.6;}}
a{{color:inherit;text-decoration:none;}}
.wrap{{max-width:var(--maxw);margin:0 auto;padding:0 28px;position:relative;z-index:1;}}
.eyebrow{{font-family:var(--font-mono);font-size:12.5px;letter-spacing:0.14em;color:var(--amber);display:flex;align-items:center;gap:10px;margin-bottom:14px;}}
.eyebrow::before{{content:"";width:22px;height:1px;background:var(--amber);display:inline-block;}}
.eyebrow.cyan{{color:var(--cyan);}}
.eyebrow.cyan::before{{background:var(--cyan);}}
h2.section-title{{font-size:clamp(26px,3.6vw,40px);line-height:1.22;margin-bottom:14px;}}
h2.section-title em{{font-style:normal;color:var(--amber);}}
p.section-lede{{color:var(--text-dim);font-size:16px;max-width:760px;line-height:1.75;margin-bottom:40px;}}
section{{padding:60px 0;border-bottom:1px solid var(--border-soft);}}
.section-head{{margin-bottom:44px;}}
header.site-nav{{position:sticky;top:0;z-index:50;background:rgba(243,244,240,0.88);border-bottom:1px solid var(--border-soft);}}
.nav-inner{{max-width:var(--maxw);margin:0 auto;padding:14px 28px;display:flex;align-items:center;}}
.nav-brand{{font-family:var(--font-mono);font-size:13px;color:var(--text);display:flex;align-items:center;gap:10px;}}
.nav-brand .dot{{width:8px;height:8px;border-radius:50%;background:var(--cyan);}}
.grid-2{{display:grid;grid-template-columns:repeat(2,1fr);gap:16px;}}
@media (max-width:640px){{.grid-2{{grid-template-columns:1fr;}}}}
.card{{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-m);padding:22px 20px;}}
.card .tag{{font-family:var(--font-mono);font-size:10.5px;letter-spacing:.09em;color:var(--amber);margin-bottom:10px;display:block;}}
.card.cyan .tag{{color:var(--cyan);}}
.card h4{{font-size:16.5px;font-weight:700;margin-bottom:8px;color:var(--text);}}
.card p{{font-size:13.8px;color:var(--text-dim);}}
.card.border-l{{border-left:3px solid var(--amber);}}
.card.border-l.cy{{border-left:3px solid var(--cyan);}}
.check-list{{display:grid;gap:11px;margin:18px 0;}}
.check-list li{{display:flex;gap:10px;font-size:14px;color:var(--text-dim);}}
.check-list li::before{{content:"✓";color:var(--good);font-weight:700;}}

/* Table Styles */
.table-container {{ overflow-x: auto; background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-m); padding: 10px; }}
.data-table {{ width: 100%; border-collapse: collapse; font-size: 11px; font-family: var(--font-mono); white-space: nowrap; }}
.data-table th, .data-table td {{ border-bottom: 1px solid var(--border-soft); padding: 8px 12px; text-align: left; }}
.data-table th {{ background: var(--bg-soft); color: var(--text); font-weight: bold; border-bottom: 2px solid var(--border); }}
.data-table tbody tr:hover {{ background: var(--surface-2); }}
</style>
</head>
<body>

<header class="site-nav">
  <div class="nav-inner">
    <div class="nav-brand"><span class="dot"></span><span class="full">AI DATA ANALYSIS</span><span>· EDA Report</span></div>
  </div>
</header>

<section id="data-preview">
  <div class="wrap">
    <div class="section-head">
      <div class="eyebrow cyan">01 · DATA PREVIEW</div>
      <h2 class="section-title">전체 데이터 <em>5행 미리보기</em></h2>
      <p class="section-lede">결측치와 이상값이 제거된 전체 학습 데이터(train_cleaned.csv)의 상위 5개 행입니다. 이 데이터를 바탕으로 모델 타겟 변수인 CD값(resist_line_cd_nm)을 분석합니다.</p>
    </div>
    
    <div class="table-container">
      {head_html}
    </div>
  </div>
</section>

<section id="analysis">
  <div class="wrap">
    <div class="section-head">
      <div class="eyebrow cyan">02 · DATA ANALYSIS</div>
      <h2 class="section-title">PR Tone별 <em>CD 상관관계</em></h2>
      <p class="section-lede">Positive와 Negative PR은 특성이 정반대이므로 데이터셋을 분리하여 분석합니다. 타겟 변수인 임계 치수(resist_line_cd_nm)와 상관관계가 높은 주요 변수들입니다.</p>
    </div>
    
    <div class="grid-2">
      <div class="card border-l cy">
        <span class="tag cyan">POSITIVE PR</span>
        <h4>노광량(Dose) 증가 → CD 감소</h4>
        <ul class="check-list" style="margin-top:14px;">
"""

# POSITIVE correlations
for feature, val in corr_results['POSITIVE']:
    html_template += f"          <li><code>{feature}</code>: <b>{val:+.4f}</b></li>\n"

html_template += """        </ul>
        <p style="margin-top:14px;">Positive PR은 노광된 부분이 씻겨 나가므로, 빛(Dose)을 많이 받을수록 트렌치가 넓어지고 남겨지는 라인 CD는 작아지는 음(-)의 상관관계가 나타납니다.</p>
      </div>
      <div class="card border-l" style="border-left-color:var(--amber);">
        <span class="tag" style="color:var(--amber);">NEGATIVE PR</span>
        <h4>노광량(Dose) 증가 → CD 증가</h4>
        <ul class="check-list" style="margin-top:14px;">
"""

# NEGATIVE correlations
for feature, val in corr_results['NEGATIVE']:
    html_template += f"          <li><code>{feature}</code>: <b>{val:+.4f}</b></li>\n"

html_template += """        </ul>
        <p style="margin-top:14px;">Negative PR은 반대로 노광된 부분이 경화되어 남게 되므로, 빛(Dose)을 많이 받을수록 남겨지는 라인 CD가 두꺼워지는 양(+)의 상관관계가 나타납니다.</p>
      </div>
    </div>
  </div>
</section>

<footer style="padding: 40px; text-align: center; color: var(--text-faint); font-family: var(--font-mono); font-size: 12px;">
  Generated for Semiconductor AI Project
</footer>

</body>
</html>"""

with open('analysis_index.html', 'w', encoding='utf-8') as f:
    f.write(html_template)
