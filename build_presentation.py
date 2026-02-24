"""
Build HSSU Statistics Partnership Presentation
Generates index.html with course comparison, data analysis charts,
and R/data science benefits narrative.
"""
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
FOCUS_DL = ROOT / "focus_downloads"
AFFLUENCE = ROOT / "affluence_analysis_results"
OUT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _latest_clean_csv() -> Optional[Path]:
    candidates = list(FOCUS_DL.glob("final_grades_*_clean.csv"))
    if not candidates:
        return None
    # Pick the file with the most rows (most marking periods / most data)
    best, best_size = None, 0
    for p in candidates:
        sz = p.stat().st_size
        if sz > best_size:
            best, best_size = p, sz
    return best


def load_current_math_grades() -> pd.DataFrame:
    csv = _latest_clean_csv()
    if csv is None:
        return pd.DataFrame()
    df = pd.read_csv(csv)
    math_pat = r"Algebra|Geometry|Calculus|Pre\s?Calc|PreCalc"
    eca_pat = r"ECA.*(?:Algebra|Calculus|Precalculus)"
    mask = (df["Course"].str.contains(math_pat, case=False, na=False) |
            df["Course"].str.contains(eca_pat, case=False, na=False))
    return df[mask].copy()


def load_affluence_data() -> pd.DataFrame:
    p = AFFLUENCE / "processed_student_data.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def load_zip_income() -> pd.DataFrame:
    p = AFFLUENCE / "zip_code_analysis.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def load_historical_data() -> Optional[pd.DataFrame]:
    """Load historical math grades if available (from focus_login_hssu.py export-history)."""
    p = FOCUS_DL / "historical_math_grades.csv"
    if p.exists():
        return pd.read_csv(p)
    return None


def load_demographics() -> Optional[pd.DataFrame]:
    """Load student demographics if available (from focus_login_hssu.py export-demographics)."""
    p = FOCUS_DL / "student_demographics.csv"
    if p.exists():
        return pd.read_csv(p)
    return None


# ---------------------------------------------------------------------------
# Chart data builders
# ---------------------------------------------------------------------------

def math_grade_distribution(mdf: pd.DataFrame) -> dict:
    """Current year math grade distribution across marking periods."""
    periods = []
    for col_base in ["Quarter 1", "Quarter 2", "Semester 1", "Progress Report 3"]:
        ltr_col = f"{col_base} (Letter)"
        if ltr_col not in mdf.columns:
            continue
        dist = mdf[ltr_col].value_counts()
        periods.append({
            "label": col_base,
            "A": int(dist.get("A", 0)),
            "B": int(dist.get("B", 0)),
            "C": int(dist.get("C", 0)),
            "D": int(dist.get("D", 0)),
            "F": int(dist.get("F", 0)),
        })
    return {"periods": periods}


def math_grade_by_course(mdf: pd.DataFrame) -> dict:
    """Average math grade by course type."""
    # Use Semester 1 % for the most complete picture
    col = "Semester 1 (%)"
    if col not in mdf.columns:
        col = "Quarter 2 (%)"
    if col not in mdf.columns:
        return {"courses": []}

    # Simplify course names
    def simplify(name):
        name = re.sub(r"\s+1-[12]$", "", name)
        name = re.sub(r"\s+VC-CR$", "", name)
        name = re.sub(r"\s+DD$", "", name)
        return name.strip()

    mdf = mdf.copy()
    mdf["SimpleCourse"] = mdf["Course"].apply(simplify)
    agg = mdf.groupby("SimpleCourse")[col].agg(["mean", "count"]).reset_index()
    agg = agg[agg["count"] >= 5].sort_values("mean", ascending=False)
    courses = []
    for _, r in agg.iterrows():
        courses.append({
            "name": r["SimpleCourse"],
            "avg": round(float(r["mean"]), 1),
            "n": int(r["count"]),
        })
    return {"courses": courses, "col_used": col}


def _build_equity_merged(mdf: pd.DataFrame, adf: pd.DataFrame,
                         demo: pd.DataFrame) -> pd.DataFrame:
    """Merge math grades + affluence (income) + demographics into one frame."""
    if mdf.empty or adf.empty or demo is None or demo.empty:
        return pd.DataFrame()
    demo = demo.copy()
    demo["Race"] = demo.apply(_primary_race, axis=1)
    pct_col = "Semester 1 (%)"
    if pct_col not in mdf.columns:
        pct_col = "Quarter 2 (%)"
    if pct_col not in mdf.columns:
        return pd.DataFrame()
    ltr_col = pct_col.replace("(%)", "(Letter)")
    m = mdf.merge(adf[["ID", "MedianZipIncome", "ZipCode"]],
                  left_on="Student ID", right_on="ID", how="inner")
    m = m.merge(demo[["StudentID", "Race"]],
                left_on="Student ID", right_on="StudentID", how="inner")
    m = m.dropna(subset=[pct_col, "MedianZipIncome"])
    m["_pct"] = m[pct_col]
    m["_ltr"] = m.get(ltr_col)
    return m


def math_score_by_income(merged: pd.DataFrame) -> dict:
    """Chart 1: Average math score by household-income quartile."""
    if merged.empty:
        return {"quartiles": [], "correlation": 0}
    corr = float(merged["MedianZipIncome"].corr(merged["_pct"]))
    m = merged.copy()
    m["Q"] = pd.qcut(m["MedianZipIncome"], q=4, duplicates="drop")
    agg = m.groupby("Q", observed=True).agg(
        avg=("_pct", "mean"), n=("_pct", "count"),
    ).reset_index()
    quartiles = []
    for _, r in agg.iterrows():
        lo, hi = r["Q"].left, r["Q"].right
        quartiles.append({
            "label": f"${int(lo/1000)}K\u2013${int(hi/1000)}K",
            "avg": round(float(r["avg"]), 1),
            "n": int(r["n"]),
        })
    return {"quartiles": quartiles, "correlation": round(corr, 3)}


def math_df_by_income(merged: pd.DataFrame) -> dict:
    """Chart 2: Math D/F rate by household-income quartile."""
    if merged.empty or "_ltr" not in merged.columns:
        return {"quartiles": []}
    m = merged.dropna(subset=["_ltr"]).copy()
    m["Q"] = pd.qcut(m["MedianZipIncome"], q=4, duplicates="drop")
    quartiles = []
    for q, grp in m.groupby("Q", observed=True):
        total = len(grp)
        df_n = int(grp["_ltr"].isin(["D", "F"]).sum())
        lo, hi = q.left, q.right
        quartiles.append({
            "label": f"${int(lo/1000)}K\u2013${int(hi/1000)}K",
            "df_rate": round(float(df_n) / total * 100, 1),
            "df_count": df_n,
            "n": total,
        })
    return {"quartiles": quartiles}


def race_by_income_quartile(merged: pd.DataFrame) -> dict:
    """Chart 3: Racial composition (%) of each income quartile — stacked bar."""
    if merged.empty:
        return {"quartiles": [], "races": []}
    m = merged.drop_duplicates(subset=["Student ID"]).copy()
    m["Q"] = pd.qcut(m["MedianZipIncome"], q=4, duplicates="drop")
    races = [r for r in m["Race"].value_counts().index if m[m["Race"] == r].shape[0] >= 3]
    quartiles = []
    for q, grp in m.groupby("Q", observed=True):
        lo, hi = q.left, q.right
        total = len(grp)
        entry = {"label": f"${int(lo/1000)}K\u2013${int(hi/1000)}K", "n": total}
        for race in races:
            entry[race] = round(float((grp["Race"] == race).sum()) / total * 100, 1)
        quartiles.append(entry)
    return {"quartiles": quartiles, "races": races}


def df_rate_by_course(mdf: pd.DataFrame) -> dict:
    """D/F rate per course for current year."""
    col = "Semester 1 (Letter)"
    if col not in mdf.columns:
        col = "Quarter 2 (Letter)"
    if col not in mdf.columns:
        return {"courses": []}

    def simplify(name):
        name = re.sub(r"\s+1-[12]$", "", name)
        name = re.sub(r"\s+VC-CR$", "", name)
        name = re.sub(r"\s+DD$", "", name)
        return name.strip()

    mdf = mdf.copy()
    mdf["SimpleCourse"] = mdf["Course"].apply(simplify)
    results = []
    for course, grp in mdf.groupby("SimpleCourse"):
        total = grp[col].notna().sum()
        if total < 5:
            continue
        df_count = grp[col].isin(["D", "F"]).sum()
        results.append({
            "name": course,
            "df_rate": round(float(df_count / total * 100), 1),
            "df_count": int(df_count),
            "total": int(total),
        })
    results.sort(key=lambda x: x["df_rate"], reverse=True)
    return {"courses": results}


def _primary_race(row):
    """Derive a single race label from demographic flag columns."""
    if row.get("Single_Ethnicity") == "Hispanic":
        return "Hispanic/Latino"
    if row.get("Race_Black") == "Yes" and row.get("Race_White") == "Yes":
        return "Two or More"
    if row.get("Race_Black") == "Yes":
        return "Black"
    if row.get("Race_White") == "Yes":
        return "White"
    if row.get("Race_Asian") == "Yes":
        return "Asian"
    if row.get("Race_AmIndian") == "Yes":
        return "Am. Indian/Alaska Native"
    if row.get("Race_Pacific") == "Yes":
        return "Pacific Islander"
    eth = str(row.get("Single_Ethnicity", "")).strip()
    if eth and eth != "nan":
        return eth
    return "Unknown"


def math_df_by_race(merged: pd.DataFrame) -> dict:
    """Chart 4: Math D/F rate per racial group — uses the equity merged frame."""
    if merged.empty or "_ltr" not in merged.columns:
        return {"groups": []}
    m = merged.dropna(subset=["_ltr"])
    groups = []
    for race, grp in m.groupby("Race"):
        total = len(grp)
        if total < 5:
            continue
        df_n = int(grp["_ltr"].isin(["D", "F"]).sum())
        groups.append({
            "race": race,
            "df_rate": round(float(df_n) / total * 100, 1),
            "df_count": df_n,
            "n": total,
        })
    groups.sort(key=lambda g: g["df_rate"], reverse=True)
    return {"groups": groups}


# ---------------------------------------------------------------------------
# HTML slide template
# ---------------------------------------------------------------------------

def build_html(chart_data: dict) -> str:
    data_json = json.dumps(chart_data)
    return r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HSSU Statistics Partnership | Collegiate School of Medicine &amp; Bioscience</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
  :root {
    --navy: #0a1628; --navy2: #111d33; --navy3: #18263e;
    --gold: #c9a227; --gold-light: #e8d07a; --gold-dim: #c9a22730;
    --accent: #4a9eff; --accent2: #2d7dd2;
    --text: #e8ecf1; --text2: #a0aec0; --text3: #718096;
    --danger: #fc5c65; --warn: #f7b731; --ok: #26de81;
    --card: #15202e; --card2: #1a2940;
    --radius: 12px;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  html { font-size:16px; }
  body { font-family:'Inter',sans-serif; background:var(--navy); color:var(--text); line-height:1.6; height:100vh; overflow:hidden; }

  /* Navigation */
  nav { position:fixed; top:0; left:0; right:0; z-index:100; background:rgba(10,22,40,.92); backdrop-filter:blur(12px); border-bottom:1px solid rgba(201,162,39,.15); padding:12px 24px; display:flex; justify-content:space-between; align-items:center; }
  nav .brand { font-size:.9rem; font-weight:700; color:var(--gold); letter-spacing:.03em; }
  nav .dots { display:flex; gap:8px; }
  nav .dot { width:10px; height:10px; border-radius:50%; border:2px solid var(--text3); cursor:pointer; transition:all .2s; }
  nav .dot.on { background:var(--gold); border-color:var(--gold); }
  nav .dot:hover { border-color:var(--gold); }

  /* Slides */
  .slide { height:100vh; padding:48px 4vw 10px; display:flex; flex-direction:column; justify-content:center; position:absolute; top:0; left:0; width:100%; overflow:hidden; opacity:0; pointer-events:none; transition:opacity .35s ease; }
  .slide.active { opacity:1; pointer-events:auto; z-index:1; }
  .slide:nth-child(even) { background:var(--navy2); }

  /* Title slide */
  .hero { text-align:center; }
  .hero .tag { display:inline-block; background:var(--gold-dim); color:var(--gold); padding:4px 14px; border-radius:20px; font-size:.7rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; margin-bottom:10px; }
  .hero h1 { font-size:clamp(1.6rem,4vw,2.8rem); font-weight:900; line-height:1.1; margin-bottom:8px; }
  .hero h1 span { color:var(--gold); }
  .hero .sub { font-size:.92rem; color:var(--text2); max-width:700px; margin:0 auto 16px; line-height:1.5; }
  .hero .schools { display:flex; justify-content:center; gap:40px; flex-wrap:wrap; margin-top:12px; }
  .hero .school { text-align:center; }
  .hero .school .name { font-weight:700; font-size:1rem; }
  .hero .school .role { font-size:.78rem; color:var(--text3); }

  /* Section headers */
  .sec-hdr { margin-bottom:10px; }
  .sec-hdr .num { font-size:.68rem; color:var(--gold); font-weight:700; letter-spacing:.1em; text-transform:uppercase; }
  .sec-hdr h2 { font-size:clamp(1.2rem,2.5vw,1.7rem); font-weight:800; margin-top:2px; }
  .sec-hdr p { color:var(--text2); max-width:640px; margin-top:2px; font-size:.82rem; line-height:1.4; }

  /* Cards grid */
  .grid2 { display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:10px; }
  .grid3 { display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:10px; }

  /* Card */
  .card { background:var(--card); border:1px solid rgba(255,255,255,.06); border-radius:var(--radius); padding:12px 14px; transition:transform .2s, box-shadow .2s; }
  .card:hover { transform:translateY(-2px); box-shadow:0 8px 30px rgba(0,0,0,.3); }
  .card h3 { font-size:.85rem; font-weight:700; margin-bottom:4px; }
  .card .lbl { font-size:.72rem; color:var(--text3); font-weight:600; text-transform:uppercase; letter-spacing:.05em; }

  /* Course comparison table */
  .cmp-tbl { width:100%; border-collapse:collapse; font-size:.72rem; margin-top:6px; }
  .cmp-tbl th { background:var(--card2); padding:4px 8px; text-align:left; font-size:.64rem; text-transform:uppercase; letter-spacing:.05em; color:var(--text3); font-weight:700; border-bottom:2px solid var(--gold-dim); }
  .cmp-tbl td { padding:4px 8px; border-bottom:1px solid rgba(255,255,255,.04); vertical-align:top; }
  .cmp-tbl tr:hover td { background:rgba(74,158,255,.04); }
  .cmp-tbl .course-name { font-weight:700; color:var(--gold-light); }
  .cmp-tbl .check { color:var(--ok); font-weight:700; }
  .cmp-tbl .x { color:var(--danger); font-weight:700; }

  /* Stat boxes */
  .stat-box { text-align:center; padding:10px; }
  .stat-box .big { font-size:1.8rem; font-weight:900; }
  .stat-box .big.gold { color:var(--gold); }
  .stat-box .big.danger { color:var(--danger); }
  .stat-box .big.ok { color:var(--ok); }
  .stat-box .desc { font-size:.75rem; color:var(--text2); margin-top:2px; }

  /* Chart containers */
  .chart-wrap { position:relative; width:100%; max-width:700px; margin:4px auto; }
  .chart-wrap.wide { max-width:900px; }
  .chart-wrap canvas { width:100% !important; max-height:200px; }

  /* Benefit list */
  .benefits { list-style:none; }
  .benefits li { padding:4px 0; border-bottom:1px solid rgba(255,255,255,.04); display:flex; gap:8px; align-items:flex-start; }
  .benefits li:last-child { border-bottom:none; }
  .benefits .icon { font-size:1rem; flex-shrink:0; margin-top:1px; }
  .benefits .txt h4 { font-size:.8rem; font-weight:700; }
  .benefits .txt p { font-size:.72rem; color:var(--text2); margin-top:0; line-height:1.35; }

  /* Timeline */
  .timeline { position:relative; padding-left:30px; }
  .timeline::before { content:''; position:absolute; left:8px; top:0; bottom:0; width:2px; background:var(--gold-dim); }
  .timeline .step { position:relative; margin-bottom:10px; }
  .timeline .step::before { content:''; position:absolute; left:-26px; top:4px; width:12px; height:12px; border-radius:50%; background:var(--gold); border:3px solid var(--navy); }
  .timeline .step h4 { font-weight:700; font-size:.82rem; }
  .timeline .step p { font-size:.72rem; color:var(--text2); margin-top:0; line-height:1.35; }

  /* Footer */
  .foot { display:none; }
  .foot a { color:var(--gold); text-decoration:none; }

  /* Pill/badge */
  .pill { display:inline-block; padding:3px 10px; border-radius:10px; font-size:.7rem; font-weight:700; }
  .pill.r { background:rgba(38,222,129,.12); color:var(--ok); }
  .pill.py { background:rgba(74,158,255,.12); color:var(--accent); }
  .pill.gold { background:var(--gold-dim); color:var(--gold); }

  /* Slide nav buttons (in header) */
  .slide-nav { display:flex; align-items:center; gap:8px; }
  .slide-nav button { width:30px; height:30px; border-radius:50%; border:1.5px solid var(--gold-dim); background:transparent; color:var(--gold); font-size:.85rem; cursor:pointer; display:flex; align-items:center; justify-content:center; transition:all .2s; }
  .slide-nav button:hover { background:var(--gold); color:var(--navy); border-color:var(--gold); }
  .slide-nav button:disabled { opacity:.25; cursor:default; pointer-events:none; }
  .slide-nav .counter { font-size:.72rem; color:var(--text3); font-weight:600; min-width:36px; text-align:center; }

  /* Responsive */
  @media (max-width:640px) {
    .slide { padding:48px 12px 8px; height:100vh; }
    .grid2, .grid3 { grid-template-columns:1fr; }
    .slide-nav button { width:26px; height:26px; font-size:.75rem; }
  }
</style>
</head>
<body>

<!-- Navigation -->
<nav>
  <div class="brand">HSSU &times; CSMB Statistics Partnership</div>
  <div style="display:flex;align-items:center;gap:16px">
    <div class="dots" id="nav-dots"></div>
    <div class="slide-nav" id="slide-nav"></div>
  </div>
</nav>

<!-- ===== SLIDE 1: Title ===== -->
<section class="slide hero" id="s0">
  <div class="tag">Dual-Credit Partnership Proposal</div>
  <h1>Bringing <span>Statistics &amp; Data Science</span><br>to Collegiate Students</h1>
  <div class="sub">A partnership between Harris-Stowe State University and Collegiate School of Medicine &amp; Bioscience to offer college-credit statistics courses with R programming and data science applications</div>
  <div class="schools">
    <div class="school"><div class="name">Harris-Stowe State University</div><div class="role">HBCU &middot; Statistics &amp; Data Science Faculty</div></div>
    <div class="school"><div class="name">Collegiate School of Medicine &amp; Bioscience</div><div class="role">SLPS Magnet &middot; STEM Focus</div></div>
  </div>
</section>

<!-- ===== SLIDE 2: Why Statistics ===== -->
<section class="slide" id="s1">
  <div class="sec-hdr">
    <div class="num">01 &mdash; The Opportunity</div>
    <h2>Why Statistics Matters for Collegiate Students</h2>
    <p>Statistics is foundational to biomedical research, public health, and data-driven decision making &mdash; all core to CSMB's mission.</p>
  </div>
  <div class="grid3">
    <div class="card stat-box"><div class="big gold">86%</div><div class="desc">of STEM careers require statistical literacy (BLS, 2024)</div></div>
    <div class="card stat-box"><div class="big gold">$98K</div><div class="desc">median salary for statisticians &amp; data scientists (BLS)</div></div>
    <div class="card stat-box"><div class="big gold">35%</div><div class="desc">projected job growth in data science through 2032</div></div>
  </div>
  <div class="grid2" style="margin-top:8px">
    <div class="card">
      <h3>College Credit Advantage</h3>
      <ul class="benefits">
        <li><span class="icon">&#x1F393;</span><div class="txt"><h4>Earn 3-4 college credits</h4><p>Transfer-ready credits from an accredited HBCU, satisfying general education math requirements at most universities</p></div></li>
        <li><span class="icon">&#x1F4B0;</span><div class="txt"><h4>Save $1,500+ in tuition</h4><p>College statistics courses cost $500-$1,500+; dual-credit offers significant savings for families</p></div></li>
        <li><span class="icon">&#x1F52C;</span><div class="txt"><h4>Research-ready skills</h4><p>Statistical analysis is required for PLTW Biomedical capstone projects and summer research internships</p></div></li>
      </ul>
    </div>
    <div class="card">
      <h3>CSMB Curriculum Alignment</h3>
      <ul class="benefits">
        <li><span class="icon">&#x1F9EC;</span><div class="txt"><h4>Biomedical Science Pathway</h4><p>Biostatistics directly supports Human Body Systems, Medical Interventions, and Biomedical Innovation coursework</p></div></li>
        <li><span class="icon">&#x1F4CA;</span><div class="txt"><h4>Data Literacy Across Subjects</h4><p>R programming skills transfer to AP Research, AP Environmental Science, and college-level science labs</p></div></li>
        <li><span class="icon">&#x1F91D;</span><div class="txt"><h4>HBCU Pipeline</h4><p>Build direct relationships with Harris-Stowe faculty, creating pathways for continued enrollment and mentorship</p></div></li>
      </ul>
    </div>
  </div>
</section>

<!-- ===== SLIDE 3: Course Comparison ===== -->
<section class="slide" id="s2">
  <div class="sec-hdr">
    <div class="num">02 &mdash; Course Comparison</div>
    <h2>Three Paths to Statistical Mastery</h2>
    <p>Comparing the standard community college statistics course with two HSSU options that include R programming and data science components.</p>
  </div>
  <div style="overflow-x:auto">
  <table class="cmp-tbl">
    <thead>
      <tr>
        <th>Feature</th>
        <th>STLCC MTH 180<br><small>Intro Statistics (MOTR MATH 110)</small></th>
        <th>HSSU STAT0260<br><small>Data Analysis &amp; Stats w/Lab</small></th>
        <th>HSSU MATH0301<br><small>Biostatistics</small></th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td class="lbl">Credits</td>
        <td>3</td>
        <td>4 (includes lab)</td>
        <td>3</td>
      </tr>
      <tr>
        <td class="lbl">Institution Type</td>
        <td>Community College</td>
        <td>HBCU (4-year university)</td>
        <td>HBCU (4-year university)</td>
      </tr>
      <tr>
        <td class="lbl">Descriptive Statistics</td>
        <td><span class="check">&#10003;</span> Data collection, organization, representation</td>
        <td><span class="check">&#10003;</span> Data analysis methods</td>
        <td><span class="check">&#10003;</span> Descriptive statistics with bio applications</td>
      </tr>
      <tr>
        <td class="lbl">Probability Theory</td>
        <td><span class="check">&#10003;</span> Elementary probability, distributions</td>
        <td><span class="check">&#10003;</span> Probability &amp; distributions</td>
        <td><span class="check">&#10003;</span> Probability in biological contexts</td>
      </tr>
      <tr>
        <td class="lbl">Inferential Statistics</td>
        <td><span class="check">&#10003;</span> Confidence intervals, hypothesis testing</td>
        <td><span class="check">&#10003;</span> Statistical inference</td>
        <td><span class="check">&#10003;</span> One &amp; two sample parameter inferences</td>
      </tr>
      <tr>
        <td class="lbl">Regression &amp; Correlation</td>
        <td><span class="check">&#10003;</span> Correlation &amp; regression analysis</td>
        <td><span class="check">&#10003;</span> Correlation &amp; regression</td>
        <td><span class="check">&#10003;</span> Simple regression, covariance, correlation</td>
      </tr>
      <tr>
        <td class="lbl">ANOVA</td>
        <td><span class="x">&#10007;</span></td>
        <td><span class="check">&#10003;</span></td>
        <td><span class="check">&#10003;</span> Analysis of Variance</td>
      </tr>
      <tr>
        <td class="lbl">Non-Parametric Methods</td>
        <td><span class="x">&#10007;</span></td>
        <td><span class="check">&#10003;</span></td>
        <td><span class="check">&#10003;</span> Non-parametric techniques</td>
      </tr>
      <tr>
        <td class="lbl">Experimental Design</td>
        <td><span class="x">&#10007;</span></td>
        <td><span class="check">&#10003;</span></td>
        <td><span class="check">&#10003;</span> Research design</td>
      </tr>
      <tr>
        <td class="lbl">R Programming</td>
        <td><span class="x">&#10007;</span> Calculator-based</td>
        <td><span class="check">&#10003;</span> <span class="pill r">R / RStudio</span></td>
        <td><span class="check">&#10003;</span> <span class="pill r">R</span> + SAS, SPSS, Excel</td>
      </tr>
      <tr>
        <td class="lbl">Data Science Lab</td>
        <td><span class="x">&#10007;</span></td>
        <td><span class="check">&#10003;</span> Integrated lab component</td>
        <td><span class="check">&#10003;</span> Software-based analysis</td>
      </tr>
      <tr>
        <td class="lbl">Biological Applications</td>
        <td><span class="x">&#10007;</span> General examples</td>
        <td>General data analysis</td>
        <td><span class="check">&#10003;</span> <strong>Biological Sciences focus</strong></td>
      </tr>
      <tr>
        <td class="lbl">Contingency Tables</td>
        <td><span class="x">&#10007;</span></td>
        <td><span class="check">&#10003;</span></td>
        <td><span class="check">&#10003;</span></td>
      </tr>
      <tr>
        <td class="lbl">MOTR Transfer</td>
        <td><span class="check">&#10003;</span> MOTR MATH 110</td>
        <td>University credit</td>
        <td>University credit</td>
      </tr>
      <tr>
        <td class="lbl">Instructor</td>
        <td>Various</td>
        <td>Ann Podleski, Ph.D.</td>
        <td>Ann Podleski, Ph.D.</td>
      </tr>
    </tbody>
  </table>
  </div>
  <div class="grid3" style="margin-top:8px">
    <div class="card" style="border-left:3px solid var(--gold)">
      <div class="lbl">Key Advantage</div>
      <h3>R Programming &amp; Lab</h3>
      <p style="font-size:.78rem;color:var(--text2);line-height:1.35">Both HSSU courses include hands-on statistical software (R, SAS, SPSS) that MTH 180 lacks entirely. Students learn to code real data analyses.</p>
    </div>
    <div class="card" style="border-left:3px solid var(--ok)">
      <div class="lbl">Key Advantage</div>
      <h3>Biomedical Focus</h3>
      <p style="font-size:.78rem;color:var(--text2);line-height:1.35">MATH0301 Biostatistics is specifically designed for Biological Sciences students &mdash; a perfect fit for CSMB's biomedical curriculum.</p>
    </div>
    <div class="card" style="border-left:3px solid var(--accent)">
      <div class="lbl">Key Advantage</div>
      <h3>Advanced Methods</h3>
      <p style="font-size:.78rem;color:var(--text2);line-height:1.35">ANOVA, non-parametric tests, and experimental design go beyond the standard intro stats course, preparing students for college research.</p>
    </div>
  </div>
</section>

<!-- ===== SLIDE 4: Current Math Performance ===== -->
<section class="slide" id="s3">
  <div class="sec-hdr">
    <div class="num">03 &mdash; Math Performance at CSMB</div>
    <h2>Current Year Math Grade Analysis</h2>
    <p>Data from Focus SIS showing math achievement patterns across courses and marking periods at Collegiate School of Medicine &amp; Bioscience.</p>
  </div>
  <div class="grid2">
    <div class="card">
      <h3>Grade Distribution by Marking Period</h3>
      <div class="chart-wrap"><canvas id="chartGradeDist"></canvas></div>
    </div>
    <div class="card">
      <h3>D/F Rate by Math Course</h3>
      <div class="chart-wrap"><canvas id="chartDFRate"></canvas></div>
    </div>
  </div>
  <div class="grid2" style="margin-top:8px">
    <div class="card">
      <h3>Average Score by Course</h3>
      <div class="chart-wrap"><canvas id="chartAvgCourse"></canvas></div>
    </div>
    <div class="card">
      <h3>Key Insight</h3>
      <div style="padding:10px 0">
        <p style="font-size:.78rem;color:var(--text2);margin-bottom:8px;line-height:1.35">Students in lower-level math courses show significantly higher D/F rates, indicating a need for <strong style="color:var(--gold)">alternative pathways</strong> that build statistical thinking through applied, real-world data analysis rather than abstract algebraic manipulation.</p>
        <p style="font-size:.78rem;color:var(--text2);margin-bottom:8px;line-height:1.35"><strong style="color:var(--accent)">Statistics courses with R programming</strong> provide a concrete, applied approach to mathematics that engages students through hands-on data exploration &mdash; particularly effective for students who struggle with traditional algebra sequences.</p>
        <p style="font-size:.78rem;color:var(--text2);line-height:1.35">The biostatistics pathway directly connects mathematical concepts to the biomedical curriculum students are already passionate about.</p>
      </div>
    </div>
  </div>
</section>

<!-- ===== SLIDE 5: Equity & Access ===== -->
<section class="slide" id="s4">
  <div class="sec-hdr">
    <div class="num">04 &mdash; Equity &amp; Access</div>
    <h2>Income, Race &amp; Math Performance</h2>
    <p>Does zip-code income predict math outcomes? Who are the students in each income bracket? The data connects poverty, demographics, and achievement.</p>
  </div>
  <div class="grid2">
    <div class="card">
      <h3>Avg Math Score by Income Quartile</h3>
      <div class="chart-wrap"><canvas id="chartScoreIncome"></canvas></div>
    </div>
    <div class="card">
      <h3>Math D/F Rate by Income Quartile</h3>
      <div class="chart-wrap"><canvas id="chartDFIncome"></canvas></div>
    </div>
  </div>
  <div class="grid2" style="margin-top:8px">
    <div class="card">
      <h3>Who Lives in Each Income Bracket?</h3>
      <div class="chart-wrap"><canvas id="chartRaceIncome"></canvas></div>
    </div>
    <div class="card">
      <h3>Math D/F Rate by Race</h3>
      <div class="chart-wrap"><canvas id="chartDFRace"></canvas></div>
    </div>
  </div>
  <div class="card" style="margin-top:8px">
    <h3>The Story These Charts Tell</h3>
    <div class="grid3" style="margin-top:6px">
      <div>
        <p style="font-size:.74rem;color:var(--text2);line-height:1.35"><strong style="color:var(--danger)">Lower-income zip codes &rarr; lower math scores &amp; higher D/F rates.</strong> Students from economically disadvantaged neighborhoods face steeper barriers to math success.</p>
      </div>
      <div>
        <p style="font-size:.74rem;color:var(--text2);line-height:1.35"><strong style="color:var(--gold)">Lower-income brackets are disproportionately students of color.</strong> The income gap and the racial achievement gap are the same gap. On-site HSSU instruction removes transportation and cost barriers.</p>
      </div>
      <div>
        <p style="font-size:.74rem;color:var(--text2);line-height:1.35"><strong style="color:var(--ok)">An HBCU partnership addresses both.</strong> Harris-Stowe brings culturally responsive pedagogy, affordable dual-credit, and a direct pathway &mdash; meeting students where they are.</p>
      </div>
    </div>
  </div>
</section>

<!-- ===== SLIDE 6: R & Data Science Benefits ===== -->
<section class="slide" id="s5">
  <div class="sec-hdr">
    <div class="num">05 &mdash; R Programming &amp; Data Science</div>
    <h2>Why R Changes Everything</h2>
    <p>The R programming language is the gold standard for statistical computing in biomedical research, and it's what sets the HSSU courses apart.</p>
  </div>
  <div class="grid2">
    <div class="card">
      <h3>R in the CSMB Curriculum</h3>
      <ul class="benefits">
        <li><span class="icon">&#x1F9EC;</span><div class="txt"><h4>PLTW Biomedical Capstone</h4><p>Students can use R to analyze experimental data for their capstone projects, producing publication-quality statistical analyses and visualizations</p></div></li>
        <li><span class="icon">&#x1F3E5;</span><div class="txt"><h4>Summer Research Internships</h4><p>Washington University, SLU, and BJC health system internships expect statistical software proficiency. R is the #1 tool in biomedical research.</p></div></li>
        <li><span class="icon">&#x1F4DD;</span><div class="txt"><h4>AP Research &amp; Science Courses</h4><p>R enables sophisticated data analysis for AP Research papers and AP Environmental Science field studies that go far beyond spreadsheet calculations</p></div></li>
        <li><span class="icon">&#x1F30D;</span><div class="txt"><h4>Health Equity Analysis</h4><p>Students can analyze real public health datasets &mdash; COVID disparities, environmental justice, food access &mdash; connecting stats to their community</p></div></li>
      </ul>
    </div>
    <div class="card">
      <h3>What Students Will Learn in R</h3>
      <div style="background:var(--navy);border-radius:8px;padding:10px;margin-bottom:8px;font-family:monospace;font-size:.7rem;color:var(--accent);overflow-x:auto">
        <pre style="margin:0;white-space:pre-wrap"><span style="color:var(--text3)"># Load patient data</span>
library(tidyverse)
patients &lt;- read_csv("clinical_trial.csv")

<span style="color:var(--text3)"># Descriptive statistics</span>
patients %&gt;%
  group_by(treatment_group) %&gt;%
  summarise(
    mean_bp = mean(blood_pressure),
    sd_bp   = sd(blood_pressure),
    n       = n()
  )

<span style="color:var(--text3)"># Hypothesis test: does treatment lower BP?</span>
t.test(blood_pressure ~ treatment_group,
       data = patients)

<span style="color:var(--text3)"># Visualize results</span>
ggplot(patients, aes(x = treatment_group,
                     y = blood_pressure,
                     fill = treatment_group)) +
  geom_boxplot() +
  labs(title = "Blood Pressure by Treatment Group")</pre>
      </div>
      <div class="grid3">
        <div style="text-align:center"><div class="pill r" style="font-size:.8rem;padding:6px 14px">tidyverse</div><div style="font-size:.75rem;color:var(--text3);margin-top:4px">Data wrangling</div></div>
        <div style="text-align:center"><div class="pill r" style="font-size:.8rem;padding:6px 14px">ggplot2</div><div style="font-size:.75rem;color:var(--text3);margin-top:4px">Visualization</div></div>
        <div style="text-align:center"><div class="pill r" style="font-size:.8rem;padding:6px 14px">rmarkdown</div><div style="font-size:.75rem;color:var(--text3);margin-top:4px">Reports &amp; papers</div></div>
      </div>
    </div>
  </div>
  <div class="card" style="margin-top:8px">
    <h3>Skills Comparison: Calculator vs. R Programming</h3>
    <table class="cmp-tbl">
      <thead><tr><th>Skill</th><th>Calculator-Based (MTH 180)</th><th>R Programming (HSSU)</th><th>Career Impact</th></tr></thead>
      <tbody>
        <tr><td>Data import</td><td>Manual entry (small datasets)</td><td>Read CSV, databases, APIs</td><td>Handle real-world data at scale</td></tr>
        <tr><td>Visualization</td><td>Basic histograms on screen</td><td>Publication-quality charts (ggplot2)</td><td>Communicate findings effectively</td></tr>
        <tr><td>Reproducibility</td><td>Re-enter everything each time</td><td>Scripts run again with one click</td><td>Scientific rigor &amp; collaboration</td></tr>
        <tr><td>Automation</td><td>Not possible</td><td>Batch process 1000s of datasets</td><td>Efficiency in research labs</td></tr>
        <tr><td>Resume value</td><td>Not listed</td><td>"Proficient in R" is a major asset</td><td>Immediate employability signal</td></tr>
      </tbody>
    </table>
  </div>
</section>

<!-- ===== SLIDE 7: Implementation ===== -->
<section class="slide" id="s6">
  <div class="sec-hdr">
    <div class="num">06 &mdash; Implementation Roadmap</div>
    <h2>Making It Happen</h2>
    <p>A phased approach to launching the HSSU statistics partnership at Collegiate School of Medicine &amp; Bioscience.</p>
  </div>
  <div class="grid2">
    <div class="card">
      <h3>Proposed Course Sequence</h3>
      <div class="timeline">
        <div class="step"><h4>Junior Year &mdash; Fall Semester</h4><p><strong>STAT0260: Data Analysis &amp; Statistics w/Lab</strong> (4 credits) &mdash; Foundational statistics with R programming. Students learn data wrangling, visualization, and inference while building coding skills.</p></div>
        <div class="step"><h4>Junior Year &mdash; Spring Semester</h4><p><strong>MATH0301: Biostatistics</strong> (3 credits) &mdash; Advanced statistical methods applied to biological sciences. ANOVA, experimental design, and non-parametric tests using R, SAS, and SPSS.</p></div>
        <div class="step"><h4>Senior Year</h4><p><strong>Apply skills to PLTW Capstone, AP Research, and summer internships.</strong> Students enter college with 7 credits and proven data science competency.</p></div>
      </div>
    </div>
    <div class="card">
      <h3>Logistics &amp; Requirements</h3>
      <ul class="benefits">
        <li><span class="icon">&#x1F468;&#x200D;&#x1F3EB;</span><div class="txt"><h4>Instructor: Dr. Ann Podleski</h4><p>HSSU faculty member already teaching both courses. Can deliver instruction on-site at CSMB or via hybrid model.</p></div></li>
        <li><span class="icon">&#x1F4BB;</span><div class="txt"><h4>Technology: R &amp; RStudio</h4><p>Free, open-source software. Runs on existing CSMB laptops. RStudio Cloud available as backup for any hardware issues.</p></div></li>
        <li><span class="icon">&#x1F4C5;</span><div class="txt"><h4>Schedule: 4 days/week</h4><p>STAT0260 meets Mon-Thu 1:00-1:50 PM. Can be adapted to CSMB bell schedule. MATH0301 meets Mon/Wed 2:00-3:15 PM.</p></div></li>
        <li><span class="icon">&#x1F4B5;</span><div class="txt"><h4>Cost: Dual-Credit Rate</h4><p>HSSU dual-credit tuition is significantly less than standard university tuition, making this accessible to all CSMB families.</p></div></li>
      </ul>
    </div>
  </div>
</section>

<!-- ===== SLIDE 8: People & Partnerships ===== -->
<section class="slide" id="s7">
  <div class="sec-hdr">
    <div class="num">07 &mdash; The People Behind the Partnership</div>
    <h2>Built on <span style="color:var(--gold)">Relationships</span>, Driven by Mission</h2>
    <p>This partnership is grounded in decades of collaboration between HSSU and SLPS &mdash; and personal relationships that make execution seamless.</p>
  </div>
  <div class="grid2">
    <div class="card" style="border-left:3px solid var(--gold)">
      <h3 style="margin-bottom:4px">Dr. Ann Podleski</h3>
      <div style="font-size:.7rem;color:var(--gold);font-weight:600;margin-bottom:4px">PROFESSOR OF MATHEMATICS &amp; DEPARTMENT CHAIR &middot; HARRIS-STOWE STATE UNIVERSITY</div>
      <ul class="benefits">
        <li><span class="icon" style="font-size:1.1rem">&#x1F393;</span><div class="txt"><h4>Ph.D. in Mathematics</h4><p>Extensive academic career in mathematics education with a focus on applied statistics and data analysis</p></div></li>
        <li><span class="icon" style="font-size:1.1rem">&#x1F3EB;</span><div class="txt"><h4>Department Chair</h4><p>Leads the HSSU Mathematics Department, overseeing curriculum for STAT0260 (Data Analysis &amp; Statistics w/Lab) and MATH0301 (Biostatistics)</p></div></li>
        <li><span class="icon" style="font-size:1.1rem">&#x1F52C;</span><div class="txt"><h4>STEM Research Mentor</h4><p>Has mentored HSSU students to national STEM competitions including the Emerging Researchers National Competition, integrating geospatial analysis and data science</p></div></li>
        <li><span class="icon" style="font-size:1.1rem">&#x1F4CA;</span><div class="txt"><h4>R, SAS &amp; SPSS Expert</h4><p>Teaches hands-on statistical computing across multiple platforms &mdash; exactly the skills CSMB students need for biomedical research careers</p></div></li>
      </ul>
      <div style="font-size:.7rem;color:var(--text3);margin-top:4px">PodleskA@hssu.edu &middot; (314) 340-3565 &middot; HGA 317</div>
    </div>
    <div class="card" style="border-left:3px solid var(--accent)">
      <h3 style="margin-bottom:4px">Dr. Harvey R. Fields, Jr.</h3>
      <div style="font-size:.7rem;color:var(--accent);font-weight:600;margin-bottom:4px">FOUNDING DEAN, COLLEGE OF STEM &middot; HARRIS-STOWE STATE UNIVERSITY</div>
      <ul class="benefits">
        <li><span class="icon" style="font-size:1.1rem">&#x1F3DB;</span><div class="txt"><h4>Morehouse &rarr; Georgia Tech &rarr; Washington University</h4><p>B.S. Chemistry (Morehouse College), B.S. Chemical Engineering (Georgia Tech), M.S. &amp; Ph.D. Chemistry (Washington University in St. Louis)</p></div></li>
        <li><span class="icon" style="font-size:1.1rem">&#x1F31F;</span><div class="txt"><h4>Former Associate Dean, WashU</h4><p>Served as Associate Dean for Student Success at Washington University before being named inaugural Dean of HSSU's College of STEM in 2023</p></div></li>
        <li><span class="icon" style="font-size:1.1rem">&#x1F3D7;</span><div class="txt"><h4>$62M STEM Building</h4><p>Leading the construction of HSSU's new $62 million STEM education facility, breaking ground January 2026 with completion expected summer 2027</p></div></li>
        <li><span class="icon" style="font-size:1.1rem">&#x1F91D;</span><div class="txt"><h4>SLPS Partnership Champion</h4><p>Deeply committed to creating STEM pathways for underrepresented students in the St. Louis region</p></div></li>
      </ul>
    </div>
  </div>
  <div class="card" style="margin-top:8px;border:1px solid var(--gold-dim)">
    <h3 style="margin-bottom:8px;text-align:center">The CSMB &harr; HSSU Connection</h3>
    <div class="grid3">
      <div style="text-align:center;padding:6px">
        <div style="font-size:1.4rem;margin-bottom:2px">&#x1F9D1;&#x200D;&#x1F3EB;</div>
        <h4 style="color:var(--gold);margin-bottom:3px;font-size:.82rem">Everett Stuckey</h4>
        <div style="font-size:.7rem;color:var(--text3);margin-bottom:4px">9th &amp; 10th Grade Counselor, CSMB</div>
        <p style="font-size:.74rem;color:var(--text2);line-height:1.35">Has known Dr. Fields for <strong style="color:var(--gold)">29 years</strong>. Has worked closely with Dr. Fields on student recruitment, grant development, internship placement for SLPS students, and growing HSSU STEM opportunities across the district.</p>
      </div>
      <div style="text-align:center;padding:6px">
        <div style="font-size:1.4rem;margin-bottom:2px">&#x1F393;</div>
        <h4 style="color:var(--accent);margin-bottom:3px;font-size:.82rem">Monica Nickolai</h4>
        <div style="font-size:.7rem;color:var(--text3);margin-bottom:4px">College Admissions Specialist, CSMB</div>
        <p style="font-size:.74rem;color:var(--text2);line-height:1.35">Previously worked for Dr. Fields at <strong style="color:var(--accent)">Washington University</strong>, bringing firsthand knowledge of his leadership approach and deep familiarity with university partnership logistics.</p>
      </div>
      <div style="text-align:center;padding:6px">
        <div style="font-size:1.4rem;margin-bottom:2px">&#x1F517;</div>
        <h4 style="color:var(--ok);margin-bottom:3px;font-size:.82rem">Why This Matters</h4>
        <div style="font-size:.7rem;color:var(--text3);margin-bottom:4px">Trusted Relationships = Fast Execution</div>
        <p style="font-size:.74rem;color:var(--text2);line-height:1.35">This isn't a cold partnership. Decades of personal and professional trust between CSMB staff and HSSU leadership means <strong style="color:var(--ok)">fewer barriers, faster decisions, and genuine investment</strong> in our students' success.</p>
      </div>
    </div>
  </div>
</section>

<!-- ===== SLIDE 9: Call to Action ===== -->
<section class="slide hero" id="s8" style="background:linear-gradient(135deg,var(--navy) 0%,var(--navy3) 100%)">
  <div class="tag">Next Steps</div>
  <h1>Let's Build the <span>Future of Statistics Education</span> at CSMB</h1>
  <div class="sub">Partner with Harris-Stowe State University to give Collegiate students the statistical thinking, R programming skills, and data science competency they need to excel in biomedical research and beyond.</div>
  <div class="grid3" style="max-width:900px;margin:16px auto 0">
    <div class="card stat-box" style="border:1px solid var(--gold-dim)">
      <div class="big gold">7</div>
      <div class="desc">College credits earned at CSMB</div>
    </div>
    <div class="card stat-box" style="border:1px solid var(--gold-dim)">
      <div class="big gold">2</div>
      <div class="desc">Software platforms mastered (R + SPSS)</div>
    </div>
    <div class="card stat-box" style="border:1px solid var(--gold-dim)">
      <div class="big gold">1</div>
      <div class="desc">HBCU partnership for life</div>
    </div>
  </div>
</section>

<div class="foot">
  HSSU Statistics Partnership Proposal &middot; Collegiate School of Medicine &amp; Bioscience &middot; SLPS<br>
  Data sourced from Focus SIS and U.S. Census Bureau ACS &middot; <a href="https://www.hssu.edu">hssu.edu</a>
</div>

<script>
const CD = """ + data_json + r""";
Chart.defaults.color = '#a0aec0';
Chart.defaults.borderColor = 'rgba(255,255,255,0.04)';
Chart.defaults.font.family = 'Inter, sans-serif';

// --- Grade Distribution Stacked Bar ---
!function(){
  const p = CD.grade_dist.periods;
  if(!p.length) return;
  new Chart(document.getElementById('chartGradeDist'), {
    type:'bar',
    data:{
      labels: p.map(d=>d.label),
      datasets:[
        {label:'A', data:p.map(d=>d.A), backgroundColor:'#26de81'},
        {label:'B', data:p.map(d=>d.B), backgroundColor:'#4a9eff'},
        {label:'C', data:p.map(d=>d.C), backgroundColor:'#f7b731'},
        {label:'D', data:p.map(d=>d.D), backgroundColor:'#fc8c3c'},
        {label:'F', data:p.map(d=>d.F), backgroundColor:'#fc5c65'},
      ]
    },
    options:{responsive:true, plugins:{legend:{position:'top',labels:{boxWidth:12,padding:8,font:{size:11}}}}, scales:{x:{stacked:true,grid:{display:false}},y:{stacked:true,title:{display:true,text:'Students'}}}}
  });
}();

// --- D/F Rate by Course ---
!function(){
  const c = CD.df_rate.courses;
  if(!c.length) return;
  new Chart(document.getElementById('chartDFRate'), {
    type:'bar',
    data:{
      labels: c.map(d=>d.name),
      datasets:[{
        label:'D/F Rate %',
        data:c.map(d=>d.df_rate),
        backgroundColor: c.map(d=>d.df_rate>30?'#fc5c65':d.df_rate>15?'#fc8c3c':'#26de81'),
        borderRadius:4,
      }]
    },
    options:{indexAxis:'y', responsive:true, plugins:{legend:{display:false}}, scales:{x:{title:{display:true,text:'D/F Rate (%)'},max:60,grid:{color:'rgba(255,255,255,0.04)'}},y:{grid:{display:false}}}}
  });
}();

// --- Avg Score by Course ---
!function(){
  const c = CD.avg_course.courses;
  if(!c.length) return;
  new Chart(document.getElementById('chartAvgCourse'), {
    type:'bar',
    data:{
      labels: c.map(d=>d.name),
      datasets:[{
        label:'Avg %',
        data:c.map(d=>d.avg),
        backgroundColor: c.map(d=>d.avg>=80?'#26de81':d.avg>=70?'#4a9eff':d.avg>=60?'#f7b731':'#fc5c65'),
        borderRadius:4,
      }]
    },
    options:{indexAxis:'y', responsive:true, plugins:{legend:{display:false}}, scales:{x:{title:{display:true,text:'Average Score (%)'},min:40,max:100,grid:{color:'rgba(255,255,255,0.04)'}},y:{grid:{display:false}}}}
  });
}();

// --- Chart 1: Avg Math Score by Income Quartile ---
!function(){
  const q = CD.score_income.quartiles;
  if(!q.length) return;
  const corr = CD.score_income.correlation;
  new Chart(document.getElementById('chartScoreIncome'), {
    type:'bar',
    data:{
      labels: q.map(d=>d.label),
      datasets:[{
        label:'Avg Math %',
        data:q.map(d=>d.avg),
        backgroundColor:['#fc5c65','#fc8c3c','#f7b731','#26de81'],
        borderRadius:4,
      }]
    },
    options:{responsive:true, plugins:{legend:{display:false},tooltip:{callbacks:{afterLabel:function(ctx){return q[ctx.dataIndex].n+' students'}}},subtitle:{display:true,text:'r = '+corr+' (income vs math score)',color:'#a0aec0',font:{size:10},padding:{bottom:4}}}, scales:{x:{title:{display:true,text:'Zip-Code Median Household Income'},grid:{display:false}},y:{title:{display:true,text:'Avg Math Score (%)'},min:50,max:100,grid:{color:'rgba(255,255,255,0.04)'}}}}
  });
}();

// --- Chart 2: Math D/F Rate by Income Quartile ---
!function(){
  const q = CD.df_income.quartiles;
  if(!q.length) return;
  new Chart(document.getElementById('chartDFIncome'), {
    type:'bar',
    data:{
      labels: q.map(d=>d.label),
      datasets:[{
        label:'D/F Rate %',
        data:q.map(d=>d.df_rate),
        backgroundColor: q.map(d=>d.df_rate>30?'#fc5c65':d.df_rate>15?'#fc8c3c':'#26de81'),
        borderRadius:4,
      }]
    },
    options:{responsive:true, plugins:{legend:{display:false},tooltip:{callbacks:{afterLabel:function(ctx){return q[ctx.dataIndex].df_count+'/'+q[ctx.dataIndex].n+' students'}}}}, scales:{x:{title:{display:true,text:'Zip-Code Median Household Income'},grid:{display:false}},y:{title:{display:true,text:'D/F Rate (%)'},max:60,grid:{color:'rgba(255,255,255,0.04)'}}}}
  });
}();

// --- Chart 3: Race Composition by Income Quartile (Stacked %) ---
!function(){
  const d = CD.race_income;
  if(!d.quartiles.length) return;
  const raceColors = {'Black':'#4a9eff','White':'#26de81','Hispanic/Latino':'#f7b731','Asian':'#fc8c3c','Two or More':'#a855f7','Am. Indian/Alaska Native':'#ec4899','Unknown':'#718096'};
  const datasets = d.races.map(race=>({
    label:race,
    data:d.quartiles.map(q=>q[race]||0),
    backgroundColor:raceColors[race]||'#718096',
  }));
  new Chart(document.getElementById('chartRaceIncome'), {
    type:'bar',
    data:{
      labels: d.quartiles.map(q=>q.label+'\n(n='+q.n+')'),
      datasets: datasets,
    },
    options:{responsive:true, plugins:{legend:{position:'top',labels:{boxWidth:10,padding:6,font:{size:9}}},tooltip:{callbacks:{label:function(ctx){return ctx.dataset.label+': '+ctx.raw+'%'}}}}, scales:{x:{stacked:true,grid:{display:false}},y:{stacked:true,max:100,title:{display:true,text:'% of Students'},grid:{color:'rgba(255,255,255,0.04)'}}}}
  });
}();

// --- Chart 4: Math D/F Rate by Race ---
!function(){
  const g = CD.df_race.groups;
  if(!g.length) return;
  new Chart(document.getElementById('chartDFRace'), {
    type:'bar',
    data:{
      labels: g.map(d=>d.race),
      datasets:[{
        label:'D/F Rate %',
        data:g.map(d=>d.df_rate),
        backgroundColor: g.map(d=>d.df_rate>30?'#fc5c65':d.df_rate>15?'#fc8c3c':'#26de81'),
        borderRadius:4,
      }]
    },
    options:{indexAxis:'y', responsive:true, plugins:{legend:{display:false},tooltip:{callbacks:{afterLabel:function(ctx){return g[ctx.dataIndex].df_count+'/'+g[ctx.dataIndex].n+' students'}}}}, scales:{x:{title:{display:true,text:'Math D/F Rate (%)'},max:60,grid:{color:'rgba(255,255,255,0.04)'}},y:{grid:{display:false}}}}
  });
}();

// --- Slide Navigation ---
!function(){
  const slides=document.querySelectorAll('.slide');
  const dots=document.getElementById('nav-dots');
  let cur=0;

  // Build dots
  slides.forEach((s,i)=>{
    const d=document.createElement('div');
    d.className='dot'+(i===0?' on':'');
    d.onclick=()=>goTo(i);
    dots.appendChild(d);
  });

  // Build prev/next buttons (in header)
  const nav=document.getElementById('slide-nav');
  nav.innerHTML='<button id="btnPrev" title="Previous slide">&#x276E;</button><span class="counter" id="slideCounter">1 / '+slides.length+'</span><button id="btnNext" title="Next slide">&#x276F;</button>';
  const btnP=document.getElementById('btnPrev');
  const btnN=document.getElementById('btnNext');
  const ctr=document.getElementById('slideCounter');

  function goTo(idx){
    cur=Math.max(0,Math.min(slides.length-1,idx));
    slides.forEach((s,i)=>s.classList.toggle('active',i===cur));
    sync();
  }
  function sync(){
    dots.querySelectorAll('.dot').forEach((d,i)=>d.classList.toggle('on',i===cur));
    btnP.disabled=cur===0;
    btnN.disabled=cur===slides.length-1;
    ctr.textContent=(cur+1)+' / '+slides.length;
  }

  btnP.onclick=(e)=>{e.stopPropagation();goTo(cur-1);};
  btnN.onclick=(e)=>{e.stopPropagation();goTo(cur+1);};

  // Click anywhere to advance (skip if clicking interactive elements)
  document.addEventListener('click',(e)=>{
    if(e.target.closest('.slide-nav, nav, a, button, canvas, input, select, textarea')) return;
    goTo(cur+1);
  });

  // Keyboard navigation
  document.addEventListener('keydown',(e)=>{
    if(e.key==='ArrowRight'||e.key===' '||e.key==='PageDown'){e.preventDefault();goTo(cur+1);}
    if(e.key==='ArrowLeft'||e.key==='PageUp'){e.preventDefault();goTo(cur-1);}
    if(e.key==='Home'){e.preventDefault();goTo(0);}
    if(e.key==='End'){e.preventDefault();goTo(slides.length-1);}
  });

  slides[0].classList.add('active');
  sync();
}();

</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data...")
    mdf = load_current_math_grades()
    adf = load_affluence_data()
    zdf = load_zip_income()

    demo = load_demographics()

    print(f"  Math grades: {len(mdf)} rows")
    print(f"  Affluence data: {len(adf)} students")
    print(f"  Zip analysis: {len(zdf)} rows")
    print(f"  Demographics: {len(demo) if demo is not None else 0} students")

    # Build equity merged dataset (math grades + income + demographics)
    eq = _build_equity_merged(mdf, adf, demo)
    print(f"  Equity merged: {len(eq)} rows, {eq['Student ID'].nunique() if not eq.empty else 0} students")

    # Build chart data
    chart_data = {
        "grade_dist": math_grade_distribution(mdf),
        "avg_course": math_grade_by_course(mdf),
        "df_rate": df_rate_by_course(mdf),
        "score_income": math_score_by_income(eq),
        "df_income": math_df_by_income(eq),
        "race_income": race_by_income_quartile(eq),
        "df_race": math_df_by_race(eq),
    }

    html = build_html(chart_data)
    out = OUT_DIR / "index.html"
    out.write_text(html, encoding="utf-8")
    print(f"\nPresentation written to {out}")
    print(f"  File size: {out.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
