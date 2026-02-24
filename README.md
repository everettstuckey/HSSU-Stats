# HSSU Statistics Partnership Proposal

A presentation making the case for a dual-credit statistics partnership between **Harris-Stowe State University** (HBCU) and **Collegiate School of Medicine & Bioscience** (SLPS).

## View the Presentation

**[Live Presentation](https://everettstuckey.github.io/HSSU-Stats/)**

## Course Comparison

| Course | Institution | Credits | R Programming | Bio Focus |
|--------|------------|---------|---------------|-----------|
| MTH 180 Intro Statistics | STLCC | 3 | No | No |
| STAT0260 Data Analysis & Stats w/Lab | HSSU | 4 | Yes | General |
| MATH0301 Biostatistics | HSSU | 3 | Yes | Yes |

## Data Analysis

The presentation includes analysis of:
- Current year math grade distributions at CSMB
- D/F rates by math course
- Correlation between zip code median income and GPA
- GPA patterns by zip code

## Scripts

- `build_presentation.py` — Generates `index.html` from Focus SIS grade data and census income data
- `focus_historical_scraper.py` — Scrapes historical math grades and student demographics from Focus SIS (requires credentials)

## Regenerate

```bash
python build_presentation.py
```
