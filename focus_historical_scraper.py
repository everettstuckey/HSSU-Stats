"""
Focus SIS Historical Math Grades & Demographics Scraper

Pulls math grades and student demographics across multiple school years
from Focus SIS for the HSSU Statistics Partnership analysis.

Usage:
    python focus_historical_scraper.py --years 10 --headed
    python focus_historical_scraper.py --demographics-only --headed
"""
import argparse
import io
import json
import os
import re
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Add parent dir to path for focus_login imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from focus_login import (
    get_authenticated_context,
    _fetch_csv_with_cookies,
    DEFAULT_STORAGE,
    FINAL_GRADES_URL,
)
from playwright.sync_api import (
    Page,
    BrowserContext,
    TimeoutError as PlaywrightTimeoutError,
    sync_playwright,
)

OUT_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# School year switching
# ---------------------------------------------------------------------------

def get_available_years(page: Page) -> List[dict]:
    """Find the school year selector and return available years."""
    # Focus SIS typically has a year dropdown in the top nav
    # Look for the year selector element
    selectors = [
        'select#side_syear',           # common Focus year dropdown
        'select[name="side_syear"]',
        '#year_select',
        'select.year-selector',
        '.school-year-dropdown select',
    ]

    for sel in selectors:
        el = page.query_selector(sel)
        if el:
            options = page.query_selector_all(f'{sel} option')
            years = []
            for opt in options:
                val = opt.get_attribute('value')
                text = opt.inner_text().strip()
                selected = opt.get_attribute('selected') is not None
                years.append({
                    'value': val,
                    'text': text,
                    'selected': selected,
                })
            print(f"Found year selector ({sel}): {len(years)} years available")
            return years

    # Fallback: look for year links in navigation
    year_links = page.query_selector_all('a[href*="syear="]')
    if year_links:
        years = []
        for link in year_links:
            href = link.get_attribute('href')
            text = link.inner_text().strip()
            m = re.search(r'syear=(\d+)', href)
            if m:
                years.append({'value': m.group(1), 'text': text, 'selected': False})
        print(f"Found year links: {len(years)} years available")
        return years

    print("WARNING: Could not find school year selector!")
    # Try to identify from page content
    page.screenshot(path=OUT_DIR / "year_selector_debug.png", full_page=True)
    return []


def switch_school_year(page: Page, year_value: str) -> bool:
    """Switch to a specific school year."""
    selectors = [
        'select#side_syear',
        'select[name="side_syear"]',
        '#year_select',
    ]

    for sel in selectors:
        el = page.query_selector(sel)
        if el:
            page.select_option(sel, year_value)
            page.wait_for_load_state("networkidle", timeout=30_000)
            page.wait_for_timeout(2_000)
            print(f"  Switched to year: {year_value}")
            return True

    # Fallback: navigate with URL parameter
    current_url = page.url
    if '?' in current_url:
        new_url = re.sub(r'syear=\d+', f'syear={year_value}', current_url)
        if new_url == current_url:
            new_url += f'&syear={year_value}'
    else:
        new_url = current_url + f'?syear={year_value}'

    page.goto(new_url, wait_until="load", timeout=30_000)
    page.wait_for_load_state("networkidle", timeout=30_000)
    page.wait_for_timeout(2_000)
    print(f"  Navigated to year via URL: {year_value}")
    return True


# ---------------------------------------------------------------------------
# Grade export for a specific year
# ---------------------------------------------------------------------------

def export_grades_for_year(
    page: Page,
    context: BrowserContext,
    year_text: str,
) -> Optional[pd.DataFrame]:
    """Export all grades for the current school year."""
    print(f"\n  Navigating to Student Final Grades for {year_text}...")
    page.goto(FINAL_GRADES_URL, wait_until="load", timeout=30_000)
    page.wait_for_load_state("networkidle", timeout=30_000)
    page.wait_for_timeout(2_000)

    if "StudentFinalGrades" not in page.url:
        print(f"  WARNING: Not on Final Grades page ({page.url})")
        return None

    # Open Search Screen
    try:
        page.click("text=Search Screen", timeout=10_000)
        page.wait_for_load_state("networkidle", timeout=15_000)
        page.wait_for_timeout(2_000)
    except PlaywrightTimeoutError:
        print("  WARNING: Could not open Search Screen")
        return None

    # Find and check all available marking period checkboxes
    mp_checkboxes = page.query_selector_all('input[id^="mp_arr"]')
    checked = 0
    for cb in mp_checkboxes:
        try:
            cb.check()
            checked += 1
        except Exception:
            pass
    print(f"  Checked {checked} marking period checkboxes")

    # Check grade and percent fields
    for field in ["elements[grade]", "elements[percent]"]:
        cb = page.query_selector(f'input[name="{field}"]')
        if cb:
            try:
                cb.check()
            except Exception:
                pass

    # Click Continue/Search
    try:
        btn = page.locator('button[data-test="search-button"]')
        if btn.count():
            btn.click()
        else:
            page.click("text=Continue", timeout=5_000)
        page.wait_for_load_state("networkidle", timeout=120_000)
        page.wait_for_timeout(3_000)
    except PlaywrightTimeoutError:
        print("  WARNING: Grade list generation timed out")
        return None

    # Find CSV export link
    csv_link = page.query_selector("a.lo_export_csv")
    if not csv_link:
        print("  WARNING: No CSV export link found")
        page.screenshot(path=OUT_DIR / f"no_csv_{year_text}.png", full_page=True)
        return None

    href = csv_link.get_attribute("href")
    print("  Downloading CSV...")
    raw = _fetch_csv_with_cookies(context, href)
    print(f"  Downloaded {len(raw):,} bytes")

    df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
    df["SchoolYear"] = year_text
    return df


# ---------------------------------------------------------------------------
# Demographics export
# ---------------------------------------------------------------------------

DEMOGRAPHICS_URL = (
    "https://slps.focusschoolsoftware.com/focus/Modules.php"
    "?modname=Students/Student.php&modfunc=none"
    "&search_modfunc=list"
)


def export_demographics(
    page: Page,
    context: BrowserContext,
) -> Optional[pd.DataFrame]:
    """Export student demographics (race, ethnicity, zip code)."""
    print("\nNavigating to Student Information...")
    page.goto(DEMOGRAPHICS_URL, wait_until="load", timeout=30_000)
    page.wait_for_load_state("networkidle", timeout=30_000)
    page.wait_for_timeout(2_000)

    # Try to configure the report to include demographic fields
    try:
        page.click("text=Search Screen", timeout=10_000)
        page.wait_for_load_state("networkidle", timeout=15_000)
        page.wait_for_timeout(2_000)
    except PlaywrightTimeoutError:
        print("  Trying alternative navigation...")

    # Look for demographic field checkboxes
    demo_fields = [
        "elements[ethnicity]", "elements[race]",
        "elements[zipcode]", "elements[zip]",
        "elements[address]", "elements[city]",
        "elements[custom_200000027]",  # common custom field for ethnicity
    ]
    for field in demo_fields:
        cb = page.query_selector(f'input[name="{field}"]')
        if cb:
            try:
                cb.check()
                print(f"  Checked: {field}")
            except Exception:
                pass

    # Search/Continue
    try:
        btn = page.locator('button[data-test="search-button"]')
        if btn.count():
            btn.click()
        else:
            page.click("text=Continue", timeout=5_000)
        page.wait_for_load_state("networkidle", timeout=120_000)
        page.wait_for_timeout(3_000)
    except PlaywrightTimeoutError:
        print("  WARNING: Demographics search timed out")

    # Find CSV export
    csv_link = page.query_selector("a.lo_export_csv")
    if not csv_link:
        print("  WARNING: No CSV export link found for demographics")
        page.screenshot(path=OUT_DIR / "no_demo_csv.png", full_page=True)
        return None

    href = csv_link.get_attribute("href")
    raw = _fetch_csv_with_cookies(context, href)
    df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
    print(f"  Demographics: {len(df)} students, columns: {list(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# Filter to math courses
# ---------------------------------------------------------------------------

def filter_math_courses(df: pd.DataFrame) -> pd.DataFrame:
    """Filter grade DataFrame to math courses only."""
    if "Course" not in df.columns:
        return df
    math_pat = r"Algebra|Geometry|Calculus|Pre\s?Calc|PreCalc|Math|Trig|Stat"
    eca_pat = r"ECA.*(?:Algebra|Calculus|Precalculus|Math|Stat)"
    mask = (df["Course"].str.contains(math_pat, case=False, na=False) |
            df["Course"].str.contains(eca_pat, case=False, na=False))
    # Exclude non-math ECA courses
    exclude = r"ECA.*(?:English|History|Art|Biology|Psychology|Sociology|Communication|Philosophy|German|Photography|Drawing|Fitness|Sex|Java|College Comp|Smart Start|Public Speaking)"
    mask = mask & ~df["Course"].str.contains(exclude, case=False, na=False)
    return df[mask].copy()


# ---------------------------------------------------------------------------
# Main scraper
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Scrape historical math grades from Focus SIS")
    parser.add_argument("--years", type=int, default=10, help="Number of years to scrape (default: 10)")
    parser.add_argument("--headed", action="store_true", help="Show browser window")
    parser.add_argument("--demographics-only", action="store_true", help="Only scrape demographics")
    parser.add_argument("--grades-only", action="store_true", help="Only scrape grades")
    args = parser.parse_args()

    with sync_playwright() as playwright:
        browser, context, page = get_authenticated_context(
            playwright, headed=args.headed or True
        )

        try:
            if not args.grades_only:
                # --- Demographics ---
                demo_df = export_demographics(page, context)
                if demo_df is not None:
                    demo_path = OUT_DIR / "student_demographics.csv"
                    demo_df.to_csv(demo_path, index=False)
                    print(f"Saved demographics to {demo_path}")

            if args.demographics_only:
                return

            # --- Historical Grades ---
            print("\n" + "=" * 60)
            print("Discovering available school years...")
            print("=" * 60)

            years = get_available_years(page)
            if not years:
                print("Could not detect school years. Exporting current year only.")
                df = export_grades_for_year(page, context, "current")
                if df is not None:
                    math_df = filter_math_courses(df)
                    math_df.to_csv(OUT_DIR / "historical_math_grades.csv", index=False)
                    print(f"Saved {len(math_df)} math grade rows")
                return

            # Limit to requested number of years
            years_to_scrape = years[:args.years]
            print(f"Will scrape {len(years_to_scrape)} school years:")
            for y in years_to_scrape:
                print(f"  {y['text']} (value={y['value']})")

            all_dfs = []
            for y in years_to_scrape:
                print(f"\n{'='*60}")
                print(f"Processing: {y['text']}")
                print(f"{'='*60}")

                switch_school_year(page, y["value"])
                df = export_grades_for_year(page, context, y["text"])
                if df is not None:
                    math_df = filter_math_courses(df)
                    print(f"  Found {len(math_df)} math grade rows")
                    all_dfs.append(math_df)
                else:
                    print(f"  No data for {y['text']}")

            if all_dfs:
                combined = pd.concat(all_dfs, ignore_index=True)
                out_path = OUT_DIR / "historical_math_grades.csv"
                combined.to_csv(out_path, index=False)
                print(f"\n{'='*60}")
                print(f"Combined: {len(combined)} math grade rows across {len(all_dfs)} years")
                print(f"Saved to {out_path}")
                print(f"{'='*60}")
            else:
                print("\nNo grade data collected.")

        finally:
            context.storage_state(path=DEFAULT_STORAGE)
            browser.close()


if __name__ == "__main__":
    main()
