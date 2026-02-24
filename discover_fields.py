"""
Scrape student demographics (#!1) and addresses (#!addresses) from Focus SIS.
Iterates through all students and saves to focus_downloads/student_demographics.csv.

Demographics tab fields observed:
  - Ethnicity (Hispanic/Latino Y/N)
  - Single Ethnicity (e.g. Asian, Black, White, etc.)
  - Race: Black or African American (Yes/No)
  - Race: Asian (Yes/No)
  - Race: American Indian or Alaska Native (Yes/No)
  - Race: White (Yes/No)
  - Race: Native Hawaiian or Other Pacific Islander (Yes/No)

Addresses tab:
  - Street address, City, State, Zip
"""
import csv
import re
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

from focus_login_hssu import *
from playwright.sync_api import sync_playwright

DL = Path("focus_downloads")
BASE = "https://slps.focusschoolsoftware.com/focus/Modules.php"


def get_all_student_ids(page):
    """Get list of (student_id, name) from the student list."""
    page.goto(STUDENT_INFO_URL, wait_until="load", timeout=30_000)
    page.wait_for_load_state("networkidle", timeout=30_000)
    page.wait_for_timeout(2_000)

    # Click Search to load all students
    try:
        page.locator('button:has-text("Search")').click(timeout=5_000)
    except Exception:
        page.click("text=Search", timeout=5_000)
    page.wait_for_load_state("networkidle", timeout=30_000)
    page.wait_for_timeout(2_000)

    links = page.query_selector_all('a[href*="student_id="]')
    students = []
    seen = set()
    for link in links:
        href = link.get_attribute("href") or ""
        m = re.search(r'student_id=(\d+)', href)
        if m and m.group(1) not in seen:
            seen.add(m.group(1))
            name = link.inner_text().strip()
            students.append((m.group(1), name))
    return students


def scrape_demographics(page, student_id):
    """Navigate to #!1 tab and extract race/ethnicity fields."""
    url = f"{BASE}?modname=Students/Student.php&include_top=&student_id={student_id}#!1"
    page.goto(url, wait_until="load", timeout=30_000)
    page.wait_for_load_state("networkidle", timeout=30_000)
    page.wait_for_timeout(1_500)

    body = page.inner_text("body")

    demo = {}

    # Extract ethnicity (Hispanic/Latino)
    m = re.search(r'Hispanic/Latino\s*\n?\s*([YN]\s*-\s*\w+)', body)
    if m:
        demo["Hispanic_Latino"] = m.group(1).strip()

    # Extract single ethnicity
    m = re.search(r'Single Ethnicity\s*\n?\s*(\w[\w\s]*?)(?:\n|$)', body)
    if m:
        demo["Single_Ethnicity"] = m.group(1).strip()

    # Extract individual race flags
    race_fields = [
        ("Race_Black", r"Race:\s*Black or African American\s*\n?\s*(Yes|No)"),
        ("Race_Asian", r"Race:\s*Asian\s*\n?\s*(Yes|No)"),
        ("Race_AmIndian", r"Race:\s*American Indian or Alaska Native\s*\n?\s*(Yes|No)"),
        ("Race_White", r"Race:\s*White\s*\n?\s*(Yes|No)"),
        ("Race_Pacific", r"Race:\s*Native Hawaiian or Other Pacific Islander\s*\n?\s*(Yes|No)"),
    ]
    for key, pattern in race_fields:
        m = re.search(pattern, body, re.IGNORECASE)
        if m:
            demo[key] = m.group(1).strip()

    # Extract gender
    m = re.search(r'Gender\s*\n?\s*(Male|Female|M|F)', body)
    if m:
        demo["Gender"] = m.group(1).strip()

    return demo


def scrape_address(page, student_id):
    """Navigate to #!addresses tab and extract zip code + address."""
    url = f"{BASE}?modname=Students/Student.php&include_top=&student_id={student_id}#!addresses"
    page.goto(url, wait_until="load", timeout=30_000)
    page.wait_for_load_state("networkidle", timeout=30_000)
    page.wait_for_timeout(1_500)

    body = page.inner_text("body")
    addr = {}

    # Look for zip code pattern (5-digit starting with 6 for MO)
    zips = re.findall(r'\b(6\d{4})\b', body)
    if zips:
        addr["ZipCode"] = zips[0]

    # Try to extract full address
    # Look for "MO" state pattern: street, city MO zip
    m = re.search(r'(\d+\s+[\w\s]+(?:ST|AV|AVE|BLVD|DR|LN|PL|RD|CT|WAY|TER|CIR)[\w\s]*)\s*,?\s*([\w\s]+),?\s*MO\s*(\d{5})', body, re.IGNORECASE)
    if m:
        addr["Street"] = m.group(1).strip()
        addr["City"] = m.group(2).strip()
        addr["State"] = "MO"
        addr["ZipCode"] = m.group(3).strip()

    return addr


def main():
    DL.mkdir(exist_ok=True)

    with sync_playwright() as pw:
        browser, context, page = get_authenticated_context(pw, headed=True)
        try:
            # Get all student IDs
            print("Getting student list...")
            students = get_all_student_ids(page)
            print(f"Found {len(students)} unique students", flush=True)

            rows = []
            for i, (sid, name) in enumerate(students):
                try:
                    demo = scrape_demographics(page, sid)
                    addr = scrape_address(page, sid)

                    row = {
                        "StudentID": sid,
                        "Student": name,
                        **demo,
                        **addr,
                    }
                    rows.append(row)

                    # Progress
                    race = demo.get("Single_Ethnicity", "?")
                    zc = addr.get("ZipCode", "?")
                    print(f"  [{i+1}/{len(students)}] {name}: Race={race}, Zip={zc}", flush=True)

                except Exception as e:
                    print(f"  [{i+1}/{len(students)}] ERROR {name}: {e}", flush=True)
                    rows.append({"StudentID": sid, "Student": name, "Error": str(e)})

            # Save to CSV
            if rows:
                import pandas as pd
                df = pd.DataFrame(rows)
                out_path = DL / "student_demographics.csv"
                df.to_csv(out_path, index=False)
                print(f"\nSaved {len(df)} students to {out_path}")
                print(f"Columns: {list(df.columns)}")
                print(f"\nRace distribution:")
                if "Single_Ethnicity" in df.columns:
                    print(df["Single_Ethnicity"].value_counts().to_string())
                print(f"\nZip code distribution:")
                if "ZipCode" in df.columns:
                    print(df["ZipCode"].value_counts().to_string())

        finally:
            context.storage_state(path=DEFAULT_STORAGE)
            browser.close()


if __name__ == "__main__":
    main()
