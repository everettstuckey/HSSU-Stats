"""
Geocode student addresses using the US Census Bureau batch geocoder,
download Missouri zip-code boundary GeoJSON, and save everything
as map_data.json for the presentation build script.

Student dots are jittered ±0.002° (~220 m) for anonymization.
"""

import csv, io, json, random, time
from pathlib import Path
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
AFFLUENCE = ROOT / "affluence_analysis_results"
OUT = Path(__file__).resolve().parent / "map_data.json"

CENSUS_BATCH_URL = (
    "https://geocoding.geo.census.gov/geocoder/locations/addressbatch"
)
CENSUS_SINGLE_URL = (
    "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"
)
MO_ZCTA_URL = (
    "https://raw.githubusercontent.com/OpenDataDE/State-zip-code-GeoJSON/"
    "master/mo_missouri_zip_codes_geo.min.json"
)

JITTER = 0.002  # ~220 meters of random offset for privacy

random.seed(42)


def _geocode_batch(addresses: list[dict]) -> dict[str, tuple]:
    """Batch-geocode via Census Bureau. Returns {id: (lat, lng)} dict."""
    # Build CSV for batch geocoder: id, street, city, state, zip
    buf = io.StringIO()
    writer = csv.writer(buf)
    for a in addresses:
        writer.writerow([
            a["id"], a["street"], a["city"], a["state"], a["zip"]
        ])
    buf.seek(0)

    print(f"  Sending {len(addresses)} addresses to Census batch geocoder...")
    resp = requests.post(
        CENSUS_BATCH_URL,
        files={"addressFile": ("addresses.csv", buf.getvalue(), "text/csv")},
        data={"benchmark": "Public_AR_Current"},
        timeout=120,
    )
    resp.raise_for_status()

    results = {}
    for line in resp.text.strip().split("\n"):
        parts = line.strip('"').split('","')
        if len(parts) < 6:
            continue
        uid = parts[0].strip('"')
        match_flag = parts[2].strip('"')
        if match_flag == "Match":
            coords = parts[5].strip('"')  # "lng,lat"
            try:
                lng_s, lat_s = coords.split(",")
                results[uid] = (float(lat_s), float(lng_s))
            except (ValueError, IndexError):
                pass
    return results


def _geocode_single(street: str, city: str, state: str, zipcode: str):
    """Fallback: geocode one address at a time."""
    params = {
        "address": f"{street}, {city}, {state} {zipcode}",
        "benchmark": "Public_AR_Current",
        "format": "json",
    }
    try:
        resp = requests.get(CENSUS_SINGLE_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        matches = data.get("result", {}).get("addressMatches", [])
        if matches:
            c = matches[0]["coordinates"]
            return (c["y"], c["x"])
    except Exception:
        pass
    return None


def _jitter(lat: float, lng: float) -> tuple:
    """Add random offset for anonymization."""
    return (
        lat + random.uniform(-JITTER, JITTER),
        lng + random.uniform(-JITTER, JITTER),
    )


def main():
    # --- Load student data ---
    sdf = pd.read_csv(AFFLUENCE / "processed_student_data.csv")
    print(f"Loaded {len(sdf)} students")

    # Parse addresses for the batch geocoder
    addresses = []
    for _, row in sdf.iterrows():
        street = str(row["Address1"]).strip()
        addr2 = str(row["Address2"]).strip()
        # addr2 looks like "ST LOUIS, MO  63109"
        parts = addr2.replace(",", " ").split()
        city = "ST LOUIS"
        state = "MO"
        zipcode = str(int(row["ZipCode"]))
        addresses.append({
            "id": str(row["ID"]),
            "street": street,
            "city": city,
            "state": state,
            "zip": zipcode,
        })

    # --- Batch geocode ---
    geo = _geocode_batch(addresses)
    print(f"  Batch matched: {len(geo)} / {len(addresses)}")

    # Fallback: single-geocode any misses
    missing = [a for a in addresses if a["id"] not in geo]
    if missing:
        print(f"  Retrying {len(missing)} misses individually...")
        for a in missing:
            result = _geocode_single(a["street"], a["city"], a["state"], a["zip"])
            if result:
                geo[a["id"]] = result
            time.sleep(0.5)
        print(f"  Total geocoded: {len(geo)} / {len(addresses)}")

    # --- Build student dots (anonymized) ---
    dots = []
    for _, row in sdf.iterrows():
        sid = str(row["ID"])
        if sid not in geo:
            continue
        lat, lng = _jitter(*geo[sid])
        dots.append({
            "lat": round(lat, 5),
            "lng": round(lng, 5),
            "zip": str(int(row["ZipCode"])),
            "grade": int(row["GradeLevel"]),
        })
    print(f"  Student dots: {len(dots)}")

    # --- Fetch Missouri zip-code boundaries ---
    print("  Downloading Missouri ZCTA GeoJSON...")
    resp = requests.get(MO_ZCTA_URL, timeout=60)
    resp.raise_for_status()
    mo_geo = resp.json()

    target_zips = set(str(int(z)) for z in sdf["ZipCode"].unique())
    filtered_features = [
        f for f in mo_geo["features"]
        if f["properties"].get("ZCTA5CE10") in target_zips
    ]
    print(f"  Zip polygons matched: {len(filtered_features)} / {len(target_zips)} target zips")

    zip_geojson = {
        "type": "FeatureCollection",
        "features": filtered_features,
    }

    # Attach income + student count to each zip feature
    zip_income = sdf.groupby("ZipCode").agg(
        median_income=("MedianZipIncome", "first"),
        n_students=("ID", "nunique"),
        avg_gpa=("Acum-GPA", "mean"),
    ).reset_index()
    income_map = {}
    for _, r in zip_income.iterrows():
        zc = str(int(r["ZipCode"]))
        income_map[zc] = {
            "income": int(r["median_income"]) if pd.notna(r["median_income"]) else 0,
            "n": int(r["n_students"]),
            "gpa": round(float(r["avg_gpa"]), 2),
        }

    for f in zip_geojson["features"]:
        zc = f["properties"].get("ZCTA5CE10", "")
        info = income_map.get(zc, {})
        f["properties"]["income"] = info.get("income", 0)
        f["properties"]["n_students"] = info.get("n", 0)
        f["properties"]["avg_gpa"] = info.get("gpa", 0)

    # --- Save ---
    out_data = {
        "dots": dots,
        "zip_geojson": zip_geojson,
    }
    OUT.write_text(json.dumps(out_data, separators=(",", ":")), encoding="utf-8")
    sz = OUT.stat().st_size
    print(f"\nSaved {OUT} ({sz:,} bytes)")
    print(f"  {len(dots)} student dots, {len(filtered_features)} zip polygons")


if __name__ == "__main__":
    main()
