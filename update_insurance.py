"""Check IRDAI for a newer 'Handbook on Indian Insurance Statistics' and rebuild the insurer data if one exists.

Run by GitHub Actions once a month (.github/workflows/update_insurance.yml), or by hand:
    python update_insurance.py
The result is written to data/insurance/update_status.json. If anything looks wrong, the existing data is kept.
"""
import json
import re
import shutil
import tempfile
import time
import urllib.parse
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
INS = ROOT / "data" / "insurance"
PAGE = "https://irdai.gov.in/handbook-of-indian-insurance"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/126 Safari/537.36"}


def status(result: str) -> None:
    (INS / "update_status.json").write_text(json.dumps({"checked": time.strftime("%Y-%m-%d"), "result": result}))
    print(result)


def latest_handbook():
    """(year, url) of the newest handbook zip listed on IRDAI's page, e.g. ('2025-26', 'https://...zip...')."""
    html = requests.get(PAGE, headers=UA, timeout=60).text
    found = {}
    for href in re.findall(r'href="([^"]+\.zip[^"]*)"', html):
        name = urllib.parse.unquote_plus(href)
        m = re.search(r"Handbook on Indian Insurance Statistics (\d{4}-\d{2})", name)
        if m:
            found.setdefault(m.group(1), href.replace("&amp;", "&"))
    if not found:
        raise RuntimeError("no handbook links found on the IRDAI page")
    year = max(found)
    return year, found[year]


def main():
    current = json.loads((INS / "meta.json").read_text())["year"]
    try:
        year, url = latest_handbook()
    except Exception as e:
        return status(f"failed: could not read the IRDAI page ({str(e)[:120]})")
    if year <= current:
        return status(f"up to date ({current})")

    backup = {f: (INS / f).read_bytes() for f in ["life.csv", "health.csv", "meta.json"]}
    tmp = Path(tempfile.mkdtemp())
    try:
        z = tmp / "handbook.zip"
        z.write_bytes(requests.get(url, headers=UA, timeout=300).content)
        zipfile.ZipFile(z).extractall(tmp / "x")
        folder = next(p.parent for p in (tmp / "x").rglob("Part I.xlsx"))
        from core.insurance_data import build
        life, health = build(str(folder), year)
        ok = (len(life) >= 15 and len(health) >= 15
              and 0.9 < life["csr_number"].median() <= 1.0 and health["icr_retail"].notna().sum() >= 15)
        if not ok:
            raise RuntimeError("new tables did not pass the sanity checks (layout may have changed)")
        status(f"updated to {year}")
    except Exception as e:
        for f, data in backup.items():
            (INS / f).write_bytes(data)
        status(f"failed: handbook {year} found but could not be processed, keeping {current} data ({str(e)[:150]})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
