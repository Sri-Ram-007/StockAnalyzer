"""Latest news headlines from Google News (free RSS, no key), cached for one hour."""
import json
import time
import urllib.parse
import xml.etree.ElementTree as ET

import pandas as pd
import requests

from core.data import DATA_DIR

NEWS_DIR = DATA_DIR / "news"
NEWS_DIR.mkdir(exist_ok=True)
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"


def get_news(query: str, india: bool = True, limit: int = 8) -> list[dict]:
    key = "".join(ch for ch in query.lower() if ch.isalnum())[:60] + ("_in" if india else "_us")
    cache = NEWS_DIR / f"{key}.json"
    if cache.exists() and time.time() - cache.stat().st_mtime < 3600:
        return json.loads(cache.read_text())[:limit]
    region = "hl=en-IN&gl=IN&ceid=IN:en" if india else "hl=en-US&gl=US&ceid=US:en"
    url = f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}&{region}"
    items = []
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=20)
        for it in ET.fromstring(r.content).findall(".//item"):
            title, source = it.findtext("title") or "", it.findtext("source") or ""
            if source and title.endswith(" - " + source):
                title = title[: -len(source) - 3]
            items.append({"title": title, "link": it.findtext("link"), "source": source,
                          "date": str(pd.to_datetime(it.findtext("pubDate"), errors="coerce"))})
        items.sort(key=lambda x: x["date"], reverse=True)
        cache.write_text(json.dumps(items[:30]))
    except Exception:
        if cache.exists():
            return json.loads(cache.read_text())[:limit]
    return items[:limit]


def news_query(name: str, ticker: str) -> tuple[str, bool]:
    """Search words and region for a stock: Indian stocks search Indian news for '<company> share'."""
    clean = name.replace(" Limited", "").replace(" Ltd.", "").replace(" Ltd", "").replace(", Inc.", "").replace(" Inc.", "")
    india = ticker.endswith((".NS", ".BO"))
    return (f'"{clean}" share' if india else f'"{clean}" stock'), india
