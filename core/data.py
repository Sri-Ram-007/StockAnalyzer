"""Downloading and caching market data (free sources, no API keys)."""
import io
import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Official index constituent lists published by NSE Indices
UNIVERSES = {
    "Nifty 50": "ind_nifty50list.csv",
    "Nifty 100": "ind_nifty100list.csv",
    "Nifty 200": "ind_nifty200list.csv",
    "Nifty 500": "ind_nifty500list.csv",
    "All NSE stocks": None,
    "BSE-only (over ₹500 Cr)": None,
}
UNIVERSE_URLS = [
    "https://archives.nseindia.com/content/indices/{}",
    "https://www.niftyindices.com/IndexConstituent/{}",
]
HEADERS = {"User-Agent": "Mozilla/5.0"}

UNIVERSE_MAX_AGE_DAYS = 7   # stock lists refresh weekly, so new listings appear within days
PRICE_MAX_AGE_HOURS = 12


def _age_hours(path: Path) -> float:
    return (time.time() - path.stat().st_mtime) / 3600 if path.exists() else float("inf")


ALL_NSE = "All NSE stocks"
ALL_NSE_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"


def get_all_nse() -> pd.DataFrame:
    """Every equity listed on NSE (series EQ, about 2,350 companies), refreshed monthly.
    Industry comes from the Nifty 500 list where available."""
    cache = DATA_DIR / "universe_All_NSE.csv"
    if _age_hours(cache) > UNIVERSE_MAX_AGE_DAYS * 24:
        try:
            r = requests.get(ALL_NSE_URL, headers=HEADERS, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            df.columns = [c.strip() for c in df.columns]
            df = df[df["SERIES"].str.strip().isin(["EQ", "BE", "BZ"])]  # BE/BZ = trade-for-trade series
            df = df.rename(columns={"SYMBOL": "Symbol", "NAME OF COMPANY": "Company Name"})[["Company Name", "Symbol"]]
            df.to_csv(cache, index=False)
        except Exception:
            pass
    if not cache.exists():
        raise RuntimeError("Could not download the NSE equity list and no cached copy exists.")
    df = pd.read_csv(cache)
    try:
        ind = get_universe("Nifty 500").set_index("Symbol")["Industry"]
        df["Industry"] = df["Symbol"].map(ind).fillna("")
    except Exception:
        df["Industry"] = ""
    df["Ticker"] = df["Symbol"].str.strip() + ".NS"
    return df[["Company Name", "Industry", "Symbol", "Ticker"]]


BSE_ONLY = "BSE-only (over ₹500 Cr)"
BSE_URL = ("https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w?Group=&Scripcode=&industry=&segment=Equity"
           "&status=Active")


def get_bse_only() -> pd.DataFrame:
    """Companies listed on BSE but not on NSE (matched by ISIN), refreshed monthly.
    Yahoo Finance tickers for BSE use the BSE symbol plus '.BO'."""
    cache = DATA_DIR / "universe_BSE_only.csv"
    if _age_hours(cache) > UNIVERSE_MAX_AGE_DAYS * 24:
        try:
            r = requests.get(BSE_URL, headers={**HEADERS, "Referer": "https://www.bseindia.com/",
                                               "Origin": "https://www.bseindia.com"}, timeout=40)
            b = pd.DataFrame(r.json())
            nse_raw = requests.get(ALL_NSE_URL, headers=HEADERS, timeout=30).text
            nse = pd.read_csv(io.StringIO(nse_raw))
            nse.columns = [c.strip() for c in nse.columns]
            only = b[~b["ISIN_NUMBER"].isin(set(nse["ISIN NUMBER"].str.strip()))]
            out = pd.DataFrame({"Company Name": only["Scrip_Name"].str.replace(r"-\$$", "", regex=True).str.strip(),
                                "Industry": only["INDUSTRY"].fillna(""), "Symbol": only["scrip_id"].str.strip(),
                                "Ticker": only["scrip_id"].str.strip() + ".BO",
                                "MarketCapCr": pd.to_numeric(only["Mktcap"], errors="coerce")})
            out.to_csv(cache, index=False)
        except Exception:
            pass
    if not cache.exists():
        return pd.DataFrame(columns=["Company Name", "Industry", "Symbol", "Ticker", "MarketCapCr"])
    df = pd.read_csv(cache)
    df["Industry"] = df["Industry"].fillna("")
    return df


def get_universe(name: str) -> pd.DataFrame:
    """Return Company Name, Industry, Symbol and Yahoo ticker for an NSE index.

    Downloads the official list once a month and falls back to the cached copy if offline.
    """
    if name == ALL_NSE:
        return get_all_nse()
    if name == BSE_ONLY:
        b = get_bse_only()
        return b[b["MarketCapCr"] > 500][["Company Name", "Industry", "Symbol", "Ticker"]].reset_index(drop=True)
    if name in ("S&P 500", "Indian ETFs", "US ETFs"):
        from core import markets
        if name == "S&P 500":
            return markets.get_sp500()
        return markets.get_nse_etfs() if name == "Indian ETFs" else markets.etf_universe(markets.US_ETFS)
    cache = DATA_DIR / f"universe_{name.replace(' ', '_')}.csv"
    if _age_hours(cache) > UNIVERSE_MAX_AGE_DAYS * 24:
        for url in UNIVERSE_URLS:
            try:
                r = requests.get(url.format(UNIVERSES[name]), headers=HEADERS, timeout=20)
                r.raise_for_status()
                df = pd.read_csv(io.StringIO(r.text))
                if "Symbol" in df.columns and len(df) > 10:
                    df.to_csv(cache, index=False)
                    break
            except Exception:
                continue
    if not cache.exists():
        raise RuntimeError(f"Could not download the {name} list and no cached copy exists.")
    df = pd.read_csv(cache)
    df["Ticker"] = df["Symbol"].str.strip() + ".NS"
    return df[["Company Name", "Industry", "Symbol", "Ticker"]]


def _download_closes(tickers: list[str], period: str) -> pd.DataFrame:
    """Download adjusted daily closes in batches; returns dates x tickers."""
    frames = []
    for i in range(0, len(tickers), 100):
        batch = tickers[i:i + 100]
        raw = yf.download(batch, period=period, auto_adjust=True, progress=False,
                          threads=True, group_by="column")
        if raw.empty:
            continue
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].rename(columns={"Close": batch[0]})
        frames.append(close)
    if not frames:
        return pd.DataFrame()
    closes = pd.concat(frames, axis=1)
    closes = closes.loc[:, ~closes.columns.duplicated()]
    closes.index = pd.to_datetime(closes.index).tz_localize(None)
    # Drop rows where almost nothing traded (e.g. an incomplete current day)
    return closes.dropna(how="all").loc[lambda d: d.notna().mean(axis=1) > 0.5]


def get_prices(name: str, tickers: list[str], period: str = "10y", force: bool = False) -> pd.DataFrame:
    """Cached adjusted close prices for a universe. Refreshes at most every 12 hours."""
    cache = DATA_DIR / f"prices_{name.replace(' ', '_')}.parquet"
    if force or _age_hours(cache) > PRICE_MAX_AGE_HOURS:
        closes = _download_closes(tickers, period)
        if not closes.empty:
            closes.to_parquet(cache)
    if not cache.exists():
        raise RuntimeError("Price download failed and no cached data exists. Check your internet connection.")
    return pd.read_parquet(cache)


def get_index(ticker: str = "^NSEI", period: str = "10y") -> pd.Series:
    """Daily close of a market index (default Nifty 50), cached for 12 hours."""
    cache = DATA_DIR / f"index_{ticker.strip('^')}.parquet"
    if _age_hours(cache) > PRICE_MAX_AGE_HOURS:
        closes = _download_closes([ticker], period)
        if not closes.empty:
            closes.to_parquet(cache)
    return pd.read_parquet(cache).iloc[:, 0].dropna()


def get_ohlc(ticker: str, period: str = "10y") -> pd.DataFrame:
    """Daily Open/High/Low/Close/Volume for one stock (adjusted), cached for 12 hours."""
    cache = DATA_DIR / f"ohlc_{ticker.replace('^', '').replace('&', '_')}.parquet"
    if _age_hours(cache) > PRICE_MAX_AGE_HOURS:
        try:
            df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
            if not df.empty:
                df.index = pd.to_datetime(df.index).tz_localize(None)
                df[["Open", "High", "Low", "Close", "Volume"]].dropna().to_parquet(cache)
        except Exception:
            pass
    if not cache.exists():
        return pd.DataFrame()
    return pd.read_parquet(cache)


MARKET_INDICES = {"NIFTY 50": "^NSEI", "SENSEX": "^BSESN", "BANK NIFTY": "^NSEBANK"}


def get_index_quotes() -> list[dict]:
    """Latest close and day change for the main Indian indices."""
    quotes = []
    for name, ticker in MARKET_INDICES.items():
        df = get_ohlc(ticker, period="1y")
        if len(df) >= 2:
            last, prev = float(df["Close"].iloc[-1]), float(df["Close"].iloc[-2])
            quotes.append({"name": name, "value": last, "change": last - prev, "pct": last / prev - 1})
    return quotes


WATCHLIST_FILE = DATA_DIR / "watchlist.json"


def load_watchlist() -> list[str]:
    import json
    try:
        return json.loads(WATCHLIST_FILE.read_text())
    except Exception:
        return []


def save_watchlist(tickers: list[str]) -> None:
    import json
    WATCHLIST_FILE.write_text(json.dumps(tickers))


def last_updated(name: str) -> str:
    cache = DATA_DIR / f"prices_{name.replace(' ', '_')}.parquet"
    return time.strftime("%d %b %Y, %H:%M", time.localtime(cache.stat().st_mtime)) if cache.exists() else "never"
