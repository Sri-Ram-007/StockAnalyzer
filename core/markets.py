"""Phase 4: markets beyond Indian stocks - US stocks, Indian and US ETFs, and worldwide search."""
import time

import pandas as pd
import yfinance as yf

from core.data import DATA_DIR

SP500_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"

# Popular, liquid ETFs. Each entry: (Yahoo ticker, name, what it holds)
INDIAN_ETFS = [
    ("NIFTYBEES.NS", "Nippon India Nifty 50 BeES", "Large cap index"),
    ("SETFNIF50.NS", "SBI Nifty 50 ETF", "Large cap index"),
    ("JUNIORBEES.NS", "Nippon India Nifty Next 50 BeES", "Large cap index"),
    ("MIDCAPETF.NS", "Mirae Asset Nifty Midcap 150 ETF", "Mid cap index"),
    ("BANKBEES.NS", "Nippon India Bank BeES", "Banking"),
    ("PSUBNKBEES.NS", "Nippon India PSU Bank BeES", "Banking"),
    ("ITBEES.NS", "Nippon India Nifty IT BeES", "Information technology"),
    ("CPSEETF.NS", "CPSE ETF", "Public sector companies"),
    ("GOLDBEES.NS", "Nippon India Gold BeES", "Gold"),
    ("SILVERBEES.NS", "Nippon India Silver BeES", "Silver"),
    ("MON100.NS", "Motilal Oswal Nasdaq 100 ETF", "US technology (Nasdaq 100)"),
    ("MASPTOP50.NS", "Mirae Asset S&P 500 Top 50 ETF", "US large caps"),
    ("HNGSNGBEES.NS", "Nippon India Hang Seng BeES", "Hong Kong stocks"),
    ("LIQUIDBEES.NS", "Nippon India Liquid BeES", "Cash / liquid"),
]
US_ETFS = [
    ("SPY", "SPDR S&P 500 ETF", "US large caps"), ("VOO", "Vanguard S&P 500 ETF", "US large caps"),
    ("VTI", "Vanguard Total Stock Market ETF", "Whole US market"), ("QQQ", "Invesco QQQ (Nasdaq 100)", "US technology"),
    ("IWM", "iShares Russell 2000 ETF", "US small caps"), ("DIA", "SPDR Dow Jones ETF", "US blue chips"),
    ("SCHD", "Schwab US Dividend Equity ETF", "Dividend stocks"), ("VIG", "Vanguard Dividend Appreciation ETF", "Dividend growth"),
    ("XLK", "Technology Select Sector SPDR", "Technology"), ("XLF", "Financial Select Sector SPDR", "Financials"),
    ("XLE", "Energy Select Sector SPDR", "Energy"), ("XLV", "Health Care Select Sector SPDR", "Healthcare"),
    ("SMH", "VanEck Semiconductor ETF", "Semiconductors"), ("VEA", "Vanguard Developed Markets ETF", "Developed markets ex-US"),
    ("VWO", "Vanguard Emerging Markets ETF", "Emerging markets"), ("INDA", "iShares MSCI India ETF", "India"),
    ("EWJ", "iShares MSCI Japan ETF", "Japan"), ("FXI", "iShares China Large-Cap ETF", "China"),
    ("BND", "Vanguard Total Bond Market ETF", "US bonds"), ("TLT", "iShares 20+ Year Treasury Bond ETF", "Long-term US bonds"),
    ("GLD", "SPDR Gold Shares", "Gold"), ("SLV", "iShares Silver Trust", "Silver"), ("VNQ", "Vanguard Real Estate ETF", "Real estate"),
]
ETF_TICKERS = {t for t, _, _ in INDIAN_ETFS + US_ETFS}


def _age_hours(path):
    return (time.time() - path.stat().st_mtime) / 3600 if path.exists() else float("inf")


def get_sp500() -> pd.DataFrame:
    cache = DATA_DIR / "universe_SP500.csv"
    if _age_hours(cache) > 30 * 24:
        try:
            df = pd.read_csv(SP500_URL)
            df.to_csv(cache, index=False)
        except Exception:
            pass
    if not cache.exists():
        raise RuntimeError("Could not download the S&P 500 list and no cached copy exists.")
    df = pd.read_csv(cache)
    return pd.DataFrame({"Company Name": df["Security"], "Industry": df["GICS Sector"], "Symbol": df["Symbol"],
                         "Ticker": df["Symbol"].str.replace(".", "-", regex=False)})


def etf_universe(items) -> pd.DataFrame:
    return pd.DataFrame({"Company Name": [n for _, n, _ in items], "Industry": [c for _, _, c in items],
                         "Symbol": [t.split(".")[0] for t, _, _ in items], "Ticker": [t for t, _, _ in items]})


def global_search(query: str, limit: int = 10) -> list[dict]:
    """Search Yahoo Finance worldwide (stocks and ETFs on any exchange)."""
    try:
        quotes = yf.Search(query, max_results=limit).quotes
    except Exception:
        return []
    return [{"ticker": q["symbol"], "name": q.get("longname") or q.get("shortname") or q["symbol"],
             "exchange": q.get("exchDisp") or q.get("exchange", ""), "type": q.get("quoteType", "")}
            for q in quotes if q.get("quoteType") in ("EQUITY", "ETF") and q.get("symbol")]


CURRENCY_SYMBOLS = {"INR": "₹", "USD": "$", "EUR": "€", "GBP": "£", "GBp": "GBp ", "JPY": "¥", "CNY": "¥",
                    "HKD": "HK$", "SGD": "S$", "AUD": "A$", "CAD": "C$", "CHF": "CHF ", "KRW": "₩"}


def currency_code(ticker: str) -> str:
    """ISO currency code for a ticker, e.g. INR, USD, JPY (cached)."""
    if ticker.endswith((".NS", ".BO")):
        return "INR"
    currency_of(ticker)
    cache = DATA_DIR / "currencies.csv"
    return pd.read_csv(cache, index_col=0)["currency"].to_dict().get(ticker, "USD")


def currency_of(ticker: str) -> str:
    """Currency symbol for a ticker (Indian tickers are rupees; others asked from Yahoo, cached)."""
    if ticker.endswith((".NS", ".BO")):
        return "₹"
    cache = DATA_DIR / "currencies.csv"
    known = pd.read_csv(cache, index_col=0)["currency"].to_dict() if cache.exists() else {}
    if ticker not in known:
        try:
            known[ticker] = yf.Ticker(ticker).fast_info.get("currency") or "USD"
        except Exception:
            known[ticker] = "USD"
        pd.Series(known, name="currency").to_csv(cache)
    code = known[ticker]
    return CURRENCY_SYMBOLS.get(code, f"{code} ")
