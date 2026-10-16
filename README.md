# Stock Analyzer

Personal stock analysis tool. Runs on your own computer, uses free data, needs no API keys.

## Start it
Double-click **Start Stock Analyzer.bat**. It opens in your browser at http://localhost:8501.
Close the black window to stop it.

The first run on a new computer creates a Python environment automatically (needs Python 3.10+ installed).

## Phase 1 features
- **Recovery Screener**: stocks that fell from their 1-year high, with how often each one (and all stocks) recovered from a fall that big in the past.
- **Stock Deep Dive**: price chart, 1-year high, 200-day average, every past fall and whether it recovered.

## Data
- Index lists: NSE Indices (Nifty 50/100/200/500), refreshed monthly.
- Prices: Yahoo Finance via `yfinance`, 10 years, adjusted for splits and dividends, cached in `data/` and refreshed every 12 hours.

Educational tool, not investment advice.

## Phase 2 features
- **Search any NSE stock** (about 2,350 companies), not just Nifty lists.
- **All NSE stocks** option in Settings for Explore and Fallen Stocks (first load about 10 minutes, then cached).
- **Business health check** on every stock page: profit, profit trend, sales trend, debt, margin, valuation vs its own history, and whether the fall was broad or company-specific.
- **Recovery score (0-100)** combining past recoveries, business health, valuation and fall type. Change the weights with "Change what matters to you".
- Fallen Stocks: **Check business health** button adds Score and Business health columns.

Company financials come from Yahoo Finance via yfinance and are cached for 7 days in `data/fundamentals/`.

## Phase 3: Indian mutual funds
- Every open-ended scheme from AMFI (about 9,700), searchable.
- **Explore categories**: rank all funds in a category by 1/3/5-year return, return per risk, consistency (share of 3-year periods beating 10%) or smallest worst fall. Direct/Growth filters.
- **Fund page**: growth of Rs 10,000 vs Nifty 50, returns, volatility, worst fall, and every fall of 10%+ with how long it took to recover.
- **Compare** up to 4 funds side by side.
- Data: AMFI NAVAll.txt (official list) and mfapi.in (NAV history), no keys.

## Phase 4: US and global
- **US**: S&P 500 stocks and popular US ETFs, with the same Explore / Fallen / health check / score.
- **Indian ETFs** list under India settings.
- **Global**: search any stock or ETF worldwide (Japan, Korea, UK, Europe, ...), prices in local currency.

## Phase 5: portfolio and daily alerts
- **💼 Portfolio → Holdings**: value, invested, total and today's profit/loss (all in rupees; foreign holdings converted
  at the buy-date and current exchange rates), per-holding table, allocation chart, CSV download.
- **Add holding**: Indian stocks/ETFs, US or global stocks/ETFs, Indian mutual funds.
- **Alerts**: choose rules (fall below 1-year high, big daily move, holding loss, price targets), set up Telegram and/or
  Gmail, preview and send a test message.
- **Automatic daily alerts**:
  - GitHub Actions (`.github/workflows/daily_alerts.yml`), runs weekdays ~5 pm IST even when the laptop is off.
    Add repository secrets `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` (and/or `EMAIL_ADDRESS`, `EMAIL_APP_PASSWORD`, `EMAIL_TO`).
  - Or Windows Task Scheduler running `Send Alerts Now.bat`.
- Command line: `python run_alerts.py --dry-run` prints today's message without sending.

Private files that never leave your computer (see `.gitignore`): `data/alerts_config.json` (bot token / app password).

## Phase 6: ML outlook (honestly back-tested)
- **🤖 ML outlook** page (India and US): a gradient-boosted tree model using 15 price-based signals estimates the chance a
  stock is higher, or beats the index, in about 3 months.
- **Walk-forward testing** on years the model never saw, with a gap so answers never leak, compared against
  "always guess the most common outcome". Per-year accuracy, AUC, returns of the model's top picks, signal importance.
- Result when built (Oct 2026): **no real edge** on Nifty 100 or S&P 500 (AUC about 0.50). Predictions are therefore hidden
  behind a warning. Re-train any time with the latest prices.
- Models and reports are saved in `data/models/`.
