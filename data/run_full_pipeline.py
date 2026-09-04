"""
=========================================================================
run_full_pipeline.py
=========================================================================
Script UNICO e completo: scarica i dati reali da Yahoo Finance, costruisce
il database SQL, calcola tutte le metriche (performance, rischio, VaR/CVaR,
esposizioni, attribution, benchmark), ed esporta lo star schema per Power BI.

REQUISITI:
    pip install yfinance pandas openpyxl --break-system-packages

USO:
    python run_full_pipeline.py

OUTPUT (nella cartella corrente):
    positions.csv, prices.csv, benchmark.csv   -> dati grezzi scaricati
    portfolio.db                                -> database SQLite
    analytics_results.json                      -> tutte le metriche calcolate
    correlation_matrix_top10.csv                -> matrice di correlazione
    powerbi/*.csv                                -> star schema per Power BI
=========================================================================
"""
import os
import json
import sqlite3
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf

# =========================================================================
# CONFIGURAZIONE
# =========================================================================
START_DATE = "2026-05-10"  # data di creazione del portafoglio
END_DATE = date.today().strftime("%Y-%m-%d")
BENCHMARK_TICKER = "^GSPC"   # S&P 500
RISK_FREE_RATE = 0.04
TRADING_DAYS = 252
CONFIDENCE_LEVELS = [0.95, 0.99]

OUTPUT_DIR = "."
POWERBI_DIR = "powerbi"
os.makedirs(POWERBI_DIR, exist_ok=True)

# =========================================================================
# 1) DEFINIZIONE PORTAFOGLIO
#    ticker, nome, blocco, asset_class, tema, area, valuta, peso_target
# =========================================================================
PORTFOLIO = [
    # --- TECH GROWTH: CORE (20%) ---
    ("NVDA", "NVIDIA Corp",            "Tech Growth - Core", "Equity", "AI/Semiconductors", "US",     "USD", 0.040),
    ("MSFT", "Microsoft Corp",         "Tech Growth - Core", "Equity", "Cloud/Software",     "US",     "USD", 0.040),
    ("GOOGL","Alphabet Inc",           "Tech Growth - Core", "Equity", "Cloud/AI",           "US",     "USD", 0.035),
    ("AMZN", "Amazon.com Inc",         "Tech Growth - Core", "Equity", "Cloud/E-commerce",   "US",     "USD", 0.035),
    ("TSM",  "Taiwan Semiconductor",   "Tech Growth - Core", "Equity", "Semiconductors",     "Asia",   "USD", 0.025),
    ("ASML", "ASML Holding",           "Tech Growth - Core", "Equity", "Semiconductors",     "Europe", "USD", 0.015),
    ("AMD",  "Advanced Micro Devices", "Tech Growth - Core", "Equity", "Semiconductors",     "US",     "USD", 0.010),

    # --- TECH GROWTH: CONTORNO (10%) ---
    ("EQIX", "Equinix Inc",            "Tech Growth - Contorno", "Equity", "Data Center REIT", "US", "USD", 0.020),
    ("DLR",  "Digital Realty Trust",   "Tech Growth - Contorno", "Equity", "Data Center REIT", "US", "USD", 0.020),
    ("VRT",  "Vertiv Holdings",        "Tech Growth - Contorno", "Equity", "Power & Cooling",   "US", "USD", 0.020),
    ("ETN",  "Eaton Corp",             "Tech Growth - Contorno", "Equity", "Infrastructure",    "US", "USD", 0.015),
    ("PANW", "Palo Alto Networks",     "Tech Growth - Contorno", "Equity", "Cybersecurity",     "US", "USD", 0.015),
    ("CRWD", "CrowdStrike Holdings",   "Tech Growth - Contorno", "Equity", "Cybersecurity",     "US", "USD", 0.010),

    # --- CORE DIVERSIFICATO (20%) ---
    ("JPM",   "JPMorgan Chase & Co",     "Core Diversificato", "Equity", "Financials",  "US",     "USD", 0.030),
    ("ALV.DE","Allianz SE",              "Core Diversificato", "Equity", "Financials",  "Europe", "EUR", 0.020),
    ("SIE.DE","Siemens AG",              "Core Diversificato", "Equity", "Industrials", "Europe", "EUR", 0.020),
    ("HON",   "Honeywell International", "Core Diversificato", "Equity", "Industrials", "US",     "USD", 0.020),
    ("JNJ",   "Johnson & Johnson",       "Core Diversificato", "Equity", "Healthcare",  "US",     "USD", 0.025),
    ("NVO",   "Novo Nordisk",            "Core Diversificato", "Equity", "Healthcare",  "Europe", "USD", 0.025),
    ("LDO.MI","Leonardo SpA",            "Core Diversificato", "Equity", "Defense",     "Europe", "EUR", 0.015),
    ("LMT",   "Lockheed Martin",         "Core Diversificato", "Equity", "Defense",     "US",     "USD", 0.015),
    ("NOC",   "Northrop Grumman",        "Core Diversificato", "Equity", "Defense",     "US",     "USD", 0.010),
    ("GD",    "General Dynamics",        "Core Diversificato", "Equity", "Defense",     "US",     "USD", 0.010),
    ("RTX",   "RTX Corp",                "Core Diversificato", "Equity", "Defense",     "US",     "USD", 0.010),

    # --- FRONTIERA / ALTERNATIVI (5%) ---
    ("ARKX",   "ARK Space Exploration ETF", "Frontiera/Alternativi", "Equity", "Space Economy",  "US",     "USD", 0.010),
    ("ILMN",   "Illumina Inc",              "Frontiera/Alternativi", "Equity", "Genomics",       "US",     "USD", 0.010),
    ("TXG",    "10x Genomics",              "Frontiera/Alternativi", "Equity", "Genomics",       "US",     "USD", 0.005),
    ("BTC-USD","Bitcoin",                   "Frontiera/Alternativi", "Crypto", "Digital Assets", "Global", "USD", 0.015),
    ("ETH-USD","Ethereum",                  "Frontiera/Alternativi", "Crypto", "Digital Assets", "Global", "USD", 0.010),

    # --- COMMODITIES & TERRE RARE (4%) ---
    ("REMX", "VanEck Rare Earth/Strategic Metals ETF",  "Commodities & Terre Rare", "Commodities", "Rare Earth/Metals", "Global", "USD", 0.020),
    ("PICK", "iShares MSCI Global Metals & Mining ETF", "Commodities & Terre Rare", "Commodities", "Metals & Mining",   "Global", "USD", 0.020),

    # --- FIXED INCOME (25%) ---
    ("IEI",  "iShares 3-7Y Treasury Bond ETF",     "Fixed Income - Govt", "Fixed Income", "Government/Treasury", "US", "USD", 0.100),
    ("VCIT", "Vanguard Interm-Term Corp Bond ETF", "Fixed Income - IG",   "Fixed Income", "Corporate IG",         "US", "USD", 0.100),
    ("HYG",  "iShares iBoxx High Yield Corp ETF",  "Fixed Income - HY",   "Fixed Income", "High Yield",           "US", "USD", 0.050),

    # --- LIQUIDITA (16%) ---
    ("CASH", "Liquidita / Cash Equivalent", "Liquidita", "Liquidita", "Cash", "US", "USD", 0.160),
]

# normalizzo i pesi a 1.0 esatto
_raw_total = sum(row[7] for row in PORTFOLIO)
PORTFOLIO = [(t, n, blk, ac, s, g, c, round(w / _raw_total, 5)) for (t, n, blk, ac, s, g, c, w) in PORTFOLIO]

positions_df = pd.DataFrame(
    PORTFOLIO, columns=["ticker", "name", "block", "asset_class", "theme", "region", "currency", "target_weight"]
)
positions_df.to_csv(f"{OUTPUT_DIR}/positions.csv", index=False)
print(f"[1/6] positions.csv creato: {len(positions_df)} posizioni, peso totale {positions_df['target_weight'].sum():.4f}")

# =========================================================================
# 2) DOWNLOAD PREZZI DA YAHOO FINANCE
# =========================================================================
import time

print(f"\n[2/6] Download prezzi da Yahoo Finance ({START_DATE} -> {END_DATE})...")
all_records = []
failed = []

def download_one(ticker, start, end, max_retries=3):
    """Scarica un ticker con retry e fallback. I server cloud (come GitHub Actions)
    sono spesso limitati piu' aggressivamente da Yahoo Finance rispetto a un PC normale."""
    for attempt in range(max_retries):
        try:
            data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True, threads=False)
            if data.empty:
                data = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
            if data.empty:
                time.sleep(2.5)
                continue
            if isinstance(data.columns, pd.MultiIndex):
                if "Close" in data.columns.get_level_values(0):
                    close_col = data["Close"]
                    if isinstance(close_col, pd.DataFrame):
                        close_col = close_col.iloc[:, 0]
                else:
                    close_col = data.iloc[:, 0]
            else:
                close_col = data["Close"] if "Close" in data.columns else data.iloc[:, 0]
            return close_col
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(2.5)
    return pd.Series(dtype=float)

for ticker, name, block, asset_class, theme, region, ccy, weight in PORTFOLIO:
    if ticker == "CASH":
        continue
    try:
        close_col = download_one(ticker, START_DATE, END_DATE)
        if close_col is None or close_col.empty:
            print(f"  VUOTO {ticker:10s} - nessun dato restituito")
            failed.append(ticker)
            time.sleep(1.0)
            continue
        n_rows = 0
        for dt, price in close_col.items():
            if pd.notna(price):
                all_records.append((pd.Timestamp(dt).strftime("%Y-%m-%d"), ticker, round(float(price), 4)))
                n_rows += 1
        print(f"  OK  {ticker:10s} - {n_rows} righe")
    except Exception as e:
        print(f"  ERRORE {ticker}: {e}")
        failed.append(ticker)
    time.sleep(1.0)  # pausa generosa tra le richieste (contesto cloud = piu' rate limiting)

# Liquidita': NAV costante a 100 (cash puro, nessuna serie di prezzo)
dates_ref = pd.bdate_range(start=START_DATE, end=END_DATE)
for dt in dates_ref:
    all_records.append((dt.strftime("%Y-%m-%d"), "CASH", 100.0))

prices_df = pd.DataFrame(all_records, columns=["date", "ticker", "close"])
# --- FIX: allinea tutti i ticker al calendario borsistico (Lun-Ven) ---
# Le crypto (BTC/ETH) hanno prezzi anche nel weekend; le equity/ETF no.
# Senza questo fix, un giorno con copertura parziale (es. solo crypto in un
# weekend) puo' distorcere pesantemente NAV/performance.
prices_df["date"] = pd.to_datetime(prices_df["date"])
prices_df = prices_df[prices_df["date"].dt.dayofweek < 5]  # solo Lun-Ven

_pivot_check = prices_df.pivot(index="date", columns="ticker", values="close").sort_index()
_pivot_check = _pivot_check.ffill()  # porta avanti l'ultimo prezzo nei giorni di festivita' locale

_min_coverage = int(len(_pivot_check.columns) * 0.9)
_coverage = _pivot_check.notna().sum(axis=1)
_valid_dates = _coverage[_coverage >= _min_coverage].index
_pivot_check = _pivot_check.loc[_valid_dates].dropna(how="any")

prices_df = _pivot_check.reset_index().melt(id_vars="date", var_name="ticker", value_name="close")
prices_df["date"] = prices_df["date"].dt.strftime("%Y-%m-%d")

prices_df.to_csv(f"{OUTPUT_DIR}/prices.csv", index=False)
print(f"prices.csv creato: {len(prices_df)} righe totali")
if failed:
    print(f"\nATTENZIONE - ticker falliti: {failed}")
    print("Verifica il simbolo esatto su finance.yahoo.com (specialmente ALV.DE, SIE.DE, LDO.MI)")

# =========================================================================
# 3) DOWNLOAD BENCHMARK (S&P 500)
# =========================================================================
print(f"\n[3/6] Download benchmark {BENCHMARK_TICKER}...")
bench_data = yf.download(BENCHMARK_TICKER, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
bench_close = bench_data["Close"] if "Close" in bench_data.columns else bench_data.iloc[:, 0]
benchmark_df = pd.DataFrame({
    "date": [d.strftime("%Y-%m-%d") for d in bench_close.index],
    "close": bench_close.values.flatten(),
})
benchmark_df.to_csv(f"{OUTPUT_DIR}/benchmark.csv", index=False)
print(f"benchmark.csv creato: {len(benchmark_df)} righe")

# =========================================================================
# 4) COSTRUZIONE DATABASE SQL
# =========================================================================
print("\n[4/6] Costruzione database SQL...")
conn = sqlite3.connect(f"{OUTPUT_DIR}/portfolio.db")
conn.executescript("""
DROP TABLE IF EXISTS positions;
DROP TABLE IF EXISTS prices;

CREATE TABLE positions (
    ticker TEXT PRIMARY KEY, name TEXT NOT NULL, block TEXT NOT NULL,
    asset_class TEXT NOT NULL, theme TEXT NOT NULL, region TEXT NOT NULL,
    currency TEXT NOT NULL, target_weight REAL NOT NULL
);
CREATE TABLE prices (
    date TEXT NOT NULL, ticker TEXT NOT NULL, close REAL NOT NULL,
    PRIMARY KEY (date, ticker),
    FOREIGN KEY (ticker) REFERENCES positions(ticker)
);
CREATE INDEX idx_prices_ticker ON prices(ticker);
CREATE INDEX idx_prices_date ON prices(date);
""")
positions_df.to_sql("positions", conn, if_exists="append", index=False)
prices_df.to_sql("prices", conn, if_exists="append", index=False)

conn.executescript("""
DROP VIEW IF EXISTS v_daily_returns;
CREATE VIEW v_daily_returns AS
SELECT ticker, date, close,
    close / LAG(close) OVER (PARTITION BY ticker ORDER BY date) - 1 AS daily_return
FROM prices;

DROP VIEW IF EXISTS v_position_value;
CREATE VIEW v_position_value AS
WITH first_price AS (SELECT ticker, MIN(date) AS first_date FROM prices GROUP BY ticker),
base AS (SELECT p.ticker, p.close AS base_close FROM prices p
         JOIN first_price f ON p.ticker=f.ticker AND p.date=f.first_date)
SELECT pr.date, pr.ticker, pos.block, pos.asset_class, pos.theme, pos.region, pos.currency,
       pos.target_weight, pos.target_weight*100.0*(pr.close/b.base_close) AS position_value
FROM prices pr JOIN positions pos ON pr.ticker=pos.ticker JOIN base b ON pr.ticker=b.ticker;

DROP VIEW IF EXISTS v_portfolio_nav;
CREATE VIEW v_portfolio_nav AS
SELECT date, SUM(position_value) AS nav FROM v_position_value GROUP BY date ORDER BY date;

DROP VIEW IF EXISTS v_asset_class_exposure;
CREATE VIEW v_asset_class_exposure AS
WITH last_date AS (SELECT MAX(date) AS d FROM v_position_value)
SELECT asset_class, SUM(position_value) AS value,
    SUM(position_value)*100.0/(SELECT SUM(position_value) FROM v_position_value WHERE date=(SELECT d FROM last_date)) AS weight_pct
FROM v_position_value WHERE date=(SELECT d FROM last_date) GROUP BY asset_class ORDER BY weight_pct DESC;

DROP VIEW IF EXISTS v_block_exposure;
CREATE VIEW v_block_exposure AS
WITH last_date AS (SELECT MAX(date) AS d FROM v_position_value)
SELECT block, SUM(position_value) AS value,
    SUM(position_value)*100.0/(SELECT SUM(position_value) FROM v_position_value WHERE date=(SELECT d FROM last_date)) AS weight_pct
FROM v_position_value WHERE date=(SELECT d FROM last_date) GROUP BY block ORDER BY weight_pct DESC;

DROP VIEW IF EXISTS v_geo_exposure;
CREATE VIEW v_geo_exposure AS
WITH last_date AS (SELECT MAX(date) AS d FROM v_position_value)
SELECT region, SUM(position_value) AS value,
    SUM(position_value)*100.0/(SELECT SUM(position_value) FROM v_position_value WHERE date=(SELECT d FROM last_date)) AS weight_pct
FROM v_position_value WHERE date=(SELECT d FROM last_date) GROUP BY region ORDER BY weight_pct DESC;

DROP VIEW IF EXISTS v_currency_exposure;
CREATE VIEW v_currency_exposure AS
WITH last_date AS (SELECT MAX(date) AS d FROM v_position_value)
SELECT currency, SUM(position_value) AS value,
    SUM(position_value)*100.0/(SELECT SUM(position_value) FROM v_position_value WHERE date=(SELECT d FROM last_date)) AS weight_pct
FROM v_position_value WHERE date=(SELECT d FROM last_date) GROUP BY currency ORDER BY weight_pct DESC;
""")
conn.commit()
print("portfolio.db creato con schema e viste")

# =========================================================================
# 5) CALCOLO DI TUTTE LE METRICHE (ANALYTICS)
# =========================================================================
print("\n[5/6] Calcolo metriche (performance, rischio, esposizioni, benchmark)...")

positions = pd.read_sql("SELECT * FROM positions", conn)
prices = pd.read_sql("SELECT * FROM prices", conn)
prices["date"] = pd.to_datetime(prices["date"])
pivot = prices.pivot(index="date", columns="ticker", values="close").sort_index()
returns_matrix = pivot.pct_change().dropna()

nav = pd.read_sql("SELECT * FROM v_portfolio_nav ORDER BY date", conn)
nav["date"] = pd.to_datetime(nav["date"])
nav = nav.set_index("date")
nav["nav_return"] = nav["nav"].pct_change()
daily_returns = nav["nav_return"].dropna()

# --- Performance ---
total_return = nav["nav"].iloc[-1] / nav["nav"].iloc[0] - 1
n_years = (nav.index[-1] - nav.index[0]).days / 365.25
annualized_return = (1 + total_return) ** (1 / n_years) - 1
annualized_vol = daily_returns.std() * np.sqrt(TRADING_DAYS)
sharpe_ratio = (annualized_return - RISK_FREE_RATE) / annualized_vol if annualized_vol > 0 else float("nan")
downside_vol = daily_returns[daily_returns < 0].std() * np.sqrt(TRADING_DAYS)
sortino_ratio = (annualized_return - RISK_FREE_RATE) / downside_vol if downside_vol > 0 else np.nan
cum_max = nav["nav"].cummax()
drawdown = nav["nav"] / cum_max - 1
max_drawdown = drawdown.min()
max_dd_date = drawdown.idxmin()

# --- VaR / CVaR ---
var_results, cvar_results = {}, {}
for cl in CONFIDENCE_LEVELS:
    pct = (1 - cl) * 100
    var_hist = np.percentile(daily_returns, pct)
    z = {0.95: 1.645, 0.99: 2.326}[cl]
    var_param = daily_returns.mean() - z * daily_returns.std()
    cvar = daily_returns[daily_returns <= var_hist].mean()
    var_results[f"VaR_{int(cl*100)}_historical_daily_pct"] = round(var_hist * 100, 3)
    var_results[f"VaR_{int(cl*100)}_parametric_daily_pct"] = round(var_param * 100, 3)
    cvar_results[f"CVaR_{int(cl*100)}_daily_pct"] = round(float(cvar) * 100, 3)

# --- Esposizioni ---
def exposure_table(df, key_col):
    return (df.groupby(key_col)["target_weight"].sum() * 100).round(2).sort_values(ascending=False).to_dict()

asset_class_exposure = exposure_table(positions, "asset_class")
block_exposure = exposure_table(positions, "block")
region_exposure = exposure_table(positions, "region")
currency_exposure = exposure_table(positions, "currency")

# --- Concentrazione ---
weights_pct = positions["target_weight"] * 100
hhi = float((weights_pct ** 2).sum())
top5_weight = float(positions.sort_values("target_weight", ascending=False).head(5)["target_weight"].sum() * 100)
top10_weight = float(positions.sort_values("target_weight", ascending=False).head(10)["target_weight"].sum() * 100)
concentration_risk = {
    "hhi": round(hhi, 1),
    "hhi_label": "Diversificato" if hhi < 1500 else ("Moderatamente Concentrato" if hhi < 2500 else "Concentrato"),
    "top5_weight_pct": round(top5_weight, 2), "top10_weight_pct": round(top10_weight, 2),
}

# --- Attribution ---
first_last = prices.groupby("ticker")["date"].agg(["min", "max"]).reset_index()
first_prices = prices.merge(first_last, left_on=["ticker","date"], right_on=["ticker","min"])[["ticker","close"]].rename(columns={"close":"first_close"})
last_prices = prices.merge(first_last, left_on=["ticker","date"], right_on=["ticker","max"])[["ticker","close"]].rename(columns={"close":"last_close"})
perf = positions.merge(first_prices, on="ticker").merge(last_prices, on="ticker")
perf["stock_return"] = perf["last_close"] / perf["first_close"] - 1
perf["contribution"] = perf["target_weight"] * perf["stock_return"]
block_attribution = (perf.groupby("block")["contribution"].sum() * 100).round(3).sort_values(ascending=False).to_dict()
asset_class_attribution = (perf.groupby("asset_class")["contribution"].sum() * 100).round(3).sort_values(ascending=False).to_dict()
top_5_performers = perf.sort_values("stock_return", ascending=False)[["ticker","name","block","stock_return"]].head(5).assign(stock_return=lambda d: (d["stock_return"]*100).round(2)).to_dict("records")
bottom_5_performers = perf.sort_values("stock_return")[["ticker","name","block","stock_return"]].head(5).assign(stock_return=lambda d: (d["stock_return"]*100).round(2)).to_dict("records")

# --- Benchmark: Alpha, Beta, Tracking Error, Information Ratio ---
benchmark_metrics = {}
rolling_metrics = []
try:
    bench = pd.read_csv(f"{OUTPUT_DIR}/benchmark.csv")
    bench["date"] = pd.to_datetime(bench["date"])
    bench = bench.set_index("date")
    bench_ret = bench["close"].pct_change().dropna()
    merged = pd.concat([daily_returns.rename("port"), bench_ret.rename("bench")], axis=1).dropna()
    beta = merged["port"].cov(merged["bench"]) / merged["bench"].var()
    ann_bench_ret = (1 + merged["bench"]).prod() ** (TRADING_DAYS/len(merged)) - 1
    alpha = (annualized_return - RISK_FREE_RATE) - beta * (ann_bench_ret - RISK_FREE_RATE)
    te = (merged["port"] - merged["bench"]).std() * np.sqrt(TRADING_DAYS)
    ir = (annualized_return - ann_bench_ret) / te if te > 0 else np.nan
    benchmark_metrics = {
        "beta": round(float(beta), 3), "alpha_annualized_pct": round(float(alpha)*100, 2),
        "tracking_error_pct": round(float(te)*100, 2), "information_ratio": round(float(ir), 3),
        "benchmark_annualized_return_pct": round(float(ann_bench_ret)*100, 2),
    }
    WINDOW = 20  # ridotta: storico breve dal 10 maggio
    rolling_vol = merged["port"].rolling(WINDOW).std() * np.sqrt(TRADING_DAYS)
    rolling_ann_ret = merged["port"].rolling(WINDOW).apply(lambda x: (1+x).prod()**(TRADING_DAYS/WINDOW)-1, raw=True)
    rolling_sharpe = (rolling_ann_ret - RISK_FREE_RATE) / rolling_vol
    rolling_beta = merged["port"].rolling(WINDOW).cov(merged["bench"]) / merged["bench"].rolling(WINDOW).var()
    rdf = pd.DataFrame({
        "date": merged.index.strftime("%Y-%m-%d"),
        "rolling_vol_pct": (rolling_vol*100).round(2), "rolling_sharpe": rolling_sharpe.round(3),
        "rolling_beta": rolling_beta.round(3),
    }).dropna()
    rolling_metrics = rdf.iloc[::3].to_dict("records")
except Exception as e:
    print(f"  Nota: benchmark non disponibile per Alpha/Beta/TE/IR ({e})")

# --- Correlazione (Top 10 per peso) ---
top10_tickers_raw = positions.sort_values("target_weight", ascending=False).head(10)["ticker"].tolist()
top10_tickers = [t for t in top10_tickers_raw if t in returns_matrix.columns]
missing_from_corr = [t for t in top10_tickers_raw if t not in returns_matrix.columns]
if missing_from_corr:
    print(f"ATTENZIONE - questi ticker top10 non hanno dati prezzo: {missing_from_corr}")
if top10_tickers:
    corr_matrix = returns_matrix[top10_tickers].corr().round(3)
    corr_matrix.to_csv(f"{OUTPUT_DIR}/correlation_matrix_top10.csv")
else:
    print("Nessun ticker disponibile per la matrice di correlazione.")
    corr_matrix = pd.DataFrame()

results = {
    "period": {"start": str(nav.index[0].date()), "end": str(nav.index[-1].date()), "years": round(n_years, 2)},
    "performance": {
        "total_return_pct": round(total_return*100, 2), "annualized_return_pct": round(annualized_return*100, 2),
        "annualized_volatility_pct": round(annualized_vol*100, 2), "sharpe_ratio": round(sharpe_ratio, 3),
        "sortino_ratio": round(sortino_ratio, 3), "max_drawdown_pct": round(max_drawdown*100, 2),
        "max_drawdown_date": str(max_dd_date.date()),
    },
    "risk": var_results, "cvar": cvar_results,
    "asset_class_exposure": asset_class_exposure, "block_exposure": block_exposure,
    "region_exposure": region_exposure, "currency_exposure": currency_exposure,
    "concentration_risk": concentration_risk,
    "block_attribution_pct": block_attribution, "asset_class_attribution_pct": asset_class_attribution,
    "top_5_performers": top_5_performers, "bottom_5_performers": bottom_5_performers,
    "benchmark": benchmark_metrics, "rolling_metrics": rolling_metrics,
}
with open(f"{OUTPUT_DIR}/analytics_results.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("analytics_results.json salvato")
print(json.dumps(results["performance"], indent=2))
print("Esposizione Asset Class:", asset_class_exposure)

# =========================================================================
# 6) EXPORT STAR SCHEMA PER POWER BI
# =========================================================================
print("\n[6/6] Export star schema per Power BI...")
positions.to_csv(f"{POWERBI_DIR}/dim_positions.csv", index=False)

dates = pd.read_sql("SELECT DISTINCT date FROM prices ORDER BY date", conn)
dates["date"] = pd.to_datetime(dates["date"])
dim_date = pd.DataFrame({"date": dates["date"]})
dim_date["year"] = dim_date["date"].dt.year
dim_date["month"] = dim_date["date"].dt.month
dim_date["month_name"] = dim_date["date"].dt.strftime("%b")
dim_date["quarter"] = dim_date["date"].dt.quarter
dim_date["year_month"] = dim_date["date"].dt.strftime("%Y-%m")
dim_date["weekday"] = dim_date["date"].dt.day_name()
dim_date.to_csv(f"{POWERBI_DIR}/dim_date.csv", index=False)

fact_prices = pd.read_sql("SELECT * FROM v_daily_returns", conn)
fact_prices.to_csv(f"{POWERBI_DIR}/fact_prices.csv", index=False)

nav_export = pd.read_sql("SELECT * FROM v_portfolio_nav ORDER BY date", conn)
nav_export["daily_return"] = nav_export["nav"].pct_change()
nav_export["cumulative_return_pct"] = (nav_export["nav"] / nav_export["nav"].iloc[0] - 1) * 100
nav_export["drawdown_pct"] = (nav_export["nav"] / nav_export["nav"].cummax() - 1) * 100
nav_export.to_csv(f"{POWERBI_DIR}/fact_portfolio_nav.csv", index=False)

for view, fname in [
    ("v_asset_class_exposure", "dim_asset_class_exposure.csv"),
    ("v_block_exposure", "dim_block_exposure.csv"),
    ("v_geo_exposure", "dim_geo_exposure.csv"),
    ("v_currency_exposure", "dim_currency_exposure.csv"),
]:
    df = pd.read_sql(f"SELECT * FROM {view}", conn)
    df.to_csv(f"{POWERBI_DIR}/{fname}", index=False)

conn.close()

print("\n" + "="*60)
print("PIPELINE COMPLETATA CON SUCCESSO")
print("="*60)
print("File generati:")
print("  positions.csv, prices.csv, benchmark.csv")
print("  portfolio.db")
print("  analytics_results.json, correlation_matrix_top10.csv")
print(f"  {POWERBI_DIR}/*.csv (8 file, star schema)")
print("\nProssimo passo: carica positions.csv, prices.csv e")
print("analytics_results.json in chat per rigenerare dashboard,")
print("Excel e IC Memo con i dati reali.")
