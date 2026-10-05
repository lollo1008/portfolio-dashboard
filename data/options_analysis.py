#!/usr/bin/env python3
"""
options_analysis.py
===================
Option-market analytics for the Portfolio Dashboard ("Options" page in Power BI).

For each underlying and each target expiry it computes:

  * implied volatility per strike (Black-Scholes on the FORWARD, Brent root-finding)
  * a smoothed volatility smile
  * the risk-neutral density via Breeden-Litzenberger:  f(K) = e^{rT} * d2C/dK2
    (same idea as the course notebook: interpolate/smooth in VOLATILITY space,
     convert back to prices with Black-Scholes, then take the 2nd derivative)
  * the risk-neutral CDF  P(S_T < K) = 1 + e^{rT} * dC/dK  (no truncation needed)
  * a lognormal benchmark (flat vol = ATM IV) to show skew / fat tails
  * open-interest statistics: put/call ratios, max pain, OI "walls"

Differences from the course notebook (all deliberate):
  1. Forward from put-call parity (handles dividends) instead of spot with r = 0.
  2. Out-of-the-money options only (puts below the forward, calls above):
     ITM quotes are illiquid and carry little time value, so their IV is noisy.
  3. Brent root-finder instead of minimize_scalar (exact, faster, flags no-solution).
  4. Strikes are resampled to a uniform grid BEFORE Gaussian smoothing, because
     listed strikes are unevenly spaced ($1 near the money, $5-10 in the wings).
  5. Expiry and time to expiry are inputs, not hard-coded (t = 3/52 in the notebook).

Outputs (CSV, written to --outdir, ready for Power BI):
  fact_options_summary.csv   one row per (as_of, ticker, target_dte)  -> APPENDED (history)
  fact_options_smile.csv     latest snapshot: strike, raw IV, smoothed IV
  fact_options_density.csv   latest snapshot: strike, risk-neutral density, CDF, lognormal benchmark
  fact_options_oi.csv        latest snapshot: open interest and volume by strike

Usage:
  python options_analysis.py --tickers SPY --target-dte 30 60 90 --outdir data/powerbi
  python options_analysis.py --tickers SPY NVDA --rate 0.04
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import brentq
from scipy.stats import norm

_trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy 2.x / 1.x

# --------------------------------------------------------------------------- #
# Black-Scholes on the forward
# --------------------------------------------------------------------------- #

def bs_price(F, K, T, sigma, r, cp="c"):
    """Black (1976) price on forward F. Vectorised in K and sigma."""
    F = np.asarray(F, float)
    K = np.asarray(K, float)
    sigma = np.asarray(sigma, float)
    df = np.exp(-r * T)
    v = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * v * v) / v
    d2 = d1 - v
    if cp == "c":
        return df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def implied_vol(price, F, K, T, r, cp="c"):
    """Implied vol by Brent's method. Returns NaN if no arbitrage-free solution."""
    if not (np.isfinite(price) and price > 0 and T > 0):
        return np.nan
    df = np.exp(-r * T)
    intrinsic = df * max(F - K, 0.0) if cp == "c" else df * max(K - F, 0.0)
    upper = df * F if cp == "c" else df * K
    if price <= intrinsic + 1e-10 or price >= upper:
        return np.nan
    lo, hi = 1e-4, 5.0
    f = lambda s: float(bs_price(F, K, T, s, r, cp)) - price
    if f(lo) > 0 or f(hi) < 0:
        return np.nan
    return brentq(f, lo, hi, xtol=1e-9)


# --------------------------------------------------------------------------- #
# Chain cleaning
# --------------------------------------------------------------------------- #

@dataclass
class Params:
    max_rel_spread: float = 0.60   # drop two-sided quotes with (ask-bid)/mid above this
    min_oi: int = 5                # drop strikes with open interest below this (dead strikes)
    m_lo: float = 0.80             # use strikes within [m_lo, m_hi] * forward
    m_hi: float = 1.20
    smooth_frac: float = 0.020     # Gaussian width as a fraction of the forward (tuned on synthetic chains)
    n_grid: int = 801
    min_strikes: int = 8


def add_mid(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    """Mid price from a valid two-sided quote; else last trade. Flags the source."""
    df = df.copy()
    for c in ("bid", "ask", "lastPrice", "volume", "openInterest"):
        if c not in df:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    two_sided = (df.bid > 0) & (df.ask >= df.bid)
    mid2 = (df.bid + df.ask) / 2
    df["mid"] = np.where(two_sided, mid2, np.where(df.lastPrice > 0, df.lastPrice, np.nan))
    df["price_source"] = np.where(two_sided, "mid", "last")
    df["rel_spread"] = np.where(two_sided, (df.ask - df.bid) / np.maximum(mid2, 1e-12), np.nan)
    bad_spread = two_sided & (df.rel_spread > p.max_rel_spread)
    return df[df.mid.notna() & ~bad_spread].copy()


def forward_from_parity(calls, puts, spot, T, r, n=6):
    """F = K + (C - P) e^{rT}, median over the n strikes closest to spot."""
    m = calls[["strike", "mid"]].merge(puts[["strike", "mid"]], on="strike", suffixes=("_c", "_p"))
    carry = spot * np.exp(r * T)
    if len(m) < 3:
        return float(carry), "spot_carry"
    m = m.assign(dist=(m.strike - spot).abs()).nsmallest(n, "dist")
    F = float((m.strike + (m.mid_c - m.mid_p) * np.exp(r * T)).median())
    if abs(F / carry - 1) > 0.05:          # implausible -> stale quotes
        return float(carry), "spot_carry"
    return F, "parity"


def build_smile(calls, puts, F, T, r, p: Params) -> pd.DataFrame:
    """OTM implied vols: puts for K < F, calls for K >= F."""
    rows = []
    for _, x in calls[calls.strike >= F].iterrows():
        rows.append((x.strike, implied_vol(x.mid, F, x.strike, T, r, "c"), "call", x.mid,
                     x.openInterest, x.volume, x.price_source))
    for _, x in puts[puts.strike < F].iterrows():
        rows.append((x.strike, implied_vol(x.mid, F, x.strike, T, r, "p"), "put", x.mid,
                     x.openInterest, x.volume, x.price_source))
    s = pd.DataFrame(rows, columns=["strike", "iv", "side", "mid", "oi", "volume", "price_source"])
    s = s.dropna(subset=["iv"])
    s = s[(s.strike >= p.m_lo * F) & (s.strike <= p.m_hi * F)]
    # liquidity filter on open interest - but Yahoo often returns OI = 0 outside market hours;
    # if the filter would leave too few strikes, skip it (and say so in the summary)
    s_oi = s[s.oi >= p.min_oi]
    applied = len(s_oi) >= p.min_strikes
    s = s_oi if applied else s
    s = s.assign(oi_filter_applied=applied)
    s = s[(s.iv > 0.02) & (s.iv < 3.0)]
    return s.sort_values("strike").drop_duplicates("strike").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Breeden-Litzenberger
# --------------------------------------------------------------------------- #

def implied_density(strikes, ivs, F, T, r, smooth_frac=0.02, n_grid=801):
    """
    Risk-neutral density and CDF from a set of (strike, IV) points.

    1. resample IV on a uniform strike grid   2. Gaussian-smooth in vol space
    3. back to call prices (Black-Scholes)    4. finite differences in K
         f(K)      = e^{rT} * C''(K)
         P(S_T<K)  = 1 + e^{rT} * C'(K)
    The outer 2 smoothing-widths are trimmed (boundary effect of the filter).
    """
    strikes = np.asarray(strikes, float)
    ivs = np.asarray(ivs, float)
    grid = np.linspace(strikes.min(), strikes.max(), n_grid)
    h = grid[1] - grid[0]
    iv_lin = np.interp(grid, strikes, ivs)
    sig_pts = max(smooth_frac * F / h, 1e-6)
    iv_s = gaussian_filter1d(iv_lin, sig_pts, mode="nearest")

    C = bs_price(F, grid, T, iv_s, r, "c")
    d1 = (C[2:] - C[:-2]) / (2 * h)
    d2 = (C[2:] - 2 * C[1:-1] + C[:-2]) / h ** 2
    K, iv_in = grid[1:-1], iv_s[1:-1]
    pdf_raw = np.exp(r * T) * d2
    cdf = 1 + np.exp(r * T) * d1

    trim = int(2 * sig_pts) + 1
    sl = slice(trim, len(K) - trim) if len(K) > 2 * trim + 10 else slice(None)
    K, iv_in, pdf_raw, cdf = K[sl], iv_in[sl], pdf_raw[sl], cdf[sl]
    pdf = np.clip(pdf_raw, 0, None)
    neg_share = float(_trapz(np.clip(-pdf_raw, 0, None), K) /
                      max(_trapz(np.abs(pdf_raw), K), 1e-300))
    return dict(K=K, iv=iv_in, pdf=pdf, cdf=cdf, mass=float(_trapz(pdf, K)), neg_share=neg_share)


def lognormal_benchmark(K, F, T, sigma):
    """Flat-vol (Black-Scholes) risk-neutral density and CDF with the same forward."""
    mu = np.log(F) - 0.5 * sigma ** 2 * T
    s = sigma * np.sqrt(T)
    z = (np.log(K) - mu) / s
    return norm.pdf(z) / (K * s), norm.cdf(z)


def _moments(K, w, F):
    """Mean, std, skew, excess kurtosis of the log-return x = ln(K/F) under density weights w."""
    x = np.log(K / F)
    w = w / w.sum()
    m = (w * x).sum()
    v = (w * (x - m) ** 2).sum()
    sd = np.sqrt(v)
    return m, sd, (w * (x - m) ** 3).sum() / sd ** 3, (w * (x - m) ** 4).sum() / sd ** 4 - 3


def prob_below(K, cdf, level):
    """P(S_T < level), NaN outside the range covered by the density."""
    if level < K[0] or level > K[-1]:
        return np.nan
    return float(np.interp(level, K, cdf))


# --------------------------------------------------------------------------- #
# Open interest statistics
# --------------------------------------------------------------------------- #

def oi_table(calls, puts) -> pd.DataFrame:
    c = calls.groupby("strike")[["openInterest", "volume"]].sum()
    c.columns = ["call_oi", "call_volume"]
    q = puts.groupby("strike")[["openInterest", "volume"]].sum()
    q.columns = ["put_oi", "put_volume"]
    return c.join(q, how="outer").fillna(0.0).reset_index().sort_values("strike")


def max_pain(oi: pd.DataFrame) -> float:
    """Expiry price that minimises the total intrinsic value paid to option holders."""
    if float(oi.call_oi.sum()) <= 0 or float(oi.put_oi.sum()) <= 0:
        return float('nan')          # need OI on both sides, else the result is an artefact
    ks = oi.strike.values
    best_k, best_pay = np.nan, np.inf
    for k in ks:
        pay = (oi.call_oi.values * np.clip(k - ks, 0, None)).sum() + \
              (oi.put_oi.values * np.clip(ks - k, 0, None)).sum()
        if pay < best_pay:
            best_k, best_pay = k, pay
    return float(best_k)


# --------------------------------------------------------------------------- #
# One expiry, end to end
# --------------------------------------------------------------------------- #

def _diagnose(calls, puts, calls_c, puts_c, spot, F, f_src, T, r, p):
    """One-line account of where the strikes were lost (printed when an expiry is skipped)."""
    def n(df, col, cond):
        return int(cond(pd.to_numeric(df[col], errors="coerce").fillna(0)).sum()) if col in df else -1
    ivs = [implied_vol(x.mid, F, x.strike, T, r, "c" if x.strike >= F else "p")
           for df in (calls_c, puts_c) for _, x in df.iterrows()]
    n_iv = int(np.isfinite(ivs).sum())
    return (f"spot={spot:.2f} F={F:.2f}({f_src}) raw calls/puts={len(calls)}/{len(puts)} "
            f"bid>0={n(calls,'bid',lambda x:x>0)}/{n(puts,'bid',lambda x:x>0)} "
            f"last>0={n(calls,'lastPrice',lambda x:x>0)}/{n(puts,'lastPrice',lambda x:x>0)} "
            f"OI>={p.min_oi}={n(calls,'openInterest',lambda x:x>=p.min_oi)}/{n(puts,'openInterest',lambda x:x>=p.min_oi)} "
            f"after_mid={len(calls_c)}/{len(puts_c)} valid_iv={n_iv}")


def analyze_expiry(calls, puts, spot, T, r, p: Params | None = None):
    """Returns (summary dict, smile df, density df, oi df) or (None, reason) on failure."""
    p = p or Params()
    calls_c, puts_c = add_mid(calls, p), add_mid(puts, p)
    oi = oi_table(calls, puts)                    # OI/volume use the full chain
    F, f_src = forward_from_parity(calls_c, puts_c, spot, T, r)
    smile = build_smile(calls_c, puts_c, F, T, r, p)
    if len(smile) < p.min_strikes:
        return None, (f"only {len(smile)} usable strikes (< {p.min_strikes}) | " + _diagnose(calls, puts, calls_c, puts_c, spot, F, f_src, T, r, p))

    d = implied_density(smile.strike, smile.iv, F, T, r, p.smooth_frac, p.n_grid)
    K = d["K"]
    atm_iv = float(np.interp(F, K, d["iv"])) if K[0] <= F <= K[-1] else float(smile.iv.median())
    ln_pdf, ln_cdf = lognormal_benchmark(K, F, T, atm_iv)

    mom_rn = _moments(K, d["pdf"], F)
    mom_ln = _moments(K, ln_pdf, F)          # same truncation -> like-for-like

    def iv_at(m):
        k = m * F
        return float(np.interp(k, K, d["iv"])) if K[0] <= k <= K[-1] else np.nan

    s = dict(
        spot=spot, forward=F, forward_source=f_src, rate=r, T_years=T, atm_iv=atm_iv,
        expected_move_1sd_pct=atm_iv * np.sqrt(T) * 100,
        iv_90pct=iv_at(0.90), iv_110pct=iv_at(1.10),
        n_strikes_used=len(smile), share_last_price=float((smile.price_source == "last").mean()),
        density_mass=d["mass"], density_neg_share=d["neg_share"],
        range_low=float(K[0]), range_high=float(K[-1]),
        rn_std_logret=mom_rn[1], rn_skew=mom_rn[2], rn_exkurt=mom_rn[3],
        ln_std_logret=mom_ln[1], ln_skew=mom_ln[2], ln_exkurt=mom_ln[3],
        total_call_oi=float(oi.call_oi.sum()), total_put_oi=float(oi.put_oi.sum()),
        total_call_volume=float(oi.call_volume.sum()), total_put_volume=float(oi.put_volume.sum()),
        max_pain=max_pain(oi),
    )
    s["oi_available"] = bool(s["total_call_oi"] > 0 and s["total_put_oi"] > 0)
    s["oi_filter_applied"] = bool(smile.oi_filter_applied.iloc[0])
    s["quote_quality"] = "live" if s["share_last_price"] < 0.5 else "stale_last_prices"
    s["iv_skew_90_110"] = s["iv_90pct"] - s["iv_110pct"]
    s["put_call_oi_ratio"] = (s["total_put_oi"] / s["total_call_oi"]
                              if s["oi_available"] else np.nan)
    s["put_call_volume_ratio"] = (s["total_put_volume"] / s["total_call_volume"]
                                  if s["total_call_volume"] else np.nan)
    above = oi[oi.strike >= spot]
    below = oi[oi.strike <= spot]
    s["call_oi_wall"] = (float(above.loc[above.call_oi.idxmax(), "strike"])
                         if len(above) and above.call_oi.max() > 0 else np.nan)
    s["put_oi_wall"] = (float(below.loc[below.put_oi.idxmax(), "strike"])
                        if len(below) and below.put_oi.max() > 0 else np.nan)

    # market-implied probabilities of moves vs the lognormal benchmark
    for pct in (5, 10, 15):
        lvl = spot * (1 - pct / 100)
        s[f"p_down_{pct}_rn"] = prob_below(K, d["cdf"], lvl)
        s[f"p_down_{pct}_ln"] = prob_below(K, ln_cdf, lvl) if K[0] <= lvl <= K[-1] else np.nan
    for pct in (5, 10):
        lvl = spot * (1 + pct / 100)
        pb = prob_below(K, d["cdf"], lvl)
        s[f"p_up_{pct}_rn"] = 1 - pb if np.isfinite(pb) else np.nan
        pbl = prob_below(K, ln_cdf, lvl) if K[0] <= lvl <= K[-1] else np.nan
        s[f"p_up_{pct}_ln"] = 1 - pbl if np.isfinite(pbl) else np.nan

    smile_out = smile.assign(iv_smooth=np.interp(smile.strike, K, d["iv"], left=np.nan, right=np.nan))
    smile_out["moneyness"] = smile_out.strike / F
    dens = pd.DataFrame(dict(strike=K, moneyness=K / F, density=d["pdf"], cdf=d["cdf"],
                             density_lognormal=ln_pdf, cdf_lognormal=ln_cdf, iv_smooth=d["iv"]))
    return s, smile_out, dens, oi


# --------------------------------------------------------------------------- #
# Data download (yfinance) and orchestration
# --------------------------------------------------------------------------- #

def pick_expiries(expiries, today: dt.date, targets, min_dte=5):
    """For each target DTE choose the closest listed expiry (deduplicated)."""
    cands = [(e, (dt.date.fromisoformat(e) - today).days) for e in expiries]
    cands = [c for c in cands if c[1] >= min_dte]
    chosen = {}
    for t in targets:
        if not cands:
            break
        best = min(cands, key=lambda c: abs(c[1] - t))
        chosen.setdefault(best[0], t)
    return [(e, (dt.date.fromisoformat(e) - today).days, t) for e, t in chosen.items()]


def get_risk_free(yf, default):
    """13-week T-bill (^IRX, quoted in %) with a fallback; always printed."""
    try:
        h = yf.Ticker("^IRX").history(period="5d")["Close"].dropna()
        if len(h):
            r = float(h.iloc[-1]) / 100
            print(f"[rate] ^IRX = {r:.4f}")
            return r
    except Exception as e:                                  # noqa: BLE001
        print(f"[rate] ^IRX failed ({e})")
    print(f"[rate] using default {default:.4f}")
    return default


def run(tickers, targets, outdir, rate, params: Params):
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("yfinance not installed: pip install yfinance")

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    today = dt.date.today()
    r = get_risk_free(yf, 0.04) if rate is None else rate
    summaries, smiles, denss, ois = [], [], [], []

    for tkr in tickers:
        tk = yf.Ticker(tkr)
        px = tk.history(period="5d")["Close"].dropna()
        if px.empty or not tk.options:
            print(f"[{tkr}] no price or no listed options -> skipped")
            continue
        spot = float(px.iloc[-1])
        for exp, dte, target in pick_expiries(tk.options, today, targets):
            try:
                ch = tk.option_chain(exp)
            except Exception as e:                           # noqa: BLE001
                print(f"[{tkr} {exp}] download failed: {e}")
                continue
            res = analyze_expiry(ch.calls, ch.puts, spot, dte / 365.0, r, params)
            if res[0] is None:
                print(f"[{tkr} {exp}] skipped: {res[1]}")
                continue
            s, sm, de, oi = res
            key = dict(as_of=today.isoformat(), ticker=tkr, expiry=exp, dte=dte, target_dte=target)
            summaries.append({**key, **s})
            smiles.append(sm.assign(**key))
            denss.append(de.assign(**key))
            ois.append(oi.assign(**key))
            print(f"[{tkr} {exp}] DTE={dte} F={s['forward']:.2f} ATM IV={s['atm_iv']:.1%} "
                  f"strikes={s['n_strikes_used']} mass={s['density_mass']:.3f} "
                  f"P(-10%) rn={s['p_down_10_rn']:.3f} ln={s['p_down_10_ln']:.3f}")

    if not summaries:
        sys.exit("No expiry could be analysed - nothing written.")

    # snapshots overwrite, summary appends (history for time-series visuals)
    pd.concat(smiles).to_csv(outdir / "fact_options_smile.csv", index=False)
    pd.concat(denss).to_csv(outdir / "fact_options_density.csv", index=False)
    pd.concat(ois).to_csv(outdir / "fact_options_oi.csv", index=False)
    new = pd.DataFrame(summaries)
    path = outdir / "fact_options_summary.csv"
    if path.exists():
        old = pd.read_csv(path)
        keys = ["as_of", "ticker", "target_dte"]
        old = old.merge(new[keys].drop_duplicates(), on=keys, how="left", indicator=True)
        old = old[old["_merge"] == "left_only"].drop(columns="_merge")
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)
    print(f"written to {outdir.resolve()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="+", default=["SPY"])
    ap.add_argument("--target-dte", nargs="+", type=int, default=[30, 60, 90])
    ap.add_argument("--outdir", default="data/powerbi")
    ap.add_argument("--rate", type=float, default=None,
                    help="annual risk-free rate; default = latest ^IRX, fallback 0.04")
    a = ap.parse_args()
    run(a.tickers, a.target_dte, a.outdir, a.rate, Params())


if __name__ == "__main__":
    main()
