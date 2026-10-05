"""
Validation of options_analysis.py on synthetic chains whose true answer is known.

Run:  python tests/test_options_analysis.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
import options_analysis as oa  # noqa: E402

_trapz = getattr(np, "trapezoid", None) or np.trapz
R, T, SPOT = 0.04, 60 / 365, 600.0


# ----- ground truth: mixture of two lognormals (martingale-consistent) -------
def mixture(F, w=(0.8, 0.2), sig=(0.15, 0.32), shift=(0.0, -0.10)):
    """Components with forwards F_i, mixture mean = F. Returns pdf, cdf, call-price functions."""
    f2 = F * (1 + shift[1])
    f1 = (F - w[1] * f2) / w[0]
    Fs = (f1, f2)

    def pdf(K):
        K = np.asarray(K, float)
        out = 0
        for wi, Fi, si in zip(w, Fs, sig):
            mu = np.log(Fi) - 0.5 * si ** 2 * T
            s = si * np.sqrt(T)
            out = out + wi * norm.pdf((np.log(K) - mu) / s) / (K * s)
        return out

    def cdf(K):
        K = np.asarray(K, float)
        out = 0
        for wi, Fi, si in zip(w, Fs, sig):
            mu = np.log(Fi) - 0.5 * si ** 2 * T
            out = out + wi * norm.cdf((np.log(K) - mu) / (si * np.sqrt(T)))
        return out

    def call(K):
        return sum(wi * oa.bs_price(Fi, K, T, si, R, "c") for wi, Fi, si in zip(w, Fs, sig))

    def put(K):
        return call(K) - np.exp(-R * T) * (F - np.asarray(K, float))

    return pdf, cdf, call, put


def make_chain(call_fn, put_fn, F, tick=0.0, spread=0.0, oi=500):
    ks = np.concatenate([np.arange(round(F * 0.70), round(F * 0.90), 5.0),
                         np.arange(round(F * 0.90), round(F * 1.10) + 1, 1.0),
                         np.arange(round(F * 1.10), round(F * 1.30) + 1, 5.0)])
    ks = np.unique(ks)
    out = []
    for fn in (call_fn, put_fn):
        px = np.maximum(fn(ks), 0.0)
        if tick:
            px = np.round(px / tick) * tick
        bid = np.maximum(px - spread / 2, 0)
        ask = px + spread / 2
        out.append(pd.DataFrame(dict(strike=ks, bid=bid, ask=ask, lastPrice=px,
                                     volume=100.0, openInterest=float(oi))))
    return out


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    return bool(cond)


def main():
    ok = True
    F_true = SPOT * np.exp(R * T)

    # 1. implied vol round trip -------------------------------------------------
    for cp in ("c", "p"):
        for K in (480, 600, 720):
            px = float(oa.bs_price(F_true, K, T, 0.23, R, cp))
            iv = oa.implied_vol(px, F_true, K, T, R, cp)
            ok &= check(f"IV round trip {cp} K={K}", abs(iv - 0.23) < 1e-6, f"iv={iv:.8f}")
    ok &= check("IV rejects price below intrinsic",
                np.isnan(oa.implied_vol(1.0, F_true, 500, T, R, "c")))

    # 2. flat vol -> density must equal the lognormal ---------------------------
    flat_call = lambda K: oa.bs_price(F_true, K, T, 0.20, R, "c")
    flat_put = lambda K: oa.bs_price(F_true, K, T, 0.20, R, "p")
    calls, puts = make_chain(flat_call, flat_put, F_true)
    s, sm, de, oi = oa.analyze_expiry(calls, puts, SPOT, T, R)
    ln_pdf, _ = oa.lognormal_benchmark(de.strike.values, F_true, T, 0.20)
    l1 = _trapz(np.abs(de.density.values - ln_pdf), de.strike.values)
    ok &= check("forward recovered from put-call parity", abs(s["forward"] / F_true - 1) < 1e-4,
                f"F={s['forward']:.3f} vs {F_true:.3f}")
    ok &= check("flat vol: ATM IV = 20%", abs(s["atm_iv"] - 0.20) < 1e-3, f"{s['atm_iv']:.5f}")
    ok &= check("flat vol: density matches lognormal (L1 < 0.01)", l1 < 0.01, f"L1={l1:.5f}")
    ok &= check("flat vol: skew/kurtosis ~ lognormal", abs(s["rn_skew"] - s["ln_skew"]) < 0.02
                and abs(s["rn_exkurt"] - s["ln_exkurt"]) < 0.05,
                f"skew {s['rn_skew']:.3f} vs {s['ln_skew']:.3f}")

    # 3. skewed truth, noise-free, then tick-rounded + spread --------------------
    pdf_t, cdf_t, call_t, put_t = mixture(F_true)
    for label, tick, spread, tol_l1, tol_p in (("noise-free", 0.0, 0.0, 0.03, 0.01),
                                               ("tick 0.01 + spread 0.02", 0.01, 0.02, 0.06, 0.02)):
        calls, puts = make_chain(call_t, put_t, F_true, tick=tick, spread=spread)
        s, sm, de, oi = oa.analyze_expiry(calls, puts, SPOT, T, R)
        K = de.strike.values
        l1 = _trapz(np.abs(de.density.values - pdf_t(K)), K) / _trapz(pdf_t(K), K)
        ok &= check(f"[{label}] density vs truth: relative L1 < {tol_l1}", l1 < tol_l1, f"L1={l1:.4f}")
        ok &= check(f"[{label}] density mass in range ~ true mass", abs(s["density_mass"] - (cdf_t(K[-1]) - cdf_t(K[0]))) < 0.02,
                    f"{s['density_mass']:.4f} vs {cdf_t(K[-1]) - cdf_t(K[0]):.4f}")
        for pct in (5, 10):
            lvl = SPOT * (1 - pct / 100)
            err = abs(s[f"p_down_{pct}_rn"] - float(cdf_t(lvl)))
            ok &= check(f"[{label}] P(down {pct}%) vs truth, abs err < {tol_p}", err < tol_p,
                        f"{s[f'p_down_{pct}_rn']:.4f} vs {float(cdf_t(lvl)):.4f}")
        ok &= check(f"[{label}] negative-density share small (< 5%)", s["density_neg_share"] < 0.05,
                    f"{s['density_neg_share']:.4f}")
        ok &= check(f"[{label}] skew is negative and below lognormal", s["rn_skew"] < s["ln_skew"] - 0.1,
                    f"{s['rn_skew']:.3f} vs {s['ln_skew']:.3f}")
        ok &= check(f"[{label}] downside prob > lognormal (fat left tail)", s["p_down_10_rn"] > s["p_down_10_ln"],
                    f"{s['p_down_10_rn']:.4f} vs {s['p_down_10_ln']:.4f}")


    # 3b. noisy quotes (multiplicative 2% noise, coarse tick, wide spread): shape must stay usable
    errs = []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        calls, puts = make_chain(call_t, put_t, F_true, tick=0.05, spread=0.10)
        for d in (calls, puts):
            n = 1 + rng.normal(0, 0.02, len(d))
            for col in ("bid", "ask", "lastPrice"):
                d[col] = d[col] * n
        s, sm, de, oi = oa.analyze_expiry(calls, puts, SPOT, T, R)
        K = de.strike.values
        errs.append((_trapz(np.abs(de.density.values - pdf_t(K)), K) / _trapz(pdf_t(K), K),
                     abs(s["p_down_10_rn"] - float(cdf_t(SPOT * 0.9)))))
    e = np.array(errs)
    ok &= check("[2% quote noise] median relative L1 < 0.08", np.median(e[:, 0]) < 0.08, f"{np.median(e[:, 0]):.3f}")
    ok &= check("[2% quote noise] worst-case P(down 10%) error < 0.02", e[:, 1].max() < 0.02, f"{e[:, 1].max():.4f}")

    # 4. open-interest statistics on a hand-checkable chain ----------------------
    ks = np.array([90., 100., 110.])
    c = pd.DataFrame(dict(strike=ks, openInterest=[10., 20., 70.], volume=[1., 2., 3.]))
    p = pd.DataFrame(dict(strike=ks, openInterest=[70., 20., 10.], volume=[3., 2., 1.]))
    t = oa.oi_table(c, p)
    # pain at 100: calls 10*10=100 ; puts 10*10=100 -> 200. at 90: calls 20*10+70*20=1600 ; at 110: puts 20*10+70*20=1600
    ok &= check("max pain on symmetric chain = 100", oa.max_pain(t) == 100.0, f"{oa.max_pain(t)}")
    ok &= check("totals", t.call_oi.sum() == 100 and t.put_oi.sum() == 100)

    # 5. degenerate input does not crash -----------------------------------------
    res = oa.analyze_expiry(calls.iloc[:4], puts.iloc[:4], SPOT, T, R)
    ok &= check("too few strikes -> clean failure, no exception", res[0] is None, str(res[1]))

    # 6. expiry picker ---------------------------------------------------------------
    import datetime as dt
    ex = ["2026-10-09", "2026-10-16", "2026-11-06", "2026-12-18", "2027-01-15"]
    got = oa.pick_expiries(ex, dt.date(2026, 10, 5), [30, 60, 90])
    ok &= check("expiry picker returns distinct expiries", len({g[0] for g in got}) == len(got), str(got))

    print("\nALL PASSED" if ok else "\nSOME TESTS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
