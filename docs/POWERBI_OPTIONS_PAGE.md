# Power BI – "Options" page (6th page)

All 4 CSVs share the keys `as_of, ticker, expiry, dte, target_dte`.
- fact_options_summary  : one row per (as_of, ticker, target_dte) – **history, appended each run**
- fact_options_smile    : strike, iv, side, mid, oi, volume, iv_smooth, moneyness (snapshot)
- fact_options_density  : strike, moneyness, density, cdf, density_lognormal, cdf_lognormal (snapshot)
- fact_options_oi       : strike, call_oi, put_oi, call_volume, put_volume (snapshot)

## 1. Load (Get data > Blank query > Advanced editor), one per file
```
let
  Source = Csv.Document(Web.Contents("https://raw.githubusercontent.com/lollo1008/portfolio-dashboard/main/data/powerbi/fact_options_summary.csv"),
           [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]),
  Hdr = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
  Typed = Table.TransformColumnTypes(Hdr, {{"as_of", type date}}, "en-US"),
  Num = Table.TransformColumnTypes(Typed,
          List.Transform(List.RemoveItems(Table.ColumnNames(Typed), {"as_of","ticker","expiry","forward_source"}),
          each {_, type number}), "en-US")
in Num
```
Repeat for smile/density/oi (change file name; skip the `Num` step's exclusion list: keep as_of, ticker, expiry, side, price_source as text/date).
Locale "en-US" avoids the decimal-comma problem you had on the other CSVs.

## 2. Relationships
Create `dim_option_chain` = DISTINCT(SELECTCOLUMNS(summary, "as_of", ..., "ticker", ..., "target_dte", ...)) and relate it 1→* to smile, density, oi on (as_of, ticker, target_dte) – or simpler: use the page slicers on each table (same column names, "Edit interactions" off not needed).

## 3. Measures
```
Latest As Of = CALCULATE(MAX(fact_options_summary[as_of]), ALL(fact_options_summary))
-- apply to snapshot tables so only the latest run shows:
Is Latest = IF(MAX(fact_options_density[as_of]) = [Latest As Of], 1, 0)   // page-level filter = 1
ATM IV            = AVERAGE(fact_options_summary[atm_iv])
Expected Move 1σ  = AVERAGE(fact_options_summary[expected_move_1sd_pct]) / 100
Put/Call OI       = AVERAGE(fact_options_summary[put_call_oi_ratio])
Put/Call Volume   = AVERAGE(fact_options_summary[put_call_volume_ratio])
Max Pain          = AVERAGE(fact_options_summary[max_pain])
IV Skew 90-110    = AVERAGE(fact_options_summary[iv_skew_90_110])
P(down 10%) RN    = AVERAGE(fact_options_summary[p_down_10_rn])
P(down 10%) LN    = AVERAGE(fact_options_summary[p_down_10_ln])
Tail Premium      = [P(down 10%) RN] / [P(down 10%) LN] - 1      // "x% more likely than lognormal"
Skew RN           = AVERAGE(fact_options_summary[rn_skew])
Ex Kurtosis RN    = AVERAGE(fact_options_summary[rn_exkurt])
```
Format ratios as 0.00, probabilities as 0.0%, IV as 0.0%.

## 4. Layout (same style: dark cards #1B3A4B, light page, sidebar)
Slicers (top): ticker, target_dte (30/60/90, single select, default 30).
KPI cards row (6): ATM IV · 1σ Expected Move · Put/Call OI · Max Pain · IV Skew 90–110 · P(down 10%) RN vs LN.
Charts:
1. **Implied distribution** – line chart, X=strike (continuous), Y=density and density_lognormal. Title: "Risk-neutral density vs lognormal". Add a constant line at spot (from summary).
2. **Volatility smile** – scatter/line: X=moneyness, Y=iv (markers) + iv_smooth (line).
3. **Open interest by strike** – clustered column: call_oi vs put_oi (calls in green #1B6B52, puts in red #B0492E); filter strikes to ±15% of spot.
4. **Tail probabilities** – clustered bar: P(down 5/10/15%) RN vs lognormal (use a small table with unpivoted columns or 6 measures).
5. **History** (fills as weeks pass) – line: ATM IV and IV Skew by as_of. Empty at first run: normal.
Footnote text box (mandatory): "Risk-neutral probabilities embed a risk premium and are NOT real-world forecasts. Density shape is sensitive to quote noise; tail probabilities are the robust read. Source: Yahoo Finance option chains (snapshot)."
Sidebar: duplicate a button, label "Options", Action → Page navigation → Options; active-page style as on the others.

## 5. Reading it (interview talking points)
- P(down 10%) RN above lognormal = market pays for crash protection (negative skew).
- Put/Call OI > 1 = more hedging positions; walls and max pain = strikes where dealer hedging may pin price near expiry (a heuristic, not a law).
- Expected move = ATM IV·√T.
