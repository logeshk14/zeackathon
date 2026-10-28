"""
FinGuard AI - back-test on real market events (slide 9)
Run next to finguard.py:
    python backtest.py              (live data via yfinance)
    python backtest.py --synthetic  (offline test)
Outputs a table and backtest_results.csv.

Fairness: the Isolation Forest is trained ONLY on data before each event's
look-back window, so it never sees the event it is being tested on.
Verify event dates yourself before presenting.
"""
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import finguard as fg

HIGH = 70            # risk threshold for a High alert
LOOKBACK_DAYS = 30   # calendar days before the event to search
AFTER_DAYS = 5       # calendar days after the event still counted as detection

EVENTS = [
    # name, ticker, event date (crash / announcement day)
    ("COVID crash", "^NSEI", "2020-03-23"),
    ("Yes Bank moratorium", "YESBANK.NS", "2020-03-05"),
    ("Adani-Hindenburg report", "ADANIENT.NS", "2023-01-24"),
]
ALERT_RATE_TICKERS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"]


def fetch_range(ticker, start="2017-01-01"):
    import yfinance as yf
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def score_as_of(df, cutoff):
    """Score every day, with the ML model fitted only on data before cutoff."""
    f = fg.make_features(df)
    cols = fg.Z_FEATURES + ["vol_ratio", "volatility_chg"]
    X = f[cols].clip(-10, 10)
    train = X[X.index < cutoff]
    model = IsolationForest(n_estimators=300, contamination=0.02, random_state=42).fit(train)
    raw = -model.score_samples(X)
    raw_train = raw[X.index < cutoff]
    lo, hi = np.percentile(raw_train, 50), np.percentile(raw_train, 99.5)
    out = f.copy()
    out["stat_score"] = fg.statistical_score(f)
    out["ml_score"] = ((raw - lo) / (hi - lo)).clip(0, 1) * 100
    out["risk"] = 0.5 * out["stat_score"] + 0.5 * out["ml_score"]
    return out


def first_alert_offset(series, event_date):
    """Trading-day offset of the first score >= HIGH near the event.
    Negative = before the event, 0 = on the day, None = no alert."""
    event_date = pd.Timestamp(event_date)
    idx = series.index
    pos_event = idx.searchsorted(event_date)
    if pos_event >= len(idx):
        return None
    win = series[(idx >= event_date - pd.Timedelta(days=LOOKBACK_DAYS)) &
                 (idx <= event_date + pd.Timedelta(days=AFTER_DAYS))]
    hits = win[win >= HIGH]
    if hits.empty:
        return None
    return int(idx.searchsorted(hits.index[0]) - pos_event)


def fmt(off):
    if off is None:
        return "no alert"
    if off < 0:
        return f"{abs(off)} days before"
    return "on event day" if off == 0 else f"{off} days after"


def alert_rates(use_synth):
    blended, baseline = [], []
    names = ["SYNTH"] if use_synth else ALERT_RATE_TICKERS
    for t in names:
        try:
            df = fg.synthetic() if use_synth else fetch_range(t, "2021-01-01")
            r = fg.score_stock(df)
            blended.append((r["risk"] >= HIGH).mean())
            baseline.append((r["stat_score"] >= HIGH).mean())
        except Exception as e:
            print(f"Alert-rate skip {t}: {e}")
    return (np.mean(blended) if blended else np.nan,
            np.mean(baseline) if baseline else np.nan)


def main():
    use_synth = "--synthetic" in sys.argv
    events = EVENTS
    if use_synth:
        d = fg.synthetic()
        events = [("Planted shock A", "SYNTH", str(d.index[400].date())),
                  ("Planted shock B", "SYNTH", str(d.index[800].date()))]
    rows = []
    for name, ticker, ev in events:
        try:
            df = fg.synthetic() if use_synth else fetch_range(ticker)
            cutoff = pd.Timestamp(ev) - pd.Timedelta(days=LOOKBACK_DAYS + 60)
            sc = score_as_of(df, cutoff)
            blended = first_alert_offset(sc["risk"], ev)
            baseline = first_alert_offset(sc["stat_score"], ev)
            peak = sc.loc[(sc.index >= pd.Timestamp(ev) - pd.Timedelta(days=LOOKBACK_DAYS)) &
                          (sc.index <= pd.Timestamp(ev) + pd.Timedelta(days=AFTER_DAYS)), "risk"].max()
            rows.append({"event": name, "ticker": ticker, "event_date": ev,
                         "FinGuard_first_alert": fmt(blended),
                         "zscore_baseline_first_alert": fmt(baseline),
                         "peak_risk": round(float(peak), 1)})
        except Exception as e:
            print(f"Skipping {name}: {e}")

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n=== Event back-test ===")
    print(res.to_string(index=False))
    res.to_csv("backtest_results.csv", index=False)

    fin, base = alert_rates(use_synth)
    print("\n=== Alert rate (share of stock-days at High) ===")
    print(f"FinGuard blended : {fin:.2%}")
    print(f"z-score baseline : {base:.2%}")
    print("\nSaved backtest_results.csv. Copy these numbers into slide 9.")


if __name__ == "__main__":
    main()
