"""
FinGuard AI - starter pipeline
Install:  pip install yfinance pandas numpy scikit-learn
Run:      python finguard.py                (live data via yfinance)
          python finguard.py --synthetic    (offline test data)
Flags unusual market behaviour only. It does not prove fraud or predict crashes.
"""
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

TICKERS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ADANIENT.NS", "YESBANK.NS"]
WINDOW = 20            # rolling baseline (trading days)
Z_FEATURES = ["ret_z", "gap_z", "range_z", "vol_z"]
LABELS = {
    "ret_z": "price move",
    "gap_z": "overnight gap",
    "range_z": "intraday range",
    "vol_z": "trading volume",
}


def fetch(ticker, years=5):
    import yfinance as yf
    df = yf.download(ticker, period=f"{years}y", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def synthetic(n=1250, seed=0):
    """Random-walk data with a few planted shocks, for offline testing."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, n)))
    vol = rng.lognormal(13, 0.25, n)
    for i in (400, 800, n - 3):
        close[i:] *= 1.09
        vol[i] *= 6
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * (1 - abs(rng.normal(0, 0.004, n)))
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close, "Volume": vol}, index=idx)


def zscore(s, window=WINDOW):
    """Z-score against the PREVIOUS window (shift avoids look-ahead)."""
    mean = s.rolling(window).mean().shift(1)
    std = s.rolling(window).std().shift(1)
    return (s - mean) / std.replace(0, np.nan)


def make_features(df):
    f = pd.DataFrame(index=df.index)
    ret = df["Close"].pct_change()
    f["ret"] = ret
    f["ret_z"] = zscore(ret)
    f["gap_z"] = zscore(df["Open"] / df["Close"].shift(1) - 1)
    f["range_z"] = zscore((df["High"] - df["Low"]) / df["Close"])
    f["vol_z"] = zscore(np.log1p(df["Volume"]))
    f["vol_ratio"] = df["Volume"] / df["Volume"].rolling(WINDOW).mean().shift(1)
    f["volatility"] = ret.rolling(WINDOW).std()
    f["volatility_chg"] = f["volatility"] / f["volatility"].shift(WINDOW) - 1
    return f.replace([np.inf, -np.inf], np.nan).dropna()


def statistical_score(f):
    """0-100 from the largest absolute z-score (1 sigma -> 0, 5 sigma -> 100)."""
    worst = f[Z_FEATURES].abs().max(axis=1)
    return ((worst - 1) / 4).clip(0, 1) * 100


def ml_score(f, contamination=0.02, seed=42):
    """Isolation Forest anomaly score scaled to 0-100."""
    cols = Z_FEATURES + ["vol_ratio", "volatility_chg"]
    X = f[cols].clip(-10, 10)
    model = IsolationForest(n_estimators=300, contamination=contamination,
                            random_state=seed).fit(X)
    raw = -model.score_samples(X)                 # higher = more unusual
    lo, hi = np.percentile(raw, 50), np.percentile(raw, 99.5)
    return pd.Series(((raw - lo) / (hi - lo)).clip(0, 1) * 100, index=f.index)


def tier(score):
    return pd.cut(score, [-1, 40, 70, 101], labels=["Low", "Watch", "High"])


def score_stock(df):
    f = make_features(df)
    out = f.copy()
    out["stat_score"] = statistical_score(f)
    out["ml_score"] = ml_score(f)
    out["risk"] = (0.5 * out["stat_score"] + 0.5 * out["ml_score"]).round(1)
    out["tier"] = tier(out["risk"])
    return out


def explain(row, k=3):
    """Plain-English reasons: the k largest z-score drivers on that day."""
    drivers = row[Z_FEATURES].abs().sort_values(ascending=False).head(k).index
    parts = []
    for d in drivers:
        direction = "above" if row[d] > 0 else "below"
        parts.append(f"{LABELS[d]} was {abs(row[d]):.1f} sigma {direction} normal")
    text = "; ".join(parts)
    if row["vol_ratio"] > 2:
        text += f". Volume was {row['vol_ratio']:.1f}x its {WINDOW}-day average"
    return text


def main():
    use_synth = "--synthetic" in sys.argv
    results = {}
    for t in (["SYNTH"] if use_synth else TICKERS):
        try:
            df = synthetic() if use_synth else fetch(t)
            results[t] = score_stock(df)
        except Exception as e:
            print(f"Skipping {t}: {e}")

    print("\n=== Latest risk by stock ===")
    for t, r in results.items():
        last = r.iloc[-1]
        print(f"{t:<12} risk={last['risk']:5.1f}  {last['tier']}")

    print("\n=== Top 5 alerts (all stocks, all dates) ===")
    allr = pd.concat({t: r for t, r in results.items()}, names=["ticker", "date"])
    for (t, d), row in allr.sort_values("risk", ascending=False).head(5).iterrows():
        print(f"{t} {d.date()}  risk={row['risk']:.1f} ({row['tier']})")
        print(f"   why: {explain(row)}")

    print("\nAlert rate (High tier):",
          f"{(allr['tier'] == 'High').mean():.2%} of stock-days")


if __name__ == "__main__":
    main()
