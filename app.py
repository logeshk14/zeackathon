"""
FinGuard AI - Streamlit dashboard
Put this file next to finguard.py, then:
    pip install streamlit plotly yfinance pandas numpy scikit-learn
    streamlit run app.py
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import finguard as fg

st.set_page_config(page_title="FinGuard AI", page_icon=":shield:", layout="wide")
TIER_COLORS = {"Low": "#16A34A", "Watch": "#F59E0B", "High": "#DC2626"}


@st.cache_data(ttl=3600, show_spinner="Scoring stocks...")
def load(ticker, synthetic, seed):
    df = fg.synthetic(seed=seed) if synthetic else fg.fetch(ticker)
    return df, fg.score_stock(df)


st.title("FinGuard AI")
st.caption("Early warning system for unusual behaviour in Indian stocks. "
           "Not proof of fraud, not a crash predictor, not investment advice.")

with st.sidebar:
    st.header("Settings")
    synthetic = st.checkbox("Use demo (synthetic) data", value=False)
    if synthetic:
        names = [f"DEMO{i}" for i in range(1, 6)]
    else:
        names = st.multiselect("Stocks", fg.TICKERS, default=fg.TICKERS)
    st.markdown("**Risk tiers**  \nLow 0-40  |  Watch 40-70  |  High 70+")

data = {}
for i, t in enumerate(names):
    try:
        data[t] = load(t, synthetic, i)
    except Exception as e:
        st.sidebar.warning(f"Could not load {t}: {e}")

if not data:
    st.info("No data loaded. Tick 'Use demo data' or check your internet connection.")
    st.stop()

scores = {t: v[1] for t, v in data.items()}
tab1, tab2, tab3 = st.tabs(["Market overview", "Stock detail", "Alert history"])

# ---------- Tab 1: overview ----------
with tab1:
    latest = pd.DataFrame(
        [{"Stock": t, "Risk": r["risk"].iloc[-1], "Tier": str(r["tier"].iloc[-1]),
          "As of": r.index[-1].date()} for t, r in scores.items()]
    ).sort_values("Risk", ascending=False)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Latest risk by stock")
        fig = px.bar(latest, x="Stock", y="Risk", color="Tier",
                     color_discrete_map=TIER_COLORS, range_y=[0, 100])
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Risk heatmap (last 60 days)")
        mat = pd.DataFrame({t: r["risk"] for t, r in scores.items()}).tail(60).T
        heat = px.imshow(mat, aspect="auto", zmin=0, zmax=100,
                         color_continuous_scale=[(0, "#16A34A"), (0.5, "#F59E0B"), (1, "#DC2626")])
        st.plotly_chart(heat, use_container_width=True)
    st.dataframe(latest, hide_index=True, use_container_width=True)

# ---------- Tab 2: stock detail ----------
with tab2:
    t = st.selectbox("Stock", list(data))
    df, r = data[t]
    days = st.slider("Days to show", 60, min(750, len(r)), min(250, len(r)))
    r = r.tail(days)
    d = df.loc[r.index]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25],
                        vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=d.index, open=d["Open"], high=d["High"], low=d["Low"],
                                 close=d["Close"], name="Price"), row=1, col=1)
    hi = r[r["risk"] >= 70]
    fig.add_trace(go.Scatter(x=hi.index, y=d.loc[hi.index, "High"] * 1.02, mode="markers",
                             marker=dict(symbol="triangle-down", size=11, color="#DC2626"),
                             name="High-risk day"), row=1, col=1)
    fig.add_trace(go.Bar(x=d.index, y=d["Volume"], name="Volume", marker_color="#64748B"),
                  row=2, col=1)
    fig.update_layout(height=520, xaxis_rangeslider_visible=False, margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns([1, 2])
    day = st.selectbox("Inspect date", list(r.index[::-1]), format_func=lambda x: x.date().isoformat())
    row = r.loc[day]
    with left:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=float(row["risk"]), title={"text": "Risk score"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#0B1F3A"},
                   "steps": [{"range": [0, 40], "color": "#BBF7D0"},
                             {"range": [40, 70], "color": "#FDE68A"},
                             {"range": [70, 100], "color": "#FCA5A5"}]}))
        gauge.update_layout(height=260, margin=dict(t=40, b=10))
        st.plotly_chart(gauge, use_container_width=True)
    with right:
        st.subheader(f"Why flagged: {row['tier']}")
        st.write(fg.explain(row))
        m1, m2, m3 = st.columns(3)
        m1.metric("Statistical score", f"{row['stat_score']:.0f}")
        m2.metric("ML score", f"{row['ml_score']:.0f}")
        m3.metric("Volume vs average", f"{row['vol_ratio']:.1f}x")
        zs = row[fg.Z_FEATURES].rename(index=fg.LABELS)
        bar = px.bar(x=zs.values, y=zs.index, orientation="h",
                     labels={"x": "Sigma from normal", "y": ""})
        bar.update_layout(height=220, margin=dict(t=10, b=10))
        st.plotly_chart(bar, use_container_width=True)

# ---------- Tab 3: alert history ----------
with tab3:
    allr = pd.concat(scores, names=["ticker", "date"]).reset_index()
    allr["tier"] = allr["tier"].astype(str)
    f1, f2 = st.columns(2)
    tiers = f1.multiselect("Tier", ["High", "Watch", "Low"], default=["High"])
    picks = f2.multiselect("Stock", list(data), default=list(data))
    view = allr[allr["tier"].isin(tiers) & allr["ticker"].isin(picks)]
    view = view.sort_values("risk", ascending=False).head(200).copy()
    if view.empty:
        st.info("No alerts match these filters.")
    else:
        view["why"] = view.apply(fg.explain, axis=1)
        view["date"] = view["date"].dt.date
        st.dataframe(view[["ticker", "date", "risk", "tier", "why"]],
                     hide_index=True, use_container_width=True)
        st.caption(f"High-tier alert rate: {(allr['tier'] == 'High').mean():.2%} of stock-days")
