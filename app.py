"""
app.py - Streamlit dashboard
Run:  streamlit run app.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

import core

st.set_page_config(page_title="Smart Electricity Optimizer", page_icon="⚡", layout="wide")
sns.set_theme(style="whitegrid")


# ------------------------------------------------------------------ sidebar
st.sidebar.title("⚡ Settings")
src = st.sidebar.radio("Data source", ["Demo data (synthetic)", "Upload CSV"])
df = None
if src == "Upload CSV":
    up = st.sidebar.file_uploader("CSV (date, kwh, [temp]) ya UCI household file", type=["csv", "txt"])
    st.sidebar.caption("Daily / hourly / minute data - sab chalega (auto daily me convert hoga).")
    if up is not None:
        try:
            df = core.load_csv(up)
        except Exception as e:
            st.sidebar.error(str(e))
if df is None:
    if src == "Upload CSV":
        st.sidebar.info("CSV upload hone tak demo data dikha raha hoon.")
    df = core.generate_demo_data()

st.sidebar.subheader("Tariff (apne DISCOM ke hisaab se)")
limits = [st.sidebar.number_input(f"Slab {i+1} upto (units)", value=v, step=50)
          for i, v in enumerate([100, 200, 400])]
rates = [st.sidebar.number_input(f"Slab {i+1} rate (Rs/unit)", value=v, step=0.5)
         for i, v in enumerate([3.5, 5.0, 7.0, 9.0])]
slabs = list(zip(limits + [float("inf")], rates))
fixed = st.sidebar.number_input("Fixed charge (Rs)", value=100.0, step=10.0)
tax = st.sidebar.number_input("Tax / duty (%)", value=5.0, step=0.5)
budget = st.sidebar.number_input("Monthly budget (Rs)", value=1000, step=100)

# save history in SQLite (SQL concept)
core.save_to_sqlite(df, "usage.db")

st.title("⚡ AI-Powered Smart Electricity Consumption & Optimization")
st.caption(f"Data: {df['date'].min():%d %b %Y} → {df['date'].max():%d %b %Y}  |  "
           f"{len(df)} days")

# ------------------------------------------------------------------ pipeline
@st.cache_data(show_spinner="Models train ho rahe hain...")
def run_pipeline(data: pd.DataFrame):
    res = core.train_and_compare(data)
    end = data["date"].max() + pd.offsets.MonthEnd(2)
    fc = core.forecast_until(data, res["best_model"], res["features"], res["has_temp"], end)
    an = core.detect_anomalies(data)
    return res, fc, an


res, fc, an = run_pipeline(df)
proj = core.bill_projection(df, fc, res["best_mape"], budget, slabs, fixed, tax)
monthly = core.monthly_summary(df, slabs, fixed, tax)
full_months = monthly[monthly["days"] >= 25]

st.caption(f"Best model: {res['best_name']} (MAPE {res['best_mape']}%)")

# alert banner (last 7 days)
recent = an[(an["date"] > df["date"].max() - pd.Timedelta(days=7)) & an["is_anomaly"]]
if len(recent):
    st.error(f"🚨 Pichle 7 din me {len(recent)} din abnormal usage mila. Anomalies tab dekho.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Overview", "🔮 Forecast & Bill", "🔌 Appliances & What-if", "🚨 Anomalies", "🧪 Model Report"])

# ------------------------------------------------------------------ TAB 1
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total consumption", f"{df['kwh'].sum():,.0f} kWh")
    c2.metric("Avg / day", f"{df['kwh'].mean():.1f} kWh")
    peak = full_months.loc[full_months["units"].idxmax()]
    c3.metric("Highest month", peak["month"], f"{peak['units']:.0f} units")
    last_full = full_months.iloc[-1]
    c4.metric(f"Bill ({last_full['month']})", f"Rs {last_full['bill']:,.0f}")

    st.subheader("Daily usage")
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.plot(df["date"], df["kwh"], alpha=0.35, label="Daily")
    ax.plot(df["date"], df["kwh"].rolling(7).mean(), color="C1", label="7-day avg")
    ax.set_ylabel("kWh"); ax.legend()
    st.pyplot(fig); plt.close(fig)

    colA, colB = st.columns(2)
    with colA:
        st.subheader("Monthly comparison")
        fig, ax = plt.subplots(figsize=(6, 3.6))
        sns.barplot(data=full_months.tail(12), x="month", y="units", ax=ax, color="C0")
        ax.tick_params(axis="x", rotation=60); ax.set_xlabel("")
        st.pyplot(fig); plt.close(fig)
    with colB:
        st.subheader("Weekly pattern (avg kWh)")
        wd = df.assign(day=df["date"].dt.day_name()).groupby("day")["kwh"].mean()
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        fig, ax = plt.subplots(figsize=(6, 3.6))
        sns.barplot(x=wd.reindex(order).index.str[:3], y=wd.reindex(order).values, ax=ax, color="C2")
        ax.set_ylabel("kWh")
        st.pyplot(fig); plt.close(fig)

    st.subheader("Month × weekday heatmap")
    hm = df.assign(m=df["date"].dt.month, d=df["date"].dt.dayofweek).pivot_table(
        index="m", columns="d", values="kwh", aggfunc="mean")
    hm.columns = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][: len(hm.columns)]
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(hm, cmap="YlOrRd", annot=True, fmt=".1f", ax=ax)
    ax.set_ylabel("Month")
    st.pyplot(fig); plt.close(fig)

    st.subheader("Monthly table")
    st.dataframe(monthly.rename(columns={"units": "Units (kWh)", "bill": "Bill (Rs)",
                                         "avg_daily": "Avg/day", "days": "Days"}), use_container_width=True)

# ------------------------------------------------------------------ TAB 2
with tab2:
    st.subheader(f"Next month ({proj['next_month']}) estimated bill")
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted units", f"{proj['next_units']:.0f} kWh")
    c2.metric("Predicted bill", f"Rs {proj['next_bill']:,.0f}")
    c3.metric("Likely range", f"Rs {proj['next_bill_low']:,.0f} - {proj['next_bill_high']:,.0f}")

    st.subheader(f"Budget tracker - {proj['current_month']}")
    d1, d2, d3 = st.columns(3)
    d1.metric("Units so far", f"{proj['cur_actual']:.0f} kWh")
    d2.metric("Month-end projected bill", f"Rs {proj['cur_projected_bill']:,.0f}")
    d3.metric("Budget", f"Rs {budget:,.0f}", delta=f"{budget - proj['cur_projected_bill']:,.0f}")
    st.progress(min(proj["cur_projected_bill"] / max(budget, 1), 1.0))
    if proj["over_budget"]:
        st.warning(f"Budget cross hone wala hai. ~{proj['units_to_cut']} units kam karne honge "
                   f"(budget = max {proj['budget_units']} units).")
    else:
        st.success("Is rate par aap budget ke andar rahoge. 👍")

    st.subheader("Forecast (next ~2 months)")
    fig, ax = plt.subplots(figsize=(11, 3.4))
    hist = df.tail(60)
    ax.plot(hist["date"], hist["kwh"], label="Actual")
    ax.plot(fc["date"], fc["kwh"], color="C3", label="Forecast")
    ax.set_ylabel("kWh"); ax.legend()
    st.pyplot(fig); plt.close(fig)

    st.subheader("Kaunse factors usage badhate hain?")
    imp = res["importance"].head(8)[::-1]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.barh(imp.index, imp.values, color="C4")
    ax.set_xlabel("Importance")
    st.pyplot(fig); plt.close(fig)
    if res["has_temp"] is False:
        st.info("Temperature column nahi hai - CSV me 'temp' add karoge to accuracy badhegi.")

# ------------------------------------------------------------------ TAB 3
with tab3:
    st.subheader("Appliance-wise estimate")
    st.caption("Formula: Watts × Qty × Hours/day × 30 ÷ 1000. Table edit karo - apne ghar ke numbers daalo.")
    if "apps" not in st.session_state:
        st.session_state["apps"] = core.default_appliances()
    edited = st.data_editor(st.session_state["apps"], num_rows="dynamic", use_container_width=True, key="ed")
    edited = edited.dropna().reset_index(drop=True)
    calibrate = st.checkbox("Actual units se calibrate karo (recommended)", value=True)
    atab, base = core.appliance_estimate(
        edited, proj["cur_projected_units"] if calibrate else None) if len(edited) else (None, None)

    if atab is not None:
        top = atab.iloc[0]
        st.success(f"Sabse zyada bijli: **{top['Appliance']}** ({top['Share %']}%)")
        cA, cB = st.columns([1, 1])
        with cA:
            st.dataframe(atab[["Appliance", base, "Share %"]], use_container_width=True)
        with cB:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.pie(atab[base], labels=atab["Appliance"], autopct="%1.0f%%", startangle=90)
            st.pyplot(fig); plt.close(fig)

        st.subheader("🎛️ What-if simulator")
        w1, w2 = st.columns(2)
        pick = w1.selectbox("Appliance", atab["Appliance"])
        cut = w2.slider("Roz kitne ghante kam?", 0.0, 8.0, 1.0, 0.5)
        r = atab[atab["Appliance"] == pick].iloc[0]
        scale = r[base] / r["Est. kWh/month"] if r["Est. kWh/month"] else 1
        saved = min(r["Watts"] * r["Qty"] * cut * 30 / 1000 * scale, r[base])
        wi = core.what_if(proj["cur_projected_units"], saved, slabs, fixed, tax)
        m1, m2, m3 = st.columns(3)
        m1.metric("Units saved", f"{wi['saved_units']} kWh")
        m2.metric("New bill", f"Rs {wi['after']:,.0f}", delta=f"-Rs {wi['saved_rs']:,.0f}", delta_color="inverse")
        m3.metric("Yearly saving", f"Rs {wi['saved_rs'] * 12:,.0f}")

        st.subheader("💡 Smart recommendations")
        for t in core.recommendations(atab, base, proj, slabs, fixed, tax):
            st.markdown(f"- {t}")

# ------------------------------------------------------------------ TAB 4
with tab4:
    st.subheader("Abnormal usage detection (Isolation Forest)")
    st.caption("Model pehle weather/calendar se 'expected' usage nikalta hai, phir jo din expected se bahut upar ho use flag karta hai.")
    flagged = an[an["is_anomaly"]]
    st.metric("Flagged days", len(flagged))
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(an["date"], an["kwh"], alpha=0.5, label="Usage")
    ax.plot(an["date"], an["expected"], color="C2", alpha=0.6, label="Expected")
    ax.scatter(flagged["date"], flagged["kwh"], color="red", zorder=5, label="Anomaly")
    ax.legend(); ax.set_ylabel("kWh")
    st.pyplot(fig); plt.close(fig)
    show = flagged[["date", "kwh", "expected", "excess_pct", "reason"]].sort_values("date", ascending=False)
    show = show.assign(date=show["date"].dt.date, expected=show["expected"].round(1),
                       excess_pct=show["excess_pct"].round(0))
    st.dataframe(show, use_container_width=True)

# ------------------------------------------------------------------ TAB 5
with tab5:
    st.subheader("Model comparison (time-based split: pehle 80% train, last 20% test)")
    st.dataframe(res["results"], use_container_width=True)
    st.caption("Baseline = 'pichle hafte same din jitna use hua tha'. Model ko baseline se behtar hona chahiye.")
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.plot(res["test_dates"], res["test_actual"], label="Actual (test)")
    ax.plot(res["test_dates"], res["test_pred"], label=f"Predicted ({res['best_name']})", alpha=0.85)
    ax.legend(); ax.set_ylabel("kWh")
    st.pyplot(fig); plt.close(fig)
