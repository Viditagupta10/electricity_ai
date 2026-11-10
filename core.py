"""
core.py - Smart Electricity Consumption & Optimization System
All data-science logic lives here (no Streamlit code), so it is easy to test.
"""
import sqlite3
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------
# 1. TARIFF (slab-wise billing)
# --------------------------------------------------------------------------
# (upper_limit_of_slab_in_units, rate_per_unit)  -> edit as per your DISCOM
DEFAULT_SLABS = [(100, 3.5), (200, 5.0), (400, 7.0), (float("inf"), 9.0)]
DEFAULT_FIXED = 100.0   # fixed monthly charge (Rs)
DEFAULT_TAX = 5.0       # electricity duty / tax (%)


def calc_bill(units, slabs=DEFAULT_SLABS, fixed=DEFAULT_FIXED, tax_pct=DEFAULT_TAX):
    """Slab-wise bill: each slab's units are charged at that slab's rate."""
    units = max(float(units), 0.0)
    energy, prev = 0.0, 0.0
    for limit, rate in slabs:
        if units > prev:
            energy += (min(units, limit) - prev) * rate
        prev = limit
    return round((energy + fixed) * (1 + tax_pct / 100), 2)


def units_for_budget(budget, slabs=DEFAULT_SLABS, fixed=DEFAULT_FIXED, tax_pct=DEFAULT_TAX):
    """Inverse of calc_bill: max units you can use to stay within a budget."""
    lo, hi = 0.0, 5000.0
    for _ in range(60):  # binary search
        mid = (lo + hi) / 2
        if calc_bill(mid, slabs, fixed, tax_pct) <= budget:
            lo = mid
        else:
            hi = mid
    return round(lo, 1)


# --------------------------------------------------------------------------
# 2. DATA (synthetic demo / CSV loader / SQLite)
# --------------------------------------------------------------------------
def generate_demo_data(days=730, seed=42, end_date=None):
    """Realistic daily household data (temperature-driven AC, weekends, spikes)."""
    rng = np.random.default_rng(seed)
    end = pd.Timestamp(end_date or pd.Timestamp.today().normalize())
    dates = pd.date_range(end=end, periods=days, freq="D")
    doy = dates.dayofyear.values
    temp = 25 + 11 * np.sin(2 * np.pi * (doy - 105) / 365) + rng.normal(0, 1.8, days)

    ac = np.clip((temp - 26) * 0.9, 0, 10) * 1.3 * rng.uniform(0.8, 1.15, days)
    fan = 0.075 * 3 * np.clip((temp - 18) * 0.9, 0, 12)
    fridge = 1.3 + 0.03 * (temp - 25) + rng.normal(0, 0.05, days)
    geyser = np.where(temp < 20, 2.0 * rng.uniform(0.3, 0.8, days), 0)
    washing = np.where(rng.random(days) < 0.45, 0.6, 0.0)
    base = 1.6 + rng.normal(0, 0.15, days)  # lights, TV, charger, misc
    weekend = np.asarray(dates.dayofweek >= 5)
    base = base + weekend * 0.8
    kwh = ac + fan + fridge + geyser + washing + base

    # inject abnormal days (faulty appliance / appliance left ON)
    idx = rng.choice(np.arange(60, days), size=max(5, days // 90), replace=False)
    kwh[idx] *= rng.uniform(1.6, 2.1, len(idx))
    return pd.DataFrame({"date": dates, "kwh": kwh.round(2), "temp": temp.round(1)})


def load_csv(file) -> pd.DataFrame:
    """
    Accepts:
      * simple CSV with columns: date, kwh  (optional: temp)
      * hourly / minute data (auto-aggregated to daily)
      * UCI 'Household Power Consumption' file (Date, Time, Global_active_power)
    Returns daily dataframe: date, kwh, [temp]
    """
    raw = pd.read_csv(file, sep=None, engine="python", na_values=["?", "", "NA"])
    raw.columns = [c.strip().lower() for c in raw.columns]

    # --- UCI format ---
    if "global_active_power" in raw.columns:
        ts = pd.to_datetime(raw["date"] + " " + raw["time"], dayfirst=True, errors="coerce")
        kw = pd.to_numeric(raw["global_active_power"], errors="coerce")
        s = pd.Series((kw / 60.0).values, index=ts).dropna()      # kW per minute -> kWh
        daily = s.resample("D").sum().rename("kwh").reset_index()
        daily.columns = ["date", "kwh"]
        return _finalize(daily)

    date_col = next((c for c in raw.columns if c in ("date", "datetime", "timestamp", "time", "day")), None)
    kwh_col = next((c for c in raw.columns if c in ("kwh", "units", "consumption", "energy", "usage", "unit")), None)
    if date_col is None or kwh_col is None:
        raise ValueError("CSV me 'date' aur 'kwh' (ya units/consumption) column hona chahiye.")
    dates = pd.to_datetime(raw[date_col], errors="coerce")           # ISO (YYYY-MM-DD) first
    if dates.isna().mean() > 0.2:                                      # fallback: DD/MM/YYYY
        dates = pd.to_datetime(raw[date_col], errors="coerce", dayfirst=True)
    df = pd.DataFrame({"date": dates, "kwh": pd.to_numeric(raw[kwh_col], errors="coerce")})
    temp_col = next((c for c in raw.columns if c in ("temp", "temperature")), None)
    if temp_col:
        df["temp"] = pd.to_numeric(raw[temp_col], errors="coerce")
    df = df.dropna(subset=["date", "kwh"])
    agg = {"kwh": "sum"}
    if "temp" in df:
        agg["temp"] = "mean"
    daily = df.set_index("date").resample("D").agg(agg).reset_index()
    return _finalize(daily)


def _finalize(daily):
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["kwh"] = daily["kwh"].replace(0, np.nan).interpolate(limit_direction="both")
    if "temp" in daily:
        daily["temp"] = daily["temp"].interpolate(limit_direction="both")
    if len(daily) < 90:
        raise ValueError("Kam se kam 90 din ka data chahiye (better: 1+ saal).")
    return daily


def save_to_sqlite(df, db_path="usage.db", table="daily_usage"):
    with sqlite3.connect(db_path) as con:
        df.to_sql(table, con, if_exists="replace", index=False)


def load_from_sqlite(db_path="usage.db", table="daily_usage"):
    with sqlite3.connect(db_path) as con:
        return pd.read_sql(f"SELECT * FROM {table} ORDER BY date", con, parse_dates=["date"])


def monthly_summary(df, slabs=DEFAULT_SLABS, fixed=DEFAULT_FIXED, tax=DEFAULT_TAX):
    m = df.assign(month=df["date"].dt.to_period("M")).groupby("month").agg(
        units=("kwh", "sum"), days=("kwh", "count"), avg_daily=("kwh", "mean"))
    m["bill"] = m["units"].apply(lambda u: calc_bill(u, slabs, fixed, tax))
    m = m.round(1).reset_index()
    m["month"] = m["month"].astype(str)
    return m


# --------------------------------------------------------------------------
# 3. FEATURE ENGINEERING + FORECASTING (time-series split, leakage-free)
# --------------------------------------------------------------------------
def _feat_row(window, date, temp):
    """window = last 30 daily values (oldest -> newest). Same fn for train & forecast."""
    w = np.asarray(window, dtype=float)
    doy = date.dayofyear
    row = {
        "lag1": w[-1], "lag2": w[-2], "lag7": w[-7], "lag14": w[-14],
        "roll7": w[-7:].mean(), "roll30": w[-30:].mean(),
        "dow": date.dayofweek, "month": date.month,
        "is_weekend": int(date.dayofweek >= 5),
        "doy_sin": np.sin(2 * np.pi * doy / 365.25), "doy_cos": np.cos(2 * np.pi * doy / 365.25),
    }
    if temp is not None:
        row["temp"] = float(temp)
    return row


def build_features(df):
    has_temp = "temp" in df.columns and df["temp"].notna().all()
    vals = df["kwh"].values
    rows, ys, dts = [], [], []
    for i in range(30, len(df)):
        d = df["date"].iloc[i]
        rows.append(_feat_row(vals[i - 30:i], d, df["temp"].iloc[i] if has_temp else None))
        ys.append(vals[i])
        dts.append(d)
    X = pd.DataFrame(rows)
    return X, pd.Series(ys, name="kwh"), pd.Series(dts, name="date"), has_temp


def _metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-6, None))) * 100
    return round(mae, 3), round(rmse, 3), round(mape, 2)


def _models():
    return {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Random Forest": RandomForestRegressor(n_estimators=250, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=250, learning_rate=0.05, max_depth=3, random_state=42),
    }


def train_and_compare(df, test_frac=0.2):
    """Chronological split -> compare baseline + 3 models -> refit best on all data."""
    X, y, dts, has_temp = build_features(df)
    split = int(len(X) * (1 - test_frac))
    Xtr, Xte, ytr, yte = X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]

    rows = []
    mae, rmse, mape = _metrics(yte, Xte["lag7"])
    rows.append({"Model": "Baseline (same day last week)", "MAE": mae, "RMSE": rmse, "MAPE %": mape})
    test_preds = {}
    for name, mdl in _models().items():
        mdl.fit(Xtr, ytr)
        pred = mdl.predict(Xte)
        test_preds[name] = pred
        mae, rmse, mape = _metrics(yte, pred)
        rows.append({"Model": name, "MAE": mae, "RMSE": rmse, "MAPE %": mape})
    results = pd.DataFrame(rows).sort_values("MAPE %").reset_index(drop=True)

    best_name = results[results["Model"] != "Baseline (same day last week)"].iloc[0]["Model"]
    best = _models()[best_name].fit(X, y)  # refit on full data for real forecasting
    best_mape = float(results.loc[results["Model"] == best_name, "MAPE %"].iloc[0])

    importance = None
    est = best.steps[-1][1] if hasattr(best, "steps") else best
    if hasattr(est, "feature_importances_"):
        importance = pd.Series(est.feature_importances_, index=X.columns).sort_values(ascending=False)
    else:
        importance = pd.Series(np.abs(est.coef_), index=X.columns).sort_values(ascending=False)
        importance = importance / importance.sum()

    return {
        "results": results, "best_name": best_name, "best_model": best, "best_mape": best_mape,
        "features": list(X.columns), "has_temp": has_temp, "importance": importance,
        "test_dates": dts.iloc[split:].values, "test_actual": yte.values,
        "test_pred": test_preds[best_name],
    }


def forecast_until(df, model, features, has_temp, end_date):
    """Recursive day-by-day forecast from last date in df until end_date."""
    hist = list(df["kwh"].values[-30:])
    last = df["date"].max()
    clim = None
    if has_temp:
        clim = df.groupby(df["date"].dt.dayofyear)["temp"].mean().reindex(range(1, 367))
        clim = clim.interpolate(limit_direction="both")
    out = []
    for d in pd.date_range(last + pd.Timedelta(days=1), pd.Timestamp(end_date), freq="D"):
        t = clim[d.dayofyear] if has_temp else None
        x = pd.DataFrame([_feat_row(hist[-30:], d, t)])[features]
        p = float(max(model.predict(x)[0], 0.0))
        hist.append(p)
        out.append((d, p))
    return pd.DataFrame(out, columns=["date", "kwh"])


def bill_projection(df, fc_df, mape, budget, slabs, fixed, tax):
    """Current-month projection (actual so far + forecast) and next-month estimate."""
    last = df["date"].max()
    cur = last.to_period("M")
    nxt = cur + 1
    cur_actual = df[df["date"].dt.to_period("M") == cur]["kwh"].sum()
    cur_fc = fc_df[fc_df["date"].dt.to_period("M") == cur]["kwh"].sum()
    nxt_units = fc_df[fc_df["date"].dt.to_period("M") == nxt]["kwh"].sum()
    cur_total = cur_actual + cur_fc
    band = mape / 100.0
    res = {
        "current_month": str(cur), "next_month": str(nxt),
        "cur_actual": round(float(cur_actual), 1), "cur_projected_units": round(float(cur_total), 1),
        "cur_projected_bill": calc_bill(cur_total, slabs, fixed, tax),
        "next_units": round(float(nxt_units), 1),
        "next_bill": calc_bill(nxt_units, slabs, fixed, tax),
        "next_bill_low": calc_bill(nxt_units * (1 - band), slabs, fixed, tax),
        "next_bill_high": calc_bill(nxt_units * (1 + band), slabs, fixed, tax),
        "budget": budget, "budget_units": units_for_budget(budget, slabs, fixed, tax),
    }
    res["over_budget"] = res["cur_projected_bill"] > budget
    res["units_to_cut"] = max(0.0, round(res["cur_projected_units"] - res["budget_units"], 1))
    return res


# --------------------------------------------------------------------------
# 4. ANOMALY DETECTION (Isolation Forest on residual vs expected usage)
# --------------------------------------------------------------------------
def detect_anomalies(df, contamination=0.03):
    d = df.copy()
    has_temp = "temp" in d.columns
    feats = pd.DataFrame({
        "dow": d["date"].dt.dayofweek, "month": d["date"].dt.month,
        "doy_sin": np.sin(2 * np.pi * d["date"].dt.dayofyear / 365.25),
        "doy_cos": np.cos(2 * np.pi * d["date"].dt.dayofyear / 365.25),
    })
    if has_temp:
        feats["temp"] = d["temp"].values
    expected = cross_val_predict(
        RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=0, n_jobs=-1),
        feats, d["kwh"], cv=KFold(5, shuffle=True, random_state=0))
    d["expected"] = expected
    d["excess_pct"] = (d["kwh"] - d["expected"]) / d["expected"].clip(lower=0.1) * 100
    iso = IsolationForest(contamination=contamination, random_state=42)
    d["iso"] = iso.fit_predict(d[["excess_pct", "kwh"]])
    d["is_anomaly"] = (d["iso"] == -1) & (d["excess_pct"] > 25)   # only abnormally HIGH usage

    def reason(r):
        if not r["is_anomaly"]:
            return ""
        base = f"{r['kwh']:.1f} kWh vs expected {r['expected']:.1f} (+{r['excess_pct']:.0f}%). "
        if has_temp and r["temp"] >= 32:
            return base + "Garmi zyada thi - AC lagatar/zyada chala hoga."
        if has_temp and r["temp"] <= 15:
            return base + "Thand thi - geyser/heater zyada chala hoga."
        return base + "Weather se explain nahi hota - koi appliance ON reh gaya ya kharab (geyser/AC/pump) check karo."
    d["reason"] = d.apply(reason, axis=1)
    return d


# --------------------------------------------------------------------------
# 5. APPLIANCES, WHAT-IF, RECOMMENDATIONS
# --------------------------------------------------------------------------
def default_appliances():
    return pd.DataFrame([
        {"Appliance": "AC",               "Watts": 1500, "Qty": 1, "Hours/day": 6.0},
        {"Appliance": "Ceiling Fan",      "Watts": 75,   "Qty": 3, "Hours/day": 10.0},
        {"Appliance": "Refrigerator",     "Watts": 150,  "Qty": 1, "Hours/day": 8.0},   # compressor duty cycle
        {"Appliance": "Washing Machine",  "Watts": 500,  "Qty": 1, "Hours/day": 0.7},
        {"Appliance": "LED Lights",       "Watts": 10,   "Qty": 8, "Hours/day": 6.0},
        {"Appliance": "TV",               "Watts": 100,  "Qty": 1, "Hours/day": 4.0},
        {"Appliance": "Geyser",           "Watts": 2000, "Qty": 1, "Hours/day": 0.5},
    ])


def appliance_estimate(app_df, actual_monthly_units=None):
    """kWh/month = Watts x Qty x Hours x 30 / 1000. Optionally calibrate to actual bill units."""
    a = app_df.copy()
    a["Est. kWh/month"] = (a["Watts"] * a["Qty"] * a["Hours/day"] * 30 / 1000).round(1)
    total = a["Est. kWh/month"].sum()
    if actual_monthly_units and total > 0:
        a["Calibrated kWh/month"] = (a["Est. kWh/month"] * actual_monthly_units / total).round(1)
    base = "Calibrated kWh/month" if "Calibrated kWh/month" in a else "Est. kWh/month"
    a["Share %"] = (a[base] / a[base].sum() * 100).round(1)
    return a.sort_values(base, ascending=False).reset_index(drop=True), base


def what_if(monthly_units, saved_units, slabs, fixed, tax):
    before = calc_bill(monthly_units, slabs, fixed, tax)
    after = calc_bill(monthly_units - saved_units, slabs, fixed, tax)
    return {"before": before, "after": after, "saved_rs": round(before - after, 2),
            "saved_units": round(saved_units, 1)}


def recommendations(app_table, base_col, proj, slabs, fixed, tax, top_n=3):
    """Appliance-wise, budget-aware tips with Rs savings from 1 hour/day less usage."""
    tips = []
    monthly_units = proj["cur_projected_units"]
    candidates = app_table[~app_table["Appliance"].str.contains("fridge|refrigerator", case=False)]
    for _, r in candidates.head(top_n).iterrows():
        one_hr_kwh = r["Watts"] * r["Qty"] * 30 / 1000          # saving of 1 hr/day
        scale = r[base_col] / r["Est. kWh/month"] if r["Est. kWh/month"] else 1
        w = what_if(monthly_units, one_hr_kwh * scale, slabs, fixed, tax)
        tips.append(f"**{r['Appliance']}** ({r['Share %']}% of usage): roz 1 ghanta kam chalane se "
                    f"~{w['saved_units']} units / ~Rs {w['saved_rs']:.0f} per month bachega.")
    if proj["over_budget"]:
        tips.insert(0, f"Is month ka projected bill Rs {proj['cur_projected_bill']:.0f} hai, budget Rs {proj['budget']:.0f} "
                       f"se zyada. Budget me rehne ke liye ~{proj['units_to_cut']} units kam karne honge.")
    else:
        tips.insert(0, f"Aap budget me ho: projected Rs {proj['cur_projected_bill']:.0f} vs budget Rs {proj['budget']:.0f}.")
    tips.append("AC ko 24-26 C par rakho aur filter saaf rakho; peak shaam ke ghanton me geyser/washing machine avoid karo.")
    tips.append("Purane tube-light/bulb ko LED se replace karo aur standby par devices ko switch-off karo.")
    return tips
