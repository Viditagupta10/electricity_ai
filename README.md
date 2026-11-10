# ⚡ AI-Powered Smart Electricity Consumption & Optimization System

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
Dashboard browser me `http://localhost:8501` par dikhega; model training ke dauran spinner dikhai dega.
Bina kisi file ke demo data se chalega. Apna data: sidebar -> Upload CSV
(columns: `date`, `kwh`, optional `temp`) ya UCI Household Power Consumption file.

## Features
- Daily / weekly / monthly analysis, highest month, heatmap
- Next-month units forecast (Linear Reg vs Random Forest vs Gradient Boosting, time-based split)
- Slab-wise bill prediction + budget tracker
- Appliance-wise estimate + what-if simulator
- Isolation Forest anomaly detection with reasons
- SQLite storage (usage.db), Streamlit dashboard

## Files
- `core.py` - saara data-science logic (testable)
- `app.py`  - Streamlit UI
