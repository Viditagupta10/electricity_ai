# Smart Parking Availability & Prediction System

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dashboard will be available in your browser at `http://localhost:8501`. A
loading spinner is displayed while the machine learning models are being
trained; the first run may take approximately 20–30 seconds. The application
can also run without uploading a file by using demo data.

## How to Upload Your Own Data

In the sidebar, select **Upload CSV**.

Required columns:

- `timestamp`
- `lot_id`
- `occupied`

Optional columns:

- `capacity`
- `rain`
- `is_holiday`
- `event`
- `lat`
- `lon`

Each parking lot needs at least 14 days of data.

## Features

- **Live Status:** Shows available spaces in each lot and predicts occupancy
  for the next 1, 2, and 3 hours.
- **Future Planning:** Predicts occupancy for a selected date and time,
  identifies less crowded time slots, and suggests dynamic parking prices.
- **Smart Recommendations:** Recommends a parking lot based on the
  destination and available spaces.
- **Analytics:** Analyzes weekday and weekend patterns, creates occupancy
  heatmaps, explores rain, events, and holidays, and clusters parking lots
  using KMeans.
- **Model Report:** Compares a baseline, Linear Regression, Random Forest,
  and Gradient Boosting with a time-based data split.
- **Database Storage:** Saves the active parking data to the SQLite database
  `parking.db`.

## Project Files

- `core.py`: Data processing, model training, and prediction logic.
- `app.py`: Streamlit dashboard and user interface.
- `parking.db`: SQLite database for parking data.
