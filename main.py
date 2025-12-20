# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import pandas as pd
import joblib
import os

from src.pipeline import preprocess_and_engineer  # نستخدم البايبلاين اللي عملناه

app = FastAPI(title="Driver Risk Scoring API")

# === Load model + feature list (saved from the notebook) ===
MODEL_PATH = os.path.join("models", "rf_is_risky.pkl")
#FEATURES_PATH = os.path.join("models", "features_no_leak.pkl")

rf_model = joblib.load(MODEL_PATH)
MODEL_FEATURES = list(rf_model.feature_names_in_)

#features_no_leak = joblib.load(FEATURES_PATH)


# ---------- Request & Response schemas ----------

class DriverRiskRequest(BaseModel):
    driver_id: int
    date_from: datetime
    date_to: datetime


class DriverRiskResponse(BaseModel):
    driver_id: int
    date_from: datetime
    date_to: datetime
    trips_count: int
    risky_trips: int
    risk_ratio: float
    avg_risk_probability: float


# ---------- Load data (for now: from CSV) ----------

def load_trips_for_driver(driver_id: int, date_from: datetime, date_to: datetime) -> pd.DataFrame:
    """
    For the task submission, we read from data/driver.csv and filter.
    In real production, this would be a SQL query.
    """
    df_all = pd.read_csv("data/driver.csv")
    df_all["date_from"] = pd.to_datetime(df_all["date_from"])
    df_all["date_to"] = pd.to_datetime(df_all["date_to"])

    mask = (
        (df_all["driver_id"] == driver_id) &
        (df_all["date_from"] >= date_from) &
        (df_all["date_to"] <= date_to)
    )
    return df_all[mask].copy()


# ---------- Main endpoint ----------

@app.post("/driver-risk", response_model=DriverRiskResponse)
def driver_risk(request: DriverRiskRequest):
    # 1) Get raw trips for this driver + period
    df_raw = load_trips_for_driver(request.driver_id, request.date_from, request.date_to)

    if df_raw.empty:
        raise HTTPException(status_code=404, detail="No trips found for this driver in the given period.")

    # 2) Apply same preprocessing + feature engineering as in the notebook
    df_eng = preprocess_and_engineer(df_raw)

    # 3) Build feature matrix for the model
    X = df_eng[MODEL_FEATURES].copy()
    proba = rf_model.predict_proba(X)[:, 1]       # probability of risky=1
    preds = (proba >= 0.5).astype(int)           # simple 0.5 threshold

    # 4) Aggregate trip-level predictions into a driver summary
    trips_count = int(len(df_eng))
    risky_trips = int(preds.sum())
    risk_ratio = risky_trips / trips_count
    avg_prob = float(proba.mean())

    # 5) Return structured response
    return DriverRiskResponse(
        driver_id=request.driver_id,
        date_from=request.date_from,
        date_to=request.date_to,
        trips_count=trips_count,
        risky_trips=risky_trips,
        risk_ratio=round(risk_ratio, 3),
        avg_risk_probability=round(avg_prob, 3),
    )
