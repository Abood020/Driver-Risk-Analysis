# Driver Risk Profiling & Behavior Analysis

## 📌 Project Overview
This project analyzes telematics trip data to classify driver behavior and safety risks using machine learning.  
We applied unsupervised clustering (K-Means) to discover risk patterns without manual labeling, and validated the results using a Random Forest classifier.  

The optimal number of clusters was **K = 2**, selected using the Silhouette score.  
A FastAPI backend was developed to allow real-time scoring of driver risk based on date range queries.

---

## 🚗 Identified Driver Profiles (K = 2)

### 1️⃣ Lower-risk (Short/Medium Trips)
- Majority of trips
- Low overspeeding
- Shorter distances
- Mainly daytime trips
- Low fatigue risk

**Interpretation:** Represents normal and safer driving behavior.

---

### 2️⃣ High-risk (Long-distance & Fatigued)
- Smaller proportion of trips
- Trips with higher distance
- Increased night driving
- Higher fatigue likelihood (>4 hours)

**Interpretation:** Long-haul drivers with increased safety risk exposure.

---

## ⚙️ Methodology

### 1️⃣ Data Preprocessing
- Removed impossible trips (negative distance, zero duration)
- Filtered trips with unrealistic speeds (>120 km/h)
- Ensured consistency in `engine_idle <= duration_traveled`
- Handled NaN and infinite values

---

### 2️⃣ Feature Engineering

Key engineered features include:

#### Trip-level
- `avg_speed_kmh`
- `moving_speed_kmh`
- `idle_ratio`
- `seatbelt_ratio`
- `context_weight` (rush hours & night)
- `is_fatigued`

#### Driver-level
- `driver_total_km`
- `driver_seatbelt_habit`
- `driver_risk_tendency`

> Note: The manual weighted risk score was NOT used in the final clustering.  
The model learned feature importance directly.

---

### 3️⃣ Clustering (Unsupervised)

We evaluated K from 2 to 6:


- Metric: Silhouette Score
- Decision: K = 2 provided the clearest separation and simplest interpretability.

---

### 4️⃣ Validation (Supervised)

To validate cluster meaning:

- Trained RandomForestClassifier to predict cluster labels
- Achieved high accuracy (strong separation)
- Feature importance highlighted key drivers:
  - overspeed_standard
  - distance_traveled
  - driver_risk_tendency
  - avg_speed_kmh
  - seat_belt
  - is_fatigued

---

## 🖥 Deployment — FastAPI

The system includes a production-style API for real-time risk scoring.

### Endpoint

### Example Request
```json
{
  "driver_id": 24,
  "date_from": "2025-11-09 08:40:21",
  "date_to": "2025-11-12 00:00:00"
}

{
  "driver_id": 24,
  "date_from": "2025-11-09T08:40:21",
  "date_to": "2025-11-12T00:00:00",
  "trips_count": 3,
  "risky_trips": 1,
  "risk_ratio": 0.333,
  "avg_risk_probability": 0.336
}
