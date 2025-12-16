# Driver Risk Profiling & Behavior Analysis

## 📌 Project Overview
This project analyzes telematics data to classify drivers based on their behavior, efficiency, and safety risks. Using unsupervised machine learning (**K-Means Clustering**), we identified distinct driver profiles without prior labels. The results were validated using a supervised **Random Forest Classifier**, achieving **98% accuracy**.

## 📊 Key Findings (Driver Profiles)
We identified 4 distinct clusters of drivers:
1.  **High Risk (Speeders):**
    * Smallest group but most dangerous.
    * **Characteristics:** Highest average speed and extreme risk scores (>90).
    * **Action:** Requires immediate intervention or training.
2.  **Fatigued (Long Haul):**
    * **Characteristics:** Driving for >4 hours continuously (Fatigue Risk).
    * **Action:** Enforce mandatory breaks.
3.  **Inefficient (High Idle):**
    * Largest group (The majority).
    * **Characteristics:** High idling ratios (engine running while stopped), leading to fuel wastage.
    * **Action:** Awareness campaigns on engine idling.
4.  **Safe (Local):**
    * **Characteristics:** Short distances, low speeds, and minimal risk scores.
    * **Action:** Eligible for safety bonuses.

## ⚙️ Methodology
### 1. Data Preprocessing
* Removed physics-defying errors (e.g., negative speeds, speeds > 200 km/h).
* Fixed logical inconsistencies where `idle_time > duration`.

### 2. Feature Engineering
Created domain-specific features to enhance model performance:
* **Weighted Risk Score:** A composite score combining overspeeding, idling, and seatbelt violations (weighted by time of day).
* **Fatigue Indicator:** Flagging trips exceeding 4 hours.
* **Seatbelt Ratio:** Percentage of trip time without a seatbelt.

### 3. Modeling (Unsupervised)
* **Algorithm:** K-Means Clustering.
* **Selection:** Used the **Elbow Method** to determine that **K=4** was the optimal number of clusters.

### 4. Evaluation (Supervised Validation)
* To validate the quality of the clusters, we trained a **Random Forest Classifier** to predict the assigned profiles.
* **Result:** The model achieved **98% Accuracy**, confirming that the identified profiles are distinct and robust.
* **Top Features:** `hour`, `overspeed_standard`, and `time_category` were the most critical factors in distinguishing drivers.



## 🛠 Technologies Used
* **Python:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (K-Means, Random Forest, PCA)
* **Visualization:** Matplotlib, Seaborn

