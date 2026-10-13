# Real-World Electric Vehicle Range Prediction Using Machine Learning

An end-to-end Machine Learning web application designed and built for **AI23521 – Build and Deploy for Machine Learning Application**.

The application predicts the real-world remaining driving range (in kilometers) of production **Battery Electric Vehicles (BEVs)** based on verified manufacturer vehicle specifications (battery capacity, chemistry, motor power, curb weight, drivetrain, body type) and multi-factor operational conditions (State of Charge, speed, temperature, traffic congestion, terrain, driving style, HVAC draw, passenger load, battery degradation, and tyre pressure).

---

## 📌 Project Highlights & Validation Benchmark Results
* **Vehicle Scope:** Pure Battery Electric Vehicles (BEVs) only — **8 Manufacturers, 17 Models, 28 Production Variants** (Tata, Mahindra, MG, Hyundai, Kia, BYD, Tesla, BMW). Zero hybrid contamination.
* **Dataset Foundation:** 2,800 drive-cycle observations synthesized via standardized automotive longitudinal dynamics equations (aerodynamic drag, rolling resistance, inertial acceleration, elevation changes, HVAC auxiliary load, and battery degradation) parameterized with verified OEM specifications.
* **Winning Model:** **HistGradientBoosting Regressor Pipeline**
* **Validation Methodology & Anti-Leakage Evaluation:**
  * **Standard 80/20 Holdout Split ($n=560$):**
    * Validation $R^2$: **`0.9664`**
    * Mean Absolute Error (MAE): **`15.10 km`**
    * Root Mean Squared Error (RMSE): **`21.63 km`**
    * Mean Absolute Percentage Error (MAPE): **`11.57%`**
  * **Group-Aware Unseen Vehicle Generalization (GroupShuffleSplit / Zero-Leakage Test):**
    * Generalization $R^2$: **`0.9497`**
    * Mean Absolute Error (MAE): **`20.31 km`**
    * Root Mean Squared Error (RMSE): **`31.78 km`**
    * 5-Fold GroupKFold Cross-Validation: **`Mean R² = 0.9559 ± 0.0041`**
* **Statistical Uncertainty:** 90% prediction interval derived from model residual RMSE ($\pm 1.645 \times \text{RMSE}$).
* **Real-World Calibration:** Transparently benchmarks predictions against certified cycle ratings (ARAI MIDC / WLTP) and verified real-world road tests (Tata C75, Autocar India).
* **Automated Test Coverage:** 42 passing unit and audit tests across `test_audit_suite.py` (20 Section-35 tests) and `test_app.py` (22 application tests) (**100% Passing**).

---

## 🗂️ Project Directory Structure

```text
EV_Range_Prediction/
│
├── app.py                      # Flask application (Auth, REST APIs, ML Inferences, Port 7000)
├── train_model_v2.py           # 6-Algorithm training & dual-validation benchmark pipeline
├── eda_report.py               # Exploratory Data Analysis & statistical audit script
├── generate_real_ev_dataset.py # Multi-factor BEV drive-cycle physics dataset generator
├── test_audit_suite.py         # 20-scenario Section 35 technical audit test suite
├── test_app.py                 # 22-test application and security verification suite
│
├── vehicles_catalogue.json     # Verified OEM technical specs and real-world reference catalogue
├── DATASET_SOURCE.md           # Formal dataset documentation & physics derivations
├── real_ev_dataset.csv         # 2,800 drive-cycle observation records (20 features)
├── model_v2.pkl / model.pkl    # Serialized winning HistGradientBoosting pipeline
├── model_metadata.json         # Machine-readable metrics & OOD feature bounds
├── requirements.txt            # Minimal required dependencies
│
├── README.md                   # Project overview & execution manual
├── PROJECT_REPORT.md           # Complete academic project report
├── DEPLOYMENT_GUIDE.md         # Production WSGI & cloud deployment manual
│
├── templates/
│   ├── base.html               # Shared Neumorphic navbar, user status, alert banners, footer
│   ├── home.html               # Hero live EV preview, feature cards & 6-model benchmark table
│   ├── register.html           # Neumorphic user registration form
│   ├── login.html              # Neumorphic driver login form
│   ├── dashboard.html          # Cockpit with cascading EV selector, tactile sliders & steppers
│   ├── result.html             # Neumorphic circular range gauge & derived energy analytics
│   ├── garage.html             # Personal EV Garage with saved vehicles & default selection
│   ├── history.html            # Personal prediction history telemetry log with detail modal
│   ├── profile.html            # Driver profile, statistics & garage overview
│   └── about.html              # Physics methodology & dual validation benchmark analysis
│
├── static/
│   ├── css/style.css           # Authentic Neumorphic / Soft UI Design System (Dual shadows, Insets)
│   └── images/
│       ├── model_comparison.png      # 6-algorithm R² vs RMSE benchmark graph
│       ├── actual_vs_predicted.png   # 1:1 scatter plot of predicted vs observed range
│       ├── feature_importance.png    # Top relative feature importance rankings
│       ├── target_distribution.png   # Frequency histogram of remaining range
│       ├── manufacturer_distribution.png # Bar chart of records per OEM
│       └── correlation_matrix.png    # Heatmap of Pearson correlation coefficients
│
└── instance/
    └── database.db             # Persistent SQLite user credentials & history database
```

---

## ⚙️ Installation & Setup Instructions

### 1. Prerequisites
Ensure you have **Python 3.10+** (tested on Python 3.13) installed on Windows.

### 2. Install Required Dependencies
Open PowerShell in the project directory:

```powershell
pip install -r requirements.txt
```

### 3. (Optional) Run Dataset Generation, EDA, or Training
All models and data are already pre-generated and verified. If you wish to re-execute any stage:

```powershell
# Re-generate the 2,800-record physics simulation dataset
python generate_real_ev_dataset.py

# Re-run Exploratory Data Analysis & generate distribution charts
python eda_report.py

# Re-train & benchmark all 6 models with dual random & group splits
python train_model_v2.py
```

### 4. Run Automated Test Suites (42 Tests Total)
```powershell
# Run Section 35 Technical Audit Test Suite (20 Tests)
python test_audit_suite.py

# Run Full Application & Security Test Suite (22 Tests)
python test_app.py
```

### 5. Launch the Web Application
```powershell
python app.py
```

Open your browser and navigate to:
👉 **`http://127.0.0.1:7000`**

---

## 🔌 REST API Endpoints

The Flask backend exposes clean JSON endpoints for vehicle cascading:

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/api/manufacturers` | `GET` | Returns list of all 8 supported BEV manufacturers. |
| `/api/models/<manufacturer>` | `GET` | Returns commercial model lines for the specified brand. |
| `/api/variants/<model_name>` | `GET` | Returns list of trims/variants with unique IDs. |
| `/api/vehicle-specs/<variant_id>` | `GET` | Returns verified battery capacity, motor power, curb weight, drivetrain, and official test range. |

---

## 📊 Model Benchmark Summary

### A. Standard Holdout Evaluation (80/20 Random Split, $n=560$)
| Model | MAE (km) | RMSE (km) | Validation $R^2$ | MAPE (%) | Result |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Linear Regression | 42.18 | 58.11 | 0.7578 | 67.01% | Baseline |
| Decision Tree | 38.06 | 56.16 | 0.7738 | 28.94% | Underperformed |
| Random Forest | 27.30 | 37.63 | 0.8984 | 21.28% | Strong |
| Gradient Boosting | 23.81 | 35.24 | 0.9109 | 21.53% | Competitive |
| XGBoost | 19.45 | 27.85 | 0.9444 | 15.63% | Excellent |
| **HistGradientBoosting** | **15.10** | **21.63** | **0.9664** | **11.57%** | 🏆 **Selected** |

### B. Group-Aware Generalization (Zero Vehicle Leakage Split)
| Model | MAE (km) | RMSE (km) | Unseen Group $R^2$ | 5-Fold CV Mean $R^2$ | MAPE (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Linear Regression | 44.81 | 69.22 | 0.7612 | 0.7635 ± 0.005 | 48.47% |
| Decision Tree | 40.05 | 59.86 | 0.8214 | — | 25.69% |
| Random Forest | 31.64 | 45.58 | 0.8965 | 0.8982 ± 0.017 | 22.38% |
| Gradient Boosting | 27.15 | 44.73 | 0.9002 | 0.9003 ± 0.010 | 19.20% |
| XGBoost | 22.94 | 34.59 | 0.9403 | — | 15.89% |
| **HistGradientBoosting** | **20.31** | **31.78** | **0.9497** | **0.9559 ± 0.004** | **12.43%** |
