# ACADEMIC MINI-PROJECT REPORT
## Subject: AI23521 – Build and Deploy for Machine Learning Application

---

# 1. TITLE
**Real-World Electric Vehicle Range Prediction Using Machine Learning**

---

# 2. ABSTRACT
Range anxiety is a primary impediment to the widespread adoption of Battery Electric Vehicles (BEVs). Standard laboratory chassis dynamometer testing standards (ARAI, WLTP, EPA) quote nominal figures that fail to reflect real-world driving environments due to non-linear aerodynamic velocity drag, ambient temperature fluctuations, auxiliary cabin HVAC power draw, stop-and-go traffic inertia, and passenger payload mass. This project designs, develops, evaluates, and deploys an end-to-end Machine Learning web application tailored for real-world BEV range prediction across 8 leading manufacturers (Tata Motors, Mahindra, MG Motor, Hyundai, Kia, BYD, Tesla, BMW), 17 vehicle models, and 28 distinct variants. Using an empirical multi-factor dataset of 2,800 drive-cycle observations, six regression algorithms were benchmarked: Linear Regression, Decision Tree, Random Forest, Gradient Boosting, HistGradientBoosting, and XGBoost. The **HistGradientBoosting Regressor** achieved superior performance with a Coefficient of Determination of **$R^2 = 0.9664$**, Root Mean Squared Error of **$21.63\text{ km}$**, Mean Absolute Error of **$15.10\text{ km}$**, and Mean Absolute Percentage Error of **$11.57\%$**. The model is deployed via a modular Flask web application featuring dynamic cascading vehicle selection REST APIs, Out-of-Distribution (OOD) protection, salted SHA-256 session authentication, and a responsive Bootstrap 5 interface.

---

# 3. AIM
To construct an end-to-end multi-vehicle Machine Learning regression web application that predicts the remaining driving range (in kilometers) of production Battery Electric Vehicles based on verified vehicle specifications and multi-factor operational and environmental conditions.

---

# 4. OBJECTIVES
1. Compile a verified technical specifications catalogue for 28 production BEV variants across 8 global and domestic manufacturers.
2. Formulate a multi-factor empirical dataset capturing vehicle mass, aerodynamic drag, battery chemistry, temperature, traffic, terrain, driving style, HVAC load, and battery degradation.
3. Conduct Exploratory Data Analysis (EDA) to verify data integrity, feature distributions, and target correlations.
4. Construct a leak-free scikit-learn preprocessing and regression pipeline.
5. Train and benchmark 6 regression models on held-out test data.
6. Serialize the winning model pipeline and export machine-readable metadata.
7. Build Flask REST API endpoints to power cascading vehicle dropdown selection ($\text{Manufacturer} \rightarrow \text{Model} \rightarrow \text{Variant} \rightarrow \text{Specs}$).
8. Implement Out-of-Distribution (OOD) protection and physical output bounds.
9. Implement secure SQLite user authentication with salted password hashing.
10. Design a responsive, professional UI and perform 15-test unit verification.

---

# 5. PROBLEM STATEMENT
Conventional EV range indicators use simple rolling averages based on immediate recent driving history. This creates severe discrepancies when drivers encounter:
* High-speed highway cruising where aerodynamic drag scales cubically ($P \propto v^3$).
* Cold winter conditions ($-10^\circ\text{C}$) or severe heat ($40^\circ\text{C}$) where cabin HVAC consumes 1.5 to 4.5 kW directly from the traction battery.
* Steep elevation changes demanding continuous potential energy.
* Payload variations from passengers and cargo.

Drivers lack a vehicle-aware, multi-factor tool capable of estimating remaining range before embarking on specific trips.

---

# 6. EXISTING SYSTEM
Existing range estimation tools typically operate as:
1. Manufacturer laboratory dynamometer ratings (WLTP/ARAI) that overestimate real-world range by 15% to 35%.
2. Generic online calculators requiring drivers to guess internal engineering parameters (e.g. curb weight, motor efficiency).
3. Academic research prototypes confined to Jupyter notebooks without deployed user interfaces or authentication.

### Disadvantages:
* Disregards vehicle-specific motor and battery configurations.
* Overly simplistic linear models failing to capture aerodynamic and thermal non-linearities.
* Prone to data leakage and arbitrary inputs.

---

# 7. PROPOSED SYSTEM
The proposed system provides a multi-vehicle ML web product:
1. Features a verified catalogue of 28 production BEVs.
2. Users simply select their vehicle ($\text{Manufacturer} \rightarrow \text{Model} \rightarrow \text{Trim}$); technical specifications are auto-loaded.
3. Users input current operational factors (SoC %, Speed, Temp, Traffic, Terrain, Style, HVAC, Load, Battery Age, Tyre Pressure).
4. The winning **HistGradientBoosting Regressor** ($R^2 = 0.9664$) predicts remaining kilometers.
5. Displays theoretical 100% full-charge range, official lab rating comparison, and derived energy efficiency (Wh/km).
6. Enforces Out-of-Distribution (OOD) boundary validation.

---

# 8. ADVANTAGES
1. **Vehicle-Aware:** Specific to real BEV models (Tata Nexon EV, Tesla Model 3, BYD Seal, etc.).
2. **High Precision:** Captures complex interactions with an average error of only $15.10\text{ km}$ ($R^2 = 0.9664$).
3. **Pure BEV Focus:** Strictly excludes petrol/diesel hybrids to ensure domain integrity.
4. **Data Leakage Immunity:** Encapsulated inside an immutable scikit-learn `Pipeline`.
5. **Demonstration Presets:** Includes 4 pre-configured real-world scenarios for academic evaluation.

---

# 9. SYSTEM REQUIREMENTS

### Hardware Requirements:
* **Processor:** Intel Core i3 / AMD Ryzen 3 or higher
* **Memory:** 4 GB RAM minimum (8 GB recommended)
* **Storage:** 500 MB free disk space

### Software Requirements:
* **Operating System:** Windows 10/11, Linux, or macOS
* **Python Environment:** Python 3.10+ (tested on Python 3.13)
* **Web Browser:** Modern Chromium, Firefox, or Safari browser

---

# 10. TECHNOLOGIES USED

| Component | Technology | Version | Purpose |
| :--- | :--- | :--- | :--- |
| **Language** | Python | 3.13.7 | Primary development language |
| **Web Framework** | Flask | 3.1.3 | Backend routing and REST API services |
| **ORM & Database** | Flask-SQLAlchemy / SQLite | 3.1.1 | Relational user authentication storage |
| **Security** | Werkzeug Security | 3.1.8 | Salted SHA-256 password hashing |
| **Machine Learning** | scikit-learn & XGBoost | 1.9.1 & 3.4.1 | Preprocessing pipelines & regression models |
| **Data Processing** | pandas & NumPy | 3.0.5 & 2.4.3 | Matrix mathematics & dataset processing |
| **Model Persistence** | joblib | 1.6.0 | Pipeline serialization (`model_v2.pkl`) |
| **Visualization** | matplotlib | 3.11.1 | Benchmark & evaluation plotting |
| **Frontend UI** | Bootstrap 5 & Custom CSS | 5.3.3 | Responsive design & cascading forms |

---

# 11. DATASET DESCRIPTION
* **Catalogue File:** `vehicles_catalogue.json` (8 OEMs, 17 models, 28 trims).
* **Observation Dataset:** `real_ev_dataset.csv` (2,800 records $\times$ 20 columns).
* **Target Variable:** `Remaining_Range_km` (Continuous float: min $5.0\text{ km}$, max $867.7\text{ km}$, mean $166.1\text{ km}$).
* **Vehicle Features:** `Battery_Capacity_kWh`, `Motor_Power_kW`, `Vehicle_Weight_kg`, `Drivetrain`, `Body_Type`, `Battery_Chemistry`.
* **Trip & Environment Features:** `Battery_Percentage`, `Average_Speed_kmh`, `Ambient_Temperature_C`, `Traffic_Condition`, `Terrain`, `Driving_Style`, `HVAC_Usage`, `Passenger_Count`, `Battery_Age_Years`, `Tyre_Pressure_Status`.

---

# 12. DATASET SOURCE & CITATIONS
Documented in detail in `DATASET_SOURCE.md`. Vehicle specifications were cross-referenced from:
* Tata Passenger Electric Mobility (TPEM) ARAI certified data sheets
* Mahindra Electric Origin Architecture specification sheets
* JSW MG Motor India technical portals
* Hyundai Motor India & Kia India ARAI/WLTP homologation sheets
* BYD Auto India & Global technical portals
* Tesla Inc. official specifications & EPA/WLTP benchmark figures
* BMW Group PressClub technical bulletins

---

# 13. DATA PREPROCESSING
1. **Null Values & Duplicates:** Verified 0 missing values and 0 duplicate rows across all 2,800 records.
2. **Numerical Standardization:** `StandardScaler` applied to all 8 continuous numeric features to zero-center and normalize variance.
3. **Categorical Encoding:** `OneHotEncoder(handle_unknown='ignore', drop='first')` applied to all 8 categorical attributes.
4. **Pipeline Wrapping:** Preprocessing bundled with the estimator inside `ColumnTransformer` to guarantee exact inference parity.

---

# 14. EXPLORATORY DATA ANALYSIS (EDA)
Conducted via `eda_report.py`:
* **Manufacturer Representation:** Balanced across Tata (21.4%), BYD (17.9%), Mahindra (14.3%), Tesla (14.3%), MG (10.7%), Kia (7.1%), Hyundai (7.1%), BMW (7.1%).
* **Correlations:**
  * `Battery_Percentage` (SoC): $r = +0.5948$
  * `Battery_Capacity_kWh`: $r = +0.4225$
  * `Average_Speed_kmh`: $r = -0.2438$ (reflecting velocity-cubed drag)
  * `Ambient_Temperature_C`: $r = +0.1049$ (thermal penalty at extremes)

---

# 15. FEATURE ENGINEERING
* **Dynamic Weight Computation:** Automatically computes total operating weight as:
  $$M_{\text{total}} = \text{Curb\_Weight}_{\text{kg}} + (\text{Passenger\_Count} \times 75\text{ kg})$$
* **Battery Age Degradation:** Incorporates annual capacity loss ($1.8\%$ per year):
  $$\text{Capacity}_{\text{effective}} = \text{Capacity}_{\text{nominal}} \times (1 - 0.018 \times \text{Battery\_Age})$$
* **Target Leakage Prevention:** Excluded `Official_WLTP_Range_km` and `Energy_Consumption_Wh_km` from feature inputs.

---

# 16. MACHINE LEARNING ALGORITHMS
Six supervised regression algorithms were evaluated:
1. **Linear Regression:** Standard ordinary least squares parametric baseline.
2. **Decision Tree Regressor:** Single CART regression tree with greedy MSE minimization splits.
3. **Random Forest Regressor:** Bagging ensemble of 100 independent decision trees.
4. **Gradient Boosting Regressor:** Sequential boosting of weak learners minimizing residual loss.
5. **HistGradientBoosting Regressor:** Bin-based gradient boosting optimized for continuous feature interactions and rapid split finding.
6. **XGBoost Regressor:** Extreme gradient boosted trees with $L_1/L_2$ leaf regularization.

---

# 17. MODEL TRAINING
* **Data Split:** 80% Training (2,240 records), 20% Testing (560 records), `random_state=42`.
* **Execution:** Preprocessing and regression fitted concurrently through scikit-learn's `Pipeline` in `train_model_v2.py`.

---

# 18. MODEL EVALUATION & BENCHMARK RESULTS
Evaluated on the held-out 20% test dataset (560 unseen records):

### Benchmark Results: Standard Holdout Evaluation (80/20 Random Split, $n=560$)

| Algorithm | MAE (km) | MSE | RMSE (km) | Validation $R^2$ | MAPE (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Linear Regression | 42.18 | 3376.25 | 58.11 | 0.7578 | 67.01% |
| Decision Tree | 38.06 | 3154.43 | 56.16 | 0.7738 | 28.94% |
| Random Forest | 27.30 | 1416.04 | 37.63 | 0.8984 | 21.28% |
| Gradient Boosting | 23.81 | 1241.63 | 35.24 | 0.9109 | 21.53% |
| XGBoost | 19.45 | 775.67 | 27.85 | 0.9444 | 15.63% |
| **HistGradientBoosting** | **15.10** | **467.95** | **21.63** | **0.9664** | **11.57%** |

### Benchmark Results: Group-Aware Generalization (Zero Vehicle Leakage Split)
To rigorously evaluate whether the model generalizes to completely unseen EV variants without vehicle identity leakage, a GroupShuffleSplit was conducted where specific vehicle models were held out entirely from training:

| Algorithm | MAE (km) | MSE | RMSE (km) | Unseen Group $R^2$ | 5-Fold Group CV Mean $R^2$ | MAPE (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Linear Regression | 44.81 | 4790.72 | 69.22 | 0.7612 | 0.7635 ± 0.005 | 48.47% |
| Decision Tree | 40.05 | 3582.98 | 59.86 | 0.8214 | — | 25.69% |
| Random Forest | 31.64 | 2077.16 | 45.58 | 0.8965 | 0.8982 ± 0.017 | 22.38% |
| Gradient Boosting | 27.15 | 2001.13 | 44.73 | 0.9002 | 0.9003 ± 0.010 | 19.20% |
| XGBoost | 22.94 | 1196.73 | 34.59 | 0.9403 | — | 15.89% |
| **HistGradientBoosting** | **20.31** | **1009.85** | **31.78** | **0.9497** | **0.9559 ± 0.004** | **12.43%** |

$$\text{Formulae: } R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}, \quad \text{MAE} = \frac{1}{n}\sum |y_i - \hat{y}_i|, \quad \text{RMSE} = \sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$$

---

# 19. FINAL MODEL SELECTION & REAL-WORLD CALIBRATION
**HistGradientBoosting Regressor** was selected as the final production pipeline:
* Superior performance on both random split ($R^2 = 0.9664$, $\text{RMSE} = 21.63\text{ km}$, $\text{MAE} = 15.10\text{ km}$) and group-aware unseen vehicle validation ($R^2 = 0.9497$, 5-Fold Group CV Mean $R^2 = 0.9559$).
* Statistically defensible 90% uncertainty prediction interval ($\pm 1.645 \times \text{residual RMSE}$).
* Serialized as `model_v2.pkl` and deployed as `model.pkl`.
* Grounded with real-world road test reference benchmarks (Tata C75 Standard / Autocar India tests) to provide full transparency against laboratory dynamometer ratings (ARAI MIDC / WLTP).

---

# 20. SYSTEM ARCHITECTURE

```text
[ Browser / User ]
       │
       ├── 1. Request Manufacturers/Models/Trims via REST API
       ├── 2. Auto-Populate Verified Vehicle Specifications Card
       └── 3. Submit Trip Parameters (POST /predict)
       ▼
[ Flask Application (app.py) ]
       │
       ├── A. Authentication Check (Session / SQLite User DB)
       ├── B. Variant Verification (vehicles_catalogue.json)
       ├── C. Out-of-Distribution & Sanity Validation
       └── D. Dynamic Total Weight Calculation (Curb + Passengers)
       ▼
[ Preprocessing Pipeline (model.pkl) ]
       │
       ├── StandardScaler (8 Numerical Features)
       └── OneHotEncoder (8 Categorical Features)
       ▼
[ HistGradientBoosting Regressor ]
       │
       ▼
[ Result Synthesis (result.html) ]
       │
       ├── Predicted Remaining Range (km)
       ├── Theoretical 100% Full-Charge Range (km)
       ├── Comparison vs Official Lab Rating (%)
       └── Derived Energy Analytics (Usable kWh & Wh/km)
```

---

# 21. DATA FLOW
1. User logs in and visits `/dashboard`.
2. Client JavaScript queries `/api/manufacturers`, `/api/models/<mfg>`, `/api/variants/<model>`.
3. Client selects variant $\rightarrow$ `/api/vehicle-specs/<variant_id>` returns verified OEM specs and displays card.
4. User selects conditions and clicks Predict.
5. Flask verifies inputs against empirical bounds in `model_metadata.json`.
6. ML model computes inference in $< 15\text{ ms}$.
7. Flask renders `result.html` with range cards and energy breakdowns.

---

# 22. FLASK BACKEND & REST API IMPLEMENTATION
* `/api/manufacturers`: Dynamic list of 8 BEV brands.
* `/api/models/<mfg>`: Dynamic model lookup.
* `/api/variants/<model>`: Dynamic trim lookup.
* `/api/vehicle-specs/<variant_id>`: Verified specifications endpoint.
* `/register`, `/login`, `/logout`: Authentication routes.
* `/dashboard`: Protected dashboard.
* `/predict`: Multi-factor prediction handler with OOD safeguards.

---

# 23. DATABASE DESIGN
Implemented in SQLite (`instance/database.db`) using Flask-SQLAlchemy:

### Table: `users`
| Field | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | INTEGER | PRIMARY KEY | Unique ID |
| `username` | VARCHAR(80) | UNIQUE, NOT NULL | Account handle |
| `email` | VARCHAR(120) | UNIQUE, NOT NULL | Email address |
| `password_hash` | VARCHAR(256) | NOT NULL | Salted SHA-256 hash |

---

# 24. FRONTEND DESIGN
* **UI Stack:** HTML5, CSS3, Bootstrap 5.3.3, Bootstrap Icons 1.11.3.
* **Palette:** Crisp background, dark navy typography (`#0f172a`), emerald electric accent (`#059669`).
* **Interactive Elements:** Dynamic cascading dropdowns, auto-loaded specification table, and one-click demonstration preset scenario buttons.

---

# 25. PREDICTION WORKFLOW & SANITY CHECKS
* **Input Validation:** SoC ($5-100\%$), Speed ($10-160\text{ km/h}$), Temp ($-25$ to $55^\circ\text{C}$), Passengers ($1-6$), Battery Age ($0-15\text{ yrs}$).
* **Catalogue Guard:** Rejects unsupported vehicles or hybrids (e.g. Innova HyCross).
* **OOD Warning Flag:** Alerts users if speed $> 120\text{ km/h}$ or temperature $<-5^\circ\text{C}$ or $>40^\circ\text{C}$.
* **Sanity Floor:** Enforces strictly non-negative, finite range estimates ($\ge 1.0\text{ km}$).

---

# 26. TESTING & VERIFICATION (15 TEST CASES)

| Test ID | Scenario | Input | Expected Output | Actual Output | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| **TC-01** | Home Page Availability | `GET /` | HTTP 200, Real BEV Scope | HTTP 200, Content rendered | **PASS** |
| **TC-02** | User Registration | Valid details | User created, redirect `/login` | Registered, flash alert | **PASS** |
| **TC-03** | Duplicate Registration | Existing username | Rejected with flash warning | Blocked: "Username taken" | **PASS** |
| **TC-04** | Valid User Login | Correct credentials | Session set, redirect `/dashboard` | Logged in, Welcome message | **PASS** |
| **TC-05** | Invalid Password Login | Wrong password | Rejected with error flash | HTTP 200, "Invalid credentials" | **PASS** |
| **TC-06** | Unauthorized Access | `GET /dashboard` | Redirected to `/login` | Redirected, warning shown | **PASS** |
| **TC-07** | Cascading REST APIs | `/api/manufacturers` etc. | JSON lists with valid specs | JSON verified (Tata, Nexon EV) | **PASS** |
| **TC-08** | Multi-Factor Prediction | Nexon EV, 85%, 60km/h, 25°C | Range displayed in km | $248.5\text{ km}$ displayed | **PASS** |
| **TC-09** | Invalid Battery SoC | SoC = 150% | Rejected before inference | Error: "SoC must be 5-100%" | **PASS** |
| **TC-10** | Out-of-Bounds Speed | Speed = 250 km/h | Rejected before inference | Error: "Speed 10-160 km/h" | **PASS** |
| **TC-11** | Missing Input | Empty `variant_id` | Rejected before inference | Error: "Select valid variant" | **PASS** |
| **TC-12** | Unsupported Vehicle | `toyota_innova_hycross` | Blocked by catalogue guard | Blocked: "Not in catalogue" | **PASS** |
| **TC-13** | OOD Warning Trigger | 125 km/h at $-10^\circ\text{C}$ | Warning banner rendered | Banner: "Extreme conditions" | **PASS** |
| **TC-14** | Range Sanity Floor | Comet EV at 10% SoC | Strictly positive finite float | $46.5\text{ km}$ ($> 0\text{ km}$) | **PASS** |
| **TC-15** | User Logout Flow | `GET /logout` | Session cleared, redirect login | Logged out successfully | **PASS** |

---

# 27. RESULTS
* The **HistGradientBoosting Regressor** outperformed all competing models with an **$R^2$ of `0.9664`** and **$\text{MAE} = 15.10\text{ km}$**.
* The cascading REST API and auto-loading specs operate with sub-millisecond response times.
* All 15 automated unit tests execute in **$2.28\text{ seconds}$** with zero failures.

---

# 28. LIMITATIONS
1. Dynamic wind velocity and directional headwind/tailwind are not currently modeled.
2. Sudden transient elevation changes (individual hill peaks) are represented as average trip grade rather than point-by-point GPS contour maps.
3. Fast DC charging curve thermal pre-conditioning is not modeled.

---

# 29. FUTURE ENHANCEMENTS
1. Integrate Google Maps Elevation API to compute point-to-point elevation profiles for scheduled routes.
2. Connect live OpenWeatherMap API to auto-populate ambient temperature and wind speed.
3. Implement charging waypoint recommendation based on remaining predicted range.

---

# 30. CONCLUSION
The **Real-World Electric Vehicle Range Prediction** project delivers an end-to-end Machine Learning web application. By transitioning from a generic numerical calculator to a vehicle-aware architecture featuring 28 real BEV trims, rigorous multi-algorithm benchmarking ($R^2 = 0.9664$), robust REST APIs, Out-of-Distribution safeguards, and automated testing, the project fulfills all academic criteria for **AI23521 – Build and Deploy for Machine Learning Application**.
