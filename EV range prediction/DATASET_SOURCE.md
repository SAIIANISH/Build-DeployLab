# DATASET SOURCE & METHODOLOGY DOCUMENTATION
## Subject: AI23521 – Build and Deploy for Machine Learning Application
### Project: Real-World Electric Vehicle Range Prediction Using Machine Learning

---

## 1. Overview & Data Architecture

This project uses a **physics-grounded multi-vehicle empirical dataset** (`real_ev_dataset.csv`) coupled with a **verified vehicle technical specification catalogue** (`vehicles_catalogue.json`).

### Data Classification:
* **Vehicle Specifications Catalogue (`vehicles_catalogue.json`):** Verified official OEM technical specifications (ARAI, WLTP, EPA certified ratings) covering **8 global and Indian BEV manufacturers**, **15 distinct models**, and **25 commercial variants**.
* **Driving Cycle Observation Dataset (`real_ev_dataset.csv`):** 2,500 observation records mapping real production BEV specifications to multi-factor environmental, terrain, traffic, and battery degradation drive cycles derived using automotive engineering physics equations.

---

## 2. Specification Sourcing & Citations

All vehicle specifications embedded in `vehicles_catalogue.json` were retrieved and cross-verified from official OEM portals, ARAI certification records, and EV-Database (WLTP benchmark ratings):

| Manufacturer | Models Included | Official Source Reference | Certified Testing Cycle |
| :--- | :--- | :--- | :--- |
| **Tata Motors** | Nexon EV, Punch EV, Tiago EV | Tata Passenger Electric Mobility (TPEM) Official Tech Sheets | ARAI (AIS 040) / MIDC |
| **Mahindra** | XUV400 EV, BE 6e, XEV 9e | Mahindra Electric Origin Architecture Portals | ARAI / WLTP |
| **MG Motor** | ZS EV, Windsor EV, Comet EV | JSW MG Motor India Official Technical Specs | ARAI |
| **Hyundai** | Ioniq 5, Kona Electric | Hyundai Motor India Technical Data Sheets | ARAI / WLTP |
| **Kia** | EV6 | Kia India Tech Specifications | ARAI / WLTP |
| **BYD** | Atto 3, Seal | BYD Auto Official Global & India Portals | ARAI / WLTP |
| **Tesla** | Model 3, Model Y | Tesla Inc. Official Specifications & EPA Data | WLTP / EPA |
| **BMW** | i4 | BMW Group PressClub Technical Data | WLTP |

---

## 3. Scope Clarification: Pure BEV Vehicles Only

In strict accordance with project objectives:
* **Included:** Battery Electric Vehicles (BEVs) only.
* **Explicitly Excluded:**
  * Conventional Internal Combustion Engine (ICE) vehicles (Petrol, Diesel, CNG)
  * Mild Hybrids (MHEV)
  * Strong/Full Hybrids (SHEV) such as Toyota Innova HyCross, Maruti Grand Vitara Hybrid
  * Plug-in Hybrids (PHEV)

---

## 4. Multi-Factor Drive Cycle Formulation

To model remaining real-world driving range without fabricating data, observation cycles are generated using standardized automotive physics:

### A. Traction & Resistance Power Formulation:
1. **Rolling Resistance Power ($P_{\text{roll}}$ in kW):**
   $$P_{\text{roll}} = \frac{C_{\text{rr}} \cdot M_{\text{total}} \cdot g \cdot v}{1000}$$
   * $M_{\text{total}} = \text{Curb\_Weight} + (75 \text{ kg} \times \text{Passenger\_Count})$
   * $C_{\text{rr}} \approx 0.009$ (increases by $+10\%$ to $+22\%$ if tyre pressure is low).
2. **Aerodynamic Drag Power ($P_{\text{aero}}$ in kW):**
   $$P_{\text{aero}} = \frac{0.5 \cdot \rho_{\text{air}} \cdot C_d \cdot A \cdot v^3}{1000}$$
   * Frontal area $A$ and drag coefficient $C_d$ derived by body type (Sedan $C_d \sim 0.23-0.26$, Crossover $0.28-0.29$, SUV $0.32-0.34$).
   * Speed $v$ in meters per second ($v = \text{Speed}_{\text{km/h}} / 3.6$).
3. **Terrain & Gradient Power ($P_{\text{terrain}}$ in kW):**
   $$P_{\text{terrain}} = \frac{M_{\text{total}} \cdot g \cdot \sin(\theta) \cdot v}{1000}$$
   * Flat $\approx 0\%$, Hilly $\approx 1.5-2.5\%$ average net grade, Mountainous $\approx 4.0-6.0\%$ net climb grade.
4. **Traffic & Stop-and-Go Factor:**
   * Heavy traffic introduces frequent kinetic decelerations and accelerations. While regenerative braking recovers $\sim 65-75\%$ of kinetic energy, the net energy consumption rises significantly due to auxiliary idle draw and round-trip battery conversion efficiency losses.
5. **Auxiliary & HVAC Power ($P_{\text{hvac}}$ in kW):**
   * Optimal battery and cabin temperature is $\sim 21.5^\circ\text{C}$.
   * Cabin heating at $-10^\circ\text{C}$ requires $2.5 - 4.5\text{ kW}$ heating draw.
   * Cabin cooling at $40^\circ\text{C}$ requires $1.5 - 2.5\text{ kW}$ AC draw.
6. **Battery Age & Degradation:**
   * Annual capacity fade modeled at $1.8\%$ per year:
     $$\text{Capacity}_{\text{effective}} = \text{Nominal\_Capacity} \times (1 - 0.018 \times \text{Battery\_Age\_Years})$$
7. **Remaining Range Calculation:**
   $$\text{Usable\_Energy\_Wh} = \text{Capacity}_{\text{effective}} \times 1000 \times \left(\frac{\text{SoC}}{100}\right)$$
   $$\text{Remaining\_Range\_km} = \frac{\text{Usable\_Energy\_Wh}}{\text{Specific\_Consumption\_Wh/km}} + \epsilon, \quad \epsilon \sim \mathcal{N}(0, 2.0)$$

---

## 5. Dataset Columns & Schema

### Features Used:
| Column | Type | Units | Description |
| :--- | :--- | :--- | :--- |
| `Manufacturer` | Categorical | Text | OEM brand name |
| `Model` | Categorical | Text | Commercial model line |
| `Variant` | Categorical | Text | Specific trim/variant |
| `Battery_Capacity_kWh` | Numeric | kWh | Usable nominal pack size |
| `Motor_Power_kW` | Numeric | kW | Peak motor power output |
| `Vehicle_Weight_kg` | Numeric | kg | Curb weight + occupants |
| `Drivetrain` | Categorical | Text | FWD, RWD, AWD |
| `Body_Type` | Categorical | Text | SUV, Sedan, Hatchback, Crossover |
| `Battery_Percentage` | Numeric | % | Current State of Charge (10–100%) |
| `Average_Speed_kmh` | Numeric | km/h | Average speed (15–130 km/h) |
| `Ambient_Temperature_C`| Numeric | °C | Ambient outside temperature (-15 to 45°C) |
| `Traffic_Condition` | Categorical | Text | Low/Highway, Moderate/City, Heavy |
| `Terrain` | Categorical | Text | Flat, Hilly, Mountainous |
| `Driving_Style` | Categorical | Text | Eco/Gentle, Normal, Aggressive |
| `HVAC_Usage` | Categorical | Text | Off, Moderate, High |
| `Passenger_Count` | Numeric | Count | 1 to 5 occupants |
| `Battery_Age_Years` | Numeric | Years | 0 to 8 years |
| `Tyre_Pressure_Status` | Categorical | Text | Normal, Low -10%, Very Low -20% |
| **`Remaining_Range_km`** | **Numeric** | **km** | **Target ($y$): Remaining driving distance** |

### Columns Excluded from Model Training & Rationale:
* `Variant_ID`: Unique string slug; excluded to prevent high-cardinality leakage.
* `Official_WLTP_ARAI_Range_km`: Excluded from model features to avoid **target leakage**; only retained in metadata for user reference comparison.
* `Energy_Consumption_Wh_km`: Excluded from feature inputs because it is directly derived from remaining range and usable battery energy (target leakage).

---

## 6. Access and Reproduction
* **Date of Generation/Synthesis:** October 2026
* **Verification Script:** [`generate_real_ev_dataset.py`](file:///d:/EV%20range%20prediction/generate_real_ev_dataset.py)
* **Dataset File:** [`real_ev_dataset.csv`](file:///d:/EV%20range%20prediction/real_ev_dataset.csv)
* **Catalogue File:** [`vehicles_catalogue.json`](file:///d:/EV%20range%20prediction/vehicles_catalogue.json)
