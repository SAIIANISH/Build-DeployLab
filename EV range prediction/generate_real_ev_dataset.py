import json
import numpy as np
import pandas as pd

# Set fixed seed for 100% reproducibility
np.random.seed(42)

# Load verified vehicle catalogue
with open('vehicles_catalogue.json', 'r') as f:
    catalogue = json.load(f)

# Flatten catalogue into a list of variants
variants = []
for mfg in catalogue['manufacturers']:
    mfg_name = mfg['name']
    for model in mfg['models']:
        model_name = model['name']
        for var in model['variants']:
            var_data = var.copy()
            var_data['manufacturer'] = mfg_name
            var_data['model'] = model_name
            variants.append(var_data)

print(f"Loaded {len(variants)} verified BEV variants across {len(catalogue['manufacturers'])} manufacturers.")

# Generate 100 diverse driving scenarios per variant (Total: 25 * 100 = 2,500 records)
samples_per_variant = 100
records = []

traffic_options = ['Low/Highway', 'Moderate/City', 'Heavy/Stop-and-Go']
traffic_probs = [0.40, 0.40, 0.20]

terrain_options = ['Flat', 'Hilly', 'Mountainous']
terrain_probs = [0.60, 0.30, 0.10]

driving_styles = ['Eco/Gentle', 'Normal', 'Aggressive']
style_probs = [0.30, 0.50, 0.20]

hvac_options = ['Off', 'Moderate', 'High']
hvac_probs = [0.25, 0.50, 0.25]

tyre_options = ['Normal', 'Low -10%', 'Very Low -20%']
tyre_probs = [0.75, 0.20, 0.05]

for v in variants:
    mfg = v['manufacturer']
    model = v['model']
    variant_name = v['name']
    cap_kwh = v['battery_capacity_kwh']
    chem = v['battery_chemistry']
    power_kw = v['motor_power_kw']
    curb_wt = v['curb_weight_kg']
    drivetrain = v['drivetrain']
    body_type = v['body_type']
    official_range = v['official_range_km']

    # Drag area Cd * A by body type
    if body_type == 'Sedan':
        cda = 0.53
    elif body_type == 'Hatchback':
        cda = 0.61
    elif body_type == 'Crossover':
        cda = 0.67
    else:  # SUV
        cda = 0.86

    for _ in range(samples_per_variant):
        # 1. State of Charge (10% to 100%)
        soc = np.random.uniform(10.0, 100.0)

        # 2. Average Speed (15 to 130 km/h)
        speed = np.random.uniform(15.0, 130.0)

        # 3. Ambient Temperature (-15°C to 45°C)
        temp = np.random.uniform(-15.0, 45.0)

        # 4. Traffic Condition
        traffic = np.random.choice(traffic_options, p=traffic_probs)

        # 5. Terrain
        terrain = np.random.choice(terrain_options, p=terrain_probs)

        # 6. Driving Style
        style = np.random.choice(driving_styles, p=style_probs)

        # 7. HVAC Usage
        hvac = np.random.choice(hvac_options, p=hvac_probs)

        # 8. Passengers (1 to 5)
        passengers = int(np.random.choice([1, 2, 3, 4, 5], p=[0.45, 0.30, 0.12, 0.08, 0.05]))

        # 9. Battery Age (0 to 8 years)
        age = float(np.random.choice(np.arange(0, 9), p=[0.25, 0.20, 0.18, 0.12, 0.10, 0.06, 0.04, 0.03, 0.02]))

        # 10. Tyre Pressure
        tyre = np.random.choice(tyre_options, p=tyre_probs)

        # --- PHYSICS CALCULATIONS ---
        total_weight = curb_wt + (passengers * 75.0)
        v_ms = speed / 3.6

        # Rolling resistance
        crr = 0.009
        if tyre == 'Low -10%':
            crr *= 1.12
        elif tyre == 'Very Low -20%':
            crr *= 1.25
        p_roll = (crr * total_weight * 9.81 * v_ms) / 1000.0  # kW

        # Aerodynamic power
        p_aero = (0.5 * 1.225 * cda * (v_ms ** 3)) / 1000.0  # kW

        # Terrain power
        if terrain == 'Flat':
            p_terrain = 0.0
        elif terrain == 'Hilly':
            p_terrain = (total_weight * 9.81 * 0.016 * v_ms) / 1000.0  # ~1.6% grade
        else:  # Mountainous
            p_terrain = (total_weight * 9.81 * 0.042 * v_ms) / 1000.0  # ~4.2% grade

        # Traffic multiplier
        traffic_mult = 1.0 if traffic == 'Low/Highway' else (1.14 if traffic == 'Moderate/City' else 1.34)

        # Style multiplier
        style_mult = 0.90 if style == 'Eco/Gentle' else (1.00 if style == 'Normal' else 1.22)

        # Drivetrain loss multiplier
        dt_mult = 1.00 if drivetrain == 'FWD' else (1.02 if drivetrain == 'RWD' else 1.07)

        # HVAC power draw (kW)
        temp_diff = abs(temp - 21.5)
        if hvac == 'Off':
            p_hvac = 0.35  # baseline infotainment/instruments
        elif hvac == 'Moderate':
            p_hvac = np.clip(0.35 + 0.07 * temp_diff, 0.6, 2.5)
        else:  # High
            p_hvac = np.clip(0.35 + 0.14 * temp_diff, 1.2, 4.5)

        # Chemistry cold penalty (LFP loses slightly more capacity/efficiency at sub-zero)
        chem_factor = 1.0
        if temp < 0:
            chem_factor = 1.12 if chem == 'LFP' else 1.06

        # Effective total power required (kW)
        traction_power = (p_roll + p_aero + p_terrain) * traffic_mult * style_mult * dt_mult
        total_power = (traction_power + p_hvac) * chem_factor

        # Specific consumption (Wh/km)
        consumption_wh_km = (total_power * 1000.0) / speed
        consumption_wh_km = np.clip(consumption_wh_km, 90.0, 480.0)

        # Available energy with battery age degradation (~1.8% annual degradation)
        effective_capacity = cap_kwh * (1.0 - (0.018 * age))
        usable_energy_wh = effective_capacity * 1000.0 * (soc / 100.0)

        # Remaining range in km
        ideal_range = usable_energy_wh / consumption_wh_km

        # Real-world observation noise (Gaussian +/- 2.0 km)
        noise = np.random.normal(0, 2.0)
        remaining_range = np.round(np.clip(ideal_range + noise, 1.0, 950.0), 1)

        records.append({
            'Manufacturer': mfg,
            'Model': model,
            'Variant': variant_name,
            'Battery_Capacity_kWh': round(cap_kwh, 1),
            'Motor_Power_kW': round(power_kw, 1),
            'Vehicle_Weight_kg': int(total_weight),
            'Drivetrain': drivetrain,
            'Body_Type': body_type,
            'Battery_Chemistry': chem,
            'Battery_Percentage': round(soc, 1),
            'Average_Speed_kmh': round(speed, 1),
            'Ambient_Temperature_C': round(temp, 1),
            'Traffic_Condition': traffic,
            'Terrain': terrain,
            'Driving_Style': style,
            'HVAC_Usage': hvac,
            'Passenger_Count': passengers,
            'Battery_Age_Years': round(age, 1),
            'Tyre_Pressure_Status': tyre,
            'Remaining_Range_km': remaining_range
        })

df = pd.DataFrame(records)
# Shuffle rows with fixed seed
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

df.to_csv('real_ev_dataset.csv', index=False)
print(f"Successfully generated 'real_ev_dataset.csv' with {df.shape[0]} rows and {df.shape[1]} columns!")
print(f"Unique Manufacturers: {df['Manufacturer'].nunique()}")
print(f"Unique Models:        {df['Model'].nunique()}")
print(f"Unique Variants:      {df['Variant'].nunique()}")
print("\nFirst 3 rows:")
print(df.head(3))
