import numpy as np
import pandas as pd

# Set fixed random seed for reproducibility
np.random.seed(42)

# Number of sample driving cycles/vehicle records
n_samples = 1500

# 1. Battery Capacity (kWh): Typical 2020-2026 EV capacities (e.g., Nexon EV 30kWh to Lucid/Tesla 100+ kWh)
battery_capacity = np.random.uniform(30.0, 110.0, n_samples)

# 2. State of Charge / Battery Percentage (%): Range from 10% to 100%
battery_percentage = np.random.uniform(10.0, 100.0, n_samples)

# 3. Vehicle Speed (km/h): Average trip speed from 20 km/h (city traffic) to 125 km/h (expressway)
speed = np.random.uniform(20.0, 125.0, n_samples)

# 4. Vehicle Weight (kg): Curb weight + passengers/cargo (1300 kg to 2600 kg)
weight = np.random.uniform(1300.0, 2600.0, n_samples)

# 5. Ambient Temperature (°C): Operating temperatures from -10°C to 45°C
temperature = np.random.uniform(-10.0, 45.0, n_samples)

# 6. Driving Mode: Eco, Normal, Sport
modes = ['Eco', 'Normal', 'Sport']
driving_mode = np.random.choice(modes, size=n_samples, p=[0.35, 0.45, 0.20])

# Physics-informed Energy Consumption Calculation (Wh/km):
# Base rolling resistance and vehicle weight contribution
rolling_power_kw = (0.01 * weight * 9.81 * (speed / 3.6)) / 1000.0

# Aerodynamic drag power contribution (drag increases with cube of velocity)
aero_power_kw = (0.5 * 1.225 * 0.27 * 2.3 * ((speed / 3.6) ** 3)) / 1000.0

# Auxiliary & HVAC power based on ambient temperature (optimal around 21°C)
temp_diff = np.abs(temperature - 21.0)
hvac_power_kw = 0.5 + 0.08 * temp_diff  # heating/cooling draw

# Driving mode energy factor
mode_factors = {'Eco': 0.90, 'Normal': 1.00, 'Sport': 1.18}
mode_multiplier = np.array([mode_factors[m] for m in driving_mode])

# Total power needed (kW) and energy consumption (Wh/km)
total_power_kw = (rolling_power_kw + aero_power_kw + hvac_power_kw) * mode_multiplier
# Specific consumption (Wh/km) = (total_power_kw * 1000) / speed_kmh
consumption_wh_per_km = (total_power_kw * 1000.0) / speed

# Clip to realistic modern EV consumption bounds (110 Wh/km to 320 Wh/km)
consumption_wh_per_km = np.clip(consumption_wh_per_km, 110.0, 320.0)

# Available usable energy in battery (kWh)
usable_energy_kwh = battery_capacity * (battery_percentage / 100.0)

# Theoretical Range (km) = (Usable Energy in Wh) / (Consumption in Wh/km)
ideal_range = (usable_energy_kwh * 1000.0) / consumption_wh_per_km

# Add realistic real-world variance/noise (Gaussian noise +/- 2.5 km)
noise = np.random.normal(0, 2.5, n_samples)
range_km = np.round(np.clip(ideal_range + noise, 5.0, 950.0), 1)

# Build DataFrame
df = pd.DataFrame({
    'Battery_Capacity_kWh': np.round(battery_capacity, 1),
    'Battery_Percentage': np.round(battery_percentage, 1),
    'Vehicle_Speed_kmh': np.round(speed, 1),
    'Vehicle_Weight_kg': np.round(weight, 0).astype(int),
    'Ambient_Temperature_C': np.round(temperature, 1),
    'Driving_Mode': driving_mode,
    'Range_km': range_km
})

# Save to CSV
df.to_csv('dataset.csv', index=False)
print(f"Dataset successfully created with {df.shape[0]} rows and {df.shape[1]} columns!")
print("First 5 rows:")
print(df.head())
