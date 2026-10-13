import datetime
import json
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GroupShuffleSplit, GroupKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

print("=" * 80)
print("EV RANGE PREDICTION ENGINE: RE-TRAINING & SCIENTIFIC AUDIT PIPELINE")
print("=" * 80)

# 1. Load Dataset
df = pd.read_csv('real_ev_dataset.csv')
print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Unique variants: {df['Variant'].nunique()} across {df['Manufacturer'].nunique()} manufacturers\n")

# 2. Define Features & Target
numerical_features = [
    'Battery_Capacity_kWh',
    'Motor_Power_kW',
    'Vehicle_Weight_kg',
    'Battery_Percentage',
    'Average_Speed_kmh',
    'Ambient_Temperature_C',
    'Passenger_Count',
    'Battery_Age_Years'
]

categorical_features = [
    'Drivetrain',
    'Body_Type',
    'Battery_Chemistry',
    'Traffic_Condition',
    'Terrain',
    'Driving_Style',
    'HVAC_Usage',
    'Tyre_Pressure_Status'
]

target = 'Remaining_Range_km'

X = df[numerical_features + categorical_features]
y = df[target]
groups = df['Variant']

# 3. Compute Feature Bounds for Out-of-Distribution (OOD) Protection
feature_bounds = {}
for col in numerical_features:
    feature_bounds[col] = {
        'min': float(df[col].min()),
        'max': float(df[col].max()),
        'mean': float(round(df[col].mean(), 2)),
        'std': float(round(df[col].std(), 2))
    }

categorical_options = {}
for col in categorical_features:
    categorical_options[col] = df[col].unique().tolist()

# 4. Preprocessing Setup
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical_features)
    ]
)

# 5. Define Model Candidates
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'HistGradientBoosting': HistGradientBoostingRegressor(random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, objective='reg:squarederror')
}

# -------------------------------------------------------------
# 6. Evaluation A: Random Row-Wise Split (80/20)
# -------------------------------------------------------------
print("--- EVALUATION A: RANDOM ROW-WISE SPLIT (80% Train, 20% Test) ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

results_random = {}
best_model_name = None
best_rmse = float('inf')
best_pipeline = None

for name, model in models.items():
    pipe = Pipeline([('preprocessor', preprocessor), ('regressor', model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    
    mae = float(mean_absolute_error(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, y_pred))
    mape = float(mean_absolute_percentage_error(y_test, y_pred) * 100.0)
    
    results_random[name] = {
        'MAE': round(mae, 2),
        'MSE': round(mse, 2),
        'RMSE': round(rmse, 2),
        'R2': round(r2, 4),
        'MAPE': round(mape, 2),
        'pipeline': pipe,
        'y_pred': y_pred
    }
    
    print(f"  {name:<22} | R²: {r2:.4f} | RMSE: {rmse:>6.2f} km | MAE: {mae:>6.2f} km | MAPE: {mape:>5.2f}%")
    
    if rmse < best_rmse:
        best_rmse = rmse
        best_model_name = name
        best_pipeline = pipe

# -------------------------------------------------------------
# 7. Evaluation B: Group-Aware Split (Vehicle Leakage Check)
# -------------------------------------------------------------
print("\n--- EVALUATION B: GROUP-AWARE SPLIT (Holding out 20% unseen vehicle variants) ---")
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train_g, X_test_g = X.iloc[train_idx], X.iloc[test_idx]
y_train_g, y_test_g = y.iloc[train_idx], y.iloc[test_idx]

results_group = {}
for name, model in models.items():
    pipe = Pipeline([('preprocessor', preprocessor), ('regressor', model)])
    pipe.fit(X_train_g, y_train_g)
    y_pred_g = pipe.predict(X_test_g)
    
    mae_g = float(mean_absolute_error(y_test_g, y_pred_g))
    mse_g = float(mean_squared_error(y_test_g, y_pred_g))
    rmse_g = float(np.sqrt(mse_g))
    r2_g = float(r2_score(y_test_g, y_pred_g))
    mape_g = float(mean_absolute_percentage_error(y_test_g, y_pred_g) * 100.0)
    
    results_group[name] = {
        'MAE': round(mae_g, 2),
        'MSE': round(mse_g, 2),
        'RMSE': round(rmse_g, 2),
        'R2': round(r2_g, 4),
        'MAPE': round(mape_g, 2)
    }
    print(f"  {name:<22} | R²: {r2_g:.4f} | RMSE: {rmse_g:>6.2f} km | MAE: {mae_g:>6.2f} km | MAPE: {mape_g:>5.2f}%")

# -------------------------------------------------------------
# 8. Evaluation C: 5-Fold GroupKFold Cross-Validation
# -------------------------------------------------------------
print("\n--- EVALUATION C: 5-FOLD GROUPKFOLD CROSS-VALIDATION ---")
gkf = GroupKFold(n_splits=5)
results_cv = {}

for name in ['HistGradientBoosting', 'Gradient Boosting', 'Random Forest', 'Linear Regression']:
    cv_r2, cv_rmse, cv_mae, cv_mape = [], [], [], []
    for tr, te in gkf.split(X, y, groups=groups):
        p = Pipeline([('preprocessor', preprocessor), ('regressor', models[name])])
        p.fit(X.iloc[tr], y.iloc[tr])
        preds = p.predict(X.iloc[te])
        cv_r2.append(r2_score(y.iloc[te], preds))
        cv_rmse.append(np.sqrt(mean_squared_error(y.iloc[te], preds)))
        cv_mae.append(mean_absolute_error(y.iloc[te], preds))
        cv_mape.append(mean_absolute_percentage_error(y.iloc[te], preds) * 100.0)
    
    results_cv[name] = {
        'mean_r2': round(float(np.mean(cv_r2)), 4),
        'std_r2': round(float(np.std(cv_r2)), 4),
        'mean_rmse': round(float(np.mean(cv_rmse)), 2),
        'mean_mae': round(float(np.mean(cv_mae)), 2),
        'mean_mape': round(float(np.mean(cv_mape)), 2)
    }
    print(f"  {name:<22} | Mean R²: {results_cv[name]['mean_r2']} (±{results_cv[name]['std_r2']}) | Mean RMSE: {results_cv[name]['mean_rmse']} km | Mean MAE: {results_cv[name]['mean_mae']} km")

# -------------------------------------------------------------
# 9. Retrain Winning Pipeline on Entire Dataset for Production
# -------------------------------------------------------------
print(f"\n--> Selected Model: {best_model_name}")
final_production_pipeline = Pipeline([('preprocessor', preprocessor), ('regressor', models[best_model_name])])
final_production_pipeline.fit(X, y)

# Export Pipelines
joblib.dump(final_production_pipeline, 'model_v2.pkl')
joblib.dump(final_production_pipeline, 'model.pkl')
print(f"Saved Production Pipeline to 'model_v2.pkl' and 'model.pkl'.")

# 10. Export Comprehensive Metadata
metadata = {
    'model_name': best_model_name,
    'version': '2.1.0',
    'training_date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'target_variable': target,
    'target_description': 'Remaining driving range in km under current State of Charge and trip operating conditions',
    'dataset_description': 'Physics-grounded empirical simulation dataset based on verified OEM BEV specifications and standardized vehicle dynamics equations',
    'numerical_features': numerical_features,
    'categorical_features': categorical_features,
    'feature_bounds': feature_bounds,
    'categorical_options': categorical_options,
    'random_split_metrics': {
        k: {m: v[m] for m in ['MAE', 'MSE', 'RMSE', 'R2', 'MAPE']} for k, v in results_random.items()
    },
    'group_split_metrics': results_group,
    'group_kfold_cv_metrics': results_cv,
    'winning_metrics': {
        'model_name': best_model_name,
        'random_split_r2': results_random[best_model_name]['R2'],
        'random_split_rmse_km': results_random[best_model_name]['RMSE'],
        'random_split_mae_km': results_random[best_model_name]['MAE'],
        'random_split_mape_pct': results_random[best_model_name]['MAPE'],
        'group_split_r2': results_group[best_model_name]['R2'],
        'group_split_rmse_km': results_group[best_model_name]['RMSE'],
        'group_cv_r2_mean': results_cv[best_model_name]['mean_r2'],
        'residual_std_error_km': results_random[best_model_name]['RMSE']
    },
    'total_samples': int(df.shape[0]),
    'train_samples_random': int(X_train.shape[0]),
    'test_samples_random': int(X_test.shape[0]),
    'dataset_file': 'real_ev_dataset.csv'
}

with open('model_metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)
print("Updated 'model_metadata.json' with comprehensive scientific validation metrics.")

# 11. Regenerate Evaluation Figures with Honest Terminology
os.makedirs('static/images', exist_ok=True)

# Chart 1: Model Comparison (R² and RMSE)
names = list(results_random.keys())
r2_vals = [results_random[n]['R2'] for n in names]
rmse_vals = [results_random[n]['RMSE'] for n in names]

fig, ax1 = plt.subplots(figsize=(10, 5))
color = '#0284c7'
ax1.set_xlabel('Regression Model Candidate', fontweight='bold', fontsize=11)
ax1.set_ylabel('Validation R² Score', color=color, fontweight='bold', fontsize=11)
bars = ax1.bar(names, r2_vals, color=color, alpha=0.75, width=0.45, align='center', label='Validation R²')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(0.7, 1.02)
plt.xticks(rotation=20, ha='right', fontsize=9.5)

ax2 = ax1.twinx()
color = '#dc2626'
ax2.set_ylabel('RMSE in km (Lower is Better)', color=color, fontweight='bold', fontsize=11)
line = ax2.plot(names, rmse_vals, color=color, marker='o', linewidth=2.5, markersize=8, label='RMSE (km)')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0, max(rmse_vals) * 1.25)

plt.title('Multi-Algorithm Regression Benchmark (Held-Out Evaluation Split)', fontsize=12, fontweight='bold', pad=14)
fig.tight_layout()
chart_comp_path = os.path.join('static', 'images', 'model_comparison.png')
plt.savefig(chart_comp_path, dpi=300)
plt.close()

# Chart 2: Actual vs Predicted Range
y_best_pred = results_random[best_model_name]['y_pred']
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_best_pred, color='#0284c7', alpha=0.55, edgecolors='k', s=38, label='Validation Observations')
min_val = min(y_test.min(), y_best_pred.min())
max_val = max(y_test.max(), y_best_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], color='#dc2626', linestyle='--', linewidth=2.2, label='Ideal 1:1 Parity Line')
plt.title(f'Observed vs Predicted Range ({best_model_name} — R² = {results_random[best_model_name]["R2"]})', fontsize=12, fontweight='bold', pad=12)
plt.xlabel('Ground-Truth Simulated Range (km)', fontsize=11)
plt.ylabel('Model Predicted Range (km)', fontsize=11)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
chart_fit_path = os.path.join('static', 'images', 'actual_vs_predicted.png')
plt.savefig(chart_fit_path, dpi=300)
plt.close()

# Chart 3: Feature Importance
if hasattr(final_production_pipeline.named_steps['regressor'], 'feature_importances_'):
    imp_model = final_production_pipeline.named_steps['regressor']
else:
    imp_model = results_random['Random Forest']['pipeline'].named_steps['regressor']

cat_encoder = final_production_pipeline.named_steps['preprocessor'].named_transformers_['cat']
cat_encoded_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
all_features = numerical_features + cat_encoded_names

importances = imp_model.feature_importances_
top_idx = np.argsort(importances)[-12:]

plt.figure(figsize=(9, 6))
plt.barh(range(len(top_idx)), importances[top_idx], color='#0d9488', align='center', edgecolor='#0f172a', alpha=0.85)
plt.yticks(range(len(top_idx)), [all_features[i] for i in top_idx], fontsize=10)
plt.xlabel('Relative Feature Importance Score', fontsize=11)
plt.title(f'Top Predictor Importance Ranking ({best_model_name})', fontsize=12, fontweight='bold', pad=12)
plt.grid(axis='x', linestyle=':', alpha=0.6)
plt.tight_layout()
chart_imp_path = os.path.join('static', 'images', 'feature_importance.png')
plt.savefig(chart_imp_path, dpi=300)
plt.close()

print("All benchmark visualizations updated successfully.")
print("=" * 80)
