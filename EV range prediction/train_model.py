import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------------------------------------
# 1. LOAD DATASET
# -------------------------------------------------------------
print("="*65)
print("STEP 1: LOADING DATASET")
print("="*65)
df = pd.read_csv('dataset.csv')
print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.\n")

# -------------------------------------------------------------
# 2. FEATURE SELECTION (Separating Features X and Target y)
# -------------------------------------------------------------
print("="*65)
print("STEP 2: FEATURE SELECTION")
print("="*65)
numerical_features = [
    'Battery_Capacity_kWh',
    'Battery_Percentage',
    'Vehicle_Speed_kmh',
    'Vehicle_Weight_kg',
    'Ambient_Temperature_C'
]
categorical_features = ['Driving_Mode']

X = df[numerical_features + categorical_features]
y = df['Range_km']

print("Input Features (X):")
print(f"  - Numerical:   {numerical_features}")
print(f"  - Categorical: {categorical_features}")
print(f"Target Variable (y): Range_km\n")

# -------------------------------------------------------------
# 3. TRAIN-TEST SPLIT (80% Train, 20% Test)
# -------------------------------------------------------------
print("="*65)
print("STEP 3: TRAIN-TEST SPLIT")
print("="*65)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples: {X_train.shape[0]} (80%)")
print(f"Testing samples:  {X_test.shape[0]} (20%)\n")

# -------------------------------------------------------------
# 4. PREPROCESSING PIPELINE
# -------------------------------------------------------------
# We use ColumnTransformer:
# - Standardize numerical features using StandardScaler
# - Encode categorical 'Driving_Mode' using OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ]
)

# -------------------------------------------------------------
# 5. MODEL SELECTION & COMPARISON
# -------------------------------------------------------------
print("="*65)
print("STEP 4: MODEL TRAINING & COMPARISON")
print("="*65)

models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
    'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}
best_model_name = None
best_r2 = -float('inf')
best_pipeline = None

for name, model in models.items():
    # Build complete pipeline (Preprocessing + Model)
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = pipeline.predict(X_test)
    
    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'pipeline': pipeline,
        'y_pred': y_pred
    }
    
    print(f"--- {name} ---")
    print(f"  MAE:       {mae:.2f} km")
    print(f"  MSE:       {mse:.2f}")
    print(f"  RMSE:      {rmse:.2f} km")
    print(f"  R2 Score:  {r2:.4f} ({r2*100:.2f}% variance explained)")
    print()
    
    if r2 > best_r2:
        best_r2 = r2
        best_model_name = name
        best_pipeline = pipeline

print(f"--> BEST MODEL: {best_model_name} with R2 Score = {best_r2:.4f}")

# -------------------------------------------------------------
# 6. SAVE THE BEST MODEL PIPELINE
# -------------------------------------------------------------
print("\n" + "="*65)
print("STEP 5: SAVING BEST MODEL PIPELINE")
print("="*65)
model_filename = 'model.pkl'
joblib.dump(best_pipeline, model_filename)
print(f"Successfully saved {best_model_name} pipeline to '{model_filename}'!")

# -------------------------------------------------------------
# 7. GENERATE VISUALIZATIONS FOR REPORT & DASHBOARD
# -------------------------------------------------------------
print("\n" + "="*65)
print("STEP 6: GENERATING EVALUATION CHARTS")
print("="*65)
os.makedirs('static/images', exist_ok=True)

# Chart 1: Actual vs Predicted Range
best_preds = results[best_model_name]['y_pred']

plt.figure(figsize=(8, 6))
plt.scatter(y_test, best_preds, color='#0284c7', alpha=0.6, edgecolors='k', s=45, label='Predicted Points')
min_val = min(y_test.min(), best_preds.min())
max_val = max(y_test.max(), best_preds.max())
plt.plot([min_val, max_val], [min_val, max_val], color='#dc2626', linestyle='--', linewidth=2, label='Perfect Fit (Ideal)')
plt.title(f'Actual vs Predicted EV Range ({best_model_name})', fontsize=14, fontweight='bold', pad=12)
plt.xlabel('Actual Range (km)', fontsize=12)
plt.ylabel('Predicted Range (km)', fontsize=12)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
chart1_path = os.path.join('static', 'images', 'actual_vs_predicted.png')
plt.savefig(chart1_path, dpi=300)
plt.close()
print(f"Saved: {chart1_path}")

# Chart 2: Feature Importance (From Random Forest)
rf_model = results['Random Forest Regressor']['pipeline'].named_steps['regressor']
# Extract feature names after one-hot encoding
cat_encoder = results['Random Forest Regressor']['pipeline'].named_steps['preprocessor'].named_transformers_['cat']
cat_encoded_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
all_feature_names = numerical_features + cat_encoded_names

importances = rf_model.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(9, 5.5))
plt.barh(range(len(indices)), importances[indices], color='#0d9488', align='center')
plt.yticks(range(len(indices)), [all_feature_names[i] for i in indices], fontsize=11)
plt.xlabel('Relative Importance Score', fontsize=12)
plt.title('Feature Importance for EV Range Prediction', fontsize=14, fontweight='bold', pad=12)
plt.grid(axis='x', linestyle=':', alpha=0.6)
plt.tight_layout()
chart2_path = os.path.join('static', 'images', 'feature_importance.png')
plt.savefig(chart2_path, dpi=300)
plt.close()
print(f"Saved: {chart2_path}")

print("\nMODEL TRAINING & EVALUATION COMPLETED SUCCESSFULLY!")
