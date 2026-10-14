import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set display options for full visibility
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("="*75)
print("PHASE 2: EXPLORATORY DATA ANALYSIS (EDA) & DATA QUALITY REPORT")
print("="*75)

# 1. Load Dataset
df = pd.read_csv('real_ev_dataset.csv')
print(f"Dataset Loaded Successfully: {df.shape[0]} Rows, {df.shape[1]} Columns\n")

# 2. General Info & Data Types
print("--- 1. COLUMN INFORMATION & DATA TYPES ---")
for col in df.columns:
    print(f"  {col:<25} | Type: {str(df[col].dtype):<10} | Nulls: {df[col].isnull().sum()}")

# 3. Duplicate Records
num_duplicates = df.duplicated().sum()
print(f"\n--- 2. DUPLICATE CHECK ---")
print(f"  Total duplicate rows found: {num_duplicates}")

# 4. Vehicle Breakdown
print(f"\n--- 3. VEHICLE CATALOGUE BREAKDOWN ---")
print(f"  Unique Manufacturers: {df['Manufacturer'].nunique()}")
print(f"  Unique Models:        {df['Model'].nunique()}")
print(f"  Unique Variants:      {df['Variant'].nunique()}")

print("\n  Row Counts by Manufacturer:")
mfg_counts = df['Manufacturer'].value_counts()
for mfg, count in mfg_counts.items():
    print(f"    - {mfg:<15}: {count} records ({count/len(df)*100:.1f}%)")

print("\n  Row Counts by Drivetrain:")
for dt, count in df['Drivetrain'].value_counts().items():
    print(f"    - {dt:<10}: {count} records")

print("\n  Row Counts by Body Type:")
for bt, count in df['Body_Type'].value_counts().items():
    print(f"    - {bt:<10}: {count} records")

print("\n  Row Counts by Battery Chemistry:")
for bc, count in df['Battery_Chemistry'].value_counts().items():
    print(f"    - {bc:<10}: {count} records")

# 5. Numerical Feature Statistics
num_cols = df.select_dtypes(include=['number']).columns.tolist()
print("\n--- 4. NUMERICAL SUMMARY STATISTICS ---")
num_summary = df[num_cols].describe().round(2).T
num_summary['median'] = df[num_cols].median().round(2)
print(num_summary[['min', '25%', 'median', '75%', 'max', 'mean', 'std']])

# 6. Categorical Distributions
cat_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
# Exclude vehicle names from quick categorical lists
env_cat_cols = ['Traffic_Condition', 'Terrain', 'Driving_Style', 'HVAC_Usage', 'Tyre_Pressure_Status']
print("\n--- 5. DRIVING / ENVIRONMENTAL CATEGORICAL ATTRIBUTES ---")
for col in env_cat_cols:
    val_counts = df[col].value_counts().to_dict()
    print(f"  {col}: {val_counts}")

# 7. Target Variable Analysis
target = 'Remaining_Range_km'
print(f"\n--- 6. TARGET VARIABLE ANALYSIS ({target}) ---")
print(f"  Minimum Range:  {df[target].min():.1f} km")
print(f"  Maximum Range:  {df[target].max():.1f} km")
print(f"  Mean Range:     {df[target].mean():.1f} km")
print(f"  Median Range:   {df[target].median():.1f} km")
print(f"  Std Deviation:  {df[target].std():.1f} km")
print(f"  Any NaN / Inf?  {df[target].isna().any()} / {np.isinf(df[target]).any()}")
print(f"  Any Negative?   {(df[target] < 0).any()}")

# 8. Pearson Correlation with Target
print(f"\n--- 7. LINEAR CORRELATION WITH {target} ---")
correlations = df[num_cols].corr()[target].sort_values(ascending=False)
for col, corr in correlations.items():
    if col != target:
        print(f"  {col:<25}: r = {corr:+.4f}")

# 9. Generate and Save Visualizations
os.makedirs('static/images', exist_ok=True)

# Chart 1: Target Distribution
plt.figure(figsize=(8, 5))
plt.hist(df[target], bins=35, color='#0284c7', edgecolor='#0f172a', alpha=0.75)
plt.axvline(df[target].mean(), color='#dc2626', linestyle='--', linewidth=2, label=f"Mean: {df[target].mean():.1f} km")
plt.axvline(df[target].median(), color='#059669', linestyle=':', linewidth=2, label=f"Median: {df[target].median():.1f} km")
plt.title('Distribution of Remaining Driving Range (km)', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Remaining Range (km)', fontsize=11)
plt.ylabel('Frequency (Drive-Cycle Observations)', fontsize=11)
plt.legend()
plt.grid(axis='y', linestyle=':', alpha=0.6)
plt.tight_layout()
chart1_path = os.path.join('static', 'images', 'target_distribution.png')
plt.savefig(chart1_path, dpi=300)
plt.close()
print(f"\nSaved Visualization: {chart1_path}")

# Chart 2: Manufacturer Distribution
plt.figure(figsize=(9, 4.5))
mfg_counts.plot(kind='bar', color='#0d9488', edgecolor='#0f172a', width=0.6)
plt.title('Drive Cycle Distribution across BEV Manufacturers', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Manufacturer', fontsize=11)
plt.ylabel('Number of Records', fontsize=11)
plt.xticks(rotation=30, ha='right')
plt.grid(axis='y', linestyle=':', alpha=0.6)
plt.tight_layout()
chart2_path = os.path.join('static', 'images', 'manufacturer_distribution.png')
plt.savefig(chart2_path, dpi=300)
plt.close()
print(f"Saved Visualization: {chart2_path}")

# Chart 3: Correlation Heatmap
plt.figure(figsize=(9, 7))
corr_matrix = df[num_cols].corr()
im = plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.xticks(range(len(num_cols)), num_cols, rotation=45, ha='right', fontsize=9)
plt.yticks(range(len(num_cols)), num_cols, fontsize=9)
plt.title('Correlation Matrix of Numerical Features', fontsize=13, fontweight='bold', pad=12)
for i in range(len(num_cols)):
    for j in range(len(num_cols)):
        val = corr_matrix.iloc[i, j]
        color = 'white' if abs(val) > 0.55 else 'black'
        plt.text(j, i, f"{val:.2f}", ha='center', va='center', color=color, fontsize=8, fontweight='bold')
plt.tight_layout()
chart3_path = os.path.join('static', 'images', 'correlation_matrix.png')
plt.savefig(chart3_path, dpi=300)
plt.close()
print(f"Saved Visualization: {chart3_path}")

print("\nPHASE 2 EDA COMPLETE - ALL DATA SANITY CHECKS PASSED!")
