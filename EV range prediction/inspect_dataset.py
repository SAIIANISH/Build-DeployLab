import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Load dataset
df = pd.read_csv('dataset.csv')

print("="*60)
print("1. DATASET SHAPE")
print("="*60)
print(f"Total Rows (Records): {df.shape[0]}")
print(f"Total Columns (Features + Target): {df.shape[1]}")

print("\n" + "="*60)
print("2. COLUMN NAMES & DATA TYPES")
print("="*60)
print(df.dtypes)

print("\n" + "="*60)
print("3. FIRST 5 SAMPLE ROWS")
print("="*60)
print(df.head())

print("\n" + "="*60)
print("4. MISSING VALUES CHECK (NaNs)")
print("="*60)
print(df.isnull().sum())

print("\n" + "="*60)
print("5. DUPLICATE ROWS CHECK")
print("="*60)
print(f"Number of duplicate rows: {df.duplicated().sum()}")

print("\n" + "="*60)
print("6. CATEGORICAL & NUMERICAL COLUMNS")
print("="*60)
cat_cols = [c for c in df.columns if df[c].dtype == 'object' or df[c].dtype == 'string' or str(df[c].dtype) == 'str']
num_cols = [c for c in df.columns if c not in cat_cols]
print(f"Numerical Columns: {num_cols}")
print(f"Categorical Columns: {cat_cols}")
if cat_cols:
    for col in cat_cols:
        print(f"  Values in '{col}': {df[col].unique().tolist()}")

print("\n" + "="*60)
print("7. SUMMARY STATISTICS")
print("="*60)
print(df.describe().round(2))
