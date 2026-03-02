"""
Stage 1: Data Loading, Initial Profiling, and Cleaning

This script performs comprehensive data loading, profiling, and cleaning for the
pet insurance recommendation system EDA.

Author: Agentic Data Scientist
Date: 2026-02-26
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Configure plotting
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = (12, 8)

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_FILE = BASE_DIR / "user_data" / "ds_agent_test_set.csv"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
DATA_DIR = BASE_DIR / "data"

# Create directories
for dir_path in [RESULTS_DIR, FIGURES_DIR, DATA_DIR]:
    dir_path.mkdir(exist_ok=True)

print("=" * 80)
print("STAGE 1: DATA LOADING, INITIAL PROFILING, AND CLEANING")
print("=" * 80)
print()

# ============================================================================
# STEP 1: DATA LOADING
# ============================================================================
print("STEP 1: Loading data...")
print(f"Data file: {DATA_FILE}")
print(f"File size: {DATA_FILE.stat().st_size / (1024*1024):.2f} MB")

# Load data in chunks to avoid memory issues (file is 12MB)
df = pd.read_csv(DATA_FILE)
print(f"✓ Loaded {len(df):,} rows and {len(df.columns)} columns")
print()

# ============================================================================
# STEP 2: INITIAL DATA STRUCTURE INSPECTION
# ============================================================================
print("STEP 2: Initial data structure inspection...")
print("\nColumn names:")
print(df.columns.tolist())
print()

print("Data types:")
print(df.dtypes)
print()

print("First 3 rows:")
print(df.head(3))
print()

print("Basic statistics:")
print(df.describe())
print()

# ============================================================================
# STEP 3: DATA QUALITY CHECKS
# ============================================================================
print("STEP 3: Data quality checks...")

# Check for missing values
print("\n3.1 Missing values analysis:")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'Column': missing.index,
    'Missing_Count': missing.values,
    'Missing_Percent': missing_pct.values
})
missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)

if len(missing_df) > 0:
    print(f"Found missing values in {len(missing_df)} columns:")
    print(missing_df.to_string(index=False))
else:
    print("✓ No missing values found")
print()

# Check for duplicate rows
print("3.2 Duplicate rows analysis:")
duplicate_count = df.duplicated().sum()
print(f"Total duplicate rows: {duplicate_count:,} ({duplicate_count/len(df)*100:.2f}%)")
if duplicate_count > 0:
    print("Sample duplicate rows:")
    print(df[df.duplicated(keep=False)].head(10))
print()

# Check for duplicate columns
print("3.3 Duplicate column names:")
duplicate_cols = [col for col in df.columns if df.columns.tolist().count(col) > 1]
if duplicate_cols:
    print(f"⚠ Found duplicate column names: {set(duplicate_cols)}")
    # Keep only the first occurrence
    df = df.loc[:, ~df.columns.duplicated()]
    print(f"✓ Removed duplicate columns. Now have {len(df.columns)} unique columns")
else:
    print("✓ No duplicate column names")
print()

# Check data type consistency
print("3.4 Data type consistency checks:")
print("\nNumeric columns that might need conversion:")
for col in df.columns:
    if df[col].dtype == 'object':
        # Try to identify numeric columns stored as strings
        try:
            numeric_test = pd.to_numeric(df[col].head(100), errors='coerce')
            if numeric_test.notna().sum() > 50:  # If >50% are numeric
                print(f"  - {col}: appears to contain numeric values")
        except:
            pass
print()

# ============================================================================
# STEP 4: TREATMENT VARIANT ANALYSIS
# ============================================================================
print("STEP 4: Treatment variant analysis...")

# Parse treatment variant
print("\n4.1 Treatment variant distribution:")
print(df['COVERAGE_TREATMENT'].value_counts().sort_index())
print()

# Verify treatment components match the variant name
print("4.2 Verifying treatment components match variant name...")
df['variant_check'] = (
    df['TREATMENT_COINSURANCE'].astype(str) + '_' +
    df['TREATMENT_DEDUCTIBLE'].astype(str) + '_' +
    df['TREATMENT_COVERAGE_LIMIT'].astype(str)
)
mismatches = df[df['COVERAGE_TREATMENT'] != df['variant_check']]
if len(mismatches) > 0:
    print(f"⚠ Found {len(mismatches)} rows where COVERAGE_TREATMENT doesn't match components")
    print("Sample mismatches:")
    print(mismatches[['COVERAGE_TREATMENT', 'variant_check',
                      'TREATMENT_COINSURANCE', 'TREATMENT_DEDUCTIBLE',
                      'TREATMENT_COVERAGE_LIMIT']].head())
else:
    print("✓ All treatment variants match their components")
print()

# Check for all 27 expected variants
print("4.3 Expected variants (3 deductibles × 3 limits × 3 coinsurance = 27):")
expected_variants = []
for coinsurance in [70, 80, 90]:
    for deductible in [100, 250, 500]:
        for limit in [10000, 20000, 50000]:
            expected_variants.append(f"{coinsurance}_{deductible}_{limit}")

actual_variants = set(df['COVERAGE_TREATMENT'].unique())
missing_variants = set(expected_variants) - actual_variants
extra_variants = actual_variants - set(expected_variants)

print(f"Expected variants: {len(expected_variants)}")
print(f"Actual variants: {len(actual_variants)}")
if missing_variants:
    print(f"⚠ Missing variants: {missing_variants}")
if extra_variants:
    print(f"⚠ Extra variants: {extra_variants}")
if not missing_variants and not extra_variants:
    print("✓ All 27 expected variants present")
print()

# ============================================================================
# STEP 5: TARGET VARIABLE DEFINITION
# ============================================================================
print("STEP 5: Target variable definition and construction...")

# Analyze conversion and sales columns
print("\n5.1 Conversion and sales analysis:")
print(f"CONVERTED column - unique values: {df['CONVERTED'].unique()}")
print(f"CONVERTED value counts:\n{df['CONVERTED'].value_counts()}")
print()
print(f"SALES column - unique values: {df['SALES'].unique()}")
print(f"SALES value counts:\n{df['SALES'].value_counts()}")
print()

# Check if CONVERTED and SALES are aligned
print("5.2 Checking CONVERTED vs SALES alignment:")
print(pd.crosstab(df['CONVERTED'], df['SALES']))
print()

# Create target variable: is_purchased (binary)
df['is_purchased'] = (df['CONVERTED'] == 1).astype(int)

# Create premium_amount variable (for calculating sales per quote)
# SALES appears to be the premium amount when purchased
df['premium_amount'] = df['SALES']

print("5.3 Target variables created:")
print(f"  - is_purchased: binary indicator (1 = purchased, 0 = not purchased)")
print(f"  - premium_amount: premium amount (equals SALES when purchased)")
print()

print("5.4 Overall conversion rate:")
conversion_rate = df['is_purchased'].mean() * 100
print(f"  Overall conversion: {conversion_rate:.2f}%")
print(f"  Purchased: {df['is_purchased'].sum():,}")
print(f"  Not purchased: {(~df['is_purchased'].astype(bool)).sum():,}")
print()

# ============================================================================
# STEP 6: IDENTIFY IRRELEVANT/FORWARD-LOOKING FEATURES
# ============================================================================
print("STEP 6: Identifying irrelevant or forward-looking features...")

# Features that are clearly forward-looking or not useful for prediction
forward_looking = []
raw_features = []
identifier_features = []

# Analyze each column
for col in df.columns:
    # Check for forward-looking features (final values that customer selected)
    if 'FINAL_' in col:
        forward_looking.append(col)

    # Check for raw identifiers (encrypted IDs, long strings)
    elif col in ['ENCRYPTED_USER_ID', 'QUOTE_ID', 'ENCRYPTED_QUOTE_ID']:
        identifier_features.append(col)

    # Check for very long text fields
    elif df[col].dtype == 'object':
        avg_length = df[col].astype(str).str.len().mean()
        if avg_length > 50:  # Arbitrary threshold
            raw_features.append(col)

print(f"\n6.1 Forward-looking features (should not be used for prediction):")
print(f"  Count: {len(forward_looking)}")
print(f"  Features: {forward_looking}")
print()

print(f"6.2 Identifier features (not useful for modeling):")
print(f"  Count: {len(identifier_features)}")
print(f"  Features: {identifier_features}")
print()

print(f"6.3 Raw text features (may need processing):")
print(f"  Count: {len(raw_features)}")
print(f"  Features: {raw_features}")
print()

# ============================================================================
# STEP 7: SAVE INITIAL PROFILING RESULTS
# ============================================================================
print("STEP 7: Saving initial profiling results...")

# Save data quality report
quality_report = {
    'total_rows': len(df),
    'total_columns': len(df.columns),
    'missing_values': missing_df.to_dict('records') if len(missing_df) > 0 else [],
    'duplicate_rows': int(duplicate_count),
    'duplicate_columns': duplicate_cols,
    'conversion_rate': float(conversion_rate),
    'treatment_variants': {
        'expected': len(expected_variants),
        'actual': len(actual_variants),
        'missing': list(missing_variants),
        'extra': list(extra_variants)
    },
    'feature_categories': {
        'forward_looking': forward_looking,
        'identifiers': identifier_features,
        'raw_text': raw_features
    }
}

with open(RESULTS_DIR / '01_data_quality_report.json', 'w') as f:
    json.dump(quality_report, f, indent=2)
print("✓ Saved data quality report: results/01_data_quality_report.json")

# Save column metadata
column_metadata = []
for col in df.columns:
    col_info = {
        'column_name': col,
        'dtype': str(df[col].dtype),
        'missing_count': int(df[col].isnull().sum()),
        'missing_percent': float(df[col].isnull().mean() * 100),
        'unique_values': int(df[col].nunique()),
        'sample_values': df[col].dropna().head(5).tolist()
    }
    column_metadata.append(col_info)

metadata_df = pd.DataFrame(column_metadata)
metadata_df.to_csv(RESULTS_DIR / '01_column_metadata.csv', index=False)
print("✓ Saved column metadata: results/01_column_metadata.csv")

# Save cleaned dataset
df.to_csv(DATA_DIR / '01_cleaned_data.csv', index=False)
print(f"✓ Saved cleaned dataset: data/01_cleaned_data.csv")
print(f"  Shape: {df.shape}")
print()

print("=" * 80)
print("STEP 1 DATA LOADING AND PROFILING COMPLETE")
print("=" * 80)
