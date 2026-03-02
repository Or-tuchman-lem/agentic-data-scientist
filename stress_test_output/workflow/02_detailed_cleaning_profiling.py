"""
Stage 1 (Part 2): Detailed Data Cleaning and Feature Profiling

This script performs detailed data cleaning, feature profiling, and generates
comprehensive analysis of all features in the dataset.

Author: Agentic Data Scientist
Date: 2026-02-26
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

# Configure plotting
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"

print("=" * 80)
print("STAGE 1 (PART 2): DETAILED CLEANING AND FEATURE PROFILING")
print("=" * 80)
print()

# ============================================================================
# LOAD DATA FROM PREVIOUS STEP
# ============================================================================
print("Loading data from previous step...")
df = pd.read_csv(DATA_DIR / '01_cleaned_data.csv')
print(f"✓ Loaded {len(df):,} rows and {len(df.columns)} columns")
print()

# ============================================================================
# STEP 1: HANDLE MISSING VALUES
# ============================================================================
print("STEP 1: Handling missing values...")

# Drop SOURCE column (99.66% missing, not useful)
if 'SOURCE' in df.columns:
    df = df.drop('SOURCE', axis=1)
    print("✓ Dropped SOURCE column (99.66% missing)")

# Drop duplicate STATE.1 column (duplicate of STATE)
if 'STATE.1' in df.columns:
    df = df.drop('STATE.1', axis=1)
    print("✓ Dropped STATE.1 column (duplicate of STATE)")

# Handle IMPUTED_AGE missing values (1.1% missing)
imputed_age_missing = df['IMPUTED_AGE'].isnull().sum()
if imputed_age_missing > 0:
    # Fill with median age
    median_age = df['IMPUTED_AGE'].median()
    df['IMPUTED_AGE'] = df['IMPUTED_AGE'].fillna(median_age)
    print(f"✓ Filled {imputed_age_missing} missing IMPUTED_AGE values with median: {median_age}")

# Handle MEDIAN_HOUSEHOLD_INCOME_2020 missing values (3.1%)
income_missing = df['MEDIAN_HOUSEHOLD_INCOME_2020'].isnull().sum()
if income_missing > 0:
    # Fill with state median
    state_median_income = df.groupby('STATE')['MEDIAN_HOUSEHOLD_INCOME_2020'].transform('median')
    df['MEDIAN_HOUSEHOLD_INCOME_2020'] = df['MEDIAN_HOUSEHOLD_INCOME_2020'].fillna(state_median_income)
    # If still missing, fill with overall median
    overall_median = df['MEDIAN_HOUSEHOLD_INCOME_2020'].median()
    df['MEDIAN_HOUSEHOLD_INCOME_2020'] = df['MEDIAN_HOUSEHOLD_INCOME_2020'].fillna(overall_median)
    print(f"✓ Filled {income_missing} missing MEDIAN_HOUSEHOLD_INCOME_2020 values")

# Handle TOTAL_VET_CLINICS missing values (6.9%)
vet_missing = df['TOTAL_VET_CLINICS'].isnull().sum()
if vet_missing > 0:
    # Fill with state median
    state_median_vets = df.groupby('STATE')['TOTAL_VET_CLINICS'].transform('median')
    df['TOTAL_VET_CLINICS'] = df['TOTAL_VET_CLINICS'].fillna(state_median_vets)
    # If still missing, fill with overall median
    overall_median_vets = df['TOTAL_VET_CLINICS'].median()
    df['TOTAL_VET_CLINICS'] = df['TOTAL_VET_CLINICS'].fillna(overall_median_vets)
    print(f"✓ Filled {vet_missing} missing TOTAL_VET_CLINICS values")

print(f"\n✓ Missing values after cleaning: {df.isnull().sum().sum()}")
print()

# ============================================================================
# STEP 2: FEATURE CATEGORIZATION
# ============================================================================
print("STEP 2: Categorizing features for analysis...")

# Categorize features by type and relevance
feature_categories = {
    'target_variables': ['is_purchased', 'premium_amount', 'CONVERTED', 'SALES'],

    'treatment_assigned': [
        'COVERAGE_TREATMENT', 'TREATMENT_COINSURANCE',
        'TREATMENT_DEDUCTIBLE', 'TREATMENT_COVERAGE_LIMIT'
    ],

    'forward_looking': [
        'FINAL_COINSURANCE_PERCENT', 'FINAL_BASE_DEDUCTIBLE',
        'FINAL_COVERAGE_LIMIT', 'FINAL_ANNUAL_PREMIUM'
    ],

    'pricing_features': [
        'PIT_ANNUAL_PREMIUM', 'BASE_PREMIUM'
    ],

    'customer_demographics': [
        'IMPUTED_AGE', 'STATE', 'HAS_DEBIT_CARD'
    ],

    'customer_network': [
        'HAS_STRONGLY_CONNECTED_USERS', 'HAS_MULTIPLE_PET_DISCOUNT'
    ],

    'pet_characteristics': [
        'PET_BREED_CLEAN', 'PET_AGE_YEARS', 'DESIGNER_BREED', 'PET_SEX'
    ],

    'location_features': [
        'MEDIAN_HOUSEHOLD_INCOME_2020', 'TOTAL_VET_CLINICS'
    ],

    'temporal_features': [
        'COVERAGE_SENT_AT', 'QUOTE_MONTH_COS'
    ],

    'identifiers': [
        'ENCRYPTED_USER_ID', 'QUOTE_ID', 'ENCRYPTED_QUOTE_ID',
        'PET_LAST_NAME', 'EMAIL_END'
    ]
}

print("Feature categories:")
for category, features in feature_categories.items():
    existing_features = [f for f in features if f in df.columns]
    print(f"  {category}: {len(existing_features)} features")

print()

# ============================================================================
# STEP 3: CATEGORICAL FEATURE PROFILING
# ============================================================================
print("STEP 3: Profiling categorical features...")

categorical_features = df.select_dtypes(include=['object', 'bool']).columns.tolist()
# Remove identifiers and forward-looking
categorical_features = [c for c in categorical_features
                       if c not in feature_categories['identifiers']
                       and c not in feature_categories['forward_looking']]

categorical_profile = []

for col in categorical_features:
    value_counts = df[col].value_counts()
    profile = {
        'feature': col,
        'unique_values': int(df[col].nunique()),
        'most_common': str(value_counts.index[0]) if len(value_counts) > 0 else None,
        'most_common_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
        'most_common_pct': float(value_counts.iloc[0] / len(df) * 100) if len(value_counts) > 0 else 0,
        'top_5_values': value_counts.head(5).to_dict()
    }
    categorical_profile.append(profile)

cat_profile_df = pd.DataFrame(categorical_profile)
cat_profile_df.to_csv(RESULTS_DIR / '02_categorical_feature_profile.csv', index=False)
print(f"✓ Saved categorical feature profile: {len(categorical_features)} features")
print("\nTop categorical features by unique values:")
print(cat_profile_df.sort_values('unique_values', ascending=False).head(10)[
    ['feature', 'unique_values', 'most_common', 'most_common_pct']
].to_string(index=False))
print()

# ============================================================================
# STEP 4: NUMERICAL FEATURE PROFILING
# ============================================================================
print("STEP 4: Profiling numerical features...")

numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
# Remove identifiers and forward-looking
numerical_features = [c for c in numerical_features
                     if c not in feature_categories['identifiers']
                     and c not in feature_categories['forward_looking']]

numerical_profile = []

for col in numerical_features:
    stats = df[col].describe()
    profile = {
        'feature': col,
        'count': int(stats['count']),
        'mean': float(stats['mean']),
        'std': float(stats['std']),
        'min': float(stats['min']),
        'q25': float(stats['25%']),
        'median': float(stats['50%']),
        'q75': float(stats['75%']),
        'max': float(stats['max']),
        'zeros_count': int((df[col] == 0).sum()),
        'zeros_pct': float((df[col] == 0).mean() * 100),
        'cv': float(stats['std'] / stats['mean']) if stats['mean'] != 0 else np.inf
    }
    numerical_profile.append(profile)

num_profile_df = pd.DataFrame(numerical_profile)
num_profile_df.to_csv(RESULTS_DIR / '02_numerical_feature_profile.csv', index=False)
print(f"✓ Saved numerical feature profile: {len(numerical_features)} features")
print("\nNumerical features summary:")
print(num_profile_df[['feature', 'mean', 'std', 'min', 'max', 'zeros_pct']].to_string(index=False))
print()

# ============================================================================
# STEP 5: FEATURE DISTRIBUTIONS BY TARGET
# ============================================================================
print("STEP 5: Analyzing feature distributions by target (is_purchased)...")

# Conversion rates by categorical features
conversion_by_category = {}

for col in categorical_features:
    if df[col].nunique() < 50:  # Only for features with reasonable number of categories
        conv_rates = df.groupby(col)['is_purchased'].agg(['sum', 'count', 'mean'])
        conv_rates.columns = ['purchases', 'total_quotes', 'conversion_rate']
        conv_rates = conv_rates.sort_values('conversion_rate', ascending=False)
        conversion_by_category[col] = conv_rates.to_dict()

# Save conversion analysis
with open(RESULTS_DIR / '02_conversion_by_category.json', 'w') as f:
    json.dump(conversion_by_category, f, indent=2)
print(f"✓ Saved conversion analysis by categorical features")

# Numerical feature statistics by purchase status
numeric_by_target = {}

for col in numerical_features:
    if col not in ['is_purchased', 'CONVERTED']:
        purchased = df[df['is_purchased'] == 1][col]
        not_purchased = df[df['is_purchased'] == 0][col]

        numeric_by_target[col] = {
            'purchased': {
                'mean': float(purchased.mean()),
                'median': float(purchased.median()),
                'std': float(purchased.std())
            },
            'not_purchased': {
                'mean': float(not_purchased.mean()),
                'median': float(not_purchased.median()),
                'std': float(not_purchased.std())
            },
            'mean_diff': float(purchased.mean() - not_purchased.mean()),
            'mean_diff_pct': float((purchased.mean() - not_purchased.mean()) / not_purchased.mean() * 100)
                            if not_purchased.mean() != 0 else np.inf
        }

with open(RESULTS_DIR / '02_numeric_by_target.json', 'w') as f:
    json.dump(numeric_by_target, f, indent=2)
print(f"✓ Saved numerical feature analysis by target")
print()

# ============================================================================
# STEP 6: VARIANT PERFORMANCE ANALYSIS
# ============================================================================
print("STEP 6: Analyzing treatment variant performance...")

# Calculate metrics by variant
variant_performance = df.groupby('COVERAGE_TREATMENT').agg({
    'QUOTE_ID': 'count',  # Number of quotes
    'is_purchased': ['sum', 'mean'],  # Purchases and conversion rate
    'premium_amount': ['sum', 'mean']  # Total sales and avg premium
}).round(4)

variant_performance.columns = ['quotes', 'purchases', 'conversion_rate', 'total_sales', 'avg_premium']

# Calculate average sales per quote (the target metric)
variant_performance['avg_sales_per_quote'] = (
    variant_performance['total_sales'] / variant_performance['quotes']
).round(2)

# Parse variant components for easier analysis
variant_performance = variant_performance.reset_index()
variant_performance[['coinsurance', 'deductible', 'coverage_limit']] = (
    variant_performance['COVERAGE_TREATMENT'].str.split('_', expand=True).astype(int)
)

# Sort by performance metric
variant_performance = variant_performance.sort_values('avg_sales_per_quote', ascending=False)

variant_performance.to_csv(RESULTS_DIR / '02_variant_performance.csv', index=False)
print(f"✓ Saved variant performance analysis")
print("\nTop 10 variants by avg_sales_per_quote:")
print(variant_performance[['COVERAGE_TREATMENT', 'quotes', 'conversion_rate',
                          'avg_sales_per_quote']].head(10).to_string(index=False))
print("\nBottom 10 variants by avg_sales_per_quote:")
print(variant_performance[['COVERAGE_TREATMENT', 'quotes', 'conversion_rate',
                          'avg_sales_per_quote']].tail(10).to_string(index=False))
print()

# Analyze current production variant
prod_variant = '80_250_20000'
if prod_variant in variant_performance['COVERAGE_TREATMENT'].values:
    prod_perf = variant_performance[variant_performance['COVERAGE_TREATMENT'] == prod_variant].iloc[0]
    print(f"Current production variant: {prod_variant}")
    print(f"  Quotes: {prod_perf['quotes']}")
    print(f"  Conversion rate: {prod_perf['conversion_rate']:.4f}")
    print(f"  Avg sales per quote: ${prod_perf['avg_sales_per_quote']:.2f}")
    print(f"  Rank: {variant_performance[variant_performance['COVERAGE_TREATMENT'] == prod_variant].index[0] + 1} / 27")
    print()

# ============================================================================
# STEP 7: SAVE CLEANED DATASET
# ============================================================================
print("STEP 7: Saving final cleaned dataset...")

# Create final feature set (excluding identifiers and forward-looking)
features_to_exclude = (
    feature_categories['identifiers'] +
    feature_categories['forward_looking'] +
    ['variant_check']  # Remove helper column
)
features_to_exclude = [f for f in features_to_exclude if f in df.columns]

df_clean = df.drop(features_to_exclude, axis=1)

print(f"Original shape: {df.shape}")
print(f"Cleaned shape: {df_clean.shape}")
print(f"Removed {len(features_to_exclude)} columns: {features_to_exclude}")
print()

# Save cleaned dataset
df_clean.to_csv(DATA_DIR / '02_cleaned_features.csv', index=False)
print(f"✓ Saved cleaned dataset: data/02_cleaned_features.csv")

# Save feature list
feature_list = {
    'total_features': len(df_clean.columns),
    'target_variables': [c for c in feature_categories['target_variables'] if c in df_clean.columns],
    'treatment_features': [c for c in feature_categories['treatment_assigned'] if c in df_clean.columns],
    'predictive_features': [c for c in df_clean.columns
                           if c not in feature_categories['target_variables']
                           and c not in feature_categories['treatment_assigned']],
    'excluded_features': features_to_exclude
}

with open(RESULTS_DIR / '02_feature_list.json', 'w') as f:
    json.dump(feature_list, f, indent=2)
print(f"✓ Saved feature list: results/02_feature_list.json")
print()

print("=" * 80)
print("DETAILED CLEANING AND PROFILING COMPLETE")
print("=" * 80)
print(f"\nFinal dataset: {df_clean.shape[0]:,} rows × {df_clean.shape[1]} columns")
print(f"Target: is_purchased (conversion rate: {df_clean['is_purchased'].mean()*100:.2f}%)")
print(f"Metric: Average sales per quote across 27 variants")
