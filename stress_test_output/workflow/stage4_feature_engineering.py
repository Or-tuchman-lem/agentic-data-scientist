"""
Stage 4: Feature Engineering and Hypothesis Generation

This script creates new, potentially more predictive features from existing data:
1. Time-based features from timestamps
2. Interaction terms between key variables
3. Binary flags and categorical encodings
4. Non-linear transformations
5. Composite/aggregated features

Author: Agentic Data Scientist
Date: 2026-02-26
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json
from scipy import stats
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("="*80)
print("STAGE 4: FEATURE ENGINEERING AND HYPOTHESIS GENERATION")
print("="*80)
print()

# Load cleaned data
print("[1/8] Loading cleaned data...")
df = pd.read_csv('data/03_cleaned_fixed.csv')
print(f"✓ Loaded: {len(df):,} rows × {len(df.columns)} columns")
print()

# Create a copy for feature engineering
df_eng = df.copy()
original_features = set(df.columns)

# ============================================================================
# PART 1: TIME-BASED FEATURES FROM TIMESTAMP
# ============================================================================
print("[2/8] Creating time-based features from TIMESTAMP...")

# Convert to datetime (COVERAGE_SENT_AT is the timestamp column)
df_eng['TIMESTAMP_DT'] = pd.to_datetime(df_eng['COVERAGE_SENT_AT'])

# Extract temporal components
df_eng['QUOTE_YEAR'] = df_eng['TIMESTAMP_DT'].dt.year
df_eng['QUOTE_MONTH'] = df_eng['TIMESTAMP_DT'].dt.month
df_eng['QUOTE_DAY'] = df_eng['TIMESTAMP_DT'].dt.day
df_eng['QUOTE_DAYOFWEEK'] = df_eng['TIMESTAMP_DT'].dt.dayofweek  # 0=Monday, 6=Sunday
df_eng['QUOTE_HOUR'] = df_eng['TIMESTAMP_DT'].dt.hour
df_eng['QUOTE_DAYOFYEAR'] = df_eng['TIMESTAMP_DT'].dt.dayofyear

# Business-relevant temporal features
df_eng['IS_WEEKEND'] = (df_eng['QUOTE_DAYOFWEEK'] >= 5).astype(int)
df_eng['IS_BUSINESS_HOURS'] = ((df_eng['QUOTE_HOUR'] >= 9) & (df_eng['QUOTE_HOUR'] <= 17)).astype(int)
df_eng['IS_EVENING'] = ((df_eng['QUOTE_HOUR'] >= 18) & (df_eng['QUOTE_HOUR'] <= 23)).astype(int)
df_eng['IS_MONTH_START'] = (df_eng['QUOTE_DAY'] <= 5).astype(int)  # First 5 days
df_eng['IS_MONTH_END'] = (df_eng['QUOTE_DAY'] >= 26).astype(int)  # Last ~5 days

# Quarter and season
df_eng['QUOTE_QUARTER'] = df_eng['TIMESTAMP_DT'].dt.quarter
df_eng['QUOTE_SEASON'] = df_eng['QUOTE_MONTH'].map({
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Fall', 10: 'Fall', 11: 'Fall'
})

# Days since campaign start (May 30, 2025)
campaign_start = pd.Timestamp('2025-05-30')
df_eng['DAYS_SINCE_CAMPAIGN_START'] = (df_eng['TIMESTAMP_DT'] - campaign_start).dt.days

time_features = [
    'QUOTE_YEAR', 'QUOTE_MONTH', 'QUOTE_DAY', 'QUOTE_DAYOFWEEK', 'QUOTE_HOUR',
    'QUOTE_DAYOFYEAR', 'IS_WEEKEND', 'IS_BUSINESS_HOURS', 'IS_EVENING',
    'IS_MONTH_START', 'IS_MONTH_END', 'QUOTE_QUARTER', 'QUOTE_SEASON',
    'DAYS_SINCE_CAMPAIGN_START'
]

print(f"✓ Created {len(time_features)} time-based features:")
for f in time_features:
    print(f"  - {f}")
print()

# ============================================================================
# PART 2: INTERACTION TERMS BETWEEN KEY PREDICTORS
# ============================================================================
print("[3/8] Generating interaction terms between key predictors...")

interactions_created = []

# Top predictors from Stage 3 analysis:
# 1. HAS_MULTIPLE_PET_DISCOUNT (strongest)
# 2. HAS_DEBIT_CARD
# 3. MEDIAN_HOUSEHOLD_INCOME_2020 (income)
# 4. HAS_STRONGLY_CONNECTED_USERS
# 5. IMPUTED_AGE (age)

# Binary × Binary interactions
df_eng['MULTIPET_X_DEBIT'] = df_eng['HAS_MULTIPLE_PET_DISCOUNT'] * df_eng['HAS_DEBIT_CARD']
interactions_created.append('MULTIPET_X_DEBIT')

df_eng['MULTIPET_X_CONNECTED'] = df_eng['HAS_MULTIPLE_PET_DISCOUNT'] * df_eng['HAS_STRONGLY_CONNECTED_USERS']
interactions_created.append('MULTIPET_X_CONNECTED')

df_eng['DEBIT_X_CONNECTED'] = df_eng['HAS_DEBIT_CARD'] * df_eng['HAS_STRONGLY_CONNECTED_USERS']
interactions_created.append('DEBIT_X_CONNECTED')

# Binary × Continuous interactions
# Income interactions (normalized for interpretability)
df_eng['INCOME_NORMALIZED'] = (df_eng['MEDIAN_HOUSEHOLD_INCOME_2020'] - df_eng['MEDIAN_HOUSEHOLD_INCOME_2020'].mean()) / df_eng['MEDIAN_HOUSEHOLD_INCOME_2020'].std()

df_eng['MULTIPET_X_INCOME'] = df_eng['HAS_MULTIPLE_PET_DISCOUNT'] * df_eng['INCOME_NORMALIZED']
interactions_created.append('MULTIPET_X_INCOME')

df_eng['DEBIT_X_INCOME'] = df_eng['HAS_DEBIT_CARD'] * df_eng['INCOME_NORMALIZED']
interactions_created.append('DEBIT_X_INCOME')

# Age interactions
df_eng['AGE_NORMALIZED'] = (df_eng['IMPUTED_AGE'] - df_eng['IMPUTED_AGE'].mean()) / df_eng['IMPUTED_AGE'].std()

df_eng['MULTIPET_X_AGE'] = df_eng['HAS_MULTIPLE_PET_DISCOUNT'] * df_eng['AGE_NORMALIZED']
interactions_created.append('MULTIPET_X_AGE')

# Premium interactions (price sensitivity varies by segment)
df_eng['PREMIUM_NORMALIZED'] = (df_eng['PIT_ANNUAL_PREMIUM'] - df_eng['PIT_ANNUAL_PREMIUM'].mean()) / df_eng['PIT_ANNUAL_PREMIUM'].std()

df_eng['INCOME_X_PREMIUM'] = df_eng['INCOME_NORMALIZED'] * df_eng['PREMIUM_NORMALIZED']
interactions_created.append('INCOME_X_PREMIUM')

df_eng['AGE_X_PREMIUM'] = df_eng['AGE_NORMALIZED'] * df_eng['PREMIUM_NORMALIZED']
interactions_created.append('AGE_X_PREMIUM')

# Variant parameter interactions (coinsurance × deductible, etc.)
df_eng['COINSURANCE_X_DEDUCTIBLE'] = df_eng['TREATMENT_COINSURANCE'] * df_eng['TREATMENT_DEDUCTIBLE']
interactions_created.append('COINSURANCE_X_DEDUCTIBLE')

df_eng['COINSURANCE_X_LIMIT'] = df_eng['TREATMENT_COINSURANCE'] * df_eng['TREATMENT_COVERAGE_LIMIT']
interactions_created.append('COINSURANCE_X_LIMIT')

df_eng['DEDUCTIBLE_X_LIMIT'] = df_eng['TREATMENT_DEDUCTIBLE'] * df_eng['TREATMENT_COVERAGE_LIMIT']
interactions_created.append('DEDUCTIBLE_X_LIMIT')

# Pet age × customer age interaction
df_eng['PET_AGE_NORMALIZED'] = (df_eng['PET_AGE_YEARS'] - df_eng['PET_AGE_YEARS'].mean()) / df_eng['PET_AGE_YEARS'].std()

df_eng['CUSTOMER_AGE_X_PET_AGE'] = df_eng['AGE_NORMALIZED'] * df_eng['PET_AGE_NORMALIZED']
interactions_created.append('CUSTOMER_AGE_X_PET_AGE')

print(f"✓ Created {len(interactions_created)} interaction features:")
for f in interactions_created:
    print(f"  - {f}")
print()

# ============================================================================
# PART 3: BINARY FLAGS AND CATEGORICAL ENCODINGS
# ============================================================================
print("[4/8] Creating binary flags and categorical encodings...")

binary_flags = []

# Income-based segments (from Stage 3: Q4 converts at 19.9% vs Q1 at 12.3%)
income_quartiles = df_eng['MEDIAN_HOUSEHOLD_INCOME_2020'].quantile([0.25, 0.5, 0.75])
df_eng['IS_HIGH_INCOME'] = (df_eng['MEDIAN_HOUSEHOLD_INCOME_2020'] >= income_quartiles[0.75]).astype(int)
df_eng['IS_LOW_INCOME'] = (df_eng['MEDIAN_HOUSEHOLD_INCOME_2020'] <= income_quartiles[0.25]).astype(int)
df_eng['IS_MIDDLE_INCOME'] = ((df_eng['MEDIAN_HOUSEHOLD_INCOME_2020'] > income_quartiles[0.25]) &
                               (df_eng['MEDIAN_HOUSEHOLD_INCOME_2020'] < income_quartiles[0.75])).astype(int)
binary_flags.extend(['IS_HIGH_INCOME', 'IS_LOW_INCOME', 'IS_MIDDLE_INCOME'])

# Income bins
df_eng['INCOME_BIN'] = pd.qcut(df_eng['MEDIAN_HOUSEHOLD_INCOME_2020'], q=10, labels=False, duplicates='drop')
binary_flags.append('INCOME_BIN')

# Age-based segments
age_median = df_eng['IMPUTED_AGE'].median()
df_eng['IS_SENIOR'] = (df_eng['IMPUTED_AGE'] >= 65).astype(int)
df_eng['IS_YOUNG_ADULT'] = (df_eng['IMPUTED_AGE'] <= 35).astype(int)
df_eng['IS_MIDDLE_AGE'] = ((df_eng['IMPUTED_AGE'] > 35) & (df_eng['IMPUTED_AGE'] < 65)).astype(int)
binary_flags.extend(['IS_SENIOR', 'IS_YOUNG_ADULT', 'IS_MIDDLE_AGE'])

# Premium-based segments (price sensitivity)
premium_quartiles = df_eng['PIT_ANNUAL_PREMIUM'].quantile([0.25, 0.5, 0.75])
df_eng['IS_HIGH_PREMIUM'] = (df_eng['PIT_ANNUAL_PREMIUM'] >= premium_quartiles[0.75]).astype(int)
df_eng['IS_LOW_PREMIUM'] = (df_eng['PIT_ANNUAL_PREMIUM'] <= premium_quartiles[0.25]).astype(int)
binary_flags.extend(['IS_HIGH_PREMIUM', 'IS_LOW_PREMIUM'])

# Pet age segments (puppies/kittens vs seniors)
df_eng['IS_YOUNG_PET'] = (df_eng['PET_AGE_YEARS'] <= 2).astype(int)
df_eng['IS_SENIOR_PET'] = (df_eng['PET_AGE_YEARS'] >= 8).astype(int)
binary_flags.extend(['IS_YOUNG_PET', 'IS_SENIOR_PET'])

# Existing customer composite score (0-3 based on indicators)
df_eng['EXISTING_CUSTOMER_SCORE'] = (
    df_eng['HAS_MULTIPLE_PET_DISCOUNT'] +
    df_eng['HAS_DEBIT_CARD'] +
    df_eng['HAS_STRONGLY_CONNECTED_USERS']
)
binary_flags.append('EXISTING_CUSTOMER_SCORE')

# High-value customer flag (existing + high income)
df_eng['IS_HIGH_VALUE_CUSTOMER'] = (
    (df_eng['EXISTING_CUSTOMER_SCORE'] >= 2) &
    (df_eng['IS_HIGH_INCOME'] == 1)
).astype(int)
binary_flags.append('IS_HIGH_VALUE_CUSTOMER')

# Variant quality tiers (from Stage 2: best variants are 70% coinsurance + low deductible)
df_eng['IS_TOP_TIER_VARIANT'] = (
    (df_eng['TREATMENT_COINSURANCE'] == 70) &
    (df_eng['TREATMENT_DEDUCTIBLE'] == 100)
).astype(int)

df_eng['IS_BOTTOM_TIER_VARIANT'] = (
    (df_eng['TREATMENT_COINSURANCE'] == 90) &
    (df_eng['TREATMENT_DEDUCTIBLE'] == 500)
).astype(int)

binary_flags.extend(['IS_TOP_TIER_VARIANT', 'IS_BOTTOM_TIER_VARIANT'])

# State grouping by conversion performance (from Stage 3)
# High performers: DC, NY, CA, etc.
# Low performers: HI, MT, etc.
high_conversion_states = ['DC', 'NY', 'CA', 'MA', 'NJ', 'CT', 'MD', 'CO']
low_conversion_states = ['HI', 'MT', 'WY', 'ID', 'ND', 'SD']

df_eng['IS_HIGH_CONVERSION_STATE'] = df_eng['STATE'].isin(high_conversion_states).astype(int)
df_eng['IS_LOW_CONVERSION_STATE'] = df_eng['STATE'].isin(low_conversion_states).astype(int)
binary_flags.extend(['IS_HIGH_CONVERSION_STATE', 'IS_LOW_CONVERSION_STATE'])

print(f"✓ Created {len(binary_flags)} binary/categorical features:")
for f in binary_flags:
    print(f"  - {f}")
print()

# ============================================================================
# PART 4: NON-LINEAR TRANSFORMATIONS FOR NUMERICAL FEATURES
# ============================================================================
print("[5/8] Engineering non-linear transformations...")

nonlinear_features = []

# Log transformations (for right-skewed distributions)
df_eng['LOG_INCOME'] = np.log1p(df_eng['MEDIAN_HOUSEHOLD_INCOME_2020'])
df_eng['LOG_PREMIUM'] = np.log1p(df_eng['PIT_ANNUAL_PREMIUM'])
nonlinear_features.extend(['LOG_INCOME', 'LOG_PREMIUM'])

# Square root transformations
df_eng['SQRT_INCOME'] = np.sqrt(df_eng['MEDIAN_HOUSEHOLD_INCOME_2020'])
df_eng['SQRT_PREMIUM'] = np.sqrt(df_eng['PIT_ANNUAL_PREMIUM'])
nonlinear_features.extend(['SQRT_INCOME', 'SQRT_PREMIUM'])

# Polynomial features (quadratic) for key predictors
df_eng['INCOME_SQUARED'] = df_eng['INCOME_NORMALIZED'] ** 2
df_eng['AGE_SQUARED'] = df_eng['AGE_NORMALIZED'] ** 2
df_eng['PREMIUM_SQUARED'] = df_eng['PREMIUM_NORMALIZED'] ** 2
df_eng['PET_AGE_SQUARED'] = df_eng['PET_AGE_NORMALIZED'] ** 2
nonlinear_features.extend(['INCOME_SQUARED', 'AGE_SQUARED', 'PREMIUM_SQUARED', 'PET_AGE_SQUARED'])

# Ratio features
df_eng['PREMIUM_TO_INCOME_RATIO'] = df_eng['PIT_ANNUAL_PREMIUM'] / (df_eng['MEDIAN_HOUSEHOLD_INCOME_2020'] + 1)  # Affordability
df_eng['DEDUCTIBLE_TO_LIMIT_RATIO'] = df_eng['TREATMENT_DEDUCTIBLE'] / (df_eng['TREATMENT_COVERAGE_LIMIT'] + 1)
nonlinear_features.extend(['PREMIUM_TO_INCOME_RATIO', 'DEDUCTIBLE_TO_LIMIT_RATIO'])

# Binned versions of continuous features (non-linear discretization)
df_eng['AGE_BIN'] = pd.cut(df_eng['IMPUTED_AGE'], bins=[0, 35, 50, 65, 100], labels=['Young', 'Middle', 'Senior', 'Elder'])
df_eng['PET_AGE_BIN'] = pd.cut(df_eng['PET_AGE_YEARS'], bins=[0, 2, 5, 8, 20], labels=['Puppy', 'Young', 'Adult', 'Senior'])
nonlinear_features.extend(['AGE_BIN', 'PET_AGE_BIN'])

print(f"✓ Created {len(nonlinear_features)} non-linear features:")
for f in nonlinear_features:
    print(f"  - {f}")
print()

# ============================================================================
# PART 5: COMPOSITE/AGGREGATED FEATURES
# ============================================================================
print("[6/8] Creating composite and aggregated features...")

composite_features = []

# Risk score composite (combination of factors that increase conversion)
# Positive factors: existing customer indicators, high income
# Negative factors: high premium
df_eng['PROPENSITY_SCORE'] = (
    df_eng['HAS_MULTIPLE_PET_DISCOUNT'] * 3.0 +  # Strongest predictor
    df_eng['HAS_DEBIT_CARD'] * 2.0 +
    df_eng['HAS_STRONGLY_CONNECTED_USERS'] * 1.0 +
    df_eng['IS_HIGH_INCOME'] * 1.5 -
    df_eng['IS_HIGH_PREMIUM'] * 1.0
)
composite_features.append('PROPENSITY_SCORE')

# Value score (potential lifetime value proxy)
df_eng['CUSTOMER_VALUE_SCORE'] = (
    df_eng['INCOME_NORMALIZED'] * 0.3 +
    df_eng['EXISTING_CUSTOMER_SCORE'] * 0.4 +
    (1 - df_eng['PREMIUM_NORMALIZED']) * 0.3  # Lower premium = less price-sensitive
)
composite_features.append('CUSTOMER_VALUE_SCORE')

# Engagement composite (time + existing customer signals)
df_eng['ENGAGEMENT_SCORE'] = (
    df_eng['HAS_DEBIT_CARD'] * 2 +
    df_eng['HAS_STRONGLY_CONNECTED_USERS'] * 1 +
    df_eng['IS_BUSINESS_HOURS'] * 0.5
)
composite_features.append('ENGAGEMENT_SCORE')

# Geographic features (state-level aggregations will be done next)
# For now, create region based on state
state_to_region = {
    'ME': 'Northeast', 'NH': 'Northeast', 'VT': 'Northeast', 'MA': 'Northeast', 'RI': 'Northeast',
    'CT': 'Northeast', 'NY': 'Northeast', 'NJ': 'Northeast', 'PA': 'Northeast',
    'OH': 'Midwest', 'IN': 'Midwest', 'IL': 'Midwest', 'MI': 'Midwest', 'WI': 'Midwest',
    'MN': 'Midwest', 'IA': 'Midwest', 'MO': 'Midwest', 'ND': 'Midwest', 'SD': 'Midwest',
    'NE': 'Midwest', 'KS': 'Midwest',
    'DE': 'South', 'MD': 'South', 'DC': 'South', 'VA': 'South', 'WV': 'South', 'NC': 'South',
    'SC': 'South', 'GA': 'South', 'FL': 'South', 'KY': 'South', 'TN': 'South', 'AL': 'South',
    'MS': 'South', 'AR': 'South', 'LA': 'South', 'OK': 'South', 'TX': 'South',
    'MT': 'West', 'ID': 'West', 'WY': 'West', 'CO': 'West', 'NM': 'West', 'AZ': 'West',
    'UT': 'West', 'NV': 'West', 'WA': 'West', 'OR': 'West', 'CA': 'West', 'AK': 'West', 'HI': 'West'
}
df_eng['REGION'] = df_eng['STATE'].map(state_to_region)
composite_features.append('REGION')

# Breed complexity (designer breed + specific breeds that might be higher risk)
df_eng['IS_COMPLEX_BREED'] = (df_eng['DESIGNER_BREED'] == 1).astype(int)
composite_features.append('IS_COMPLEX_BREED')

print(f"✓ Created {len(composite_features)} composite features:")
for f in composite_features:
    print(f"  - {f}")
print()

# ============================================================================
# PART 6: STATE-LEVEL AGGREGATIONS (TARGET ENCODING PROXY)
# ============================================================================
print("[6.5/8] Creating state-level aggregate features...")

# Calculate state-level statistics (like target encoding but for EDA)
state_stats = df_eng.groupby('STATE').agg({
    'CONVERTED': ['mean', 'count'],  # Conversion rate and sample size
    'MEDIAN_HOUSEHOLD_INCOME_2020': 'median',
    'IMPUTED_AGE': 'median',
    'PIT_ANNUAL_PREMIUM': 'median'
}).round(4)

state_stats.columns = ['STATE_CONVERSION_RATE', 'STATE_SAMPLE_SIZE',
                       'STATE_MEDIAN_INCOME', 'STATE_MEDIAN_AGE',
                       'STATE_MEDIAN_PREMIUM']

# Add state-level features back to main dataframe
df_eng = df_eng.merge(state_stats, left_on='STATE', right_index=True, how='left')

state_agg_features = ['STATE_CONVERSION_RATE', 'STATE_SAMPLE_SIZE',
                      'STATE_MEDIAN_INCOME', 'STATE_MEDIAN_AGE',
                      'STATE_MEDIAN_PREMIUM']

# Create deviation features (how much does individual differ from state average)
df_eng['INCOME_VS_STATE_MEDIAN'] = df_eng['MEDIAN_HOUSEHOLD_INCOME_2020'] - df_eng['STATE_MEDIAN_INCOME']
df_eng['AGE_VS_STATE_MEDIAN'] = df_eng['IMPUTED_AGE'] - df_eng['STATE_MEDIAN_AGE']
df_eng['PREMIUM_VS_STATE_MEDIAN'] = df_eng['PIT_ANNUAL_PREMIUM'] - df_eng['STATE_MEDIAN_PREMIUM']

state_agg_features.extend(['INCOME_VS_STATE_MEDIAN', 'AGE_VS_STATE_MEDIAN', 'PREMIUM_VS_STATE_MEDIAN'])

print(f"✓ Created {len(state_agg_features)} state-level aggregate features:")
for f in state_agg_features:
    print(f"  - {f}")
print()

# ============================================================================
# SUMMARY OF ALL NEW FEATURES
# ============================================================================
print("="*80)
print("FEATURE ENGINEERING SUMMARY")
print("="*80)
print()

new_features = set(df_eng.columns) - original_features
print(f"Original features: {len(original_features)}")
print(f"New features created: {len(new_features)}")
print(f"Total features: {len(df_eng.columns)}")
print()

# Categorize new features
feature_categories = {
    'Time-based': time_features,
    'Interactions': interactions_created,
    'Binary Flags': binary_flags,
    'Non-linear Transforms': nonlinear_features,
    'Composite Scores': composite_features,
    'State Aggregations': state_agg_features
}

print("Feature categories:")
for category, features in feature_categories.items():
    print(f"  {category}: {len(features)} features")
print()

# Save engineered dataset
output_path = 'data/04_engineered_features.csv'
df_eng.to_csv(output_path, index=False)
print(f"✓ Saved engineered dataset: {output_path}")
print(f"  Shape: {df_eng.shape[0]:,} rows × {df_eng.shape[1]} columns")
print()

# Save feature metadata
feature_metadata = {
    'original_features': list(original_features),
    'new_features': list(new_features),
    'feature_categories': {k: v for k, v in feature_categories.items()},
    'total_features': len(df_eng.columns),
    'timestamp': datetime.now().isoformat()
}

metadata_path = 'results/stage4_feature_metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(feature_metadata, f, indent=2)
print(f"✓ Saved feature metadata: {metadata_path}")
print()

print("="*80)
print("FEATURE ENGINEERING COMPLETE")
print("="*80)
