"""
Stage 3 Part 2: Feature Importance Ranking and Deep Dives

This script ranks features by predictive power, creates visualizations,
and performs deep dives into high-signal features.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# File paths
DATA_FILE = Path("data/03_cleaned_fixed.csv")
RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

print("=" * 80)
print("STAGE 3 PART 2: FEATURE IMPORTANCE AND VISUALIZATIONS")
print("=" * 80)

# ============================================================================
# 1. LOAD DATA AND PREVIOUS RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("1. LOADING DATA AND PREVIOUS RESULTS")
print("=" * 80)

df = pd.read_csv(DATA_FILE)
print(f"✓ Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

# Load previous analysis results
num_bivariate = pd.read_csv(RESULTS_DIR / "stage3_numerical_bivariate.csv")
cat_bivariate = pd.read_csv(RESULTS_DIR / "stage3_categorical_bivariate.csv")
num_stats = pd.read_csv(RESULTS_DIR / "stage3_numerical_stats.csv")
cat_stats = pd.read_csv(RESULTS_DIR / "stage3_categorical_stats.csv")

print(f"✓ Loaded previous analysis results")

# ============================================================================
# 2. UNIFIED FEATURE IMPORTANCE RANKING
# ============================================================================
print("\n" + "=" * 80)
print("2. UNIFIED FEATURE IMPORTANCE RANKING")
print("=" * 80)

feature_importance = []

# Process numerical features
for _, row in num_bivariate.iterrows():
    feature_importance.append({
        'feature': row['feature'],
        'type': 'numerical',
        'effect_size': abs(row['cohens_d']),
        'effect_size_metric': 'cohens_d',
        'p_value': row['t_pvalue'],
        'significant': row['significant'],
        'correlation': abs(row['correlation']),
        'direction': 'positive' if row['mean_diff'] > 0 else 'negative',
        'mean_diff_pct': abs(row['mean_diff_pct'])
    })

# Process categorical features
for _, row in cat_bivariate.iterrows():
    feature_importance.append({
        'feature': row['feature'],
        'type': 'categorical',
        'effect_size': row['cramers_v'],
        'effect_size_metric': 'cramers_v',
        'p_value': row['chi2_pvalue'],
        'significant': row['significant'],
        'correlation': np.nan,
        'direction': 'categorical',
        'mean_diff_pct': row['range_pct']
    })

# Create unified ranking
feature_importance_df = pd.DataFrame(feature_importance)
feature_importance_df = feature_importance_df.sort_values('effect_size', ascending=False)

# Add ranking
feature_importance_df['rank'] = range(1, len(feature_importance_df) + 1)

# Categorize importance
def categorize_importance(effect_size, metric):
    if metric == 'cohens_d':
        if effect_size >= 0.8:
            return 'Large'
        elif effect_size >= 0.5:
            return 'Medium'
        elif effect_size >= 0.2:
            return 'Small'
        else:
            return 'Negligible'
    elif metric == 'cramers_v':
        if effect_size >= 0.5:
            return 'Large'
        elif effect_size >= 0.3:
            return 'Medium'
        elif effect_size >= 0.1:
            return 'Small'
        else:
            return 'Negligible'
    return 'Unknown'

feature_importance_df['importance_category'] = feature_importance_df.apply(
    lambda x: categorize_importance(x['effect_size'], x['effect_size_metric']), axis=1
)

# Save feature importance ranking
feature_importance_df.to_csv(RESULTS_DIR / "stage3_feature_importance_ranking.csv", index=False)
print(f"✓ Saved feature importance ranking")

print("\nTop 10 Features by Effect Size:")
print("=" * 80)
for _, row in feature_importance_df.head(10).iterrows():
    sig_marker = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
    print(f"{row['rank']:2d}. {row['feature']:35s} | Effect: {row['effect_size']:.3f} ({row['effect_size_metric']:10s}) | {row['importance_category']:10s} {sig_marker}")

# ============================================================================
# 3. FEATURE GROUP ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("3. FEATURE GROUP ANALYSIS")
print("=" * 80)

# Define feature groups
feature_groups = {
    'customer_demographics': ['IMPUTED_AGE', 'STATE', 'MEDIAN_HOUSEHOLD_INCOME_2020'],
    'pet_characteristics': ['PET_AGE_YEARS', 'PET_BREED_CLEAN', 'DESIGNER_BREED', 'PET_SEX'],
    'pricing': ['PIT_ANNUAL_PREMIUM', 'BASE_PREMIUM'],
    'existing_customer': ['HAS_MULTIPLE_PET_DISCOUNT', 'HAS_DEBIT_CARD', 'HAS_STRONGLY_CONNECTED_USERS'],
    'context': ['TOTAL_VET_CLINICS', 'QUOTE_MONTH_COS', 'COVERAGE_SENT_AT']
}

# Calculate average effect size by group
group_importance = []
for group_name, features in feature_groups.items():
    group_features = feature_importance_df[feature_importance_df['feature'].isin(features)]
    avg_effect = group_features['effect_size'].mean()
    n_significant = group_features['significant'].sum()
    n_total = len(group_features)

    group_importance.append({
        'group': group_name,
        'n_features': n_total,
        'n_significant': n_significant,
        'avg_effect_size': avg_effect,
        'features': ', '.join(group_features.sort_values('effect_size', ascending=False)['feature'].tolist())
    })

group_importance_df = pd.DataFrame(group_importance).sort_values('avg_effect_size', ascending=False)
group_importance_df.to_csv(RESULTS_DIR / "stage3_feature_group_importance.csv", index=False)

print("\nFeature Group Rankings:")
for _, row in group_importance_df.iterrows():
    print(f"  • {row['group']:25s}: Avg Effect = {row['avg_effect_size']:.3f} ({row['n_significant']}/{row['n_features']} significant)")

# ============================================================================
# 4. DEEP DIVE: TOP CATEGORICAL FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("4. DEEP DIVE: TOP CATEGORICAL FEATURES")
print("=" * 80)

# Focus on HAS_MULTIPLE_PET_DISCOUNT, HAS_DEBIT_CARD, HAS_STRONGLY_CONNECTED_USERS, STATE
top_categorical = ['HAS_MULTIPLE_PET_DISCOUNT', 'HAS_DEBIT_CARD', 'HAS_STRONGLY_CONNECTED_USERS', 'STATE']

categorical_deep_dive = []

for feat in top_categorical:
    if feat not in df.columns:
        continue

    # Conversion analysis
    conversion_by_cat = df.groupby(feat).agg({
        'is_purchased': ['sum', 'count', 'mean'],
        'premium_amount': 'mean'
    }).round(4)
    conversion_by_cat.columns = ['conversions', 'total_quotes', 'conversion_rate', 'avg_premium']
    conversion_by_cat['avg_sales_per_quote'] = conversion_by_cat['conversion_rate'] * conversion_by_cat['avg_premium']

    print(f"\n{feat}:")
    print("-" * 80)

    if df[feat].nunique() <= 10:
        # Show all categories
        conversion_by_cat = conversion_by_cat.sort_values('conversion_rate', ascending=False)
        print(conversion_by_cat)

        # Save detailed breakdown
        conversion_by_cat.to_csv(RESULTS_DIR / f"stage3_deep_dive_{feat.lower()}.csv")
    else:
        # Show top and bottom 10
        print("Top 10 by conversion rate:")
        print(conversion_by_cat.sort_values('conversion_rate', ascending=False).head(10))
        print("\nBottom 10 by conversion rate:")
        print(conversion_by_cat.sort_values('conversion_rate', ascending=False).tail(10))

        # Save detailed breakdown
        conversion_by_cat.to_csv(RESULTS_DIR / f"stage3_deep_dive_{feat.lower()}.csv")

# ============================================================================
# 5. CORRELATION MATRIX FOR NUMERICAL FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("5. CORRELATION MATRIX FOR NUMERICAL FEATURES")
print("=" * 80)

numerical_features = num_stats['feature'].tolist()
numerical_features.append('is_purchased')

# Calculate correlation matrix
corr_matrix = df[numerical_features].corr()

# Save correlation matrix
corr_matrix.to_csv(RESULTS_DIR / "stage3_numerical_correlations.csv")
print(f"✓ Saved numerical correlation matrix")

# Show correlations with target
target_corr = corr_matrix['is_purchased'].drop('is_purchased').sort_values(key=abs, ascending=False)
print("\nCorrelations with Conversion (is_purchased):")
for feat, corr in target_corr.items():
    print(f"  • {feat:35s}: {corr:6.3f}")

# ============================================================================
# 6. INTERACTION EFFECTS: TOP FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("6. INTERACTION EFFECTS ANALYSIS")
print("=" * 80)

# Analyze interaction between top features
# Example: HAS_MULTIPLE_PET_DISCOUNT × HAS_DEBIT_CARD
if 'HAS_MULTIPLE_PET_DISCOUNT' in df.columns and 'HAS_DEBIT_CARD' in df.columns:
    interaction_analysis = df.groupby(['HAS_MULTIPLE_PET_DISCOUNT', 'HAS_DEBIT_CARD']).agg({
        'is_purchased': ['sum', 'count', 'mean']
    }).round(4)
    interaction_analysis.columns = ['conversions', 'total', 'conversion_rate']

    print("\nInteraction: HAS_MULTIPLE_PET_DISCOUNT × HAS_DEBIT_CARD")
    print(interaction_analysis)

    # Calculate additive vs actual effect
    base_rate = df['is_purchased'].mean()
    print(f"\nBase conversion rate: {base_rate:.4f}")

    for (multi_pet, debit), row in interaction_analysis.iterrows():
        additive_effect = (
            df[df['HAS_MULTIPLE_PET_DISCOUNT'] == multi_pet]['is_purchased'].mean() - base_rate +
            df[df['HAS_DEBIT_CARD'] == debit]['is_purchased'].mean() - base_rate +
            base_rate
        )
        actual_effect = row['conversion_rate']
        interaction_lift = actual_effect - additive_effect

        print(f"  Multi-Pet={multi_pet}, Debit={debit}:")
        print(f"    Actual: {actual_effect:.4f} | Additive expected: {additive_effect:.4f} | Interaction: {interaction_lift:+.4f}")

# Example: MEDIAN_HOUSEHOLD_INCOME × STATE (high-level)
print("\nInteraction: Income × Top States")
top_states = df['STATE'].value_counts().head(5).index.tolist()

for state in top_states:
    state_data = df[df['STATE'] == state]
    # Split income into quartiles
    state_data_copy = state_data.copy()
    state_data_copy['income_quartile'] = pd.qcut(state_data['MEDIAN_HOUSEHOLD_INCOME_2020'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

    income_effect = state_data_copy.groupby('income_quartile')['is_purchased'].mean()

    print(f"\n  {state}:")
    for quartile, rate in income_effect.items():
        print(f"    {quartile}: {rate:.4f}")

print("\n" + "=" * 80)
print("STAGE 3 PART 2 COMPLETE")
print("=" * 80)
print("✓ Feature importance ranking generated")
print("✓ Deep dives into top features completed")
print("✓ Correlation analysis completed")
print("✓ Interaction effects analyzed")
print("\nNext: Create visualizations")
