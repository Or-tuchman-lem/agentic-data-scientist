"""
Stage 4: Feature Testing and Validation

This script tests the predictive power of newly engineered features by:
1. Calculating effect sizes (Cohen's d) for continuous features
2. Computing conversion rate differences for binary features
3. Statistical tests for significance
4. Ranking features by predictive strength

Author: Agentic Data Scientist
Date: 2026-02-26
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

print("="*80)
print("STAGE 4: TESTING NEW FEATURES FOR PREDICTIVE POWER")
print("="*80)
print()

# Load engineered data
print("[1/5] Loading engineered dataset...")
df = pd.read_csv('data/04_engineered_features.csv')
print(f"✓ Loaded: {len(df):,} rows × {len(df.columns)} columns")
print()

# Load feature metadata to identify new features
import json
with open('results/stage4_feature_metadata.json', 'r') as f:
    metadata = json.load(f)

new_features = metadata['new_features']
print(f"Testing {len(new_features)} new features for predictive power")
print()

# Helper function to calculate Cohen's d
def cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0

# ============================================================================
# PART 1: TEST CONTINUOUS FEATURES
# ============================================================================
print("[2/5] Testing continuous/numerical features...")

# Identify continuous features (exclude categorical)
categorical_features = ['QUOTE_SEASON', 'AGE_BIN', 'PET_AGE_BIN', 'REGION']
continuous_new_features = [f for f in new_features if f not in categorical_features and f in df.columns]

# Remove normalized/intermediate features for cleaner results
normalized_features = [f for f in continuous_new_features if 'NORMALIZED' in f]
continuous_new_features = [f for f in continuous_new_features if 'NORMALIZED' not in f]

results_continuous = []

for feature in continuous_new_features:
    try:
        # Skip if feature has too many missing values
        if df[feature].isna().sum() / len(df) > 0.5:
            continue

        # Get converters and non-converters
        converters = df[df['CONVERTED'] == 1][feature].dropna()
        non_converters = df[df['CONVERTED'] == 0][feature].dropna()

        if len(converters) < 10 or len(non_converters) < 10:
            continue

        # Calculate statistics
        mean_converters = converters.mean()
        mean_non_converters = non_converters.mean()
        diff = mean_converters - mean_non_converters

        # Effect size
        effect_size = cohens_d(converters, non_converters)

        # Statistical test (t-test or Mann-Whitney depending on normality)
        # Use Mann-Whitney U for robustness
        statistic, p_value = stats.mannwhitneyu(converters, non_converters, alternative='two-sided')

        # Correlation with target
        correlation = df[[feature, 'CONVERTED']].corr().iloc[0, 1]

        results_continuous.append({
            'feature': feature,
            'mean_converters': mean_converters,
            'mean_non_converters': mean_non_converters,
            'difference': diff,
            'effect_size': abs(effect_size),
            'correlation': correlation,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'n_converters': len(converters),
            'n_non_converters': len(non_converters)
        })
    except Exception as e:
        print(f"  Warning: Could not test {feature}: {e}")
        continue

df_continuous = pd.DataFrame(results_continuous)
df_continuous = df_continuous.sort_values('effect_size', ascending=False)

print(f"✓ Tested {len(df_continuous)} continuous features")
print()
print("Top 10 continuous features by effect size:")
for i, row in df_continuous.head(10).iterrows():
    sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
    print(f"  {row['feature']:40s} | Cohen's d = {row['effect_size']:6.3f} | r = {row['correlation']:6.3f} {sig}")
print()

# ============================================================================
# PART 2: TEST BINARY FEATURES
# ============================================================================
print("[3/5] Testing binary/categorical features...")

# Identify binary features (0/1 encoded)
binary_features = []
for feature in new_features:
    if feature not in df.columns:
        continue
    unique_vals = df[feature].dropna().unique()
    if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
        binary_features.append(feature)

results_binary = []

for feature in binary_features:
    try:
        # Calculate conversion rates for each group
        grouped = df.groupby(feature)['CONVERTED'].agg(['mean', 'count'])

        if len(grouped) < 2 or grouped['count'].min() < 10:
            continue

        conv_rate_1 = grouped.loc[1, 'mean'] if 1 in grouped.index else 0
        conv_rate_0 = grouped.loc[0, 'mean'] if 0 in grouped.index else 0
        diff = conv_rate_1 - conv_rate_0

        # Chi-square test
        contingency = pd.crosstab(df[feature], df['CONVERTED'])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

        # Cramér's V (effect size for categorical)
        n = contingency.sum().sum()
        cramers_v = np.sqrt(chi2 / n)

        results_binary.append({
            'feature': feature,
            'conv_rate_1': conv_rate_1,
            'conv_rate_0': conv_rate_0,
            'difference': diff,
            'cramers_v': cramers_v,
            'chi2': chi2,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'n_1': grouped.loc[1, 'count'] if 1 in grouped.index else 0,
            'n_0': grouped.loc[0, 'count'] if 0 in grouped.index else 0
        })
    except Exception as e:
        print(f"  Warning: Could not test {feature}: {e}")
        continue

df_binary = pd.DataFrame(results_binary)
df_binary = df_binary.sort_values('cramers_v', ascending=False)

print(f"✓ Tested {len(df_binary)} binary features")
print()
print("Top 15 binary features by effect size (Cramér's V):")
for i, row in df_binary.head(15).iterrows():
    sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
    print(f"  {row['feature']:40s} | V = {row['cramers_v']:6.3f} | Δconv = {row['difference']:7.1%} {sig}")
print()

# ============================================================================
# PART 3: TEST INTERACTION FEATURES
# ============================================================================
print("[4/5] Analyzing interaction features specifically...")

interaction_features = [f for f in new_features if '_X_' in f]
interaction_results = []

for feature in interaction_features:
    if feature not in df.columns:
        continue

    try:
        # For interactions, calculate correlation with target
        correlation = df[[feature, 'CONVERTED']].corr().iloc[0, 1]

        # Also calculate effect size
        converters = df[df['CONVERTED'] == 1][feature].dropna()
        non_converters = df[df['CONVERTED'] == 0][feature].dropna()
        effect_size = cohens_d(converters, non_converters)

        # Statistical test
        statistic, p_value = stats.mannwhitneyu(converters, non_converters, alternative='two-sided')

        interaction_results.append({
            'feature': feature,
            'correlation': correlation,
            'effect_size': abs(effect_size),
            'p_value': p_value,
            'significant': p_value < 0.05
        })
    except Exception as e:
        continue

df_interactions = pd.DataFrame(interaction_results)
df_interactions = df_interactions.sort_values('effect_size', ascending=False)

print(f"✓ Tested {len(df_interactions)} interaction features")
print()
print("Top 10 interaction features:")
for i, row in df_interactions.head(10).iterrows():
    sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
    print(f"  {row['feature']:40s} | d = {row['effect_size']:6.3f} | r = {row['correlation']:6.3f} {sig}")
print()

# ============================================================================
# PART 4: COMPOSITE SCORE FEATURES
# ============================================================================
print("[5/5] Testing composite score features...")

score_features = ['PROPENSITY_SCORE', 'CUSTOMER_VALUE_SCORE', 'ENGAGEMENT_SCORE',
                  'EXISTING_CUSTOMER_SCORE']

score_results = []

for feature in score_features:
    if feature not in df.columns:
        continue

    try:
        # Calculate correlation and effect size
        correlation = df[[feature, 'CONVERTED']].corr().iloc[0, 1]

        converters = df[df['CONVERTED'] == 1][feature].dropna()
        non_converters = df[df['CONVERTED'] == 0][feature].dropna()
        effect_size = cohens_d(converters, non_converters)

        # Statistical test
        statistic, p_value = stats.mannwhitneyu(converters, non_converters, alternative='two-sided')

        # Also show conversion rate by score level
        score_groups = df.groupby(feature)['CONVERTED'].agg(['mean', 'count'])

        score_results.append({
            'feature': feature,
            'correlation': correlation,
            'effect_size': abs(effect_size),
            'p_value': p_value,
            'significant': p_value < 0.05,
            'score_range': f"{df[feature].min():.1f} to {df[feature].max():.1f}",
            'n_levels': len(score_groups)
        })
    except Exception as e:
        continue

df_scores = pd.DataFrame(score_results)
df_scores = df_scores.sort_values('effect_size', ascending=False)

print(f"✓ Tested {len(df_scores)} composite score features")
print()
print("Composite score features performance:")
for i, row in df_scores.iterrows():
    sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
    print(f"  {row['feature']:30s} | d = {row['effect_size']:6.3f} | r = {row['correlation']:6.3f} {sig}")
    print(f"    Range: {row['score_range']} ({row['n_levels']} levels)")
print()

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("="*80)
print("SAVING RESULTS")
print("="*80)
print()

# Save detailed results
df_continuous.to_csv('results/stage4_continuous_features_tested.csv', index=False)
print(f"✓ Saved continuous feature results: results/stage4_continuous_features_tested.csv")

df_binary.to_csv('results/stage4_binary_features_tested.csv', index=False)
print(f"✓ Saved binary feature results: results/stage4_binary_features_tested.csv")

df_interactions.to_csv('results/stage4_interaction_features_tested.csv', index=False)
print(f"✓ Saved interaction feature results: results/stage4_interaction_features_tested.csv")

df_scores.to_csv('results/stage4_composite_scores_tested.csv', index=False)
print(f"✓ Saved composite score results: results/stage4_composite_scores_tested.csv")

# Create unified ranking of all new features
all_features = []

# Add continuous features
for _, row in df_continuous.iterrows():
    all_features.append({
        'feature': row['feature'],
        'type': 'continuous',
        'effect_size': row['effect_size'],
        'p_value': row['p_value'],
        'metric': 'cohens_d'
    })

# Add binary features
for _, row in df_binary.iterrows():
    all_features.append({
        'feature': row['feature'],
        'type': 'binary',
        'effect_size': row['cramers_v'],
        'p_value': row['p_value'],
        'metric': 'cramers_v'
    })

df_all_features = pd.DataFrame(all_features)
df_all_features = df_all_features.sort_values('effect_size', ascending=False)

df_all_features.to_csv('results/stage4_all_features_ranked.csv', index=False)
print(f"✓ Saved unified feature ranking: results/stage4_all_features_ranked.csv")
print()

# Summary statistics
print("="*80)
print("FEATURE TESTING SUMMARY")
print("="*80)
print()

print(f"Total new features tested: {len(df_all_features)}")
print(f"Significant features (p<0.05): {df_all_features['p_value'].lt(0.05).sum()} ({df_all_features['p_value'].lt(0.05).sum()/len(df_all_features)*100:.1f}%)")
print(f"Highly significant (p<0.001): {df_all_features['p_value'].lt(0.001).sum()}")
print()

# Top features overall
print("TOP 20 NEW FEATURES BY EFFECT SIZE:")
print("-" * 80)
for i, row in df_all_features.head(20).iterrows():
    sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
    print(f"{i+1:2d}. {row['feature']:45s} | {row['metric']:10s} = {row['effect_size']:6.3f} {sig}")
print()

print("="*80)
print("FEATURE TESTING COMPLETE")
print("="*80)
