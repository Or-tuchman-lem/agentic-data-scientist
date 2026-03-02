"""
Randomization Quality Analysis for Pet Insurance A/B Test
=========================================================

Analyzes whether customers were randomly assigned to treatment variants
by checking balance across:
- Sample sizes (chi-square goodness of fit)
- Customer demographics (age, income, state)
- Pet characteristics (age, breed, sex)
- Temporal distribution
- Existing customer indicators

Author: Agentic Data Scientist
Date: 2026-02-26
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Paths
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

print("="*70)
print("RANDOMIZATION QUALITY ANALYSIS")
print("="*70)
print()

# Load cleaned data
print("[1/7] Loading cleaned dataset...")
df = pd.read_csv(DATA_DIR / "03_cleaned_fixed.csv")
print(f"   ✓ Loaded {len(df):,} records with {len(df.columns)} features")
print()

# ============================================================================
# 1. SAMPLE SIZE BALANCE
# ============================================================================
print("[2/7] Analyzing sample size balance across variants...")
print()

variant_counts = df['COVERAGE_TREATMENT'].value_counts().sort_index()
expected_per_variant = len(df) / len(variant_counts)

# Chi-square goodness of fit test
chi2_stat, chi2_p = stats.chisquare(variant_counts)

print("Sample Size Distribution:")
print(f"   Total samples: {len(df):,}")
print(f"   Number of variants: {len(variant_counts)}")
print(f"   Expected per variant (if balanced): {expected_per_variant:.1f}")
print(f"   Actual range: {variant_counts.min():,} - {variant_counts.max():,}")
print(f"   Mean: {variant_counts.mean():.1f}")
print(f"   Std Dev: {variant_counts.std():.1f}")
print(f"   Coefficient of Variation: {variant_counts.std()/variant_counts.mean():.2%}")
print()
print(f"   Chi-square test: χ²={chi2_stat:.2f}, p={chi2_p:.4e}")
if chi2_p < 0.05:
    print(f"   ⚠️  IMBALANCED: Sample sizes differ significantly (p<0.05)")
else:
    print(f"   ✓ BALANCED: Sample sizes are reasonably uniform")
print()

# Save variant counts
variant_summary = pd.DataFrame({
    'variant': variant_counts.index,
    'count': variant_counts.values,
    'percent': variant_counts.values / len(df) * 100,
    'expected': expected_per_variant,
    'difference': variant_counts.values - expected_per_variant,
    'pct_difference': (variant_counts.values - expected_per_variant) / expected_per_variant * 100
})
variant_summary.to_csv(RESULTS_DIR / "randomization_variant_counts.csv", index=False)
print(f"   Saved: results/randomization_variant_counts.csv")
print()

# ============================================================================
# 2. COVARIATE BALANCE - CONTINUOUS FEATURES
# ============================================================================
print("[3/7] Testing covariate balance for continuous features...")
print()

continuous_features = [
    'IMPUTED_AGE',
    'MEDIAN_HOUSEHOLD_INCOME_2020',
    'PET_AGE_YEARS',
    'BASE_PREMIUM',
    'PIT_ANNUAL_PREMIUM'
]

continuous_balance_results = []

for feature in continuous_features:
    # Remove missing values
    feature_data = df[[feature, 'COVERAGE_TREATMENT']].dropna()

    # Group by variant
    groups = [group[feature].values for name, group in feature_data.groupby('COVERAGE_TREATMENT')]

    # ANOVA F-test (parametric)
    f_stat, f_p = stats.f_oneway(*groups)

    # Kruskal-Wallis H-test (non-parametric)
    h_stat, h_p = stats.kruskal(*groups)

    # Calculate means and std devs per variant
    variant_means = feature_data.groupby('COVERAGE_TREATMENT')[feature].mean()
    variant_stds = feature_data.groupby('COVERAGE_TREATMENT')[feature].std()

    # Calculate coefficient of variation across variant means
    cv_means = variant_means.std() / variant_means.mean()

    continuous_balance_results.append({
        'feature': feature,
        'f_statistic': f_stat,
        'f_pvalue': f_p,
        'h_statistic': h_stat,
        'h_pvalue': h_p,
        'mean_across_variants': feature_data[feature].mean(),
        'cv_of_variant_means': cv_means,
        'min_variant_mean': variant_means.min(),
        'max_variant_mean': variant_means.max(),
        'balanced': 'Yes' if h_p >= 0.05 else 'No'
    })

    status = "✓ BALANCED" if h_p >= 0.05 else "⚠️  IMBALANCED"
    print(f"   {feature}:")
    print(f"      Kruskal-Wallis: H={h_stat:.2f}, p={h_p:.4f} {status}")
    print(f"      Variant means range: {variant_means.min():.1f} - {variant_means.max():.1f}")
    print(f"      CV of means: {cv_means:.2%}")
    print()

continuous_balance_df = pd.DataFrame(continuous_balance_results)
continuous_balance_df.to_csv(RESULTS_DIR / "randomization_continuous_balance.csv", index=False)
print(f"   Saved: results/randomization_continuous_balance.csv")
print()

# ============================================================================
# 3. COVARIATE BALANCE - CATEGORICAL FEATURES
# ============================================================================
print("[4/7] Testing covariate balance for categorical features...")
print()

categorical_features = [
    'HAS_MULTIPLE_PET_DISCOUNT',
    'HAS_DEBIT_CARD',
    'HAS_STRONGLY_CONNECTED_USERS',
    'PET_SEX',
    'STATE',
    'DESIGNER_BREED'
]

categorical_balance_results = []

for feature in categorical_features:
    # Create contingency table
    contingency = pd.crosstab(df['COVERAGE_TREATMENT'], df[feature])

    # Chi-square test of independence
    chi2, chi2_p, dof, expected = stats.chi2_contingency(contingency)

    # Cramér's V effect size
    n = contingency.sum().sum()
    min_dim = min(contingency.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

    categorical_balance_results.append({
        'feature': feature,
        'chi2_statistic': chi2,
        'chi2_pvalue': chi2_p,
        'degrees_of_freedom': dof,
        'cramers_v': cramers_v,
        'n_categories': len(df[feature].unique()),
        'balanced': 'Yes' if chi2_p >= 0.05 else 'No'
    })

    status = "✓ BALANCED" if chi2_p >= 0.05 else "⚠️  IMBALANCED"
    print(f"   {feature}:")
    print(f"      Chi-square: χ²={chi2:.2f}, p={chi2_p:.4f} {status}")
    print(f"      Cramér's V: {cramers_v:.4f}")
    print(f"      Categories: {len(df[feature].unique())}")
    print()

categorical_balance_df = pd.DataFrame(categorical_balance_results)
categorical_balance_df.to_csv(RESULTS_DIR / "randomization_categorical_balance.csv", index=False)
print(f"   Saved: results/randomization_categorical_balance.csv")
print()

# ============================================================================
# 4. TEMPORAL BALANCE
# ============================================================================
print("[5/7] Testing temporal balance...")
print()

# Convert timestamp to datetime
df['timestamp_dt'] = pd.to_datetime(df['COVERAGE_SENT_AT'])
df['date'] = df['timestamp_dt'].dt.date
df['month'] = df['timestamp_dt'].dt.to_period('M')
df['week'] = df['timestamp_dt'].dt.to_period('W')

# Test if variants are balanced over time (by month)
monthly_variant_dist = pd.crosstab(df['month'], df['COVERAGE_TREATMENT'])
chi2_month, p_month, _, _ = stats.chi2_contingency(monthly_variant_dist)

# Test if variants are balanced over time (by week)
weekly_variant_dist = pd.crosstab(df['week'], df['COVERAGE_TREATMENT'])
chi2_week, p_week, _, _ = stats.chi2_contingency(weekly_variant_dist)

print("Temporal Distribution:")
print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
print(f"   Duration: {(df['date'].max() - df['date'].min()).days} days")
print(f"   Unique months: {df['month'].nunique()}")
print(f"   Unique weeks: {df['week'].nunique()}")
print()
print(f"   Monthly balance test: χ²={chi2_month:.2f}, p={p_month:.4f}")
status_month = "✓ BALANCED" if p_month >= 0.05 else "⚠️  IMBALANCED"
print(f"      {status_month}")
print()
print(f"   Weekly balance test: χ²={chi2_week:.2f}, p={p_week:.4f}")
status_week = "✓ BALANCED" if p_week >= 0.05 else "⚠️  IMBALANCED"
print(f"      {status_week}")
print()

# Save temporal summary
temporal_summary = {
    'date_min': str(df['date'].min()),
    'date_max': str(df['date'].max()),
    'duration_days': int((df['date'].max() - df['date'].min()).days),
    'n_months': int(df['month'].nunique()),
    'n_weeks': int(df['week'].nunique()),
    'monthly_chi2': float(chi2_month),
    'monthly_pvalue': float(p_month),
    'monthly_balanced': bool(p_month >= 0.05),
    'weekly_chi2': float(chi2_week),
    'weekly_pvalue': float(p_week),
    'weekly_balanced': bool(p_week >= 0.05)
}

with open(RESULTS_DIR / "randomization_temporal_balance.json", 'w') as f:
    json.dump(temporal_summary, f, indent=2)
print(f"   Saved: results/randomization_temporal_balance.json")
print()

# ============================================================================
# 5. OVERALL RANDOMIZATION QUALITY SCORE
# ============================================================================
print("[6/7] Computing overall randomization quality score...")
print()

# Count balanced features
n_continuous_balanced = sum(continuous_balance_df['balanced'] == 'Yes')
n_categorical_balanced = sum(categorical_balance_df['balanced'] == 'Yes')
n_temporal_balanced = int(temporal_summary['monthly_balanced']) + int(temporal_summary['weekly_balanced'])

total_tests = len(continuous_features) + len(categorical_features) + 2  # +2 for temporal
total_balanced = n_continuous_balanced + n_categorical_balanced + n_temporal_balanced

randomization_score = total_balanced / total_tests * 100

print("Randomization Quality Summary:")
print(f"   Sample size balance: {'✓ Pass' if chi2_p >= 0.05 else '⚠️  Fail'}")
print(f"   Continuous features balanced: {n_continuous_balanced}/{len(continuous_features)}")
print(f"   Categorical features balanced: {n_categorical_balanced}/{len(categorical_features)}")
print(f"   Temporal balance: {n_temporal_balanced}/2")
print()
print(f"   OVERALL SCORE: {randomization_score:.1f}% ({total_balanced}/{total_tests} tests passed)")
print()

# Quality grade
if randomization_score >= 90:
    grade = "A - EXCELLENT"
    quality = "Excellent randomization. Causal inference is valid."
elif randomization_score >= 75:
    grade = "B - GOOD"
    quality = "Good randomization with minor imbalances. Causal inference likely valid."
elif randomization_score >= 60:
    grade = "C - FAIR"
    quality = "Fair randomization. Some imbalances present. Consider covariate adjustment."
elif randomization_score >= 40:
    grade = "D - POOR"
    quality = "Poor randomization. Significant imbalances. Require covariate adjustment."
else:
    grade = "F - FAILED"
    quality = "Failed randomization. Severe imbalances. Causal inference not valid without adjustment."

print(f"   GRADE: {grade}")
print(f"   ASSESSMENT: {quality}")
print()

# Create summary report
summary_report = {
    'overall_score': randomization_score,
    'grade': grade,
    'assessment': quality,
    'sample_size_balanced': bool(chi2_p >= 0.05),
    'continuous_features': {
        'n_tested': len(continuous_features),
        'n_balanced': int(n_continuous_balanced),
        'pct_balanced': n_continuous_balanced / len(continuous_features) * 100
    },
    'categorical_features': {
        'n_tested': len(categorical_features),
        'n_balanced': int(n_categorical_balanced),
        'pct_balanced': n_categorical_balanced / len(categorical_features) * 100
    },
    'temporal_balance': {
        'n_tested': 2,
        'n_balanced': int(n_temporal_balanced),
        'pct_balanced': n_temporal_balanced / 2 * 100
    },
    'imbalanced_features': [
        row['feature'] for _, row in continuous_balance_df[continuous_balance_df['balanced'] == 'No'].iterrows()
    ] + [
        row['feature'] for _, row in categorical_balance_df[categorical_balance_df['balanced'] == 'No'].iterrows()
    ]
}

with open(RESULTS_DIR / "randomization_quality_summary.json", 'w') as f:
    json.dump(summary_report, f, indent=2)
print(f"   Saved: results/randomization_quality_summary.json")
print()

# ============================================================================
# 6. VISUALIZATIONS
# ============================================================================
print("[7/7] Creating visualizations...")
print()

fig = plt.figure(figsize=(20, 12))

# 1. Sample size distribution
ax1 = plt.subplot(3, 3, 1)
variant_counts_sorted = variant_counts.sort_values(ascending=False)
bars = ax1.bar(range(len(variant_counts_sorted)), variant_counts_sorted.values,
               color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
ax1.axhline(expected_per_variant, color='red', linestyle='--', linewidth=2,
            label=f'Expected: {expected_per_variant:.0f}')
ax1.set_xlabel('Variant Rank', fontsize=11, fontweight='bold')
ax1.set_ylabel('Sample Count', fontsize=11, fontweight='bold')
ax1.set_title(f'A. Sample Size Distribution (χ²={chi2_stat:.1f}, p={chi2_p:.2e})',
              fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Color bars by imbalance
for i, (bar, count) in enumerate(zip(bars, variant_counts_sorted.values)):
    pct_diff = abs(count - expected_per_variant) / expected_per_variant
    if pct_diff > 0.10:  # >10% imbalance
        bar.set_color('salmon')
    elif pct_diff > 0.05:  # >5% imbalance
        bar.set_color('khaki')

# 2. Continuous feature balance (p-values)
ax2 = plt.subplot(3, 3, 2)
continuous_balance_sorted = continuous_balance_df.sort_values('h_pvalue')
colors = ['green' if p >= 0.05 else 'red' for p in continuous_balance_sorted['h_pvalue']]
bars = ax2.barh(range(len(continuous_balance_sorted)),
                continuous_balance_sorted['h_pvalue'],
                color=colors, alpha=0.6, edgecolor='black', linewidth=0.5)
ax2.axvline(0.05, color='black', linestyle='--', linewidth=2, label='α=0.05')
ax2.set_yticks(range(len(continuous_balance_sorted)))
ax2.set_yticklabels(continuous_balance_sorted['feature'], fontsize=9)
ax2.set_xlabel('P-value (Kruskal-Wallis)', fontsize=11, fontweight='bold')
ax2.set_title('B. Continuous Feature Balance', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='x')
ax2.set_xlim(0, max(continuous_balance_sorted['h_pvalue'].max() * 1.1, 0.1))

# 3. Categorical feature balance (p-values)
ax3 = plt.subplot(3, 3, 3)
categorical_balance_sorted = categorical_balance_df.sort_values('chi2_pvalue')
colors = ['green' if p >= 0.05 else 'red' for p in categorical_balance_sorted['chi2_pvalue']]
bars = ax3.barh(range(len(categorical_balance_sorted)),
                categorical_balance_sorted['chi2_pvalue'],
                color=colors, alpha=0.6, edgecolor='black', linewidth=0.5)
ax3.axvline(0.05, color='black', linestyle='--', linewidth=2, label='α=0.05')
ax3.set_yticks(range(len(categorical_balance_sorted)))
ax3.set_yticklabels(categorical_balance_sorted['feature'], fontsize=9)
ax3.set_xlabel('P-value (Chi-square)', fontsize=11, fontweight='bold')
ax3.set_title('C. Categorical Feature Balance', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, axis='x')
ax3.set_xlim(0, max(categorical_balance_sorted['chi2_pvalue'].max() * 1.1, 0.1))

# 4. Customer age distribution by variant
ax4 = plt.subplot(3, 3, 4)
# Sample 10 variants for visibility
sampled_variants = df['COVERAGE_TREATMENT'].unique()[:10]
violin_data = [df[df['COVERAGE_TREATMENT'] == v]['IMPUTED_AGE'].dropna()
               for v in sampled_variants]
parts = ax4.violinplot(violin_data, positions=range(len(sampled_variants)),
                       showmeans=True, showmedians=True)
ax4.set_xlabel('Variant (showing first 10)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Customer Age', fontsize=11, fontweight='bold')
ax4.set_title('D. Customer Age by Variant', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

# 5. Customer income distribution by variant
ax5 = plt.subplot(3, 3, 5)
violin_data = [df[df['COVERAGE_TREATMENT'] == v]['MEDIAN_HOUSEHOLD_INCOME_2020'].dropna() / 1000
               for v in sampled_variants]
parts = ax5.violinplot(violin_data, positions=range(len(sampled_variants)),
                       showmeans=True, showmedians=True)
ax5.set_xlabel('Variant (showing first 10)', fontsize=11, fontweight='bold')
ax5.set_ylabel('Customer Income ($1000s)', fontsize=11, fontweight='bold')
ax5.set_title('E. Customer Income by Variant', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')

# 6. Premium distribution by variant
ax6 = plt.subplot(3, 3, 6)
violin_data = [df[df['COVERAGE_TREATMENT'] == v]['PIT_ANNUAL_PREMIUM'].dropna()
               for v in sampled_variants]
parts = ax6.violinplot(violin_data, positions=range(len(sampled_variants)),
                       showmeans=True, showmedians=True)
ax6.set_xlabel('Variant (showing first 10)', fontsize=11, fontweight='bold')
ax6.set_ylabel('Annual Premium ($)', fontsize=11, fontweight='bold')
ax6.set_title('F. Premium by Variant', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')

# 7. Temporal distribution (daily variant counts)
ax7 = plt.subplot(3, 3, 7)
daily_counts = df.groupby(['date', 'COVERAGE_TREATMENT']).size().reset_index(name='count')
# Plot total daily counts
daily_total = df.groupby('date').size()
ax7.plot(daily_total.index, daily_total.values, color='steelblue', linewidth=2)
ax7.set_xlabel('Date', fontsize=11, fontweight='bold')
ax7.set_ylabel('Daily Quote Count', fontsize=11, fontweight='bold')
ax7.set_title('G. Temporal Distribution of Quotes', fontsize=12, fontweight='bold')
ax7.grid(True, alpha=0.3)
plt.setp(ax7.xaxis.get_majorticklabels(), rotation=45, ha='right')

# 8. Existing customer features balance
ax8 = plt.subplot(3, 3, 8)
existing_customer_features = ['HAS_MULTIPLE_PET_DISCOUNT', 'HAS_DEBIT_CARD',
                               'HAS_STRONGLY_CONNECTED_USERS']
balance_summary = []
for feat in existing_customer_features:
    pct_by_variant = df.groupby('COVERAGE_TREATMENT')[feat].mean() * 100
    balance_summary.append({
        'feature': feat.replace('HAS_', '').replace('_', ' ').title(),
        'mean': pct_by_variant.mean(),
        'std': pct_by_variant.std(),
        'cv': pct_by_variant.std() / pct_by_variant.mean()
    })

balance_df = pd.DataFrame(balance_summary)
x_pos = range(len(balance_df))
bars = ax8.bar(x_pos, balance_df['mean'], yerr=balance_df['std'],
               color='steelblue', alpha=0.7, capsize=5, edgecolor='black', linewidth=0.5)
ax8.set_xticks(x_pos)
ax8.set_xticklabels([f.split()[0] for f in balance_df['feature']],
                     rotation=0, ha='center', fontsize=9)
ax8.set_ylabel('Mean % Across Variants', fontsize=11, fontweight='bold')
ax8.set_title('H. Existing Customer Features Balance', fontsize=12, fontweight='bold')
ax8.grid(True, alpha=0.3, axis='y')

# 9. Overall quality scorecard
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')

scorecard_text = f"""
RANDOMIZATION QUALITY SCORECARD

Overall Score: {randomization_score:.1f}%
Grade: {grade}

Tests Passed: {total_balanced}/{total_tests}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Details:

  ✓ Sample Size: {'Pass' if chi2_p >= 0.05 else 'Fail'}

  ✓ Continuous: {n_continuous_balanced}/{len(continuous_features)} balanced

  ✓ Categorical: {n_categorical_balanced}/{len(categorical_features)} balanced

  ✓ Temporal: {n_temporal_balanced}/2 balanced

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Assessment:

{quality}
"""

ax9.text(0.05, 0.95, scorecard_text, transform=ax9.transAxes,
         fontsize=11, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'randomization_quality_analysis.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: figures/randomization_quality_analysis.png")
plt.close()

print()
print("="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print()
print("Output files:")
print("   1. results/randomization_variant_counts.csv")
print("   2. results/randomization_continuous_balance.csv")
print("   3. results/randomization_categorical_balance.csv")
print("   4. results/randomization_temporal_balance.json")
print("   5. results/randomization_quality_summary.json")
print("   6. figures/randomization_quality_analysis.png")
print()
print(f"FINAL VERDICT: {grade} - {quality}")
print()
