"""
Stage 3: Exploratory Feature Analysis and Relationship to Target

This script performs comprehensive univariate and bivariate analysis of all predictive features
to understand their characteristics and relationships with conversion and premium outcomes.
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
plt.rcParams['figure.figsize'] = (12, 8)

# File paths
DATA_FILE = Path("data/03_cleaned_fixed.csv")
RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("STAGE 3: EXPLORATORY FEATURE ANALYSIS AND RELATIONSHIP TO TARGET")
print("=" * 80)

# Load feature metadata
with open(RESULTS_DIR / "02_feature_list.json", "r") as f:
    feature_metadata = json.load(f)

predictive_features = feature_metadata["predictive_features"]
print(f"\n✓ Loaded feature metadata: {len(predictive_features)} predictive features to analyze")

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n" + "=" * 80)
print("1. LOADING DATA")
print("=" * 80)

# Load data
df = pd.read_csv(DATA_FILE)
print(f"✓ Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

# Verify targets
target_conversion = "is_purchased"
target_premium = "premium_amount"
print(f"✓ Target variables: {target_conversion} (conversion), {target_premium} (premium)")

# Basic statistics
n_total = len(df)
n_converted = df[target_conversion].sum()
conversion_rate = n_converted / n_total
avg_premium_all = df[target_premium].mean()
avg_premium_converted = df[df[target_conversion] == 1][target_premium].mean()

print(f"\nTarget Statistics:")
print(f"  • Total quotes: {n_total:,}")
print(f"  • Conversions: {n_converted:,} ({conversion_rate:.2%})")
print(f"  • Avg premium (all): ${avg_premium_all:.2f}")
print(f"  • Avg premium (converted): ${avg_premium_converted:.2f}")

# ============================================================================
# 2. CATEGORIZE FEATURES BY TYPE
# ============================================================================
print("\n" + "=" * 80)
print("2. CATEGORIZING FEATURES")
print("=" * 80)

# Categorize features by data type
numerical_features = []
categorical_features = []
boolean_features = []

for feat in predictive_features:
    if feat not in df.columns:
        print(f"⚠ Warning: Feature {feat} not in dataset")
        continue

    dtype = df[feat].dtype
    n_unique = df[feat].nunique()

    if dtype in ['int64', 'float64']:
        if n_unique == 2 and set(df[feat].unique()).issubset({0, 1, np.nan}):
            boolean_features.append(feat)
        else:
            numerical_features.append(feat)
    else:
        categorical_features.append(feat)

print(f"✓ Feature categorization:")
print(f"  • Numerical: {len(numerical_features)} features")
print(f"  • Categorical: {len(categorical_features)} features")
print(f"  • Boolean: {len(boolean_features)} features")

print(f"\nNumerical features: {numerical_features}")
print(f"Categorical features: {categorical_features}")
print(f"Boolean features: {boolean_features}")

# ============================================================================
# 3. UNIVARIATE ANALYSIS - NUMERICAL FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("3. UNIVARIATE ANALYSIS - NUMERICAL FEATURES")
print("=" * 80)

numerical_stats = []

for feat in numerical_features:
    series = df[feat]

    stats_dict = {
        'feature': feat,
        'count': series.count(),
        'missing': series.isna().sum(),
        'mean': series.mean(),
        'std': series.std(),
        'min': series.min(),
        'q25': series.quantile(0.25),
        'median': series.median(),
        'q75': series.quantile(0.75),
        'max': series.max(),
        'skewness': series.skew(),
        'kurtosis': series.kurtosis(),
        'n_unique': series.nunique(),
        'cv': series.std() / series.mean() if series.mean() != 0 else np.nan
    }

    # Detect outliers using IQR method
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = ((series < lower_bound) | (series > upper_bound)).sum()
    stats_dict['outliers_n'] = outliers
    stats_dict['outliers_pct'] = outliers / len(series) * 100

    numerical_stats.append(stats_dict)

    print(f"\n{feat}:")
    print(f"  Range: [{stats_dict['min']:.2f}, {stats_dict['max']:.2f}]")
    print(f"  Mean: {stats_dict['mean']:.2f} ± {stats_dict['std']:.2f}")
    print(f"  Median: {stats_dict['median']:.2f}")
    print(f"  Skewness: {stats_dict['skewness']:.3f}")
    print(f"  Outliers: {stats_dict['outliers_n']} ({stats_dict['outliers_pct']:.2f}%)")

# Save numerical statistics
numerical_stats_df = pd.DataFrame(numerical_stats)
numerical_stats_df.to_csv(RESULTS_DIR / "stage3_numerical_stats.csv", index=False)
print(f"\n✓ Saved numerical statistics to stage3_numerical_stats.csv")

# ============================================================================
# 4. UNIVARIATE ANALYSIS - CATEGORICAL & BOOLEAN FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("4. UNIVARIATE ANALYSIS - CATEGORICAL & BOOLEAN FEATURES")
print("=" * 80)

categorical_stats = []

for feat in categorical_features + boolean_features:
    series = df[feat]
    value_counts = series.value_counts()

    stats_dict = {
        'feature': feat,
        'type': 'boolean' if feat in boolean_features else 'categorical',
        'count': series.count(),
        'missing': series.isna().sum(),
        'n_unique': series.nunique(),
        'top_value': value_counts.index[0] if len(value_counts) > 0 else None,
        'top_freq': value_counts.iloc[0] if len(value_counts) > 0 else 0,
        'top_pct': value_counts.iloc[0] / len(series) * 100 if len(value_counts) > 0 else 0
    }

    # For boolean, report True/False split
    if feat in boolean_features:
        true_count = (series == 1).sum()
        false_count = (series == 0).sum()
        stats_dict['true_count'] = true_count
        stats_dict['true_pct'] = true_count / len(series) * 100
        stats_dict['false_count'] = false_count
        stats_dict['false_pct'] = false_count / len(series) * 100

    categorical_stats.append(stats_dict)

    print(f"\n{feat} ({stats_dict['type']}):")
    print(f"  Unique values: {stats_dict['n_unique']}")
    if feat in boolean_features:
        print(f"  True: {stats_dict['true_count']} ({stats_dict['true_pct']:.2f}%)")
        print(f"  False: {stats_dict['false_count']} ({stats_dict['false_pct']:.2f}%)")
    else:
        print(f"  Top value: {stats_dict['top_value']} ({stats_dict['top_pct']:.2f}%)")
        print(f"  Top 5 values:")
        for val, count in value_counts.head().items():
            print(f"    • {val}: {count:,} ({count/len(series)*100:.2f}%)")

# Save categorical statistics
categorical_stats_df = pd.DataFrame(categorical_stats)
categorical_stats_df.to_csv(RESULTS_DIR / "stage3_categorical_stats.csv", index=False)
print(f"\n✓ Saved categorical statistics to stage3_categorical_stats.csv")

# ============================================================================
# 5. BIVARIATE ANALYSIS - NUMERICAL vs TARGET
# ============================================================================
print("\n" + "=" * 80)
print("5. BIVARIATE ANALYSIS - NUMERICAL FEATURES vs TARGET")
print("=" * 80)

numerical_bivariate = []

for feat in numerical_features:
    # Compare feature distribution between converters and non-converters
    converted = df[df[target_conversion] == 1][feat]
    not_converted = df[df[target_conversion] == 0][feat]

    # T-test for difference in means
    t_stat, p_value = stats.ttest_ind(converted, not_converted, nan_policy='omit')

    # Mann-Whitney U test (non-parametric alternative)
    u_stat, u_pvalue = stats.mannwhitneyu(converted.dropna(), not_converted.dropna(), alternative='two-sided')

    # Cohen's d effect size
    mean_diff = converted.mean() - not_converted.mean()
    pooled_std = np.sqrt(((len(converted) - 1) * converted.std()**2 +
                          (len(not_converted) - 1) * not_converted.std()**2) /
                         (len(converted) + len(not_converted) - 2))
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

    # Point-biserial correlation (continuous vs binary)
    correlation, corr_pvalue = stats.pointbiserialr(df[target_conversion], df[feat].fillna(df[feat].median()))

    bivariate_dict = {
        'feature': feat,
        'mean_converted': converted.mean(),
        'mean_not_converted': not_converted.mean(),
        'mean_diff': mean_diff,
        'mean_diff_pct': (mean_diff / not_converted.mean() * 100) if not_converted.mean() != 0 else 0,
        't_statistic': t_stat,
        't_pvalue': p_value,
        'u_statistic': u_stat,
        'u_pvalue': u_pvalue,
        'cohens_d': cohens_d,
        'correlation': correlation,
        'correlation_pvalue': corr_pvalue,
        'significant': p_value < 0.05
    }

    numerical_bivariate.append(bivariate_dict)

    print(f"\n{feat}:")
    print(f"  Mean (converted): {converted.mean():.2f}")
    print(f"  Mean (not converted): {not_converted.mean():.2f}")
    print(f"  Difference: {mean_diff:.2f} ({bivariate_dict['mean_diff_pct']:.2f}%)")
    print(f"  Cohen's d: {cohens_d:.3f}")
    print(f"  t-test p-value: {p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''}")
    print(f"  Correlation: {correlation:.3f} (p={corr_pvalue:.4f})")

# Save numerical bivariate analysis
numerical_bivariate_df = pd.DataFrame(numerical_bivariate)
numerical_bivariate_df = numerical_bivariate_df.sort_values('cohens_d', key=abs, ascending=False)
numerical_bivariate_df.to_csv(RESULTS_DIR / "stage3_numerical_bivariate.csv", index=False)
print(f"\n✓ Saved numerical bivariate analysis to stage3_numerical_bivariate.csv")

# ============================================================================
# 6. BIVARIATE ANALYSIS - CATEGORICAL vs TARGET
# ============================================================================
print("\n" + "=" * 80)
print("6. BIVARIATE ANALYSIS - CATEGORICAL & BOOLEAN vs TARGET")
print("=" * 80)

categorical_bivariate = []

for feat in categorical_features + boolean_features:
    # Conversion rate by category
    conversion_by_cat = df.groupby(feat)[target_conversion].agg(['sum', 'count', 'mean'])
    conversion_by_cat.columns = ['conversions', 'total', 'conversion_rate']
    conversion_by_cat = conversion_by_cat.sort_values('conversion_rate', ascending=False)

    # Chi-square test of independence
    contingency_table = pd.crosstab(df[feat], df[target_conversion])
    chi2, chi2_pvalue, dof, expected = stats.chi2_contingency(contingency_table)

    # Cramér's V effect size
    n = contingency_table.sum().sum()
    cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))

    bivariate_dict = {
        'feature': feat,
        'n_categories': df[feat].nunique(),
        'chi2_statistic': chi2,
        'chi2_pvalue': chi2_pvalue,
        'cramers_v': cramers_v,
        'significant': chi2_pvalue < 0.05,
        'best_category': conversion_by_cat.index[0] if len(conversion_by_cat) > 0 else None,
        'best_conversion_rate': conversion_by_cat.iloc[0]['conversion_rate'] if len(conversion_by_cat) > 0 else 0,
        'worst_category': conversion_by_cat.index[-1] if len(conversion_by_cat) > 0 else None,
        'worst_conversion_rate': conversion_by_cat.iloc[-1]['conversion_rate'] if len(conversion_by_cat) > 0 else 0,
        'range_pct': (conversion_by_cat.iloc[0]['conversion_rate'] -
                      conversion_by_cat.iloc[-1]['conversion_rate']) * 100 if len(conversion_by_cat) > 0 else 0
    }

    categorical_bivariate.append(bivariate_dict)

    print(f"\n{feat}:")
    print(f"  Categories: {bivariate_dict['n_categories']}")
    print(f"  Chi-square p-value: {chi2_pvalue:.4f} {'***' if chi2_pvalue < 0.001 else '**' if chi2_pvalue < 0.01 else '*' if chi2_pvalue < 0.05 else ''}")
    print(f"  Cramér's V: {cramers_v:.3f}")
    print(f"  Best category: {bivariate_dict['best_category']} ({bivariate_dict['best_conversion_rate']:.2%})")
    print(f"  Worst category: {bivariate_dict['worst_category']} ({bivariate_dict['worst_conversion_rate']:.2%})")
    print(f"  Conversion rate range: {bivariate_dict['range_pct']:.2f} percentage points")

    # Show top categories for high-cardinality features
    if bivariate_dict['n_categories'] <= 10:
        print(f"  Conversion rates by category:")
        for cat, row in conversion_by_cat.head(10).iterrows():
            print(f"    • {cat}: {row['conversion_rate']:.2%} ({int(row['conversions'])}/{int(row['total'])})")

# Save categorical bivariate analysis
categorical_bivariate_df = pd.DataFrame(categorical_bivariate)
categorical_bivariate_df = categorical_bivariate_df.sort_values('cramers_v', ascending=False)
categorical_bivariate_df.to_csv(RESULTS_DIR / "stage3_categorical_bivariate.csv", index=False)
print(f"\n✓ Saved categorical bivariate analysis to stage3_categorical_bivariate.csv")

print("\n" + "=" * 80)
print("STAGE 3 ANALYSIS COMPLETE - Part 1")
print("=" * 80)
print(f"✓ Generated 4 analysis files in {RESULTS_DIR}/")
print("  Next: Feature importance ranking, visualizations, and deep dives")
