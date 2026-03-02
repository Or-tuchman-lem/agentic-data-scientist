"""
Stage 1 (Part 3): Fix Income Sentinel Values and Create Visualizations

This script fixes the income data sentinel values and creates comprehensive
visualizations for understanding the dataset.

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
print("STAGE 1 (PART 3): FIX INCOME DATA AND CREATE VISUALIZATIONS")
print("=" * 80)
print()

# ============================================================================
# LOAD DATA
# ============================================================================
print("Loading data...")
df = pd.read_csv(DATA_DIR / '02_cleaned_features.csv')
print(f"✓ Loaded {len(df):,} rows and {len(df.columns)} columns")
print()

# ============================================================================
# FIX INCOME SENTINEL VALUES
# ============================================================================
print("STEP 1: Fixing income sentinel values...")

# Check for sentinel values
income_col = 'MEDIAN_HOUSEHOLD_INCOME_2020'
sentinel_count = (df[income_col] < 0).sum()
print(f"Found {sentinel_count:,} negative/sentinel values in {income_col}")
print(f"Min value: {df[income_col].min()}")
print(f"Sentinel values distribution:")
print(df[df[income_col] < 0][income_col].value_counts())
print()

# Replace sentinel values with NaN, then fill with state median
df.loc[df[income_col] < 0, income_col] = np.nan
missing_after_sentinel = df[income_col].isnull().sum()
print(f"Converted {sentinel_count} sentinel values to NaN")

# Fill with state median
state_median_income = df.groupby('STATE')[income_col].transform('median')
df[income_col] = df[income_col].fillna(state_median_income)

# If still missing, fill with overall median
overall_median = df[income_col].median()
df[income_col] = df[income_col].fillna(overall_median)

print(f"✓ Fixed income data:")
print(f"  New min: ${df[income_col].min():,.0f}")
print(f"  New median: ${df[income_col].median():,.0f}")
print(f"  New max: ${df[income_col].max():,.0f}")
print(f"  New mean: ${df[income_col].mean():,.0f}")
print()

# Save fixed dataset
df.to_csv(DATA_DIR / '03_cleaned_fixed.csv', index=False)
print(f"✓ Saved fixed dataset: data/03_cleaned_fixed.csv")
print()

# ============================================================================
# VISUALIZATION 1: VARIANT PERFORMANCE
# ============================================================================
print("STEP 2: Creating visualizations...")
print("\n2.1 Variant performance visualization...")

# Calculate variant metrics
variant_perf = df.groupby('COVERAGE_TREATMENT').agg({
    'is_purchased': ['count', 'sum', 'mean'],
    'premium_amount': 'sum'
}).round(4)

variant_perf.columns = ['quotes', 'purchases', 'conversion_rate', 'total_sales']
variant_perf['avg_sales_per_quote'] = (variant_perf['total_sales'] / variant_perf['quotes']).round(2)
variant_perf = variant_perf.reset_index()

# Parse components
variant_perf[['coinsurance', 'deductible', 'coverage_limit']] = (
    variant_perf['COVERAGE_TREATMENT'].str.split('_', expand=True).astype(int)
)

# Sort by performance
variant_perf_sorted = variant_perf.sort_values('avg_sales_per_quote', ascending=False)

# Create figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Avg sales per quote by variant
ax1 = axes[0, 0]
colors = ['red' if v == '80_250_20000' else 'steelblue' for v in variant_perf_sorted['COVERAGE_TREATMENT']]
ax1.barh(range(len(variant_perf_sorted)), variant_perf_sorted['avg_sales_per_quote'], color=colors)
ax1.set_yticks(range(len(variant_perf_sorted)))
ax1.set_yticklabels(variant_perf_sorted['COVERAGE_TREATMENT'], fontsize=8)
ax1.set_xlabel('Average Sales per Quote ($)', fontsize=10)
ax1.set_title('Variant Performance (Red = Current Production)', fontsize=12, fontweight='bold')
ax1.axvline(x=variant_perf_sorted['avg_sales_per_quote'].mean(), color='orange', linestyle='--', label='Mean')
ax1.legend()
ax1.grid(axis='x', alpha=0.3)

# Plot 2: Conversion rate by variant
ax2 = axes[0, 1]
variant_perf_sorted_conv = variant_perf.sort_values('conversion_rate', ascending=False)
colors2 = ['red' if v == '80_250_20000' else 'green' for v in variant_perf_sorted_conv['COVERAGE_TREATMENT']]
ax2.barh(range(len(variant_perf_sorted_conv)), variant_perf_sorted_conv['conversion_rate'], color=colors2)
ax2.set_yticks(range(len(variant_perf_sorted_conv)))
ax2.set_yticklabels(variant_perf_sorted_conv['COVERAGE_TREATMENT'], fontsize=8)
ax2.set_xlabel('Conversion Rate', fontsize=10)
ax2.set_title('Conversion Rate by Variant (Red = Current Production)', fontsize=12, fontweight='bold')
ax2.axvline(x=variant_perf_sorted_conv['conversion_rate'].mean(), color='orange', linestyle='--', label='Mean')
ax2.legend()
ax2.grid(axis='x', alpha=0.3)

# Plot 3: Avg sales per quote by coinsurance level
ax3 = axes[1, 0]
coin_perf = variant_perf.groupby('coinsurance').agg({
    'avg_sales_per_quote': 'mean',
    'conversion_rate': 'mean'
}).reset_index()
ax3.bar(coin_perf['coinsurance'].astype(str), coin_perf['avg_sales_per_quote'], color='steelblue')
ax3.set_xlabel('Coinsurance Level (%)', fontsize=10)
ax3.set_ylabel('Avg Sales per Quote ($)', fontsize=10)
ax3.set_title('Performance by Coinsurance Level', fontsize=12, fontweight='bold')
for i, v in enumerate(coin_perf['avg_sales_per_quote']):
    ax3.text(i, v + 1, f'${v:.2f}', ha='center', va='bottom', fontweight='bold')

# Plot 4: Avg sales per quote by deductible
ax4 = axes[1, 1]
ded_perf = variant_perf.groupby('deductible').agg({
    'avg_sales_per_quote': 'mean',
    'conversion_rate': 'mean'
}).reset_index()
ax4.bar(ded_perf['deductible'].astype(str), ded_perf['avg_sales_per_quote'], color='coral')
ax4.set_xlabel('Deductible ($)', fontsize=10)
ax4.set_ylabel('Avg Sales per Quote ($)', fontsize=10)
ax4.set_title('Performance by Deductible', fontsize=12, fontweight='bold')
for i, v in enumerate(ded_perf['avg_sales_per_quote']):
    ax4.text(i, v + 1, f'${v:.2f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(FIGURES_DIR / '03_variant_performance.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: figures/03_variant_performance.png")

# ============================================================================
# VISUALIZATION 2: CUSTOMER & PET CHARACTERISTICS
# ============================================================================
print("\n2.2 Customer and pet characteristics visualization...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Age distribution by purchase status
ax1 = axes[0, 0]
df[df['is_purchased'] == 1]['IMPUTED_AGE'].hist(bins=30, alpha=0.6, label='Purchased', ax=ax1, color='green')
df[df['is_purchased'] == 0]['IMPUTED_AGE'].hist(bins=30, alpha=0.6, label='Not Purchased', ax=ax1, color='red')
ax1.set_xlabel('Customer Age', fontsize=10)
ax1.set_ylabel('Frequency', fontsize=10)
ax1.set_title('Age Distribution by Purchase Status', fontsize=11, fontweight='bold')
ax1.legend()

# Pet age distribution
ax2 = axes[0, 1]
df[df['is_purchased'] == 1]['PET_AGE_YEARS'].hist(bins=15, alpha=0.6, label='Purchased', ax=ax2, color='green')
df[df['is_purchased'] == 0]['PET_AGE_YEARS'].hist(bins=15, alpha=0.6, label='Not Purchased', ax=ax2, color='red')
ax2.set_xlabel('Pet Age (Years)', fontsize=10)
ax2.set_ylabel('Frequency', fontsize=10)
ax2.set_title('Pet Age Distribution by Purchase Status', fontsize=11, fontweight='bold')
ax2.legend()

# Conversion by pet sex
ax3 = axes[0, 2]
sex_conv = df.groupby('PET_SEX')['is_purchased'].mean()
ax3.bar(sex_conv.index, sex_conv.values, color=['steelblue', 'coral'])
ax3.set_ylabel('Conversion Rate', fontsize=10)
ax3.set_title('Conversion Rate by Pet Sex', fontsize=11, fontweight='bold')
for i, v in enumerate(sex_conv.values):
    ax3.text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

# Conversion by multiple pet discount
ax4 = axes[1, 0]
mpd_conv = df.groupby('HAS_MULTIPLE_PET_DISCOUNT')['is_purchased'].mean()
ax4.bar(['No', 'Yes'], mpd_conv.values, color=['lightcoral', 'lightgreen'])
ax4.set_ylabel('Conversion Rate', fontsize=10)
ax4.set_title('Conversion: Multiple Pet Discount', fontsize=11, fontweight='bold')
for i, v in enumerate(mpd_conv.values):
    ax4.text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

# Conversion by connected users
ax5 = axes[1, 1]
conn_conv = df.groupby('HAS_STRONGLY_CONNECTED_USERS')['is_purchased'].mean()
ax5.bar(['No', 'Yes'], conn_conv.values, color=['lightcoral', 'lightgreen'])
ax5.set_ylabel('Conversion Rate', fontsize=10)
ax5.set_title('Conversion: Strongly Connected Users', fontsize=11, fontweight='bold')
for i, v in enumerate(conn_conv.values):
    ax5.text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

# Top 10 states by quotes
ax6 = axes[1, 2]
top_states = df['STATE'].value_counts().head(10)
ax6.barh(range(len(top_states)), top_states.values, color='steelblue')
ax6.set_yticks(range(len(top_states)))
ax6.set_yticklabels(top_states.index, fontsize=9)
ax6.set_xlabel('Number of Quotes', fontsize=10)
ax6.set_title('Top 10 States by Quote Volume', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(FIGURES_DIR / '03_customer_pet_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: figures/03_customer_pet_analysis.png")

# ============================================================================
# VISUALIZATION 3: PRICING AND PREMIUM ANALYSIS
# ============================================================================
print("\n2.3 Pricing and premium analysis...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Premium distribution by purchase status
ax1 = axes[0, 0]
purchased_premiums = df[df['is_purchased'] == 1]['PIT_ANNUAL_PREMIUM']
not_purchased_premiums = df[df['is_purchased'] == 0]['PIT_ANNUAL_PREMIUM']
ax1.hist([purchased_premiums, not_purchased_premiums], bins=50, label=['Purchased', 'Not Purchased'],
         color=['green', 'red'], alpha=0.6)
ax1.set_xlabel('Annual Premium ($)', fontsize=10)
ax1.set_ylabel('Frequency', fontsize=10)
ax1.set_title('Premium Distribution by Purchase Status', fontsize=11, fontweight='bold')
ax1.legend()
ax1.axvline(purchased_premiums.mean(), color='darkgreen', linestyle='--', linewidth=2, label=f'Purchased Mean: ${purchased_premiums.mean():.0f}')
ax1.axvline(not_purchased_premiums.mean(), color='darkred', linestyle='--', linewidth=2, label=f'Not Purchased Mean: ${not_purchased_premiums.mean():.0f}')

# Scatter: Premium vs Conversion by variant
ax2 = axes[0, 1]
avg_premium_by_variant = df.groupby('COVERAGE_TREATMENT').agg({
    'PIT_ANNUAL_PREMIUM': 'mean',
    'is_purchased': 'mean'
}).reset_index()
scatter = ax2.scatter(avg_premium_by_variant['PIT_ANNUAL_PREMIUM'],
                     avg_premium_by_variant['is_purchased'],
                     c=avg_premium_by_variant['is_purchased'],
                     cmap='RdYlGn', s=100, alpha=0.7, edgecolors='black')
ax2.set_xlabel('Average Annual Premium ($)', fontsize=10)
ax2.set_ylabel('Conversion Rate', fontsize=10)
ax2.set_title('Premium vs Conversion Rate by Variant', fontsize=11, fontweight='bold')
plt.colorbar(scatter, ax=ax2, label='Conversion Rate')

# Premium by coverage limit
ax3 = axes[1, 0]
limit_premium = variant_perf.groupby('coverage_limit')['avg_sales_per_quote'].mean()
ax3.bar(limit_premium.index.astype(str), limit_premium.values, color='purple', alpha=0.7)
ax3.set_xlabel('Coverage Limit ($)', fontsize=10)
ax3.set_ylabel('Avg Sales per Quote ($)', fontsize=10)
ax3.set_title('Performance by Coverage Limit', fontsize=11, fontweight='bold')
for i, v in enumerate(limit_premium.values):
    ax3.text(i, v + 1, f'${v:.2f}', ha='center', va='bottom', fontweight='bold')

# Income vs Conversion
ax4 = axes[1, 1]
# Create income bins
df['income_bin'] = pd.cut(df[income_col], bins=10)
income_conv = df.groupby('income_bin')['is_purchased'].mean()
income_labels = [f'${int(i.left/1000)}-{int(i.right/1000)}K' for i in income_conv.index]
ax4.plot(range(len(income_conv)), income_conv.values, marker='o', linewidth=2, markersize=8, color='teal')
ax4.set_xticks(range(len(income_conv)))
ax4.set_xticklabels(income_labels, rotation=45, ha='right', fontsize=8)
ax4.set_ylabel('Conversion Rate', fontsize=10)
ax4.set_title('Conversion Rate by Household Income', fontsize=11, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / '03_pricing_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: figures/03_pricing_analysis.png")

# ============================================================================
# VISUALIZATION 4: CORRELATION HEATMAP
# ============================================================================
print("\n2.4 Feature correlation heatmap...")

# Select numeric features for correlation
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# Remove redundant columns
numeric_cols = [c for c in numeric_cols if c not in ['CONVERTED', 'SALES', 'premium_amount']]

# Calculate correlations with target
correlations = df[numeric_cols].corr()['is_purchased'].sort_values(ascending=False)

# Create heatmap for top features
top_features = correlations.abs().sort_values(ascending=False).head(15).index.tolist()
corr_matrix = df[top_features].corr()

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Feature Correlation Heatmap (Top 15 by Target Correlation)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / '03_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: figures/03_correlation_heatmap.png")

# Save correlation rankings
correlations_df = pd.DataFrame({
    'feature': correlations.index,
    'correlation_with_purchase': correlations.values
})
correlations_df.to_csv(RESULTS_DIR / '03_feature_correlations.csv', index=False)
print("✓ Saved: results/03_feature_correlations.csv")

print()
print("=" * 80)
print("VISUALIZATIONS COMPLETE")
print("=" * 80)
print("\nGenerated 4 visualization files:")
print("  1. figures/03_variant_performance.png")
print("  2. figures/03_customer_pet_analysis.png")
print("  3. figures/03_pricing_analysis.png")
print("  4. figures/03_correlation_heatmap.png")
