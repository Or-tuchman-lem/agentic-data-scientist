"""
Stage 3 Part 3: Comprehensive Feature Visualizations

Create publication-quality visualizations for feature analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
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
print("STAGE 3 PART 3: COMPREHENSIVE VISUALIZATIONS")
print("=" * 80)

# Load data
df = pd.read_csv(DATA_FILE)
feature_importance = pd.read_csv(RESULTS_DIR / "stage3_feature_importance_ranking.csv")
num_bivariate = pd.read_csv(RESULTS_DIR / "stage3_numerical_bivariate.csv")
cat_bivariate = pd.read_csv(RESULTS_DIR / "stage3_categorical_bivariate.csv")

print(f"✓ Loaded data and analysis results")

# ============================================================================
# VISUALIZATION 1: Feature Importance Ranking
# ============================================================================
print("\n1. Creating feature importance ranking visualization...")

fig, ax = plt.subplots(figsize=(14, 10))

# Exclude COVERAGE_SENT_AT (timestamp - not useful for modeling)
feature_importance_viz = feature_importance[feature_importance['feature'] != 'COVERAGE_SENT_AT'].head(14)

# Color by type
colors = ['#1f77b4' if t == 'numerical' else '#ff7f0e' for t in feature_importance_viz['type']]

# Horizontal bar chart
y_pos = np.arange(len(feature_importance_viz))
ax.barh(y_pos, feature_importance_viz['effect_size'], color=colors, alpha=0.7)

# Add value labels
for i, (idx, row) in enumerate(feature_importance_viz.iterrows()):
    ax.text(row['effect_size'] + 0.005, i, f"{row['effect_size']:.3f}",
            va='center', fontsize=9, fontweight='bold')

ax.set_yticks(y_pos)
ax.set_yticklabels(feature_importance_viz['feature'], fontsize=10)
ax.set_xlabel('Effect Size (Cohen\'s d or Cramér\'s V)', fontsize=12, fontweight='bold')
ax.set_title('Feature Importance Ranking: Predictive Power for Conversion\n(excluding timestamp features)',
             fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#1f77b4', alpha=0.7, label='Numerical'),
    Patch(facecolor='#ff7f0e', alpha=0.7, label='Categorical')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "stage3_feature_importance_ranking.png", dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved stage3_feature_importance_ranking.png")

# ============================================================================
# VISUALIZATION 2: Numerical Features Distribution - Converted vs Non-Converted
# ============================================================================
print("\n2. Creating numerical features distribution comparison...")

numerical_features = num_bivariate['feature'].tolist()

fig, axes = plt.subplots(3, 3, figsize=(18, 14))
axes = axes.flatten()

for i, feat in enumerate(numerical_features[:9]):  # Top 9 numerical features
    ax = axes[i]

    # Get data for converters and non-converters
    converted = df[df['is_purchased'] == 1][feat]
    not_converted = df[df['is_purchased'] == 0][feat]

    # Create violin plot
    parts = ax.violinplot([not_converted, converted], positions=[0, 1], widths=0.7,
                          showmeans=True, showmedians=True)

    # Color the violin plots
    for pc in parts['bodies']:
        pc.set_facecolor('#3498db')
        pc.set_alpha(0.6)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Not Converted', 'Converted'], fontsize=9)
    ax.set_title(f'{feat}', fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add mean values as text
    ax.text(0, converted.max() * 0.95, f'μ={not_converted.mean():.1f}',
            ha='center', fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    ax.text(1, converted.max() * 0.95, f'μ={converted.mean():.1f}',
            ha='center', fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

# Remove empty subplots
for j in range(len(numerical_features), len(axes)):
    fig.delaxes(axes[j])

plt.suptitle('Numerical Features: Distribution Comparison by Conversion Status',
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "stage3_numerical_distributions.png", dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved stage3_numerical_distributions.png")

# ============================================================================
# VISUALIZATION 3: Top Categorical Features - Conversion Rates
# ============================================================================
print("\n3. Creating categorical features conversion rate visualization...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

# Feature 1: HAS_MULTIPLE_PET_DISCOUNT
ax = axes[0]
multi_pet_data = df.groupby('HAS_MULTIPLE_PET_DISCOUNT')['is_purchased'].agg(['mean', 'count'])
multi_pet_data.index = ['No Multi-Pet Discount', 'Has Multi-Pet Discount']
bars = ax.bar(range(len(multi_pet_data)), multi_pet_data['mean'] * 100, color=['#e74c3c', '#2ecc71'], alpha=0.7)
ax.set_xticks(range(len(multi_pet_data)))
ax.set_xticklabels(multi_pet_data.index, fontsize=11)
ax.set_ylabel('Conversion Rate (%)', fontsize=11, fontweight='bold')
ax.set_title('HAS_MULTIPLE_PET_DISCOUNT\n(Strongest Predictor)', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for i, (idx, row) in enumerate(multi_pet_data.iterrows()):
    ax.text(i, row['mean'] * 100 + 2, f"{row['mean']*100:.1f}%\n(n={int(row['count']):,})",
            ha='center', fontsize=10, fontweight='bold')

# Feature 2: HAS_DEBIT_CARD
ax = axes[1]
debit_data = df.groupby('HAS_DEBIT_CARD')['is_purchased'].agg(['mean', 'count'])
debit_data.index = ['No Debit Card', 'Has Debit Card']
bars = ax.bar(range(len(debit_data)), debit_data['mean'] * 100, color=['#e74c3c', '#2ecc71'], alpha=0.7)
ax.set_xticks(range(len(debit_data)))
ax.set_xticklabels(debit_data.index, fontsize=11)
ax.set_ylabel('Conversion Rate (%)', fontsize=11, fontweight='bold')
ax.set_title('HAS_DEBIT_CARD\n(Existing Customer Signal)', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for i, (idx, row) in enumerate(debit_data.iterrows()):
    ax.text(i, row['mean'] * 100 + 1, f"{row['mean']*100:.1f}%\n(n={int(row['count']):,})",
            ha='center', fontsize=10, fontweight='bold')

# Feature 3: HAS_STRONGLY_CONNECTED_USERS (Graph DB feature)
ax = axes[2]
graph_data = df.groupby('HAS_STRONGLY_CONNECTED_USERS')['is_purchased'].agg(['mean', 'count'])
graph_data.index = ['Not Connected', 'Strongly Connected']
bars = ax.bar(range(len(graph_data)), graph_data['mean'] * 100, color=['#e74c3c', '#2ecc71'], alpha=0.7)
ax.set_xticks(range(len(graph_data)))
ax.set_xticklabels(graph_data.index, fontsize=11)
ax.set_ylabel('Conversion Rate (%)', fontsize=11, fontweight='bold')
ax.set_title('HAS_STRONGLY_CONNECTED_USERS\n(Graph Database Feature)', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for i, (idx, row) in enumerate(graph_data.iterrows()):
    ax.text(i, row['mean'] * 100 + 0.5, f"{row['mean']*100:.1f}%\n(n={int(row['count']):,})",
            ha='center', fontsize=10, fontweight='bold')

# Feature 4: Top States
ax = axes[3]
state_data = df.groupby('STATE')['is_purchased'].agg(['mean', 'count'])
state_data = state_data[state_data['count'] >= 100].sort_values('mean', ascending=False).head(10)
bars = ax.barh(range(len(state_data)), state_data['mean'] * 100, color='#3498db', alpha=0.7)
ax.set_yticks(range(len(state_data)))
ax.set_yticklabels(state_data.index, fontsize=10)
ax.set_xlabel('Conversion Rate (%)', fontsize=11, fontweight='bold')
ax.set_title('Top 10 States by Conversion Rate\n(min 100 quotes)', fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
for i, (idx, row) in enumerate(state_data.iterrows()):
    ax.text(row['mean'] * 100 + 0.3, i, f"{row['mean']*100:.1f}%",
            va='center', fontsize=9, fontweight='bold')

plt.suptitle('Top Categorical Features: Conversion Rate Analysis',
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "stage3_categorical_conversion_rates.png", dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved stage3_categorical_conversion_rates.png")

# ============================================================================
# VISUALIZATION 4: Correlation Heatmap
# ============================================================================
print("\n4. Creating correlation heatmap...")

numerical_features_for_corr = num_bivariate['feature'].tolist()
numerical_features_for_corr.append('is_purchased')

corr_matrix = df[numerical_features_for_corr].corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.3f', cmap='coolwarm',
            center=0, vmin=-0.5, vmax=0.5, square=True, linewidths=0.5,
            cbar_kws={'label': 'Correlation Coefficient'}, ax=ax)
ax.set_title('Numerical Features: Correlation Matrix\n(with Target Variable: is_purchased)',
             fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "stage3_correlation_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved stage3_correlation_heatmap.png")

# ============================================================================
# VISUALIZATION 5: Income Distribution by Conversion Status
# ============================================================================
print("\n5. Creating income distribution visualization...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: Distribution comparison
ax = axes[0]
converted = df[df['is_purchased'] == 1]['MEDIAN_HOUSEHOLD_INCOME_2020']
not_converted = df[df['is_purchased'] == 0]['MEDIAN_HOUSEHOLD_INCOME_2020']

ax.hist(not_converted, bins=50, alpha=0.5, label='Not Converted', color='#e74c3c', density=True)
ax.hist(converted, bins=50, alpha=0.5, label='Converted', color='#2ecc71', density=True)
ax.axvline(not_converted.mean(), color='#e74c3c', linestyle='--', linewidth=2, label=f'Mean (NC): ${not_converted.mean():,.0f}')
ax.axvline(converted.mean(), color='#2ecc71', linestyle='--', linewidth=2, label=f'Mean (C): ${converted.mean():,.0f}')
ax.set_xlabel('Median Household Income (2020)', fontsize=11, fontweight='bold')
ax.set_ylabel('Density', fontsize=11, fontweight='bold')
ax.set_title('Income Distribution by Conversion Status', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

# Right: Conversion rate by income quartile
ax = axes[1]
df_copy = df.copy()
df_copy['income_quartile'] = pd.qcut(df['MEDIAN_HOUSEHOLD_INCOME_2020'], q=4, labels=['Q1\n(Lowest)', 'Q2', 'Q3', 'Q4\n(Highest)'])
income_quartile_conv = df_copy.groupby('income_quartile')['is_purchased'].agg(['mean', 'count'])
bars = ax.bar(range(len(income_quartile_conv)), income_quartile_conv['mean'] * 100, color='#3498db', alpha=0.7)
ax.set_xticks(range(len(income_quartile_conv)))
ax.set_xticklabels(income_quartile_conv.index, fontsize=11)
ax.set_ylabel('Conversion Rate (%)', fontsize=11, fontweight='bold')
ax.set_title('Conversion Rate by Income Quartile', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for i, (idx, row) in enumerate(income_quartile_conv.iterrows()):
    ax.text(i, row['mean'] * 100 + 0.5, f"{row['mean']*100:.1f}%",
            ha='center', fontsize=10, fontweight='bold')

plt.suptitle('Household Income: Strongest Numerical Predictor (Cohen\'s d = 0.270)',
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "stage3_income_analysis.png", dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved stage3_income_analysis.png")

# ============================================================================
# VISUALIZATION 6: Feature Group Importance
# ============================================================================
print("\n6. Creating feature group importance visualization...")

group_importance = pd.read_csv(RESULTS_DIR / "stage3_feature_group_importance.csv")

fig, ax = plt.subplots(figsize=(12, 8))
y_pos = np.arange(len(group_importance))
colors_map = {'context': '#e74c3c', 'pricing': '#3498db', 'customer_demographics': '#2ecc71',
              'existing_customer': '#f39c12', 'pet_characteristics': '#9b59b6'}
colors = [colors_map.get(g, '#95a5a6') for g in group_importance['group']]

bars = ax.barh(y_pos, group_importance['avg_effect_size'], color=colors, alpha=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels([g.replace('_', ' ').title() for g in group_importance['group']], fontsize=11)
ax.set_xlabel('Average Effect Size', fontsize=12, fontweight='bold')
ax.set_title('Feature Group Importance: Average Effect Size by Category',
             fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

for i, (idx, row) in enumerate(group_importance.iterrows()):
    ax.text(row['avg_effect_size'] + 0.01, i,
            f"{row['avg_effect_size']:.3f}  ({row['n_significant']}/{row['n_features']} sig.)",
            va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(FIGURES_DIR / "stage3_feature_group_importance.png", dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved stage3_feature_group_importance.png")

# ============================================================================
# VISUALIZATION 7: Interaction Effects
# ============================================================================
print("\n7. Creating interaction effects visualization...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: Multi-Pet × Debit Card
ax = axes[0]
interaction_data = df.groupby(['HAS_MULTIPLE_PET_DISCOUNT', 'HAS_DEBIT_CARD'])['is_purchased'].agg(['mean', 'count'])
interaction_pivot = interaction_data['mean'].unstack() * 100

x = np.arange(2)
width = 0.35
bars1 = ax.bar(x - width/2, interaction_pivot[False], width, label='No Debit Card', color='#e74c3c', alpha=0.7)
bars2 = ax.bar(x + width/2, interaction_pivot[True], width, label='Has Debit Card', color='#2ecc71', alpha=0.7)

ax.set_xticks(x)
ax.set_xticklabels(['No Multi-Pet\nDiscount', 'Has Multi-Pet\nDiscount'], fontsize=11)
ax.set_ylabel('Conversion Rate (%)', fontsize=11, fontweight='bold')
ax.set_title('Interaction: Multi-Pet Discount × Debit Card', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Right: Income Quartile × Top States
ax = axes[1]
df_copy = df.copy()
df_copy['income_quartile'] = pd.qcut(df['MEDIAN_HOUSEHOLD_INCOME_2020'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
top_states = ['CA', 'TX', 'FL', 'PA', 'IL']

for state in top_states:
    state_data = df_copy[df_copy['STATE'] == state]
    income_conv = state_data.groupby('income_quartile')['is_purchased'].mean() * 100
    ax.plot(income_conv.index, income_conv.values, marker='o', linewidth=2, label=state, markersize=8)

ax.set_xlabel('Income Quartile', fontsize=11, fontweight='bold')
ax.set_ylabel('Conversion Rate (%)', fontsize=11, fontweight='bold')
ax.set_title('Interaction: Income Quartile × State\n(Top 5 States by Volume)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10, title='State')
ax.grid(True, alpha=0.3)

plt.suptitle('Interaction Effects: Feature Combinations Show Non-Additive Patterns',
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "stage3_interaction_effects.png", dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved stage3_interaction_effects.png")

print("\n" + "=" * 80)
print("STAGE 3 VISUALIZATIONS COMPLETE")
print("=" * 80)
print("✓ Created 7 comprehensive visualizations in figures/")
print("  1. Feature importance ranking")
print("  2. Numerical distributions by conversion")
print("  3. Categorical conversion rates")
print("  4. Correlation heatmap")
print("  5. Income analysis")
print("  6. Feature group importance")
print("  7. Interaction effects")
