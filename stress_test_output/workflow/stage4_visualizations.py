"""
Stage 4: Visualizations for Feature Engineering Results

Creates publication-quality visualizations showing:
1. Top engineered features by effect size
2. Composite score distributions by conversion
3. Key interaction effects
4. Feature category performance summary

Author: Agentic Data Scientist
Date: 2026-02-26
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

print("="*80)
print("STAGE 4: CREATING VISUALIZATIONS")
print("="*80)
print()

# Load data
print("[1/4] Loading data...")
df = pd.read_csv('data/04_engineered_features.csv')
df_all_features = pd.read_csv('results/stage4_all_features_ranked.csv')
df_continuous = pd.read_csv('results/stage4_continuous_features_tested.csv')
df_scores = pd.read_csv('results/stage4_composite_scores_tested.csv')
print(f"✓ Loaded data: {len(df):,} rows")
print()

# ============================================================================
# FIGURE 1: Top 20 Engineered Features by Effect Size
# ============================================================================
print("[2/4] Creating Figure 1: Top engineered features...")

fig, ax = plt.subplots(figsize=(12, 8))

# Get top 20 features
top_features = df_all_features.head(20).copy()
top_features['feature_short'] = top_features['feature'].str.replace('_', ' ').str.title()

# Color by significance
colors = ['#d62728' if p < 0.001 else '#ff7f0e' if p < 0.01 else '#2ca02c'
          for p in top_features['p_value']]

# Horizontal bar chart
bars = ax.barh(range(len(top_features)), top_features['effect_size'], color=colors)
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['feature_short'], fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Effect Size (Cohen\'s d or Cramér\'s V)', fontsize=11, fontweight='bold')
ax.set_title('Top 20 Engineered Features by Effect Size', fontsize=13, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

# Add value labels
for i, (idx, row) in enumerate(top_features.iterrows()):
    sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*"
    ax.text(row['effect_size'] + 0.01, i, f"{row['effect_size']:.3f}{sig}",
            va='center', fontsize=8)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#d62728', label='p < 0.001 (highly significant)'),
    Patch(facecolor='#ff7f0e', label='p < 0.01 (very significant)'),
    Patch(facecolor='#2ca02c', label='p < 0.05 (significant)')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig('figures/stage4_top_features_ranked.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: figures/stage4_top_features_ranked.png")
print()

# ============================================================================
# FIGURE 2: Composite Score Distributions by Conversion
# ============================================================================
print("[3/4] Creating Figure 2: Composite score distributions...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

composite_scores = ['PROPENSITY_SCORE', 'CUSTOMER_VALUE_SCORE', 'ENGAGEMENT_SCORE', 'EXISTING_CUSTOMER_SCORE']

for idx, score in enumerate(composite_scores):
    ax = axes[idx]

    # Get data and ensure numeric type
    converters = df[df['CONVERTED'] == 1][score].dropna().astype(float)
    non_converters = df[df['CONVERTED'] == 0][score].dropna().astype(float)

    # Violin plots
    parts = ax.violinplot([non_converters, converters], positions=[0, 1],
                           showmeans=True, showmedians=True, widths=0.7)

    # Color the violins
    for pc, color in zip(parts['bodies'], ['#1f77b4', '#ff7f0e']):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)

    # Formatting
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Not Converted', 'Converted'])
    ax.set_ylabel(score.replace('_', ' ').title(), fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add effect size
    effect_size = df_scores[df_scores['feature'] == score]['effect_size'].values[0]
    correlation = df_scores[df_scores['feature'] == score]['correlation'].values[0]
    ax.text(0.5, 0.95, f"Cohen's d = {effect_size:.3f}\nr = {correlation:.3f}",
            transform=ax.transAxes, fontsize=9, va='top', ha='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

fig.suptitle('Composite Score Distributions by Conversion Status',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('figures/stage4_composite_scores.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: figures/stage4_composite_scores.png")
print()

# ============================================================================
# FIGURE 3: Key Interaction Effects
# ============================================================================
print("[4/4] Creating Figure 3: Interaction effects...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: MULTIPET_X_DEBIT interaction
ax = axes[0, 0]
interaction_data = df.groupby(['HAS_MULTIPLE_PET_DISCOUNT', 'HAS_DEBIT_CARD'])['CONVERTED'].agg(['mean', 'count'])
interaction_data = interaction_data.reset_index()
interaction_data['label'] = interaction_data.apply(
    lambda x: f"MultiPet={x['HAS_MULTIPLE_PET_DISCOUNT']}, Debit={x['HAS_DEBIT_CARD']}", axis=1
)

bars = ax.bar(range(len(interaction_data)), interaction_data['mean'],
              color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
ax.set_xticks(range(len(interaction_data)))
ax.set_xticklabels(interaction_data['label'], rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Conversion Rate', fontsize=10, fontweight='bold')
ax.set_title('MULTIPET × DEBIT Interaction (d=0.369)', fontsize=11, fontweight='bold')
ax.set_ylim(0, 0.7)
ax.grid(axis='y', alpha=0.3)

for i, (idx, row) in enumerate(interaction_data.iterrows()):
    ax.text(i, row['mean'] + 0.02, f"{row['mean']:.1%}\n(n={int(row['count'])})",
            ha='center', fontsize=8)

# Panel 2: PROPENSITY_SCORE by quantiles
ax = axes[0, 1]
df['PROPENSITY_QUANTILE'] = pd.qcut(df['PROPENSITY_SCORE'], q=5, labels=False, duplicates='drop')
quantile_conv = df.groupby('PROPENSITY_QUANTILE')['CONVERTED'].agg(['mean', 'count'])

bars = ax.bar(range(len(quantile_conv)), quantile_conv['mean'])
ax.set_xticks(range(len(quantile_conv)))
ax.set_xticklabels(quantile_conv.index)
ax.set_ylabel('Conversion Rate', fontsize=10, fontweight='bold')
ax.set_xlabel('Propensity Score Quantile', fontsize=10, fontweight='bold')
ax.set_title('Propensity Score by Quantile (d=0.522)', fontsize=11, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

for i, (idx, row) in enumerate(quantile_conv.iterrows()):
    ax.text(i, row['mean'] + 0.01, f"{row['mean']:.1%}", ha='center', fontsize=9)

# Panel 3: Income × Premium interaction (affordability)
ax = axes[1, 0]
df['INCOME_QUARTILE'] = pd.qcut(df['MEDIAN_HOUSEHOLD_INCOME_2020'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
df['PREMIUM_QUARTILE'] = pd.qcut(df['PIT_ANNUAL_PREMIUM'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')

# Heatmap of conversion rates
heatmap_data = df.groupby(['INCOME_QUARTILE', 'PREMIUM_QUARTILE'])['CONVERTED'].mean().unstack()

im = ax.imshow(heatmap_data.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=0.25)
ax.set_xticks(range(len(heatmap_data.columns)))
ax.set_yticks(range(len(heatmap_data.index)))
ax.set_xticklabels(heatmap_data.columns)
ax.set_yticklabels(heatmap_data.index)
ax.set_xlabel('Premium Quartile (Q1=Low, Q4=High)', fontsize=10, fontweight='bold')
ax.set_ylabel('Income Quartile (Q1=Low, Q4=High)', fontsize=10, fontweight='bold')
ax.set_title('Income × Premium Interaction\n(Affordability Effect)', fontsize=11, fontweight='bold')

# Add text annotations
for i in range(len(heatmap_data.index)):
    for j in range(len(heatmap_data.columns)):
        text = ax.text(j, i, f"{heatmap_data.iloc[i, j]:.1%}",
                      ha="center", va="center", color="black", fontsize=8)

# Colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Conversion Rate', rotation=270, labelpad=15)

# Panel 4: Feature Category Performance
ax = axes[1, 1]

# Calculate average effect size by category
category_performance = []
categories = {
    'Composite Scores': ['PROPENSITY_SCORE', 'CUSTOMER_VALUE_SCORE', 'ENGAGEMENT_SCORE', 'EXISTING_CUSTOMER_SCORE'],
    'Interactions': [f for f in df_all_features['feature'] if '_X_' in f],
    'Binary Flags': [f for f in df_all_features['feature'] if f.startswith('IS_')],
    'Transformations': ['LOG_INCOME', 'SQRT_INCOME', 'LOG_PREMIUM', 'SQRT_PREMIUM', 'PREMIUM_TO_INCOME_RATIO'],
    'State Features': [f for f in df_all_features['feature'] if 'STATE_' in f],
    'Time Features': [f for f in df_all_features['feature'] if any(x in f for x in ['QUOTE_', 'IS_WEEKEND', 'IS_BUSINESS', 'IS_EVENING', 'DAYS_SINCE'])]
}

for cat_name, features in categories.items():
    cat_features = df_all_features[df_all_features['feature'].isin(features)]
    if len(cat_features) > 0:
        avg_effect = cat_features['effect_size'].mean()
        pct_sig = (cat_features['p_value'] < 0.05).mean() * 100
        category_performance.append({
            'category': cat_name,
            'avg_effect_size': avg_effect,
            'pct_significant': pct_sig,
            'n_features': len(cat_features)
        })

df_cat = pd.DataFrame(category_performance).sort_values('avg_effect_size', ascending=True)

bars = ax.barh(range(len(df_cat)), df_cat['avg_effect_size'])
ax.set_yticks(range(len(df_cat)))
ax.set_yticklabels(df_cat['category'])
ax.set_xlabel('Average Effect Size', fontsize=10, fontweight='bold')
ax.set_title('Feature Category Performance', fontsize=11, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

for i, (idx, row) in enumerate(df_cat.iterrows()):
    ax.text(row['avg_effect_size'] + 0.005, i,
            f"{row['avg_effect_size']:.3f}\n({int(row['n_features'])} features)",
            va='center', fontsize=8)

fig.suptitle('Key Interaction Effects and Category Performance',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('figures/stage4_interaction_effects.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: figures/stage4_interaction_effects.png")
print()

# ============================================================================
# Summary
# ============================================================================
print("="*80)
print("VISUALIZATION SUMMARY")
print("="*80)
print()
print("Created 3 publication-quality figures:")
print("  1. figures/stage4_top_features_ranked.png - Top 20 engineered features")
print("  2. figures/stage4_composite_scores.png - Composite score distributions")
print("  3. figures/stage4_interaction_effects.png - Interaction effects and categories")
print()
print("="*80)
print("VISUALIZATIONS COMPLETE")
print("="*80)
