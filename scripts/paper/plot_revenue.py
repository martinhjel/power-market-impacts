"""
Plot bar chart of expected revenue from nuclear and offshore wind revenue analysis.

This script reads the nuclear_offshore_revenue.csv file and creates visualizations
showing the expected revenue values for different scenarios, technologies, and areas.
"""

import sys
from pathlib import Path

# Add workspace root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configuration
MODEL_FOLDER = "PowerGamaMSc_2025_BM_1H_serial_TrueEXO_load"
OUTPUT_DIR = "visualizations"

# Setup paths
base_path = Path.cwd()
# Handle running from scripts/paper or workspace root
if base_path.name == "paper":
    base_path = base_path.parent.parent
elif base_path.name == "scripts":
    base_path = base_path.parent

input_path = base_path / OUTPUT_DIR / MODEL_FOLDER / "paper"
output_path = input_path
output_path.mkdir(parents=True, exist_ok=True)

# Read the CSV file
csv_file = input_path / "nuclear_offshore_revenue.csv"
df = pd.read_csv(csv_file)

# Filter for SMR and LMR scenarios only
smr_lmr_scenarios = [s for s in df['scenario'].unique() if s.startswith('SMR') or s.startswith('LMR')]
df = df[df['scenario'].isin(smr_lmr_scenarios)].copy()

print(f"Loaded {len(df)} rows from {csv_file}")
print(f"\nSMR/LMR Scenarios: {sorted(df['scenario'].unique())}")
print(f"Technologies: {df['technology'].unique()}")
print(f"Areas: {df['area'].unique()}")

# Set up the plot style
plt.rcParams['font.size'] = 10

# Create single figure with BA and LLPS grouped together
fig, ax = plt.subplots(figsize=(16, 8))

# Extract capacity levels and load types
df['load_type'] = df['scenario'].apply(lambda x: 'BA' if '-BA' in x else 'LLPS')
df['capacity'] = df['scenario'].apply(lambda x: ''.join(filter(str.isdigit, x.split('-')[0])))

# Get unique capacity levels in order
capacity_levels = sorted(df['capacity'].unique(), key=int)

# Calculate average revenue per MWh per scenario
scenario_revenue = df.groupby(['capacity', 'load_type']).agg({
    'revenue_per_mwh': 'mean'
}).reset_index()

# Prepare data for grouped bar chart
x = np.arange(len(capacity_levels))
width = 0.35

ba_values = []
llps_values = []

for cap in capacity_levels:
    ba_val = scenario_revenue[(scenario_revenue['capacity'] == cap) & 
                              (scenario_revenue['load_type'] == 'BA')]['revenue_per_mwh']
    llps_val = scenario_revenue[(scenario_revenue['capacity'] == cap) & 
                                (scenario_revenue['load_type'] == 'LLPS')]['revenue_per_mwh']
    
    ba_values.append(ba_val.values[0] if len(ba_val) > 0 else 0)
    llps_values.append(llps_val.values[0] if len(llps_val) > 0 else 0)

# Plot bars
bars1 = ax.bar(x - width/2, ba_values, width, label='BA Load', 
               color='#FF6B35', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, llps_values, width, label='LLPS Load', 
               color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels on top of bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

# Customize plot
ax.set_xlabel('Nuclear Capacity (MW per area)', fontsize=12, fontweight='bold')
ax.set_ylabel('Revenue (EUR/MWh)', fontsize=12, fontweight='bold')
ax.set_title('SMR/LMR Revenue per MWh: BA vs LLPS Load Scenarios', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(capacity_levels)
ax.legend(fontsize=11, loc='upper left')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(bottom=0)

plt.tight_layout()

# Save figure
output_file = output_path / "smr_lmr_revenue.pdf"
plt.savefig(output_file, format='pdf', bbox_inches='tight')
print(f"\nSaved figure to {output_file}")

# Create a detailed plot by area with separate subplots for SMR and LMR
fig2, (ax_smr, ax_lmr) = plt.subplots(1, 2, figsize=(24, 10))

# Split data into SMR and LMR
smr_data = df[df['scenario'].str.startswith('SMR')].copy()
lmr_data = df[df['scenario'].str.startswith('LMR')].copy()

# Calculate common y-axis scale
max_revenue = df['revenue_per_mwh'].max()
y_max = max_revenue * 1.1  # Add 10% margin

# Plot SMR scenarios
if not smr_data.empty:
    pivot_smr = smr_data.pivot_table(
        index='scenario',
        columns='area',
        values='revenue_per_mwh',
        fill_value=0
    )
    
    # Sort SMR scenarios
    smr_order = sorted(pivot_smr.index, key=lambda x: (
        int(''.join(filter(str.isdigit, x.split('-')[0]))),  # Sort by capacity
        0 if 'BA' in x else 1  # BA before LLPS
    ))
    pivot_smr = pivot_smr.reindex(smr_order)
    
    pivot_smr.plot(kind='bar', ax=ax_smr, width=0.8, edgecolor='black', linewidth=0.8)
    
    ax_smr.set_xlabel('Scenario', fontsize=12, fontweight='bold')
    ax_smr.set_ylabel('Revenue (EUR/MWh)', fontsize=12, fontweight='bold')
    ax_smr.set_title('SMR Revenue per MWh by Scenario and Area', 
                     fontsize=14, fontweight='bold')
    ax_smr.legend(title='Area', loc='upper left', fontsize=10)
    ax_smr.grid(axis='y', alpha=0.3)
    ax_smr.tick_params(axis='x', rotation=45)
    ax_smr.set_ylim(0, y_max)
else:
    ax_smr.text(0.5, 0.5, 'No SMR data available', 
                ha='center', va='center', transform=ax_smr.transAxes, fontsize=14)
    ax_smr.set_title('SMR Scenarios', fontsize=14, fontweight='bold')
    ax_smr.set_ylim(0, y_max)

# Plot LMR scenarios
if not lmr_data.empty:
    pivot_lmr = lmr_data.pivot_table(
        index='scenario',
        columns='area',
        values='revenue_per_mwh',
        fill_value=0
    )
    
    # Sort LMR scenarios
    lmr_order = sorted(pivot_lmr.index, key=lambda x: (
        int(''.join(filter(str.isdigit, x.split('-')[0]))),  # Sort by capacity
        0 if 'BA' in x else 1  # BA before LLPS
    ))
    pivot_lmr = pivot_lmr.reindex(lmr_order)
    
    pivot_lmr.plot(kind='bar', ax=ax_lmr, width=0.8, edgecolor='black', linewidth=0.8)
    
    ax_lmr.set_xlabel('Scenario', fontsize=12, fontweight='bold')
    ax_lmr.set_ylabel('Revenue (EUR/MWh)', fontsize=12, fontweight='bold')
    ax_lmr.set_title('LMR Revenue per MWh by Scenario and Area', 
                     fontsize=14, fontweight='bold')
    ax_lmr.legend(title='Area', loc='upper left', fontsize=10)
    ax_lmr.grid(axis='y', alpha=0.3)
    ax_lmr.tick_params(axis='x', rotation=45)
    ax_lmr.set_ylim(0, y_max)
else:
    ax_lmr.text(0.5, 0.5, 'No LMR data available', 
                ha='center', va='center', transform=ax_lmr.transAxes, fontsize=14)
    ax_lmr.set_title('LMR Scenarios', fontsize=14, fontweight='bold')
    ax_lmr.set_ylim(0, y_max)

plt.tight_layout()

output_file2 = output_path / "smr_lmr_revenue_by_area.pdf"
plt.savefig(output_file2, format='pdf', bbox_inches='tight')
print(f"Saved detailed figure to {output_file2}")

# Print summary statistics
print("\n" + "="*80)
print("SMR/LMR REVENUE PER MWh SUMMARY STATISTICS")
print("="*80)

summary = df.groupby('scenario').agg({
    'revenue_per_mwh': ['mean', 'min', 'max']
}).round(1)

print(summary)

# Compare BA vs LLPS load scenarios
print("\n" + "="*80)
print("BA vs LLPS Load Comparison")
print("="*80)

load_comparison = df.groupby(['capacity', 'load_type']).agg({
    'revenue_per_mwh': 'mean'
}).round(1)

print(load_comparison)

plt.show()

print("\n" + "="*80)
print("Analysis complete!")
print("="*80)
