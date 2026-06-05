#!/usr/bin/env python3
"""
Script to read and plot Dutch power prices from nl_price.parquet
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    # Define data path
    data_file = Path(__file__).parent / "data" / "nl_price.parquet"
    
    # Read the parquet file
    print(f"Reading data from {data_file}...")
    df = pd.read_parquet(data_file)
    
    # Display basic information
    print("\nData Info:")
    print(df.info())
    print("\nFirst few rows:")
    print(df.head())
    print("\nBasic statistics:")
    print(df.describe())
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Time series of prices
    ax1 = axes[0]
    if isinstance(df.index, pd.DatetimeIndex):
        df.plot(ax=ax1, linewidth=0.8)
    else:
        # Assume first column is datetime if index is not
        if pd.api.types.is_datetime64_any_dtype(df.iloc[:, 0]):
            df.set_index(df.columns[0]).plot(ax=ax1, linewidth=0.8)
        else:
            df.plot(ax=ax1, linewidth=0.8)
    
    ax1.set_title('Dutch Power Prices - Time Series', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price (€/MWh)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    # Plot 2: Distribution histogram
    ax2 = axes[1]
    df.hist(ax=ax2, bins=50, alpha=0.7, edgecolor='black')
    ax2.set_title('Dutch Power Prices - Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Price (€/MWh)')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path(__file__).parent.parent / "visualizations"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "nl_price_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_file}")
    
    # Show plot
    plt.show()

if __name__ == "__main__":
    main()
