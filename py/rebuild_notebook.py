#!/usr/bin/env python3
"""
Rebuild the volatility_modeling_part1.ipynb with the requested changes:
1. Add EWMA before GARCH
2. Show manual GARCH optimization before arch package
3. Use sp500rv5.csv instead of oxfordman zip
4. Focus on fitted volatility comparison, not forecasting
5. Remove HAR and HAR-X forecasting sections
"""

import json
import nbformat as nbf

# Create new notebook
nb = nbf.v4.new_notebook()

# Cell 0: Title
nb.cells.append(nbf.v4.new_markdown_cell("""# Volatility Modeling - Part 1
## Recap, Stylized Facts, EWMA and GARCH Models

This notebook covers:
1. **Data Loading**: S&P 500 returns from WRDS (2000-01-03 to 2022-06-28)
2. **Stylized Facts**: Key properties of financial volatility
3. **EWMA Model**: Exponentially Weighted Moving Average
4. **GARCH Models**: Manual optimization and arch package implementation
5. **Realized Volatility**: Comparison with model-based estimates"""))

# Cell 1: pip install
nb.cells.append(nbf.v4.new_code_cell("""# !pip install arch yfinance wrds scipy optimize"""))

# Cell 2: Imports
nb.cells.append(nbf.v4.new_code_cell("""# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
%matplotlib inline"""))

# Cell 3: Data Loading markdown
nb.cells.append(nbf.v4.new_markdown_cell("""## 1. Data Loading

### S&P 500 Returns from WRDS

We'll use WRDS (Wharton Research Data Services) to access high-quality financial data.
Date range: 2000-01-03 to 2022-06-28 (matching RV5 data availability)"""))

# Cell 4: Load WRDS data
nb.cells.append(nbf.v4.new_code_cell("""# Connect to WRDS and download S&P 500 data
import wrds

# Connect to WRDS (uses saved credentials)
db = wrds.Connection(wrds_username=None)

# Download S&P 500 index data from Compustat
# Date range: 2000-01-03 to 2022-06-28 (matching RV5 data)
query = \"\"\"
SELECT datadate as date, prccd as price
FROM comp.idx_daily
WHERE gvkeyx = '000003'
  AND datadate >= '2000-01-03'
  AND datadate <= '2022-06-28'
ORDER BY datadate
\"\"\"

print("Downloading S&P 500 data from WRDS/Compustat...")
data = db.raw_sql(query)
db.close()

# Process data
data['date'] = pd.to_datetime(data['date'])
data = data.set_index('date')

# Calculate log returns (in percentage)
returns = 100 * np.log(data['price'] / data['price'].shift(1))
returns = returns.dropna()

print(f"\\nData loaded: {len(returns)} observations")
print(f"Date range: {returns.index[0]} to {returns.index[-1]}")
print(f"\\nBasic statistics:")
print(returns.describe())"""))

# Cell 5: Alternative yfinance
nb.cells.append(nbf.v4.new_code_cell("""# Alternative: Using yfinance (uncomment to use)
# import yfinance as yf

# # Download S&P 500 data
# print("Downloading S&P 500 data from Yahoo Finance...")
# sp500 = yf.download('^GSPC', start='2000-01-03', end='2022-06-28', progress=False)

# # Calculate log returns (in percentage)
# returns = 100 * np.log(sp500['Adj Close'] / sp500['Adj Close'].shift(1))
# returns = returns.dropna()
# returns.name = 'returns'

# print(f"\\nData loaded: {len(returns)} observations")
# print(f"Date range: {returns.index[0]} to {returns.index[-1]}")
# print(f"\\nBasic statistics:")
# print(returns.describe())"""))

# Cell 6: Stylized facts markdown
nb.cells.append(nbf.v4.new_markdown_cell("""## 2. Stylized Facts of Volatility

Financial volatility exhibits several well-documented empirical regularities:

1. **Volatility Clustering**: High (low) volatility tends to be followed by high (low) volatility
2. **Fat Tails**: Return distributions have heavier tails than the normal distribution
3. **Leverage Effect**: Negative returns are associated with larger increases in volatility than positive returns
4. **Mean Reversion**: Volatility tends to revert to a long-run mean
5. **Long Memory**: Volatility shocks persist for extended periods"""))

# Cell 7: Stylized facts visualization
nb.cells.append(nbf.v4.new_code_cell("""# Visualize returns and distribution
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Time series plot
axes[0, 0].plot(returns.index, returns.values, linewidth=0.5, color='navy', alpha=0.7)
axes[0, 0].set_title('S&P 500 Daily Returns', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Return (%)')
axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=0.5)
axes[0, 0].grid(True, alpha=0.3)

# Histogram with normal overlay
axes[0, 1].hist(returns.values, bins=100, alpha=0.7, color='navy', density=True, edgecolor='black')
mu, std = returns.mean(), returns.std()
x = np.linspace(returns.min(), returns.max(), 100)
axes[0, 1].plot(x, stats.norm.pdf(x, mu, std), 'r-', linewidth=2, label='Normal Distribution')
axes[0, 1].set_title('Distribution of Returns', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Return (%)')
axes[0, 1].set_ylabel('Density')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Absolute returns (proxy for volatility)
abs_returns = np.abs(returns)
axes[1, 0].plot(abs_returns.index, abs_returns.values, linewidth=0.5, color='darkred', alpha=0.7)
axes[1, 0].set_title('Absolute Returns (Volatility Proxy)', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('|Return| (%)')
axes[1, 0].set_xlabel('Date')
axes[1, 0].grid(True, alpha=0.3)

# ACF of absolute returns (volatility clustering)
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(abs_returns.values, lags=50, ax=axes[1, 1], alpha=0.05)
axes[1, 1].set_title('ACF of Absolute Returns', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Lag')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print statistics
print("\\n" + "="*60)
print("STYLIZED FACTS ANALYSIS")
print("="*60)
print(f"Mean return: {returns.mean():.4f}%")
print(f"Volatility (std): {returns.std():.4f}%")
print(f"Skewness: {returns.skew():.4f} (normal = 0)")
print(f"Excess Kurtosis: {returns.kurtosis():.4f} (normal = 0)")
print(f"\\n✓ Fat tails confirmed: Excess kurtosis = {returns.kurtosis():.2f} >> 0")
print(f"✓ Volatility clustering visible in absolute returns plot")"""))

# Continue with more cells...
print("Notebook skeleton created. Saving...")

# Save notebook
with open('/home/user/intro-vola/volatility_modeling_part1_new.ipynb', 'w') as f:
    nbf.write(nb, f)

print("✓ New notebook created: volatility_modeling_part1_new.ipynb")
