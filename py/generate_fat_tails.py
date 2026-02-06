"""
Generate comparison of empirical S&P 500 return distribution vs Normal distribution
to illustrate fat tails (excess kurtosis)
Uses actual S&P 500 data
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde

# Set style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

# Download S&P 500 data
print("Downloading S&P 500 data...")
returns = None

# Try yfinance
try:
    import yfinance as yf
    sp500 = yf.download('^GSPC', start='2010-01-01', end='2024-12-31', progress=False)
    returns = 100 * np.log(sp500['Adj Close'] / sp500['Adj Close'].shift(1))
    returns = returns.dropna()
    print(f"✓ Downloaded {len(returns)} data points from {returns.index[0]} to {returns.index[-1]}")
except Exception as e:
    print(f"  yfinance failed: {e}")

# Try pandas_datareader
if returns is None:
    try:
        import pandas_datareader.data as web
        sp500 = web.DataReader('^GSPC', 'yahoo', start='2010-01-01', end='2024-12-31')
        returns = 100 * np.log(sp500['Adj Close'] / sp500['Adj Close'].shift(1))
        returns = returns.dropna()
        print(f"✓ Downloaded {len(returns)} data points from {returns.index[0]} to {returns.index[-1]}")
    except Exception as e:
        print(f"  pandas_datareader failed: {e}")

# Try direct CSV download from Yahoo
if returns is None:
    try:
        import urllib.request
        from io import StringIO
        # Yahoo Finance CSV download URL
        url = "https://query1.finance.yahoo.com/v7/finance/download/%5EGSPC?period1=1262304000&period2=1735689600&interval=1d&events=history"
        with urllib.request.urlopen(url) as response:
            data = response.read().decode('utf-8')
        sp500 = pd.read_csv(StringIO(data), index_col=0, parse_dates=True)
        returns = 100 * np.log(sp500['Adj Close'] / sp500['Adj Close'].shift(1))
        returns = returns.dropna()
        print(f"✓ Downloaded {len(returns)} data points from {returns.index[0]} to {returns.index[-1]}")
    except Exception as e:
        print(f"  Direct download failed: {e}")

# If all downloads failed, exit with instructions
if returns is None:
    print("\n❌ All download methods failed.")
    print("\nPlease download S&P 500 data manually:")
    print("1. Go to https://finance.yahoo.com/quote/%5EGSPC/history")
    print("2. Download historical data from 2010-01-01 to present")
    print("3. Save as 'sp500_data.csv' in the same directory as this script")
    print("4. Run this script again")
    import sys
    sys.exit(1)

# Calculate moments
mu = np.mean(returns)
sigma = np.std(returns)
skewness = stats.skew(returns)
kurtosis = stats.kurtosis(returns)  # Excess kurtosis

print(f"\nSample statistics:")
print(f"Mean: {mu:.4f}%")
print(f"Std Dev: {sigma:.4f}%")
print(f"Skewness: {skewness:.4f}")
print(f"Excess Kurtosis: {kurtosis:.4f}")

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Create empirical distribution using KDE
kde = gaussian_kde(returns, bw_method='scott')
x = np.linspace(returns.min(), returns.max(), 1000)
empirical_pdf = kde(x)

# Plot empirical distribution
ax.plot(x, empirical_pdf, 'b-', linewidth=2.5, label='Empirical Distribution')

# Plot normal distribution with same mean and variance
normal_pdf = stats.norm.pdf(x, mu, sigma)
ax.plot(x, normal_pdf, 'r-', linewidth=2.5, label='Normal Distribution')

# Labels and legend
ax.set_xlabel('Daily Returns (%)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.legend(loc='upper right', framealpha=0.95, fontsize=11)

# Grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Tight layout
plt.tight_layout()

# Save
plt.savefig('assets/other/fat_tails.pdf', dpi=300, bbox_inches='tight')
plt.savefig('assets/other/fat_tails.png', dpi=300, bbox_inches='tight')

print("\n✓ Fat tails plot saved successfully!")
print("  Files: assets/other/fat_tails.pdf and .png")
