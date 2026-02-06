"""
Generate S&P 500 volatility clustering visualization using real data
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    sp500 = yf.download('^GSPC', start='2005-01-01', end='2024-12-31', progress=False)
    returns = 100 * np.log(sp500['Adj Close'] / sp500['Adj Close'].shift(1))
    returns = returns.dropna()
    print(f"✓ Downloaded {len(returns)} data points from {returns.index[0]} to {returns.index[-1]}")
except Exception as e:
    print(f"  yfinance failed: {e}")

# Try pandas_datareader
if returns is None:
    try:
        import pandas_datareader.data as web
        sp500 = web.DataReader('^GSPC', 'yahoo', start='2005-01-01', end='2024-12-31')
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
        url = "https://query1.finance.yahoo.com/v7/finance/download/%5EGSPC?period1=1104537600&period2=1735689600&interval=1d&events=history"
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
    print("2. Download historical data from 2005-01-01 to present")
    print("3. Save as 'sp500_data.csv' in the same directory as this script")
    print("4. Run this script again")
    import sys
    sys.exit(1)

# Create the plot
fig, ax = plt.subplots(figsize=(10, 4))

# Plot returns
ax.plot(returns.index, returns.values, linewidth=0.5, color='#1f77b4', alpha=0.7)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

# Add shaded regions for high volatility periods
# 2008 Financial Crisis
crisis_start = pd.Timestamp('2008-09-01')
crisis_end = pd.Timestamp('2009-03-31')
if crisis_start in returns.index or any(returns.index > crisis_start):
    ax.axvspan(crisis_start, crisis_end, alpha=0.15, color='red', label='High Volatility Period')

# COVID-19 Crash
covid_start = pd.Timestamp('2020-02-15')
covid_end = pd.Timestamp('2020-04-30')
if covid_start in returns.index or any(returns.index > covid_start):
    ax.axvspan(covid_start, covid_end, alpha=0.15, color='red')

# Labels
ax.set_xlabel('Date', fontsize=11)
ax.set_ylabel('Daily Returns (%)', fontsize=11)
ax.set_title('S&P 500 Daily Returns: Volatility Clustering', fontsize=12, fontweight='bold')

# Grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Legend
ax.legend(loc='upper right', framealpha=0.9, fontsize=9)

# Add text annotation
textstr = 'Large changes followed by large changes;\nsmall changes followed by small changes'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=props)

# Tight layout
plt.tight_layout()

# Save
plt.savefig('assets/other/volatility_clustering.pdf', dpi=300, bbox_inches='tight')
plt.savefig('assets/other/volatility_clustering.png', dpi=300, bbox_inches='tight')

print("\n✓ Volatility clustering plot saved successfully!")
print("  Files: assets/other/volatility_clustering.pdf and .png")

# Print statistics
print(f"\nStatistics:")
print(f"Full sample volatility: {np.std(returns):.3f}%")
if crisis_start in returns.index or any(returns.index > crisis_start):
    crisis_returns = returns[(returns.index >= crisis_start) & (returns.index <= crisis_end)]
    print(f"2008 crisis volatility: {np.std(crisis_returns):.3f}%")
if covid_start in returns.index or any(returns.index > covid_start):
    covid_returns = returns[(returns.index >= covid_start) & (returns.index <= covid_end)]
    print(f"COVID-19 crash volatility: {np.std(covid_returns):.3f}%")
