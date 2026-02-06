#!/usr/bin/env python3
"""
Generate News Impact Curve for GARCH, GJR-GARCH, and EGARCH models
Using S&P 500 data with fallback mechanism:
1. Try WRDS (Wharton Research Data Services)
2. Try yfinance (Yahoo Finance)
3. Fall back to realistic simulated S&P 500-like data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model

# Set random seed for reproducibility
np.random.seed(42)

# Set style for publication-quality plots
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14


def fetch_wrds_data():
    """
    Fetch S&P 500 data from WRDS/Compustat.
    Returns log returns in percentage terms.
    """
    try:
        import wrds
        print("Attempting to fetch data from WRDS...")
        db = wrds.Connection(wrds_username=None)

        query = """
        SELECT datadate as date, prccd as price
        FROM comp.idx_daily
        WHERE gvkeyx = '000003'
          AND datadate >= '2004-01-01'
          AND datadate <= '2024-12-31'
        ORDER BY datadate
        """

        data = db.raw_sql(query)
        db.close()

        if len(data) < 100:
            raise ValueError("Insufficient data from WRDS")

        data['date'] = pd.to_datetime(data['date'])
        data = data.set_index('date')

        # Calculate log returns in percentage terms
        returns = 100 * np.log(data['price'] / data['price'].shift(1))
        returns = returns.dropna().values

        print(f"Successfully fetched {len(returns)} observations from WRDS")
        return returns, "WRDS/Compustat"

    except Exception as e:
        print(f"WRDS fetch failed: {e}")
        return None, None


def fetch_yfinance_data():
    """
    Fetch S&P 500 data from Yahoo Finance using yfinance.
    Returns log returns in percentage terms.
    """
    try:
        import yfinance as yf
        print("Attempting to fetch data from Yahoo Finance...")

        # Download S&P 500 data (^GSPC)
        sp500 = yf.download('^GSPC', start='2004-01-01', end='2024-12-31',
                           progress=False, auto_adjust=True)

        if len(sp500) < 100:
            raise ValueError("Insufficient data from Yahoo Finance")

        # Calculate log returns from adjusted closing prices
        returns = 100 * np.log(sp500['Close'] / sp500['Close'].shift(1))
        returns = returns.dropna().values

        print(f"Successfully fetched {len(returns)} observations from Yahoo Finance")
        return returns, "Yahoo Finance"

    except Exception as e:
        print(f"Yahoo Finance fetch failed: {e}")
        return None, None


def generate_simulated_data():
    """
    Generate realistic S&P 500-like returns using GJR-GARCH process.
    Uses empirically-validated parameters typical for S&P 500:
    - omega = 0.01, alpha = 0.05, beta = 0.93, gamma = 0.03 (leverage effect)
    Returns log returns in percentage terms.
    """
    print("Falling back to simulated S&P 500-like data...")

    n_obs = 5000  # ~20 years of daily data

    # Empirically-validated GARCH parameters typical for S&P 500
    # These must satisfy stationarity: alpha + beta + gamma/2 < 1
    omega_true = 0.01
    alpha_true = 0.05
    beta_true = 0.93
    gamma_true = 0.03  # For asymmetry (leverage effect)

    # Initialize
    h = np.zeros(n_obs)
    returns = np.zeros(n_obs)
    unconditional_var = omega_true / (1 - alpha_true - beta_true - gamma_true/2)
    h[0] = unconditional_var

    print(f"Simulation parameters (typical S&P 500):")
    print(f"  omega={omega_true}, alpha={alpha_true}, beta={beta_true}, gamma={gamma_true}")
    print(f"  Unconditional variance: {unconditional_var:.4f}")
    print(f"  Persistence: {alpha_true + beta_true + gamma_true/2:.4f}")

    # Generate data with leverage effect using GJR-GARCH process
    for t in range(1, n_obs):
        eps = np.random.standard_normal()  # Standard normal innovations
        returns[t-1] = np.sqrt(h[t-1]) * eps

        # GJR-GARCH data generating process
        h[t] = omega_true + alpha_true * returns[t-1]**2
        if returns[t-1] < 0:
            h[t] += gamma_true * returns[t-1]**2  # Leverage effect
        h[t] += beta_true * h[t-1]

        # Safety check
        h[t] = max(h[t], 0.001)

    # Last return
    eps = np.random.standard_normal()
    returns[n_obs-1] = np.sqrt(h[n_obs-1]) * eps

    print(f"Generated {n_obs} simulated observations (~20 years of daily data)")
    return returns, "Simulated (S&P 500-like)"


def get_sp500_returns():
    """
    Get S&P 500 returns with fallback mechanism:
    1. Try WRDS
    2. Try yfinance
    3. Fall back to simulated data

    Returns:
        returns: numpy array of log returns (percentage)
        source: string describing the data source
    """
    # Try WRDS first
    returns, source = fetch_wrds_data()
    if returns is not None:
        return returns, source

    # Try yfinance second
    returns, source = fetch_yfinance_data()
    if returns is not None:
        return returns, source

    # Fall back to simulated data
    returns, source = generate_simulated_data()
    return returns, source


# =============================================================================
# Main execution
# =============================================================================

print("="*60)
print("Fetching S&P 500 returns data")
print("="*60)

# Get returns with fallback mechanism
returns, data_source = get_sp500_returns()

print(f"\nData source: {data_source}")
print(f"Observations: {len(returns)}")
print(f"Mean return: {returns.mean():.4f}%")
print(f"Std dev: {returns.std():.4f}%")
print(f"Skewness: {pd.Series(returns).skew():.4f}")
print(f"Kurtosis: {pd.Series(returns).kurtosis():.4f}")
print(f"Min: {returns.min():.2f}%, Max: {returns.max():.2f}%")

# Estimate GARCH(1,1) model
print("\n" + "="*60)
print("Estimating GARCH(1,1) model...")
print("="*60)
garch_model = arch_model(returns, vol='Garch', p=1, q=1, rescale=False, mean='Zero')
garch_fit = garch_model.fit(disp='off', show_warning=False)
print("GARCH(1,1) estimation completed")

# Estimate GJR-GARCH(1,1) model
print("\n" + "="*60)
print("Estimating GJR-GARCH(1,1) model...")
print("="*60)
gjr_model = arch_model(returns, vol='Garch', p=1, o=1, q=1, rescale=False, mean='Zero')
gjr_fit = gjr_model.fit(disp='off', show_warning=False)
print("GJR-GARCH(1,1) estimation completed")

# Estimate EGARCH(1,1) model
print("\n" + "="*60)
print("Estimating EGARCH(1,1) model...")
print("="*60)
egarch_model = arch_model(returns, vol='EGARCH', p=1, o=1, q=1, rescale=False, mean='Zero')
egarch_fit = egarch_model.fit(disp='off', show_warning=False)
print("EGARCH(1,1) estimation completed")

# Extract parameters
garch_params = garch_fit.params
gjr_params = gjr_fit.params
egarch_params = egarch_fit.params

print("\n" + "="*60)
print("Model Parameters Summary")
print("="*60)
print(f"\nGARCH(1,1):")
print(f"  omega: {garch_params['omega']:.6f}")
print(f"  alpha: {garch_params['alpha[1]']:.6f}")
print(f"  beta:  {garch_params['beta[1]']:.6f}")
persistence_garch = garch_params['alpha[1]'] + garch_params['beta[1]']
print(f"  Persistence (alpha+beta): {persistence_garch:.6f}")

print(f"\nGJR-GARCH(1,1):")
print(f"  omega: {gjr_params['omega']:.6f}")
print(f"  alpha: {gjr_params['alpha[1]']:.6f}")
print(f"  gamma: {gjr_params['gamma[1]']:.6f}")
print(f"  beta:  {gjr_params['beta[1]']:.6f}")

print(f"\nEGARCH(1,1):")
for param in egarch_params.index:
    print(f"  {param}: {egarch_params[param]:.6f}")

# Generate news impact curves
# Create a grid of shocks
shocks = np.linspace(-6, 6, 400)

# Use unconditional variance as the starting point
garch_uncond_var = garch_params['omega'] / (1 - garch_params['alpha[1]'] - garch_params['beta[1]'])
gjr_uncond_var = gjr_params['omega'] / (1 - gjr_params['alpha[1]'] - gjr_params['gamma[1]']/2 - gjr_params['beta[1]'])

print(f"\nUnconditional variances:")
print(f"  GARCH: {garch_uncond_var:.4f}")
print(f"  GJR:   {gjr_uncond_var:.4f}")

# Calculate news impact for GARCH(1,1)
garch_impact = garch_params['omega'] + garch_params['alpha[1]'] * shocks**2 + garch_params['beta[1]'] * garch_uncond_var

# Calculate news impact for GJR-GARCH(1,1)
gjr_impact = gjr_params['omega'] + gjr_params['alpha[1]'] * shocks**2
gjr_impact += gjr_params['gamma[1]'] * shocks**2 * (shocks < 0)
gjr_impact += gjr_params['beta[1]'] * gjr_uncond_var

# For EGARCH - compute news impact properly
log_h_lag = np.log(gjr_uncond_var)  # Use GJR unconditional as reference
egarch_impact = np.zeros_like(shocks)

for i, shock in enumerate(shocks):
    std_shock = shock / np.sqrt(gjr_uncond_var)
    log_h = egarch_params['omega']

    # Add terms that exist in the model
    if 'alpha[1]' in egarch_params.index:
        log_h += egarch_params['alpha[1]'] * (np.abs(std_shock) - np.sqrt(2/np.pi))
    if 'gamma[1]' in egarch_params.index:
        log_h += egarch_params['gamma[1]'] * std_shock
    if 'beta[1]' in egarch_params.index:
        log_h += egarch_params['beta[1]'] * log_h_lag

    egarch_impact[i] = np.exp(log_h)

# Create the plot
fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(shocks, garch_impact, 'b-', linewidth=3.0, label='GARCH(1,1)', alpha=0.9)
ax.plot(shocks, gjr_impact, 'r--', linewidth=3.5, label=f'GJR-GARCH(1,1) ($\\gamma$={gjr_params["gamma[1]"]:.4f})', alpha=0.9)
# Make EGARCH more visible with different color and style
# EGARCH uses gamma[1] for asymmetry if it exists
egarch_label = 'EGARCH(1,1)'
if 'gamma[1]' in egarch_params.index:
    egarch_label = f'EGARCH(1,1) ($\\gamma$={egarch_params["gamma[1]"]:.4f})'
ax.plot(shocks, egarch_impact, color='darkorange', linestyle='-.', linewidth=3.5,
        label=egarch_label, alpha=0.9)

# Add horizontal line at unconditional variance for reference
ax.axhline(y=garch_uncond_var, color='gray', linestyle=':', alpha=0.6, linewidth=1.8, label='Long-run variance', zorder=1)
ax.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=1.3, zorder=1)

ax.set_xlabel('Shock to Returns ($\\varepsilon_{t-1}$)', fontsize=14, fontweight='bold')
ax.set_ylabel('Next Period Conditional Variance ($h_t$)', fontsize=14, fontweight='bold')
ax.set_title(f'News Impact Curves (Data: {data_source})', fontsize=16, fontweight='bold', pad=15)
ax.legend(loc='upper center', frameon=True, shadow=True, fontsize=11, ncol=2, framealpha=0.95, edgecolor='black')
ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.6)
ax.set_xlim(-6, 6)

# Adjust y-axis to prevent overflow
y_max = max(garch_impact.max(), gjr_impact.max(), egarch_impact.max())
ax.set_ylim(0, y_max * 1.1)

plt.tight_layout()

# Save the figure
output_file = 'news_impact_curve.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"\nPlot saved as: {output_file}")

# Also save as PDF for better quality in LaTeX
output_pdf = 'news_impact_curve.pdf'
plt.savefig(output_pdf, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"Plot also saved as: {output_pdf}")

print("\nDone!")
