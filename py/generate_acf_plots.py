#!/usr/bin/env python3
"""
Generate ACF plots for returns and squared returns
To illustrate the long memory stylized fact
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf

# Set random seed for reproducibility
np.random.seed(42)

# Set style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (14, 5)
plt.rcParams['font.size'] = 11

print("="*60)
print("Generating ACF plots for Long Memory Stylized Fact")
print("="*60)

# Generate realistic S&P 500-like returns (same as before)
n_obs = 5000

# Parameters for data generation
omega_true = 0.01
alpha_true = 0.05
beta_true = 0.93
gamma_true = 0.03

# Initialize
h = np.zeros(n_obs)
returns = np.zeros(n_obs)
unconditional_var = omega_true / (1 - alpha_true - beta_true - gamma_true/2)
h[0] = unconditional_var

# Generate data with GARCH dynamics
for t in range(1, n_obs):
    eps = np.random.standard_normal()
    returns[t-1] = np.sqrt(h[t-1]) * eps

    # GJR-GARCH data generating process
    h[t] = omega_true + alpha_true * returns[t-1]**2
    if returns[t-1] < 0:
        h[t] += gamma_true * returns[t-1]**2
    h[t] += beta_true * h[t-1]
    h[t] = max(h[t], 0.001)

# Last return
eps = np.random.standard_normal()
returns[n_obs-1] = np.sqrt(h[n_obs-1]) * eps

print(f"Data generated: {n_obs} observations")
print(f"Mean return: {returns.mean():.4f}%")
print(f"Std dev: {returns.std():.4f}%")

# Calculate squared returns
squared_returns = returns**2

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot ACF of returns
lags = 50
plot_acf(returns, lags=lags, ax=ax1, alpha=0.05)
ax1.set_title('ACF of Returns ($r_t$)', fontsize=14, fontweight='bold', pad=10)
ax1.set_xlabel('Lag', fontsize=12, fontweight='bold')
ax1.set_ylabel('Autocorrelation', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_ylim(-0.15, 0.15)

# Add text annotation
textstr1 = 'Observation:\nNo significant\nautocorrelation\n(random walk)'
props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.text(0.98, 0.97, textstr1, transform=ax1.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right', bbox=props, family='monospace')

# Plot ACF of squared returns
plot_acf(squared_returns, lags=lags, ax=ax2, alpha=0.05)
ax2.set_title('ACF of Squared Returns ($r_t^2$)', fontsize=14, fontweight='bold', pad=10)
ax2.set_xlabel('Lag', fontsize=12, fontweight='bold')
ax2.set_ylabel('Autocorrelation', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')

# Add text annotation
textstr2 = 'Long Memory:\nSignificant ACF\nfor many lags\n(hyperbolic decay)'
props2 = dict(boxstyle='round', facecolor='#fffacd', alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.text(0.98, 0.97, textstr2, transform=ax2.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right', bbox=props2, family='monospace')

plt.tight_layout()

# Save the figure
output_file = 'acf_long_memory.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"\nPlot saved as: {output_file}")

# Also save as PDF
output_pdf = 'acf_long_memory.pdf'
plt.savefig(output_pdf, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"Plot also saved as: {output_pdf}")

# Print some statistics
print("\n" + "="*60)
print("Autocorrelation Statistics")
print("="*60)

# Calculate ACF values
acf_returns = acf(returns, nlags=50, fft=False)
acf_squared = acf(squared_returns, nlags=50, fft=False)

print(f"\nReturns ACF at lag 1: {acf_returns[1]:.4f}")
print(f"Returns ACF at lag 10: {acf_returns[10]:.4f}")
print(f"Returns ACF at lag 20: {acf_returns[20]:.4f}")

print(f"\nSquared Returns ACF at lag 1: {acf_squared[1]:.4f}")
print(f"Squared Returns ACF at lag 10: {acf_squared[10]:.4f}")
print(f"Squared Returns ACF at lag 20: {acf_squared[20]:.4f}")

# Count significant lags (beyond 95% confidence)
critical_value = 1.96 / np.sqrt(n_obs)
sig_lags_returns = np.sum(np.abs(acf_returns[1:]) > critical_value)
sig_lags_squared = np.sum(np.abs(acf_squared[1:]) > critical_value)

print(f"\nNumber of significant lags (out of 50):")
print(f"  Returns: {sig_lags_returns}")
print(f"  Squared Returns: {sig_lags_squared}")

print("\nDone!")
