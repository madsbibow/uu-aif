#!/usr/bin/env python3
"""
Generate Value-at-Risk (VaR) Distribution Chart
Shows a loss distribution with shaded VaR region

Matches the example from the lecture:
- $10M portfolio
- Daily volatility σt = 2%
- 95% VaR = 1.645 × 0.02 × $10M = $329,000

Convention: Losses are positive, Gains are negative
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Apply UU-AIF slide style (dark background)
from uuaif_style import slide_mode, colors
slide_mode()

# Parameters from the slide example
portfolio_value = 10_000_000  # $10M
sigma = 0.02  # 2% daily volatility
mu = 0  # Expected return (often assumed zero for daily)
confidence = 0.95
z_alpha = stats.norm.ppf(confidence)  # 1.645 for 95% VaR (right tail)

# Calculate VaR (as a loss, so positive)
var_pct = z_alpha * sigma  # VaR as percentage (positive number)
var_dollar = var_pct * portfolio_value  # $329,000

print(f"Portfolio: ${portfolio_value:,.0f}")
print(f"Daily volatility: {sigma*100:.1f}%")
print(f"z-quantile (95%): {z_alpha:.3f}")
print(f"VaR (percentage): {var_pct*100:.2f}%")
print(f"VaR (dollars): ${var_dollar:,.0f}")

# Create the distribution (in returns, then flip to losses)
x_returns = np.linspace(-0.08, 0.08, 1000)  # Return range: -8% to +8%
y_returns = stats.norm.pdf(x_returns, mu, sigma)

# Convert to dollar losses (flip sign: losses are positive, gains are negative)
x = -x_returns * portfolio_value  # Flip and convert to dollar amounts
y = y_returns / portfolio_value  # Scale PDF accordingly (due to change of variables)

# Calculate maximum y value for scaling
max_y = max(y)

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the full distribution (yellow - first in slide cycle)
ax.plot(x, y, linewidth=2.5, label='Loss Distribution')

# Shade the VaR region (right tail - worst losses)
var_threshold_dollar = var_dollar
x_var = x[x >= var_threshold_dollar]
y_var = y[x >= var_threshold_dollar]
ax.fill_between(x_var, y_var, alpha=0.5, color=colors.LIGHT_CORAL,
                label='VaR Region (5% probability)')

# Add vertical line at VaR
ax.axvline(x=var_threshold_dollar, color=colors.LIGHT_CORAL, linestyle='--', linewidth=2)

# Add annotation for VaR value
y_at_var = stats.norm.pdf(z_alpha, 0, 1) * sigma / portfolio_value
ax.annotate(f'VaR = ${var_dollar/1000:,.0f}K',
            xy=(var_threshold_dollar, y_at_var),
            xytext=(0.05 * portfolio_value, max_y * 0.7),
            fontsize=12,
            fontweight='bold',
            color=colors.UUYELLOW,
            arrowprops=dict(arrowstyle='->', color=colors.UUYELLOW, lw=1.5))

# Formatting
ax.set_xlabel('Loss ($)')
ax.set_ylabel('Probability Density')
ax.set_title('Value-at-Risk: Loss Distribution', color=colors.UUYELLOW)

# Format x-axis as dollar amounts (negative = gains, positive = losses)
x_tick_positions = np.array([-0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06]) * portfolio_value
x_tick_labels = []
for val in x_tick_positions:
    if val == 0:
        x_tick_labels.append('$0')
    elif val < 0:
        x_tick_labels.append(f'-${abs(val)/1000:,.0f}K')
    else:
        x_tick_labels.append(f'${val/1000:,.0f}K')
ax.set_xticks(x_tick_positions)
ax.set_xticklabels(x_tick_labels)

ax.set_xlim(-0.075 * portfolio_value, 0.075 * portfolio_value)
ax.set_ylim(0, max_y * 1.15)

ax.legend(loc='upper left')

plt.tight_layout()

# Save the figure to assets/figures/
output_file = '../assets/figures/var_distribution.pdf'
plt.savefig(output_file)
print(f"\nPlot saved as: {output_file}")

print("\nDone!")
