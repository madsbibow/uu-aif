#!/usr/bin/env python3
"""Extract SP500 RV5 data from oxfordman dataset and save to CSV"""

import zipfile
import pandas as pd
import numpy as np

# Extract and read the CSV from the zip file
zip_path = 'assets/other/oxfordmanrealizedvolatilityindices.zip'
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    with zip_ref.open('oxfordmanrealizedvolatilityindices.csv') as f:
        rv_data = pd.read_csv(f, index_col=0, parse_dates=True)

# Filter for S&P 500 (.SPX) and select RV5
spx_rv = rv_data[rv_data['Symbol'] == '.SPX'][['rv5']].copy()

# Convert to percentage volatility (sqrt of variance * 100)
spx_rv['rv5_vol'] = np.sqrt(spx_rv['rv5']) * 100

# Save to CSV
output_path = 'assets/other/sp500rv5.csv'
spx_rv[['rv5_vol']].to_csv(output_path)

print(f"SP500 RV5 data extracted and saved to {output_path}")
print(f"Total observations: {len(spx_rv)}")
print(f"Date range: {spx_rv.index.min()} to {spx_rv.index.max()}")
print(f"\nFirst few rows:")
print(spx_rv.head())
print(f"\nLast few rows:")
print(spx_rv.tail())
