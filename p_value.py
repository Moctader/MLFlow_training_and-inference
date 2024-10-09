import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import yfinance as yf

# Fetch historical stock data for two different time windows
stock1 = yf.download('AAPL', start='2022-01-01', end='2022-12-31')
stock2 = yf.download('AAPL', start='2021-01-01', end='2021-12-31')

# Extract closing prices
data1 = stock1['Close'].values
data2 = stock2['Close'].values

# Perform KS test
ks_stat, p_value = ks_2samp(data1, data2)

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(20, 16))

# Plot time series for stock closing prices
axs[0, 0].plot(stock1.index, data1, label='AAPL 2022', color='blue')
axs[0, 0].set_title('AAPL Closing Prices 2022', fontsize=16)
axs[0, 0].set_xlabel('Date', fontsize=14)
axs[0, 0].set_ylabel('Closing Price', fontsize=14)
axs[0, 0].legend(fontsize=12)

axs[0, 1].plot(stock2.index, data2, label='AAPL 2021', color='red')
axs[0, 1].set_title('AAPL Closing Prices 2021', fontsize=16)
axs[0, 1].set_xlabel('Date', fontsize=14)
axs[0, 1].set_ylabel('Closing Price', fontsize=14)
axs[0, 1].legend(fontsize=12)

# Plot Empirical Cumulative Distribution Function(ECDF) for stock closing prices
ecdf1 = np.sort(data1)
ecdf2 = np.sort(data2)
y1 = np.arange(1, len(ecdf1) + 1) / len(ecdf1)
y2 = np.arange(1, len(ecdf2) + 1) / len(ecdf2)
D = np.max(np.abs(y1 - y2))

axs[1, 0].step(ecdf1, y1, where='post', label='ECDF AAPL 2022', color='blue')
axs[1, 0].step(ecdf2, y2, where='post', label='ECDF AAPL 2021', color='red')
axs[1, 0].axhline(y=D, color='purple', linestyle='--', label='Max Difference (D)')
axs[1, 0].set_title(f'AAPL Closing Prices\nKS statistic: {ks_stat:.2f}, p-value: {p_value:.2f}', fontsize=16)
axs[1, 0].set_xlabel('Closing Price', fontsize=14)
axs[1, 0].set_ylabel('ECDF', fontsize=14)
axs[1, 0].legend(fontsize=12)

# Generate data for p-value > 0.8 (for comparison)
np.random.seed(0)
data1_nonsignificant = np.random.normal(loc=0, scale=1, size=100)
data2_nonsignificant = np.random.normal(loc=0, scale=1, size=100)  # Identical distribution

# Perform KS test for non-significant difference
ks_stat_nonsignificant, p_value_nonsignificant = ks_2samp(data1_nonsignificant, data2_nonsignificant)

# Plot ECDF for non-significant difference
ecdf1_nonsignificant = np.sort(data1_nonsignificant)
ecdf2_nonsignificant = np.sort(data2_nonsignificant)
y1_nonsignificant = np.arange(1, len(ecdf1_nonsignificant) + 1) / len(ecdf1_nonsignificant)
y2_nonsignificant = np.arange(1, len(ecdf2_nonsignificant) + 1) / len(ecdf2_nonsignificant)
D_nonsignificant = np.max(np.abs(y1_nonsignificant - y2_nonsignificant))

axs[1, 1].step(ecdf1_nonsignificant, y1_nonsignificant, where='post', label='ECDF Data1', color='blue')
axs[1, 1].step(ecdf2_nonsignificant, y2_nonsignificant, where='post', label='ECDF Data2', color='red')
axs[1, 1].axhline(y=D_nonsignificant, color='purple', linestyle='--', label='Max Difference (D)')
axs[1, 1].set_title(f'Non-Significant Difference (p-value > 0.8)\nKS statistic: {ks_stat_nonsignificant:.2f}, p-value: {p_value_nonsignificant:.2f}', fontsize=16)
axs[1, 1].set_xlabel('Value', fontsize=14)
axs[1, 1].set_ylabel('ECDF', fontsize=14)
axs[1, 1].legend(fontsize=12)

plt.tight_layout()
plt.show()