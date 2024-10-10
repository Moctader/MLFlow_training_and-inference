import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
from pylab import rcParams

# Step 2: Define the stationarity test function
def test_stationarity(timeseries):
    rolmean = timeseries.rolling(12).mean()  # Rolling mean
    rolstd = timeseries.rolling(12).std()    # Rolling std deviation
    
    # Plot rolling statistics
    plt.figure(figsize=(10, 6))
    #plt.plot(timeseries, color='blue', label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean and Standard Deviation')
    plt.show()

    # Perform Dickey-Fuller test
    print("Results of Dickey-Fuller test:")
    adft = adfuller(timeseries, autolag='AIC')
    output = pd.Series(adft[0:4], index=['Test Statistics', 'p-value', 'No. of lags used', 'Number of observations used'])
    print(output)

# Fetch the data from yfinance
ticker = 'AAPL'  # You can change this to any stock ticker you want
df = yf.download(ticker, start='2010-01-01', end='2023-01-01')

# Use the 'Close' price
df_close = df['Close']

# Apply log transformation
df_log = np.log(df_close)

# Calculate rolling mean and standard deviation
moving_avg = df_log.rolling(12).mean()
std_dev = df_log.rolling(12).std()

# Plot the original log-transformed data, rolling mean, and standard deviation
plt.figure(figsize=(10, 6))
plt.plot(df_log, label='Log-transformed')
plt.plot(moving_avg, color="red", label='Rolling Mean')
plt.plot(std_dev, color="black", label='Rolling Std')
plt.legend(loc='best')
plt.title('Log-transformed Data, Rolling Mean, and Standard Deviation')
plt.show()

# Calculate the difference between log-transformed data and rolling mean
df_log_moving_avg_diff = df_log - moving_avg
df_log_moving_avg_diff.dropna(inplace=True)

# Test stationarity on the differenced data
test_stationarity(df_log_moving_avg_diff)

# Calculate exponentially weighted moving average
weighted_average = df_log.ewm(halflife=12, min_periods=0, adjust=True).mean()

# Plot the original log-transformed data and exponentially weighted moving average
plt.figure(figsize=(10, 6))
plt.plot(df_log, label='Log-transformed')
plt.plot(weighted_average, color='red', label='Weighted Average')
plt.xlabel("Date")
plt.ylabel("Consumption")
rcParams['figure.figsize'] = 10, 6
plt.legend()
plt.show(block=False)

# Calculate the difference between log-transformed data and its shifted version
df_log_diff = df_log - df_log.shift()

# Plot the differenced log-transformed data
plt.figure(figsize=(10, 6))
plt.title("Shifted timeseries")
plt.xlabel("Date")
plt.ylabel("Consumption")
plt.plot(df_log_diff)
plt.show()

# Test stationarity on the differenced log-transformed data
df_log_diff.dropna(inplace=True)
test_stationarity(df_log_diff)