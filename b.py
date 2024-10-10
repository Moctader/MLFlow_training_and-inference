import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler
from pylab import rcParams

# Fetch the data from yfinance
ticker = 'AAPL'  
df = yf.download(ticker, start='2010-01-01', end='2021-01-01')

# Use the 'Close' price
df_close = df['Close']

# Apply log transformation
df_log = np.log(df_close)

# Scale the log-transformed data
scaler = StandardScaler()
df_log_scaled = scaler.fit_transform(df_log.values.reshape(-1, 1))
df_log_scaled = pd.Series(df_log_scaled.flatten(), index=df_log.index)

# Perform seasonal decomposition
result = seasonal_decompose(df_log_scaled, model='additive', period=12)
result.plot()
plt.show()

# Define the stationarity test function
def test_stationarity(timeseries):
    # Determining rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()
    
    # Plot rolling statistics
    plt.figure(figsize=(10, 6))
    plt.plot(timeseries, color='blue', label='Original')
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

# Calculate rolling mean and standard deviation
moving_avg = df_log_scaled.rolling(12).mean()
std_dev = df_log_scaled.rolling(12).std()

# Calculate the difference between log-transformed data and rolling mean
df_log_moving_avg_diff = df_log_scaled - moving_avg
df_log_moving_avg_diff.dropna(inplace=True)

# Test stationarity on the differenced data
test_stationarity(df_log_moving_avg_diff)

# Calculate exponentially weighted moving average
weighted_average = df_log_scaled.ewm(halflife=12, min_periods=0, adjust=True).mean()
logScale_weightedMean = df_log_scaled - weighted_average

# Plot and test stationarity of the log-scale weighted mean
rcParams['figure.figsize'] = 10, 6
test_stationarity(logScale_weightedMean)