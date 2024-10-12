import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose

class DataProcessor:
    def preprocess_data_log_return(self, data):
        data = np.log(data / data.shift(1))
        data = data.dropna()
        scaler = StandardScaler()
        scaled_result = scaler.fit_transform(data.values.reshape(-1, 1))
        return pd.Series(scaled_result.flatten())

    def preprocess_data_differencing(self, data):
        data = data.diff().dropna()
        scaler = StandardScaler()
        scaled_result = scaler.fit_transform(data.values.reshape(-1, 1))
        return pd.Series(scaled_result.flatten())

    def preprocess_data_moving_average_deviation(self, data, window=5):
        moving_avg = data.rolling(window=window).mean()
        deviation = data - moving_avg
        deviation = deviation.dropna()
        scaler = StandardScaler()
        scaled_result = scaler.fit_transform(deviation.values.reshape(-1, 1))
        return pd.Series(scaled_result.flatten())

    def preprocess_data_decompose(self, data):
        data = data[data > 0]
        log_data = np.log(data)
        log_data = log_data.dropna()
        result = seasonal_decompose(log_data, model='additive', period=12).resid
        result = pd.Series(result).dropna()
        scaler = StandardScaler()
        scaled_result = scaler.fit_transform(result.values.reshape(-1, 1))
        return pd.Series(scaled_result.flatten())

def split_data(data, reference_ratio=0.002, current_ratio=0.01):
    total_length = len(data)
    reference_data = data[:int(total_length * reference_ratio)]
    current_data_1 = data[int(total_length * reference_ratio):int(total_length * (reference_ratio + current_ratio))]
    current_data_2 = data[int(total_length * (reference_ratio + current_ratio)):]
    return reference_data, current_data_1, current_data_2

if __name__ == "__main__":
    data = pd.read_csv('EODHD_EURUSD_HISTORICAL_2019_2024_1min.csv')
    data['target'] = (data['close'].shift(-1) > data['close']).astype(int)
    data = data.dropna()

    reference_data, current_data_1, current_data_2 = split_data(data['close'])
    reference_target, current_target_1, current_target_2 = split_data(data['target'])

    # Preprocess the reference data using different methods
    processor = DataProcessor()
    reference_data_log_return = processor.preprocess_data_log_return(reference_data)
    reference_data_differencing = processor.preprocess_data_differencing(reference_data)
    reference_data_moving_avg_dev = processor.preprocess_data_moving_average_deviation(reference_data)
    reference_data_decompose = processor.preprocess_data_decompose(reference_data)

    # Plot the comparison between original reference data after different preprocessing methods
    plt.figure(figsize=(12, 10))

    # Subplot 1: Original Reference Data
    plt.subplot(5, 1, 1)
    plt.plot(reference_data.reset_index(drop=True), label='Original Reference Data', marker='.')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Original Reference Data')
    plt.legend()

    # Subplot 2: After Log Return
    plt.subplot(5, 1, 2)
    plt.plot(reference_data_log_return.reset_index(drop=True), label='Reference Data - Log Return', marker='.')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('After Log Return')
    plt.legend()

    # Subplot 3: After Differencing
    plt.subplot(5, 1, 3)
    plt.plot(reference_data_differencing.reset_index(drop=True), label='Reference Data - Differencing', marker='.')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('After Differencing')
    plt.legend()

    # Subplot 4: After Moving Average Deviation
    plt.subplot(5, 1, 4)
    plt.plot(reference_data_moving_avg_dev.reset_index(drop=True), label='Reference Data - Moving Avg Deviation', marker='.')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('After Moving Average Deviation')
    plt.legend()

    # Subplot 5: After Decomposition
    plt.subplot(5, 1, 5)
    plt.plot(reference_data_decompose.reset_index(drop=True), label='Reference Data - Decomposition', marker='.')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('After Decomposition')
    plt.legend()

    plt.tight_layout()
    plt.show()