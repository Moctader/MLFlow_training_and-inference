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

    def consolidate(self, dataset, target_length):
        total_length = len(dataset)
        assert target_length <= total_length, 'THE TARGET LENGTH CAN ONLY BE SMALLER THAN THE DATASET'
        original_indices = np.linspace(0, total_length - 1, num=total_length)
        target_indices = np.linspace(0, total_length - 1, num=target_length)
        consolidated_data = np.interp(target_indices, original_indices, dataset)
        return consolidated_data

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

# Consolidate the reference data to two-thirds of its original length for demonstration
target_length = int(len(reference_data) * 2 / 5)
consolidated_reference_data_log_return = processor.consolidate(reference_data_log_return, target_length)
consolidated_reference_data_differencing = processor.consolidate(reference_data_differencing, target_length)
consolidated_reference_data_moving_avg_dev = processor.consolidate(reference_data_moving_avg_dev, target_length)
consolidated_reference_data_decompose = processor.consolidate(reference_data_decompose, target_length)
consolidated_reference_data_original = processor.consolidate(reference_data, target_length)

# Plot the reference data before and after consolidation for each preprocessing method
methods = [
    ('Original', reference_data, consolidated_reference_data_original),
    ('Log Return', reference_data_log_return, consolidated_reference_data_log_return),
    ('Differencing', reference_data_differencing, consolidated_reference_data_differencing),
    ('Moving Average Deviation', reference_data_moving_avg_dev, consolidated_reference_data_moving_avg_dev),
    ('Decomposition', reference_data_decompose, consolidated_reference_data_decompose)
]

plt.figure(figsize=(24, 18))
for i, (method_name, original_data, consolidated_data) in enumerate(methods, start=1):
    plt.subplot(len(methods), 2, 2*i-1)
    plt.plot(original_data.reset_index(drop=True), label=f'Original Reference Data - {method_name}', marker='o')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title(f'Original Data - {method_name}')
    plt.legend()
    
    plt.subplot(len(methods), 2, 2*i)
    plt.plot(np.arange(target_length), consolidated_data, label=f'Consolidated Reference Data - {method_name}', marker='x')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title(f'Consolidated Data - {method_name}')
    plt.legend()
    
plt.tight_layout()
plt.show()