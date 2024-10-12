import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose

class DataProcessor:
    def preprocess_data_decompose(self, data):
        data = data[data > 0]
        log_data = np.log(data)
        log_data = log_data.dropna()
        result = seasonal_decompose(log_data, model='additive', period=12).resid
        result = pd.Series(result).dropna()
        scaler = StandardScaler()
        scaled_result = scaler.fit_transform(result.values.reshape(-1, 1))
        return pd.Series(scaled_result.flatten())

    def preprocess_data_log_return(self, data):
        data = np.log(data / data.shift(1))
        data = data.dropna()
        scaler = StandardScaler()
        scaled_result = scaler.fit_transform(data.values.reshape(-1, 1))
        return pd.Series(scaled_result.flatten())

def split_data(data, reference_ratio=0.02, current_ratio=0.01):
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

    # Preprocess the data using both methods
    processor = DataProcessor()
    reference_data_decompose = processor.preprocess_data_decompose(reference_data)
    current_data_1_decompose = processor.preprocess_data_decompose(current_data_1)

    reference_data_log_return = processor.preprocess_data_log_return(reference_data)
    current_data_1_log_return = processor.preprocess_data_log_return(current_data_1)

    # Ensure the targets have the same length as the preprocessed data
    reference_target = reference_target[:len(reference_data_decompose)]
    current_target_1 = current_target_1[:len(current_data_1_decompose)]

    # Plot the decompose and log_return methods separately for reference and current data
    plt.figure(figsize=(12, 12))

    # Subplot 1: Reference Data Decompose Method
    plt.subplot(4, 1, 1)
    plt.plot(reference_data_decompose, label='Decompose Method', marker='o')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Reference Data - Decompose Method')
    plt.legend()

    # Subplot 2: Reference Data Log Return Method
    plt.subplot(4, 1, 2)
    plt.plot(reference_data_log_return, label='Log Return Method', marker='x')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Reference Data - Log Return Method')
    plt.legend()

    # Subplot 3: Current Data 1 Decompose Method
    plt.subplot(4, 1, 3)
    plt.plot(current_data_1_decompose, label='Decompose Method', marker='o')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Current Data 1 - Decompose Method')
    plt.legend()

    # Subplot 4: Current Data 1 Log Return Method
    plt.subplot(4, 1, 4)
    plt.plot(current_data_1_log_return, label='Log Return Method', marker='x')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Current Data 1 - Log Return Method')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Plot the comparison between decompose and log_return methods
    plt.figure(figsize=(12, 12))

    # Subplot 1: Reference Data Comparison
    plt.subplot(2, 1, 1)
    plt.plot(reference_data_decompose, label='Decompose Method', marker='o')
    plt.plot(reference_data_log_return, label='Log Return Method', marker='x')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Reference Data Comparison')
    plt.legend()

    # Subplot 2: Current Data 1 Comparison
    plt.subplot(2, 1, 2)
    plt.plot(current_data_1_decompose, label='Decompose Method', marker='o')
    plt.plot(current_data_1_log_return, label='Log Return Method', marker='x')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Current Data 1 Comparison')
    plt.legend()

    plt.tight_layout()
    plt.show()