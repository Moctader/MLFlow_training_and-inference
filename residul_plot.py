import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class DataProcessor:
    def consolidate(self, dataset, target_length):
        total_length = len(dataset)
        assert target_length <= total_length, 'THE TARGET LENGTH CAN ONLY BE SMALLER THAN THE DATASET'
        original_indices = np.linspace(0, total_length - 1, num=total_length)
        target_indices = np.linspace(0, total_length - 1, num=target_length)
        consolidated_data = np.interp(target_indices, original_indices, dataset)
        return consolidated_data

def split_data(data, reference_ratio=0.02, current_ratio=0.01):
    total_length = len(data)
    reference_data = data[:int(total_length * reference_ratio)]
    current_data_1 = data[int(total_length * reference_ratio):int(total_length * (reference_ratio + current_ratio))]
    current_data_2 = data[int(total_length * (reference_ratio + current_ratio)):]
    return reference_data, current_data_1, current_data_2

if __name__ == "__main__":
    data = pd.read_csv('EODHD_EURUSD_HISTORICAL_2019_2024_1min.csv')
    data = data['close']
    reference_data, current_data_1, current_data_2 = split_data(data)
    print(len(reference_data), len(current_data_1), len(current_data_2))

    processor = DataProcessor()

    # Consolidate the current_data_1 and current_data_2 to a smaller length for demonstration
    target_length = 100  # Example target length
    consolidated_data_1 = processor.consolidate(current_data_1, target_length)
    consolidated_data_2 = processor.consolidate(current_data_2, target_length)

    # Plot the original and consolidated datasets
    plt.figure(figsize=(12, 12))

    # Subplot 1: Reference vs Current Data 1
    plt.subplot(2, 1, 1)
    plt.plot(reference_data, label='Reference Data', marker='o')
    plt.plot(np.linspace(0, len(reference_data) - 1, num=target_length), consolidated_data_1, label='Consolidated Current Data 1', marker='x')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Reference Data vs Current Data 1')
    plt.legend()

    # Subplot 2: Reference vs Current Data 2
    plt.subplot(2, 1, 2)
    plt.plot(reference_data, label='Reference Data', marker='o')
    plt.plot(np.linspace(0, len(reference_data) - 1, num=target_length), consolidated_data_2, label='Consolidated Current Data 2', marker='x')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Reference Data vs Current Data 2')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Consolidate the reference data to a smaller length for demonstration
    consolidated_reference_data = processor.consolidate(reference_data, target_length)

    # Plot the reference data before and after consolidation
    plt.figure(figsize=(12, 6))
    plt.plot(reference_data, label='Original Reference Data', marker='o')
    plt.plot(np.linspace(0, len(reference_data) - 1, num=target_length), consolidated_reference_data, label='Consolidated Reference Data', marker='x')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Data Before and After Consolidation')
    plt.legend()
    plt.tight_layout()
    plt.show()