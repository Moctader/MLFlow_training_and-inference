import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from frouros.detectors.data_drift import KSTest
import math

class DataDriftDetector:
    def __init__(self, file_path, window_size=20):
        self.file_path = file_path
        self.window_size = window_size
        self.data = None
        self.reference_train_data = None
        self.test_data = None
        self.moving_average_train = None
        self.moving_average_test = None
        self.deviations_train = None
        self.deviations_test = None
        self.ks_stats = []
        self.ks_p_values = []

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        split_index = int(len(self.data) * 0.7)
        self.reference_train_data = self.data['close'].iloc[:split_index]
        self.test_data = self.data['close'].iloc[split_index:]
        print("Data is loaded")

    def calculate_moving_average(self):
        self.moving_average_train = self.reference_train_data.rolling(window=self.window_size).mean().dropna()
        self.moving_average_test = self.test_data.rolling(window=self.window_size).mean().dropna()
        print("Moving averages calculated")

    def calculate_deviations(self):
        self.deviations_train = self.reference_train_data[self.window_size-1:] - self.moving_average_train
        self.deviations_test = self.test_data[self.window_size-1:] - self.moving_average_test
        print("Deviations calculated")

    def consolidate(self, dataset, target_length):
        total_length = len(dataset)
        assert target_length <= total_length, 'THE TARGET LENGTH CAN ONLY BE SMALLER THAN THE DATASET'

        # Create an array of indices for the original dataset
        original_indices = np.linspace(0, total_length - 1, num=total_length)
        # Create an array of indices for the target length
        target_indices = np.linspace(0, total_length - 1, num=target_length)

        # Interpolate the dataset to the target length
        consolidated_data = np.interp(target_indices, original_indices, dataset)

        return consolidated_data

    def normalize(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def perform_ks_test(self):
        # Ensure the test data length is smaller than the length of the deviations_train
        if len(self.deviations_test) >= len(self.deviations_train):
            print("Test data length is too large for the reference data. Please choose a smaller test data length.")
            return

        self.consolidated_deviations_train = self.consolidate(self.deviations_train, len(self.deviations_test))
        self.consolidated_deviations_test = self.consolidate(self.deviations_test, len(self.deviations_test))

        print(len(self.consolidated_deviations_train), len(self.consolidated_deviations_test))

        # Convert to numpy arrays for KS test
        consolidated_deviations_train = np.array(self.consolidated_deviations_train).reshape(-1, 1)
        consolidated_deviations_test = np.array(self.consolidated_deviations_test).reshape(-1, 1)

        # Normalize deviations
        consolidated_deviations_train = self.normalize(consolidated_deviations_train)
        consolidated_deviations_test = self.normalize(consolidated_deviations_test)

        # Perform KS test
        ks_test = KSTest()
        ks_test.fit(consolidated_deviations_train)
        ks_result, _ = ks_test.compare(consolidated_deviations_test)
        ks_stat = ks_result.statistic[0]
        ks_p_value = ks_result.p_value[0]
        self.ks_stats.append(ks_stat)
        self.ks_p_values.append(ks_p_value)
        print(f"KS Statistic: {ks_stat}, p-value: {ks_p_value}")

    def plot_consolidated_data(self):
        # Normalize deviations
        normalized_deviations_train = self.normalize(self.consolidated_deviations_train)
        normalized_deviations_test = self.normalize(self.consolidated_deviations_test)

        # Create a combined index for plotting
        total_length = len(normalized_deviations_train) + len(normalized_deviations_test)
        train_index = range(len(normalized_deviations_train))
        test_index = range(len(normalized_deviations_train), total_length)

        # Plot the consolidated deviations
        plt.figure(figsize=(12, 6))
        plt.plot(train_index, normalized_deviations_train, label='Normalized Deviations Train Data')
        plt.plot(test_index, normalized_deviations_test, label='Normalized Deviations Test Data')
        plt.axvline(x=len(normalized_deviations_train), color='r', linestyle='--', label='Train/Test')
        plt.title('Normalized Deviations from Moving Average')
        plt.xlabel('Index')
        plt.ylabel('Normalized Deviation')
        plt.legend()
        plt.show()

    def run(self):
        self.load_data()
        self.calculate_moving_average()
        self.calculate_deviations()
        self.perform_ks_test()
        self.plot_consolidated_data()



detector = DataDriftDetector('EODHD_EURUSD_HISTORICAL_2019_2024_1min.csv')
detector.run()

