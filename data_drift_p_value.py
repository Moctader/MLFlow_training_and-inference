import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from frouros.detectors.data_drift import KSTest
import math

class DataDriftDetector:
    def __init__(self, file_path, window_size=20, num_chunks=5, desired_length=100):
        self.file_path = file_path
        self.window_size = window_size
        self.num_chunks = num_chunks
        self.desired_length = desired_length
        self.data = None
        self.reference_train_data = None
        self.test_data = None
        self.moving_average = None
        self.deviations = None
        self.chunks = None
        self.ks_stats = []
        self.ks_p_values = []

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        self.reference_train_data = self.data['close'].head(15000000)
        self.test_data = self.data['close'].tail(2000)

    def calculate_moving_average(self):
        self.moving_average = self.reference_train_data.rolling(window=self.window_size).mean().dropna()

    def calculate_deviations(self):
        self.deviations = self.reference_train_data[self.window_size-1:] - self.moving_average

    def consolidate(self, dataset, desired_length):
        container = []

        total_length = len(dataset)
        assert desired_length < total_length, 'THE DESIRED LENGTH CAN ONLY BE SMALLER THAN THE DATASET'

        subset_start = 0
        bucket_size = math.ceil(total_length / desired_length)
        
        while subset_start < total_length:
            subset_end = subset_start + bucket_size
            
            if subset_end > total_length:
                subset_end = total_length

            subset = dataset[subset_start:subset_end]
            average = sum(subset) / len(subset)
            container.append(average)
            
            subset_start = subset_end

        return container

    def perform_ks_test(self):
        # Ensure desired_length is smaller than the length of the datasets
        if self.desired_length >= len(self.reference_train_data) or self.desired_length >= len(self.test_data):
            print("Desired length is too large for the dataset. Please choose a smaller desired length.")
            return

        # Consolidate reference and test data
        consolidated_reference = self.consolidate(self.reference_train_data, self.desired_length)
        consolidated_test = self.consolidate(self.test_data, self.desired_length)

        # Convert to numpy arrays for KS test
        consolidated_reference = np.array(consolidated_reference).reshape(-1, 1)
        consolidated_test = np.array(consolidated_test).reshape(-1, 1)

        # Perform KS test
        ks_test = KSTest()
        ks_test.fit(consolidated_reference)
        ks_result, _ = ks_test.compare(consolidated_test)
        ks_stat = ks_result.statistic[0]
        ks_p_value = ks_result.p_value[0]
        self.ks_stats.append(ks_stat)
        self.ks_p_values.append(ks_p_value)
        if ks_p_value < 0.05:
            print(f"Data drift detected between reference and test data (p-value: {ks_p_value:.4f})")

    def plot_results(self):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(range(len(self.ks_p_values)), self.ks_p_values, label='KS p-values', color='blue')
        ax.set_title('KS Test p-values over time')
        ax.set_xlabel('Chunk Index')
        ax.set_ylabel('p-value')
        ax.axhline(y=0.05, color='r', linestyle='--', label='Significance Level (0.05)')
        ax.legend()
        plt.tight_layout()
        plt.show()

    def run(self):
        self.load_data()
        self.calculate_moving_average()
        self.calculate_deviations()
        self.perform_ks_test()
        self.plot_results()

# Usage
detector = DataDriftDetector('EODHD_EURUSD_HISTORICAL_2019_2024_1min.csv')
detector.run()