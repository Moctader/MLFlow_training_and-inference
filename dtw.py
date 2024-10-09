import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import anderson
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape

class DataDriftDetector:
    def __init__(self, file_path, window_size=20, num_chunks=5):
        self.file_path = file_path
        self.window_size = window_size
        self.num_chunks = num_chunks
        self.data = None
        self.last_1000_data = None
        self.moving_average = None
        self.deviations = None
        self.chunks = None
        self.ad_stats = []
        self.ad_p_values = []
        self.wd_distances = []
        self.dtw_distances = []
        self.darts_mape = []

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        self.last_1000_data = self.data['close'].tail(1000)

    def calculate_moving_average(self):
        self.moving_average = self.last_1000_data.rolling(window=self.window_size).mean().dropna()

    def calculate_deviations(self):
        self.deviations = self.last_1000_data[self.window_size-1:] - self.moving_average

    def split_chunks(self):
        self.chunks = [chunk.dropna() for chunk in np.array_split(self.deviations, self.num_chunks) if not chunk.empty]

    def perform_darts_test(self):
        for i in range(len(self.chunks) - 1):
            if self.chunks[i].empty or self.chunks[i + 1].empty:
                print(f"Chunk {i+1} or Chunk {i+2} is empty, skipping MAPE calculation.")
                self.darts_mape.append(np.nan)
                continue

            series1 = TimeSeries.from_series(self.chunks[i])
            series2 = TimeSeries.from_series(self.chunks[i + 1])

            print(f"Chunk {i+1} data:\n{self.chunks[i]}")
            print(f"Chunk {i+2} data:\n{self.chunks[i + 1]}")

            if pd.isna(series1.values()).sum() > 0 or pd.isna(series2.values()).sum() > 0:
                print(f"Chunk {i+1} or Chunk {i+2} contains NaN values, skipping MAPE calculation.")
                self.darts_mape.append(np.nan)
                continue

            scaler = Scaler()
            series1 = scaler.fit_transform(series1)
            series2 = scaler.transform(series2)

            print(f"Scaled Chunk {i+1} data:\n{series1.values()}")
            print(f"Scaled Chunk {i+2} data:\n{series2.values()}")

            # Check for NaN values in the scaled data
            if np.isnan(series1.values()).any() or np.isnan(series2.values()).any():
                print(f"Scaled Chunk {i+1} or Chunk {i+2} contains NaN values, skipping MAPE calculation.")
                self.darts_mape.append(np.nan)
                continue

            # Check for empty slices in the scaled data
            if series1.values().size == 0 or series2.values().size == 0:
                print(f"Scaled Chunk {i+1} or Chunk {i+2} is empty, skipping MAPE calculation.")
                self.darts_mape.append(np.nan)
                continue

            try:
                mape_value = mape(series1, series2)
                self.darts_mape.append(mape_value)
                print(f"MAPE between Chunk {i+1} and Chunk {i+2}: {mape_value:.4f}")
            except Exception as e:
                print(f"Error calculating MAPE: {e}")
                self.darts_mape.append(np.nan)

    def plot_results(self):
        fig, axes = plt.subplots(len(self.chunks) - 1, 1, figsize=(12, 6 * (len(self.chunks) - 1)))
        for i in range(len(self.chunks) - 1):
            label1 = f'Chunk {i+1}, AD stat: {self.ad_stats[i]:.4f}' if i < len(self.ad_stats) else f'Chunk {i+1}'
            label2 = f'Chunk {i+2}, WD: {self.wd_distances[i]:.4f}, DTW: {self.dtw_distances[i]:.4f}, MAPE: {self.darts_mape[i]:.4f}' if i < len(self.wd_distances) and i < len(self.dtw_distances) and i < len(self.darts_mape) else f'Chunk {i+2}'
            
            axes[i].plot(self.chunks[i].index, self.chunks[i], label=label1, color='blue')
            axes[i].plot(self.chunks[i + 1].index, self.chunks[i + 1], label=label2, color='orange')
            axes[i].set_title(f'Comparison between Chunk {i+1} and Chunk {i+2}')
            axes[i].set_xlabel('Index')
            axes[i].set_ylabel('Deviation')
            axes[i].legend()
        plt.tight_layout()
        plt.show()

    def run(self):
        self.load_data()
        self.calculate_moving_average()
        self.calculate_deviations()
        self.split_chunks()
        self.perform_darts_test()
        self.plot_results()

# Usage
detector = DataDriftDetector('EODHD_EURUSD_HISTORICAL_2019_2024_1min.csv')
detector.run()