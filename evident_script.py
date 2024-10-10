import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler
from pylab import rcParams

from evidently.pipeline.column_mapping import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import ColumnDriftMetric
import mlflow

def consolidate(dataset, target_length):
    total_length = len(dataset)
    assert target_length <= total_length, 'THE TARGET LENGTH CAN ONLY BE SMALLER THAN THE DATASET'

    # Create an array of indices for the original dataset
    original_indices = np.linspace(0, total_length - 1, num=total_length)
    # Create an array of indices for the target length
    target_indices = np.linspace(0, total_length - 1, num=target_length)

    # Interpolate the dataset to the target length
    consolidated_data = np.interp(target_indices, original_indices, dataset)

    return consolidated_data

# Fetch the data from yfinance
ticker = 'AAPL'  
df = yf.download(ticker, start='2010-01-01', end='2021-01-01')

# Use the 'Close' price
df_close = df['Close']

# Apply log transformation
df_log = np.log(df_close)

# Perform seasonal decomposition
result = seasonal_decompose(df_log, model='additive', period=12)
result = result.resid

# Drop NaN values from the result
result.dropna(inplace=True)

# Split the data
total_length = len(result)
reference_data = result[:int(total_length * 0.6)]
current_data_1 = result[int(total_length * 0.6):int(total_length * 0.8)]
current_data_2 = result[int(total_length * 0.8):]

# Consolidate the data to a target length
target_length = min(len(reference_data), len(current_data_1), len(current_data_2))
reference_data = consolidate(reference_data, target_length)
current_data_1 = consolidate(current_data_1, target_length)
current_data_2 = consolidate(current_data_2, target_length)

# Create DataFrames for evidently
reference_df = pd.DataFrame(reference_data, columns=['value'])
current_df_1 = pd.DataFrame(current_data_1, columns=['value'])
current_df_2 = pd.DataFrame(current_data_2, columns=['value'])

# Initialize DriftDetector with ColumnDriftMetric
column_mapping = ColumnMapping()
drift_report_1 = Report(metrics=[
    ColumnDriftMetric(column_name='value', stattest='wasserstein', stattest_threshold=0.02)
])
drift_report_2 = Report(metrics=[
    ColumnDriftMetric(column_name='value', stattest='wasserstein', stattest_threshold=0.3)
])

# Calculate the drift reports
drift_report_1.run(reference_data=reference_df, current_data=current_df_1, column_mapping=column_mapping)
drift_report_2.run(reference_data=reference_df, current_data=current_df_2, column_mapping=column_mapping)

# Save the drift reports as HTML files
drift_report_1.save_html("drift_report_1.html")
drift_report_2.save_html("drift_report_2.html")

# Set the MLflow tracking URI to the local MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5001")

# Extract drift scores, stattest names, and drift detected flags
drift_report_1_dict = drift_report_1.as_dict()
drift_report_2_dict = drift_report_2.as_dict()

drift_score_1 = drift_report_1_dict['metrics'][0]['result']['drift_score']
drift_score_2 = drift_report_2_dict['metrics'][0]['result']['drift_score']

stattest_name_1 = drift_report_1_dict['metrics'][0]['result']['stattest_name']
stattest_name_2 = drift_report_2_dict['metrics'][0]['result']['stattest_name']

drift_detected_1 = drift_report_1_dict['metrics'][0]['result']['drift_detected']
drift_detected_2 = drift_report_2_dict['metrics'][0]['result']['drift_detected']

# Log the drift reports, drift scores, stattest names, and drift detected flags in MLflow
with mlflow.start_run():
    mlflow.log_artifact("drift_report_1.html")
    mlflow.log_artifact("drift_report_2.html")
    mlflow.log_metric("drift_score_1", drift_score_1)
    mlflow.log_metric("drift_score_2", drift_score_2)
    mlflow.log_param("stattest_name_1", stattest_name_1)
    mlflow.log_param("stattest_name_2", stattest_name_2)
    mlflow.log_param("drift_detected_1", drift_detected_1)
    mlflow.log_param("drift_detected_2", drift_detected_2)

# Display the drift reports
print(drift_report_1_dict)
print(drift_report_2_dict)