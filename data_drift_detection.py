import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.seasonal import seasonal_decompose
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric
import mlflow

class DataDriftDetector:
    def __init__(self, ticker, start_date, end_date, reference_ratio=0.6, current_ratio=0.2):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.reference_ratio = reference_ratio
        self.current_ratio = current_ratio
        self.data = None
        self.reference_data = None
        self.current_data_1 = None
        self.current_data_2 = None

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

    def fetch_data(self):
        df = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        self.data = df['Close']

    def preprocess_data(self):
        log_data = np.log(self.data)
        result = seasonal_decompose(log_data, model='additive', period=12).resid
        result.dropna(inplace=True)
        self.data = result

    def split_data(self):
        total_length = len(self.data)
        self.reference_data = self.data[:int(total_length * self.reference_ratio)]
        self.current_data_1 = self.data[int(total_length * self.reference_ratio):int(total_length * (self.reference_ratio + self.current_ratio))]
        self.current_data_2 = self.data[int(total_length * (self.reference_ratio + self.current_ratio)):]

    def create_drift_report(self, reference_data, current_data, column_name='value', stattest='wasserstein', threshold=0.3):
        reference_df = pd.DataFrame(reference_data, columns=[column_name])
        current_df = pd.DataFrame(current_data, columns=[column_name])
        column_mapping = ColumnMapping()
        drift_report = Report(metrics=[
            ColumnDriftMetric(column_name=column_name, stattest=stattest, stattest_threshold=threshold)
        ])
        drift_report.run(reference_data=reference_df, current_data=current_df, column_mapping=column_mapping)
        return drift_report

    def log_drift_report(self, drift_report, report_name):
        drift_report.save_html(f"{report_name}.html")
        drift_report_dict = drift_report.as_dict()
        drift_score = drift_report_dict['metrics'][0]['result']['drift_score']
        stattest_name = drift_report_dict['metrics'][0]['result']['stattest_name']
        drift_detected = drift_report_dict['metrics'][0]['result']['drift_detected']
        return drift_score, stattest_name, drift_detected

    def run(self, stattest_threshold_1=0.02, stattest_threshold_2=0.3):
        # Fetch and preprocess data
        self.fetch_data()
        self.preprocess_data()

        # Split data
        self.split_data()

        # Consolidate data to the same length
        target_length = min(len(self.reference_data), len(self.current_data_1), len(self.current_data_2))
        self.reference_data = self.consolidate(self.reference_data, target_length)
        self.current_data_1 = self.consolidate(self.current_data_1, target_length)
        self.current_data_2 = self.consolidate(self.current_data_2, target_length)

        # Create drift reports
        drift_report_1 = self.create_drift_report(self.reference_data, self.current_data_1, threshold=stattest_threshold_1)
        drift_report_2 = self.create_drift_report(self.reference_data, self.current_data_2, threshold=stattest_threshold_2)

        # Set the MLflow tracking URI to the local MLflow server
        mlflow.set_tracking_uri("http://127.0.0.1:5001")

        # Log the drift reports and metrics in MLflow
        with mlflow.start_run():
            drift_score_1, stattest_name_1, drift_detected_1 = self.log_drift_report(drift_report_1, "drift_report_1")
            drift_score_2, stattest_name_2, drift_detected_2 = self.log_drift_report(drift_report_2, "drift_report_2")
            
            mlflow.log_artifact("drift_report_1.html")
            mlflow.log_artifact("drift_report_2.html")
            mlflow.log_metric("drift_score_1", drift_score_1)
            mlflow.log_metric("drift_score_2", drift_score_2)
            mlflow.log_param("stattest_name_1", stattest_name_1)
            mlflow.log_param("stattest_name_2", stattest_name_2)
            mlflow.log_param("drift_detected_1", drift_detected_1)
            mlflow.log_param("drift_detected_2", drift_detected_2)

    

if __name__ == "__main__":
    detector = DataDriftDetector(ticker='AAPL', start_date='2010-01-01', end_date='2021-01-01')
    detector.run()