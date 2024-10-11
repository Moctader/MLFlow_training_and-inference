import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.seasonal import seasonal_decompose
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric
import mlflow
from sklearn.preprocessing import StandardScaler


class DataDriftDetector:
    def __init__(self, reference_data, current_data_1, current_data_2):
        self.reference_data = reference_data
        self.current_data_1 = current_data_1
        self.current_data_2 = current_data_2

    def consolidate(self, dataset, target_length):
        total_length = len(dataset)
        assert target_length <= total_length, 'THE TARGET LENGTH CAN ONLY BE SMALLER THAN THE DATASET'
        original_indices = np.linspace(0, total_length - 1, num=total_length)
        target_indices = np.linspace(0, total_length - 1, num=target_length)
        consolidated_data = np.interp(target_indices, original_indices, dataset)
        return consolidated_data

    def preprocess_data(self, data):
        log_data = np.log(data)
        result = seasonal_decompose(log_data, model='additive', period=12).resid
        result = pd.Series(result).dropna()
        # Scaling the data
        scaler = StandardScaler()
        scaled_result = scaler.fit_transform(result.values.reshape(-1, 1))
        return pd.Series(scaled_result.flatten())
    
    # ks(p-value, so sensitive ideal for less than 1000 values), 
    # wasserstein(distance, ideal for more than 1000 values)
    # kl_div("degree of drift," but cannot compare the "drift sizes" between each other )
    # psi(react only to "major changes.")
    # jensenshannon(good measure to detect significant changes)


    def create_drift_report(self, reference_data, current_data, column_name='value', stattest='wasserstein', threshold=0.1):
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


    def run(self, stattest_threshold_1=0.02, stattest_threshold_2=0.8):
        # Consolidate data to the same length
        target_length = min(len(self.reference_data), len(self.current_data_1), len(self.current_data_2))
        self.reference_data = self.consolidate(self.reference_data, target_length)
        self.current_data_1 = self.consolidate(self.current_data_1, target_length)
        self.current_data_2 = self.consolidate(self.current_data_2, target_length)

        # Preprocess data
        self.reference_data = self.preprocess_data(self.reference_data)
        self.current_data_1 = self.preprocess_data(self.current_data_1)
        self.current_data_2 = self.preprocess_data(self.current_data_2)

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

     

# Helper functions to fetch and split data
def fetch_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    return df['Close']

def split_data(data, reference_ratio=0.02, current_ratio=0.01):
    total_length = len(data)
    reference_data = data[:int(total_length * reference_ratio)]
    current_data_1 = data[int(total_length * reference_ratio):int(total_length * (reference_ratio + current_ratio))]
    current_data_2 = data[int(total_length * (reference_ratio + current_ratio)):]
    return reference_data, current_data_1, current_data_2

if __name__ == "__main__":
    data=pd.read_csv('EODHD_EURUSD_HISTORICAL_2019_2024_1min.csv')
    data=data['close']
    reference_data, current_data_1, current_data_2 = split_data(data)
    print(len(reference_data), len(current_data_1), len(current_data_2))
    detector = DataDriftDetector(reference_data, current_data_1, current_data_2)
    detector.run()