import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.seasonal import seasonal_decompose
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric
import mlflow
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# DataProcessor for preprocessing and consolidation
class DataProcessor:
    def consolidate(self, dataset, target_length):
        total_length = len(dataset)
        assert target_length <= total_length, 'THE TARGET LENGTH CAN ONLY BE SMALLER THAN THE DATASET'
        original_indices = np.linspace(0, total_length - 1, num=total_length)
        target_indices = np.linspace(0, total_length - 1, num=target_length)
        consolidated_data = np.interp(target_indices, original_indices, dataset)
        return consolidated_data

    def preprocess_data(self, data):
        data = data[data > 0]
        log_data = np.log(data)
        result = seasonal_decompose(log_data, model='additive', period=12).resid
        result = pd.Series(result).dropna()

        # Scaling the data
        scaler = StandardScaler()
        scaled_result = scaler.fit_transform(result.values.reshape(-1, 1))
        return pd.Series(scaled_result.flatten())

# Splitting the data into reference and current datasets
def split_data(data, reference_ratio=0.02, current_ratio=0.01):
    total_length = len(data)
    reference_length = int(total_length * reference_ratio)
    current_length = int(total_length * current_ratio)

    reference_data = data[:reference_length]
    current_data_1 = data[reference_length:reference_length + current_length]
    current_data_2 = data[reference_length + current_length:reference_length + 2 * current_length]

    return reference_data, current_data_1, current_data_2

# Base class for input drift detection
class InputDataDrift:
    def __init__(self, reference_data, current_data_1, current_data_2):
        self.reference_data = reference_data
        self.current_data_1 = current_data_1
        self.current_data_2 = current_data_2
        self.processor = DataProcessor()

    def consolidate_data(self):
        # Consolidate data to the same length
        target_length = min(len(self.current_data_1), len(self.current_data_2))
        self.reference_data = self.processor.consolidate(self.reference_data, target_length)
        self.current_data_1 = self.processor.consolidate(self.current_data_1, target_length)
        self.current_data_2 = self.processor.consolidate(self.current_data_2, target_length)

    def preprocess_data(self):
        # Preprocess data
        self.reference_data = self.processor.preprocess_data(self.reference_data)
        self.current_data_1 = self.processor.preprocess_data(self.current_data_1)
        self.current_data_2 = self.processor.preprocess_data(self.current_data_2)

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

    def run(self, stattest_threshold=0.1):
        self.consolidate_data()
        self.preprocess_data()

        # Create drift reports
        drift_report_1 = self.create_drift_report(self.reference_data, self.current_data_1, threshold=stattest_threshold)
        drift_report_2 = self.create_drift_report(self.reference_data, self.current_data_2, threshold=stattest_threshold)

        mlflow.set_tracking_uri("http://127.0.0.1:5001")

        with mlflow.start_run():
            drift_score_1, stattest_name_1, drift_detected_1 = self.log_drift_report(drift_report_1, "input_drift_report_1")
            drift_score_2, stattest_name_2, drift_detected_2 = self.log_drift_report(drift_report_2, "input_drift_report_2")

            mlflow.log_artifact("input_drift_report_1.html")
            mlflow.log_artifact("input_drift_report_2.html")
            mlflow.log_metric("drift_score_1", drift_score_1)
            mlflow.log_metric("drift_score_2", drift_score_2)
            mlflow.log_param("stattest_name_1", stattest_name_1)
            mlflow.log_param("stattest_name_2", stattest_name_2)
            mlflow.log_param("input_drift_detected_1", drift_detected_1)
            mlflow.log_param("input_drift_detected_2", drift_detected_2)

# Prediction drift detection
class PredictionDataDrift:
    def __init__(self, model, reference_data, reference_target, current_data_1, current_target_1, current_data_2, current_target_2):
        self.model = model
        self.reference_data = reference_data
        self.reference_target = reference_target
        self.current_data_1 = current_data_1
        self.current_target_1 = current_target_1
        self.current_data_2 = current_data_2
        self.current_target_2 = current_target_2

    def run(self):
        # Train the model on reference data
        self.model.fit(self.reference_data.values.reshape(-1, 1), self.reference_target)

        # Predict on current data
        predictions_1 = self.model.predict(self.current_data_1.values.reshape(-1, 1))
        predictions_2 = self.model.predict(self.current_data_2.values.reshape(-1, 1))

        # Calculate accuracy
        accuracy_1 = accuracy_score(self.current_target_1, predictions_1)
        accuracy_2 = accuracy_score(self.current_target_2, predictions_2)

        mlflow.set_tracking_uri("http://127.0.0.1:5001")
        with mlflow.start_run():
            mlflow.log_metric("accuracy_reference", accuracy_score(self.reference_target, self.model.predict(self.reference_data.values.reshape(-1, 1))))
            mlflow.log_metric("accuracy_current_1", accuracy_1)
            mlflow.log_metric("accuracy_current_2", accuracy_2)

            drift_detected_1 = accuracy_1 < (accuracy_score(self.reference_target, self.model.predict(self.reference_data.values.reshape(-1, 1))) - 0.1)
            drift_detected_2 = accuracy_2 < (accuracy_score(self.reference_target, self.model.predict(self.reference_data.values.reshape(-1, 1))) - 0.1)

            mlflow.log_param("prediction_drift_detected_1", drift_detected_1)
            mlflow.log_param("prediction_drift_detected_2", drift_detected_2)

from evidently.report import Report
from evidently.metrics import ColumnDriftMetric
from evidently.pipeline.column_mapping import ColumnMapping
import mlflow
import numpy as np
import pandas as pd

class ChangeCorrelationInputVsPrediction:
    def __init__(self, reference_data, current_data_1, current_data_2, model):
        self.reference_data = reference_data
        self.current_data_1 = current_data_1
        self.current_data_2 = current_data_2
        self.model = model

    def run(self):
        # Predict on reference and current data
        prediction_reference = self.model.predict(self.reference_data.values.reshape(-1, 1))
        prediction_current_1 = self.model.predict(self.current_data_1.values.reshape(-1, 1))
        prediction_current_2 = self.model.predict(self.current_data_2.values.reshape(-1, 1))

        # Create DataFrames for Evidently
        reference_df = pd.DataFrame({'input': self.reference_data, 'prediction': prediction_reference})
        current_df_1 = pd.DataFrame({'input': self.current_data_1, 'prediction': prediction_current_1})
        current_df_2 = pd.DataFrame({'input': self.current_data_2, 'prediction': prediction_current_2})

        # Define column mapping
        column_mapping = ColumnMapping()
        column_mapping.prediction = 'prediction'

        # Create Evidently reports
        report_ref = Report(metrics=[ColumnDriftMetric(column_name='input', stattest='wasserstein', stattest_threshold=0.1)])
        report_current_1 = Report(metrics=[ColumnDriftMetric(column_name='input', stattest='wasserstein', stattest_threshold=0.1)])
        report_current_2 = Report(metrics=[ColumnDriftMetric(column_name='input', stattest='wasserstein', stattest_threshold=0.1)])

        # Run reports
        report_ref.run(reference_data=reference_df, current_data=reference_df, column_mapping=column_mapping)
        report_current_1.run(reference_data=reference_df, current_data=current_df_1, column_mapping=column_mapping)
        report_current_2.run(reference_data=reference_df, current_data=current_df_2, column_mapping=column_mapping)
        report_ref.save_html("report_ref.html")
        report_current_1.save_html("report_current_1.html")
        report_current_2.save_html("report_current_2.html") 
       

        # Extract correlation results
        correlation_ref = report_ref.as_dict()['metrics'][0]['result']['drift_score']
        correlation_current_1 = report_current_1.as_dict()['metrics'][0]['result']['drift_score']
        correlation_current_2 = report_current_2.as_dict()['metrics'][0]['result']['drift_score']


        # Log correlations to MLflow
        mlflow.set_tracking_uri("http://127.0.0.1:5001")
        with mlflow.start_run():
            mlflow.log_metric("correlation_reference", correlation_ref)
            mlflow.log_metric("correlation_current_1", correlation_current_1)
            mlflow.log_metric("correlation_current_2", correlation_current_2)
            mlflow.log_artifact("report_ref.html")
            mlflow.log_artifact("report_current_1.html")
            mlflow.log_artifact("report_current_2.html")


# Class for logging model quality metrics
class ModelQualityMetrics:
    def __init__(self, reference_data, reference_target, current_data_1, current_target_1, current_data_2, current_target_2, model):
        self.reference_data = reference_data
        self.reference_target = reference_target
        self.current_data_1 = current_data_1
        self.current_target_1 = current_target_1
        self.current_data_2 = current_data_2
        self.current_target_2 = current_target_2
        self.model = model

    def log_metrics(self):
        mlflow.set_tracking_uri("http://127.0.0.1:5001")
        with mlflow.start_run():
            mlflow.log_metric("accuracy_reference", accuracy_score(self.reference_target, self.model.predict(self.reference_data.values.reshape(-1, 1))))
            mlflow.log_metric("accuracy_current_1", accuracy_score(self.current_target_1, self.model.predict(self.current_data_1.values.reshape(-1, 1))))
            mlflow.log_metric("accuracy_current_2", accuracy_score(self.current_target_2, self.model.predict(self.current_data_2.values.reshape(-1, 1))))

if __name__ == "__main__":
    data = pd.read_csv('EODHD_EURUSD_HISTORICAL_2019_2024_1min.csv')
    data['target'] = (data['close'].shift(-1) > data['close']).astype(int)  # Target: if next close > current close
    data = data.dropna()

    reference_data, current_data_1, current_data_2 = split_data(data['close'])
    reference_target, current_target_1, current_target_2 = split_data(data['target'])


    print(len(reference_data), len(reference_target), len(current_data_1), len(current_target_1), len(current_data_2), len(current_target_2))

    # Preprocess the data
    processor = DataProcessor()
    reference_data = processor.preprocess_data(reference_data)
    current_data_1 = processor.preprocess_data(current_data_1)
    current_data_2 = processor.preprocess_data(current_data_2)

    # Ensure the targets have the same length as the preprocessed data
    reference_target = reference_target[:len(reference_data)]
    current_target_1 = current_target_1[:len(current_data_1)]
    current_target_2 = current_target_2[:len(current_data_2)]

    model = LogisticRegression()

    # Input Drift Detection
    input_drift_detector = InputDataDrift(reference_data, current_data_1, current_data_2)
    input_drift_detector.run()

    # Prediction Drift Detection
    prediction_drift_detector = PredictionDataDrift(model, reference_data, reference_target, current_data_1, current_target_1, current_data_2, current_target_2)
    print(len(reference_data), len(reference_target), len(current_data_1), len(current_target_1), len(current_data_2), len(current_target_2))
    prediction_drift_detector.run()

    # Change Correlation between Input and Prediction
    change_correlation_detector = ChangeCorrelationInputVsPrediction(reference_data, current_data_1, current_data_2, model)
    change_correlation_detector.run()

    # # Model Quality Metrics
    model_quality = ModelQualityMetrics(reference_data, reference_target, current_data_1, current_target_1, current_data_2, current_target_2, model)
    model_quality.log_metrics()
