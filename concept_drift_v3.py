import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import mlflow
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric
from evidently.pipeline.column_mapping import ColumnMapping

class DataProcessor:
    def preprocess_data(self, data):
        data = np.log(data / data.shift(1))
        data = data.dropna()
        scaler = StandardScaler()
        scaled_result = scaler.fit_transform(data.values.reshape(-1, 1))
        return pd.Series(scaled_result.flatten())

    def consolidate(self, dataset, target_length):
        total_length = len(dataset)
        assert target_length <= total_length, 'THE TARGET LENGTH CAN ONLY BE SMALLER THAN THE DATASET'
        original_indices = np.linspace(0, total_length - 1, num=total_length)
        target_indices = np.linspace(0, total_length - 1, num=target_length)
        consolidated_data = np.interp(target_indices, original_indices, dataset)
        return consolidated_data

class InputDataDrift:
    def __init__(self, reference_data, current_data_1, current_data_2):
        self.reference_data = reference_data
        self.current_data_1 = current_data_1
        self.current_data_2 = current_data_2

    def run(self):
        # Ensure data is numerical
        reference_df = pd.DataFrame({'input': self.reference_data.astype(float)})
        current_df_1 = pd.DataFrame({'input': self.current_data_1.astype(float)})
        current_df_2 = pd.DataFrame({'input': self.current_data_2.astype(float)})

        # Define column mapping
        column_mapping = ColumnMapping()

        # Create Evidently reports
        report_current_1 = Report(metrics=[ColumnDriftMetric(column_name='input', stattest='wasserstein', stattest_threshold=0.1)])
        report_current_2 = Report(metrics=[ColumnDriftMetric(column_name='input', stattest='wasserstein', stattest_threshold=0.1)])

        # Run reports
        report_current_1.run(reference_data=reference_df, current_data=current_df_1, column_mapping=column_mapping)
        report_current_2.run(reference_data=reference_df, current_data=current_df_2, column_mapping=column_mapping)
        report_current_1.save_html("input_current_1.html")
        report_current_2.save_html("input_current_2.html")

        # Extract drift results
        drift_score_1 = report_current_1.as_dict()['metrics'][0]['result']['drift_score']
        drift_score_2 = report_current_2.as_dict()['metrics'][0]['result']['drift_score']

        return drift_score_1, drift_score_2

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
        # Predict on reference and current data
        prediction_reference = self.model.predict(self.reference_data.reshape(-1, 1))
        prediction_current_1 = self.model.predict(self.current_data_1.reshape(-1, 1))
        prediction_current_2 = self.model.predict(self.current_data_2.reshape(-1, 1))

        # Ensure data is numerical
        reference_df = pd.DataFrame({'target': self.reference_target, 'prediction': prediction_reference.astype(float)})
        current_df_1 = pd.DataFrame({'target': self.current_target_1, 'prediction': prediction_current_1.astype(float)})
        current_df_2 = pd.DataFrame({'target': self.current_target_2, 'prediction': prediction_current_2.astype(float)})

        # Define column mapping
        column_mapping = ColumnMapping()
        column_mapping.target = 'target'
        column_mapping.prediction = 'prediction'

        # Create Evidently reports
        report_current_1 = Report(metrics=[ColumnDriftMetric(column_name='prediction', stattest='wasserstein', stattest_threshold=0.1)])
        report_current_2 = Report(metrics=[ColumnDriftMetric(column_name='prediction', stattest='wasserstein', stattest_threshold=0.1)])

        # Run reports
        report_current_1.run(reference_data=reference_df, current_data=current_df_1, column_mapping=column_mapping)
        report_current_2.run(reference_data=reference_df, current_data=current_df_2, column_mapping=column_mapping)
        report_current_1.save_html("prediction_current_1.html")
        report_current_2.save_html("prediction_current_2.html")

        # Extract drift results
        drift_score_1 = report_current_1.as_dict()['metrics'][0]['result']['drift_score']
        drift_score_2 = report_current_2.as_dict()['metrics'][0]['result']['drift_score']

        return drift_score_1, drift_score_2

class ChangeCorrelationInputVsPrediction:
    def __init__(self, reference_data, current_data_1, current_data_2, model):
        self.reference_data = reference_data
        self.current_data_1 = current_data_1
        self.current_data_2 = current_data_2
        self.model = model

    def run(self):
        # Predict on reference and current data
        prediction_reference = self.model.predict(self.reference_data.reshape(-1, 1))
        prediction_current_1 = self.model.predict(self.current_data_1.reshape(-1, 1))
        prediction_current_2 = self.model.predict(self.current_data_2.reshape(-1, 1))

        # Ensure data is numerical
        reference_df = pd.DataFrame({'input': self.reference_data.astype(float), 'prediction': prediction_reference.astype(float)})
        current_df_1 = pd.DataFrame({'input': self.current_data_1.astype(float), 'prediction': prediction_current_1.astype(float)})
        current_df_2 = pd.DataFrame({'input': self.current_data_2.astype(float), 'prediction': prediction_current_2.astype(float)})

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
        report_ref.save_html("corelation_report_ref.html")
        report_current_1.save_html("corelation_current_1.html")
        report_current_2.save_html("corelation_current_2.html") 

        # Extract correlation results
        correlation_ref = report_ref.as_dict()['metrics'][0]['result']['drift_score']
        correlation_current_1 = report_current_1.as_dict()['metrics'][0]['result']['drift_score']
        correlation_current_2 = report_current_2.as_dict()['metrics'][0]['result']['drift_score']

        return correlation_ref, correlation_current_1, correlation_current_2

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
        accuracy_ref = accuracy_score(self.reference_target, self.model.predict(self.reference_data.reshape(-1, 1)))
        accuracy_current_1 = accuracy_score(self.current_target_1, self.model.predict(self.current_data_1.reshape(-1, 1)))
        accuracy_current_2 = accuracy_score(self.current_target_2, self.model.predict(self.current_data_2.reshape(-1, 1)))

        return accuracy_ref, accuracy_current_1, accuracy_current_2


def split_data(data, reference_ratio=0.02, current_ratio=0.01):
    total_length = len(data)
    reference_length = int(total_length * reference_ratio)
    current_length = int(total_length * current_ratio)
    
    reference_data = data[:reference_length]
    current_data_1 = data[reference_length:reference_length + current_length]
    current_data_2 = data[reference_length + current_length:reference_length + 2 * current_length]
    
    return reference_data, current_data_1, current_data_2


if __name__ == "__main__":
    data = pd.read_csv('EODHD_EURUSD_HISTORICAL_2019_2024_1min.csv')
    data['target'] = (data['close'].shift(-1) > data['close']).astype(int)  
    data = data.dropna()

    processor = DataProcessor()
    
    # Preprocess the data before splitting
    data['close'] = processor.preprocess_data(data['close'])

    reference_data, current_data_1, current_data_2 = split_data(data['close'])
    reference_target, current_target_1, current_target_2 = split_data(data['target'])
    

    # Consolidate the data to a target length for comparison
    target_length = min(len(reference_data), len(reference_target), len(current_data_1), len(current_data_2), len(current_target_1), len(current_target_2))
    reference_data = processor.consolidate(reference_data, target_length)
    current_data_1 = processor.consolidate(current_data_1, target_length)
    current_data_2 = processor.consolidate(current_data_2, target_length)
    
    # Consolidate the target data and round to nearest integer
    reference_target = np.round(processor.consolidate(reference_target, target_length)).astype(int)
    current_target_1 = np.round(processor.consolidate(current_target_1, target_length)).astype(int)
    current_target_2 = np.round(processor.consolidate(current_target_2, target_length)).astype(int)


    model = LogisticRegression()

    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://127.0.0.1:5001")

    # Log everything in a single MLflow run
    with mlflow.start_run():
        # Train the model
        model.fit(reference_data.reshape(-1, 1), reference_target)

        # Input Drift Detection
        input_drift_detector = InputDataDrift(reference_data, current_data_1, current_data_2)
        drift_score_1, drift_score_2 = input_drift_detector.run()
        mlflow.log_metric("input_drift_score_1", drift_score_1)
        mlflow.log_metric("input_drift_score_2", drift_score_2)
        mlflow.log_artifact("input_current_1.html")
        mlflow.log_artifact("input_current_2.html")

        # Prediction Data Drift Detection
        prediction_drift_detector = PredictionDataDrift(model, reference_data, reference_target, current_data_1, current_target_1, current_data_2, current_target_2)
        prediction_drift_score_1, prediction_drift_score_2 = prediction_drift_detector.run()
        mlflow.log_metric("prediction_drift_score_1", prediction_drift_score_1)
        mlflow.log_metric("prediction_drift_score_2", prediction_drift_score_2)
        mlflow.log_artifact("prediction_current_1.html")
        mlflow.log_artifact("prediction_current_2.html")

        # Change Correlation between Input and Prediction
        change_correlation_detector = ChangeCorrelationInputVsPrediction(reference_data, current_data_1, current_data_2, model)
        correlation_ref, correlation_current_1, correlation_current_2 = change_correlation_detector.run()
        mlflow.log_metric("correlation_reference", correlation_ref)
        mlflow.log_metric("correlation_current_1", correlation_current_1)
        mlflow.log_metric("correlation_current_2", correlation_current_2)
        mlflow.log_artifact("corelation_report_ref.html")
        mlflow.log_artifact("corelation_current_1.html")
        mlflow.log_artifact("corelation_current_2.html")

        # Model Quality Metrics
        model_quality = ModelQualityMetrics(reference_data, reference_target, current_data_1, current_target_1, current_data_2, current_target_2, model)
        accuracy_ref, accuracy_current_1, accuracy_current_2 = model_quality.log_metrics()
        
        mlflow.log_metric("accuracy_reference", accuracy_ref)
        mlflow.log_metric("accuracy_current_1", accuracy_current_1)
        mlflow.log_metric("accuracy_current_2", accuracy_current_2)