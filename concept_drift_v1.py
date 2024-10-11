'''
When Concept Drift Happens: The model needs retraining or updating to capture the new input-output relationships.

When Target Drift Happens: The model may need adjustments to handle the new target distribution, even if the input-output relationship remains constant.




For concept drift, you need both the input features and the target values, and you typically assess drift 
in how well a model (or some predictor) performs over time. We can simulate this by creating a model, 
evaluating its performance on different segments of data, and detecting drift in the performance (i.e., concept drift).


'''



import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import mlflow

class DataProcessor:
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

def split_data(data, reference_ratio=0.02, current_ratio=0.01):
    total_length = len(data)
    reference_length = int(total_length * reference_ratio)
    current_length = int(total_length * current_ratio)
    
    reference_data = data[:reference_length]
    current_data_1 = data[reference_length:reference_length + current_length]
    current_data_2 = data[reference_length + current_length:reference_length + 2 * current_length]
    
    return reference_data, current_data_1, current_data_2

class ConceptDriftDetector:
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
        # Log the results in MLflow
        with mlflow.start_run():
            mlflow.log_metric("accuracy_reference", accuracy_score(self.reference_target, self.model.predict(self.reference_data.values.reshape(-1, 1))))
            mlflow.log_metric("accuracy_current_1", accuracy_1)
            mlflow.log_metric("accuracy_current_2", accuracy_2)
            
            # If accuracy drops significantly between reference and current data, we can infer concept drift
            drift_detected_1 = accuracy_1 < (accuracy_score(self.reference_target, self.model.predict(self.reference_data.values.reshape(-1, 1))) - 0.1)
            drift_detected_2 = accuracy_2 < (accuracy_score(self.reference_target, self.model.predict(self.reference_data.values.reshape(-1, 1))) - 0.1)

            mlflow.log_param("concept_drift_detected_1", drift_detected_1)
            mlflow.log_param("concept_drift_detected_2", drift_detected_2)

if __name__ == "__main__":
    data = pd.read_csv('EODHD_EURUSD_HISTORICAL_2019_2024_1min.csv')
    data['target'] = (data['close'].shift(-1) > data['close']).astype(int)  # Target: if next close > current close
    data = data.dropna()

    reference_data, current_data_1, current_data_2 = split_data(data['close'])
    reference_target, current_target_1, current_target_2 = split_data(data['target'])

    # Preprocess the data
    processor = DataProcessor()
    reference_data = processor.preprocess_data(reference_data)
    current_data_1 = processor.preprocess_data(current_data_1)
    current_data_2 = processor.preprocess_data(current_data_2)

    # Ensure the targets have the same length as the preprocessed data
    reference_target = reference_target[:len(reference_data)]
    current_target_1 = current_target_1[:len(current_data_1)]
    current_target_2 = current_target_2[:len(current_data_2)]

    # Using Logistic Regression for concept drift detection
    model = LogisticRegression()
    concept_drift_detector = ConceptDriftDetector(model, reference_data, reference_target, current_data_1, current_target_1, current_data_2, current_target_2)
    concept_drift_detector.run()