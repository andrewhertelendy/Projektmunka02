import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import time

class ModelMonitor:
    def __init__(self, model_path, sequence_length=10):
      self.model_path = model_path
      self.sequence_length = sequence_length
      self.model = None
      self.feature_columns = None

    def load_model(self):
      """Load the trained model."""
      self.model = tf.keras.models.load_model(self.model_path)
      print(f"Model loaded from {self.model_path}")

    def _prepare_sequence(self, df, feature_columns):
        """Prepare sequence data for real-time predictions."""
        X = []
        for i in range(len(df) - self.sequence_length):
            seq = df[feature_columns].iloc[i:(i + self.sequence_length)].values
            X.append(seq)
        return np.array(X, dtype=np.float32)

    def monitor_data(self, data_path, feature_columns, interval=5):
      """Continuously monitor new data and make predictions."""
      if self.model is None:
        raise ValueError("Load the model first using load_model()")
      
      try:
        while True:
          try:
              new_data = pd.read_csv(data_path)

              if not self.feature_columns:
                self.feature_columns = [col for col in new_data.columns if col not in ['turn_binary', 'turn_label', 'file_id']]

              if len(new_data) < self.sequence_length:
                print("Not enough data yet.")
                time.sleep(interval)
                continue

              X_new = self._prepare_sequence(new_data, self.feature_columns)
              predictions = (self.model.predict(X_new) > 0.5).astype(int)

              y_true = new_data['turn_binary'].iloc[self.sequence_length-1:].values
              
              if len(y_true) != len(predictions):
                print(f"Shape mismatch: y_true {len(y_true)}, predictions: {len(predictions)}. Continuing.")
                time.sleep(interval)
                continue

              print("\nNew Data Predictions:")
              print(classification_report(y_true, predictions))
              
              cm = confusion_matrix(y_true, predictions)
              plt.figure(figsize=(8, 6))
              sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
              plt.title(f'Monitoring Confusion Matrix')
              plt.ylabel('True Label')
              plt.xlabel('Predicted Label')
              plt.show()
              time.sleep(interval)

          except pd.errors.EmptyDataError:
            print("Empty Data File. Sleeping.")
            time.sleep(interval)
          except Exception as e:
            print(f"An error occurred: {e}. Sleeping.")
            time.sleep(interval)
      
      except KeyboardInterrupt:
        print("Monitoring stopped.")


# Main Execution
if __name__ == "__main__":
    # Model Monitoring
    monitor = ModelMonitor('best_model', sequence_length=10)
    monitor.load_model()

    # Example Usage (replace 'new_data.csv' with your monitoring data source)
    try:
      monitor.monitor_data('test_data_engineered.csv', monitor.feature_columns)  # Using test data as an example
    except Exception as e:
        print(f"Error during monitoring: {e}")