import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
from lightgbm import LGBMClassifier
import gc

# Model Definition and Training
class TrajectoryClassifier:
    def __init__(self, sequence_length=10, batch_size=1024, learning_rate=0.001, epochs=20):
       self.sequence_length = sequence_length
       self.batch_size = batch_size
       self.learning_rate = learning_rate
       self.epochs = epochs
       self.models = {}
       self.results = {}

    def create_rnn_model(self, input_shape):
      """Create the GRU model."""
      model = Sequential([
        GRU(64, input_shape=input_shape, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),

        GRU(32, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),

        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
      ])
      
      optimizer = Adam(learning_rate=self.learning_rate)
      model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
      return model
    
    def prepare_rnn_data(self, df, feature_columns):
        """Prepare sequence data for RNN."""
        X = []
        y = []
        for i in range(len(df) - self.sequence_length):
          seq = df[feature_columns].iloc[i:(i + self.sequence_length)].values
          label = df['turn_binary'].iloc[i + self.sequence_length - 1]
          X.append(seq)
          y.append(label)
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int8)
    
    def train_rnn_model(self, train_data, feature_columns, class_weight=None):
      """Train the RNN model."""
      print("\nTraining GRU model...")
      X_train, y_train = self.prepare_rnn_data(train_data, feature_columns)
      input_shape = (self.sequence_length, X_train.shape[-1])

      model = self.create_rnn_model(input_shape)
      
      callbacks = [
          EarlyStopping(patience=5, restore_best_weights=True),
          ReduceLROnPlateau(factor=0.2, patience=3, min_lr=0.00001)
      ]
      
      model.fit(
        X_train, y_train,
        batch_size=self.batch_size,
        epochs=self.epochs,
        validation_split=0.2, # Using a simple validation split for RNN
        callbacks=callbacks,
        class_weight=class_weight,
        verbose = 1
      )
      self.models['gru'] = model

    def train_tree_based_models(self, train_data, feature_columns, class_weight=None):
      """Train tree-based models (XGBoost and LightGBM)."""
      print("\nTraining XGBoost model...")
      xgb_model = xgb.XGBClassifier(
            scale_pos_weight=class_weight[1],
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            use_label_encoder=False,
            eval_metric='logloss'
        )
      X_train = train_data[feature_columns]
      y_train = train_data['turn_binary']
      xgb_model.fit(X_train, y_train)
      self.models['xgb'] = xgb_model
      
      print("\nTraining LightGBM model...")
      lgb_model = LGBMClassifier(
            scale_pos_weight=class_weight[1],
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6
        )
      lgb_model.fit(X_train, y_train)
      self.models['lgb'] = lgb_model

    def train(self, train_data, feature_columns):
      """Train all models."""
      class_weights = compute_class_weight('balanced', classes=np.unique(train_data['turn_binary']), y=train_data['turn_binary'])
      class_weight = {0: class_weights[0], 1: class_weights[1]}
      
      self.train_rnn_model(train_data, feature_columns, class_weight)
      self.train_tree_based_models(train_data, feature_columns, class_weight)
    
    def evaluate(self, test_data, feature_columns):
      """Evaluate trained models."""
      results = {model_name: {'y_true': [], 'y_pred': []} for model_name in self.models}
      
      for model_name, model in self.models.items():
        if model_name == 'gru':
          X_test, y_test = self.prepare_rnn_data(test_data, feature_columns)
          y_pred_probs = model.predict(X_test)
          y_pred = (y_pred_probs > 0.5).astype(int)
        else:
          X_test = test_data[feature_columns]
          y_test = test_data['turn_binary']
          y_pred = model.predict(X_test)
      
        results[model_name]['y_true'].extend(y_test)
        results[model_name]['y_pred'].extend(y_pred)
      
      for model_name in self.models:
        y_true = np.array(results[model_name]['y_true'])
        y_pred = np.array(results[model_name]['y_pred'])

        print(f"\n{model_name} Results:")
        print(classification_report(y_true, y_pred))

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{model_name} Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        if model_name == 'gru':
            y_pred_probs = self.models[model_name].predict(X_test)
            fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
            roc_auc = roc_auc_score(y_true, y_pred_probs)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{model_name} ROC Curve')
            plt.legend()
            plt.show()
        
# Main Execution
if __name__ == "__main__":
    # Load engineered data
    train_data = pd.read_csv('train_data_engineered.csv')
    test_data = pd.read_csv('test_data_engineered.csv')
    feature_columns = [col for col in train_data.columns if col not in ['turn_binary', 'turn_label', 'file_id']]
    
    # Model Training and Evaluation
    classifier = TrajectoryClassifier(sequence_length=10, batch_size=1024, learning_rate=0.001, epochs=20)
    classifier.train(train_data, feature_columns)
    classifier.evaluate(test_data, feature_columns)

    # Save the best model
    best_model = classifier.models['gru']
    best_model.save('best_model')
    print("\nBest model saved to 'best_model'")