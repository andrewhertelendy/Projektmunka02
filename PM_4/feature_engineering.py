import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import RobustScaler

# Feature Engineering Class
class TrajectoryFeatureEngineering:
    def __init__(self, window_size=5):
        self.window_size = window_size

    def create_kinematic_features(self, df):
        """Create kinematic features from the input dataframe"""
        # Total velocity and acceleration
        df['total_velocity'] = np.sqrt(df['xVelocity']**2 + df['yVelocity']**2).clip(0, 100)
        df['total_acceleration'] = np.sqrt(df['xAcceleration']**2 + df['yAcceleration']**2).clip(0, 50)
        
        # Rate of heading change
        df['heading_change_rate'] = df.groupby('trackId')['heading_change'].transform(lambda x: x.diff()).fillna(0).clip(-50, 50)
        
        # Turning radius
        heading_change_safe = np.where(np.abs(df['heading_change']) > 0.001, df['heading_change'], 0.001)
        df['turning_radius'] = (df['total_velocity'] / np.abs(heading_change_safe)).clip(0, 1000)
        
        # Angular velocity
        frame_diff = df.groupby('trackId')['frame'].diff().fillna(1)
        df['angular_velocity'] = np.abs(df.groupby('trackId')['heading'].diff() / frame_diff).fillna(0).clip(0, 50)
    
        return df

    def create_trajectory_features(self, df):
        """Create trajectory features using derivatives."""
        df['dx'] = df.groupby('trackId')['xCenter'].diff().fillna(0)
        df['dy'] = df.groupby('trackId')['yCenter'].diff().fillna(0)
        df['ddx'] = df.groupby('trackId')['dx'].diff().fillna(0)
        df['ddy'] = df.groupby('trackId')['dy'].diff().fillna(0)
        
        # Curvature
        numerator = df['dx'] * df['ddy'] - df['dy'] * df['ddx']
        denominator = (df['dx']**2 + df['dy']**2)**(3/2)
        denominator = np.where(denominator > 0.001, denominator, 0.001)
        df['curvature'] = np.abs(numerator / denominator).clip(0, 100)
        
        return df
    
    def create_statistical_features(self, df):
        """Create statistical features using rolling windows."""
        grouped = df.groupby('trackId')
        for feature in ['heading_change', 'total_velocity', 'angular_velocity']:
            df[f'{feature}_mean'] = grouped[feature].transform(
                lambda x: x.rolling(window=self.window_size, center=True, min_periods=1).mean()
            ).fillna(0)
            df[f'{feature}_std'] = grouped[feature].transform(
                lambda x: x.rolling(window=self.window_size, center=True, min_periods=1).std()
            ).fillna(0)
            df[f'{feature}_max'] = grouped[feature].transform(
                lambda x: x.rolling(window=self.window_size, center=True, min_periods=1).max()
            ).fillna(0)
        return df

    def select_features(self, df, feature_columns):
        """Select relevant features for training"""
        features = [
            *feature_columns, # Original features
            'total_velocity', 'total_acceleration', 'heading_change_rate', 'turning_radius', 'angular_velocity',  # New kinematic
            'curvature', # Trajectory
            'heading_change_mean', 'heading_change_std', 'heading_change_max', # Statistical
            'total_velocity_mean', 'total_velocity_std', 'total_velocity_max',
            'angular_velocity_mean', 'angular_velocity_std', 'angular_velocity_max'
        ]
        return df[features]
        
    def smooth_features(self, df):
        """Apply Savitzky-Golay filter to smooth kinematic features."""
        grouped = df.groupby('trackId')
        
        for feature in ['total_velocity', 'total_acceleration', 'heading_change_rate', 'angular_velocity']:
          df[feature] = grouped[feature].transform(lambda x: signal.savgol_filter(x, window_length=5, polyorder=2, mode='nearest'))
        return df
    
    def scale_features(self, df):
        """Scale features using RobustScaler."""
        scaler = RobustScaler()
        numerical_cols = df.select_dtypes(include=np.number).columns
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        return df

    def process(self, df, feature_columns):
        """Main processing function"""
        df = self.create_kinematic_features(df)
        df = self.create_trajectory_features(df)
        df = self.create_statistical_features(df)
        df = self.smooth_features(df)
        df = self.select_features(df, feature_columns)
        df = self.scale_features(df)
        return df

# Main execution
if __name__ == '__main__':
  # Load processed data
    train_data = pd.read_csv('train_data_processed.csv')
    test_data = pd.read_csv('test_data_processed.csv')
    feature_columns = [col for col in train_data.columns if col not in ['turn_binary', 'turn_label', 'file_id']]
  
  # Feature engineering
    feature_engineer = TrajectoryFeatureEngineering()
    train_data_engineered = feature_engineer.process(train_data, feature_columns)
    test_data_engineered = feature_engineer.process(test_data, feature_columns)
  
  # Save engineered data
    train_data_engineered.to_csv('train_data_engineered.csv', index=False)
    test_data_engineered.to_csv('test_data_engineered.csv', index=False)
    print("Engineered data saved to 'train_data_engineered.csv' and 'test_data_engineered.csv'")