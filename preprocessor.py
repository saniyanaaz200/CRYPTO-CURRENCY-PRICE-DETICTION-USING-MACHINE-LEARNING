import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DataPreprocessor:
    def __init__(self, target_col='Close', seq_length=60):
        self.target_col = target_col
        self.seq_length = seq_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def preprocess(self, df):
        # Extract and scale target column
        target_data = df[self.target_col].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(target_data)
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_data) - self.seq_length):
            X.append(scaled_data[i:i+self.seq_length])
            y.append(scaled_data[i+self.seq_length])
            
        return np.array(X), np.array(y), self.scaler
    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
