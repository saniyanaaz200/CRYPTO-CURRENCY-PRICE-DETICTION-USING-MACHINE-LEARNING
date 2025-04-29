from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

class ModelBuilder:
    @staticmethod
    def build_lstm(input_shape):
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        
        # Second LSTM layer
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        
        # Dense layers
        model.add(Dense(25))
        model.add(Dense(1))
        
        # Compile
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse')
        
        return model
