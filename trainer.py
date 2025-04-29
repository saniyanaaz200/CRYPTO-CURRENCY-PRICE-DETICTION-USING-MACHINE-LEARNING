from PyQt5.QtCore import QThread, pyqtSignal
import tensorflow as tf
from preprocessor import DataPreprocessor
from model_builder import ModelBuilder
import numpy as np

class ModelTrainer(QThread):
    progress_updated = pyqtSignal(int)
    training_completed = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, df, target_col, seq_length, test_size):
        super().__init__()
        self.df = df
        self.target_col = target_col
        self.seq_length = seq_length
        self.test_size = test_size
        self._running = True

    def run(self):
        try:
            # Preprocess data
            preprocessor = DataPreprocessor(self.target_col, self.seq_length)
            X, y, scaler = preprocessor.preprocess(self.df)
            
            # Split data
            split_idx = int(len(X) * (1 - self.test_size))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Create dataset
            train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
            train_dataset = train_dataset.shuffle(1000).batch(32)
            
            # Build model
            model = ModelBuilder.build_lstm((X.shape[1], 1))
            
            # Callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='loss',
                    patience=5,
                    restore_best_weights=True
                ),
                self._create_progress_callback()
            ]
            
            # Train
            history = model.fit(
                train_dataset,
                epochs=50,
                callbacks=callbacks,
                verbose=0
            )
            
            self.training_completed.emit({
                'model': model,
                'preprocessor': preprocessor,
                'X_test': X_test,
                'y_test': y_test,
                'history': history.history
            })
            
        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            self._running = False

    def _create_progress_callback(self):
        class ProgressCallback(tf.keras.callbacks.Callback):
            def __init__(self, trainer):
                super().__init__()
                self.trainer = trainer
            
            def on_epoch_end(self, epoch, logs=None):
                if not self.trainer._running:
                    self.model.stop_training = True
                progress = int((epoch + 1) / 50 * 100)
                self.trainer.progress_updated.emit(progress)
        
        return ProgressCallback(self)

    def stop(self):
        self._running = False
        self.wait()
