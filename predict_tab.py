from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                            QPushButton, QGroupBox, QTextEdit, QSplitter,
                            QSpinBox, QMessageBox)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar
)
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

class PredictTab(QWidget):
    def __init__(self, data_tab, model_tab):
        super().__init__()
        self.data_tab = data_tab
        self.model_tab = model_tab
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        splitter = QSplitter(Qt.Vertical)
        
        # Plot area
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        plot_widget.setLayout(plot_layout)
        
        # Controls area
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        
        # Parameters
        params_group = QGroupBox("Prediction Parameters")
        params_layout = QHBoxLayout()
        
        params_layout.addWidget(QLabel("Days to Predict:"))
        self.days_spinbox = QSpinBox()
        self.days_spinbox.setRange(1, 365)
        self.days_spinbox.setValue(30)
        
        predict_btn = QPushButton("Predict")
        predict_btn.clicked.connect(self.predict)
        
        params_layout.addWidget(self.days_spinbox)
        params_layout.addWidget(predict_btn)
        params_group.setLayout(params_layout)
        
        # Results
        results_group = QGroupBox("Prediction Results")
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        
        results_layout = QVBoxLayout()
        results_layout.addWidget(self.results_text)
        results_group.setLayout(results_layout)
        
        controls_layout.addWidget(params_group)
        controls_layout.addWidget(results_group)
        controls_widget.setLayout(controls_layout)
        
        # Assemble
        splitter.addWidget(plot_widget)
        splitter.addWidget(controls_widget)
        splitter.setSizes([600, 200])
        
        layout.addWidget(splitter)
        self.setLayout(layout)
    
    def predict(self):
        if not hasattr(self.model_tab, 'model_result'):
            QMessageBox.warning(self, "Warning", "Train model first")
            return
            
        try:
            days = self.days_spinbox.value()
            result = self.model_tab.model_result
            model = result['model']
            preprocessor = result['preprocessor']
            target_col = self.data_tab.target_col.currentText()
            seq_length = preprocessor.seq_length
            
            # Get last sequence
            last_seq = self.data_tab.df[target_col].values[-seq_length:]
            scaled_seq = preprocessor.scaler.transform(last_seq.reshape(-1, 1))
            
            # Generate predictions
            predictions = []
            current_seq = scaled_seq.copy()
            
            for _ in range(days):
                x_input = current_seq[-seq_length:].reshape(1, seq_length, 1)
                pred = model.predict(x_input, verbose=0)[0, 0]
                predictions.append(pred)
                current_seq = np.append(current_seq, pred)
            
            # Inverse transform
            predicted_prices = preprocessor.scaler.inverse_transform(
                np.array(predictions).reshape(-1, 1)
            ).flatten()
            
            # Generate dates
            date_col = self.data_tab.date_col.currentText()
            if date_col != "None":
                last_date = pd.to_datetime(self.data_tab.df[date_col].iloc[-1])
                dates = pd.date_range(start=last_date, periods=days+1)[1:]
                date_labels = [d.strftime('%Y-%m-%d') for d in dates]
            else:
                date_labels = [f"Day {i+1}" for i in range(days)]
            
            # Update plot
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            ax.plot(date_labels, predicted_prices, 'g-', marker='o', label='Predicted Price')
            ax.set_title("Future Price Prediction")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price ($)")
            ax.legend()
            ax.grid(True)
            
            # Rotate date labels if needed
            if len(date_labels[0]) > 6:  # If using actual dates not "Day X"
                for label in ax.get_xticklabels():
                    label.set_rotation(45)
                    label.set_ha('right')
            
            self.figure.tight_layout()
            self.canvas.draw()
            
            # Update results text
            results = [f"{date}: {price:.2f}" for date, price in zip(date_labels, predicted_prices)]
            self.results_text.setPlainText("Predictions:\n" + "\n".join(results))
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Prediction failed: {str(e)}")
