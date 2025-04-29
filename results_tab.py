from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                            QPushButton, QGroupBox, QTextEdit, QSplitter,
                            QMessageBox)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar
)
from matplotlib.figure import Figure
import numpy as np

class ResultsTab(QWidget):
    def __init__(self, model_tab):
        super().__init__()
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
        
        # Metrics area
        metrics_widget = QWidget()
        metrics_layout = QVBoxLayout(metrics_widget)
        
        metrics_group = QGroupBox("Evaluation Metrics")
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        
        plot_btn = QPushButton("Plot Results")
        plot_btn.clicked.connect(self.plot_results)
        
        metrics_group_layout = QVBoxLayout()
        metrics_group_layout.addWidget(self.metrics_text)
        metrics_group.setLayout(metrics_group_layout)
        
        metrics_layout.addWidget(metrics_group)
        metrics_layout.addWidget(plot_btn)
        metrics_widget.setLayout(metrics_layout)
        
        # Assemble
        splitter.addWidget(plot_widget)
        splitter.addWidget(metrics_widget)
        splitter.setSizes([600, 200])
        
        layout.addWidget(splitter)
        self.setLayout(layout)
    
    def plot_results(self):
        if not hasattr(self.model_tab, 'model_result'):
            QMessageBox.warning(self, "Warning", "Train model first")
            return
            
        result = self.model_tab.model_result
        model = result['model']
        X_test = result['X_test']
        y_test = result['y_test']
        scaler = result['preprocessor'].scaler
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Inverse transform
        y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred = scaler.inverse_transform(y_pred).flatten()
        
        # Calculate metrics
        mse = np.mean((y_true - y_pred)**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        
        # Update metrics
        self.metrics_text.setPlainText(
            f"MSE: {mse:.4f}\n"
            f"RMSE: {rmse:.4f}\n"
            f"MAE: {mae:.4f}"
        )
        
        # Plot
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        ax.plot(y_true, label='Actual', color='blue', linewidth=2)
        ax.plot(y_pred, label='Predicted', color='red', linestyle='--', linewidth=2)
        
        ax.set_title("Actual vs Predicted Prices", fontsize=14)
        ax.set_xlabel("Time Steps", fontsize=12)
        ax.set_ylabel("Price", fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        self.figure.tight_layout()
        self.canvas.draw()
