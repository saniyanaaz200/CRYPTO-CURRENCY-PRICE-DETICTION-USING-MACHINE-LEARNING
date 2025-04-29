import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget
from data_tab import DataTab
from model_tab import ModelTab
from results_tab import ResultsTab
from predict_tab import PredictTab

class StockPredictionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stock Price Prediction")
        self.setGeometry(100, 100, 1200, 800)
        self.init_ui()
        
    def init_ui(self):
        tabs = QTabWidget()
        
        # Initialize tabs with proper dependencies
        self.data_tab = DataTab()
        self.model_tab = ModelTab(self.data_tab)
        self.results_tab = ResultsTab(self.model_tab)
        self.predict_tab = PredictTab(self.data_tab, self.model_tab)
        
        # Add tabs
        tabs.addTab(self.data_tab, "📊 Data")
        tabs.addTab(self.model_tab, "🤖 Model") 
        tabs.addTab(self.results_tab, "📈 Results")
        tabs.addTab(self.predict_tab, "🔮 Prediction")
        
        self.setCentralWidget(tabs)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StockPredictionApp()
    window.show()
    sys.exit(app.exec_())
