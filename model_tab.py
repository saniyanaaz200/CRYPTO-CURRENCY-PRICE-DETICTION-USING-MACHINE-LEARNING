from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                            QPushButton, QGroupBox, QProgressBar, QSpinBox,
                            QDoubleSpinBox, QMessageBox)
from PyQt5.QtCore import Qt
from trainer import ModelTrainer

class ModelTab(QWidget):
    def __init__(self, data_tab):
        super().__init__()
        self.data_tab = data_tab
        self.trainer = None
        self.model_result = None
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Parameters
        params_group = QGroupBox("Model Parameters")
        params_layout = QVBoxLayout()
        
        self.seq_length = QSpinBox()
        self.seq_length.setRange(1, 365)
        self.seq_length.setValue(60)
        self._add_parameter(params_layout, "Sequence Length:", self.seq_length)
        
        self.test_size = QDoubleSpinBox()
        self.test_size.setRange(0.01, 0.5)
        self.test_size.setSingleStep(0.01)
        self.test_size.setValue(0.2)
        self._add_parameter(params_layout, "Test Size:", self.test_size)
        
        params_group.setLayout(params_layout)
        
        # Training
        train_group = QGroupBox("Training")
        train_layout = QVBoxLayout()
        
        self.progress = QProgressBar()
        self.progress.setAlignment(Qt.AlignCenter)
        
        btn_layout = QHBoxLayout()
        self.train_btn = QPushButton("Train Model")
        self.train_btn.clicked.connect(self.start_training)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_training)
        self.stop_btn.setEnabled(False)
        
        btn_layout.addWidget(self.train_btn)
        btn_layout.addWidget(self.stop_btn)
        
        train_layout.addWidget(self.progress)
        train_layout.addLayout(btn_layout)
        train_group.setLayout(train_layout)
        
        layout.addWidget(params_group)
        layout.addWidget(train_group)
        self.setLayout(layout)
    
    def _add_parameter(self, layout, label, widget):
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        row.addWidget(widget)
        layout.addLayout(row)
    
    def start_training(self):
        if self.data_tab.df is None:
            QMessageBox.warning(self, "Warning", "Please load data first")
            return
            
        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress.setValue(0)
        
        self.trainer = ModelTrainer(
            df=self.data_tab.df,
            target_col=self.data_tab.target_col.currentText(),
            seq_length=self.seq_length.value(),
            test_size=self.test_size.value()
        )
        
        self.trainer.progress_updated.connect(self.progress.setValue)
        self.trainer.training_completed.connect(self.on_training_complete)
        self.trainer.error_occurred.connect(self.on_training_error)
        self.trainer.start()
    
    def stop_training(self):
        if self.trainer:
            self.trainer.stop()
            self.train_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
    
    def on_training_complete(self, result):
        self.model_result = result
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        QMessageBox.information(self, "Success", "Training completed!")
    
    def on_training_error(self, error):
        QMessageBox.critical(self, "Error", error)
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
