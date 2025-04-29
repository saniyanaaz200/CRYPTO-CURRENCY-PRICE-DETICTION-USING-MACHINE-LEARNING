from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                            QPushButton, QComboBox, QLineEdit, QGroupBox,
                            QTextEdit, QFileDialog, QMessageBox)
import pandas as pd

class DataTab(QWidget):
    def __init__(self):
        super().__init__()
        self.df = None
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # File selection
        file_group = QGroupBox("Data Source")
        file_layout = QVBoxLayout()
        
        path_layout = QHBoxLayout()
        self.file_path = QLineEdit()
        self.file_path.setPlaceholderText("Select CSV file...")
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.load_file)
        path_layout.addWidget(self.file_path)
        path_layout.addWidget(browse_btn)
        
        # Column selection
        col_group = QGroupBox("Column Selection")
        col_layout = QVBoxLayout()
        
        target_layout = QHBoxLayout()
        target_layout.addWidget(QLabel("Target Column:"))
        self.target_col = QComboBox()
        target_layout.addWidget(self.target_col)
        
        date_layout = QHBoxLayout()
        date_layout.addWidget(QLabel("Date Column:"))
        self.date_col = QComboBox()
        date_layout.addWidget(self.date_col)
        
        col_layout.addLayout(target_layout)
        col_layout.addLayout(date_layout)
        col_group.setLayout(col_layout)
        
        # Data preview
        preview_group = QGroupBox("Data Preview")
        preview_layout = QVBoxLayout()
        self.data_preview = QTextEdit()
        self.data_preview.setReadOnly(True)
        preview_layout.addWidget(self.data_preview)
        preview_group.setLayout(preview_layout)
        
        # Assemble
        file_layout.addLayout(path_layout)
        file_layout.addWidget(col_group)
        file_group.setLayout(file_layout)
        
        layout.addWidget(file_group)
        layout.addWidget(preview_group)
        self.setLayout(layout)
    
    def load_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open CSV", "", "CSV Files (*.csv);;All Files (*)"
        )
        
        if not filename:
            return
            
        try:
            self.df = pd.read_csv(filename)
            self.file_path.setText(filename)
            self.data_preview.setText(str(self.df.head()))
            
            # Update comboboxes
            cols = self.df.columns.tolist()
            self.target_col.clear()
            self.date_col.clear()
            self.target_col.addItems(cols)
            self.date_col.addItems(cols + ["None"])
            
            # Auto-select likely columns
            for col in ['Close', 'close', 'price']:
                if col in cols:
                    self.target_col.setCurrentText(col)
                    break
                    
            for col in ['Date', 'date']:
                if col in cols:
                    self.date_col.setCurrentText(col)
                    break
                    
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file: {str(e)}")
