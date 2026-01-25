"""
Home page controller
Handles data loading and navigation to variable selection
"""
import os
from PyQt6.QtWidgets import QMainWindow, QFileDialog, QMessageBox
from ui.home_ui import Ui_MainWindow
from utils.data_loader import load_data_universal
from sklearn.model_selection import train_test_split
import webbrowser


    
   


class HomeController(QMainWindow):
    def __init__(self, main_window):
        super().__init__()
        
        # Reference to main window (for navigation and data sharing)
        self.main_window = main_window
        
        # Setup UI
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        #Connect footer links
        self.setup_footer_links()
        
        # Connect signals
        self.ui.import_train_data_btn.clicked.connect(self.import_train_data)
        self.ui.import_val_data_btn.clicked.connect(self.import_val_data)
        self.ui.next_btn.clicked.connect(self.go_to_variable_selection)
        
        # Connect split toggle
        self.ui.use_train_split_checkbox.stateChanged.connect(self.on_split_toggle_changed)
        self.ui.split_percentage_dropdown.currentTextChanged.connect(self.update_split_info)
    
    def on_split_toggle_changed(self, state):
        """Handle train/val split toggle"""
        is_checked = bool(state)
        
        # Enable/disable split percentage dropdown
        self.ui.split_percentage_dropdown.setEnabled(is_checked)
        
        if is_checked:
            # Disable validation upload
            self.ui.import_val_data_btn.setEnabled(False)
            self.ui.import_val_data_btn.setStyleSheet("""
                QPushButton {
                    background-color: #BDC3C7;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    padding: 12px 24px;
                    font-size: 13px;
                    font-weight: bold;
                    font-family: 'Segoe UI';
                }
            """)
            
            # Update split info
            self.update_split_info()
            
            # Check if we can enable next button
            self.check_enable_next_button()
        else:
            # Enable validation upload
            self.ui.import_val_data_btn.setEnabled(True)
            self.ui.import_val_data_btn.setStyleSheet("""
                QPushButton {
                    background-color: #3498DB;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    padding: 12px 24px;
                    font-size: 13px;
                    font-weight: bold;
                    font-family: 'Segoe UI';
                }
                QPushButton:hover {
                    background-color: #2980B9;
                }
                QPushButton:pressed {
                    background-color: #21618C;
                }
            """)
            
            # Hide split info
            self.ui.split_info_label.setVisible(False)
            
            # Check if we can enable next button
            self.check_enable_next_button()
    
    def update_split_info(self):
        """Update split info label"""
        if not self.ui.use_train_split_checkbox.isChecked():
            self.ui.split_info_label.setVisible(False)
            return
        
        if self.main_window.train_data is None:
            self.ui.split_info_label.setVisible(False)
            return
        
        # Get split percentage
        split_text = self.ui.split_percentage_dropdown.currentText()
        val_pct = int(split_text.replace('%', ''))
        train_pct = 100 - val_pct
        
        total_rows = len(self.main_window.train_data)
        train_rows = int(total_rows * train_pct / 100)
        val_rows = total_rows - train_rows
        
        self.ui.split_info_label.setText(
            f"✓ Auto-split enabled: Training {train_pct}% ({train_rows:,} rows) | Validation {val_pct}% ({val_rows:,} rows)"
        )
        self.ui.split_info_label.setVisible(True)
    
    def import_train_data(self):
        """Handle training data import"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Training Dataset",
            "",
            "All Supported Files (*.csv *.xlsx *.xls *.json *.parquet *.pq *.feather *.sas7bdat *.xpt *.dta *.sav *.h5 *.pkl);;"
            "CSV Files (*.csv);;"
            "Excel Files (*.xlsx *.xls *.xlsm *.xlsb);;"
            "All Files (*)"
        )
        
        if file_path:
            file_name = os.path.basename(file_path)
            self.ui.statusbar.showMessage(f"Loading {file_name}...")
            
            df, error = load_data_universal(file_path)
            
            if error:
                self.ui.import_train_data_btn_status.setText("✗ Load failed")
                self.ui.import_train_data_btn_status.setStyleSheet(
                    "color: #E74C3C; background: transparent; border: none; font-style: normal; font-weight: bold;"
                )
                self.ui.statusbar.showMessage("Failed to load training dataset")
                QMessageBox.critical(self, "Error Loading File", error)
                return
            
            # Store in main window
            self.main_window.train_data = df
            self.main_window.train_data_path = file_path
            
            rows, cols = df.shape
            memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            
            self.ui.import_train_data_btn_status.setText(f"✓ {file_name}")
            self.ui.import_train_data_btn_status.setStyleSheet(
                "color: #27AE60; background: transparent; border: none; font-style: normal; font-weight: bold;"
            )
            
            self.ui.statusbar.showMessage(
                f"Training dataset loaded: {file_name} ({rows:,} rows × {cols} cols, {memory_mb:.1f} MB)"
            )
            
            QMessageBox.information(
                self,
                "Success",
                f"Training dataset loaded successfully!\n\n"
                f"File: {file_name}\n"
                f"Rows: {rows:,}\n"
                f"Columns: {cols}\n"
                f"Memory: {memory_mb:.1f} MB"
            )
            
            # Update split info if enabled
            self.update_split_info()
            
            self.check_enable_next_button()
    
    def import_val_data(self):
        """Handle validation data import"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Validation Dataset",
            "",
            "All Supported Files (*.csv *.xlsx *.xls *.json *.parquet *.pq *.feather *.sas7bdat *.xpt *.dta *.sav *.h5 *.pkl);;"
            "CSV Files (*.csv);;"
            "Excel Files (*.xlsx *.xls *.xlsm *.xlsb);;"
            "All Files (*)"
        )
        
        if file_path:
            file_name = os.path.basename(file_path)
            self.ui.statusbar.showMessage(f"Loading {file_name}...")
            
            df, error = load_data_universal(file_path)
            
            if error:
                self.ui.import_val_data_btn_status.setText("✗ Load failed")
                self.ui.import_val_data_btn_status.setStyleSheet(
                    "color: #E74C3C; background: transparent; border: none; font-style: normal; font-weight: bold;"
                )
                self.ui.statusbar.showMessage("Failed to load validation dataset")
                QMessageBox.critical(self, "Error Loading File", error)
                return
            
            # Store in main window
            self.main_window.validation_data = df
            self.main_window.validation_data_path = file_path
            
            rows, cols = df.shape
            memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            
            self.ui.import_val_data_btn_status.setText(f"✓ {file_name}")
            self.ui.import_val_data_btn_status.setStyleSheet(
                "color: #27AE60; background: transparent; border: none; font-style: normal; font-weight: bold;"
            )
            
            self.ui.statusbar.showMessage(
                f"Validation dataset loaded: {file_name} ({rows:,} rows × {cols} cols, {memory_mb:.1f} MB)"
            )
            
            QMessageBox.information(
                self,
                "Success",
                f"Validation dataset loaded successfully!\n\n"
                f"File: {file_name}\n"
                f"Rows: {rows:,}\n"
                f"Columns: {cols}\n"
                f"Memory: {memory_mb:.1f} MB"
            )
            
            self.check_enable_next_button()
    
    def check_enable_next_button(self):
        """Enable next button if both datasets are loaded OR auto-split is enabled"""
        # Check if using auto-split
        if self.ui.use_train_split_checkbox.isChecked():
            # Only need training data
            if self.main_window.train_data is not None:
                self.ui.next_btn.setEnabled(True)
                self.ui.statusbar.showMessage("Training dataset loaded. Auto-split enabled. Ready to continue!")
            else:
                self.ui.next_btn.setEnabled(False)
        else:
            # Need both datasets
            if self.main_window.train_data is not None and self.main_window.validation_data is not None:
                self.ui.next_btn.setEnabled(True)
                self.ui.statusbar.showMessage("Both datasets loaded. Ready to continue!")
            else:
                self.ui.next_btn.setEnabled(False)
    
    def perform_train_val_split(self):
        """Perform train/validation split on training data"""
        if self.main_window.train_data is None:
            return False
        
        # Get split percentage
        split_text = self.ui.split_percentage_dropdown.currentText()
        val_pct = int(split_text.replace('%', '')) / 100
        
        try:
            # Perform stratified split (maintains class distribution)
            # Note: We'll do this after target selection, but prepare here
            train_df, val_df = train_test_split(
                self.main_window.train_data,
                test_size=val_pct,
                random_state=42,
                shuffle=True
            )
            
            # Store split data
            self.main_window.validation_data = val_df.reset_index(drop=True)
            self.main_window.train_data = train_df.reset_index(drop=True)
            self.main_window.validation_data_path = "AUTO_SPLIT"
            
            print(f"\n=== Train/Validation Split ===")
            print(f"Original data: {len(self.main_window.train_data) + len(self.main_window.validation_data):,} rows")
            print(f"Training: {len(train_df):,} rows ({(1-val_pct)*100:.0f}%)")
            print(f"Validation: {len(val_df):,} rows ({val_pct*100:.0f}%)")
            
            return True
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Split Error",
                f"Error splitting data:\n\n{str(e)}"
            )
            return False
    
    def go_to_variable_selection(self):
        """Navigate to variable selection page"""
        if self.main_window.train_data is None:
            QMessageBox.warning(self, "Missing Data", "Please upload the training dataset before continuing.")
            return
        
        # Check if using auto-split
        if self.ui.use_train_split_checkbox.isChecked():
            # Perform split
            if not self.perform_train_val_split():
                return
        else:
            # Need validation dataset
            if self.main_window.validation_data is None:
                QMessageBox.warning(self, "Missing Data", "Please upload the validation dataset before continuing.")
                return
        
        # Tell main window to navigate
        self.main_window.show_variable_selection_page()
        
    def setup_footer_links(self):
            """Connect social media buttons"""
            # Replace with YOUR actual URLs
            self.ui.linkedin_btn.clicked.connect(
                lambda: webbrowser.open('https://www.linkedin.com/in/gurpreet17/')
            )
            
            self.ui.github_btn.clicked.connect(
                lambda: webbrowser.open('https://github.com/Gurpreet17')
            )
