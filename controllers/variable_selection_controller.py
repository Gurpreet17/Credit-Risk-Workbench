"""
Variable selection page controller
Handles target and feature selection with validation
"""
from PyQt6.QtWidgets import QMainWindow, QMessageBox, QListWidgetItem
from PyQt6.QtGui import QColor
from PyQt6.QtCore import Qt
from ui.variable_selection_ui import Ui_VariableSelection


class VariableSelectionController(QMainWindow):
    def __init__(self, main_window):
        super().__init__()
        
        # Reference to main window
        self.main_window = main_window
        
        # Setup UI
        self.ui = Ui_VariableSelection()
        self.ui.setupUi(self)
        
        # Connect signals
        self.ui.back_btn.clicked.connect(self.go_back)
        self.ui.next_btn.clicked.connect(self.validate_and_continue)
        self.ui.target_dropdown.currentTextChanged.connect(self.on_target_changed)
        self.ui.select_all_btn.clicked.connect(self.select_all_features)
        self.ui.deselect_all_btn.clicked.connect(self.deselect_all_features)
        self.ui.search_box.textChanged.connect(self.filter_features)
        self.ui.feature_list.itemChanged.connect(self.update_selected_count)
    
    def populate_page(self):
        """Populate the page with dataset information"""
        train_rows, train_cols = self.main_window.train_data.shape
        val_rows, val_cols = self.main_window.validation_data.shape
        
        self.ui.train_info_label.setText(f"Training Data: {train_rows:,} rows × {train_cols} columns")
        self.ui.train_info_label.setStyleSheet("color: #27AE60; font-size: 11pt; font-weight: bold;")
        
        self.ui.val_info_label.setText(f"Validation Data: {val_rows:,} rows × {val_cols} columns")
        self.ui.val_info_label.setStyleSheet("color: #27AE60; font-size: 11pt; font-weight: bold;")
        
        # Populate target dropdown
        self.ui.target_dropdown.clear()
        self.ui.target_dropdown.addItem("-- Select Target Variable --")
        self.ui.target_dropdown.addItems(self.main_window.train_data.columns.tolist())
        
        # Populate feature list
        self.populate_feature_list()
        
        self.ui.next_btn.setEnabled(False)
        self.ui.statusbar.showMessage("Select target variable and features to continue")
    
    def populate_feature_list(self):
        """Populate feature list with checkboxes"""
        self.ui.feature_list.clear()
        
        for column in self.main_window.train_data.columns:
            item = QListWidgetItem(column)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.ui.feature_list.addItem(item)
        
        self.update_selected_count()
    
    def validate_binary_target(self, series, dataset_name="dataset"):
        """
        Validate that target is binary with values 0 and 1
        
        Args:
            series: pandas Series to check
            dataset_name: Name of dataset for error messages
            
        Returns:
            tuple: (is_valid: bool, error_message: str or None)
        """
        # Get unique values excluding NaN
        unique_values = series.dropna().unique()
        num_unique = len(unique_values)
        
        # Check if binary (exactly 2 values)
        if num_unique != 2:
            if num_unique < 2:
                return False, f"Target has only {num_unique} unique value(s) in {dataset_name}. Binary target requires exactly 2 values: 0 (Non-Target) and 1 (Target)."
            else:
                return False, f"Target has {num_unique} unique values in {dataset_name}. Binary target requires exactly 2 values: 0 (Non-Target) and 1 (Target)."
        
        # Check if values are exactly 0 and 1
        unique_set = set(unique_values)
        if unique_set != {0, 1}:
            # Show what values were found (only 2 values, so safe to show)
            values_str = str(sorted(unique_values.tolist()))
            return False, f"Target must contain only 0 (Non-Target) and 1 (Target). Found values: {values_str} in {dataset_name}."
        
        # All checks passed
        return True, None
    
    def on_target_changed(self, target_name):
        """Handle target variable selection"""
        if target_name == "-- Select Target Variable --" or not target_name:
            self.main_window.target_variable = None
            self.ui.target_info_label.setText("")
            # Re-enable all items
            for i in range(self.ui.feature_list.count()):
                item = self.ui.feature_list.item(i)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
                item.setBackground(QColor(255, 255, 255, 0))
            return
        
        # Check if target exists in validation set
        if target_name not in self.main_window.validation_data.columns:
            self.main_window.target_variable = None
            self.ui.target_info_label.setText(f"⚠ Warning: '{target_name}' not found in validation dataset!")
            self.ui.target_info_label.setStyleSheet("color: #E74C3C; font-size: 10pt; font-weight: bold;")
            return
        
        # ===== BINARY VALIDATION (0 and 1 only) =====
        # Validate training data
        is_valid_train, error_train = self.validate_binary_target(
            self.main_window.train_data[target_name], 
            "training data"
        )
        
        if not is_valid_train:
            self.main_window.target_variable = None
            self.ui.target_info_label.setText(f"✗ Invalid target")
            self.ui.target_info_label.setStyleSheet("color: #E74C3C; font-size: 10pt; font-weight: bold;")
            
            QMessageBox.critical(
                self,
                "Invalid Target Variable",
                f"The target variable '{target_name}' is not valid.\n\n"
                f"{error_train}\n\n"
                f"Requirements:\n"
                f"• Target must be binary (2 unique values)\n"
                f"• Values must be exactly 0 and 1\n"
                f"• 0 = Non-Target (e.g., No Default)\n"
                f"• 1 = Target (e.g., Default)"
            )
            return
        
        # Validate validation data
        is_valid_val, error_val = self.validate_binary_target(
            self.main_window.validation_data[target_name],
            "validation data"
        )
        
        if not is_valid_val:
            self.main_window.target_variable = None
            self.ui.target_info_label.setText(f"✗ Invalid target in validation set")
            self.ui.target_info_label.setStyleSheet("color: #E74C3C; font-size: 10pt; font-weight: bold;")
            
            QMessageBox.critical(
                self,
                "Invalid Target Variable",
                f"The target variable '{target_name}' is not valid in validation set.\n\n"
                f"{error_val}\n\n"
                f"Requirements:\n"
                f"• Target must be binary (2 unique values)\n"
                f"• Values must be exactly 0 and 1\n"
                f"• 0 = Non-Target (e.g., No Default)\n"
                f"• 1 = Target (e.g., Default)"
            )
            return
        
        # All validations passed!
        self.main_window.target_variable = target_name
        
        # Calculate statistics
        train_target = self.main_window.train_data[target_name]
        null_count = train_target.isnull().sum()
        
        # Calculate class distribution
        count_0 = (train_target == 0).sum()
        count_1 = (train_target == 1).sum()
        total = count_0 + count_1
        
        pct_0 = (count_0 / total * 100) if total > 0 else 0
        pct_1 = (count_1 / total * 100) if total > 0 else 0
        
        self.ui.target_info_label.setText(
            f"✓ Binary target (0/1) | "
            f"0: {count_0:,} ({pct_0:.1f}%) | "
            f"1: {count_1:,} ({pct_1:.1f}%) | "
            f"Missing: {null_count}"
        )
        self.ui.target_info_label.setStyleSheet("color: #27AE60; font-size: 10pt; font-style: italic;")
        
        # Disable target in feature list
        for i in range(self.ui.feature_list.count()):
            item = self.ui.feature_list.item(i)
            if item.text() == target_name:
                item.setCheckState(Qt.CheckState.Unchecked)
                item.setFlags(Qt.ItemFlag.NoItemFlags)
                item.setBackground(QColor("#F0F0F0"))
            else:
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
                item.setBackground(QColor(255, 255, 255, 0))
    
    def select_all_features(self):
        """Select all enabled features"""
        for i in range(self.ui.feature_list.count()):
            item = self.ui.feature_list.item(i)
            if item.flags() & Qt.ItemFlag.ItemIsEnabled:
                item.setCheckState(Qt.CheckState.Checked)
        self.update_selected_count()
    
    def deselect_all_features(self):
        """Deselect all features"""
        for i in range(self.ui.feature_list.count()):
            item = self.ui.feature_list.item(i)
            if item.flags() & Qt.ItemFlag.ItemIsEnabled:
                item.setCheckState(Qt.CheckState.Unchecked)
        self.update_selected_count()
    
    def filter_features(self, search_text):
        """Filter feature list based on search"""
        search_text = search_text.lower()
        for i in range(self.ui.feature_list.count()):
            item = self.ui.feature_list.item(i)
            item.setHidden(search_text not in item.text().lower())
    
    def update_selected_count(self):
        """Update selected features count"""
        selected_count = sum(
            1 for i in range(self.ui.feature_list.count())
            if self.ui.feature_list.item(i).checkState() == Qt.CheckState.Checked
        )
        
        self.ui.selected_count_label.setText(f"Selected: {selected_count} variables")
        
        # Enable next button if valid selection
        self.ui.next_btn.setEnabled(bool(self.main_window.target_variable and selected_count > 0))
    
    def validate_and_continue(self):
        """Validate selections and filter datasets"""
        # Check target is selected
        if not self.main_window.target_variable:
            QMessageBox.warning(self, "No Target Selected", "Please select a target variable before continuing.")
            return
        
        # Get selected features
        self.main_window.selected_features = [
            self.ui.feature_list.item(i).text()
            for i in range(self.ui.feature_list.count())
            if self.ui.feature_list.item(i).checkState() == Qt.CheckState.Checked
        ]
        
        if not self.main_window.selected_features:
            QMessageBox.warning(self, "No Features Selected", "Please select at least one feature variable.")
            return
        
        # ===== FINAL BINARY CHECK (0 and 1) =====
        is_valid_train, error_train = self.validate_binary_target(
            self.main_window.train_data[self.main_window.target_variable],
            "training data"
        )
        is_valid_val, error_val = self.validate_binary_target(
            self.main_window.validation_data[self.main_window.target_variable],
            "validation data"
        )
        
        if not is_valid_train or not is_valid_val:
            QMessageBox.critical(
                self, "Invalid Target",
                f"Target variable validation failed.\n\n"
                f"Training: {error_train or 'Valid'}\n"
                f"Validation: {error_val or 'Valid'}\n\n"
                f"Target must contain only 0 (Non-Target) and 1 (Target)."
            )
            return
        
        # Validate target exists in validation set (redundant but safe)
        if self.main_window.target_variable not in self.main_window.validation_data.columns:
            QMessageBox.critical(
                self, "Validation Error",
                f"The validation set should have the target variable '{self.main_window.target_variable}'.\n\n"
                f"This variable exists in the training set but not in the validation set."
            )
            return
        
        # Validate features exist in validation set
        missing_features = [
            f for f in self.main_window.selected_features
            if f not in self.main_window.validation_data.columns
        ]
        
        if missing_features:
            QMessageBox.critical(
                self, "Validation Error",
                f"The validation set should have the following variables:\n\n" +
                "\n".join(f"• {var}" for var in missing_features) +
                f"\n\nThese variables exist in the training set but not in the validation set."
            )
            return
        
        # Filter datasets
        selected_columns = [self.main_window.target_variable] + self.main_window.selected_features
        
        try:
            self.main_window.filtered_train_data = self.main_window.train_data[selected_columns].copy()
            self.main_window.filtered_validation_data = self.main_window.validation_data[selected_columns].copy()
            
            # Print summary to console (NO POPUP) - NEW
            print("\n" + "="*70)
            print("VARIABLE SELECTION COMPLETE")
            print("="*70)
            print(f"Target: {self.main_window.target_variable}")
            print(f"Features: {len(self.main_window.selected_features)}")
            print(f"Feature list: {self.main_window.selected_features}")
            print(f"Training: {len(self.main_window.filtered_train_data)} rows × {len(selected_columns)} columns")
            print(f"Validation: {len(self.main_window.filtered_validation_data)} rows × {len(selected_columns)} columns")
            
            # Get target distribution for summary
            train_target = self.main_window.filtered_train_data[self.main_window.target_variable]
            count_0 = (train_target == 0).sum()
            count_1 = (train_target == 1).sum()
            total = count_0 + count_1
            pct_0 = (count_0 / total * 100) if total > 0 else 0
            pct_1 = (count_1 / total * 100) if total > 0 else 0
            print(f"Target distribution: 0={count_0} ({pct_0:.1f}%), 1={count_1} ({pct_1:.1f}%)")
            print("="*70 + "\n")
            
            # ===== NAVIGATE TO BINNING (force_rerun=True) =====
            self.main_window.show_binning_page(force_rerun=True)
            
        except Exception as e:
            import traceback
            error_msg = f"Error filtering datasets:\n\n{str(e)}\n\n{traceback.format_exc()}"
            print(error_msg)
            QMessageBox.critical(self, "Error", error_msg)
    
    def go_back(self):
        """Navigate back to home page"""
        self.main_window.show_home_page()