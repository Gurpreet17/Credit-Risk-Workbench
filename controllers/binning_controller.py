"""
Binning page controller
Handles binning execution using existing binner class
"""
from PyQt6 import QtWidgets
from PyQt6.QtWidgets import (QMainWindow, QMessageBox, QTableWidgetItem, 
                              QInputDialog, QHeaderView)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from ui.binning_ui import Ui_Binning
from ui.loading_dialog_ui import LoadingDialog
from utils.binning_engine import BinningEngineWrapper
from utils.plotting import WOEPlotter
import traceback
import pandas as pd
import copy
import numpy as np
import time


class BinningWorker(QThread):
    """Worker thread for binning computation"""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)  # Returns BinningEngineWrapper
    error = pyqtSignal(str)
    
    def __init__(self, train_data, target_variable):
        super().__init__()
        self.train_data = train_data
        self.target_variable = target_variable
    
    def run(self):
        """Run binning in background thread"""
        start_time = time.time()
        
        try:
            print("\n" + "="*70)
            print("BINNING PROCESS STARTED")
            print("="*70)
            
            self.progress.emit(5, "Initializing binning engine...")
            print(f"[INIT] Creating binning engine for {len(self.train_data)} rows")
            
            # Create binning engine wrapper
            engine = BinningEngineWrapper(self.train_data, self.target_variable)
            
            # Ensure n_threshold is set
            if engine.binner.n_threshold is None:
                from math import ceil
                engine.binner.n_threshold = max(ceil(len(self.train_data) / 20), 5)
            
            print(f"[INIT] n_threshold set to: {engine.binner.n_threshold}")
            
            self.progress.emit(10, "Analyzing variables...")
            print(f"[ANALYSIS] Analyzing variable types...")
            
            # Get feature list
            features = [col for col in self.train_data.columns if col != self.target_variable]
            
            #Print the number of numeric and categorical variables
            print(f"[ANALYSIS] Found {len(engine.numeric_cols)} numeric variables")
            print(f"[ANALYSIS] Found {len(engine.categoric_cols)} categorical variables")
            
            all_vars = engine.numeric_cols + engine.categoric_cols
            total_vars = len(all_vars)
            
            if total_vars == 0:
                self.error.emit("No valid variables to bin!")
                return
            
            self.progress.emit(15, "Starting binning process...")
            print(f"\n[BINNING] Starting binning for {total_vars} variables...")
            print("-" * 70)
            
            # Bin variables sequentially
            binned_count = 0
            successful_bins = 0
            failed_bins = []
            
            # Bin numeric variables
            print(f"\n[NUMERIC] Processing {len(engine.numeric_cols)} numeric variables:")
            for i, feature in enumerate(engine.numeric_cols, 1):
                try:
                    progress_pct = 15 + int((binned_count / total_vars) * 75)
                    self.progress.emit(progress_pct, f"Binning {feature}... ({binned_count + 1}/{total_vars})")
                    
                    print(f"  [{i}/{len(engine.numeric_cols)}] Binning '{feature}'...", end=" ")
                    
                    var_start = time.time()
                    result = engine.binner.woe_numeric_binning(feature)
                    var_time = time.time() - var_start
                    
                    if result is not None and len(result) > 0:
                        engine.binned_results[feature] = result
                        engine.binner.binned_results[feature] = result
                        n_bins = len(result)
                        print(f"✓ ({n_bins} bins, {var_time:.2f}s)")
                        successful_bins += 1
                    else:
                        print(f"✗ (empty result)")
                        failed_bins.append(feature)
                    
                    binned_count += 1
                    
                except Exception as e:
                    print(f"✗ ERROR: {str(e)}")
                    failed_bins.append(feature)
                    binned_count += 1
                    continue
            
            # Bin categorical variables
            if len(engine.categoric_cols) > 0:
                print(f"\n[CATEGORICAL] Processing {len(engine.categoric_cols)} categoric variables:")
                for i, feature in enumerate(engine.categoric_cols, 1):
                    try:
                        progress_pct = 15 + int((binned_count / total_vars) * 75)
                        self.progress.emit(progress_pct, f"Binning {feature}... ({binned_count + 1}/{total_vars})")
                        
                        print(f"  [{i}/{len(engine.categoric_cols)}] Binning '{feature}'...", end=" ")
                        
                        var_start = time.time()
                        result = engine.binner.woe_categoric_binning(feature)
                        var_time = time.time() - var_start
                        
                        if result is not None and len(result) > 0:
                            engine.binned_results[feature] = result
                            engine.binner.binned_results[feature] = result
                            n_bins = len(result)
                            print(f"✓ ({n_bins} categories, {var_time:.2f}s)")
                            successful_bins += 1
                        else:
                            print(f"✗ (empty result)")
                            failed_bins.append(feature)
                        
                        binned_count += 1
                        
                    except Exception as e:
                        print(f"✗ ERROR: {str(e)}")
                        failed_bins.append(feature)
                        binned_count += 1
                        continue
            
            print("-" * 70)
            
            self.progress.emit(92, "Calculating statistics...")
            print(f"\n[METRICS] Calculating IV, AUC, and Gini for {successful_bins} variables...")
            
            # Calculate metrics
            engine.calculate_all_metrics()
            
            self.progress.emit(98, "Finalizing...")
            
            total_time = time.time() - start_time
            
            # Print summary
            print("\n" + "="*70)
            print("BINNING SUMMARY")
            print("="*70)
            print(f"Total variables processed: {total_vars}")
            print(f"Successfully binned: {successful_bins}")
            print(f"Failed: {len(failed_bins)}")
            if failed_bins:
                print(f"Failed variables: {', '.join(failed_bins)}")
            print(f"Total time: {total_time:.2f}s")
            print(f"Average time per variable: {total_time/total_vars:.2f}s")
            print("="*70 + "\n")
            
            time.sleep(0.2)
            
            self.progress.emit(100, "Complete!")
            
            # Return the engine with all results
            self.finished.emit(engine)
            
        except Exception as e:
            print("\n" + "="*70)
            print("BINNING ERROR")
            print("="*70)
            print(traceback.format_exc())
            print("="*70 + "\n")
            
            error_msg = f"Binning error:\n\n{str(e)}\n\nCheck console for details."
            self.error.emit(error_msg)


class BinningController(QMainWindow):
    def __init__(self, main_window):
        super().__init__()
        
        # Reference to main window
        self.main_window = main_window
        
        # Setup UI
        self.ui = Ui_Binning()
        self.ui.setupUi(self)
        
        # Binning data
        self.binning_engine = None
        self.current_variable = None
        self.current_canvas = None
        
        # Store original binning results for reset and comparison
        self.original_binned_results = {}
        self.original_metrics = {}
        
        # Connect signals
        self.ui.proceed_to_scorecard_btn.clicked.connect(self.proceed_to_scorecard)
        self.ui.back_to_var_selection_btn.clicked.connect(self.back_to_variable_selection)
        self.ui.variables_table.itemSelectionChanged.connect(self.on_variable_selection_changed)
        self.ui.tab_widget.currentChanged.connect(self.on_tab_changed)
        self.ui.merge_bins_btn.clicked.connect(self.merge_bins)
        self.ui.split_bin_btn.clicked.connect(self.split_bin)
        self.ui.save_binning_btn.clicked.connect(self.save_binning)
        self.ui.reset_binning_btn.clicked.connect(self.reset_binning)
        self.ui.back_to_vars_btn.clicked.connect(self.back_to_variables_list)
        self.ui.prev_var_btn.clicked.connect(self.previous_variable)
        self.ui.next_var_btn.clicked.connect(self.next_variable)
    
    def start_binning(self):
        """Start the binning process with loading dialog"""
        # Create loading dialog (NON-MODAL to allow window resizing)
        self.loading_dialog = LoadingDialog(
            self,
            "Binning Variables",
            "Performing optimal binning on all variables..."
        )
        
        # Create worker thread
        self.worker = BinningWorker(
            self.main_window.filtered_train_data,
            self.main_window.target_variable
        )
        
        # Connect signals
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.binning_complete)
        self.worker.error.connect(self.binning_error)
        
        # Start worker
        self.worker.start()
        
        # Show loading dialog (non-modal)
        self.loading_dialog.show()
    
    def update_progress(self, value, message):
        """Update progress bar and message"""
        self.loading_dialog.set_progress(value)
        self.loading_dialog.set_message(message)
        self.ui.statusbar.showMessage(message)
    
    def binning_complete(self, engine):
        """Handle binning completion - NO POPUP MESSAGE"""
        # Close loading dialog
        self.loading_dialog.close()
        
        # Store engine
        self.binning_engine = engine
        
        # Store original results for reset and comparison
        self.original_binned_results = copy.deepcopy(engine.binned_results)
        self.original_metrics = copy.deepcopy(engine.variable_metrics)
        
        # Populate variables table
        self.populate_variables_table()
        
        # Mark binning as completed in main window
        self.main_window.binning_completed = True  # ← NEW
        
        # Just update status bar - NO POPUP
        num_vars = len(self.binning_engine.binned_results)
        self.ui.statusbar.showMessage(f"Binning complete: {num_vars} variables processed. Select a variable to view details.")
        
        print(f"[UI] Variables table populated with {num_vars} variables")
    
    def binning_error(self, error_message):
        """Handle binning error"""
        self.loading_dialog.close()
        
        QMessageBox.critical(
            self,
            "Binning Error",
            error_message
        )
        
        self.ui.statusbar.showMessage("Binning failed")
    
    
    def populate_variables_table(self):
        """Populate the variables table with metrics (sorted by IV) - NO COLOR CODING"""
        # Get sorted variables (by IV descending)
        sorted_vars = self.binning_engine.get_sorted_variables()
        
        # Set row count
        self.ui.variables_table.setRowCount(len(sorted_vars))
        
        # Populate table
        for row, (var_name, metrics) in enumerate(sorted_vars):
            # Variable name
            item = QTableWidgetItem(var_name)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.ui.variables_table.setItem(row, 0, item)
            
            # Type
            item = QTableWidgetItem(metrics['type'].capitalize())
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.ui.variables_table.setItem(row, 1, item)
            
            # IV - NO COLOR CODING
            item = QTableWidgetItem(f"{metrics['IV']:.4f}")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.ui.variables_table.setItem(row, 2, item)
            
            # AUC
            item = QTableWidgetItem(f"{metrics['AUC']:.4f}")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.ui.variables_table.setItem(row, 3, item)
            
            # Gini
            item = QTableWidgetItem(f"{metrics['Gini']:.4f}")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.ui.variables_table.setItem(row, 4, item)
            
            # Number of bins
            n_bins = len(self.binning_engine.binned_results[var_name])
            item = QTableWidgetItem(str(n_bins))
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.ui.variables_table.setItem(row, 5, item)
        
        # Resize columns
        self.ui.variables_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in range(1, 6):
            self.ui.variables_table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
    
    def on_variable_selection_changed(self):
        """Handle when user selects a different variable in the table"""
        pass
    
    def on_tab_changed(self, index):
        """Handle tab change - auto-load selected variable when switching to Groupings"""
        if index == 1:  # Groupings tab
            # Get currently selected variable
            selected_rows = self.ui.variables_table.selectedItems()
            if selected_rows:
                row = self.ui.variables_table.currentRow()
                var_name = self.ui.variables_table.item(row, 0).text()
                # Load the variable
                self.load_variable_grouping(var_name)
            else:
                QMessageBox.warning(
                    self,
                    "No Variable Selected",
                    "Please select a variable from the Variables tab first."
                )
                # Switch back to Variables tab
                self.ui.tab_widget.setCurrentIndex(0)
    
    def load_variable_grouping(self, var_name):
        """Load variable data into grouping tab"""
        self.current_variable = var_name
        
        # Get metrics
        var_type = self.binning_engine.variable_metrics[var_name]['type']
        
        # Update label
        self.ui.current_variable_label.setText(f"Selected Variable: {var_name}")
        
        # Update Variable Statistics Panel
        self.update_statistics_panel()
        
        # Create WOE plot
        self.update_woe_plot()
        
        # Populate binning table
        self.populate_binning_table()
        
        # Enable/disable split button based on type
        is_numeric = var_type == 'numeric'
        self.ui.split_bin_btn.setEnabled(is_numeric)
        if not is_numeric:
            self.ui.split_bin_btn.setToolTip("Categorical variables cannot be split")
        else:
            self.ui.split_bin_btn.setToolTip("Split selected bin at a specific value")
        
        # Update navigation buttons
        self.update_navigation_buttons()
        
        self.ui.statusbar.showMessage(f"Viewing: {var_name}")
    
    def update_statistics_panel(self):
        """Update the Variable Statistics panel - Original stays constant, New updates"""
        if not self.current_variable:
            return
        
        # Get ORIGINAL metrics (never changes unless saved)
        original_gini = self.original_metrics[self.current_variable]['Gini']
        original_iv = self.original_metrics[self.current_variable]['IV']
        original_auc = self.original_metrics[self.current_variable]['AUC']
        
        # Get CURRENT (new) metrics from current binned results
        current_binned = self.binning_engine.binned_results[self.current_variable]
        new_iv = current_binned['IV'].iloc[0] if 'IV' in current_binned.columns else original_iv
        new_gini = current_binned['GINI'].iloc[0] if 'GINI' in current_binned.columns else original_gini
        new_auc = current_binned['AUC'].iloc[0] if 'AUC' in current_binned.columns else original_auc
        
        # Store references on first call
        if not hasattr(self, '_stat_labels_initialized'):
            self._original_gini_value = self.ui.original_gini_label.findChildren(QtWidgets.QLabel)[1]
            self._new_gini_value = self.ui.new_gini_label.findChildren(QtWidgets.QLabel)[1]
            self._original_iv_value = self.ui.original_iv_label.findChildren(QtWidgets.QLabel)[1]
            self._new_iv_value = self.ui.new_iv_label.findChildren(QtWidgets.QLabel)[1]
            self._auc_value = self.ui.auc_label.findChildren(QtWidgets.QLabel)[1]
            self._stat_labels_initialized = True
        
        # Update values - ORIGINAL stays same, NEW changes
        self._original_gini_value.setText(f"{original_gini:.4f}")
        self._new_gini_value.setText(f"{new_gini:.4f}")
        self._original_iv_value.setText(f"{original_iv:.4f}")
        self._new_iv_value.setText(f"{new_iv:.4f}")
        self._auc_value.setText(f"{new_auc:.4f}")
    
    def update_woe_plot(self):
        """Update WOE plot for current variable"""
        if not self.current_variable:
            return
        
        # Clear previous plot
        if self.current_canvas:
            self.ui.plot_layout.removeWidget(self.current_canvas)
            self.current_canvas.deleteLater()
            self.current_canvas = None
        
        # Get binning data - already sorted from binning.py
        binned_data = self.binning_engine.binned_results[self.current_variable]
        
        # Create plot (data is already in correct display order)
        fig = WOEPlotter.create_woe_plot(binned_data, self.current_variable)
        
        # Add to layout
        self.current_canvas = FigureCanvas(fig)
        self.ui.plot_layout.addWidget(self.current_canvas)
    
    def populate_binning_table(self):
        """Populate binning details table"""
        # Get binning data - already sorted from binning.py
        binned_data = self.binning_engine.binned_results[self.current_variable]
        
        # Set up table
        self.ui.binning_table.setRowCount(len(binned_data))
        self.ui.binning_table.setColumnCount(9)
        self.ui.binning_table.setHorizontalHeaderLabels([
            "Bin", "Total", "Target", "Non-Target", "Target Rate", "WOE", "IV", "% Total", "Target %"
        ])
        
        # Populate rows
        for row in range(len(binned_data)):
            data = binned_data.iloc[row]
            
            # Bin (Value column)
            bin_label = str(data['Value']) if 'Value' in data else str(row)
            item = QTableWidgetItem(bin_label)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.ui.binning_table.setItem(row, 0, item)
            
            # Total
            item = QTableWidgetItem(f"{int(data['Total']):,}")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.ui.binning_table.setItem(row, 1, item)
            
            # Target
            item = QTableWidgetItem(f"{int(data['Target']):,}")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.ui.binning_table.setItem(row, 2, item)
            
            # Non-Target
            item = QTableWidgetItem(f"{int(data['Non Target']):,}")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.ui.binning_table.setItem(row, 3, item)
            
            # Target Rate
            item = QTableWidgetItem(f"{data['Target Rate']:.2%}")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.ui.binning_table.setItem(row, 4, item)
            
            # WOE
            item = QTableWidgetItem(f"{data['WOE']:.4f}")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.ui.binning_table.setItem(row, 5, item)
            
            # IV
            item = QTableWidgetItem(f"{data['iv']:.4f}")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.ui.binning_table.setItem(row, 6, item)
            
            # % Total
            pct_total = data['Total'] / binned_data['Total'].sum()
            item = QTableWidgetItem(f"{pct_total:.2%}")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.ui.binning_table.setItem(row, 7, item)
            
            # Target %
            item = QTableWidgetItem(f"{data['Target %']:.2%}")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.ui.binning_table.setItem(row, 8, item)
        
        # Resize columns
        self.ui.binning_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in range(1, 9):
            self.ui.binning_table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        
    def update_navigation_buttons(self):
        """Update Previous/Next button states"""
        if not self.binning_engine:
            return
        
        sorted_vars = self.binning_engine.get_sorted_variables()
        var_names = [v[0] for v in sorted_vars]
        
        try:
            current_index = var_names.index(self.current_variable)
            self.ui.prev_var_btn.setEnabled(current_index > 0)
            self.ui.next_var_btn.setEnabled(current_index < len(var_names) - 1)
        except ValueError:
            self.ui.prev_var_btn.setEnabled(False)
            self.ui.next_var_btn.setEnabled(False)
    
    def previous_variable(self):
        """Navigate to previous variable"""
        sorted_vars = self.binning_engine.get_sorted_variables()
        var_names = [v[0] for v in sorted_vars]
        
        try:
            current_index = var_names.index(self.current_variable)
            if current_index > 0:
                self.load_variable_grouping(var_names[current_index - 1])
        except ValueError:
            pass
    
    def next_variable(self):
        """Navigate to next variable"""
        sorted_vars = self.binning_engine.get_sorted_variables()
        var_names = [v[0] for v in sorted_vars]
        
        try:
            current_index = var_names.index(self.current_variable)
            if current_index < len(var_names) - 1:
                self.load_variable_grouping(var_names[current_index + 1])
        except ValueError:
            pass
    
    def merge_bins(self):
        """Merge selected bins"""
        if not self.current_variable:
            return
        
        # Get selected rows from the table
        selected_rows = sorted(set(item.row() for item in self.ui.binning_table.selectedItems()))
        
        if len(selected_rows) < 2:
            self.ui.statusbar.showMessage("Please select at least 2 consecutive bins to merge (Ctrl+Click)")
            return
        
        # Check if consecutive
        for i in range(len(selected_rows) - 1):
            if selected_rows[i + 1] - selected_rows[i] != 1:
                self.ui.statusbar.showMessage("Selected bins must be consecutive")
                return
        
        try:
            # Try merging the bins
            print(f"[MERGE] Merging bins at indices: {selected_rows}")
            
            self.binning_engine.merge_bins(self.current_variable, selected_rows)
            
            # Refresh display
            self.update_statistics_panel()
            self.update_woe_plot()
            self.populate_binning_table()
            self.populate_variables_table()
            
            self.ui.statusbar.showMessage(f"Merged {len(selected_rows)} bins for {self.current_variable}")
            
            print(f"[MERGE] Merge complete. New bin count: {len(self.binning_engine.binned_results[self.current_variable])}")
            
        except Exception as e:
            self.ui.statusbar.showMessage(f"Error merging bins: {str(e)}")
            print(f"[MERGE ERROR] {str(e)}")
            print(traceback.format_exc())
    
    def split_bin(self):
        """Split a numeric bin"""
        if not self.current_variable:
            return
        
        var_type = self.binning_engine.variable_metrics[self.current_variable]['type']
        if var_type != 'numeric':
            self.ui.statusbar.showMessage("Only numeric variables can be split")
            return
        
        # Get selected row
        selected_rows = self.ui.binning_table.selectedItems()
        if not selected_rows:
            self.ui.statusbar.showMessage("Please select a bin to split")
            return
        
        bin_index = self.ui.binning_table.currentRow()
        
        # Get binning data
        binned_data = self.binning_engine.binned_results[self.current_variable]
        
        # Get bin range
        bin_row = binned_data.iloc[bin_index]
        start = bin_row['interval_start_include']
        end = bin_row['interval_end_exclude']
        
        # Sort values to get actual logical range (handles reversed/backwards intervals)
        values = [start, end]
        values.sort()  # -inf will be first, +inf will be last
        lower, upper = values[0], values[1]
        
        # Handle inf values for display
        if np.isinf(lower) and lower < 0:
            lower_str = "-∞"
            lower_val = float('-inf')
        else:
            lower_str = f"{lower:.4f}"
            lower_val = float(lower)
        
        if np.isinf(upper) and upper > 0:
            upper_str = "+∞"
            upper_val = float('inf')
        else:
            upper_str = f"{upper:.4f}"
            upper_val = float(upper)
        
        # Ask for split value
        split_value, ok = QInputDialog.getDouble(
            self,
            "Split Bin",
            f"Split bin: [{lower_str}, {upper_str})\n\n"
            f"Enter split point value:\n"
            f"(Value must be between {lower_str} and {upper_str})",
            (lower_val + upper_val) / 2 if not np.isinf(lower_val) and not np.isinf(upper_val) else 0.0,
            decimals=4
        )
        
        if ok:
            # Validate split value is within range
            if split_value <= lower_val or split_value >= upper_val:
                self.ui.statusbar.showMessage(f"Split value must be between {lower_str} and {upper_str}")
                return
            
            try:
                # Call split_bins
                self.binning_engine.split_bins(self.current_variable, split_value)
                
                # Refresh display
                self.update_statistics_panel()
                self.update_woe_plot()
                self.populate_binning_table()
                self.populate_variables_table()
                
                self.ui.statusbar.showMessage(f"Split bin at {split_value:.4f}")
                
            except Exception as e:
                self.ui.statusbar.showMessage(f"Error splitting bin: {str(e)}")
                print(f"[SPLIT ERROR] {traceback.format_exc()}")
    
    def reset_binning(self):
        """Reset binning to original state"""
        if not self.current_variable:
            return
        
        # Restore original binning
        self.binning_engine.binned_results[self.current_variable] = copy.deepcopy(
            self.original_binned_results[self.current_variable]
        )
        self.binning_engine.binner.binned_results[self.current_variable] = copy.deepcopy(
            self.original_binned_results[self.current_variable]
        )
        
        # Recalculate metrics
        self.binning_engine.calculate_all_metrics()
        
        # Refresh display
        self.update_statistics_panel()
        self.update_woe_plot()
        self.populate_binning_table()
        self.populate_variables_table()
        
        self.ui.statusbar.showMessage(f"Reset {self.current_variable} to original binning")
    
    def save_binning(self):
        """Save current binning - resets Original to match New"""
        try:
            # Save binnings
            saved_binnings = self.binning_engine.save_binnings()
            
            # Update ORIGINAL to match current (NEW becomes ORIGINAL)
            self.original_binned_results = copy.deepcopy(self.binning_engine.binned_results)
            self.original_metrics = copy.deepcopy(self.binning_engine.variable_metrics)
            
            # Store in main window
            self.main_window.binned_results = saved_binnings
            self.main_window.binning_metrics = self.binning_engine.variable_metrics
            
            # Refresh statistics panel to show Original = New
            self.update_statistics_panel()
            
            self.ui.statusbar.showMessage(f"Binning saved! Original metrics updated to current values.")
            
        except Exception as e:
            self.ui.statusbar.showMessage(f"Error saving: {str(e)}")
    
    def back_to_variables_list(self):
        """Go back to variables list tab"""
        self.ui.tab_widget.setCurrentIndex(0)
    
    def proceed_to_scorecard(self):
        """Proceed to scorecard development page"""
        # Check if binning is complete
        if self.binning_engine is None or len(self.binning_engine.binned_results) == 0:
            QMessageBox.warning(
                self,
                "No Binning Results",
                "Please complete the binning process before proceeding to scorecard."
            )
            return
        
        # Save binning results to main window (NO POPUP - REMOVED)
        self.main_window.binned_results = self.binning_engine.save_binnings()
        self.main_window.binning_metrics = self.binning_engine.variable_metrics
        self.main_window.binning_engine = self.binning_engine
        
        print(f"[BINNING] Saved {len(self.main_window.binned_results)} binned variables")
        print(f"[BINNING] Binned variables: {list(self.main_window.binned_results.keys())}")
        
        # Navigate to scorecard page (NO confirmation dialog)
        self.main_window.show_scorecard_page()
    
    def back_to_variable_selection(self):
        """Navigate back to variable selection page"""
        # Ask user if they want to re-run binning when they come back
        reply = QMessageBox.question(
            self,
            "Return to Variable Selection",
            "Do you want to re-run binning when you return?\n\n"
            "• Yes: Fresh binning with new variable selection\n"
            "• No: Keep current binning results",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Reset binning flag so it re-runs
            self.main_window.binning_completed = False
        
        self.main_window.show_variable_selection_page()
    
    def go_back(self):
        """Navigate back to variable selection"""
        self.main_window.show_variable_selection_page()