"""
Scorecard Development Controller
Handles scorecard building, evaluation, and export
"""
from PyQt6 import QtWidgets
from PyQt6.QtWidgets import (QMainWindow, QMessageBox, QTableWidgetItem, 
                              QHeaderView, QFileDialog)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import seaborn as sns
from ui.scorecard_ui import Ui_Scorecard
from ui.loading_dialog_ui import LoadingDialog
from utils.scorecarding import CreditScoreModel
from utils.scorecard_exporter import ScorecardExporter
import traceback
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import time


class ScorecardWorker(QThread):
    """Worker thread for scorecard building"""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object, object, object, dict, object)  # model, train_scored, val_scored, metrics, X_train_woe
    error = pyqtSignal(str)
    
    def __init__(self, binning_engine, train_data, val_data, target, selected_vars, score_params):
        super().__init__()
        self.binning_engine = binning_engine
        self.train_data = train_data
        self.val_data = val_data
        self.target = target
        self.selected_vars = selected_vars
        self.score_params = score_params
    
    def run(self):
        """Build scorecard model"""
        start_time = time.time()
        
        try:
            print("\n" + "="*70)
            print("SCORECARD BUILDING STARTED")
            print("="*70)
            
            self.progress.emit(5, "Preparing data...")
            print(f"[INIT] Selected variables: {len(self.selected_vars)}")
            print(f"[INIT] Training samples: {len(self.train_data)}")
            print(f"[INIT] Validation samples: {len(self.val_data)}")
            
            # Apply WOE transformation
            self.progress.emit(15, "Applying WOE transformation...")
            print("[WOE] Transforming training data...")
            
            # IMPORTANT: Extract only the selected variables (NOT including target)
            X_train_raw = self.train_data[self.selected_vars].copy()
            X_val_raw = self.val_data[self.selected_vars].copy()
            
            print(f"[WOE] X_train_raw columns: {X_train_raw.columns.tolist()}")
            print(f"[WOE] X_train_raw shape: {X_train_raw.shape}")
            
            # Apply WOE transformation
            X_train_woe = self.binning_engine.binner.apply_bins(
                X_train_raw,
                self.binning_engine.binned_results
            )
            
            # STORE A COPY OF WOE DATA FOR CORRELATION CALCULATION
            stored_X_train_woe = X_train_woe.copy()
            
            print("[WOE] Transforming validation data...")
            X_val_woe = self.binning_engine.binner.apply_bins(
                X_val_raw,
                self.binning_engine.binned_results,
            )
            
            # Extract target variable
            y_train = self.train_data[self.target].values
            y_val = self.val_data[self.target].values
            
            print(f"[WOE] Training WOE shape: {X_train_woe.shape}")
            print(f"[WOE] Validation WOE shape: {X_val_woe.shape}")
            print(f"[WOE] y_train shape: {y_train.shape}")
            
            # Calculate population odds
            self.progress.emit(25, "Calculating population odds...")
            n_good_train = (y_train == 0).sum()
            n_bad_train = (y_train == 1).sum()
            pop_odds = n_good_train / n_bad_train if n_bad_train > 0 else 1
            
            print(f"[ODDS] Population odds: {pop_odds:.4f} ({n_good_train} good / {n_bad_train} bad)")
            
            # Update score params with population odds
            score_params = self.score_params.copy()
            score_params['population_odds'] = pop_odds
            
            # Create model
            self.progress.emit(35, "Initializing logistic regression...")
            print("[MODEL] Creating logistic regression classifier...")
            
            classifier = LogisticRegression(
                penalty='l2',
                C=0.01,
                solver='lbfgs',
                max_iter=2000,
                random_state=42,
                class_weight='balanced',
                warm_start=False
            )
            
            # Create scorecard model
            model = CreditScoreModel(
                classifier=classifier,
                score_params=score_params,
                binning_engine=self.binning_engine,
                higher_score_better=True,  # This is correct based on previous discussion
                impute_missing=True
            )
            
            # Store numeric/categoric features
            model.selected_numeric_cols = [
                var for var in self.selected_vars 
                if self.binning_engine.variable_metrics[var]['type'] == 'numeric'
            ]
            model.selected_categoric_cols = [
                var for var in self.selected_vars 
                if self.binning_engine.variable_metrics[var]['type'] == 'categoric'
            ]
            
            print(f"[MODEL] Numeric features: {model.binning_engine.numeric_cols}")
            print(f"[MODEL] Categoric features: {model.binning_engine.categoric_cols}")
            
            # Train model
            self.progress.emit(45, "Training logistic regression model...")
            print("[TRAIN] Fitting model...")
            
            model.train(X_train_woe, y_train)
            
            coefficients = model.get_coefficients()
            intercept = model.classifier.intercept_[0]
            print(f"[TRAIN] Model trained. Intercept: {intercept:.4f}")
            print(f"[TRAIN] Coefficients: {coefficients}")
            
            # Build scorecard table
            self.progress.emit(55, "Building scorecard table...")
            print("[SCORECARD] Constructing scorecard table...")
            
            model.build_scorecard_table(X_train_woe)
            
            # Calculate points
            self.progress.emit(65, "Calculating scorecard points...")
            print("[POINTS] Scaling points...")
            
            model.finalize_points()
            
            print(f"[POINTS] Scorecard table shape: {model.scorecard_table.shape}")
            
            # Score datasets
            self.progress.emit(75, "Scoring training data...")
            print("[SCORE] Scoring training data...")
            
            train_scored = model.assign_points_to_data(
                self.train_data,
                self.selected_vars,
                target_column=self.target
            )
            
            self.progress.emit(85, "Scoring validation data...")
            print("[SCORE] Scoring validation data...")
            
            val_scored = model.assign_points_to_data(
                self.val_data,
                self.selected_vars,
                target_column=self.target
            )
            
            print(f"[SCORE] Training scores: min={train_scored['score'].min():.0f}, max={train_scored['score'].max():.0f}")
            print(f"[SCORE] Validation scores: min={val_scored['score'].min():.0f}, max={val_scored['score'].max():.0f}")
            
            # Calculate metrics
            self.progress.emit(95, "Calculating performance metrics...")
            print("[METRICS] Calculating KS, AUC, Gini...")
            
            # Training metrics
            from sklearn.metrics import roc_auc_score, roc_curve
            
            train_auc = roc_auc_score(train_scored['target'], -train_scored['score'])
            train_gini = 2 * train_auc - 1
            train_ks, train_ks_score, _ = model.calculate_ks_statistic(train_scored)
            
            # Validation metrics
            val_auc = roc_auc_score(val_scored['target'], -val_scored['score'])
            val_gini = 2 * val_auc - 1
            val_ks, val_ks_score, _ = model.calculate_ks_statistic(val_scored)
            
            metrics = {
                'train': {
                    'count': len(train_scored),
                    'ks': train_ks,
                    'ks_score': train_ks_score,
                    'auc': train_auc,
                    'gini': train_gini
                },
                'val': {
                    'count': len(val_scored),
                    'ks': val_ks,
                    'ks_score': val_ks_score,
                    'auc': val_auc,
                    'gini': val_gini
                },
                'train_auc': train_auc,  # Add these for compatibility
                'val_auc': val_auc,
                'train_ks': train_ks,
                'val_ks': val_ks
            }
            
            total_time = time.time() - start_time
            
            print("\n" + "="*70)
            print("SCORECARD SUMMARY")
            print("="*70)
            print(f"Training Set:")
            print(f"  Count: {metrics['train']['count']}")
            print(f"  KS: {metrics['train']['ks']:.4f} at score {metrics['train']['ks_score']:.0f}")
            print(f"  AUC: {metrics['train']['auc']:.4f}")
            print(f"  Gini: {metrics['train']['gini']:.4f}")
            print(f"\nValidation Set:")
            print(f"  Count: {metrics['val']['count']}")
            print(f"  KS: {metrics['val']['ks']:.4f} at score {metrics['val']['ks_score']:.0f}")
            print(f"  AUC: {metrics['val']['auc']:.4f}")
            print(f"  Gini: {metrics['val']['gini']:.4f}")
            print(f"\nTotal time: {total_time:.2f}s")
            print("="*70 + "\n")
            
            self.progress.emit(100, "Complete!")
            
            # EMIT THE STORED WOE DATA AS WELL
            self.finished.emit(model, train_scored, val_scored, metrics, stored_X_train_woe)
            
        except Exception as e:
            print("\n" + "="*70)
            print("SCORECARD ERROR")
            print("="*70)
            print(traceback.format_exc())
            print("="*70 + "\n")
            
            error_msg = f"Scorecard building error:\n\n{str(e)}\n\nCheck console for details."
            self.error.emit(error_msg)


class ScorecardController(QMainWindow):
    def __init__(self, main_window):
        super().__init__()
        
        self.main_window = main_window
        
        # Setup UI
        self.ui = Ui_Scorecard()
        self.ui.setupUi(self)
        
        # Data
        self.model = None
        self.train_scored = None
        self.val_scored = None
        self.metrics = None
        self.X_train_woe = None  # Store WOE data for correlation
        
        # Connect signals
        self.ui.back_to_binning_btn.clicked.connect(self.back_to_binning)
        self.ui.add_var_btn.clicked.connect(self.add_variables)
        self.ui.add_all_vars_btn.clicked.connect(self.add_all_variables)
        self.ui.remove_var_btn.clicked.connect(self.remove_variables)
        self.ui.remove_all_vars_btn.clicked.connect(self.remove_all_variables)
        self.ui.build_scorecard_btn.clicked.connect(self.build_scorecard)
        self.ui.export_excel_btn.clicked.connect(self.export_to_excel)
        self.ui.export_pmml_btn.clicked.connect(self.export_to_pmml)
        
        # Disable export buttons initially
        self.ui.export_excel_btn.setEnabled(False)
        self.ui.export_pmml_btn.setEnabled(False)
    
    def populate_available_variables(self):
        """Populate available variables from binning results"""
        self.ui.available_vars_list.clear()
        
        if not self.main_window.binned_results:
            return
        
        # Get variables sorted by IV
        sorted_vars = sorted(
            self.main_window.binning_metrics.items(),
            key=lambda x: x[1]['IV'],
            reverse=True
        )
        
        for var_name, metrics in sorted_vars:
            item_text = f"{var_name} (IV: {metrics['IV']:.4f})"
            self.ui.available_vars_list.addItem(item_text)
        
        # Calculate and display population odds
        self.calculate_population_odds()
        
        self.ui.statusbar.showMessage(f"Loaded {len(sorted_vars)} variables from binning results")
    
    def calculate_population_odds(self):
        """Calculate and display population odds from training data"""
        if self.main_window.filtered_train_data is None:
            return
        
        target = self.main_window.target_variable
        n_good = (self.main_window.filtered_train_data[target] == 0).sum()
        n_bad = (self.main_window.filtered_train_data[target] == 1).sum()
        
        if n_bad > 0:
            pop_odds = n_good / n_bad
            self.ui.pop_odds_label.setText(f"{pop_odds:.2f} (Good/Bad = {n_good}/{n_bad})")
        else:
            self.ui.pop_odds_label.setText("Cannot calculate (no bad samples)")
    
    def add_variables(self):
        """Add selected variables to scorecard"""
        selected_items = self.ui.available_vars_list.selectedItems()
        for item in selected_items:
            var_name = item.text().split(" (IV:")[0]
            # Check if already in selected list
            found = False
            for i in range(self.ui.selected_vars_list.count()):
                if self.ui.selected_vars_list.item(i).text().startswith(var_name):
                    found = True
                    break
            if not found:
                self.ui.selected_vars_list.addItem(item.text())
    
    def add_all_variables(self):
        """Add all variables to scorecard"""
        for i in range(self.ui.available_vars_list.count()):
            item = self.ui.available_vars_list.item(i)
            var_name = item.text().split(" (IV:")[0]
            # Check if already in selected list
            found = False
            for j in range(self.ui.selected_vars_list.count()):
                if self.ui.selected_vars_list.item(j).text().startswith(var_name):
                    found = True
                    break
            if not found:
                self.ui.selected_vars_list.addItem(item.text())
    
    def remove_variables(self):
        """Remove selected variables from scorecard"""
        selected_items = self.ui.selected_vars_list.selectedItems()
        for item in selected_items:
            self.ui.selected_vars_list.takeItem(self.ui.selected_vars_list.row(item))
    
    def remove_all_variables(self):
        """Remove all variables from scorecard"""
        self.ui.selected_vars_list.clear()
    
    def get_selected_variable_names(self):
        """Get list of selected variable names"""
        vars_list = []
        for i in range(self.ui.selected_vars_list.count()):
            item_text = self.ui.selected_vars_list.item(i).text()
            var_name = item_text.split(" (IV:")[0]
            vars_list.append(var_name)
        return vars_list
    
    def build_scorecard(self):
        """Build scorecard model"""
        # Validate selections
        selected_vars = self.get_selected_variable_names()
        
        if len(selected_vars) == 0:
            QMessageBox.warning(self, "No Variables", "Please select at least one variable for the scorecard.")
            return
        
        # Check for multicollinearity issues
        has_issues, warning_msg = self.check_multicollinearity(selected_vars)
    
        if has_issues:
            reply = QMessageBox.warning(
                self,
                "Potential Issues Detected",
                f"{warning_msg}\n\n"
                "These issues may cause:\n"
                "• Unstable coefficients\n"
                "• Poor model performance\n"
                "• Singular matrix errors\n\n"
                "Continue anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return
        
        # Get score parameters
        score_params = {
            'base_score': self.ui.base_score_input.value(),
            'pdo': self.ui.pdo_input.value(),
            'base_odds': self.ui.base_odds_input.value()
        }
        
        print(f"\n[UI] Building scorecard with {len(selected_vars)} variables")
        print(f"[UI] Score parameters: {score_params}")
        
        # Create loading dialog
        self.loading_dialog = LoadingDialog(
            self,
            "Building Scorecard",
            "Training logistic regression model and calculating points..."
        )
        
        # Create worker
        self.worker = ScorecardWorker(
            binning_engine=self.main_window.binning_engine,
            train_data=self.main_window.filtered_train_data,
            val_data=self.main_window.filtered_validation_data,
            target=self.main_window.target_variable,
            selected_vars=selected_vars,
            score_params=score_params
        )
        
        # Connect signals
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.scorecard_complete)
        self.worker.error.connect(self.scorecard_error)
        
        # Start worker
        self.worker.start()
        
        # Show loading dialog
        self.loading_dialog.show()
    
    def update_progress(self, value, message):
        """Update progress"""
        self.loading_dialog.set_progress(value)
        self.loading_dialog.set_message(message)
        self.ui.statusbar.showMessage(message)
    
    def scorecard_complete(self, model, train_scored, val_scored, metrics, X_train_woe):
        """Handle scorecard completion"""
        self.loading_dialog.close()
        
        # Store results INCLUDING WOE DATA
        self.model = model
        self.train_scored = train_scored
        self.val_scored = val_scored
        self.metrics = metrics
        self.X_train_woe = X_train_woe  # STORE WOE DATA
        
        print(f"[DEBUG] Stored X_train_woe shape: {X_train_woe.shape}")
        print(f"[DEBUG] Stored X_train_woe columns: {list(X_train_woe.columns)}")
        
        # Populate results
        try:
            self.populate_summary_table()
            self.populate_scorecard_table()
            self.populate_coefficient_table()
            self.populate_correlation_table()
            self.create_diagnostic_plots()
            
            # Enable export buttons
            self.ui.export_excel_btn.setEnabled(True)
            self.ui.export_pmml_btn.setEnabled(True)
            
            # Switch to results tab
            self.ui.tab_widget.setCurrentIndex(1)
            
            self.ui.statusbar.showMessage("Scorecard built successfully!")
            
            QMessageBox.information(
                self,
                "Scorecard Complete",
                f"Scorecard built successfully!\n\n"
                f"Training AUC: {metrics['train_auc']:.4f}\n"
                f"Validation AUC: {metrics['val_auc']:.4f}\n"
                f"Training KS: {metrics['train_ks']:.4f}\n"
                f"Validation KS: {metrics['val_ks']:.4f}"
            )
            
            print("[UI] Scorecard results populated in interface")
            
        except Exception as e:
            print(f"[ERROR] Failed to populate results: {str(e)}")
            print(traceback.format_exc())
            QMessageBox.warning(
                self,
                "Incomplete Results",
                f"Scorecard was built but some results failed to display:\n\n{str(e)}"
            )
    
    def scorecard_error(self, error_message):
        """Handle scorecard error"""
        self.loading_dialog.close()
        
        QMessageBox.critical(self, "Scorecard Error", error_message)
        
        self.ui.statusbar.showMessage("Scorecard building failed")
    
    def populate_summary_table(self):
        """Populate summary statistics table"""
        metrics_data = [
            ("Count", self.metrics['train']['count'], self.metrics['val']['count']),
            ("KS Statistic", f"{self.metrics['train']['ks']:.4f}", f"{self.metrics['val']['ks']:.4f}"),
            ("KS Score", f"{self.metrics['train']['ks_score']:.0f}", f"{self.metrics['val']['ks_score']:.0f}"),
            ("AUC", f"{self.metrics['train']['auc']:.4f}", f"{self.metrics['val']['auc']:.4f}"),
            ("Gini", f"{self.metrics['train']['gini']:.4f}", f"{self.metrics['val']['gini']:.4f}")
        ]
        
        self.ui.summary_table.setRowCount(len(metrics_data))
        
        for row, (metric, train_val, val_val) in enumerate(metrics_data):
            # Metric name
            item = QTableWidgetItem(metric)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.ui.summary_table.setItem(row, 0, item)
            
            # Training value
            item = QTableWidgetItem(str(train_val))
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.ui.summary_table.setItem(row, 1, item)
            
            # Validation value
            item = QTableWidgetItem(str(val_val))
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.ui.summary_table.setItem(row, 2, item)
        
        # Resize columns
        self.ui.summary_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.ui.summary_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.ui.summary_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.ui.summary_table.verticalHeader().setVisible(False)
    
    def populate_scorecard_table(self):
        """Populate scorecard table"""
        scorecard_df = self.model.scorecard_table.copy()
        
        # Filter out base score row for display
        scorecard_display = scorecard_df[scorecard_df['Variable'] != 'BASE_SCORE'].copy()
        
        # Set up table
        self.ui.scorecard_table.setRowCount(len(scorecard_display))
        self.ui.scorecard_table.setColumnCount(7)
        self.ui.scorecard_table.setHorizontalHeaderLabels([
            "Variable", "Bin", "WOE", "Coefficient", "Points", "Count", "Target Rate"
        ])
        
        # Populate rows
        for row, (idx, data) in enumerate(scorecard_display.iterrows()):
            # Variable
            item = QTableWidgetItem(str(data['Variable']))
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.ui.scorecard_table.setItem(row, 0, item)
            
            # Bin value
            if isinstance(data['Value'], list):
                bin_val = str(data['Value'])
            else:
                bin_val = str(data['Value']) if pd.notna(data['Value']) else 'Missing'
            item = QTableWidgetItem(bin_val)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.ui.scorecard_table.setItem(row, 1, item)
            
            # WOE
            item = QTableWidgetItem(f"{data['WOE']:.4f}")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.ui.scorecard_table.setItem(row, 2, item)
            
            # Coefficient
            item = QTableWidgetItem(f"{data['Coefficient']:.4f}")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.ui.scorecard_table.setItem(row, 3, item)
            
            # Points
            item = QTableWidgetItem(f"{int(data['Points'])}")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            # Color code points
            points = int(data['Points'])
            if points > 0:
                item.setBackground(QColor("#D5F4E6"))  # Light green
            elif points < 0:
                item.setBackground(QColor("#FADBD8"))  # Light red
            self.ui.scorecard_table.setItem(row, 4, item)
            
            # Count
            count_val = f"{int(data['Total']):,}" if pd.notna(data['Total']) else "-"
            item = QTableWidgetItem(count_val)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.ui.scorecard_table.setItem(row, 5, item)
            
            # Target Rate
            rate_val = f"{data['Target Rate']:.2%}" if pd.notna(data['Target Rate']) else "-"
            item = QTableWidgetItem(rate_val)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.ui.scorecard_table.setItem(row, 6, item)
        
        # Resize columns
        self.ui.scorecard_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.ui.scorecard_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for col in range(2, 7):
            self.ui.scorecard_table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        self.ui.scorecard_table.verticalHeader().setVisible(False)
    
    def populate_coefficient_table(self):
        """Populate coefficient significance table"""
        # USE STORED WOE DATA
        y_train = self.main_window.filtered_train_data[self.main_window.target_variable].values
        
        # Calculate p-values using stored WOE data
        significance = self.model.calculate_significance(self.X_train_woe, y_train)
        
        self.ui.coef_table.setRowCount(len(significance))
        
        for row, (feature, p_value) in enumerate(significance):
            # Variable
            item = QTableWidgetItem(feature)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.ui.coef_table.setItem(row, 0, item)
            
            # Coefficient
            coef_idx = list(self.X_train_woe.columns).index(feature)
            coef = self.model.classifier.coef_[0][coef_idx]
            item = QTableWidgetItem(f"{coef:.6f}")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.ui.coef_table.setItem(row, 1, item)
            
            # P-value
            item = QTableWidgetItem(f"{p_value:.6f}")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.ui.coef_table.setItem(row, 2, item)
            
            # Significant
            is_sig = p_value < 0.05
            item = QTableWidgetItem("✓ Yes" if is_sig else "✗ No")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if is_sig:
                item.setBackground(QColor("#D5F4E6"))
                item.setForeground(QColor("#27AE60"))
            else:
                item.setBackground(QColor("#FADBD8"))
                item.setForeground(QColor("#E74C3C"))
            self.ui.coef_table.setItem(row, 3, item)
        
        # Resize columns
        for col in range(4):
            self.ui.coef_table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeMode.Stretch)
        self.ui.coef_table.verticalHeader().setVisible(False)
    
    def populate_correlation_table(self):
        """Populate correlation matrix table"""
        print("[UI] Populating correlation matrix...")
        
        try:
            # USE STORED WOE DATA
            if self.X_train_woe is None:
                print("[WARNING] No WOE data available. Skipping correlation matrix.")
                return
            
            print(f"[DEBUG] Using stored X_train_woe shape: {self.X_train_woe.shape}")
            print(f"[DEBUG] Using stored X_train_woe columns: {list(self.X_train_woe.columns)}")
            
            # Calculate correlation
            _, corr_df = self.model.analyze_coefficient_correlation(self.X_train_woe, threshold=0.4)
            
            if corr_df.empty:
                print("[WARNING] Correlation matrix is empty")
                return
            
            # Set up table
            n_vars = len(corr_df)
            self.ui.corr_table.setRowCount(n_vars)
            self.ui.corr_table.setColumnCount(n_vars)
            
            # Set headers
            headers = corr_df.columns.tolist()
            self.ui.corr_table.setHorizontalHeaderLabels(headers)
            self.ui.corr_table.setVerticalHeaderLabels(headers)
            
            # Enable scrolling
            self.ui.corr_table.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
            self.ui.corr_table.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
            
            # Populate cells
            for i in range(n_vars):
                for j in range(n_vars):
                    corr_value = corr_df.iloc[i, j]
                    
                    item = QtWidgets.QTableWidgetItem(f"{corr_value:.3f}")
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    
                    # Color coding
                    if i == j:
                        item.setBackground(QColor("#ECF0F1"))
                        item.setForeground(QColor("#2C3E50"))
                    elif abs(corr_value) > 0.7:
                        item.setBackground(QColor("#E74C3C"))
                        item.setForeground(QColor("white"))
                        item.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
                    elif abs(corr_value) > 0.4:
                        item.setBackground(QColor("#F39C12"))
                        item.setForeground(QColor("white"))
                    else:
                        item.setBackground(QColor("white"))
                        item.setForeground(QColor("#2C3E50"))
                    
                    self.ui.corr_table.setItem(i, j, item)
            
            # Column and row sizing
            max_var_length = max(len(var) for var in headers)
            col_width = max(100, min(max_var_length * 8, 200))
            
            for i in range(n_vars):
                self.ui.corr_table.setColumnWidth(i, col_width)
                self.ui.corr_table.setRowHeight(i, 40)
            
            # Configure headers
            h_header = self.ui.corr_table.horizontalHeader()
            h_header.setDefaultSectionSize(col_width)
            h_header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
            h_header.setMinimumSectionSize(80)
            h_header.setMaximumSectionSize(300)
            h_header.setStretchLastSection(False)
            
            v_header = self.ui.corr_table.verticalHeader()
            v_header.setDefaultSectionSize(40)
            v_header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Fixed)
            v_header.setMinimumWidth(150)
            
            self.ui.corr_table.setMinimumSize(400, 300)
            
            print(f"[UI] Correlation matrix populated: {n_vars}×{n_vars}, col_width={col_width}px")
            
        except Exception as e:
            print(f"[ERROR] Failed to populate correlation table: {str(e)}")
            print(traceback.format_exc())
    
    def create_diagnostic_plots(self):
        """Create all diagnostic plots"""
        plt.style.use('default')
        sns.set_palette(["#3498DB", "#E74C3C", "#27AE60", "#F39C12"])
        
        self.create_roc_plot()
        self.create_ks_plot()
        self.create_calibration_plot()
        self.create_distribution_plot()
    
    def create_roc_plot(self):
        """Create ROC curve plot"""
        for i in reversed(range(self.ui.roc_layout.count())): 
            self.ui.roc_layout.itemAt(i).widget().setParent(None)
        
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        fig.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.15)
        
        from sklearn.metrics import roc_curve, auc
        
        fpr_train, tpr_train, _ = roc_curve(self.train_scored['target'], -self.train_scored['score'])
        auc_train = auc(fpr_train, tpr_train)
        ax.plot(fpr_train, tpr_train, color='#3498DB', linewidth=2.5, 
                label=f'Training AUC = {auc_train:.4f}')
        
        fpr_val, tpr_val, _ = roc_curve(self.val_scored['target'], -self.val_scored['score'])
        auc_val = auc(fpr_val, tpr_val)
        ax.plot(fpr_val, tpr_val, color='#E74C3C', linewidth=2.5,
                label=f'Validation AUC = {auc_val:.4f}')
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random', alpha=0.5)
        
        ax.set_xlabel('False Positive Rate', fontsize=11, fontweight='600', color='#2C3E50')
        ax.set_ylabel('True Positive Rate', fontsize=11, fontweight='600', color='#2C3E50')
        ax.set_title('ROC Curve', fontsize=13, fontweight='bold', color='#2C3E50', pad=10)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        canvas = FigureCanvas(fig)
        self.ui.roc_layout.addWidget(canvas)
    
    def create_ks_plot(self):
        """Create KS chart"""
        for i in reversed(range(self.ui.ks_layout.count())): 
            self.ui.ks_layout.itemAt(i).widget().setParent(None)
        
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        fig.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.15)
        
        ks_stat, ks_score, ks_data = self.model.calculate_ks_statistic(self.val_scored)
        
        ax.plot(ks_data['score'], ks_data['cum_good_pct'], 
                color='#27AE60', linewidth=2.5, label='Good Customers')
        ax.plot(ks_data['score'], ks_data['cum_bad_pct'], 
                color='#E74C3C', linewidth=2.5, label='Bad Customers')
        
        ks_row = ks_data[ks_data['score'] == ks_score].iloc[0]
        ax.vlines(x=ks_score, 
                ymin=ks_row['cum_bad_pct'], 
                ymax=ks_row['cum_good_pct'],
                colors='#3498DB', linewidth=3, linestyles='dashed',
                label=f'KS = {ks_stat:.3f}')
        
        ax.annotate(f'KS = {ks_stat:.1%}\nat {ks_score:.0f}',
                xy=(ks_score, (ks_row['cum_good_pct'] + ks_row['cum_bad_pct']) / 2),
                xytext=(ks_score + 20, (ks_row['cum_good_pct'] + ks_row['cum_bad_pct']) / 2),
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#F39C12', alpha=0.7),
                arrowprops=dict(arrowstyle='->', lw=2))
        
        ax.set_xlabel('Credit Score', fontsize=11, fontweight='600', color='#2C3E50')
        ax.set_ylabel('Cumulative %', fontsize=11, fontweight='600', color='#2C3E50')
        ax.set_title('KS Chart (Validation)', fontsize=13, fontweight='bold', color='#2C3E50', pad=10)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim([0, 1])
        
        canvas = FigureCanvas(fig)
        self.ui.ks_layout.addWidget(canvas)
    
    def create_calibration_plot(self):
        """Create calibration chart"""
        for i in reversed(range(self.ui.cal_layout.count())): 
            self.ui.cal_layout.itemAt(i).widget().setParent(None)
        
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        fig.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.15)
        
        pdo = self.ui.pdo_input.value()
        base_odds = self.ui.base_odds_input.value()
        base_score = self.ui.base_score_input.value()
        
        num_quantiles = 20
        
        for dataset, color, label in [(self.val_scored, '#3498DB', 'Validation'),
                                       (self.train_scored, '#27AE60', 'Training')]:
            dataset_copy = dataset.copy()
            dataset_copy['score_quantile'], _ = pd.qcut(
                dataset_copy['score'], q=num_quantiles, labels=False, retbins=True, duplicates='drop'
            )
            
            agg = dataset_copy.groupby(['score_quantile'])[['score_quantile', 'target', 'score']].agg(['mean', 'count', 'sum'])
            agg.columns = ['_'.join(col) for col in agg.columns]
            
            agg['observed_odds'] = (1 - agg['target_mean']) / agg['target_mean']
            agg['log_observed_odds'] = np.log(agg['observed_odds'].replace([np.inf, -np.inf], np.nan))
            agg['expected_odds'] = base_odds * 2 ** ((agg['score_mean'] - base_score) / pdo)
            agg['log_expected_odds'] = np.log(agg['expected_odds'])
            
            ax.scatter(agg['score_mean'], agg['log_observed_odds'], 
                      color=color, alpha=0.6, s=50, label=f'{label} Observed')
        
        ax.scatter(agg['score_mean'], agg['log_expected_odds'], 
                  color='black', marker='x', s=50, label='Expected')
        
        ax.set_xlabel('Credit Score', fontsize=11, fontweight='600', color='#2C3E50')
        ax.set_ylabel('log(Odds)', fontsize=11, fontweight='600', color='#2C3E50')
        ax.set_title('Calibration Chart', fontsize=13, fontweight='bold', color='#2C3E50', pad=10)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        canvas = FigureCanvas(fig)
        self.ui.cal_layout.addWidget(canvas)
    
    def create_distribution_plot(self):
        """Create score distribution plot"""
        for i in reversed(range(self.ui.dist_layout.count())): 
            self.ui.dist_layout.itemAt(i).widget().setParent(None)
        
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        fig.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.15)
        
        good_scores = self.val_scored[self.val_scored['target'] == 0]['score']
        bad_scores = self.val_scored[self.val_scored['target'] == 1]['score']
        
        sns.kdeplot(data=good_scores, ax=ax, fill=True, color='#27AE60', 
                   alpha=0.5, linewidth=2.5, label='Non-Default (Good)')
        sns.kdeplot(data=bad_scores, ax=ax, fill=True, color='#E74C3C', 
                   alpha=0.5, linewidth=2.5, label='Default (Bad)')
        
        ax.set_xlabel('Credit Score', fontsize=11, fontweight='600', color='#2C3E50')
        ax.set_ylabel('Density', fontsize=11, fontweight='600', color='#2C3E50')
        ax.set_title('Score Distribution (Validation)', fontsize=13, fontweight='bold', color='#2C3E50', pad=10)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        canvas = FigureCanvas(fig)
        self.ui.dist_layout.addWidget(canvas)
    
    def export_to_excel(self):
        """Export to Excel"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Excel Scorecard", "scorecard_report.xlsx", "Excel Files (*.xlsx)"
        )
        
        if not file_path:
            return
        
        try:
            print(f"[EXPORT] Exporting to Excel: {file_path}")
            
            # Get correlation using stored WOE data
            corr_df = None
            if self.X_train_woe is not None:
                _, corr_df = self.model.analyze_coefficient_correlation(self.X_train_woe, threshold=0.4)
            
            exporter = ScorecardExporter(
                self.model, self.train_scored, self.val_scored, 
                self.metrics, correlation_df=corr_df
            )
            exporter.export_to_excel(file_path)
            
            QMessageBox.information(self, "Export Complete", f"Exported to:\n{file_path}")
            self.ui.statusbar.showMessage(f"Exported to {file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Error:\n\n{str(e)}")
            print(traceback.format_exc())
    
    def export_to_pmml(self):
        """Export to PMML"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save PMML Model", "scorecard_model.pmml", "PMML Files (*.pmml)"
        )
        
        if not file_path:
            return
        
        try:
            print(f"[EXPORT] Exporting to PMML: {file_path}")
            
            selected_vars = self.get_selected_variable_names()
            
            # CREATE EXPORTER WITH WOE DATA
            exporter = ScorecardExporter(
                self.model,
                self.train_scored,
                self.val_scored,
                self.metrics,
                correlation_df=self.correlation_df if hasattr(self, 'correlation_df') else None,
                X_train_woe=self.X_train_woe  # ← ADD THIS!
            )
            
            exporter.export_to_pmml(file_path, selected_vars)
            
            QMessageBox.information(self, "Export Complete", f"Exported to:\n{file_path}")
            self.ui.statusbar.showMessage(f"Exported to {file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Error:\n\n{str(e)}")
            print(traceback.format_exc())
    
    def back_to_binning(self):
        """Navigate back to binning"""
        self.main_window.show_binning_page(force_rerun=False)
        
    def check_multicollinearity(self, selected_vars):
        """Check for multicollinearity issues"""
        warnings = []
        
        target_leak_keywords = ['status', 'outcome', 'result', 'charged', 'default', 'paid']
        leak_vars = [v for v in selected_vars if any(kw in v.lower() for kw in target_leak_keywords)]
        
        if leak_vars:
            warnings.append(f"⚠️ Possible target leakage: {', '.join(leak_vars)}")
        
        constant_vars = []
        for var in selected_vars:
            if var in self.main_window.binning_engine.binned_results:
                n_bins = len(self.main_window.binning_engine.binned_results[var])
                if n_bins <= 1:
                    constant_vars.append(var)
        
        if constant_vars:
            warnings.append(f"⚠️ Constant variables (1 bin): {', '.join(constant_vars)}")
        
        if len(selected_vars) > 20:
            warnings.append(f"⚠️ Too many variables ({len(selected_vars)}). Recommended: 5-15")
        
        return len(warnings) > 0, "\n".join(warnings)