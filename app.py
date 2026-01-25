"""
Credit Risk Workbench Application
Main application entry point
"""
import sys
import platform
from PyQt6.QtWidgets import QApplication, QMainWindow, QStackedWidget
from PyQt6.QtGui import QIcon
from controllers.home_controller import HomeController
from controllers.variable_selection_controller import VariableSelectionController
from controllers.binning_controller import BinningController
from controllers.scorecard_controller import ScorecardController 


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Credit Risk Workbench")
        self.setWindowIcon(QIcon('assets/credit_risk_icon.png'))
        self.resize(1400, 900)
        
        # Shared data storage
        self.train_data = None
        self.validation_data = None
        self.train_data_path = None
        self.validation_data_path = None
        self.target_variable = None
        self.selected_features = []
        self.filtered_train_data = None
        self.filtered_validation_data = None
        
        # Binning results
        self.binned_results = None
        self.binning_metrics = None
        self.binning_engine = None
        
        # NEW: Track binning state
        self.binning_completed = False  # ← Track if binning has been run
        
        # Scorecard results
        self.scorecard_model = None
        self.train_scored = None
        self.val_scored = None
        
        # Create stacked widget for navigation
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        
        # Create page controllers
        self.home_controller = HomeController(self)
        self.var_sel_controller = VariableSelectionController(self)
        self.binning_controller = BinningController(self)
        self.scorecard_controller = ScorecardController(self)
        
        # Add pages to stack
        self.stacked_widget.addWidget(self.home_controller)        # Index 0
        self.stacked_widget.addWidget(self.var_sel_controller)     # Index 1
        self.stacked_widget.addWidget(self.binning_controller)     # Index 2
        self.stacked_widget.addWidget(self.scorecard_controller)   # Index 3
        
        # Start on home page
        self.show_home_page()
    
    def show_home_page(self):
        """Navigate to home page"""
        self.stacked_widget.setCurrentIndex(0)
        self.setWindowTitle("Credit Risk Workbench - Home")
    
    def show_variable_selection_page(self):
        """Navigate to variable selection page"""
        self.var_sel_controller.populate_page()
        self.stacked_widget.setCurrentIndex(1)
        self.setWindowTitle("Credit Risk Workbench - Variable Selection")
    
    def show_binning_page(self, force_rerun=False):
        """
        Navigate to binning page
        
        Args:
            force_rerun: If True, always re-run binning. If False, only run if not completed.
        """
        self.stacked_widget.setCurrentIndex(2)
        self.setWindowTitle("Credit Risk Workbench - Binning")
        
        # Only run binning if:
        # 1. force_rerun is True, OR
        # 2. binning has never been completed
        if force_rerun or not self.binning_completed:
            print("[APP] Running binning algorithm...")
            self.binning_controller.start_binning()
        else:
            print("[APP] Binning already completed. Loading previous results...")
            # Binning results are already loaded in the controller
            self.binning_controller.ui.statusbar.showMessage(
                f"Loaded existing binning results: {len(self.binning_engine.binned_results)} variables"
            )
    
    def show_scorecard_page(self):
        """Navigate to scorecard development page"""
        self.stacked_widget.setCurrentIndex(3)
        self.setWindowTitle("Credit Risk Workbench - Scorecard Development")
        
        # Populate variables from binning
        self.scorecard_controller.populate_available_variables()


def main():
    """Application entry point"""
    
    if platform.system() == 'Windows':
        import ctypes
        myappid = 'mycompany.creditriskworkbench.app.1.0'
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('credit_risk_icon.png'))
    app.setApplicationName("Credit Risk Workbench")
    app.setOrganizationName("Your Organization")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
    
    
    