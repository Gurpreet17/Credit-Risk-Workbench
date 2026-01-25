# scorecard.py
# UI for scorecard development page

from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_Scorecard(object):
    def setupUi(self, Scorecard):
        Scorecard.setObjectName("Scorecard")
        Scorecard.resize(1400, 900)
        Scorecard.setMinimumSize(QtCore.QSize(1200, 800))
        
        self.centralwidget = QtWidgets.QWidget(parent=Scorecard)
        self.centralwidget.setObjectName("centralwidget")
        
        # Main layout
        main_layout = QtWidgets.QVBoxLayout(self.centralwidget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Header
        header = self.create_header()
        main_layout.addWidget(header)
        
        # Tab widget
        self.tab_widget = QtWidgets.QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: none;
                background-color: #F5F6FA;
            }
            QTabBar::tab {
                background-color: #ECF0F1;
                color: #2C3E50;
                padding: 15px 30px;
                margin-right: 2px;
                font-size: 13pt;
                font-weight: bold;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                min-width: 120px;
            }
            QTabBar::tab:selected {
                background-color: white;
                color: #3498DB;
            }
            QTabBar::tab:hover:!selected {
                background-color: #D5DBDB;
            }
        """)
        
        # Setup tab
        self.setup_tab = QtWidgets.QWidget()
        self.setup_setup_tab()
        self.tab_widget.addTab(self.setup_tab, "Model Setup")
        
        # Results tab
        self.results_tab = QtWidgets.QWidget()
        self.setup_results_tab()
        self.tab_widget.addTab(self.results_tab, "Results & Scorecard")
        
        # Diagnostics tab
        self.diagnostics_tab = QtWidgets.QWidget()
        self.setup_diagnostics_tab()
        self.tab_widget.addTab(self.diagnostics_tab, "Model Diagnostics")
        
        main_layout.addWidget(self.tab_widget)
        
        Scorecard.setCentralWidget(self.centralwidget)
        
        # Status bar
        self.statusbar = QtWidgets.QStatusBar(parent=Scorecard)
        self.statusbar.setStyleSheet("""
            QStatusBar {
                background-color: #ECF0F1;
                color: #7F8C8D;
                border-top: 1px solid #BDC3C7;
                font-size: 10pt;
            }
        """)
        Scorecard.setStatusBar(self.statusbar)
        
        self.retranslateUi(Scorecard)
        QtCore.QMetaObject.connectSlotsByName(Scorecard)
    
    def create_header(self):
        """Create header section"""
        header = QtWidgets.QWidget()
        header.setMinimumHeight(80)
        header.setMaximumHeight(80)
        header.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #2C3E50, stop:1 #3498DB);
            }
        """)
        
        layout = QtWidgets.QHBoxLayout(header)
        layout.setContentsMargins(40, 15, 40, 15)
        
        title = QtWidgets.QLabel("Scorecard Development")
        font = QtGui.QFont("Segoe UI", 24, QtGui.QFont.Weight.Bold)
        title.setFont(font)
        title.setStyleSheet("color: white; background: transparent;")
        layout.addWidget(title)
        
        layout.addStretch()
        
        subtitle = QtWidgets.QLabel("Build and evaluate credit risk scorecards")
        font = QtGui.QFont("Segoe UI", 11)
        subtitle.setFont(font)
        subtitle.setStyleSheet("color: #ECF0F1; background: transparent;")
        layout.addWidget(subtitle)
        
        return header
    
    def setup_setup_tab(self):
        """Setup tab for model configuration"""
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll_area.setStyleSheet("QScrollArea { background-color: #F5F6FA; border: none; }")
        
        scroll_content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(scroll_content)
        layout.setContentsMargins(40, 30, 40, 30)
        layout.setSpacing(25)
        
        # Back button
        back_layout = QtWidgets.QHBoxLayout()
        self.back_to_binning_btn = QtWidgets.QPushButton("← Back to Binning")
        self.back_to_binning_btn.setMinimumHeight(40)
        self.back_to_binning_btn.setStyleSheet("""
            QPushButton {
                background-color: #7F8C8D;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 20px;
                font-size: 11pt;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5D6D7E;
            }
        """)
        back_layout.addWidget(self.back_to_binning_btn)
        back_layout.addStretch()
        layout.addLayout(back_layout)
        
        # Variable Selection Section
        var_section = self.create_section_widget("Variable Selection", 
            "Select variables to include in the scorecard model")
        var_layout = QtWidgets.QVBoxLayout()
        var_layout.setContentsMargins(20, 15, 20, 15)
        
        # Available variables list
        list_layout = QtWidgets.QHBoxLayout()
        
        # All variables
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        QtWidgets.QLabel("Available Variables:").setParent(left_widget)
        left_layout.addWidget(QtWidgets.QLabel("Available Variables:"))
        
        self.available_vars_list = QtWidgets.QListWidget()
        self.available_vars_list.setSelectionMode(QtWidgets.QListWidget.SelectionMode.MultiSelection)
        self.available_vars_list.setMinimumHeight(200)
        self.available_vars_list.setStyleSheet("""
            QListWidget {
                border: 2px solid #3498DB;
                border-radius: 6px;
                padding: 5px;
                font-size: 11pt;
            }
            QListWidget::item:selected {
                background-color: #5DADE2;
                color: white;
            }
        """)
        left_layout.addWidget(self.available_vars_list)
        list_layout.addWidget(left_widget)
        
        # Arrow buttons
        btn_widget = QtWidgets.QWidget()
        btn_layout = QtWidgets.QVBoxLayout(btn_widget)
        btn_layout.addStretch()
        
        self.add_var_btn = QtWidgets.QPushButton("→")
        self.add_var_btn.setMinimumSize(60, 40)
        self.add_all_vars_btn = QtWidgets.QPushButton("⇉")
        self.add_all_vars_btn.setMinimumSize(60, 40)
        self.remove_var_btn = QtWidgets.QPushButton("←")
        self.remove_var_btn.setMinimumSize(60, 40)
        self.remove_all_vars_btn = QtWidgets.QPushButton("⇇")
        self.remove_all_vars_btn.setMinimumSize(60, 40)
        
        for btn in [self.add_var_btn, self.add_all_vars_btn, self.remove_var_btn, self.remove_all_vars_btn]:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #3498DB;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    font-size: 16pt;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #2980B9;
                }
            """)
        
        btn_layout.addWidget(self.add_var_btn)
        btn_layout.addWidget(self.add_all_vars_btn)
        btn_layout.addSpacing(20)
        btn_layout.addWidget(self.remove_var_btn)
        btn_layout.addWidget(self.remove_all_vars_btn)
        btn_layout.addStretch()
        list_layout.addWidget(btn_widget)
        
        # Selected variables
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        right_layout.addWidget(QtWidgets.QLabel("Selected Variables:"))
        
        self.selected_vars_list = QtWidgets.QListWidget()
        self.selected_vars_list.setSelectionMode(QtWidgets.QListWidget.SelectionMode.MultiSelection)
        self.selected_vars_list.setMinimumHeight(200)
        self.selected_vars_list.setStyleSheet("""
            QListWidget {
                border: 2px solid #27AE60;
                border-radius: 6px;
                padding: 5px;
                font-size: 11pt;
            }
            QListWidget::item:selected {
                background-color: #52BE80;
                color: white;
            }
        """)
        right_layout.addWidget(self.selected_vars_list)
        list_layout.addWidget(right_widget)
        
        var_layout.addLayout(list_layout)
        var_section.layout().addLayout(var_layout)
        layout.addWidget(var_section)
        
        # Scoring Parameters Section
        params_section = self.create_section_widget("Scoring Parameters",
            "Configure scorecard scaling parameters")
        params_layout = QtWidgets.QGridLayout()
        params_layout.setContentsMargins(20, 15, 20, 15)
        params_layout.setSpacing(15)
        
        # Base Score
        params_layout.addWidget(QtWidgets.QLabel("Base Score:"), 0, 0)
        self.base_score_input = QtWidgets.QSpinBox()
        self.base_score_input.setRange(10, 2000)
        self.base_score_input.setValue(600)
        self.base_score_input.setSingleStep(10)
        self.base_score_input.setMinimumHeight(35)
        self.base_score_input.setStyleSheet(self.get_input_style())
        params_layout.addWidget(self.base_score_input, 0, 1)
        
        # PDO
        params_layout.addWidget(QtWidgets.QLabel("Points to Double Odds (PDO):"), 1, 0)
        self.pdo_input = QtWidgets.QSpinBox()
        self.pdo_input.setRange(10, 100)
        self.pdo_input.setValue(20)
        self.pdo_input.setSingleStep(5)
        self.pdo_input.setMinimumHeight(35)
        self.pdo_input.setStyleSheet(self.get_input_style())
        params_layout.addWidget(self.pdo_input, 1, 1)
        
        # Base Odds
        params_layout.addWidget(QtWidgets.QLabel("Base Odds:"), 2, 0)
        self.base_odds_input = QtWidgets.QDoubleSpinBox()
        self.base_odds_input.setRange(1, 100)
        self.base_odds_input.setValue(50)
        self.base_odds_input.setSingleStep(1)
        self.base_odds_input.setDecimals(1)
        self.base_odds_input.setMinimumHeight(35)
        self.base_odds_input.setStyleSheet(self.get_input_style())
        params_layout.addWidget(self.base_odds_input, 2, 1)
        
        # Population Odds (auto-calculated, read-only)
        params_layout.addWidget(QtWidgets.QLabel("Population Odds (calculated):"), 3, 0)
        self.pop_odds_label = QtWidgets.QLabel("Will be calculated from training data")
        self.pop_odds_label.setStyleSheet("""
            font-size: 11pt;
            color: #7F8C8D;
            padding: 8px;
            background-color: #ECF0F1;
            border-radius: 4px;
        """)
        params_layout.addWidget(self.pop_odds_label, 3, 1)
        
        params_section.layout().addLayout(params_layout)
        layout.addWidget(params_section)
        
        # Build button
        build_layout = QtWidgets.QHBoxLayout()
        build_layout.addStretch()
        
        self.build_scorecard_btn = QtWidgets.QPushButton("Build Scorecard")
        self.build_scorecard_btn.setMinimumHeight(50)
        self.build_scorecard_btn.setMinimumWidth(200)
        self.build_scorecard_btn.setStyleSheet("""
            QPushButton {
                background-color: #27AE60;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 15px 30px;
                font-size: 14pt;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #229954;
            }
            QPushButton:pressed {
                background-color: #1E8449;
            }
            QPushButton:disabled {
                background-color: #BDC3C7;
            }
        """)
        build_layout.addWidget(self.build_scorecard_btn)
        
        layout.addLayout(build_layout)
        layout.addStretch()
        
        scroll_area.setWidget(scroll_content)
        main_layout = QtWidgets.QVBoxLayout(self.setup_tab)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll_area)
    
    def setup_results_tab(self):
        """Results tab with scorecard and metrics"""
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll_area.setStyleSheet("QScrollArea { background-color: #F5F6FA; border: none; }")
        
        scroll_content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(scroll_content)
        layout.setContentsMargins(40, 30, 40, 30)
        layout.setSpacing(25)
        
        # Summary Statistics
        summary_section = self.create_section_widget("Model Performance Summary",
            "Key performance metrics for training and validation sets")
        summary_layout = QtWidgets.QVBoxLayout()
        summary_layout.setContentsMargins(20, 15, 20, 15)
        
        self.summary_table = QtWidgets.QTableWidget()
        self.summary_table.setColumnCount(3)
        self.summary_table.setHorizontalHeaderLabels(["Metric", "Training", "Validation"])
        self.summary_table.setMinimumHeight(250)
        self.summary_table.setStyleSheet("""
            QTableWidget {
                background-color: white;
                border: 2px solid #3498DB;
                border-radius: 8px;
                font-size: 11pt;
                gridline-color: #E1E4E8;
            }
            QHeaderView::section {
                background-color: #3498DB;
                color: white;
                padding: 12px;
                border: none;
                font-weight: bold;
                font-size: 11pt;
            }
        """)
        summary_layout.addWidget(self.summary_table)
        summary_section.layout().addLayout(summary_layout)
        layout.addWidget(summary_section)
        
        # Scorecard Table
        scorecard_section = self.create_section_widget("Scorecard",
            "Variable binning and point allocation")
        scorecard_layout = QtWidgets.QVBoxLayout()
        scorecard_layout.setContentsMargins(20, 15, 20, 15)
        
        self.scorecard_table = QtWidgets.QTableWidget()
        self.scorecard_table.setMinimumHeight(400)
        self.scorecard_table.setStyleSheet("""
            QTableWidget {
                background-color: white;
                border: 2px solid #3498DB;
                border-radius: 8px;
                font-size: 10pt;
                gridline-color: #E1E4E8;
            }
            QHeaderView::section {
                background-color: #3498DB;
                color: white;
                padding: 10px;
                border: none;
                font-weight: bold;
                font-size: 10pt;
            }
        """)
        scorecard_layout.addWidget(self.scorecard_table)
        scorecard_section.layout().addLayout(scorecard_layout)
        layout.addWidget(scorecard_section)
        
        # Export buttons
        export_layout = QtWidgets.QHBoxLayout()
        export_layout.setSpacing(15)
        
        self.export_excel_btn = QtWidgets.QPushButton("📊 Export to Excel")
        self.export_excel_btn.setMinimumHeight(45)
        self.export_excel_btn.setMinimumWidth(180)
        self.export_excel_btn.setStyleSheet("""
            QPushButton {
                background-color: #27AE60;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 12pt;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #229954;
            }
            QPushButton:disabled {
                background-color: #BDC3C7;
            }
        """)
        export_layout.addWidget(self.export_excel_btn)
        
        self.export_pmml_btn = QtWidgets.QPushButton("📄 Export to PMML")
        self.export_pmml_btn.setMinimumHeight(45)
        self.export_pmml_btn.setMinimumWidth(180)
        self.export_pmml_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498DB;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 12pt;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
            QPushButton:disabled {
                background-color: #BDC3C7;
            }
        """)
        export_layout.addWidget(self.export_pmml_btn)
        
        export_layout.addStretch()
        layout.addLayout(export_layout)
        
        scroll_area.setWidget(scroll_content)
        main_layout = QtWidgets.QVBoxLayout(self.results_tab)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll_area)
    
    def setup_diagnostics_tab(self):
        """Diagnostics tab with plots and statistical tests"""
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll_area.setStyleSheet("QScrollArea { background-color: #F5F6FA; border: none; }")
        
        scroll_content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(scroll_content)
        layout.setContentsMargins(40, 30, 40, 30)
        layout.setSpacing(25)
        
        # Coefficient Significance
        coef_section = self.create_section_widget("Coefficient Significance",
            "Statistical significance of model coefficients")
        coef_layout = QtWidgets.QVBoxLayout()
        coef_layout.setContentsMargins(20, 15, 20, 15)
        
        self.coef_table = QtWidgets.QTableWidget()
        self.coef_table.setColumnCount(4)
        self.coef_table.setHorizontalHeaderLabels(["Variable", "Coefficient", "P-Value", "Significant"])
        self.coef_table.setMinimumHeight(300)
        self.coef_table.setStyleSheet("""
            QTableWidget {
                background-color: white;
                border: 2px solid #3498DB;
                border-radius: 8px;
                font-size: 11pt;
                gridline-color: #E1E4E8;
            }
            QHeaderView::section {
                background-color: #3498DB;
                color: white;
                padding: 12px;
                border: none;
                font-weight: bold;
                font-size: 11pt;
            }
        """)
        coef_layout.addWidget(self.coef_table)
        coef_section.layout().addLayout(coef_layout)
        layout.addWidget(coef_section)
        
        # Correlation Matrix
        corr_section = self.create_section_widget("Coefficient Correlation Matrix",
            "Correlation between model coefficients")
        corr_layout = QtWidgets.QVBoxLayout()
        corr_layout.setContentsMargins(20, 15, 20, 15)
        
        self.corr_table = QtWidgets.QTableWidget()
        self.corr_table.setMinimumHeight(300)
        self.corr_table.setStyleSheet("""
            QTableWidget {
                background-color: white;
                border: 2px solid #3498DB;
                border-radius: 8px;
                font-size: 10pt;
                gridline-color: #E1E4E8;
            }
            QHeaderView::section {
                background-color: #3498DB;
                color: white;
                padding: 10px;
                border: none;
                font-weight: bold;
                font-size: 10pt;
            }
        """)
        corr_layout.addWidget(self.corr_table)
        corr_section.layout().addLayout(corr_layout)
        layout.addWidget(corr_section)
        
        # Plot containers
        plot_section = self.create_section_widget("Model Diagnostics Plots",
            "Visual analysis of model performance")
        plot_layout = QtWidgets.QGridLayout()
        plot_layout.setContentsMargins(20, 15, 20, 15)
        plot_layout.setSpacing(15)
        
        # ROC Plot
        self.roc_container = QtWidgets.QWidget()
        self.roc_container.setMinimumHeight(400)
        self.roc_container.setStyleSheet("""
            QWidget {
                background-color: white;
                border: 2px solid #3498DB;
                border-radius: 8px;
            }
        """)
        self.roc_layout = QtWidgets.QVBoxLayout(self.roc_container)
        plot_layout.addWidget(self.roc_container, 0, 0)
        
        # KS Plot
        self.ks_container = QtWidgets.QWidget()
        self.ks_container.setMinimumHeight(400)
        self.ks_container.setStyleSheet("""
            QWidget {
                background-color: white;
                border: 2px solid #3498DB;
                border-radius: 8px;
            }
        """)
        self.ks_layout = QtWidgets.QVBoxLayout(self.ks_container)
        plot_layout.addWidget(self.ks_container, 0, 1)
        
        # Calibration Plot
        self.cal_container = QtWidgets.QWidget()
        self.cal_container.setMinimumHeight(400)
        self.cal_container.setStyleSheet("""
            QWidget {
                background-color: white;
                border: 2px solid #3498DB;
                border-radius: 8px;
            }
        """)
        self.cal_layout = QtWidgets.QVBoxLayout(self.cal_container)
        plot_layout.addWidget(self.cal_container, 1, 0)
        
        # Score Distribution Plot
        self.dist_container = QtWidgets.QWidget()
        self.dist_container.setMinimumHeight(400)
        self.dist_container.setStyleSheet("""
            QWidget {
                background-color: white;
                border: 2px solid #3498DB;
                border-radius: 8px;
            }
        """)
        self.dist_layout = QtWidgets.QVBoxLayout(self.dist_container)
        plot_layout.addWidget(self.dist_container, 1, 1)
        
        plot_section.layout().addLayout(plot_layout)
        layout.addWidget(plot_section)
        
        scroll_area.setWidget(scroll_content)
        main_layout = QtWidgets.QVBoxLayout(self.diagnostics_tab)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll_area)
    
    def create_section_widget(self, title, description):
        """Create a styled section widget"""
        section = QtWidgets.QWidget()
        section.setStyleSheet("""
            QWidget {
                background-color: white;
                border: 2px solid #3498DB;
                border-radius: 10px;
            }
        """)
        
        layout = QtWidgets.QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header
        header = QtWidgets.QWidget()
        header.setStyleSheet("""
            QWidget {
                background-color: #3498DB;
                border: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            }
        """)
        header_layout = QtWidgets.QVBoxLayout(header)
        header_layout.setContentsMargins(20, 15, 20, 15)
        
        title_label = QtWidgets.QLabel(title)
        title_label.setStyleSheet("""
            font-size: 14pt;
            color: white;
            font-weight: bold;
            background: transparent;
            border: none;
        """)
        header_layout.addWidget(title_label)
        
        desc_label = QtWidgets.QLabel(description)
        desc_label.setStyleSheet("""
            font-size: 10pt;
            color: #ECF0F1;
            background: transparent;
            border: none;
        """)
        header_layout.addWidget(desc_label)
        
        layout.addWidget(header)
        
        return section
    
    def get_input_style(self):
        """Common input widget style"""
        return """
            QSpinBox, QDoubleSpinBox {
                border: 2px solid #BDC3C7;
                border-radius: 6px;
                padding: 8px;
                font-size: 11pt;
                background-color: white;
            }
            QSpinBox:focus, QDoubleSpinBox:focus {
                border: 2px solid #3498DB;
            }
        """
    
    def retranslateUi(self, Scorecard):
        _translate = QtCore.QCoreApplication.translate
        Scorecard.setWindowTitle(_translate("Scorecard", "Scorecard Development - Credit Risk Workbench"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    app.setFont(QtGui.QFont("Segoe UI", 10))
    
    Scorecard = QtWidgets.QMainWindow()
    ui = Ui_Scorecard()
    ui.setupUi(Scorecard)
    Scorecard.show()
    sys.exit(app.exec())