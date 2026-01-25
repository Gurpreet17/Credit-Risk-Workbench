# binning.py
# UI for binning page - SAS EM inspired design

from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_Binning(object):
    def setupUi(self, Binning):
        Binning.setObjectName("Binning")
        Binning.resize(1400, 900)
        Binning.setMinimumSize(QtCore.QSize(1200, 800))
        
        self.centralwidget = QtWidgets.QWidget(parent=Binning)
        self.centralwidget.setObjectName("centralwidget")
        
        # Main layout
        main_layout = QtWidgets.QVBoxLayout(self.centralwidget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Header
        header = self.create_header()
        main_layout.addWidget(header)
        
        # Tab widget for Variables List and Grouping
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
                min-width: 150px;
            }
            QTabBar::tab:selected {
                background-color: white;
                color: #3498DB;
            }
            QTabBar::tab:hover:!selected {
                background-color: #D5DBDB;
            }
        """)
        
        # Variables List Tab
        self.variables_tab = QtWidgets.QWidget()
        self.setup_variables_tab()
        self.tab_widget.addTab(self.variables_tab, "Variables")
        
        # Grouping Tab (Scrollable)
        self.grouping_tab = QtWidgets.QWidget()
        self.setup_grouping_tab()
        self.tab_widget.addTab(self.grouping_tab, "Groupings")
        
        main_layout.addWidget(self.tab_widget)
        
        Binning.setCentralWidget(self.centralwidget)
        
        # Status bar
        self.statusbar = QtWidgets.QStatusBar(parent=Binning)
        self.statusbar.setStyleSheet("""
            QStatusBar {
                background-color: #ECF0F1;
                color: #7F8C8D;
                border-top: 1px solid #BDC3C7;
                font-size: 10pt;
            }
        """)
        Binning.setStatusBar(self.statusbar)
        
        self.retranslateUi(Binning)
        QtCore.QMetaObject.connectSlotsByName(Binning)
    
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
        
        title = QtWidgets.QLabel("Interactive Grouping")
        font = QtGui.QFont("Segoe UI", 24, QtGui.QFont.Weight.Bold)
        title.setFont(font)
        title.setStyleSheet("color: white; background: transparent;")
        layout.addWidget(title)
        
        layout.addStretch()
        
        subtitle = QtWidgets.QLabel("Adjust variable binning and review WOE analysis")
        font = QtGui.QFont("Segoe UI", 11)
        subtitle.setFont(font)
        subtitle.setStyleSheet("color: #ECF0F1; background: transparent;")
        layout.addWidget(subtitle)
        
        return header
    
    def setup_variables_tab(self):
        """Setup variables list tab"""
        layout = QtWidgets.QVBoxLayout(self.variables_tab)
        layout.setContentsMargins(40, 30, 40, 30)
        layout.setSpacing(20)
        
        # Table for variables
        self.variables_table = QtWidgets.QTableWidget()
        self.variables_table.setColumnCount(6)
        self.variables_table.setHorizontalHeaderLabels([
            "Variable Name", "Type", "IV", "AUC", "Gini", "# Bins"
        ])
        
        # Table properties
        self.variables_table.setSelectionBehavior(QtWidgets.QTableWidget.SelectionBehavior.SelectRows)
        self.variables_table.setSelectionMode(QtWidgets.QTableWidget.SelectionMode.SingleSelection)
        self.variables_table.setAlternatingRowColors(True)
        self.variables_table.setSortingEnabled(False)
        self.variables_table.verticalHeader().setVisible(False)
        
        # Table styling - NO COLOR CODING
        self.variables_table.setStyleSheet("""
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
            QTableWidget::item {
                padding: 10px;
                border-bottom: 1px solid #E1E4E8;
            }
            QTableWidget::item:selected {
                background-color: #5DADE2;
                color: white;
            }
            QTableWidget::item:hover {
                background-color: #AED6F1;
            }
        """)
        
        self.variables_table.verticalHeader().setDefaultSectionSize(45)
        
        layout.addWidget(self.variables_table)
        
        # Button section
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.setSpacing(15)
        
        # Back to Variable Selection button
        self.back_to_var_selection_btn = QtWidgets.QPushButton("← Back to Variable Selection")
        self.back_to_var_selection_btn.setMinimumHeight(45)
        self.back_to_var_selection_btn.setMinimumWidth(240)
        self.back_to_var_selection_btn.setStyleSheet("""
            QPushButton {
                background-color: #7F8C8D;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 11pt;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5D6D7E;
            }
            QPushButton:pressed {
                background-color: #4A5568;
            }
        """)
        btn_layout.addWidget(self.back_to_var_selection_btn)
        
        btn_layout.addStretch()
        
        # Proceed to Scorecard button
        self.proceed_to_scorecard_btn = QtWidgets.QPushButton("Proceed to Scorecard →")
        self.proceed_to_scorecard_btn.setMinimumHeight(45)
        self.proceed_to_scorecard_btn.setMinimumWidth(220)
        self.proceed_to_scorecard_btn.setStyleSheet("""
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
            QPushButton:pressed {
                background-color: #1E8449;
            }
            QPushButton:disabled {
                background-color: #BDC3C7;
            }
        """)
        btn_layout.addWidget(self.proceed_to_scorecard_btn)
        
        layout.addLayout(btn_layout)
    
    def setup_grouping_tab(self):
        """Setup grouping/editing tab - SCROLLABLE"""
        # Main scroll area
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll_area.setStyleSheet("QScrollArea { background-color: #F5F6FA; border: none; }")
        
        # Scroll content widget
        scroll_content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(scroll_content)
        layout.setContentsMargins(30, 20, 30, 20)
        layout.setSpacing(15)
        
        # Variable header with navigation
        header_layout = QtWidgets.QHBoxLayout()
        
        self.back_to_vars_btn = QtWidgets.QPushButton("← Variables")
        self.back_to_vars_btn.setMinimumHeight(35)
        self.back_to_vars_btn.setStyleSheet("""
            QPushButton {
                background-color: #7F8C8D;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 10pt;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5D6D7E;
            }
        """)
        header_layout.addWidget(self.back_to_vars_btn)
        
        header_layout.addSpacing(20)
        
        self.current_variable_label = QtWidgets.QLabel("Selected Variable: None")
        self.current_variable_label.setStyleSheet("""
            font-size: 16pt;
            color: #2C3E50;
            font-weight: bold;
            background: transparent;
        """)
        header_layout.addWidget(self.current_variable_label)
        
        header_layout.addStretch()
        
        # Navigation buttons (Previous/Next variable)
        self.prev_var_btn = QtWidgets.QPushButton("◄ Previous")
        self.prev_var_btn.setMinimumHeight(35)
        self.prev_var_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498DB;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 10pt;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
            QPushButton:disabled {
                background-color: #BDC3C7;
            }
        """)
        header_layout.addWidget(self.prev_var_btn)
        
        self.next_var_btn = QtWidgets.QPushButton("Next ►")
        self.next_var_btn.setMinimumHeight(35)
        self.next_var_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498DB;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 10pt;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
            QPushButton:disabled {
                background-color: #BDC3C7;
            }
        """)
        header_layout.addWidget(self.next_var_btn)
        
        layout.addLayout(header_layout)
        
        # Main content area with plot and table side-by-side
        content_layout = QtWidgets.QHBoxLayout()
        content_layout.setSpacing(20)
        
        # LEFT SIDE: Plot
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)
        
        plot_label = QtWidgets.QLabel("WOE Analysis")
        plot_label.setStyleSheet("font-size: 13pt; color: #2C3E50; font-weight: bold;")
        left_layout.addWidget(plot_label)
        
        # Plot container
        plot_outer = QtWidgets.QWidget()
        plot_outer.setStyleSheet("""
            QWidget {
                background-color: white;
                border: 2px solid #3498DB;
                border-radius: 8px;
            }
        """)
        plot_outer_layout = QtWidgets.QVBoxLayout(plot_outer)
        plot_outer_layout.setContentsMargins(10, 10, 10, 10)
        
        self.plot_container = QtWidgets.QWidget()
        self.plot_container.setMinimumHeight(450)
        self.plot_container.setMinimumWidth(600)
        self.plot_layout = QtWidgets.QVBoxLayout(self.plot_container)
        self.plot_layout.setContentsMargins(0, 0, 0, 0)
        
        plot_outer_layout.addWidget(self.plot_container)
        left_layout.addWidget(plot_outer)
        
        content_layout.addWidget(left_widget, 3)
        
        # RIGHT SIDE: Variable Statistics (SAS EM style)
        right_widget = QtWidgets.QWidget()
        right_widget.setMaximumWidth(280)
        right_widget.setStyleSheet("""
            QWidget {
                background-color: white;
                border: 2px solid #3498DB;
                border-radius: 8px;
            }
        """)
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        right_layout.setContentsMargins(20, 20, 20, 20)
        right_layout.setSpacing(15)
        
        stats_title = QtWidgets.QLabel("Variable Statistics")
        stats_title.setStyleSheet("""
            font-size: 13pt;
            color: #2C3E50;
            font-weight: bold;
            background: transparent;
            border: none;
        """)
        stats_title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(stats_title)
        
        # Original Gini
        self.original_gini_label = self.create_stat_label("Original Gini", "0.000")
        right_layout.addWidget(self.original_gini_label)
        
        # New Gini
        self.new_gini_label = self.create_stat_label("New Gini", "0.000")
        right_layout.addWidget(self.new_gini_label)
        
        # Separator
        separator1 = QtWidgets.QFrame()
        separator1.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        separator1.setStyleSheet("background-color: #BDC3C7; border: none;")
        separator1.setMaximumHeight(2)
        right_layout.addWidget(separator1)
        
        # Original IV
        self.original_iv_label = self.create_stat_label("Original Information Value", "0.000")
        right_layout.addWidget(self.original_iv_label)
        
        # New IV
        self.new_iv_label = self.create_stat_label("New Information Value", "0.000")
        right_layout.addWidget(self.new_iv_label)
        
        # Separator
        separator2 = QtWidgets.QFrame()
        separator2.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        separator2.setStyleSheet("background-color: #BDC3C7; border: none;")
        separator2.setMaximumHeight(2)
        right_layout.addWidget(separator2)
        
        # AUC
        self.auc_label = self.create_stat_label("AUC", "0.000")
        right_layout.addWidget(self.auc_label)
        
        right_layout.addStretch()
        
        content_layout.addWidget(right_widget, 1)
        
        layout.addLayout(content_layout)
        
        # Binning Details Table
        table_header = QtWidgets.QLabel("Binning Details")
        table_header.setStyleSheet("font-size: 13pt; color: #2C3E50; font-weight: bold; margin-top: 15px;")
        layout.addWidget(table_header)
        
        # Binning table
        self.binning_table = QtWidgets.QTableWidget()
        self.binning_table.setMinimumHeight(350)
        self.binning_table.setSelectionBehavior(QtWidgets.QTableWidget.SelectionBehavior.SelectRows)
        self.binning_table.setSelectionMode(QtWidgets.QTableWidget.SelectionMode.MultiSelection)
        self.binning_table.setAlternatingRowColors(True)
        self.binning_table.verticalHeader().setVisible(False)
        
        self.binning_table.setStyleSheet("""
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
            QTableWidget::item {
                padding: 8px;
            }
            QTableWidget::item:selected {
                background-color: #5DADE2;
                color: white;
            }
            QTableWidget::item:hover {
                background-color: #AED6F1;
            }
        """)
        
        layout.addWidget(self.binning_table)
        
        # Action buttons
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.setSpacing(15)
        
        # Merge bins button
        self.merge_bins_btn = QtWidgets.QPushButton("Merge Selected Bins")
        self.merge_bins_btn.setMinimumHeight(45)
        self.merge_bins_btn.setMinimumWidth(180)
        self.merge_bins_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498DB;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 20px;
                font-size: 11pt;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
            QPushButton:pressed {
                background-color: #21618C;
            }
            QPushButton:disabled {
                background-color: #BDC3C7;
            }
        """)
        btn_layout.addWidget(self.merge_bins_btn)
        
        # Split bin button
        self.split_bin_btn = QtWidgets.QPushButton("Split Bin")
        self.split_bin_btn.setMinimumHeight(45)
        self.split_bin_btn.setMinimumWidth(150)
        self.split_bin_btn.setStyleSheet("""
            QPushButton {
                background-color: #5DADE2;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 20px;
                font-size: 11pt;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3498DB;
            }
            QPushButton:pressed {
                background-color: #2980B9;
            }
            QPushButton:disabled {
                background-color: #BDC3C7;
            }
        """)
        btn_layout.addWidget(self.split_bin_btn)
        
        btn_layout.addStretch()
        
        # Reset button
        self.reset_binning_btn = QtWidgets.QPushButton("Reset to Original")
        self.reset_binning_btn.setMinimumHeight(45)
        self.reset_binning_btn.setMinimumWidth(160)
        self.reset_binning_btn.setStyleSheet("""
            QPushButton {
                background-color: #E74C3C;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 20px;
                font-size: 11pt;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #C0392B;
            }
            QPushButton:pressed {
                background-color: #A93226;
            }
        """)
        btn_layout.addWidget(self.reset_binning_btn)
        
        # Save binning button
        self.save_binning_btn = QtWidgets.QPushButton("Save Binning")
        self.save_binning_btn.setMinimumHeight(45)
        self.save_binning_btn.setMinimumWidth(160)
        self.save_binning_btn.setStyleSheet("""
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
            QPushButton:pressed {
                background-color: #1E8449;
            }
        """)
        btn_layout.addWidget(self.save_binning_btn)
        
        layout.addLayout(btn_layout)
        
        # Add some bottom spacing
        layout.addSpacing(30)
        
        # Set scroll content
        scroll_area.setWidget(scroll_content)
        
        # Set main layout for grouping tab
        main_grouping_layout = QtWidgets.QVBoxLayout(self.grouping_tab)
        main_grouping_layout.setContentsMargins(0, 0, 0, 0)
        main_grouping_layout.addWidget(scroll_area)
    
    def create_stat_label(self, title, value):
        """Create a statistic label (SAS EM style)"""
        widget = QtWidgets.QWidget()
        widget.setStyleSheet("background: transparent; border: none;")
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 5, 0, 5)
        layout.setSpacing(5)
        
        title_label = QtWidgets.QLabel(title)
        title_label.setStyleSheet("""
            font-size: 9pt;
            color: #7F8C8D;
            background: transparent;
            border: none;
        """)
        title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(title_label)
        
        value_label = QtWidgets.QLabel(value)
        value_label.setObjectName(f"{title.replace(' ', '_').lower()}_value")
        value_label.setStyleSheet("""
            font-size: 16pt;
            color: #2C3E50;
            font-weight: bold;
            background: transparent;
            border: none;
        """)
        value_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(value_label)
        
        return widget
    
    def retranslateUi(self, Binning):
        _translate = QtCore.QCoreApplication.translate
        Binning.setWindowTitle(_translate("Binning", "Interactive Grouping - Credit Risk Workbench"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    app.setFont(QtGui.QFont("Segoe UI", 10))
    
    Binning = QtWidgets.QMainWindow()
    ui = Ui_Binning()
    ui.setupUi(Binning)
    Binning.show()
    sys.exit(app.exec())