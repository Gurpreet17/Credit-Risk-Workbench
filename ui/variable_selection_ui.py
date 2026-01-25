# variable_selection.py
# UI for variable selection page

from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_VariableSelection(object):
    def setupUi(self, VariableSelection):
        VariableSelection.setObjectName("VariableSelection")
        VariableSelection.resize(1400, 900)
        VariableSelection.setMinimumSize(QtCore.QSize(1200, 800))
        
        # Main widget
        self.centralwidget = QtWidgets.QWidget(parent=VariableSelection)
        self.centralwidget.setObjectName("centralwidget")
        
        # Main vertical layout
        main_layout = QtWidgets.QVBoxLayout(self.centralwidget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Header section
        header_widget = QtWidgets.QWidget()
        header_widget.setMinimumHeight(120)
        header_widget.setMaximumHeight(120)
        header_widget.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #2C3E50, stop:1 #3498DB);
            }
        """)
        
        header_layout = QtWidgets.QVBoxLayout(header_widget)
        header_layout.setContentsMargins(60, 20, 60, 20)
        header_layout.setSpacing(5)
        
        # Page title
        page_title = QtWidgets.QLabel("Variable Selection")
        title_font = QtGui.QFont()
        title_font.setFamily("Segoe UI")
        title_font.setPointSize(32)
        title_font.setBold(True)
        page_title.setFont(title_font)
        page_title.setStyleSheet("color: white;")
        page_title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(page_title)
        
        # Subtitle
        subtitle = QtWidgets.QLabel("Select target variable and features for model training")
        subtitle_font = QtGui.QFont()
        subtitle_font.setFamily("Segoe UI")
        subtitle_font.setPointSize(13)
        subtitle.setFont(subtitle_font)
        subtitle.setStyleSheet("color: #ECF0F1;")
        subtitle.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(subtitle)
        
        main_layout.addWidget(header_widget)
        
        # Content area with scroll
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll_area.setStyleSheet("QScrollArea { background-color: #F5F6FA; border: none; }")
        
        scroll_content = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(80, 40, 80, 40)
        scroll_layout.setSpacing(30)
        
        # Dataset Info Section
        info_widget = self.create_info_section()
        scroll_layout.addWidget(info_widget)
        
        # Target Selection Section
        target_widget = self.create_target_section()
        scroll_layout.addWidget(target_widget)
        
        # Feature Selection Section
        features_widget = self.create_features_section()
        scroll_layout.addWidget(features_widget)
        
        # Navigation Buttons
        nav_widget = self.create_navigation_section()
        scroll_layout.addWidget(nav_widget)
        
        scroll_layout.addStretch()
        
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)
        
        VariableSelection.setCentralWidget(self.centralwidget)
        
        # Status bar
        self.statusbar = QtWidgets.QStatusBar(parent=VariableSelection)
        self.statusbar.setStyleSheet("""
            QStatusBar {
                background-color: #ECF0F1;
                color: #7F8C8D;
                border-top: 1px solid #BDC3C7;
            }
        """)
        VariableSelection.setStatusBar(self.statusbar)
        self.statusbar.showMessage("Select target variable and features")
        
        self.retranslateUi(VariableSelection)
        QtCore.QMetaObject.connectSlotsByName(VariableSelection)

    def create_shadow(self):
        """Create shadow effect"""
        shadow = QtWidgets.QGraphicsDropShadowEffect()
        shadow.setBlurRadius(25)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QtGui.QColor(0, 0, 0, 30))
        return shadow

    def create_info_section(self):
        """Create dataset information section"""
        widget = QtWidgets.QWidget()
        widget.setStyleSheet("""
            QWidget {
                background-color: white;
                border-radius: 12px;
            }
        """)
        widget.setGraphicsEffect(self.create_shadow())
        
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(40, 30, 40, 30)
        layout.setSpacing(15)
        
        # Title
        title = QtWidgets.QLabel("Dataset Information")
        title_font = QtGui.QFont()
        title_font.setFamily("Segoe UI")
        title_font.setPointSize(20)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: #2C3E50;")
        layout.addWidget(title)
        
        # Info labels (will be populated dynamically)
        self.train_info_label = QtWidgets.QLabel("Training Data: Not loaded")
        self.train_info_label.setStyleSheet("color: #7F8C8D; font-size: 11pt;")
        layout.addWidget(self.train_info_label)
        
        self.val_info_label = QtWidgets.QLabel("Validation Data: Not loaded")
        self.val_info_label.setStyleSheet("color: #7F8C8D; font-size: 11pt;")
        layout.addWidget(self.val_info_label)
        
        return widget

    def create_target_section(self):
        """Create target variable selection section"""
        widget = QtWidgets.QWidget()
        widget.setStyleSheet("""
            QWidget {
                background-color: white;
                border-radius: 12px;
            }
        """)
        widget.setGraphicsEffect(self.create_shadow())
        
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(40, 30, 40, 30)
        layout.setSpacing(20)
        
        # Title
        title = QtWidgets.QLabel("Target Variable")
        title_font = QtGui.QFont()
        title_font.setFamily("Segoe UI")
        title_font.setPointSize(20)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: #2C3E50;")
        layout.addWidget(title)
        
        # Description
        desc = QtWidgets.QLabel("Select the target variable (dependent variable) for your credit risk model")
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #7F8C8D; font-size: 11pt; margin-bottom: 10px;")
        layout.addWidget(desc)
        
        # Target dropdown
        target_layout = QtWidgets.QHBoxLayout()
        target_layout.setSpacing(15)
        
        target_label = QtWidgets.QLabel("Target Variable:")
        target_label.setStyleSheet("color: #2C3E50; font-size: 12pt; font-weight: bold;")
        target_label.setFixedWidth(150)
        target_layout.addWidget(target_label)
        
        self.target_dropdown = QtWidgets.QComboBox()
        self.target_dropdown.setMinimumHeight(40)
        self.target_dropdown.setStyleSheet("""
            QComboBox {
                background-color: #F8F9FA;
                border: 2px solid #E1E4E8;
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 11pt;
                color: #2C3E50;
            }
            QComboBox:hover {
                border: 2px solid #3498DB;
            }
            QComboBox::drop-down {
                border: none;
                width: 30px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #2C3E50;
                margin-right: 10px;
            }
            QComboBox QAbstractItemView {
                background-color: white;
                border: 1px solid #E1E4E8;
                selection-background-color: #3498DB;
                selection-color: white;
            }
        """)
        target_layout.addWidget(self.target_dropdown, 1)
        
        layout.addLayout(target_layout)
        
        # Target info label
        self.target_info_label = QtWidgets.QLabel("")
        self.target_info_label.setStyleSheet("color: #27AE60; font-size: 10pt; font-style: italic;")
        layout.addWidget(self.target_info_label)
        
        return widget

    def create_features_section(self):
        """Create feature selection section"""
        widget = QtWidgets.QWidget()
        widget.setStyleSheet("""
            QWidget {
                background-color: white;
                border-radius: 12px;
            }
        """)
        widget.setGraphicsEffect(self.create_shadow())
        
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(40, 30, 40, 30)
        layout.setSpacing(20)
        
        # Title
        title = QtWidgets.QLabel("Feature Variables")
        title_font = QtGui.QFont()
        title_font.setFamily("Segoe UI")
        title_font.setPointSize(20)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: #2C3E50;")
        layout.addWidget(title)
        
        # Description
        desc = QtWidgets.QLabel("Select the feature variables (independent variables) to include in your model")
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #7F8C8D; font-size: 11pt; margin-bottom: 10px;")
        layout.addWidget(desc)
        
        # Action buttons
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.setSpacing(15)
        
        self.select_all_btn = QtWidgets.QPushButton("Select All")
        self.select_all_btn.setMinimumHeight(40)
        self.select_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498DB;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 11pt;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
            QPushButton:pressed {
                background-color: #21618C;
            }
        """)
        button_layout.addWidget(self.select_all_btn)
        
        self.deselect_all_btn = QtWidgets.QPushButton("Deselect All")
        self.deselect_all_btn.setMinimumHeight(40)
        self.deselect_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #95A5A6;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 11pt;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #7F8C8D;
            }
            QPushButton:pressed {
                background-color: #6C7A89;
            }
        """)
        button_layout.addWidget(self.deselect_all_btn)
        
        button_layout.addStretch()
        
        # Selected count label
        self.selected_count_label = QtWidgets.QLabel("Selected: 0 variables")
        self.selected_count_label.setStyleSheet("color: #3498DB; font-size: 11pt; font-weight: bold;")
        button_layout.addWidget(self.selected_count_label)
        
        layout.addLayout(button_layout)
        
        # Search box
        search_layout = QtWidgets.QHBoxLayout()
        search_layout.setSpacing(10)
        
        search_icon_label = QtWidgets.QLabel("🔍")
        search_icon_label.setStyleSheet("font-size: 16pt;")
        search_layout.addWidget(search_icon_label)
        
        self.search_box = QtWidgets.QLineEdit()
        self.search_box.setPlaceholderText("Search variables...")
        self.search_box.setMinimumHeight(40)
        self.search_box.setStyleSheet("""
            QLineEdit {
                background-color: #F8F9FA;
                border: 2px solid #E1E4E8;
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 11pt;
                color: #2C3E50;
            }
            QLineEdit:focus {
                border: 2px solid #3498DB;
            }
        """)
        search_layout.addWidget(self.search_box, 1)
        
        layout.addLayout(search_layout)
        
        # Feature list with checkboxes
        self.feature_list = QtWidgets.QListWidget()
        self.feature_list.setMinimumHeight(400)
        self.feature_list.setStyleSheet("""
            QListWidget {
                background-color: #F8F9FA;
                border: 2px solid #E1E4E8;
                border-radius: 6px;
                padding: 10px;
                font-size: 11pt;
            }
            QListWidget::item {
                padding: 8px;
                border-radius: 4px;
                margin: 2px 0px;
            }
            QListWidget::item:hover {
                background-color: #E8F4F8;
            }
            QListWidget::item:selected {
                background-color: #D6EAF8;
                color: #2C3E50;
            }
        """)
        layout.addWidget(self.feature_list)
        
        return widget

    def create_navigation_section(self):
        """Create navigation buttons"""
        widget = QtWidgets.QWidget()
        widget.setStyleSheet("background-color: transparent;")
        
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(0, 20, 0, 20)
        layout.setSpacing(20)
        
        # Back button
        self.back_btn = QtWidgets.QPushButton("← Back to Home")
        self.back_btn.setMinimumHeight(50)
        self.back_btn.setMinimumWidth(180)
        self.back_btn.setStyleSheet("""
            QPushButton {
                background-color: #95A5A6;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 13pt;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #7F8C8D;
            }
            QPushButton:pressed {
                background-color: #6C7A89;
            }
        """)
        layout.addWidget(self.back_btn)
        
        layout.addStretch()
        
        # Next button
        self.next_btn = QtWidgets.QPushButton("Continue to Binning →")
        self.next_btn.setMinimumHeight(50)
        self.next_btn.setMinimumWidth(220)
        self.next_btn.setStyleSheet("""
            QPushButton {
                background-color: #27AE60;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 13pt;
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
        layout.addWidget(self.next_btn)
        
        return widget

    def retranslateUi(self, VariableSelection):
        _translate = QtCore.QCoreApplication.translate
        VariableSelection.setWindowTitle(_translate("VariableSelection", "Variable Selection - Credit Risk Workbench"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    app.setFont(QtGui.QFont("Segoe UI", 10))
    
    VariableSelection = QtWidgets.QMainWindow()
    ui = Ui_VariableSelection()
    ui.setupUi(VariableSelection)
    VariableSelection.show()
    sys.exit(app.exec())