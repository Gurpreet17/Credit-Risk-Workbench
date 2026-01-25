# loading_dialog.py
# Non-modal loading dialog with white background

from PyQt6 import QtCore, QtGui, QtWidgets


class LoadingDialog(QtWidgets.QDialog):
    """Non-modal loading dialog with clean white background"""
    
    def __init__(self, parent=None, title="Processing", message="Please wait..."):
        super().__init__(parent)
        
        self.setWindowTitle(title)
        self.setModal(False)  # Non-modal
        self.setFixedSize(500, 220)
        
        # Remove frame but keep title bar
        self.setWindowFlags(
            QtCore.Qt.WindowType.Dialog | 
            QtCore.Qt.WindowType.CustomizeWindowHint |
            QtCore.Qt.WindowType.WindowTitleHint
        )
        
        # White background
        self.setStyleSheet("""
            QDialog {
                background-color: white;
            }
        """)
        
        # Main layout
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Content widget with border
        content = QtWidgets.QWidget()
        content.setStyleSheet("""
            QWidget {
                background-color: white;
                border: 3px solid #3498DB;
                border-radius: 12px;
            }
        """)
        
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(40, 35, 40, 35)
        layout.setSpacing(22)
        
        # Icon + Title
        header_layout = QtWidgets.QHBoxLayout()
        header_layout.setSpacing(15)
        
        # Animated spinner
        self.spinner_label = QtWidgets.QLabel("⠋")
        self.spinner_label.setStyleSheet("""
            font-size: 36pt;
            background: transparent;
            border: none;
            color: #3498DB;
        """)
        header_layout.addWidget(self.spinner_label)
        
        title_label = QtWidgets.QLabel(title)
        title_label.setStyleSheet("""
            font-size: 17pt;
            color: #2C3E50;
            font-weight: bold;
            background: transparent;
            border: none;
        """)
        header_layout.addWidget(title_label, 1)
        
        layout.addLayout(header_layout)
        
        # Message
        self.message_label = QtWidgets.QLabel(message)
        self.message_label.setStyleSheet("""
            font-size: 11pt;
            color: #34495E;
            background: transparent;
            border: none;
        """)
        self.message_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        self.message_label.setWordWrap(True)
        layout.addWidget(self.message_label)
        
        # Progress bar
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setMinimumHeight(30)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #3498DB;
                border-radius: 12px;
                text-align: center;
                font-size: 11pt;
                font-weight: bold;
                background-color: #F8F9FA;
                color: #2C3E50;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3498DB, stop:0.5 #5DADE2, stop:1 #85C1E2);
                border-radius: 10px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        main_layout.addWidget(content)
        
        # Add shadow effect
        shadow = QtWidgets.QGraphicsDropShadowEffect()
        shadow.setBlurRadius(35)
        shadow.setXOffset(0)
        shadow.setYOffset(6)
        shadow.setColor(QtGui.QColor(0, 0, 0, 50))
        content.setGraphicsEffect(shadow)
        
        # Spinner animation
        self.rotation_angle = 0
        self.animation_timer = QtCore.QTimer(self)
        self.animation_timer.timeout.connect(self.rotate_spinner)
        self.animation_timer.start(80)  # Rotate every 80ms
        
        # Center on parent
        if parent:
            self.move_to_center()
            
    def center_on_parent(self):
        """Center dialog on parent window"""
        if self.parent():
            parent_geo = self.parent().geometry()
            dialog_geo = self.geometry()
            
            # Calculate center position
            x = parent_geo.x() + (parent_geo.width() - dialog_geo.width()) // 2
            y = parent_geo.y() + (parent_geo.height() - dialog_geo.height()) // 2
            
            self.move(x, y)
    
    def showEvent(self, event):
        """Override showEvent to re-center when shown"""
        super().showEvent(event)
        # Re-center every time the dialog is shown
        if self.parent():
            self.center_on_parent()
    
    def move_to_center(self):
        """Center dialog on parent window"""
        if self.parent():
            parent_geo = self.parent().geometry()
            dialog_geo = self.geometry()
            
            x = parent_geo.x() + (parent_geo.width() - dialog_geo.width()) // 2
            y = parent_geo.y() + (parent_geo.height() - dialog_geo.height()) // 2
            
            self.move(x, y)
    
    def rotate_spinner(self):
        """Rotate the spinner icon"""
        # Braille spinner characters for smooth animation
        spinners = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        self.rotation_angle = (self.rotation_angle + 1) % len(spinners)
        self.spinner_label.setText(spinners[self.rotation_angle])
    
    def set_progress(self, value):
        """Set progress bar value (0-100)"""
        self.progress_bar.setValue(value)
    
    def set_message(self, message):
        """Update message"""
        self.message_label.setText(message)
    
    def closeEvent(self, event):
        """Stop animation when closing"""
        self.animation_timer.stop()
        super().closeEvent(event)


if __name__ == "__main__":
    import sys
    from PyQt6.QtCore import QTimer
    
    app = QtWidgets.QApplication(sys.argv)
    
    # Create main window to test non-modal behavior
    main_window = QtWidgets.QMainWindow()
    main_window.setWindowTitle("Main Window")
    main_window.resize(800, 600)
    main_window.show()
    
    dialog = LoadingDialog(main_window, "Binning Variables", "Analyzing variable distributions...")
    
    # Simulate progress
    progress = [0]
    def update_progress():
        progress[0] += 3
        if progress[0] <= 100:
            dialog.set_progress(progress[0])
            if progress[0] % 20 == 0:
                dialog.set_message(f"Processing variable {progress[0]//10}/10...")
        else:
            dialog.close()
    
    timer = QTimer()
    timer.timeout.connect(update_progress)
    timer.start(150)
    
    dialog.show()
    sys.exit(app.exec())