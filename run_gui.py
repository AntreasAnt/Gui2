#!/usr/bin/env python3
"""
Launch script for the Image Filtering GUI
"""

import sys
import os

def check_dependencies():
    """Check if required packages are installed."""
    try:
        import cv2
        import numpy as np
        from PyQt5 import QtWidgets, QtGui, QtCore
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("\nPlease install required packages:")
        print("pip install -r requirements.txt")
        return False

def main():
    if not check_dependencies():
        sys.exit(1)
    
    # Import and run the GUI
    from filtering_gui import MainWindow, QtWidgets, QtCore
    
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
    app = QtWidgets.QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Image Filtering - Linear")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("DIP Lab")
    
    win = MainWindow()
    win.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
