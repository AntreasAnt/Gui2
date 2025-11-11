# filtering_gui.py
# -*- coding: utf-8 -*-
"""
Image Filtering & Compression Lab GUI
-----------------------------------
Extended GUI for linear filters, non-linear filters, and compression
"""
import os
import sys
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore

APP_TITLE = "Image Filtering & Compression"

DARK_QSS = """
* { font-family: 'Segoe UI', Arial; font-size: 10.5pt; color: #E6E6E6; }
QWidget { background-color: #1f2023; }
QLabel#Banner { color: #f1c57a; font-weight: 700; }
QPushButton { background-color: #f1c57a; color: #2b2d31; border: 0px; padding: 8px 14px; border-radius: 6px; }
QPushButton:hover { background-color: #ffd28e; }
QPushButton:pressed { background-color: #e7b86a; }
QPushButton#Secondary { background: #3a3c42; color: #E6E6E6; font-weight: 500; }
QPushButton#Secondary:hover { background: #44474e; }

QSpinBox, QDoubleSpinBox {
 background: #1c1d21; border: 1px solid #3a3c42; padding: 6px; border-radius: 6px;
 color: white;
}
QComboBox {
 background: #1c1d21; border: 1px solid #3a3c42; padding: 6px 20px 6px 6px; border-radius: 6px;
 min-width: 120px; selection-background-color: #f1c57a; color: white;
}
QLineEdit, QSlider {
 background: #1c1d21; border: 1px solid #3a3c42; padding: 6px; border-radius: 6px;
}
QTextEdit {
 background: #1c1d21; border: 1px solid #3a3c42; color: #E6E6E6; padding: 8px;
}
QGroupBox {
 border: 1px solid #3a3c42; border-radius: 6px; margin-top: 8px; padding-top: 12px; font-weight: 600;
}
QGroupBox::title {
 subcontrol-origin: margin; left: 10px; padding: 0 5px;
}
QSlider::groove:horizontal { height: 6px; background: #3a3c42; border-radius: 3px; }
QSlider::handle:horizontal { background: #f1c57a; width: 14px; border-radius: 7px; margin: -4px 0; }
QTabWidget::pane { border: 1px solid #3a3c42; background-color: #1f2023; }
QTabBar::tab { background: #2b2d31; color: white; padding: 10px 20px; margin-right: 0px; border-top-left-radius: 4px; border-top-right-radius: 4px; }
QTabBar::tab:selected { background: #f1c57a; color: #2b2d31;  }
QTabBar::tab:hover:!selected { background: #3a3c42; color: #E6E6E6; }
"""

# ==================== TODO FUNCTIONS ====================
# Import the TODO functions from nonlinear_compression module
try:
    from nonlinear_compression import (
        apply_nonlinear_filter,
        compress_image,
        compute_diff_image,
        compute_metrics
    )
except ImportError:
    # Fallback if module not found
    def apply_nonlinear_filter(img_rgb: np.ndarray, kind: str, **params) -> np.ndarray:
        raise NotImplementedError("TODO: Create nonlinear_compression.py and implement apply_nonlinear_filter")

    def compress_image(img_rgb: np.ndarray, codec: str, quality: int) -> tuple:
        raise NotImplementedError("TODO: Create nonlinear_compression.py and implement compress_image")

    def compute_diff_image(orig_rgb: np.ndarray, comp_rgb: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO: Create nonlinear_compression.py and implement compute_diff_image")

    def compute_metrics(orig_rgb: np.ndarray, comp_rgb: np.ndarray, encoded_bytes: bytes, codec: str) -> str:
        raise NotImplementedError("TODO: Create nonlinear_compression.py and implement compute_metrics")

def np_rgb_to_qpixmap(img_rgb: np.ndarray, target_size: QtCore.QSize) -> QtGui.QPixmap:
    if img_rgb is None:
        return QtGui.QPixmap()
    h, w, ch = img_rgb.shape
    qimg = QtGui.QImage(img_rgb.data, w, h, ch*w, QtGui.QImage.Format_RGB888)
    pix = QtGui.QPixmap.fromImage(qimg)
    return pix.scaled(target_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

class ScaledImageLabel(QtWidgets.QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_rgb = None
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(560, 420)
        self.setStyleSheet("border: 1px dashed #3a3c42;")

    def set_image_rgb(self, img_rgb: np.ndarray):
        self._last_rgb = img_rgb
        self._update_pix()

    def clear_image(self, placeholder="— no image —"):
        self._last_rgb = None
        self.setText(placeholder)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._update_pix()

    def _update_pix(self):
        if self._last_rgb is None:
            return
        pix = np_rgb_to_qpixmap(self._last_rgb, self.size())
        self.setPixmap(pix)
        self.setText("")

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1200, 720)
        self._orig = None
        self._filtered = None

        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        root_layout = QtWidgets.QVBoxLayout(root)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(8)

        # Top bar
        self.topBar = QtWidgets.QFrame(objectName="TopBar")
        tlay = QtWidgets.QHBoxLayout(self.topBar)
        tlay.setContentsMargins(12, 8, 12, 8)
        self.title = QtWidgets.QLabel(APP_TITLE, objectName="Banner")
        self.btnOpen = QtWidgets.QPushButton("Load Image")
        self.btnOpen.setObjectName("Secondary")
        self.btnSave = QtWidgets.QPushButton("Save Result")
        self.btnSave.setObjectName("Secondary")
        tlay.addWidget(self.title)
        tlay.addStretch(1)
        tlay.addWidget(self.btnOpen)
        tlay.addWidget(self.btnSave)
        root_layout.addWidget(self.topBar)

        # Main content
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.setHandleWidth(8)
        root_layout.addWidget(splitter, 1)

        # Side panel
        self.side = QtWidgets.QFrame(objectName="SidePanel")
        side = QtWidgets.QVBoxLayout(self.side)


        # Form controls
        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignLeft)

        # Linear Filters Section
        form.addRow(QtWidgets.QLabel("— Linear Filters —"))
        
        self.cboFilter = QtWidgets.QComboBox()
        self.cboFilter.addItems(["Box/Average", "Gaussian", "Sobel X", "Sobel Y", "Laplacian", "Unsharp"])

        self.spnK = QtWidgets.QSpinBox()
        self.spnK.setRange(3, 101)
        self.spnK.setSingleStep(2)
        self.spnK.setValue(5)

        self.dspSigma = QtWidgets.QDoubleSpinBox()
        self.dspSigma.setRange(0.0, 25.0)
        self.dspSigma.setValue(1.0)
        self.dspSigma.setSingleStep(0.1)

        self.spnIter = QtWidgets.QSpinBox()
        self.spnIter.setRange(1, 50)
        self.spnIter.setValue(1)

        self.cboBorder = QtWidgets.QComboBox()
        self.cboBorder.addItems(["reflect", "replicate", "constant"])

        self.chkGray = QtWidgets.QCheckBox("Grayscale processing")

        # Alpha slider for Unsharp
        alphaRow = QtWidgets.QHBoxLayout()
        self.sldAlpha = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sldAlpha.setRange(0, 300)
        self.sldAlpha.setValue(100)
        self.lblAlpha = QtWidgets.QLabel("α = 1.00")
        alphaRow.addWidget(self.sldAlpha)
        alphaRow.addWidget(self.lblAlpha)

        form.addRow("Filter:", self.cboFilter)
        form.addRow("Kernel size (odd):", self.spnK)
        form.addRow("σ (Gaussian):", self.dspSigma)
        form.addRow("Iterations:", self.spnIter)
        form.addRow("Border:", self.cboBorder)
        form.addRow("", self.chkGray)
        form.addRow("Unsharp α:", alphaRow)

        # Non-Linear Filters Section
        form.addRow(QtWidgets.QLabel("— Non-Linear Filters —"))
        
        self.cboNLFilter = QtWidgets.QComboBox()
        self.cboNLFilter.addItems(["Median", "Bilateral", "NLMeans"])
        
        self.spnNLK = QtWidgets.QSpinBox()
        self.spnNLK.setRange(1, 99)
        self.spnNLK.setSingleStep(2)
        self.spnNLK.setValue(5)
        
        self.spnBilateralD = QtWidgets.QSpinBox()
        self.spnBilateralD.setRange(1, 25)
        self.spnBilateralD.setValue(7)
        
        self.dspSigmaColor = QtWidgets.QDoubleSpinBox()
        self.dspSigmaColor.setRange(1.0, 250.0)
        self.dspSigmaColor.setValue(75.0)
        
        self.dspSigmaSpace = QtWidgets.QDoubleSpinBox()
        self.dspSigmaSpace.setRange(1.0, 250.0)
        self.dspSigmaSpace.setValue(75.0)
        
        self.dspNLMeansH = QtWidgets.QDoubleSpinBox()
        self.dspNLMeansH.setRange(1.0, 30.0)
        self.dspNLMeansH.setValue(10.0)
        
        form.addRow("NL Filter:", self.cboNLFilter)
        form.addRow("Kernel (odd):", self.spnNLK)
        form.addRow("Bilateral d:", self.spnBilateralD)
        form.addRow("σ Color:", self.dspSigmaColor)
        form.addRow("σ Space:", self.dspSigmaSpace)
        form.addRow("NLMeans h:", self.dspNLMeansH)

        side.addLayout(form)

        # Buttons
        btnRow = QtWidgets.QHBoxLayout()
        self.btnApply = QtWidgets.QPushButton("Apply Linear")
        self.btnApplyNL = QtWidgets.QPushButton("Apply Non-Linear")
        self.btnReset = QtWidgets.QPushButton("Reset")
        btnRow.addWidget(self.btnApply)
        btnRow.addWidget(self.btnApplyNL)
        btnRow.addWidget(self.btnReset)
        side.addLayout(btnRow)

        splitter.addWidget(self.side)

        # Image tabs
        self.tabs = QtWidgets.QTabWidget()
        self.tabOrig = QtWidgets.QWidget()
        self.tabFilt = QtWidgets.QWidget()
        self.tabComp = QtWidgets.QWidget()
        
        self.tabs.addTab(self.tabOrig, "Original")
        self.tabs.addTab(self.tabFilt, "Filtered")
        self.tabs.addTab(self.tabComp, "Compression")

        oLay = QtWidgets.QVBoxLayout(self.tabOrig)
        oLay.setContentsMargins(8, 8, 8, 8)
        self.viewOrig = ScaledImageLabel("— Load an image —")
        oLay.addWidget(self.viewOrig, 1)

        fLay = QtWidgets.QVBoxLayout(self.tabFilt)
        fLay.setContentsMargins(8, 8, 8, 8)
        self.viewFilt = ScaledImageLabel("— Apply a filter —")
        fLay.addWidget(self.viewFilt, 1)

        # Compression Tab
        cLay = QtWidgets.QVBoxLayout(self.tabComp)
        cLay.setContentsMargins(8, 8, 8, 8)
        
        # Compression controls
        ctrlRow = QtWidgets.QHBoxLayout()
        self.cmbCodec = QtWidgets.QComboBox()
        self.cmbCodec.addItems(["JPEG", "PNG", "WebP", "TIFF-LZW"])
        self.sldQuality = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sldQuality.setRange(1, 100)
        self.sldQuality.setValue(80)
        self.lblQuality = QtWidgets.QLabel("Quality: 80")
        self.btnCompress = QtWidgets.QPushButton("Compress")
        self.btnSaveComp = QtWidgets.QPushButton("Save Compressed")
        self.btnSaveComp.setEnabled(False)
        
        ctrlRow.addWidget(QtWidgets.QLabel("Codec:"))
        ctrlRow.addWidget(self.cmbCodec)
        ctrlRow.addSpacing(20)
        ctrlRow.addWidget(QtWidgets.QLabel("Quality:"))
        ctrlRow.addWidget(self.sldQuality, 1)
        ctrlRow.addWidget(self.lblQuality)
        ctrlRow.addSpacing(20)
        ctrlRow.addWidget(self.btnCompress)
        ctrlRow.addWidget(self.btnSaveComp)
        cLay.addLayout(ctrlRow)
        
        # Three image viewers
        viewersRow = QtWidgets.QHBoxLayout()
        
        origBox = QtWidgets.QGroupBox("Original")
        origBoxLay = QtWidgets.QVBoxLayout(origBox)
        self.viewCompOrig = ScaledImageLabel("—")
        self.viewCompOrig.setMinimumSize(300, 250)
        origBoxLay.addWidget(self.viewCompOrig)
        
        compBox = QtWidgets.QGroupBox("Compressed")
        compBoxLay = QtWidgets.QVBoxLayout(compBox)
        self.viewComp = ScaledImageLabel("—")
        self.viewComp.setMinimumSize(300, 250)
        compBoxLay.addWidget(self.viewComp)
        
        diffBox = QtWidgets.QGroupBox("Difference")
        diffBoxLay = QtWidgets.QVBoxLayout(diffBox)
        self.viewDiff = ScaledImageLabel("—")
        self.viewDiff.setMinimumSize(300, 250)
        diffBoxLay.addWidget(self.viewDiff)
        
        viewersRow.addWidget(origBox)
        viewersRow.addWidget(compBox)
        viewersRow.addWidget(diffBox)
        cLay.addLayout(viewersRow, 2)
        
        # Metrics display
        metricsBox = QtWidgets.QGroupBox("Quality Metrics")
        metricsLay = QtWidgets.QVBoxLayout(metricsBox)
        self.txtMetrics = QtWidgets.QTextEdit()
        self.txtMetrics.setReadOnly(True)
        self.txtMetrics.setMaximumHeight(120)
        self.txtMetrics.setHtml("<p>Compress an image to see metrics</p>")
        metricsLay.addWidget(self.txtMetrics)
        cLay.addWidget(metricsBox)

        splitter.addWidget(self.tabs)
        splitter.setSizes([360, 800])

        # Store compressed data
        self._comp_rgb = None
        self._comp_bytes = b""

        # Connect signals
        self.btnOpen.clicked.connect(self.on_open)
        self.btnSave.clicked.connect(self.on_save)
        self.btnApply.clicked.connect(self.on_apply)
        self.btnApplyNL.clicked.connect(self.on_apply_nl)
        self.btnReset.clicked.connect(self.on_reset)
        self.sldAlpha.valueChanged.connect(lambda v: self.lblAlpha.setText(f"α = {v/100:.2f}"))
        self.sldQuality.valueChanged.connect(lambda v: self.lblQuality.setText(f"Quality: {v}"))
        self.btnCompress.clicked.connect(self.on_compress)
        self.btnSaveComp.clicked.connect(self.on_save_compressed)

        self.setStyleSheet(DARK_QSS)

    def reset_parameters_to_defaults(self):
        """Reset all parameter controls to their default values."""
        self.cboFilter.setCurrentIndex(0)  # Box/Average
        self.spnK.setValue(5)  # Kernel size
        self.dspSigma.setValue(1.0)  # Sigma
        self.spnIter.setValue(1)  # Iterations
        self.cboBorder.setCurrentIndex(0)  # reflect
        self.chkGray.setChecked(False)  # Grayscale processing
        self.sldAlpha.setValue(100)  # Alpha = 1.00

    def on_open(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff)")
        if not path:
            return
        
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            QtWidgets.QMessageBox.critical(self, "Error", "Could not load image.")
            return
        
        self._orig = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._filtered = None
        self._comp_rgb = None
        self._comp_bytes = b""
        
        self.viewOrig.set_image_rgb(self._orig)
        self.viewFilt.clear_image("— Apply a filter —")
        self.viewCompOrig.set_image_rgb(self._orig)
        self.viewComp.clear_image("—")
        self.viewDiff.clear_image("—")
        self.txtMetrics.setHtml("<p>Compress an image to see metrics</p>")
        self.btnSaveComp.setEnabled(False)
        self.tabs.setCurrentWidget(self.tabOrig)

    def on_apply(self):
        if self._orig is None:
            QtWidgets.QMessageBox.information(self, "Info", "Load an image first.")
            return

        params = dict(
            img_rgb=self._orig,
            filter_name=self.cboFilter.currentText(),
            ksize=int(self.spnK.value()),
            sigma=float(self.dspSigma.value()),
            border_mode_str=self.cboBorder.currentText(),
            grayscale_only=self.chkGray.isChecked(),
            iterations=int(self.spnIter.value()),
            unsharp_alpha=float(self.sldAlpha.value())/100.0
        )

        try:
            from linear_filters import apply_linear_filter
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Not implemented",
                "Λείπει το αρχείο ή η συνάρτηση: linear_filters.apply_linear_filter(...)\n\n" + str(e))
            return

        try:
            result = apply_linear_filter(**params)
        except NotImplementedError:
            QtWidgets.QMessageBox.information(self, "TODO", "Υλοποιήστε τη συνάρτηση apply_linear_filter")
            return
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Σφάλμα κατά το φιλτράρισμα:\n{e}")
            return

        self._filtered = result.astype(np.uint8)
        self.viewFilt.set_image_rgb(self._filtered)
        self.tabs.setCurrentWidget(self.tabFilt)

    def on_reset(self):
        if self._orig is not None:
            self.viewOrig.set_image_rgb(self._orig)
        self._filtered = None
        self.viewFilt.clear_image("— Apply a filter —")
        self.tabs.setCurrentWidget(self.tabOrig)
        # Reset all parameters to their default values
        self.reset_parameters_to_defaults()

    def on_save(self):
        if self._filtered is None:
            QtWidgets.QMessageBox.information(self, "Info", "No result to save.")
            return
        
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save", "", "PNG (*.png);;JPEG (*.jpg)")
        if not path:
            return
        
        bgr = cv2.cvtColor(self._filtered, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, bgr)
        QtWidgets.QMessageBox.information(self, "OK", f"Saved: {path}")

    def on_apply_nl(self):
        """Apply non-linear filter"""
        if self._orig is None:
            QtWidgets.QMessageBox.information(self, "Info", "Load an image first.")
            return
        
        kind = self.cboNLFilter.currentText()
        params = {
            'ksize': int(self.spnNLK.value()),
            'bilateral_d': int(self.spnBilateralD.value()),
            'bilateral_sigma_color': float(self.dspSigmaColor.value()),
            'bilateral_sigma_space': float(self.dspSigmaSpace.value()),
            'nlmeans_h': float(self.dspNLMeansH.value())
        }
        
        try:
            result = apply_nonlinear_filter(self._orig, kind, **params)
            self._filtered = result.astype(np.uint8)
            self.viewFilt.set_image_rgb(self._filtered)
            self.tabs.setCurrentWidget(self.tabFilt)
        except NotImplementedError:
            QtWidgets.QMessageBox.information(self, "TODO", 
                "Please implement apply_nonlinear_filter function")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Error applying filter:\n{e}")

    def on_compress(self):
        """Compress image and display results"""
        if self._orig is None:
            QtWidgets.QMessageBox.information(self, "Info", "Load an image first.")
            return
        
        codec = self.cmbCodec.currentText()
        quality = int(self.sldQuality.value())
        
        try:
            comp_rgb, enc_bytes = compress_image(self._orig, codec, quality)
            self._comp_rgb = comp_rgb
            self._comp_bytes = enc_bytes
            
            # Display compressed image
            self.viewComp.set_image_rgb(comp_rgb)
            
            # Compute and display difference
            diff_img = compute_diff_image(self._orig, comp_rgb)
            self.viewDiff.set_image_rgb(diff_img)
            
            # Compute and display metrics
            metrics_html = compute_metrics(self._orig, comp_rgb, enc_bytes, codec)
            self.txtMetrics.setHtml(metrics_html)
            
            self.btnSaveComp.setEnabled(True)
            self.tabs.setCurrentWidget(self.tabComp)
            
        except NotImplementedError:
            QtWidgets.QMessageBox.information(self, "TODO",
                "Please implement compression functions:\n"
                "- compress_image\n- compute_diff_image\n- compute_metrics")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Error during compression:\n{e}")

    def on_save_compressed(self):
        """Save compressed image file"""
        if not self._comp_bytes:
            QtWidgets.QMessageBox.information(self, "Info", "No compressed image to save.")
            return
        
        codec = self.cmbCodec.currentText()
        ext_map = {"JPEG": ".jpg", "PNG": ".png", "WebP": ".webp", "TIFF-LZW": ".tiff"}
        ext = ext_map.get(codec, ".bin")
        
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Compressed", "", f"{codec} (*{ext})")
        if not path:
            return
        
        with open(path, 'wb') as f:
            f.write(self._comp_bytes)
        
        QtWidgets.QMessageBox.information(self, "OK", f"Saved compressed file: {path}")

def main() -> None:
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
    app = QtWidgets.QApplication(sys.argv)

    QtWidgets.QApplication.setStyle("Fusion")

    pal = app.palette()
    for role in (QtGui.QPalette.Window, QtGui.QPalette.Base, QtGui.QPalette.Button):
        pal.setColor(role, QtGui.QColor("#1a1b1e"))
    for role in (QtGui.QPalette.Text, QtGui.QPalette.WindowText, QtGui.QPalette.ButtonText):
        pal.setColor(role, QtGui.QColor("#e6e6e6"))
    app.setPalette(pal)

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
