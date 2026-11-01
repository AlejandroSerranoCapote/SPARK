import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, 
    QLabel, QLineEdit, QCheckBox, QPushButton, QColorDialog, 
    QScrollArea, QWidget, QGridLayout, QComboBox, QSpinBox, 
    QProgressBar, QMessageBox
)

class CompareSetupDialog(QDialog):
    """
    Dialog to setup parameters for comparing kinetics across multiple datasets.
    Allows user to pick target wavelength, normalization, titles, labels, and colors.

    Parameters
    ----------
    wl_min : float
        Minimum available wavelength in the datasets.
    wl_max : float
        Maximum available wavelength in the datasets.
    default_wl : str
        Default target wavelength to pre-fill the input box.
    filenames : list of str
        List of filenames corresponding to the loaded datasets to compare.
    parent : QWidget, optional
        Parent widget for styling inheritance. Defaults to None.
    """
    def __init__(self, wl_min, wl_max, default_wl, filenames, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Compare Kinetics Setup")
        self.setMinimumWidth(400)
        
        # We try to pull the stylesheet from the parent for consistency
        if parent and hasattr(parent, 'styleSheet'):
            self.setStyleSheet(parent.styleSheet())

        self.filenames = filenames
        # Default palette matching matplotlib tab10
        self.default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                               '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        layout = QVBoxLayout(self)

        # --- 1. Target Wavelength ---
        wl_layout = QHBoxLayout()
        wl_layout.addWidget(QLabel(f"Target Wavelength ({wl_min:.1f} - {wl_max:.1f} nm):"))
        self.wl_input = QLineEdit(default_wl)
        wl_layout.addWidget(self.wl_input)
        layout.addLayout(wl_layout)

        # --- 2. Custom Title ---
        title_layout = QHBoxLayout()
        title_layout.addWidget(QLabel("Plot Title:"))
        self.title_input = QLineEdit("Kinetics Comparison")
        title_layout.addWidget(self.title_input)
        layout.addLayout(title_layout)

        # --- 3. Normalization ---
        self.chk_normalize = QCheckBox("Normalize all traces to Max = 1")
        layout.addWidget(self.chk_normalize)

        # --- 4. Dataset Configuration (Scrollable) ---
        layout.addWidget(QLabel("<b>Configure Datasets:</b>"))
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        self.grid = QGridLayout(scroll_widget)
        
        # Headers
        self.grid.addWidget(QLabel("Include"), 0, 0)
        self.grid.addWidget(QLabel("Legend Label"), 0, 1)
        self.grid.addWidget(QLabel("Color"), 0, 2)

        self.dataset_controls = []
        for i, fname in enumerate(filenames):
            chk = QCheckBox()
            chk.setChecked(True)
            
            label_input = QLineEdit(fname)
            
            color_btn = QPushButton()
            color = self.default_colors[i % len(self.default_colors)]
            color_btn.setStyleSheet(f"background-color: {color}; border: 1px solid gray; width: 25px; height: 15px;")
            color_btn.setProperty("color_val", color)
            
            # Lambda needs default argument to capture current button in loop
            color_btn.clicked.connect(lambda checked, btn=color_btn: self.choose_color(btn))

            self.grid.addWidget(chk, i+1, 0)
            self.grid.addWidget(label_input, i+1, 1)
            self.grid.addWidget(color_btn, i+1, 2)

            self.dataset_controls.append((chk, label_input, color_btn, i))

        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

        # --- 5. Buttons ---
        btn_layout = QHBoxLayout()
        btn_ok = QPushButton("Plot Comparison")
        btn_ok.clicked.connect(self.accept)
        btn_ok.setStyleSheet("background-color: #0078D7; color: white;")
        
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        
        btn_layout.addWidget(btn_ok)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)

    def choose_color(self, btn):
        """
        Opens a color picker dialog and updates the button's stored hex color.
        
        Parameters
        ----------
        btn : QPushButton
            The button widget representing the color of the selected dataset.
        """
        color = QColorDialog.getColor()
        if color.isValid():
            hex_color = color.name()
            btn.setStyleSheet(f"background-color: {hex_color}; border: 1px solid gray; width: 25px; height: 15px;")
            btn.setProperty("color_val", hex_color)

    def get_data(self):
        """
        Retrieves and packages the validated data from the dialog.

        Returns
        -------
        tuple
            A tuple containing:
            - target_wl (float or None): Target wavelength.
            - normalize (bool): Normalization flag.
            - custom_title (str): Title for the plot.
            - custom_labels (list of str): Legend labels for active datasets.
            - ordered_indices (list of int): Original indices of the selected datasets.
            - custom_colors (list of str): Hex colors for the active datasets.
        """
        try:
            target_wl = float(self.wl_input.text())
        except ValueError:
            target_wl = None

        normalize = self.chk_normalize.isChecked()
        custom_title = self.title_input.text()

        custom_labels = []
        ordered_indices = []
        custom_colors = []

        # Iterate over checked datasets
        for chk, label_input, color_btn, orig_idx in self.dataset_controls:
            if chk.isChecked():
                custom_labels.append(label_input.text())
                ordered_indices.append(orig_idx)
                custom_colors.append(color_btn.property("color_val"))

        return target_wl, normalize, custom_title, custom_labels, ordered_indices, custom_colors

class DiffMapDialog(QDialog):
    """
    Dialog to select two datasets and calculate their 2D difference map (A - B).

    Parameters
    ----------
    filenames : list of str
        List of filenames for the available datasets to subtract.
    parent : QWidget, optional
        Parent widget. Defaults to None.
    """
    def __init__(self, filenames, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Difference 2D Map")
        if parent and hasattr(parent, 'styleSheet'):
            self.setStyleSheet(parent.styleSheet())
        
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Select datasets to subtract (A - B):"))
        
        form = QFormLayout()
        self.combo_A = QComboBox()
        self.combo_A.addItems(filenames)
        
        self.combo_B = QComboBox()
        self.combo_B.addItems(filenames)
        if len(filenames) > 1:
            self.combo_B.setCurrentIndex(1)
        
        form.addRow("Dataset A (Base):", self.combo_A)
        form.addRow("Dataset B (Subtracted):", self.combo_B)
        layout.addLayout(form)
        
        btns = QHBoxLayout()
        btn_ok = QPushButton("Calculate Difference")
        btn_ok.setStyleSheet("background-color: #3B82F6; color: white; font-weight: bold;")
        btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        
        btns.addWidget(btn_ok)
        btns.addWidget(btn_cancel)
        layout.addLayout(btns)
        
    def get_indices(self):
        """
        Retrieves the selected indices from the dialog comboboxes.

        Returns
        -------
        tuple
            (index_A, index_B) representing the two selected datasets.
        """
        return self.combo_A.currentIndex(), self.combo_B.currentIndex()
    
    
class ParameterIdentifiabilityDialog(QDialog):
    """
    Dialog to analyze the identifiability of a kinetic parameter using 
    profile likelihood instead of relying purely on covariance-based errors.

    Parameters
    ----------
    panel : object
        Reference to the GlobalFitPanel (calculation engine) executing the fits.
    parent : QWidget, optional
        Parent widget. Defaults to None.
    """
    def __init__(self, panel, parent=None):
        super().__init__(parent)
        self.panel = panel  # Reference to GlobalFitPanel
        self.setWindowTitle("Parameter Identifiability Analysis")
        self.resize(700, 550)
        if parent and hasattr(parent, 'styleSheet'):
            self.setStyleSheet(parent.styleSheet())

        layout = QVBoxLayout(self)

        ctrl_layout = QHBoxLayout()
        ctrl_layout.addWidget(QLabel("Parameter:"))
        self.combo_param = QComboBox()
        self._populate_param_combo()
        ctrl_layout.addWidget(self.combo_param, stretch=2)

        ctrl_layout.addWidget(QLabel("Confidence:"))
        self.combo_confidence = QComboBox()
        self.combo_confidence.addItems(["68%", "90%", "95%", "99%"])
        self.combo_confidence.setCurrentIndex(2)
        ctrl_layout.addWidget(self.combo_confidence)

        ctrl_layout.addWidget(QLabel("Steps:"))
        self.spin_steps = QSpinBox()
        self.spin_steps.setRange(5, 50)
        self.spin_steps.setValue(15)
        ctrl_layout.addWidget(self.spin_steps)

        self.btn_run = QPushButton("Run Analysis")
        self.btn_run.setStyleSheet("background-color: #10B981; color: white; font-weight: bold;")
        self.btn_run.clicked.connect(self.run_analysis)
        ctrl_layout.addWidget(self.btn_run)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setStyleSheet("background-color: #EF4444; color: white;")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel_analysis)
        ctrl_layout.addWidget(self.btn_cancel)

        layout.addLayout(ctrl_layout)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        self.lbl_result = QLabel("Select a parameter and click 'Run Analysis'.")
        self.lbl_result.setWordWrap(True)
        self.lbl_result.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.lbl_result)

        self.fig = Figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        layout.addWidget(self.canvas)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)

    def _populate_param_combo(self):
        """Populates the combo box with the labels of all non-linear kinetic parameters."""
        num_kin = self.panel._get_num_kinetic_params()
        self.combo_param.clear()
        for i in range(num_kin):
            label = self.panel._get_kinetic_param_label(i)
            self.combo_param.addItem(f"[{i}] {label}", userData=i)

    def cancel_analysis(self):
        """Signals the global fitting engine to abort the ongoing likelihood analysis."""
        self.panel._abort_fit = True
        self.lbl_result.setText("Cancelling...")

    def run_analysis(self):
        """
        Executes the profile likelihood computation on the selected parameter,
        updating the progress bar and plotting the final Chi-squared landscape.
        """
        param_idx = self.combo_param.currentData()
        if param_idx is None:
            return

        confidence = float(self.combo_confidence.currentText().strip('%')) / 100.0
        n_steps = self.spin_steps.value()

        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress.setValue(0)
        self.lbl_result.setText("Running profile likelihood... this may take a moment.")
        QApplication.processEvents()

        def on_progress(done, total):
            self.progress.setValue(int(100 * done / total))
            QApplication.processEvents()

        self.panel._abort_fit = False
        try:
            result = self.panel.compute_profile_likelihood(
                param_idx, n_steps=n_steps, confidence=confidence,
                progress_callback=on_progress
            )
        except InterruptedError:
            self.lbl_result.setText("Analysis cancelled.")
            self.progress.setValue(0)
            self.btn_run.setEnabled(True)
            self.btn_cancel.setEnabled(False)
            return
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Identifiability analysis failed:\n{e}")
            self.btn_run.setEnabled(True)
            self.btn_cancel.setEnabled(False)
            return
        finally:
            self.panel._abort_fit = False

        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress.setValue(100)

        if result is None or result['lower_bound'] is None:
            self.lbl_result.setText(
                "Could not bracket the confidence interval within the search range.\n"
                "Try increasing the number of steps or widening the range."
            )
            return

        best, lo, hi = result['best_value'], result['lower_bound'], result['upper_bound']
        self.lbl_result.setText(
            f"{result['label']}: best = {best:.4g}   "
            f"CI({int(confidence*100)}%) = [{lo:.4g}, {hi:.4g}]   "
            f"(-{best-lo:.4g} / +{hi-best:.4g})"
        )
        self._plot_result(result)

    def _plot_result(self, result):
        """
        Draws the resulting delta-Chi-squared curve, best-fit line, and confidence 
        threshold on the Matplotlib canvas.

        Parameters
        ----------
        result : dict
            Dictionary returned by `compute_profile_likelihood` containing the evaluated grid.
        """
        self.ax.clear()
        grid, chi2, chi2_min = result['grid'], result['chi2'], result['chi2_min']

        self.ax.plot(grid, chi2 - chi2_min, 'o-', color='#3C5488', label=r'$\Delta\chi^2$')
        self.ax.axhline(result['delta_threshold'], color='red', ls='--', label="Threshold")
        self.ax.axvline(result['best_value'], color='gray', ls=':', label='Best fit')

        if result['lower_bound'] is not None:
            self.ax.axvspan(result['lower_bound'], result['upper_bound'],
                             color='#10B981', alpha=0.15, label='Confidence interval')

        self.ax.set_xlabel(result['label'])
        self.ax.set_ylabel(r'$\Delta\chi^2$')
        self.ax.set_title(f"Profile Likelihood: {result['label']}")
        self.ax.legend(frameon=True, fontsize=9)
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.canvas.draw_idle()