import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QGroupBox,
    QComboBox, QDoubleSpinBox, QCheckBox, QLineEdit, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog, QMessageBox,
    QSpinBox
)


class PaperPlotterWindow(QDialog):
    """
    Advanced Drag & Drop Publication-Quality Plotter for Kinetics Traces.

    Works 100% autonomously. Allows users to drop exported .txt trace files, 
    customize dimensions, palettes, and crop the Y-axis (ΔA) with high precision 
    for publication-ready exports.

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget. If the parent has a stylesheet, it will be inherited. 
        Defaults to None.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Publication-Quality Kinetics Plotter (Standalone Mode)")
        self.resize(980, 680)
        self.setAcceptDrops(True) 
        
        if parent and hasattr(parent, 'styleSheet'):
            self.setStyleSheet(parent.styleSheet())
            
        self.file_data = [] 
        self.nature_colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2', '#DC0000']
        
        self.initUI()
        
    def initUI(self):
        """Initializes the UI layout, including drag & drop zone, controls, and canvas."""
        # 1. Main layout is HORIZONTAL (Side by side)
        main_layout = QHBoxLayout(self)
        
        # 2. Left Panel (Vertical): Drop Zone + Controls
        left_layout = QVBoxLayout()
        
        self.drop_label = QLabel("DRAG & DROP YOUR KINETIC .TXT FILES HERE")
        self.drop_label.setAlignment(Qt.AlignCenter)
        self.drop_label.setFrameShape(QLabel.StyledPanel)
        self.drop_label.setFrameShadow(QLabel.Sunken)
        self.drop_label.setMinimumHeight(70)
        self.drop_label.setStyleSheet("""
            QLabel {
                background-color: #2D3238;
                color: #A0AAB5;
                border: 2px dashed #3C5488;
                border-radius: 6px;
                font-weight: bold;
                font-size: 11pt;
            }
        """)
        left_layout.addWidget(self.drop_label)
        
        ctrl_group = QGroupBox("Plots")
        ctrl_form = QFormLayout(ctrl_group)
        
        self.combo_palette = QComboBox()
        self.combo_palette.addItems([
            "Scientific (Nature)", 
            "Qualitative (Tab10)", 
            "Vibrant (Set1)", 
            "Sequential (Viridis)", 
            "Sequential (Plasma)",
            "Cool / Warm"
        ])
        self.combo_palette.currentIndexChanged.connect(self.replotted)
        ctrl_form.addRow("Color Palette:", self.combo_palette)
        
        self.spin_width = QDoubleSpinBox()
        self.spin_width.setRange(3.0, 15.0)
        self.spin_width.setValue(7.0)  
        self.spin_width.setSingleStep(0.5)
        self.spin_width.setSuffix(" in (Width)")
        self.spin_width.valueChanged.connect(self.update_fig_size)
        ctrl_form.addRow("Figure Width:", self.spin_width)
        
        self.spin_height = QDoubleSpinBox()
        self.spin_height.setRange(2.0, 10.0)
        self.spin_height.setValue(4.5)  
        self.spin_height.setSingleStep(0.5)
        self.spin_height.setSuffix(" in (Height)")
        self.spin_height.valueChanged.connect(self.update_fig_size)
        ctrl_form.addRow("Figure Height:", self.spin_height)
        
        self.combo_scale = QComboBox()
        self.combo_scale.addItems(["Linear", "SymLog"])
        self.combo_scale.currentIndexChanged.connect(self.replotted)
        ctrl_form.addRow("X-Axis Scale:", self.combo_scale)
        
        self.spin_thresh = QDoubleSpinBox()
        self.spin_thresh.setRange(0.01, 10.0)
        self.spin_thresh.setValue(1.0)
        self.spin_thresh.setSingleStep(0.5)
        self.spin_thresh.valueChanged.connect(self.replotted)
        ctrl_form.addRow("Linthresh (ps):", self.spin_thresh)
        
        self.chk_norm = QCheckBox("Normalize individual amplitudes")
        self.chk_norm.stateChanged.connect(self.replotted)
        ctrl_form.addRow(self.chk_norm)
        
        self.chk_no_negatives = QCheckBox("Hide negative time (< 0)")
        self.chk_no_negatives.setChecked(True) 
        self.chk_no_negatives.stateChanged.connect(self.replotted)
        ctrl_form.addRow(self.chk_no_negatives)
        
        self.line_title = QLineEdit("")
        self.line_title.setPlaceholderText("Optional Plot Title")
        self.line_title.textChanged.connect(self.replotted)
        ctrl_form.addRow("Plot Title:", self.line_title)

        self.line_xlabel = QLineEdit("Time Delay / ps")
        self.line_xlabel.textChanged.connect(self.replotted)
        ctrl_form.addRow("X-Axis Label:", self.line_xlabel)

        self.line_ylabel = QLineEdit("ΔA (a.u.)")
        self.line_ylabel.textChanged.connect(self.replotted)
        ctrl_form.addRow("Y-Axis Label:", self.line_ylabel)
        
        self.chk_auto_y = QCheckBox("Automatic Y-Axis (ΔA)")
        self.chk_auto_y.setChecked(True)
        self.chk_auto_y.stateChanged.connect(self.toggle_y_inputs)
        ctrl_form.addRow(self.chk_auto_y)
        
        self.spin_ymin = QDoubleSpinBox()
        self.spin_ymin.setRange(-10.0, 10.0)
        self.spin_ymin.setValue(-0.02)
        self.spin_ymin.setSingleStep(0.005)
        self.spin_ymin.setDecimals(3)
        self.spin_ymin.setEnabled(False)
        self.spin_ymin.valueChanged.connect(self.replotted)
        ctrl_form.addRow("Y Min Crop:", self.spin_ymin)
        
        self.spin_ymax = QDoubleSpinBox()
        self.spin_ymax.setRange(-10.0, 10.0)
        self.spin_ymax.setValue(0.20)
        self.spin_ymax.setSingleStep(0.005)
        self.spin_ymax.setDecimals(3)
        self.spin_ymax.setEnabled(False)
        self.spin_ymax.valueChanged.connect(self.replotted)
        ctrl_form.addRow("Y Max Crop:", self.spin_ymax)
        
        self.btn_clear = QPushButton("Clear Plot")
        self.btn_clear.clicked.connect(self.clear_data)
        ctrl_form.addRow(self.btn_clear)
        
        
        self.table_labels = QTableWidget(0, 2)
        self.table_labels.setHorizontalHeaderLabels(["Original", "Custom Legend"])
        self.table_labels.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_labels.setMinimumHeight(120)
        
        self.table_labels.itemChanged.connect(self.replotted)
        ctrl_form.addRow(self.table_labels)
        
        
        self.btn_export_fig = QPushButton("Export Figure (600 DPI)")
        self.btn_export_fig.clicked.connect(self.export_figure)
        self.btn_export_fig.setStyleSheet("background-color: #4A8C4A; color: white; font-weight: bold;")
        ctrl_form.addRow(self.btn_export_fig)
        
        # Add control group to left layout
        left_layout.addWidget(ctrl_group)
        
        # Optional stretch to push controls upwards
        left_layout.addStretch()

        # 3. Right Panel (Vertical): Toolbar + Plot Canvas
        right_layout = QVBoxLayout()
        
        self.fig = Figure(figsize=(self.spin_width.value(), self.spin_height.value()), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas)
        
        # 4. Assemble main layout
        main_layout.addLayout(left_layout, stretch=1)
        main_layout.addLayout(right_layout, stretch=2)
        
        self.ax = self.fig.add_subplot(111)
        self.setup_paper_style()
        
    def update_labels_table(self):
        """Populates the table with loaded files to allow editing their legends."""
        self.table_labels.blockSignals(True) # Prevent infinite redraw loops during construction
        self.table_labels.setRowCount(len(self.file_data))
        
        for i, data in enumerate(self.file_data):
            wl = data['wl']
            default_label = f"{wl} nm" if wl > 0 else data['filename']
            
            # Keep modified text if the user already changed it
            existing_item = self.table_labels.item(i, 1)
            current_custom = existing_item.text() if existing_item else default_label
            
            # Column 1: Original (Read-only)
            item_orig = QTableWidgetItem(default_label)
            item_orig.setFlags(item_orig.flags() & ~Qt.ItemIsEditable) 
            
            # Column 2: Editable
            item_custom = QTableWidgetItem(current_custom)
            
            self.table_labels.setItem(i, 0, item_orig)
            self.table_labels.setItem(i, 1, item_custom)
            
        self.table_labels.blockSignals(False)
        
    def toggle_y_inputs(self):
        """Enables or disables the Y-axis crop spinboxes based on the auto mode."""
        auto_mode = self.chk_auto_y.isChecked()
        self.spin_ymin.setEnabled(not auto_mode)
        self.spin_ymax.setEnabled(not auto_mode)
        self.replotted()

    def update_fig_size(self):
        """Updates the figure dimensions in inches."""
        w = self.spin_width.value()
        h = self.spin_height.value()
        self.fig.set_size_inches(w, h)
        self.canvas.draw_idle() 
        
    def setup_paper_style(self):
        """Applies scientific formatting and styling to the Matplotlib axes."""
        self.ax.clear()
        self.ax.tick_params(direction='in', top=True, right=True, labelsize=11, width=1.2, length=6)
        for spine in self.ax.spines.values():
            spine.set_linewidth(1.2)
            
        # Try to retrieve text from the GUI. If not created yet, use defaults.
        x_label = self.line_xlabel.text() if hasattr(self, 'line_xlabel') else "Time Delay / ps"
        y_label = self.line_ylabel.text() if hasattr(self, 'line_ylabel') else "ΔA (a.u.)"
        title = self.line_title.text() if hasattr(self, 'line_title') else ""

        self.ax.set_xlabel(x_label, fontsize=13, fontname="Arial", fontweight='bold')
        self.ax.set_ylabel(y_label, fontsize=13, fontname="Arial", fontweight='bold')
        
        # Only draw title if user wrote something
        if title.strip():
            self.ax.set_title(title, fontsize=14, fontname="Arial", fontweight='bold')
            
        self.ax.grid(False)
        
    def dragEnterEvent(self, event):
        """Accepts files dragged into the window."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            
    def dropEvent(self, event):
        """Handles dropped files, parsing them if they are text files."""
        files_added = 0
        for url in event.mimeData().urls():
            file_path = str(url.toLocalFile())
            if file_path.lower().endswith('.txt'):
                if self.parse_trace_file(file_path):
                    files_added += 1
        if files_added > 0:
            self.file_data.sort(key=lambda x: x['wl'])
            self.update_labels_table()  
            self.replotted()
            
    def parse_trace_file(self, path):
        """
        Parses a .txt trace file to extract wavelengths, delays, and amplitudes.

        Parameters
        ----------
        path : str
            File path of the text trace.

        Returns
        -------
        bool
            True if parsing was successful, False otherwise.
        """
        try:
            wl_val = None
            filename = os.path.basename(path)
            
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if "wavelength" in line.lower():
                        match_h = re.search(r"([\d.]+)", line)
                        if match_h:
                            wl_val = float(match_h.group(1))
                            break
            
            if wl_val is None:
                match_fn = re.search(r"([\d.]+)\s*nm", filename, re.IGNORECASE)
                if match_fn:
                    wl_val = float(match_fn.group(1))
            
            if wl_val is None:
                wl_val = 0.0 
                
            wl_rounded = int(round(wl_val, -1))
            
            raw_data = np.loadtxt(path)
            if raw_data.ndim != 2 or raw_data.shape[1] < 2:
                return False 
                
            td_exp = raw_data[:, 0]
            y_exp = raw_data[:, 1]
            
            # Quitar los NaNs de los datos experimentales
            valid_exp = ~np.isnan(td_exp)
            td_exp = td_exp[valid_exp]
            y_exp = y_exp[valid_exp]
            
            td_fit = None
            y_fit = None
            
            if raw_data.shape[1] >= 4:
                # Nuevo formato: leemos las columnas 3 y 4 y quitamos NaNs
                td_fit = raw_data[:, 2]
                y_fit = raw_data[:, 3]
                valid_fit = ~np.isnan(td_fit)
                td_fit = td_fit[valid_fit]
                y_fit = y_fit[valid_fit]
            elif raw_data.shape[1] == 3:
                # Retención de compatibilidad con tus archivos guardados antiguos
                td_fit = td_exp
                y_fit = raw_data[:, 2][valid_exp]
            
            self.file_data.append({
                'wl': wl_rounded,
                'td': td_exp,
                'exp': y_exp,
                'td_fit': td_fit,
                'fit': y_fit,
                'filename': filename
            })
            return True
        except Exception as e:
            print(f"Error parsing: {e}")
            return False
            
    def replotted(self):
        """Redraws the plot with the latest data, styling, and UI parameter states."""
        self.setup_paper_style()
        if not self.file_data:
            self.canvas.draw_idle()
            return
            
        is_symlog = self.combo_scale.currentText() == "SymLog"
        norm_indep = self.chk_norm.isChecked()
        hide_negatives = self.chk_no_negatives.isChecked()
        palette_choice = self.combo_palette.currentText()
        
        N = len(self.file_data)
        
        generated_colors = []
        if palette_choice == "Scientific (Nature)":
            generated_colors = [self.nature_colors[i % len(self.nature_colors)] for i in range(N)]
        else:
            cmap_map = {
                "Qualitative (Tab10)": "tab10",
                "Vibrant (Set1)": "Set1",
                "Sequential (Viridis)": "viridis",
                "Sequential (Plasma)": "plasma",
                "Cool / Warm": "coolwarm"
            }
            cmap = plt.get_cmap(cmap_map[palette_choice])
            
            if palette_choice in ["Qualitative (Tab10)", "Vibrant (Set1)"]:
                generated_colors = [cmap(i % cmap.N) for i in range(N)]
            else:
                if N == 1:
                    generated_colors = [cmap(0.5)]
                else:
                    generated_colors = [cmap(val) for val in np.linspace(0.0, 0.85, N)]

        max_td_found = -1e10
        min_td_found = 1e10

        for i, data in enumerate(self.file_data):
            color = generated_colors[i]
            td_exp = data['td']
            y_exp = data['exp']
            td_fit = data['td_fit']
            y_fit = data['fit']
            wl = data['wl']
                
            item = self.table_labels.item(i, 1)
            if item and item.text().strip():
                label_text = item.text()
            else:
                label_text = f"{wl} nm" if wl > 0 else data['filename']
                      
            max_td_found = max(max_td_found, np.max(td_exp))
            min_td_found = min(min_td_found, np.min(td_exp))
            
            if norm_indep:
                max_val = max(np.max(np.abs(y_fit)), 1e-10)
                y_exp = y_exp / max_val
                if y_fit is not None:
                    y_fit = y_fit / max_val
            
            # Dibujamos el Scatter con los puntos experimentales dispersos
            self.ax.plot(td_exp, y_exp, 'o', color=color, markersize=4, alpha=0.4, 
                         markeredgewidth=1.0, label=label_text)
            
            # Dibujamos la línea continua con el eje High-Res
            if y_fit is not None and td_fit is not None:
                self.ax.plot(td_fit, y_fit, '-', color=color, linewidth=2.0)
                
        if is_symlog:
            self.ax.set_xscale('symlog', linthresh=self.spin_thresh.value())
            
        if hide_negatives:
            self.ax.set_xlim(0.0, max_td_found)
        else:
            self.ax.set_xlim(min_td_found, max_td_found)
            
        # --- CUSTOM Y-AXIS CROP APPLICATION ---
        if not self.chk_auto_y.isChecked():
            self.ax.set_ylim(self.spin_ymin.value(), self.spin_ymax.value())
        # ---------------------------------------------------
            
        self.ax.legend(frameon=True, framealpha=0.0, edgecolor='none', fontsize=10, loc='best')
        self.fig.tight_layout()
        self.canvas.draw_idle()
        
    def clear_data(self):
        """Clears all loaded files and resets the plot canvas."""
        self.file_data = []
        self.table_labels.setRowCount(0)
        self.replotted()
        
    def export_figure(self):
        """Exports the Matplotlib figure to PDF, PNG, or TIFF format at 600 DPI."""
        if not self.file_data:
            return
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Publication Figure", "", 
            "PDF Vector Graphic (*.pdf);;PNG High-Resolution Image (*.png);;TIFF Image (*.tiff)"
        )
        if save_path:
            self.update_fig_size()
            self.fig.savefig(save_path, dpi=600, bbox_inches='tight')
            QMessageBox.information(self, "Export Successful", "High-resolution plot exported successfully with selected dimensions.")            


class SASDASPlotterWindow(QDialog):
    """
    Advanced Drag & Drop Publication-Quality Plotter for SAS/DAS Spectra.

    Allows researchers to visualize and export spectra matrices with automated 
    tau label detection and high-quality customization options.

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget. Defaults to None.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Publication-Quality SAS/DAS Plotter")
        self.resize(980, 680)
        self.setAcceptDrops(True) 
        
        if parent and hasattr(parent, 'styleSheet'):
            self.setStyleSheet(parent.styleSheet())
            
        self.spectra_data = [] 
        self.nature_colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2', '#DC0000']
        
        self.initUI()
        
    def initUI(self):
        """Initializes the UI layout, including drag & drop zone, controls, and canvas."""
        # 1. HORIZONTAL Main Layout
        main_layout = QHBoxLayout(self)
        
        # 2. Left Panel
        left_layout = QVBoxLayout()
        
        self.drop_label = QLabel("DRAG & DROP YOUR SPECTRA FILES (DAS / SAS)")
        self.drop_label.setAlignment(Qt.AlignCenter)
        self.drop_label.setFrameShape(QLabel.StyledPanel)
        self.drop_label.setFrameShadow(QLabel.Sunken)
        self.drop_label.setMinimumHeight(70)
        self.drop_label.setStyleSheet("""
            QLabel {
                background-color: #2D3238;
                color: #A0AAB5;
                border: 2px dashed #00A087;
                border-radius: 6px;
                font-weight: bold;
                font-size: 11pt;
            }
        """)
        left_layout.addWidget(self.drop_label)
        
        ctrl_group = QGroupBox("Plots")
        ctrl_form = QFormLayout(ctrl_group)
        
        # Original color and size controls
        self.combo_palette = QComboBox()
        self.combo_palette.addItems([
            "Scientific (Nature)", "Qualitative (Tab10)", "Vibrant (Set1)", 
            "Sequential (Viridis)", "Sequential (Plasma)", "Cool / Warm"
        ])
        self.combo_palette.currentIndexChanged.connect(self.replotted)
        ctrl_form.addRow("Colour palette:", self.combo_palette)
        
        self.spin_ms = QSpinBox()
        self.spin_ms.setRange(0, 15)
        self.spin_ms.setValue(4)  
        self.spin_ms.setSuffix(" px")
        self.spin_ms.valueChanged.connect(self.replotted)
        ctrl_form.addRow("Point width:", self.spin_ms)
        
        self.spin_cap = QDoubleSpinBox()
        self.spin_cap.setRange(0.0, 15.0)
        self.spin_cap.setValue(3.0)  
        self.spin_cap.setSingleStep(0.5)
        self.spin_cap.setSuffix(" pt (capsize)")
        self.spin_cap.valueChanged.connect(self.replotted)
        ctrl_form.addRow("capsize width:", self.spin_cap)
        
        self.spin_width = QDoubleSpinBox()
        self.spin_width.setRange(3.0, 15.0); self.spin_width.setValue(6.5); self.spin_width.setSingleStep(0.5)
        self.spin_width.setSuffix(" in (Width)")
        self.spin_width.valueChanged.connect(self.update_fig_size)
        ctrl_form.addRow("Figure width:", self.spin_width)
        
        self.spin_height = QDoubleSpinBox()
        self.spin_height.setRange(2.0, 10.0); self.spin_height.setValue(4.5); self.spin_height.setSingleStep(0.5)
        self.spin_height.setSuffix(" in (Height)")
        self.spin_height.valueChanged.connect(self.update_fig_size)
        ctrl_form.addRow("Figure height:", self.spin_height)
        
        # --- NEW: Custom Texts ---
        self.line_title = QLineEdit("")
        self.line_title.setPlaceholderText("Optional Plot Title")
        self.line_title.textChanged.connect(self.replotted)
        ctrl_form.addRow("Plot Title:", self.line_title)

        self.line_xlabel = QLineEdit("Wavelength / nm")
        self.line_xlabel.textChanged.connect(self.replotted)
        ctrl_form.addRow("X-Axis Label:", self.line_xlabel)

        self.line_ylabel = QLineEdit("Amplitude / a.u.")
        self.line_ylabel.textChanged.connect(self.replotted)
        ctrl_form.addRow("Y-Axis Label:", self.line_ylabel)
        # ------------------------------------
        
        self.chk_auto_axes = QCheckBox("Auto limits axis")
        self.chk_auto_axes.setChecked(True)
        self.chk_auto_axes.stateChanged.connect(self.toggle_axes_inputs)
        ctrl_form.addRow(self.chk_auto_axes)
        
        self.spin_xmin = QSpinBox(); self.spin_xmin.setRange(200, 1500); self.spin_xmin.setValue(300)
        self.spin_xmin.setSuffix(" nm (X Min)"); self.spin_xmin.setEnabled(False)
        self.spin_xmin.valueChanged.connect(self.replotted)
        ctrl_form.addRow("Crop X Min:", self.spin_xmin)
        
        self.spin_xmax = QSpinBox(); self.spin_xmax.setRange(200, 1500); self.spin_xmax.setValue(800)
        self.spin_xmax.setSuffix(" nm (X Max)"); self.spin_xmax.setEnabled(False)
        self.spin_xmax.valueChanged.connect(self.replotted)
        ctrl_form.addRow("Crop X Max:", self.spin_xmax)
        
        self.spin_ymin = QDoubleSpinBox(); self.spin_ymin.setRange(-10.0, 10.0); self.spin_ymin.setValue(-0.10)
        self.spin_ymin.setSingleStep(0.01); self.spin_ymin.setDecimals(3); self.spin_ymin.setEnabled(False)
        self.spin_ymin.valueChanged.connect(self.replotted)
        ctrl_form.addRow("Crop Y Min:", self.spin_ymin)
        
        self.spin_ymax = QDoubleSpinBox(); self.spin_ymax.setRange(-10.0, 10.0); self.spin_ymax.setValue(1.0)
        self.spin_ymax.setSingleStep(0.05); self.spin_ymax.setDecimals(3); self.spin_ymax.setEnabled(False)
        self.spin_ymax.valueChanged.connect(self.replotted)
        ctrl_form.addRow("Crop Y Max:", self.spin_ymax)
        
        self.btn_clear = QPushButton("Clean plot")
        self.btn_clear.clicked.connect(self.clear_data)
        ctrl_form.addRow(self.btn_clear)

        # --- NEW: Legend Table ---
        self.table_labels = QTableWidget(0, 2)
        self.table_labels.setHorizontalHeaderLabels(["Original", "Custom Legend"])
        self.table_labels.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_labels.setMinimumHeight(120)
        self.table_labels.itemChanged.connect(self.replotted)
        ctrl_form.addRow(self.table_labels)
        # --------------------------------
        
        self.btn_export_fig = QPushButton("Export spectra (600 DPI)")
        self.btn_export_fig.clicked.connect(self.export_figure)
        self.btn_export_fig.setStyleSheet("background-color: #4A8C4A; color: white; font-weight: bold;")
        ctrl_form.addRow(self.btn_export_fig)
        
        left_layout.addWidget(ctrl_group)
        left_layout.addStretch()
        
        # 3. Right Panel (Plot)
        right_layout = QVBoxLayout()
        self.fig = Figure(figsize=(self.spin_width.value(), self.spin_height.value()), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas)
        
        main_layout.addLayout(left_layout, stretch=1)
        main_layout.addLayout(right_layout, stretch=2)
        
        self.ax = self.fig.add_subplot(111)
        self.setup_paper_style()
        
    def toggle_axes_inputs(self):
        """Enables or disables manual axis limits when the auto checkbox is toggled."""
        state = not self.chk_auto_axes.isChecked()
        self.spin_xmin.setEnabled(state)
        self.spin_xmax.setEnabled(state)
        self.spin_ymin.setEnabled(state)
        self.spin_ymax.setEnabled(state)
        self.replotted()

    def update_fig_size(self):
        """Updates the physical dimensions of the Matplotlib figure."""
        self.fig.set_size_inches(self.spin_width.value(), self.spin_height.value())
        self.canvas.draw_idle() 
        
    def setup_paper_style(self):
        """Applies scientific formatting and styling to the Matplotlib axes."""
        self.ax.clear()
        self.ax.tick_params(direction='in', top=True, right=True, labelsize=11, width=1.2, length=6)
        for spine in self.ax.spines.values():
            spine.set_linewidth(1.2)
            
        x_label = self.line_xlabel.text() if hasattr(self, 'line_xlabel') else "Wavelength / nm"
        y_label = self.line_ylabel.text() if hasattr(self, 'line_ylabel') else "Amplitude / a.u."
        title = self.line_title.text() if hasattr(self, 'line_title') else ""

        self.ax.set_xlabel(x_label, fontsize=13, fontname="Arial", fontweight='bold')
        self.ax.set_ylabel(y_label, fontsize=13, fontname="Arial", fontweight='bold')
        
        if title.strip():
            self.ax.set_title(title, fontsize=14, fontname="Arial", fontweight='bold')
            
        self.ax.axhline(0, color='#7F7F7F', linestyle='--', linewidth=1.0, zorder=1)
        self.ax.grid(False)
    
    def update_labels_table(self):
        """Populates the table with the loaded spectra to allow editing their legends."""
        self.table_labels.blockSignals(True)
        self.table_labels.setRowCount(len(self.spectra_data))
        
        for i, data in enumerate(self.spectra_data):
            default_label = data['label']
            
            existing_item = self.table_labels.item(i, 1)
            current_custom = existing_item.text() if existing_item else default_label
            
            item_orig = QTableWidgetItem(default_label)
            item_orig.setFlags(item_orig.flags() & ~Qt.ItemIsEditable) 
            
            item_custom = QTableWidgetItem(current_custom)
            
            self.table_labels.setItem(i, 0, item_orig)
            self.table_labels.setItem(i, 1, item_custom)
            
        self.table_labels.blockSignals(False)
        
    def dragEnterEvent(self, event):
        """Accepts files dragged into the window."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            
    def dropEvent(self, event):
        """Handles dropped files, parsing them if they are text files."""
        files_added = 0
        for url in event.mimeData().urls():
            file_path = str(url.toLocalFile())
            if file_path.lower().endswith('.txt'):
                if self.parse_spectra_file(file_path):
                    files_added += 1
        if files_added > 0:
            self.update_labels_table()
            self.replotted()
            
    def parse_spectra_file(self, path):
        """
        Parses a .txt spectra file (DAS or SAS) to extract wavelengths, 
        amplitudes, errors, and metadata (lifetimes).

        Parameters
        ----------
        path : str
            File path of the text spectra.

        Returns
        -------
        bool
            True if parsing was successful, False otherwise.
        """
        try:
            taus_dict = {}
            headers = None
            data_lines = []
            
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line_str = line.strip()
                    if not line_str:
                        continue
                    
                    if line_str.startswith('#'):
                        matches = re.findall(r"tau(\d+)=([\d.e+-]+)\+-([\d.e+-]+)", line_str)
                        for m in matches:
                            idx_t = int(m[0])
                            val_t = float(m[1])
                            err_t = float(m[2])
                            taus_dict[idx_t] = (val_t, err_t)
                        continue
                    
                    if headers is None:
                        if '\t' in line_str:
                            headers = line_str.split('\t')
                        else:
                            headers = re.split(r'\s+', line_str)
                    else:
                        data_lines.append(line_str)
            
            if not data_lines:
                return False
                
            raw_data = np.loadtxt(data_lines)
            if raw_data.ndim == 1:
                return False
                
            wl = raw_data[:, 0]
            num_cols = raw_data.shape[1]
            
            j = 1
            component_idx = 1
            while j < num_cols:
                col_name = headers[j] if (headers and j < len(headers)) else f"A{component_idx}"
                if "err" in col_name.lower():
                    j += 1
                    continue
                
                amp_values = raw_data[:, j]
                err_values = None
                
                if j + 1 < num_cols and "err" in headers[j+1].lower():
                    err_values = raw_data[:, j+1]
                    j += 2  
                else:
                    j += 1  
                
                # --- MODIFIED: ROUNDING AND FORMATTING TO 2 DECIMALS (:.2f) ---
                label_text = col_name
                if component_idx in taus_dict:
                    t_val, t_err = taus_dict[component_idx]
                    if t_err > 0.00001:
                        label_text = f"$\\tau_{component_idx}$ = {t_val:.2f} $\\pm$ {t_err:.2f} ps"
                    else:
                        label_text = f"$\\tau_{component_idx}$ = {t_val:.2f} ps"
                # -----------------------------------------------------------------
                    
                self.spectra_data.append({
                    'wl': wl,
                    'amp': amp_values,
                    'err': err_values,
                    'label': label_text
                })
                component_idx += 1
                
            return True
        except Exception as e:
            print(f"Error processing spectra: {e}")
            return False
            
    def replotted(self):
        """Redraws the plot applying the chosen style, limits, and manual legend entries."""
        self.setup_paper_style()
        if not self.spectra_data:
            self.canvas.draw_idle()
            return
            
        palette_choice = self.combo_palette.currentText()
        N = len(self.spectra_data)
        
        ms = self.spin_ms.value()
        cap = self.spin_cap.value()
        marker_style = 'o' if ms > 0 else None
        
        generated_colors = []
        if palette_choice == "Scientific (Nature)":
            generated_colors = [self.nature_colors[i % len(self.nature_colors)] for i in range(N)]
        else:
            cmap_map = {
                "Qualitative (Tab10)": "tab10",
                "Vibrant (Set1)": "Set1",
                "Sequential (Viridis)": "viridis",
                "Sequential (Plasma)": "plasma",
                "Cool / Warm": "coolwarm"
            }
            cmap = plt.get_cmap(cmap_map[palette_choice])
            if palette_choice in ["Qualitative (Tab10)", "Vibrant (Set1)"]:
                generated_colors = [cmap(i % cmap.N) for i in range(N)]
            else:
                generated_colors = [cmap(val) for val in np.linspace(0.0, 0.85, N)] if N > 1 else [cmap(0.5)]

        for i, data in enumerate(self.spectra_data):
            color = generated_colors[i]
            wl = data['wl']
            amp = data['amp']
            err = data['err']
            
            if err is None:
                err = np.zeros_like(amp)
            
            item = self.table_labels.item(i, 1)
            if item and item.text().strip():
                label_text = item.text()
            else:
                label_text = data['label']
                
            self.ax.errorbar(wl, amp, yerr=err, fmt='-', marker=marker_style, color=color, 
                             linewidth=2.0, markersize=ms, capsize=cap, elinewidth=1.2, 
                             markeredgewidth=1.0, alpha=0.9, label=label_text, zorder=4) 
            
        if not self.chk_auto_axes.isChecked():
            self.ax.set_xlim(self.spin_xmin.value(), self.spin_xmax.value())
            self.ax.set_ylim(self.spin_ymin.value(), self.spin_ymax.value())
            
        self.ax.legend(frameon=True, framealpha=0.0, edgecolor='none', fontsize=10, loc='best')
        self.fig.tight_layout()
        self.canvas.draw_idle()
        
    def clear_data(self):
        """Clears all loaded files and resets the plot canvas."""
        self.spectra_data = []
        if hasattr(self, 'table_labels'):
            self.table_labels.setRowCount(0) 
        self.replotted()
        
    def export_figure(self):
        """Exports the Matplotlib figure to PDF, PNG, or TIFF format at 600 DPI."""
        if not self.spectra_data:
            return
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Publication Figure", "", 
            "PDF Vector Graphic (*.pdf);;PNG High-Resolution Image (*.png);;TIFF Image (*.tiff)"
        )
        if save_path:
            self.update_fig_size()
            self.fig.savefig(save_path, dpi=600, bbox_inches='tight')
            QMessageBox.information(self, "Export Successful", "Spectrum exported successfully at 600 DPI.")
