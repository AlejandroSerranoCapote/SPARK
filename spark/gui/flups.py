import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
from matplotlib.colors import BoundaryNorm
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from mpl_toolkits.axes_grid1 import make_axes_locatable

from PyQt5.QtWidgets import (QMainWindow, QWidget, QDialog, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QLineEdit, QFileDialog, QMessageBox, 
                             QComboBox, QDoubleSpinBox, QSlider, QDial, QGroupBox, QInputDialog,
                             QApplication)
from PyQt5.QtCore import Qt, QEvent

from spark.core.chirp import fit_t0, apply_t0_correction_nonlinear, apply_t0_correction_poly, t0_model
from spark.data.importer import load_data
from .global_fit_class_gui import GlobalFitPanel 
from spark.data.dataset import SpectroscopyDataset

MODULES_STYLESHEET = """
    QMainWindow, QWidget {
        background-color: #F8F9FA; 
        color: #222222;            
        font-family: "Segoe UI", Arial, sans-serif;
        font-size: 9pt;              
    }
    QGroupBox {
        border: none;
        border-top: 1px solid #D0D0D0;
        margin-top: 18px;             
        padding-top: 15px;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0px 5px 0px 0px;
        font-weight: bold;
        font-size: 10pt;
        color: #3C5488; 
    }
    QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {
        background-color: #FFFFFF; 
        border: 1px solid #CED4DA;
        border-radius: 4px;
        color: #212529;            
        padding: 4px 8px;                
        min-height: 22px; 
    }
    QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QLineEdit:focus {
        border: 1px solid #80BDFF; 
    }
    QPushButton {
        background-color: #E2E8F0;
        border: 1px solid #CBD5E1;
        border-radius: 4px;
        padding: 6px 12px;
        color: #0F172A;
        font-weight: bold;
    }
    QPushButton:hover {
        background-color: #CBD5E1;
        border: 1px solid #94A3B8; 
    }
    QPushButton#BtnGreen {
        background-color: #10B981; 
        border: 1px solid #059669; 
        color: white; 
    }
    QPushButton#BtnGreen:hover {
        background-color: #059669; 
    }
    /* Sliders personalizados para que encajen en el sidebar */
    QSlider::groove:horizontal {
        border: 1px solid #CED4DA;
        height: 6px;
        background: #E9ECEF;
        margin: 2px 0;
        border-radius: 3px;
    }
    QSlider::handle:horizontal {
        background: #3C5488;
        border: 1px solid #2C3E50;
        width: 14px;
        margin: -4px 0;
        border-radius: 7px;
    }
    QSlider::handle:horizontal:hover {
        background: #0078D7;
    }
"""


class ManualChirpDialog(QDialog):
    """Diálogo para introducir manualmente o cargar los parámetros del Chirp."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manual / Load Chirp Parameters")
        self.resize(350, 180)
        self.setStyleSheet(MODULES_STYLESHEET)
        
        layout = QVBoxLayout(self)
        
        # Selección del modelo
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["nonlinear (a, b, c, d)", "poly (c4, c3, c2, c1, c0)"])
        layout.addWidget(QLabel("<b>Model Type:</b>"))
        layout.addWidget(self.combo_mode)
        
        # Entrada de parámetros
        self.txt_params = QLineEdit()
        self.txt_params.setPlaceholderText("e.g. 1.873, 2.598, 0.00044, -145.03")
        layout.addWidget(QLabel("<b>Parameters (comma separated):</b>"))
        layout.addWidget(self.txt_params)
        
        # Botón para cargar desde archivo txt
        self.btn_load_file = QPushButton("Load from _fit_params.txt")
        self.btn_load_file.setStyleSheet("background-color: #3B82F6; color: white; font-weight: bold;")
        self.btn_load_file.clicked.connect(self.load_from_file)
        layout.addWidget(self.btn_load_file)
        
        layout.addSpacing(10)
        
        # Botones Aceptar / Cancelar
        btns = QHBoxLayout()
        btn_ok = QPushButton("Apply Correction")
        btn_ok.setObjectName("BtnGreen")
        btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        
        btns.addWidget(btn_ok)
        btns.addWidget(btn_cancel)
        layout.addLayout(btns)
        
    def load_from_file(self):
        """Parsea el archivo txt de parámetros generado previamente."""
        path, _ = QFileDialog.getOpenFileName(self, "Select fit_params.txt", "", "Text Files (*.txt)")
        if path:
            try:
                with open(path, 'r') as f:
                    lines = f.readlines()
                    
                params = []
                for line in lines:
                    if "Fit method:" in line:
                        if "poly" in line:
                            self.combo_mode.setCurrentIndex(1)
                        else:
                            self.combo_mode.setCurrentIndex(0)
                    elif "=" in line:
                        # Extraer el valor numérico a la derecha del igual
                        val = float(line.split("=")[1].strip())
                        params.append(str(val))
                        
                self.txt_params.setText(", ".join(params))
            except Exception as e:
                QMessageBox.warning(self, "Error parsing file", f"Could not read parameters:\n{e}")
                
    def get_data(self):
        """Devuelve el modelo y la lista de parámetros introducidos."""
        mode = "nonlinear" if self.combo_mode.currentIndex() == 0 else "poly"
        try:
            params = [float(x.strip()) for x in self.txt_params.text().split(',')]
            return mode, params
        except ValueError:
            return None, None
        
class FLUPSAnalyzer(QMainWindow):
    """
    Main application window for FLUPS (Fluorescence Upconversion Spectroscopy) analysis.
    Provides an interactive GUI to load data, visualize 2D maps, fit time-zero (t0) 
    dispersion curves, and explore kinetics/spectra dynamically.
    """

    def __init__(self):
        """Initializes the FLUPS Analyzer UI, layouts, and state variables."""
        super().__init__()
        self.setWindowTitle("SPARK - FLUPS Analyzer")
        
        screen = QApplication.primaryScreen()
        screen_geom = screen.availableGeometry() # Usable size (excluding taskbar)
        
        w_target = int(screen_geom.width() * 0.85)
        h_target = int(screen_geom.height() * 0.90)
        
        x_pos = (screen_geom.width() - w_target) // 2 + screen_geom.left()
        y_pos = screen_geom.top() + 35
        
        self.setGeometry(x_pos, y_pos, w_target, h_target)
        self.setMinimumSize(1000, 700)
    
        # --- State variables ---
        self.dataset = None             
        self.dataset_corrected = None   
        
        # Mantenemos estas temporalmente para compatibilidad con los gráficos
        self.WL = None
        self.TD = None
        self.data = None
        
        self.file_path = None
        self.result_fit = None
        self.use_discrete_levels = True

        
        self.bg_cache = None
        self.cid_draw = None 
        self._is_drawing = False

        # --- UI Widgets ---
        self.btn_load = QPushButton("Load CSV")
        self.btn_load.setObjectName("BtnGreen")
        self.btn_load.clicked.connect(self.load_file)

        self.btn_plot = QPushButton("Show Map")
        self.btn_plot.clicked.connect(self.plot_map)
        self.btn_plot.setEnabled(False)
        
        self.btn_remove_fringe = QPushButton("Remove Pump Fringe")
        self.btn_remove_fringe.clicked.connect(self.remove_pump_fringe)
        self.btn_remove_fringe.setEnabled(True)
        
        self.label_status = QLabel("No file loaded")
    
        self.btn_select = QPushButton("Select t₀ points")
        self.btn_select.clicked.connect(self.enable_point_selection)
        self.btn_select.setEnabled(False)
    
        self.btn_fit = QPushButton("Fit t₀")
        self.btn_fit.clicked.connect(self.fit_t0_points)
        self.btn_fit.setEnabled(False)
        
        self.btn_auto_chirp = QPushButton("Auto-Chirp")
        self.btn_auto_chirp.clicked.connect(self.auto_fit_chirp)
        self.btn_auto_chirp.setObjectName("BtnGreen") # Para que destaque
        self.btn_auto_chirp.setEnabled(False)
        self.btn_manual_chirp = QPushButton("Manual / Load Chirp")
        self.btn_manual_chirp.clicked.connect(self.apply_manual_chirp)
        self.btn_manual_chirp.setEnabled(False)
        self.btn_auto_chirp.setToolTip("Automatically detect and correct t0 dispersion")
        
        self.btn_show_corr = QPushButton("Show Corrected Map")
        self.btn_show_corr.clicked.connect(self.toggle_corrected_map)
        self.btn_show_corr.setEnabled(False)
        self.showing_corrected = False

        self.btn_global_fit = QPushButton("Global Fit")
        self.btn_global_fit.clicked.connect(self.open_global_fit)
        self.btn_global_fit.setObjectName("BtnGreen")

        self._last_move_time = 0.0
        self._move_min_interval = 1.0 / 25.0  
        
        self.figure = Figure(figsize=(12, 8))
        self.gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1], width_ratios=[1, 1], hspace=0.25, wspace=0.35)
        
        self.ax_map = self.figure.add_subplot(self.gs[0, :])
        self.ax_time_small = self.figure.add_subplot(self.gs[1, 0])
        self.ax_spec_small = self.figure.add_subplot(self.gs[1, 1])
        
        self.canvas = FigureCanvas(self.figure)
        self.cid_draw = self.canvas.mpl_connect('draw_event', self.on_draw)
        self.cid_move = self.canvas.mpl_connect("motion_notify_event", self.on_move_map)
        
        self.clicked_points = []   
        self.cid_click = None     
            
        
        self.pcm = None
        self.cbar = None
        self.marker_map = None
        self.vline_map = None
        self.hline_map = None
        self.fit_line_artist = None
    
        self._init_small_plots()
    
        # --- Main window layout ---
        layout = QVBoxLayout()

        # Top Layout (Buttons)
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.btn_load)
        top_layout.addWidget(self.label_status)
        top_layout.addWidget(self.btn_plot)
        top_layout.addWidget(self.btn_select)
        top_layout.addWidget(self.btn_fit)
        top_layout.addWidget(self.btn_auto_chirp)
        top_layout.addWidget(self.btn_show_corr)
        top_layout.addWidget(self.btn_remove_fringe)
        top_layout.addWidget(self.btn_global_fit)
        layout.addLayout(top_layout)
        
        # Add Canvas
        layout.addWidget(self.canvas)
        
        # --- APLICAMOS EL ESTILO PREMIUM AL MÓDULO ---
        self.setStyleSheet(MODULES_STYLESHEET)

        # ===================================================================
        # THE NEW SPLIT-SCREEN LAYOUT (SIDEBAR + CANVAS)
        # ===================================================================
        
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)
        
        # --- A. PANEL IZQUIERDO (SIDEBAR) ---
        self.sidebar_widget = QWidget()
        self.sidebar_widget.setFixedWidth(360)
        self.left_layout = QVBoxLayout(self.sidebar_widget)
        self.left_layout.setContentsMargins(0, 0, 0, 0)
        self.left_layout.setSpacing(10)
        
        # 1. Data Source
        gb_data = QGroupBox("1. Data Source")
        v_data = QVBoxLayout(gb_data)
        h_load = QHBoxLayout()
        h_load.addWidget(self.btn_load)
        self.label_status.setWordWrap(True)
        h_load.addWidget(self.label_status, stretch=1)
        v_data.addLayout(h_load)
        v_data.addWidget(self.btn_remove_fringe)
        self.left_layout.addWidget(gb_data)
        
        # 2. Data Cropping (AQUÍ CREAMOS LOS CONTROLES DE NUEVO)
        gb_crop = QGroupBox("2. Data Cropping")
        v_crop = QVBoxLayout(gb_crop)
        
        self.xmin_edit = QLineEdit("-1")
        self.xmax_edit = QLineEdit("3")
        self.xmin_edit.setFixedWidth(50)
        self.xmax_edit.setFixedWidth(50)
        self.btn_apply_xlim = QPushButton("Apply")
        self.btn_apply_xlim.clicked.connect(self.apply_x_limits)
        
        h_delay = QHBoxLayout()
        h_delay.addWidget(QLabel("Delay (ps):"))
        h_delay.addWidget(self.xmin_edit)
        h_delay.addWidget(QLabel("to"))
        h_delay.addWidget(self.xmax_edit)
        h_delay.addWidget(self.btn_apply_xlim)
        v_crop.addLayout(h_delay)
        
        self.lbl_min_value = QLabel("400") 
        self.lbl_min_value.setCursor(Qt.PointingHandCursor)
        self.lbl_min_value.installEventFilter(self)
        self.slider_min = QSlider(Qt.Horizontal)
        self.slider_min.setMinimum(400)
        self.slider_min.setMaximum(800)
        self.slider_min.setValue(500)
        self.slider_min.valueChanged.connect(self.update_wl_range)
        
        self.lbl_max_value = QLabel("800") 
        self.lbl_max_value.setCursor(Qt.PointingHandCursor)
        self.lbl_max_value.installEventFilter(self)
        self.slider_max = QSlider(Qt.Horizontal)
        self.slider_max.setMinimum(400)
        self.slider_max.setMaximum(800)
        self.slider_max.setValue(700)
        self.slider_max.valueChanged.connect(self.update_wl_range)

        h_wl_min = QHBoxLayout()
        h_wl_min.addWidget(QLabel("λ min:"))
        h_wl_min.addWidget(self.slider_min)
        h_wl_min.addWidget(self.lbl_min_value)
        v_crop.addLayout(h_wl_min)
        
        h_wl_max = QHBoxLayout()
        h_wl_max.addWidget(QLabel("λ max:"))
        h_wl_max.addWidget(self.slider_max)
        h_wl_max.addWidget(self.lbl_max_value)
        v_crop.addLayout(h_wl_max)
        self.left_layout.addWidget(gb_crop)
        
        # 3. t0 Chirp Correction
        gb_chirp = QGroupBox("3. t0 Chirp Correction")
        v_chirp = QVBoxLayout(gb_chirp)
        h_tools1 = QHBoxLayout()
        h_tools1.addWidget(self.btn_select)
        h_tools1.addWidget(self.btn_fit)
        v_chirp.addLayout(h_tools1)
        
        self.combo_model = QComboBox()
        self.combo_model.addItems(["Polynomial", "Non linear"])
        self.combo_model.setCurrentIndex(1)
        
        h_model = QHBoxLayout()
        h_model.addWidget(QLabel("Model:"))
        h_model.addWidget(self.combo_model)
        v_chirp.addLayout(h_model)
        v_chirp.addWidget(self.btn_auto_chirp)
        v_chirp.addWidget(self.btn_manual_chirp)
        v_chirp.addWidget(self.btn_show_corr)
        self.left_layout.addWidget(gb_chirp)
        
        # 4. Visualization
        gb_vis = QGroupBox("4. Visualization")
        v_vis = QVBoxLayout(gb_vis)
        
        self.combo_scale = QComboBox()
        self.combo_scale.addItems(["SymLog", "Linear"])
        self.combo_scale.setCurrentIndex(0) 
        self.combo_scale.currentIndexChanged.connect(self.apply_y_scale)
        
        h_scale = QHBoxLayout()
        h_scale.addWidget(QLabel("Y Scale:"))
        h_scale.addWidget(self.combo_scale)
        v_vis.addLayout(h_scale)
        
        self.lbl_linthresh = QLabel("Linthresh (ps):")
        self.spin_linthresh = QDoubleSpinBox()
        self.spin_linthresh.setDecimals(2)
        self.spin_linthresh.setRange(0.01, 1000.0) 
        self.spin_linthresh.setValue(1.0) 
        self.spin_linthresh.setSingleStep(0.5)
        self.spin_linthresh.valueChanged.connect(self.apply_y_scale)
        
        h_lin = QHBoxLayout()
        h_lin.addWidget(self.lbl_linthresh)
        h_lin.addWidget(self.spin_linthresh)
        v_vis.addLayout(h_lin)
        
        self.n_levels = 30
        self.dial_levels = QDial()
        self.dial_levels.setRange(2, 100)
        self.dial_levels.setValue(self.n_levels)
        self.dial_levels.setNotchesVisible(True)
        self.dial_levels.setFixedSize(35, 35)
        self.dial_levels.valueChanged.connect(self.update_n_levels)
        self.lbl_dial = QLabel(f"{self.n_levels}")
        
        h_levels = QHBoxLayout()
        h_levels.addWidget(QLabel("Map Levels:"))
        h_levels.addWidget(self.dial_levels)
        h_levels.addWidget(self.lbl_dial)
        h_levels.addStretch()
        h_levels.addWidget(self.btn_plot)
        v_vis.addLayout(h_levels)
        self.left_layout.addWidget(gb_vis)
        
        self.left_layout.addStretch()
        
        # 5. Global Fit Button
        self.btn_global_fit.setFixedHeight(40)
        self.left_layout.addWidget(self.btn_global_fit)
        
        # --- B. PANEL DERECHO (CANVAS) ---
        main_layout.addWidget(self.sidebar_widget)
        main_layout.addWidget(self.canvas, stretch=1)
        
        # Set central widget
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        
        # --- Color styling for main axes and colorbars ---
        self.ax_map.tick_params(colors="black")
        self.ax_map.xaxis.label.set_color("black")
        self.ax_map.yaxis.label.set_color("black")
        self.ax_map.title.set_color("black")
        for spine in self.ax_map.spines.values():
            spine.set_color("black")
        
        if self.cbar is not None:
            self.cbar.ax.yaxis.set_tick_params(color="black", labelcolor="black")
            self.cbar.ax.yaxis.label.set_color("black")
            for spine in self.cbar.ax.spines.values():
                spine.set_color("black")
                
        for ax in [self.ax_time_small, self.ax_spec_small]:
            ax.tick_params(colors="black")
            ax.xaxis.label.set_color("black")
            ax.yaxis.label.set_color("black")
            ax.title.set_color("black")

    def on_draw(self, event):
        """
        Captures the background for Blitting with anti-recursion protection.
        
        Args:
            event: The matplotlib draw event.
        """
        if event is not None and event.canvas != self.canvas:
            return
        
        if self._is_drawing:
            return

        self._is_drawing = True 
        try:
            self.bg_cache = self.canvas.copy_from_bbox(self.figure.bbox)
            self.draw_animated_artists()
        finally:
            self._is_drawing = False

    def apply_y_scale(self):
        """Applies the selected Y-axis scale and updates the plot instantly."""
        is_symlog = self.combo_scale.currentText() == "SymLog"
        
        self.lbl_linthresh.setVisible(is_symlog)
        self.spin_linthresh.setVisible(is_symlog)
        
        # If the map doesn't exist yet, do nothing
        if not hasattr(self, 'ax_map') or self.data is None:
            return

        # Apply scale
        if is_symlog:
            # Reads the exact value from the spinbox to set where the logarithmic part begins
            self.ax_map.set_yscale("symlog", linthresh=self.spin_linthresh.value())
            self.ax_map.set_ylabel("Delay (ps) - SymLog")
        else:
            self.ax_map.set_yscale("linear")
            self.ax_map.set_ylabel("Delay (ps) - Linear")
            
        # Redraw only what is necessary
        self.canvas.draw_idle()

    def draw_animated_artists(self):
        """Draws only the animated (moving) elements over the cached background."""
        # Map Lines
        if self.vline_map: self.ax_map.draw_artist(self.vline_map)
        if self.hline_map: self.ax_map.draw_artist(self.hline_map)
        if self.marker_map: self.ax_map.draw_artist(self.marker_map)
        
        # Small Plot Lines
        if self.cut_time_small: self.ax_time_small.draw_artist(self.cut_time_small)
        if self.vline_time_small: self.ax_time_small.draw_artist(self.vline_time_small)
        if self.cut_spec_small: self.ax_spec_small.draw_artist(self.cut_spec_small)

    def eventFilter(self, obj, event):
        """
        Intercepts specific events from monitored widgets.
        
        Args:
            obj: The QObject receiving the event.
            event: The QEvent object.
            
        Returns:
            bool: True if event was handled, False otherwise.
        """
        if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            if obj == self.lbl_min_value:
                self.prompt_exact_wl_min()
                return True # Indicate that the event has already been processed
            elif obj == self.lbl_max_value:
                self.prompt_exact_wl_max()
                return True
        
        # Let the rest of the events process normally
        return super().eventFilter(obj, event)

    def prompt_exact_wl_min(self):
        """Opens a dialog to precisely set the minimum λ value."""
        if getattr(self, "WL", None) is None: 
            return # Prevents errors if no data is loaded
            
        try:
            current_val = float(self.lbl_min_value.text().replace(" nm", ""))
        except ValueError:
            current_val = float(np.min(self.WL))
        
        val, ok = QInputDialog.getDouble(
            self, "Exact Min Wavelength", "Enter wavelength (nm):", 
            value=current_val, decimals=2, min=np.min(self.WL), max=np.max(self.WL)
        )
        
        if ok:
            idx = int(np.argmin(np.abs(self.WL - val)))
            self.slider_min.setValue(idx)
    
    def prompt_exact_wl_max(self):
        """Opens a dialog to precisely set the maximum λ value."""
        if getattr(self, "WL", None) is None: 
            return
            
        try:
            current_val = float(self.lbl_max_value.text().replace(" nm", ""))
        except ValueError:
            current_val = float(np.max(self.WL))
        
        val, ok = QInputDialog.getDouble(
            self, "Exact Max Wavelength", "Enter wavelength (nm):", 
            value=current_val, decimals=2, min=np.min(self.WL), max=np.max(self.WL)
        )
        
        if ok:
            idx = int(np.argmin(np.abs(self.WL - val)))
            self.slider_max.setValue(idx)

    def open_global_fit(self):
        """Opens the Global Fit panel dialog."""
        dlg = GlobalFitPanel(self)
        dlg.exec_()

    def _init_small_plots(self):
        """Initializes the empty state and labels of the subplots."""
        self.ax_time_small.set_xlabel("Delay (ps)")
        self.ax_time_small.set_ylabel("ΔA")
        self.ax_time_small.set_title("Kinetics (cursor)")
        self.ax_time_small.set_xlim(-1, 3)
        self.cut_time_small, = self.ax_time_small.plot([], [], '-', lw=1.5)
    
        self.vline_time_small = self.ax_time_small.axvline(
            x=0, color='k', ls='--', lw=1, visible=False, zorder=5
        )
    
        self.ax_spec_small.set_xlabel("Wavelength (nm)")
        self.ax_spec_small.set_ylabel("ΔA")
        self.ax_spec_small.set_title("Spectra (cursor)")
        self.cut_spec_small, = self.ax_spec_small.plot([], [], '-', lw=1.5)
        
    def apply_x_limits(self):
        """Applies the X-axis (Delay) limits entered by the user."""
        try:
            x_min = float(self.xmin_edit.text())
            x_max = float(self.xmax_edit.text())
            if x_min >= x_max:
                raise ValueError("x_min debe ser menor que x_max")
            
            self.ax_time_small.set_xlim(x_min, x_max)
            self.canvas.draw_idle()
    
        except ValueError:
            QMessageBox.warning(self, "Error", "Introduce valores numéricos válidos para los límites de Delay.")

    def remove_pump_fringe(self):
        """Removes the pump fringe directly from the current data."""
        if self.data is None:
            QMessageBox.warning(self, "No data", "Load data first.")
            return
    
        sWl, ok1 = QInputDialog.getDouble(
            self, "Pump wavelength", "Pump wavelength (nm):", min=0.0
        )
        if not ok1:
            return
        wisWL, ok2 = QInputDialog.getDouble(
            self, "Width of scattering", "Width of pump scattering (nm):", min=0.0
        )
        if not ok2:
            return
    
        if getattr(self, "showing_corrected", False) and self.data_corrected is not None:
            data_target = self.data_corrected
        else:
            data_target = self.data
    
        # Fringe indices
        posl1 = np.argmin(np.abs(self.WL - (sWl - wisWL / 2)))
        posl2 = np.argmin(np.abs(self.WL - (sWl + wisWL / 2)))
    
        # Modify data directly
        data_target[posl1:posl2, :] = 1e-10
    
        # Refresh map to see the effect
        if getattr(self, "showing_corrected", False):
            self.toggle_corrected_map()  # re-show corrected map
        else:
            self.plot_map()  # re-show original map
    
        QMessageBox.information(
            self, "Pump fringe removed",
            f"Fringe at {sWl} ± {wisWL/2} nm has been set to near-zero."
        )

    def load_file(self):
        """Loads the data file, cleans it, and creates a SpectroscopyDataset object."""
        # Select CSV or data.txt
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select CSV or Data File", "", 
            "CSV Files (*.csv);;Data Files (*.txt *.dat)"
        )
        if not file_path:
            return
    
        try:
            # Load data
            if file_path.endswith(".csv"):
                data, wl, td = load_data(auto_path=file_path)
            else:
                wl_path, _ = QFileDialog.getOpenFileName(self, "Select Wavelength File", "", "Text Files (*.txt)")
                td_path, _ = QFileDialog.getOpenFileName(self, "Select Delay File", "", "Text Files (*.txt)")
                if not wl_path or not td_path:
                    QMessageBox.warning(self, "Files missing", "You must select both WL and TD files.")
                    return
                data, wl, td = load_data(data_path=file_path, wl_path=wl_path, td_path=td_path)
    
            # --- REMOVE DUPLICATES AND SORT (FLUPS) ---
            wl, idx_wl = np.unique(wl, return_index=True)
            td, idx_td = np.unique(td, return_index=True)
            
            # Crop the data matrix to match the clean indices
            data = data[idx_wl, :][:, idx_td]
    
            # --- 1. CREACIÓN DEL OBJETO PRINCIPAL ---
            self.dataset = SpectroscopyDataset(data, wl, td)
            
            # --- 2. Retrocompatibilidad (temporal) ---
            # Mantenemos las variables antiguas alimentadas por el objeto para que 
            # las funciones de dibujo (blitting/mapas) no se rompan de momento.
            self.WL = self.dataset.wl
            self.TD = self.dataset.td
            self.data = self.dataset.data
            
            self.file_path = file_path

            # --- RESETEAR EL ESTADO DE AJUSTES ANTERIORES ---
            self.result_fit = None
            self.dataset_corrected = None
            self.data_corrected = None
            self.showing_corrected = False
            self.fit_line_artist = None
            self.clicked_points = []
            self.btn_show_corr.setEnabled(False)
            self.btn_show_corr.setText("Show Corrected Map")
            
            # Save the CSV path and base directory
            self.csv_path = file_path
            self.base_dir = os.path.dirname(file_path)
            
            self.label_status.setText(f"Loaded : {os.path.basename(file_path)}")
            self.btn_plot.setEnabled(True)
            self.btn_select.setEnabled(True)
            self.btn_fit.setEnabled(True)
            self.btn_auto_chirp.setEnabled(True)
            self.btn_manual_chirp.setEnabled(True)
            
            # Update sliders
            nwl = len(self.dataset.wl)
            self.slider_min.blockSignals(True) 
            self.slider_max.blockSignals(True)
            
            self.slider_min.setMinimum(0)
            self.slider_min.setMaximum(nwl - 1)
            self.slider_max.setMinimum(0)
            self.slider_max.setMaximum(nwl - 1)
            
            self.slider_min.setValue(0)
            self.slider_max.setValue(nwl - 1)
            
            self.slider_min.blockSignals(False)
            self.slider_max.blockSignals(False)
            self.update_wl_range()
            
        except Exception as e:
            QMessageBox.critical(self, "Error loading file", str(e))
            
    def apply_wl_range(self):
        """Applies the current slider values to the console printout."""
        min_val = self.slider_min.value()
        max_val = self.slider_max.value()
        print(f"Aplicando λ min={min_val}, λ max={max_val}")

    def _plot_discrete_map(self, ax, WL, TD, data, n_levels=None, cmap='jet', shading='auto', vmin=None, vmax=None):
        """
        Draws a contourf-style map using a discrete pcolormesh.
        
        Args:
            ax: The matplotlib axis object.
            WL: Wavelength array.
            TD: Time Delay array.
            data: The 2D data matrix.
            n_levels: Number of discrete color levels.
            cmap: The colormap string.
            shading: Shading style for pcolormesh.
            vmin: Minimum value for the color scale.
            vmax: Maximum value for the color scale.
            
        Returns:
            QuadMesh: The created pcolormesh object.
        """
        if n_levels is None:
            n_levels = getattr(self, "n_levels", 30)

        if vmin is None:
            vmin = np.nanmin(data)
        if vmax is None:
            vmax = np.nanmax(data)

        if np.isnan(vmin) or np.isnan(vmax) or vmin >= vmax:
            vmin = 0.0 if np.isnan(vmin) else vmin
            vmax = vmin + 1.0

        levels = np.linspace(vmin, vmax, max(2, int(n_levels)))
        norm = BoundaryNorm(levels, ncolors=plt.get_cmap(cmap).N, clip=True)

        pcm = ax.pcolormesh(WL, TD, data.T, shading=shading, cmap=cmap, norm=norm)
        return pcm

    def update_n_levels(self, value):
        """Updates the number of levels for the discrete map and redraws it, respecting the visible range."""
        self.n_levels = int(value)
        if hasattr(self, "lbl_dial"):
            self.lbl_dial.setText(f"{value}")

        if getattr(self, "data", None) is None:
            return

        self.plot_map()


    def plot_map(self):
        """Draws the main map configured for Blitting (high speed)."""
        if self.data is None: return

        # Standard cleanup
        self.ax_map.clear()
        if self.cbar:
            try: self.cbar.remove()
            except: pass
            self.cbar = None

        # --- Determine data to plot (respecting filters) ---
        WL_plot = self.WL_visible if hasattr(self, "WL_visible") and self.WL_visible is not None else self.WL
        default_data = self.data_corrected if getattr(self, "showing_corrected", False) else self.data
        data_plot = self.data_visible if hasattr(self, "data_visible") and self.data_visible is not None else default_data

        # 1. Draw Map (Static)
        if self.use_discrete_levels:
            self.pcm = self._plot_discrete_map(self.ax_map, WL_plot, self.TD, data_plot, n_levels=self.n_levels)
        else:
            self.pcm = self.ax_map.pcolormesh(WL_plot, self.TD, data_plot.T, shading="auto", cmap="jet")

        self.apply_y_scale()
        tech_name = "TAS" if getattr(self, "is_TAS_mode", False) else "FLUPS"
        if getattr(self, "showing_corrected", False):
            suffix = " (t₀ Corrected)"
        elif getattr(self, "result_fit", None) is not None:
            suffix = " (Base/Original)"
        else:
            suffix = ""
        self.ax_map.set_title(f"ΔA Map ({tech_name}){suffix}")
        self.ax_map.set_xlabel("Wavelength (nm)")
        self.ax_map.set_ylabel("Delay (ps)")

        divider = make_axes_locatable(self.ax_map)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        self.cbar = self.figure.colorbar(self.pcm, cax=cax, label="ΔA")

        # 2. Initialize Dynamic Elements (animated=True)
        if getattr(self, '_last_cursor_x', None) is not None and WL_plot.min() <= self._last_cursor_x <= WL_plot.max():
            x0 = self._last_cursor_x
        else:
            x0 = WL_plot[0]

        if getattr(self, '_last_cursor_y', None) is not None and self.TD.min() <= self._last_cursor_y <= self.TD.max():
            y0 = self._last_cursor_y
        else:
            y0 = self.TD[0]

        self.vline_map = self.ax_map.axvline(x0, color='k', ls='--', lw=1, animated=True, zorder=6)
        self.hline_map = self.ax_map.axhline(y0, color='k', ls='--', lw=1, animated=True, zorder=6)
        self.marker_map, = self.ax_map.plot([x0], [y0], 'wx', markersize=8, markeredgewidth=2, animated=True, zorder=7)

        # 3. Prepare small subplots (IMPORTANT: Set limits here)
        self.ax_time_small.clear()
        self.ax_spec_small.clear()

        # Initialize animated lines (empty or at current cursor position)
        idx_wl = int(np.abs(WL_plot - x0).argmin())
        idx_td = int(np.abs(self.TD - y0).argmin())
        self.cut_time_small, = self.ax_time_small.plot(self.TD, data_plot[idx_wl, :], 'b-', lw=1.5, animated=True)
        self.vline_time_small = self.ax_time_small.axvline(y0, color='k', ls='--', lw=1, animated=True)

        self.cut_spec_small, = self.ax_spec_small.plot(WL_plot, data_plot[:, idx_td], 'r-', lw=1.5, animated=True)

        # --- SET STATIC LIMITS ---
        vmin_g, vmax_g = np.nanmin(data_plot), np.nanmax(data_plot)
        margin = (vmax_g - vmin_g) * 0.05 if vmax_g > vmin_g else 1.0

        self.ax_time_small.set_xlim(self.TD.min(), self.TD.max())
        self.ax_time_small.set_ylim(vmin_g - margin, vmax_g + margin)
        self.ax_time_small.set_xlabel("Delay (ps)")
        self.ax_time_small.set_title("Kinetics (Preview)") # Static title

        self.ax_spec_small.set_xlim(WL_plot.min(), WL_plot.max())
        self.ax_spec_small.set_ylim(vmin_g - margin, vmax_g + margin)
        self.ax_spec_small.set_xlabel("Wavelength (nm)")
        self.ax_spec_small.set_title("Spectra (Preview)") # Static title

        # Restore clicked points if any
        if getattr(self, "clicked_points", None):
            for p in self.clicked_points:
                p['artist'], = self.ax_map.plot(p['x'], p['y'], 'wo', markeredgecolor='k', markersize=6, zorder=6)

        # Connect events
        if self.cid_click is None:
            self.cid_click = self.canvas.mpl_connect("button_press_event", self.on_click_map)
        if getattr(self, 'cid_move', None) is None:
            self.cid_move = self.canvas.mpl_connect("motion_notify_event", self.on_move_map)

        # ---  REDIBUJAR LA LÍNEA DEL FIT AUTOMÁTICAMENTE ---
        # Solo dibujamos la línea roja si tenemos un fit calculado y estamos viendo el mapa original
        showing_corrected = getattr(self, "showing_corrected", False)
        if not showing_corrected and getattr(self, 'result_fit', None) is not None:
            fit_x = self.result_fit.get('fit_x')
            fit_y = self.result_fit.get('fit_y')
            if fit_x is not None and fit_y is not None:
                # Recuperar el nombre del método para la leyenda
                metodo = self.result_fit.get('method', 't₀ fit')

                # Borrar la línea anterior por si acaso quedó algún rastro en memoria
                if getattr(self, 'fit_line_artist', None) is not None:
                    try: self.fit_line_artist.remove()
                    except: pass

                # --- SOLUCIÓN ELEGANTE (Sin errores de Linter) ---
                # 1. Guardamos el encuadre actual del mapa
                current_xlim = self.ax_map.get_xlim()

                # 2. Dibujamos la línea roja
                self.fit_line_artist, = self.ax_map.plot(fit_x, fit_y, 'r-', lw=2, label=f"Fit: {metodo}")
                self.ax_map.legend()

                # 3. Volvemos a aplicar el encuadre exacto que guardamos
                self.ax_map.set_xlim(current_xlim)
                # -------------------------------------------------
        # ---------------------------------------------------------

        # 4. Trigger the first full draw (Generates bg_cache)
        self.canvas.draw()


    def update_wl_range(self):
        """
        Updates the visible data variables based on sliders 
        and calls plot_map to redraw everything correctly.
        """
        if getattr(self, "WL", None) is None or getattr(self, "data", None) is None:
             # Update text to dashes if no data is present
            if hasattr(self, "lbl_min_value"): self.lbl_min_value.setText("- nm")
            if hasattr(self, "lbl_max_value"): self.lbl_max_value.setText("- nm")
            return

        # 1. Get slider indices
        wl_min_idx = int(self.slider_min.value())
        wl_max_idx = int(self.slider_max.value())

        # 2. Correct index crossings
        if wl_min_idx >= wl_max_idx: 
            wl_max_idx = wl_min_idx + 1
        
        # Ensure array boundaries
        wl_min_idx = max(0, min(wl_min_idx, len(self.WL) - 1))
        wl_max_idx = max(0, min(wl_max_idx, len(self.WL) - 1))

        # 3. Update Text Labels (nm)
        try:
            self.lbl_min_value.setText(f"{self.WL[wl_min_idx]:.1f} nm")
            self.lbl_max_value.setText(f"{self.WL[wl_max_idx]:.1f} nm")
        except Exception:
            pass

        # 4. DEFINE VISIBLE DATA (Global Visualization State)
        source_data = self.data_corrected if getattr(self, "showing_corrected", False) else self.data
        
        # Slice the data
        self.WL_visible = self.WL[wl_min_idx : wl_max_idx + 1]
        self.data_visible = source_data[wl_min_idx : wl_max_idx + 1, :]

        # 5. CENTRALIZED CALL
        self.plot_map()
            
    def enable_point_selection(self):
        """Activates the mode allowing the user to select t0 points on the plot."""
        self.clicked_points = []
        if self.cid_click is None:
            self.cid_click = self.canvas.mpl_connect("button_press_event", self.on_click_map)
        QMessageBox.information(self, "Mode: Select points",
                                "Left click: add point\nRight click: delete last point.\nThen press 'Fit t₀'.")

    def update_small_cuts(self, x, y, WL_sel=None, data_sel=None):
        """
        Full update after a click event.
        
        Args:
            x (float): The clicked X coordinate.
            y (float): The clicked Y coordinate.
            WL_sel (numpy.ndarray, optional): Selected Wavelength slice.
            data_sel (numpy.ndarray, optional): Selected data slice.
        """
        # Reuse movement logic by simulating an event
        # This ensures visual consistency
        class MockEvent:
            pass
        evt = MockEvent()
        evt.xdata = x
        evt.ydata = y
        evt.inaxes = self.ax_map
        
        # Call on_move_map for fast rendering
        self.on_move_map(evt)
        
        # If it was a click, ensure it stays fixed (optional)
        # self.canvas.draw_idle()

    def on_click_map(self, event):
        """Registers points on the map (left adds, right deletes last) and updates cuts."""
        if event.inaxes != self.ax_map:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        if event.button == 1:  # left click -> add point
            artist, = self.ax_map.plot(x, y, 'wo', markeredgecolor='k', markersize=6, zorder=6)
            self.clicked_points.append({'x': x, 'y': y, 'artist': artist})
        elif event.button == 3 and self.clicked_points:  # right click -> delete last
            last = self.clicked_points.pop()
            try:
                last['artist'].remove()
            except Exception:
                pass

        # update marker tracking the cursor (optional: move main marker)
        if self.marker_map is None:
            self.marker_map, = self.ax_map.plot([x], [y], 'wx', markersize=8, markeredgewidth=2)
        else:
            self.marker_map.set_data([x], [y])


        # update visual references for the vertical line
        if self.vline_map is None:
            # if it doesn't exist, create it
            self.vline_map = self.ax_map.axvline(x, color='k', ls='--', lw=1)
        else:
            # if it already exists, just update its position and ensure visibility
            self.vline_map.set_xdata([x, x])
            self.vline_map.set_visible(True)
        
        # update visual references for the horizontal line
        if self.hline_map is None:
            self.hline_map = self.ax_map.axhline(y, color='k', ls='--', lw=1)
        else:
            self.hline_map.set_ydata([y, y])
            self.hline_map.set_visible(True)

        # --- Here is the difference: update the small subplots ---
        self.update_small_cuts(x, y)
        self.update_small_cuts(
            x, y,
            WL_sel=getattr(self, "WL_visible", None),
            data_sel=getattr(self, "data_visible", None)
        )
        self.canvas.draw_idle()

    def on_move_map(self, event):
        """
        Handles mouse movement over the plot to update cursors and slices dynamically.
        
        Args:
            event: The matplotlib mouse motion event.
        """
            
        # If no cache or not on the axis, exit
        if self.bg_cache is None or self.data is None: 
            return
        if event.inaxes != self.ax_map: 
            return

        # 1. Restore clean background (erases previous cursors instantly)
        self.canvas.restore_region(self.bg_cache)

        # 2. Update mathematical positions (without drawing yet)
        x, y = event.xdata, event.ydata
        if x is None or y is None: return
        
        self._last_cursor_x = x
        self._last_cursor_y = y
        
        # Map lines
        self.vline_map.set_xdata([x, x])
        self.hline_map.set_ydata([y, y])
        self.marker_map.set_data([x], [y])
        
        # Calculate indices for subplots
        # Use WL_visible if it exists, otherwise full WL
        cur_WL = self.WL_visible if hasattr(self, 'WL_visible') and self.WL_visible is not None else self.WL
        cur_data = self.data_visible if hasattr(self, 'data_visible') and self.data_visible is not None else self.data
        
        if cur_WL is not None and len(cur_WL) > 0:
            idx_wl = int(np.abs(cur_WL - x).argmin())
            idx_td = int(np.abs(self.TD - y).argmin())

            # Update small curves
            self.cut_time_small.set_data(self.TD, cur_data[idx_wl, :])
            self.vline_time_small.set_xdata([y, y])
            self.cut_spec_small.set_data(cur_WL, cur_data[:, idx_td])

            # Info in status bar
            val = cur_data[idx_wl, idx_td]
            self.label_status.setText(f"Cursor: {x:.1f} nm, {y:.2f} ps | Val: {val:.4e}")

        # 3. Draw ONLY the animated elements and blit to screen
        self.draw_animated_artists()
        self.canvas.blit(self.figure.bbox)

    def fit_t0_points(self):
        """Fits the selected points to a t0 curve and saves the extracted/corrected data."""
        if not getattr(self, "clicked_points", None) or len(self.clicked_points) < 2:
            QMessageBox.warning(self, "Not enough points", "Select at least 2 points on the map.")
            return

        w_points = np.array([p['x'] for p in self.clicked_points])
        t0_points = np.array([p['y'] for p in self.clicked_points])

        texto_modelo = self.combo_model.currentText()
        
        if texto_modelo == "Polynomial":
            mode = 'poly'
        elif texto_modelo == "Non linear":
            mode = 'nonlinear'
        else:
            mode = 'auto'
        
        # Attempt the fit
        try:
            result = fit_t0(w_points, t0_points, self.WL, self.TD, self.data, mode=mode)
        except Exception as e:
            QMessageBox.critical(self, "Error de ajuste t₀", str(e))
            return

        self.result_fit = result
        self.data_corrected = result['corrected']

        # draw fit curve on main map
        if self.fit_line_artist is not None:
            try:
                self.fit_line_artist.remove()
            except Exception:
                pass
        self.fit_line_artist, = self.ax_map.plot(result['fit_x'], result['fit_y'], 'r-', lw=2, label="t₀ fit")
        self.ax_map.legend()
        self.canvas.draw_idle()

        # automatic saving (identical to current behavior)
        self.btn_show_corr.setEnabled(True)

        base_dir = os.path.dirname(self.file_path)
        base_name = os.path.splitext(os.path.basename(self.file_path))[0]
        self.save_dir = os.path.join(base_dir, f"{base_name}_Results") 
        os.makedirs(self.save_dir, exist_ok=True)
        
        data_corr = result['corrected']
        WL = self.WL
        TD = self.TD

        np.save(os.path.join(self.save_dir, f"{base_name}_treated_data.npy"),
                {'data_c': data_corr, 'WL': WL, 'TD': TD})

        np.savetxt(os.path.join(self.save_dir, f"{base_name}_WL.txt"), WL,
                   fmt='%.6f', header='Wavelength (nm)', comments='')
        np.savetxt(os.path.join(self.save_dir, f"{base_name}_TD.txt"), TD,
                   fmt='%.6f', header='Delay (ps)', comments='')

        with open(os.path.join(self.save_dir, f"{base_name}_kin.txt"), 'w') as f:
            f.write("\t".join([f"{base_name}_kin_{round(wl,1)}nm" for wl in WL]) + "\n")
            np.savetxt(f, data_corr.T, fmt='%.6e', delimiter='\t')

        with open(os.path.join(self.save_dir, f"{base_name}_spec.txt"), 'w') as f:
            f.write("\t".join([f"{base_name}_spec_{td:.2f}ps" for td in TD]) + "\n")
            np.savetxt(f, data_corr, fmt='%.6e', delimiter='\t')

        t0_lambda = result['t0_lambda']
        popt = result['popt']
        method = result['method']

        t0_file = os.path.join(self.save_dir, f"{base_name}_t0_fit.txt")
        np.savetxt(t0_file, np.column_stack((WL, t0_lambda)),
                   fmt='%.6f', header='Wavelength (nm)\t t0 (ps)', comments='')

        params_file = os.path.join(self.save_dir, f"{base_name}_fit_params.txt")
        with open(params_file, 'w') as f:
            f.write(f"Fit method: {method}\n")
            f.write("Fit parameters:\n")
            if method.startswith('poly'):
                names = ['c4', 'c3', 'c2', 'c1', 'c0']
            else:
                names = ['a', 'b', 'c', 'd']
            for name, val in zip(names, popt):
                f.write(f"  {name} = {val:.6g}\n")

        QMessageBox.information(self, "Files saved",
                                f"Results saved in:\n{self.save_dir}")
        QMessageBox.information(self, "t₀ Fit Result",
                                f"Fit completed using {method} model.\nParameters: {np.round(popt,4)}")

    def auto_fit_chirp(self):
        """
        Automatically detects t0 using Gaussian smoothing and a strict 
        Global Intensity Threshold to reject dead spectral zones.
        """
        if self.data is None:
            QMessageBox.warning(self, "No data", "Load data first.")
            return

        from scipy.ndimage import gaussian_filter1d

        # 1. Usar región visible
        if hasattr(self, 'WL_visible') and self.WL_visible is not None:
            wl_array = self.WL_visible
            data_array = self.data_visible
        else:
            wl_array = self.WL
            data_array = self.data

        # 2. Definir ventana de búsqueda
        global_max_idx = np.unravel_index(np.nanargmax(data_array), data_array.shape)
        global_t_max = self.TD[global_max_idx[1]]
        t_search_max = global_t_max + 3.0 
        valid_td_mask = self.TD <= t_search_max

        # EL FILTRO MAESTRO: 15% del máximo absoluto de todo el mapa
        global_max_val = np.nanmax(data_array)
        global_threshold = global_max_val * 0.15 

        w_points = []
        t0_points = []

        for i, wl in enumerate(wl_array):
            raw_kinetics = data_array[i, :]
            
            # Suavizar para no pillar ruido de alta frecuencia
            kinetics = gaussian_filter1d(raw_kinetics, sigma=2)

            kinetics_valid = kinetics[valid_td_mask]
            td_valid = self.TD[valid_td_mask]

            if len(kinetics_valid) < 5: continue 

            max_idx = np.argmax(kinetics_valid)
            max_val = kinetics_valid[max_idx]

            # LA CRIBADORA: Si esta lambda no tiene fuerza real comparada con el pico, fuera.
            if max_val < global_threshold:
                continue

            target_val = max_val * 0.5
            cross_idx = None
            
            for j in range(max_idx, 0, -1):
                if kinetics_valid[j] >= target_val and kinetics_valid[j-1] < target_val:
                    cross_idx = j
                    break

            if cross_idx is not None:
                y1, y2 = kinetics_valid[cross_idx - 1], kinetics_valid[cross_idx]
                t1, t2 = td_valid[cross_idx - 1], td_valid[cross_idx]

                if y2 != y1: 
                    t0_exact = t1 + (target_val - y1) * (t2 - t1) / (y2 - y1)
                    w_points.append(wl)
                    t0_points.append(t0_exact)

        if len(w_points) < 3:
            QMessageBox.warning(self, "Auto-Chirp failed", "No clear signals found. Adjust sliders or lower the global threshold.")
            return

        # Limpiar y dibujar
        self.clicked_points = [{'x': w, 'y': t} for w, t in zip(w_points, t0_points)]
        
        for p in self.clicked_points:
            p['artist'], = self.ax_map.plot(p['x'], p['y'], 'wo', markeredgecolor='g', markersize=4, zorder=6)
        self.canvas.draw_idle()

        # Llamar al fit
        try:
            reply = QMessageBox.question(self, 'Auto-Chirp Detection', 
                                         f"Found {len(w_points)} clean t0 points.\nProceed to fit and correct?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                self.fit_t0_points()
                
        except Exception as e:
            QMessageBox.critical(self, "Auto-Chirp Error", str(e))
            
    def apply_manual_chirp(self):
        """Aplica la corrección de dispersión t0 a partir de parámetros introducidos manualmente o de archivo."""
        if self.data is None: 
            return
            
        dlg = ManualChirpDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            mode, params = dlg.get_data()
            
            if mode is None or params is None:
                QMessageBox.warning(self, "Error", "Invalid parameters. Check comma separation.")
                return
            
            # Validación de tamaño
            if mode == 'nonlinear' and len(params) != 4:
                QMessageBox.warning(self, "Error", "Nonlinear model requires exactly 4 parameters (a, b, c, d).")
                return
            if mode == 'poly' and len(params) != 5:
                QMessageBox.warning(self, "Error", "Polynomial model requires exactly 5 parameters (c4, c3, c2, c1, c0).")
                return
            
            
            
            try:
                # Si estamos en modo TAS, recalcular la matriz base primero (resta de solvente)
                if hasattr(self, "is_TAS_mode") and self.is_TAS_mode:
                    self.update_am_sf()
                
                target_data = self.data
                    
                # Aplicar la corrección
                if mode == 'nonlinear':
                    corrected, t0_lambda = apply_t0_correction_nonlinear(params, self.WL, self.TD, target_data)
                    fit_x = np.linspace(np.min(self.WL), np.max(self.WL), 400)
                    fit_y = t0_model(fit_x, *params)
                else:
                    corrected, t0_lambda = apply_t0_correction_poly(params, self.WL, self.TD, target_data)
                    fit_x = np.linspace(np.min(self.WL), np.max(self.WL), 400)
                    fit_y = np.polyval(params, fit_x)
                    
                # Guardar el resultado en la app
                self.result_fit = {
                    'method': mode + " (manual)",
                    'popt': params,
                    'fit_x': fit_x,
                    'fit_y': fit_y,
                    'corrected': corrected,
                    't0_lambda': t0_lambda
                }
                self.data_corrected = corrected
                
                # --- SOLUCIÓN DEL PLOT ---
                # 1. Asegurarnos de que estamos viendo el mapa ORIGINAL (base),
                # ya que sobre el corregido la línea roja no tiene sentido visual.
                if getattr(self, "showing_corrected", False):
                    self.toggle_corrected_map() # Esto llama a plot_map() y nos devuelve al original
                else:
                    self.plot_map() # Forzamos un redibujado limpio del mapa original
                
                # 2. Dibujar la línea de la corrección manual en rojo DESPUÉS de dibujar el mapa
                if getattr(self, 'fit_line_artist', None) is not None:
                    try: self.fit_line_artist.remove()
                    except: pass
                    
                self.fit_line_artist, = self.ax_map.plot(fit_x, fit_y, 'r-', lw=2, label="Manual t₀")
                self.ax_map.legend()
                
                cur_WL = getattr(self, "WL_visible", self.WL)
                if cur_WL is not None and len(cur_WL) > 0:
                    self.ax_map.set_xlim(np.min(cur_WL), np.max(cur_WL))
                    
                self.canvas.draw_idle()
                
                self.btn_show_corr.setEnabled(True)
                
                # --- NUEVO: Crear la carpeta _Results y guardar los datos corregidos ---
                base_dir = os.path.dirname(self.file_path)
                base_name = os.path.splitext(os.path.basename(self.file_path))[0]
                results_dir = os.path.join(base_dir, f"{base_name}_Results")
                os.makedirs(results_dir, exist_ok=True)
                
                # Empaquetar los datos usando las claves estándar del software ('data_c', 'TD', 'WL')
                data_to_save = {
                    'data_c': corrected,
                    'TD': self.TD,
                    'WL': self.WL
                }
                
                # Guardamos como _treated_data.npy para la transición directa al Global Fit
                save_path = os.path.join(results_dir, f"{base_name}_treated_data.npy")
                np.save(save_path, data_to_save)
                # ------------------------------------------------------------------------
                    
                QMessageBox.information(self, "Success", f"Manual chirp correction applied successfully and saved in:\n{results_dir}\nMode: {mode}")
                
            except Exception as e:
                
                QMessageBox.critical(self, "Error applying manual chirp", str(e))
                
    def toggle_corrected_map(self):
        """Toggles between the original and corrected map using optimized rendering."""
        
        # 1. Safety validation
        if self.data_corrected is None:
            QMessageBox.warning(self, "No corrected data", "Run 'Fit t₀' first.")
            return

        # 2. Toggle state (boolean flag)
        self.showing_corrected = not getattr(self, "showing_corrected", False)

        # 3. Decide the data source
        # If showing_corrected is True, use corrected data. If False, use self.data (base/original).
        source_data = self.data_corrected if self.showing_corrected else self.data

        # 4. Update Texts
        if self.showing_corrected:
            self.btn_show_corr.setText("Show Base/Original Map")
            suffix = "(t₀ Corrected)"
        else:
            self.btn_show_corr.setText("Show Corrected Map")
            suffix = "(Base/Original)"

        # 5. Recalculate visible slice (Respecting Sliders)
        
        if hasattr(self, 'slider_min') and hasattr(self, 'slider_max'):
            wl_min_idx = int(self.slider_min.value())
            wl_max_idx = int(self.slider_max.value())
            
            # Index protections
            if wl_min_idx >= wl_max_idx: wl_max_idx = wl_min_idx + 1
            wl_min_idx = max(0, min(wl_min_idx, len(self.WL) - 1))
            wl_max_idx = max(0, min(wl_max_idx, len(self.WL) - 1))
            
            # Update variables used by plot_map
            self.WL_visible = self.WL[wl_min_idx:wl_max_idx+1]
            self.data_visible = source_data[wl_min_idx:wl_max_idx+1, :]
        else:
            # Fallback in case there are no sliders
            self.WL_visible = self.WL
            self.data_visible = source_data
            
        self.plot_map()
        
        # Explicitly update title to reflect state
        tech_name = "TAS" if getattr(self, "is_TAS_mode", False) else "FLUPS"
        self.ax_map.set_title(f"ΔA Map ({tech_name}) {suffix}")
        
        # A final redraw to ensure title updates
        self.canvas.draw()
    