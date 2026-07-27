# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 16:25:52 2025

@author: Alejandro
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.interpolate import RegularGridInterpolator, interp1d


import matplotlib.pyplot as plt 
from matplotlib import cm, gridspec
from matplotlib.figure import Figure
from matplotlib.colors import BoundaryNorm
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from mpl_toolkits.axes_grid1 import make_axes_locatable

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QDialog, QTabWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QMessageBox,
    QProgressBar, QTableWidget, QTableWidgetItem, QHeaderView,
    QComboBox, QDoubleSpinBox, QSpinBox, QSlider, QDial,
    QFrame, QGroupBox, QRadioButton, QCheckBox, QSpacerItem, QSizePolicy,QInputDialog
)
from PyQt5.QtGui import QFont, QPalette, QColor, QDesktopServices, QIcon
from PyQt5.QtCore import Qt, QTimer, QUrl, QSize,QEvent
from PyQt5.QtWidgets import QFrame
import fit
from core_analysis import fit_t0, load_data, eV_a_nm
from GlobalFitClassGui import GlobalFitPanel
from maps_from_timescans import AppWindow as XFELWindow

STYLESHEET = """
    QMainWindow {
        background-color: #F8FAFC; 
    }

    /* --- PANEL IZQUIERDO (BRANDING) --- */
    QFrame#LeftPanel {
        background-color: #0F172A; /* Deep Slate Navy */
    }
    QLabel#LeftTitle {
        color: #FFFFFF;
        font-size: 26px;
        font-weight: 900;
        font-family: "Segoe UI Black", "Arial Black", sans-serif;
        letter-spacing: 1px;
    }
    QLabel#LeftSub {
        color: #94A3B8;
        font-size: 13px;
        line-height: 1.5;
    }

    /* --- PANEL DERECHO (CONTENIDO) --- */
    QLabel#SectionTitle {
        color: #64748B;
        font-size: 18px;
        font-weight: bold;
        letter-spacing: 2px;
        margin-top: 5px;
    }

   
    QPushButton[cssClass="MenuCard"] {
        background-color: #FFFFFF;
        color: #0F172A;
        font-size: 14px;
        font-weight: bold;    
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        border-left: 5px solid #CBD5E1; /* Borde gris por defecto */
    }
    QPushButton[cssClass="MenuCard"]:hover {
        background-color: #F1F5F9;
        margin-left: 4px; /* Pequeña animación al pasar el ratón */
    }
    QPushButton[cssClass="MenuCard"]:pressed {
        background-color: #E2E8F0;
    }

    /* Colores únicos para cada tarjeta */
    QPushButton#CardFLUPS { border-left-color: #10B981; } /* Verde */
    QPushButton#CardTAS { border-left-color: #F59E0B; } /* Naranja */
    QPushButton#CardGlobal { border-left-color: #3B82F6; } /* Azul */
    QPushButton#CardXFEL { border-left-color: #8B5CF6; } /* Morado */

    /* --- GITHUB BUTTON --- */
    QPushButton#GithubBtn {
        background-color: rgba(255, 255, 255, 0.05);
        color: #E2E8F0;
        font-weight: bold;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 4px;
        padding: 8px;
    }
    QPushButton#GithubBtn:hover {
        background-color: rgba(255, 255, 255, 0.15);
        border: 1px solid #FFFFFF;
        color: #FFFFFF;
    }
"""

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



class MainApp(QMainWindow):
    """Main Window (LAUNCHER DASHBOARD)"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ultrafast Spectroscopy Analyzer - Launcher")
        self.setMinimumSize(900, 520) 
        self.setStyleSheet(STYLESHEET) 
        self.github_url = "https://github.com/AlejandroSerranoCapote/Ultrafast-Spectroscopy-Analyzer"
        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principal: Izquierda y Derecha sin márgenes
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ==========================================
        # PANEL IZQUIERDO (DARK BRANDING)
        # ==========================================
        left_panel = QFrame()
        left_panel.setObjectName("LeftPanel")
        left_panel.setFixedWidth(300) # Anchura fija para el menú
        
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(35, 45, 35, 30)

        title = QLabel("ULTRAFAST\nSPECTROSCOPY\nANALYZER")
        title.setObjectName("LeftTitle")
        left_layout.addWidget(title)
        
        left_layout.addSpacing(15)

        sub = QLabel("Data processing\nsuite for time-resolved\nspectroscopy & global fitting.")
        sub.setObjectName("LeftSub")
        left_layout.addWidget(sub)

        left_layout.addStretch()

        self.btn_github = QPushButton("Source on GitHub")
        self.btn_github.setObjectName("GithubBtn")
        self.btn_github.setCursor(Qt.PointingHandCursor)
        self.btn_github.clicked.connect(self.open_github)
        left_layout.addWidget(self.btn_github)

        author = QLabel("v1.4 \n© A. Serrano Capote")
        author.setStyleSheet("color: #475569; font-size: 11px; margin-top: 10px;")
        left_layout.addWidget(author)

        main_layout.addWidget(left_panel)

        # ==========================================
        # PANEL DERECHO (MODULE CARDS)
        # ==========================================
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(50, 45, 50, 45)
        right_layout.setSpacing(15)

        # --- SECCIÓN A: Técnicas Experimentales ---
        lbl_cat1 = QLabel("PRIMARY PROCESSING MODULES")
        lbl_cat1.setObjectName("SectionTitle")
        right_layout.addWidget(lbl_cat1)
        
        grid1 = QGridLayout()
        grid1.setSpacing(15)
        self.btn_flups = self.create_card("FLUPS ANALYZER", "Fluorescence Upconversion Data", "CardFLUPS")
        self.btn_tas   = self.create_card("TAS ANALYZER", "Transient Absorption Data", "CardTAS")
        grid1.addWidget(self.btn_flups, 0, 0)
        grid1.addWidget(self.btn_tas, 0, 1)
        right_layout.addLayout(grid1)
        
        right_layout.addSpacing(20)

        # --- SECCIÓN B: Herramientas Avanzadas ---
        lbl_cat2 = QLabel("ADVANCED ANALYSIS & VISUALIZATION")
        lbl_cat2.setObjectName("SectionTitle")
        right_layout.addWidget(lbl_cat2)
        
        grid2 = QGridLayout()
        grid2.setSpacing(15)
        self.btn_fit = self.create_card("GLOBAL FIT", "Target Analysis & Kinetic Modeling", "CardGlobal")
        self.btn_xfel  = self.create_card("2D MAPPER", "Create 2D maps using kinetics\n (XFEL utility)", "CardXFEL")
        grid2.addWidget(self.btn_fit, 0, 0)
        grid2.addWidget(self.btn_xfel, 0, 1)
        right_layout.addLayout(grid2)

        right_layout.addStretch()
        main_layout.addWidget(right_panel)

        # Conexiones de botones (Tus funciones de siempre)
        self.btn_flups.clicked.connect(self.launch_flups)
        self.btn_tas.clicked.connect(self.launch_tas)
        self.btn_fit.clicked.connect(self.launch_global)
        self.btn_xfel.clicked.connect(self.launch_xfel)

    def create_card(self, title_text, sub_text, object_name):
            """Crea una tarjeta de módulo usando QLabels internos para soportar formato rico."""
            btn = QPushButton()
            btn.setProperty("cssClass", "MenuCard") 
            btn.setObjectName(object_name)          
            btn.setCursor(Qt.PointingHandCursor)
            btn.setMinimumHeight(85)
            
            # Le metemos un Layout INTERNO al botón
            vbox = QVBoxLayout(btn)
            vbox.setContentsMargins(20, 15, 20, 15)
            vbox.setSpacing(4)
            
            # Creamos la etiqueta del Título
            title = QLabel(title_text)
            title.setStyleSheet("color: #0F172A; font-size: 15px; font-weight: bold; border: none; background: transparent;")
            title.setAttribute(Qt.WA_TransparentForMouseEvents) # El clic traspasa al botón
            
            # Creamos la etiqueta del Subtítulo
            sub = QLabel(sub_text)
            sub.setStyleSheet("color: #64748B; font-size: 12px; font-weight: normal; border: none; background: transparent;")
            sub.setAttribute(Qt.WA_TransparentForMouseEvents)
            
            # Añadimos los textos al botón
            vbox.addWidget(title)
            vbox.addWidget(sub)
            vbox.addStretch() # Empuja el texto hacia arriba suavemente
            
            return btn

    def open_github(self):
        if hasattr(self, 'github_url'):
            from PyQt5.QtGui import QDesktopServices
            from PyQt5.QtCore import QUrl
            QDesktopServices.openUrl(QUrl(self.github_url))

    def open_github(self):
        """Opens the repository URL in the user's default web browser."""
        if hasattr(self, 'github_url'):
            QDesktopServices.openUrl(QUrl(self.github_url))

    def open_tool(self, tool_window):
        """
        Hides the main menu, opens the selected tool, and ensures the menu 
        reappears when the child tool window is closed.
        
        Args:
            tool_window (QMainWindow or QWidget): The specific tool instance to open.
        """
        self.current_tool = tool_window
        
        # Store the original close event of the tool window
        original_close = tool_window.closeEvent
        
        def on_close_tool(event):
            # Show the main menu again when the tool closes
            self.show()             
            # Execute the tool's standard close event
            original_close(event)  
            
        # Override the tool's close event with our custom wrapper
        tool_window.closeEvent = on_close_tool
        
        tool_window.show()
        self.hide()
        
    def launch_xfel(self):
        """Instantiates and launches the XFEL 2D Mapper tool."""
        window = XFELWindow() 
        self.open_tool(window)

    def launch_flups(self):
        """Instantiates and launches the FLUPS Analyzer tool."""
        window = FLUPSAnalyzer()
        self.open_tool(window)

    def launch_tas(self):
        """Instantiates and launches the TAS Analyzer tool."""
        window = TASAnalyzer()
        self.open_tool(window)

    def launch_global(self):
        """Instantiates and launches the Global Fit tool."""
        window = GlobalFitPanel()
        self.open_tool(window)
        
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
        self.setWindowTitle("FLUPS Analyzer")
        
        screen = QApplication.primaryScreen()
        screen_geom = screen.availableGeometry() # Usable size (excluding taskbar)
        
        w_target = int(screen_geom.width() * 0.85)
        h_target = int(screen_geom.height() * 0.90)
        
        x_pos = (screen_geom.width() - w_target) // 2 + screen_geom.left()
        y_pos = screen_geom.top() + 35
        
        self.setGeometry(x_pos, y_pos, w_target, h_target)
        self.setMinimumSize(1000, 700)
    
        # --- State variables ---
        self.WL = None
        self.TD = None
        self.data = None
        self.file_path = None
        self.data_corrected = None
        self.result_fit = None
        self.use_discrete_levels = True  # Change to False for a continuous map
        
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
        """Loads the data file, cleans it, and automatically normalizes ΔA."""
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
    
            # --- Normalization ---
            
            # =============================================================================
            #                 NORMALIZATION DATA IN FLUPS
            # =============================================================================

            max_val = np.nanmax(np.abs(data))
            if max_val != 0:
                data = data / max_val
    
    
            self.WL, self.TD, self.data = wl, td, data
            self.file_path = file_path

            
            # ---  RESETEAR EL ESTADO DE AJUSTES ANTERIORES ---
            self.result_fit = None
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
            nwl = len(wl)
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

    def _plot_discrete_map(self, ax, WL, TD, data, n_levels=5, cmap='jet', shading='auto', vmin=None, vmax=None):
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
        if vmin is None:
            vmin = np.nanmin(data)
        if vmax is None:
            vmax = np.nanmax(data)
    
        levels = np.linspace(vmin, vmax, n_levels)
        norm = BoundaryNorm(levels, ncolors=plt.get_cmap(cmap).N, clip=True)
    
        pcm = ax.pcolormesh(WL, TD, data.T, shading=shading, cmap=cmap, norm=norm)
        return pcm

    def update_n_levels(self, value):
        """Updates the number of levels for the discrete map and redraws it, respecting the visible range."""
        self.n_levels = value
        self.lbl_dial.setText(f"{value} levels")  # update text
    
        if self.data is None:
            return
    
        # Determine which data and WL to use (respecting current visible range)
        if hasattr(self, "WL_visible") and self.WL_visible is not None:
            WL_used = self.WL_visible
            if getattr(self, "showing_corrected", False):
                # if we are showing the corrected map
                wl_min = self.WL_visible[0]
                wl_max = self.WL_visible[-1]
                wl_min_idx = np.argmin(np.abs(self.WL - wl_min))
                wl_max_idx = np.argmin(np.abs(self.WL - wl_max)) + 1
                data_used = self.data_corrected[wl_min_idx:wl_max_idx, :]
            else:
                data_used = self.data_visible
        else:
            WL_used = self.WL
            data_used = self.data_corrected if getattr(self, "showing_corrected", False) else self.data
    
        # Redraw map directly (without resetting)
        self.ax_map.clear()
        if self.cbar:
            try: self.cbar.remove()
            except: pass
            self.cbar = None
    
        self.pcm = self._plot_discrete_map(
            self.ax_map,
            WL_used,
            self.TD,
            data_used,
            n_levels=self.n_levels,
            shading="auto",
            vmin=-1,
            vmax=1
        )
    
        self.ax_map.set_xlabel("Wavelength (nm)")
        self.ax_map.set_ylabel("Delay (ps)")
        self.ax_map.set_title("ΔA Map")
        self.apply_y_scale()
        
        # Colorbar
        divider = make_axes_locatable(self.ax_map)
        cax = divider.append_axes("right", size="3%", pad=0.02)
        self.cbar = self.figure.colorbar(self.pcm, cax=cax, label="ΔA")
    
        # Coherent visual style
        self.ax_map.set_facecolor("white")
        for spine in self.ax_map.spines.values():
            spine.set_color("black")
        self.ax_map.tick_params(colors="black")
        self.ax_map.xaxis.label.set_color("black")
        self.ax_map.yaxis.label.set_color("black")
        self.ax_map.title.set_color("black")
    
        self.canvas.draw_idle()


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
        data_plot = self.data_visible if hasattr(self, "data_visible") and self.data_visible is not None else self.data

        # 1. Draw Map (Static)
        if self.use_discrete_levels:
            self.pcm = self._plot_discrete_map(self.ax_map, WL_plot, self.TD, data_plot, n_levels=self.n_levels)
        else:
            self.pcm = self.ax_map.pcolormesh(WL_plot, self.TD, data_plot.T, shading="auto", cmap="jet")

        self.apply_y_scale()
        self.ax_map.set_title("ΔA Map")
        self.ax_map.set_xlabel("Wavelength (nm)")
        self.ax_map.set_ylabel("Delay (ps)")

        divider = make_axes_locatable(self.ax_map)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        self.cbar = self.figure.colorbar(self.pcm, cax=cax, label="ΔA")

        # 2. Initialize Dynamic Elements (animated=True)
        x0, y0 = WL_plot[0], self.TD[0]
        
        self.vline_map = self.ax_map.axvline(x0, color='k', ls='--', lw=1, animated=True, zorder=6)
        self.hline_map = self.ax_map.axhline(y0, color='k', ls='--', lw=1, animated=True, zorder=6)
        self.marker_map, = self.ax_map.plot([x0], [y0], 'wx', markersize=8, markeredgewidth=2, animated=True, zorder=7)

        # 3. Prepare small subplots (IMPORTANT: Set limits here)
        self.ax_time_small.clear()
        self.ax_spec_small.clear()
        
        # Initialize animated lines (empty or with first value)
        self.cut_time_small, = self.ax_time_small.plot(self.TD, data_plot[0, :], 'b-', lw=1.5, animated=True)
        self.vline_time_small = self.ax_time_small.axvline(y0, color='k', ls='--', lw=1, animated=True)
        
        self.cut_spec_small, = self.ax_spec_small.plot(WL_plot, data_plot[:, 0], 'r-', lw=1.5, animated=True)

        # --- SET STATIC LIMITS ---
        vmin_g, vmax_g = np.nanmin(data_plot), np.nanmax(data_plot)
        margin = (vmax_g - vmin_g) * 0.05
        
        self.ax_time_small.set_xlim(self.TD.min(), self.TD.max())
        self.ax_time_small.set_ylim(vmin_g - margin, vmax_g + margin)
        self.ax_time_small.set_xlabel("Delay (ps)")
        self.ax_time_small.set_title("Kinetics (Preview)") # Static title

        self.ax_spec_small.set_xlim(WL_plot.min(), WL_plot.max())
        self.ax_spec_small.set_ylim(vmin_g - margin, vmax_g + margin)
        self.ax_spec_small.set_xlabel("Wavelength (nm)")
        self.ax_spec_small.set_title("Spectra (Preview)") # Static title

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
            
            # Importar los modelos del motor analítico
            from core_analysis import apply_t0_correction_nonlinear, apply_t0_correction_poly, t0_model
            
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
    
class TASAnalyzer(FLUPSAnalyzer):
    """
    Transient Absorption Spectroscopy (TAS) Analyzer.
    
    Inherits from FLUPSAnalyzer but specializes in handling TAS data, which includes
    simultaneous loading and dynamic subtraction of solvent data, pump fringe removal, 
    and real-time amplitude/shift adjustments.
    """
    
    def __init__(self):
        """Initializes the TAS Analyzer UI, extending and modifying the base FLUPS UI."""
        super().__init__()
        self.setWindowTitle("TAS Analyzer")
        self.label_status.setText("No file loaded")
        
        # --- Data State ---
        self.medida = None
        self.solvente = None
        self.pump_mask = None  # New variable to remove the pump artifact
        self.TDSol = None
        self.WLSol = None
        self.is_TAS_mode = True
        self.use_discrete_levels = False 
        
        # Hide the discrete level dial (not used in TAS continuous map mode)
        self.dial_levels.hide()
        self.lbl_dial.hide()
        
        # --- Initialize variables for Blitting (optimization) ---
        self.bg_cache = None
        self.cid_draw = None
        self.cid_click = None
        self.cid_move = None
        
        # ===================================================================
        # SLIDERS AND CLICKABLE LABELS (AM and SF)
        # ===================================================================

        # 1. Slider and Label for Amplitude (AM)
        self.slider_am = QSlider(Qt.Horizontal)
        self.slider_am.setMinimumWidth(200)
        self.slider_am.setMinimum(0)
        self.slider_am.setMaximum(200)
        self.slider_am.setValue(100)  # 100% by default
        
        self.lbl_am_value = QLabel("100 %")
        self.lbl_am_value.setCursor(Qt.PointingHandCursor)
        self.lbl_am_value.setToolTip("Click to enter exact value")
        self.lbl_am_value.installEventFilter(self)
        
        self.slider_am.valueChanged.connect(self.on_am_changed)
        
        # 2. Slider and Label for Temporal Shift (SF)
        self.slider_sf = QSlider(Qt.Horizontal)
        self.slider_sf.setMinimumWidth(200)
        self.slider_sf.setMinimum(-20000)
        self.slider_sf.setMaximum(20000)
        self.slider_sf.setValue(0)
        
        self.lbl_sf_value = QLabel("0.000 ps")
        self.lbl_sf_value.setCursor(Qt.PointingHandCursor)
        self.lbl_sf_value.setToolTip("Click to enter exact value")
        self.lbl_sf_value.installEventFilter(self)
        
        self.slider_sf.valueChanged.connect(self.on_sf_changed)
        
        # ===================================================================
        # TAS LAYOUT INJECTION IN THE SIDEBAR
        # ===================================================================
        
        gb_tas = QGroupBox("TAS Adjustments")
        tas_extra_layout = QVBoxLayout(gb_tas)
        tas_extra_layout.setSpacing(8)

        amp_row = QHBoxLayout()
        amp_row.addWidget(QLabel("Amplitude (%):"))
        amp_row.addWidget(self.slider_am)
        amp_row.addWidget(self.lbl_am_value) 
        tas_extra_layout.addLayout(amp_row)

        shift_row = QHBoxLayout()
        shift_row.addWidget(QLabel("Shift (ps):"))
        shift_row.addWidget(self.slider_sf)
        shift_row.addWidget(self.lbl_sf_value) 
        tas_extra_layout.addLayout(shift_row)
        
        # Inyectamos el Widget (QGroupBox) justo debajo de Data Cropping (índice 2)
        if hasattr(self, 'left_layout'):
            self.left_layout.insertWidget(2, gb_tas)
 
        # === Checkbox for automatic .dat --> .csv conversion ===
        self.chk_convert_dat = QCheckBox("Convert .dat → .csv (IMDEA DATA)")
        self.chk_convert_dat.setChecked(True) 
        
        # Añadir al final del panel izquierdo, justo encima del botón de Global Fit
        if hasattr(self, 'left_layout'):
            self.left_layout.insertWidget(self.left_layout.count() - 2, self.chk_convert_dat)
    def on_am_changed(self, value):
        """Updates the amplitude text label and recalculates the map."""
        self.lbl_am_value.setText(f"{value} %")
        self.update_am_sf()

    def on_sf_changed(self, value):
        """Updates the shift text label and recalculates the map."""
        self.lbl_sf_value.setText(f"{value / 100.0:.3f} ps")
        self.update_am_sf()

    def prompt_exact_am(self):
        """Opens a dialog to allow the user to enter an exact amplitude value."""
        current_val = self.slider_am.value()
        val, ok = QInputDialog.getDouble(
            self, "Exact Amplitude", "Enter amplitude (%):", 
            value=current_val, decimals=0, min=0, max=200
        )
        if ok:
            self.slider_am.setValue(int(val))

    def prompt_exact_sf(self):
        """Opens a dialog to allow the user to enter an exact shift value in ps."""
        current_val = self.slider_sf.value() / 100.0
        val, ok = QInputDialog.getDouble(
            self, "Exact Shift", "Enter shift (ps):", 
            value=current_val, decimals=3, min=-200.0, max=200.0
        )
        if ok:
            self.slider_sf.setValue(int(val * 100))

    def eventFilter(self, obj, event):
        """
        Intercepts click events on the Amplitude and Shift labels.
        
        Args:
            obj: The QObject receiving the event.
            event: The QEvent object.
            
        Returns:
            bool: True if event was handled, False otherwise.
        """
        if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            if obj == getattr(self, "lbl_am_value", None):
                self.prompt_exact_am()
                return True
            elif obj == getattr(self, "lbl_sf_value", None):
                self.prompt_exact_sf()
                return True
                
        # Pass the rest of the events (like the WL ones) to the parent class
        return super().eventFilter(obj, event)

    def switch_analyzer(self):
        """Switches between FLUPSAnalyzer and TASAnalyzer without closing the main application process."""
        try:
            target_cls_name = "FLUPSAnalyzer" if isinstance(self, TASAnalyzer) else "TASAnalyzer"
            
            if target_cls_name in globals() and callable(globals()[target_cls_name]):
                TargetCls = globals()[target_cls_name]
            else:
                raise NameError(f"{target_cls_name} not found")

            # Save the reference in self (not a local variable)
            self.new_window = TargetCls()
            self.new_window.show()

            # Close the current window
            self.close()

        except Exception as e:
            QMessageBox.critical(self, "Switch error", f"Cannot switch analyzer:\n{e}")

    def convert_dat_to_csv(self, file_path):
        """
        Converts a .dat file into a .csv file structured for TAS analysis.
        
        Args:
            file_path (str): The path to the original .dat file.
            
        Returns:
            str or None: The path to the newly created .csv file, or None if conversion fails.
        """
        try:
            data = np.loadtxt(file_path)
    
            # wl = first column
            wl = data[:, 0]
    
            # t = first row (convert to ps)
            t = data[0] * 1e-3

            # Replace in the matrix
            data[:, 0] = wl
            data[0, :] = t
    
            # Create .csv path
            csv_path = os.path.splitext(file_path)[0] + ".csv"
    
            # Save
            np.savetxt(csv_path, data, delimiter=",")
            return csv_path
    
        except Exception as e:
            QMessageBox.critical(self, "Conversion error", f"Cannot convert .dat:\n{e}")
            return None

    def get_base_dir(self):
        """
        Returns the directory containing the measurement CSV.
        Automatically creates 'fit' and 'plots' subfolders if they do not exist.
        
        Returns:
            tuple: (base_dir, fit_dir, plots_dir) paths.
        """
        if hasattr(self, 'file_path') and self.file_path:
            base_dir = os.path.dirname(self.file_path)
        else:
            base_dir = os.getcwd()
    
        fit_dir = os.path.join(base_dir, "fit")
        plots_dir = os.path.join(base_dir, "plots")
        os.makedirs(fit_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
    
        return base_dir, fit_dir, plots_dir

    def remove_pump_fringe(self):
        """
        Prompts the user for a central wavelength and width to mask out the pump scatter artifact.
        The masked region is set to near-zero (1e-10) to avoid division by zero errors.
        """
        if self.data is None:
            QMessageBox.warning(self, "No data", "Load TAS data first.")
            return
    
        sWl, ok1 = QInputDialog.getDouble(self, "Pump wavelength", "Pump wavelength (nm):", min=0.0)
        if not ok1: return
        wisWL, ok2 = QInputDialog.getDouble(self, "Width of scattering", "Width of pump scattering (nm):", min=0.0)
        if not ok2: return
    
        posl1 = np.argmin(np.abs(self.WL - (sWl - wisWL / 2)))
        posl2 = np.argmin(np.abs(self.WL - (sWl + wisWL / 2)))
    
        # Create or update mask
        if self.pump_mask is None:
            self.pump_mask = np.zeros_like(self.medida, dtype=bool)
        self.pump_mask[posl1:posl2, :] = True
    
        # Apply mask over self.data
        self.update_am_sf()
    
        QMessageBox.information(self, "Pump fringe removed",
                                f"Fringe at {sWl} ± {wisWL/2} nm will be zeroed.")

    # ------------------------------------------------------------------
    # FILE LOADING
    # ------------------------------------------------------------------
    def load_file(self):
        """
        Loads both the measurement data and the corresponding solvent data.
        Handles automatic deduplication, sorting, and initial UI setup for TAS data.
        """
        
        # --- Select measurement file ---
        file_path_medida, _ = QFileDialog.getOpenFileName(
            self,
            "Select Measurement CSV",
            "",
            "CSV Files (*.csv);;Data Files (*.txt *.dat)"
        )
        if not file_path_medida or not os.path.exists(file_path_medida):
            self.label_status.setText(" No measurement file selected.")
            return
            
        # --- Automatic .dat → .csv conversion if the option is checked ---
        if self.chk_convert_dat.isChecked() and file_path_medida.lower().endswith(".dat"):
            new_path = self.convert_dat_to_csv(file_path_medida)
            if new_path:
                file_path_medida = new_path
                
        # Save the path of the first CSV read
        self.file_path = file_path_medida
        
        # Create a specific folder for this measurement
        base_dir = os.path.dirname(self.file_path)
        base_name = os.path.splitext(os.path.basename(self.file_path))[0]
        self.results_dir = os.path.join(base_dir, f"{base_name}_results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        raw = pd.read_csv(file_path_medida, header=None)
        raw = raw.apply(pd.to_numeric, errors="coerce").dropna(how="any")
        raw = raw.values.astype(float)
        
        temp_TD = raw[0, 1:]       # Temporary delay (ps)
        temp_WL = raw[1:, 0]       # Temporary wavelength (nm)
        temp_medida = raw[1:, 1:]  # 2D Intensity Matrix
        temp_medida[np.isnan(temp_medida)] = 0
        
        # --- REMOVE DUPLICATES AND SORT (MEASUREMENT) ---
        self.WL, idx_wl = np.unique(temp_WL, return_index=True)
        self.TD, idx_td = np.unique(temp_TD, return_index=True)
        
        # Crop the data matrix to match the clean indices
        self.medida = temp_medida[idx_wl, :][:, idx_td]
        
        # --- Select solvent file ---
        file_path_solvente, _ = QFileDialog.getOpenFileName(
            self,
            "Select Solvent CSV",
            os.path.dirname(self.file_path),  # ✅ Opens dialog in the same folder
            "CSV Files (*.csv);;Data Files (*.txt *.dat)"
        )
        
        if not file_path_solvente or not os.path.exists(file_path_solvente):
            self.label_status.setText(" No solvent file selected.")
            return
            
        if self.chk_convert_dat.isChecked() and file_path_solvente.lower().endswith(".dat"):
            new_path = self.convert_dat_to_csv(file_path_solvente)
            if new_path:
                file_path_solvente = new_path        
                
        rawSol = pd.read_csv(file_path_solvente, header=None)
        rawSol = rawSol.apply(pd.to_numeric, errors="coerce").dropna(how="any")
        rawSol = rawSol.values.astype(float)
        
        temp_TDSol = rawSol[0, 1:]
        temp_WLSol = rawSol[1:, 0]
        temp_solvente = rawSol[1:, 1:]
        temp_solvente[np.isnan(temp_solvente)] = 0
        
        # ---  REMOVE DUPLICATES AND SORT (SOLVENT) ---
        self.WLSol, idx_wl_sol = np.unique(temp_WLSol, return_index=True)
        self.TDSol, idx_td_sol = np.unique(temp_TDSol, return_index=True)
        self.solvente = temp_solvente[idx_wl_sol, :][:, idx_td_sol]
        
        nwl = len(self.WL)
        self.slider_min.setMinimum(0)
        self.slider_min.setMaximum(nwl - 1)
        self.slider_max.setMinimum(0)
        self.slider_max.setMaximum(nwl - 1)
        self.slider_min.setValue(0)
        self.slider_max.setValue(nwl - 1)
        
        self.idx_min = 0
        self.idx_max = nwl - 1
        
        
        try: self.slider_min.valueChanged.disconnect()
        except: pass
        try: self.slider_max.valueChanged.disconnect()
        except: pass
        self.slider_min.valueChanged.connect(self.update_wl_range)
        self.slider_max.valueChanged.connect(self.update_wl_range)
        
        # --- RESETEAR EL ESTADO DE AJUSTES ANTERIORES ---
        self.result_fit = None
        self.data_corrected = None
        self.showing_corrected = False
        self.fit_line_artist = None
        self.clicked_points = []
        if hasattr(self, 'btn_show_corr'):
            self.btn_show_corr.setEnabled(False)
            self.btn_show_corr.setText("Show Corrected Map")
        # -------------------------------------------------------
        
        # --- Calculate initial map ---
        self.label_status.setText(" TAS data loaded")
        self.update_am_sf()
        self.plot_map()
        
        # --- Define base path for compatibility with FLUPSAnalyzer ---
        self.file_path = file_path_medida
        
        if hasattr(self, "btn_plot"):
            self.btn_plot.setEnabled(True)
        if hasattr(self, "btn_select"):
            self.btn_select.setEnabled(True)
        if hasattr(self, "btn_fit"):
            self.btn_fit.setEnabled(True)
        if hasattr(self, "btn_auto_chirp"):
            self.btn_auto_chirp.setEnabled(True)
        if hasattr(self, "btn_manual_chirp"):
            self.btn_manual_chirp.setEnabled(True)
        
        #  Display only the loaded file name
        file_name = os.path.basename(file_path_medida)
        self.label_status.setText(f"TAS data loaded from: {file_name}")


    # In TASAnalyzer (replacing the current version)
    def fit_t0_points(self):
        """
        Fits selected time-zero points and saves the corrected matrix.
        Overrides the base FLUPS method to ensure the solvent-subtracted base data is used.
        """
        if not getattr(self, "clicked_points", None) or len(self.clicked_points) < 2:
            QMessageBox.warning(self, "Not enough points", "Select at least 2 points on the map.")
            return
    
        w_points = np.array([p['x'] for p in self.clicked_points])
        t0_points = np.array([p['y'] for p in self.clicked_points])
    
        try:
            # Re-calculate the base (self.data) with the most recent solvent/shift
            self.update_am_sf() 
            
            # Use self.data (Base Data: solvent-corrected) for the fit
            result = fit_t0(w_points, t0_points, self.WL, self.TD, self.data)
        except Exception as e:
            QMessageBox.critical(self, "Fit error", str(e))
            return
    
        
        # --- Save corrected data globally ---
        self.result_fit = result
        self.data_corrected = result['corrected']
        
        self.plot_map(show_fit=True)
        self.btn_show_corr.setEnabled(True)
    
        # --- Create results folder next to the CSV and save ---
        base_dir = os.path.dirname(self.file_path)
        base_name = os.path.splitext(os.path.basename(self.file_path))[0]
        save_dir = os.path.join(base_dir, f"{base_name}_results")
        os.makedirs(save_dir, exist_ok=True)
    
        data_corr = np.copy(self.data_corrected)
        WL = self.WL
        TD = self.TD
        t0_lambda = result['t0_lambda']
        popt = result['popt']
        method = result['method']
    
        np.save(os.path.join(save_dir, f"{base_name}_treated_data.npy"),
                {'data_c': data_corr, 'WL': WL, 'TD': TD})
        np.savetxt(os.path.join(save_dir, f"{base_name}_WL.txt"), WL,
                    fmt='%.6f', header='Wavelength (nm)', comments='')
        np.savetxt(os.path.join(save_dir, f"{base_name}_TD.txt"), TD,
                    fmt='%.6f', header='Delay (ps)', comments='')
        np.savetxt(os.path.join(save_dir, f"{base_name}_kin.txt"),
                    data_corr.T, fmt='%.6e', delimiter='\t')
        np.savetxt(os.path.join(save_dir, f"{base_name}_spec.txt"),
                    data_corr, fmt='%.6e', delimiter='\t')
        np.savetxt(os.path.join(save_dir, f"{base_name}_t0_fit.txt"),
                    np.column_stack((WL, t0_lambda)),
                    fmt='%.6f', header='Wavelength (nm)\t t0 (ps)', comments='')
                    
        with open(os.path.join(save_dir, f"{base_name}_fit_params.txt"), 'w') as f:
            f.write(f"Fit method: {method}\n")
            f.write("Fit parameters:\n")
            if method.startswith('poly'):
                names = ['c4', 'c3', 'c2', 'c1', 'c0']
            else:
                names = ['a', 'b', 'c', 'd']
            for name, val in zip(names, popt):
                f.write(f"  {name} = {val:.6g}\n")
    
        QMessageBox.information(self, "Files saved",
                                f" Results saved in:\n{save_dir}")
        QMessageBox.information(self, "t₀ Fit Result",
                                f"Fit completed using {method} model.\nParameters: {np.round(popt,4)}")
        

    def update_wl_range(self):
        """Updates the crop indices based on the UI sliders and refreshes the map."""
        if self.medida is None:
            return
    
        # 1. Read slider values
        # Ensure they are integers (array indices)
        s_min = int(self.slider_min.value())
        s_max = int(self.slider_max.value())
    
        # 2. Validate crossing (Min cannot be >= Max)
        if s_min >= s_max:
            s_min = s_max - 1
            if s_min < 0: s_min = 0
            self.slider_min.blockSignals(True) # Prevent infinite loop
            self.slider_min.setValue(s_min)
            self.slider_min.blockSignals(False)
    
        # 3. Save to class variables
        self.idx_min = s_min
        self.idx_max = s_max
    
        # 4. Update text labels 
        try:
            self.lbl_min_value.setText(f"{self.WL[s_min]:.1f} nm")
            self.lbl_max_value.setText(f"{self.WL[s_max]:.1f} nm")
        except Exception:
            pass
    
        # 5. Redraw
        self.plot_map()

    # ------------------------------------------------------------------
    # MAP UPDATE AFTER SLIDERS
    # ------------------------------------------------------------------
    # In TASAnalyzer (replacing the current version)
    def update_am_sf(self):
        """
        Recalculates the base TAS data by subtracting the interpolated 
        solvent matrix scaled by Amplitude (AM) and shifted in time (SF).
        """
        if self.medida is None or self.solvente is None:
            return
    
        if hasattr(self, "_updating_am_sf") and self._updating_am_sf:
            return
        self._updating_am_sf = True
    
        am = self.slider_am.value() / 100.0
        sf = self.slider_sf.value() / 100.0
        
        interpSol = RegularGridInterpolator(
            (self.WLSol, self.TDSol),
            self.solvente,
            bounds_error=False,
            fill_value=0
        )
    
        WL_grid, TD_grid = np.meshgrid(self.WL, self.TD, indexing="ij")
        points = np.column_stack([WL_grid.ravel(), (TD_grid - sf).ravel()])
        solvente_interp = interpSol(points).reshape(len(self.WL), len(self.TD)) * am
    
        # Base: measurement - solvent
        base_data = self.medida - solvente_interp
    
        # Apply mask if it exists
        if self.pump_mask is not None:
            base_data[self.pump_mask] = 1e-10
        
        # The entire 'if hasattr(self, "data_corrected") ...' block that caused double calculation is removed.
        self.data = base_data 
    
        self.update_wl_range() 
    
        if hasattr(self, "global_fit_panel") and self.global_fit_panel is not None:
            self.global_fit_panel.update_from_parent()
    
        self._updating_am_sf = False

    # ------------------------------------------------------------------
    # DRAW ΔA MAP
    # ------------------------------------------------------------------
    def plot_map(self, show_fit=False):
        """
        Draws the main interactive 2D map (SymLog in Y) with support 
        for toggling between Corrected and Original modes.
        
        Args:
            show_fit (bool, optional): Unused flag kept for backward compatibility.
        """
        
        # 1. Determine which data to use (Base vs Corrected)
        # Check the flag that activates the toggle button
        showing_corrected = getattr(self, "showing_corrected", False)
        
        if showing_corrected and hasattr(self, "data_corrected") and self.data_corrected is not None:
            source_data = self.data_corrected
            mode_suffix = "(t₀ Corrected)"
        else:
            # If no flag or False, we use self.data (which already has the solvent subtracted)
            source_data = self.data
            mode_suffix = "(Base Data)"
    
        if source_data is None:
            return
    
        saved_x, saved_y = None, None
        if getattr(self, 'marker_map', None) is not None:
            try:
                saved_x = self.marker_map.get_xdata()[0]
                saved_y = self.marker_map.get_ydata()[0]
            except Exception:
                pass
                
        # --- Cleanup ---
        self.ax_map.clear()
        self.ax_time_small.clear()
        self.ax_spec_small.clear()
        
        # Reset variables to prevent errors
        self.vline_map = None
        self.hline_map = None
        self.marker_map = None
        self.cut_time_small = None
        self.cut_spec_small = None
        
        if self.cbar:
            try: self.cbar.remove()
            except: pass
            self.cbar = None
    
        # --- 2. Slicing ---
        if not hasattr(self, 'idx_min'): self.idx_min = 0
        if not hasattr(self, 'idx_max'): self.idx_max = len(self.WL) - 1
    
        idx_start = self.idx_min
        idx_end = self.idx_max + 1
        
        wl_plot = self.WL[idx_start:idx_end]
        
        # Use source_data instead of self.data
        data_plot = source_data[idx_start:idx_end, :]
        
        if len(wl_plot) < 2: return
    
        # --- 3. Calculate Global Limits ---
        g_min = np.nanmin(data_plot)
        g_max = np.nanmax(data_plot)
            
        data_range = g_max - g_min
        if data_range == 0: data_range = 1.0
        y_lim_min = g_min - (0.1 * data_range)
        y_lim_max = g_max + (0.1 * data_range)
    
        # --- 4. Draw Main Map ---
        self.pcm = self.ax_map.pcolormesh(
            wl_plot, self.TD, data_plot.T,
            shading="auto", cmap="jet",
        )
        
        self.apply_y_scale()
        self.ax_map.set_xlabel("Wavelength (nm)")
        self.ax_map.set_ylabel("Delay (ps) - SymLog")
        
        # Update the title dynamically depending on the mode
        self.ax_map.set_title(f"ΔA Map (TAS) {mode_suffix}")
        
        self.ax_map.set_xlim(wl_plot.min(), wl_plot.max())
        
        # Colorbar
        divider = make_axes_locatable(self.ax_map)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        self.cbar = self.figure.colorbar(self.pcm, cax=cax, label="ΔA")
        self.apply_y_scale()
    
        if saved_x is not None and saved_y is not None:
            # np.clip prevents the cursor from staying off-screen if you crop too much with the slider
            mid_x = np.clip(saved_x, wl_plot.min(), wl_plot.max())
            mid_y = saved_y
        else:
            mid_x = np.median(wl_plot)
            mid_y = np.median(self.TD)
        
        self.vline_map = self.ax_map.axvline(mid_x, color="k", ls="--", lw=1, animated=True)
        self.hline_map = self.ax_map.axhline(mid_y, color="k", ls="--", lw=1, animated=True)
        self.marker_map, = self.ax_map.plot([mid_x], [mid_y], "wx", markersize=8, markeredgewidth=2, animated=True)
    
        # --- 6. Configure Small Subplots ---
        
        # A) KINETICS (Bottom-Left)
        y_cut_time = data_plot[np.abs(wl_plot - mid_x).argmin(), :]
        self.cut_time_small, = self.ax_time_small.plot(self.TD, y_cut_time, 'b-', animated=True)
        self.vline_time_small = self.ax_time_small.axvline(mid_y, color='k', ls='--', lw=1.2, animated=True)
        
        self.ax_time_small.set_xscale('linear') 
        
        try:
            user_xmin = float(self.xmin_edit.text())
            user_xmax = float(self.xmax_edit.text())
            self.ax_time_small.set_xlim(user_xmin, user_xmax)
        except ValueError:
            self.ax_time_small.set_xlim(self.TD.min(), self.TD.max())
            

        self.ax_time_small.set_ylim(y_lim_min, y_lim_max)
        self.ax_time_small.set_title("Kinetics")
        self.ax_time_small.set_xlabel("Delay (ps)")
    
        # B) SPECTRUM (Bottom-Right)
        y_cut_spec = data_plot[:, np.abs(self.TD - mid_y).argmin()]
        self.cut_spec_small, = self.ax_spec_small.plot(wl_plot, y_cut_spec, 'r-', animated=True)
        
        self.ax_spec_small.set_xlim(wl_plot.min(), wl_plot.max())
        self.ax_spec_small.set_ylim(y_lim_min, y_lim_max)
        self.ax_spec_small.set_title("Spectrum")
        self.ax_spec_small.set_xlabel("Wavelength (nm)")
    
        # --- 7. Events ---
        self.bg_cache = None
        if self.cid_draw is not None: self.canvas.mpl_disconnect(self.cid_draw)
        self.cid_draw = self.canvas.mpl_connect('draw_event', self.on_draw)
        
        if self.cid_click is None:
            self.cid_click = self.canvas.mpl_connect("button_press_event", self.on_click_map)
        if self.cid_move is None:
            self.cid_move = self.canvas.mpl_connect("motion_notify_event", self.on_move_map)
    
        self.canvas.draw()

    def on_draw(self, event):
        """
        Captures the background for blitting when the entire figure is redrawn.
        
        Args:
            event: The matplotlib draw event.
        """
        if event is not None and event.canvas != self.canvas:
            return
        # Copy the canvas region (without animated lines)
        self.bg_cache = self.canvas.copy_from_bbox(self.figure.bbox)
        
        # Take the opportunity to redraw the animated lines once
        self.draw_animated_artists()
            
    def draw_animated_artists(self):
        """Helper function to draw only the dynamic elements over the cached background."""
        # 1. Safety check:
        # If vline_map doesn't exist or is None, we do nothing.
        # This prevents a crash when the window opens before data is loaded.
        vline = getattr(self, 'vline_map', None)
        if vline is None:
            return
    
        # 2. Draw Map elements
        # (Since we already checked vline, we assume the rest were created with it)
        try:
            self.ax_map.draw_artist(self.vline_map)
            self.ax_map.draw_artist(self.hline_map)
            self.ax_map.draw_artist(self.marker_map)
            
            # 3. Draw subplot elements
            # Verify these as well for safety
            if getattr(self, 'cut_time_small', None) is not None:
                self.ax_time_small.draw_artist(self.cut_time_small)
                self.ax_time_small.draw_artist(self.vline_time_small)
            
            if getattr(self, 'cut_spec_small', None) is not None:
                self.ax_spec_small.draw_artist(self.cut_spec_small)
    
        except AttributeError:
            # If something fails internally in matplotlib (e.g. window closed), ignore
            pass

    def update_small_cuts(self, x, y, WL_sel=None, data_sel=None):
        """
        Performs a full (slow) update for clicks or slider changes.
        
        Args:
            x (float): The X coordinate.
            y (float): The Y coordinate.
            WL_sel: Unused parameter kept for signature compatibility.
            data_sel: Unused parameter kept for signature compatibility.
        """
        self.on_move_map(type('Event', (object,), {'xdata': x, 'ydata': y, 'inaxes': self.ax_map})())
        self.canvas.draw_idle() # Ensures everything stays fixed

    # ------------------------------------------------------------------
    # MOUSE MOVEMENT EVENT
    # ------------------------------------------------------------------
    def on_move_map(self, event):
        """
        Ultra-fast update of subplots and cursors during mouse movement using Blitting.
        
        Args:
            event: The matplotlib mouse motion event.
        """
        # 1. Basic validation of axes and data
        if self.data is None or event.inaxes != self.ax_map:
            return
    
        # 2. --- BUG FIX ---
        # Verify if the lines exist. If vline_map is None,
        # it means the graph is being cleared or hasn't been created yet.
        # We use getattr for extra safety.
        if getattr(self, 'vline_map', None) is None:
            return
    
        # 3. Get coordinates
        x, y = event.xdata, event.ydata
        if x is None or y is None: return
    
        # 4. Restore clean background (erases previous lines)
        if self.bg_cache is not None:
            self.canvas.restore_region(self.bg_cache)
    
        # 5. Update line data (without redrawing axes)
    
        self.vline_map.set_xdata([x, x])
        self.hline_map.set_ydata([y, y])
        self.marker_map.set_data([x], [y])
        
        # --- Data for slices ---
        idx_wl = np.abs(self.WL - x).argmin()
        idx_td = np.abs(self.TD - y).argmin()
        
        # Validate indices (in case the mouse is outside the valid data range)
        if idx_wl >= self.data.shape[0] or idx_td >= self.data.shape[1]:
            return
    
        # Update Kinetics curve
        y_time = self.data[idx_wl, :]
        self.cut_time_small.set_data(self.TD, y_time)
        self.vline_time_small.set_xdata([y, y])
        
        # Update Spectrum curve
        y_spec = self.data[:, idx_td]
        self.cut_spec_small.set_data(self.WL, y_spec)
    
        # 6. Draw the animated elements
        self.draw_animated_artists()
    
        # 7. Blit
        self.canvas.blit(self.figure.bbox)
        
        # Status bar
        val = self.data[idx_wl, idx_td]
        self.label_status.setText(f"Cursor: {x:.1f} nm, {y:.2f} ps | ΔA: {val:.4e}")

if __name__ == "__main__":
    # =====================================================================
    # Application Entry Point
    # =====================================================================
    
    # 1. Create the application instance
    app = QApplication(sys.argv)
    
    # Apply the "Fusion" style for a modern, consistent look across different OS
    app.setStyle("Fusion") 
    
    # 2. Apply the global stylesheet to the entire application
    app.setStyleSheet(STYLESHEET) 

    # 3. Force Windows to properly recognize the application icon on the taskbar.
    # This workaround prevents Windows from grouping the app under the default Python executable icon.
    import ctypes
    myappid = 'spectroscopy.analyzer.v1' 
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    # 4. Configure the Global Application Icon
    icon_path = os.path.join(os.path.dirname(__file__), "icon.ico")
    
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    # 5. Instantiate and launch the main dashboard window
    window = MainApp()
    window.show()

    # Start the main event loop and exit safely when closed
    sys.exit(app.exec_())