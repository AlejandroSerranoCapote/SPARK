# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 16:25:52 2025

@author: Alejandro
"""

# =============================================================================
# IMPORTS
# =============================================================================

# --- LIBRERÍAS ESTÁNDAR ---
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

# --- INTERFAZ GRÁFICA (PyQt5) ---
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

# --- MÓDULOS LOCALES---
import fit
from core_analysis import fit_t0, load_data, eV_a_nm
from GlobalFitClassGui import GlobalFitPanel
from maps_from_timescans import AppWindow as XFELWindow

STYLESHEET = """
    /* FONDO GENERAL Y TEXTO */
    QMainWindow, QWidget {
        background-color: #e6e8ed; /* Gris clarito muy limpio */
        color: #222222;
        font-family: "Segoe UI", Arial, sans-serif;
        font-size: 13px;
    }

    /* CAJAS DE TEXTO, LISTAS Y DESPLEGABLES (Fondo blanco) */
    QLineEdit, QComboBox, QListWidget, QTextEdit, QTableWidget {
        background-color: #FFFFFF;
        border: 1px solid #C0C0C0;
        border-radius: 3px;
        padding: 4px;
        color: #000000;
    }
    
    QComboBox::down-arrow {
            image: none; /* Quitamos la imagen por defecto si la hubiera */
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-top: 5px solid #666666; /* Dibuja un triangulito gris */
            width: 0px;
            height: 0px;
            margin-top: 2px;
        }
    
    QComboBox:hover {
            border: 1px solid #0078D7; /* Se ilumina en azul sutil al pasar el ratón */
        }
    
        /* La zona de la flecha a la derecha */
        QComboBox::drop-down {
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 25px;
            border-left: 1px solid #E5E5E5; /* Una línea divisoria sutil en lugar de un bloque gris */
            background-color: #FAFAFA;
            border-top-right-radius: 3px;
            border-bottom-right-radius: 3px;
        }
    
        /* LA LISTA QUE SE ABRE AL HACER CLIC */
        QComboBox QAbstractItemView {
            border: 1px solid #C0C0C0;
            background-color: #FFFFFF;
            selection-background-color: #E5F1FB; /* Azul muy clarito al pasar el ratón por las opciones */
            selection-color: #000000;
            outline: none; /* Elimina la caja punteada fea de Windows */
        }
    
        /* Las opciones individuales dentro de la lista */
        QComboBox QAbstractItemView::item {
            padding: 8px 10px; /* Hace que las opciones sean más altas y fáciles de leer */
            min-height: 25px;
        }
        
    /* BOTONES ESTÁNDAR */
    QPushButton {
        background-color: #E1E1E1;
        border: 1px solid #ADADAD;
        border-radius: 3px;
        padding: 6px 12px;
        color: #222222;
    }
    QPushButton:hover {
        background-color: #D4D4D4;
        border: 1px solid #0078D7; /* Borde azul sutil al pasar el ratón (estilo Windows) */
    }
    QPushButton:pressed {
        background-color: #C8C8C8;
    }

    /* ESTILO PARA LOS 4 BOTONES PRINCIPALES (Efecto Tarjeta Blanca) */
    QPushButton#MenuCard {
        background-color: #FFFFFF; /* Blanco puro para que destaquen del fondo gris */
        color: #2B2B2B;
        border: 1px solid #D2D2D2; /* Borde gris muy suave */
        border-radius: 6px; 
        font-size: 15px;
        font-weight: bold;
    }
    QPushButton#MenuCard:hover {
        background-color: #F8FBFF; /* Un fondo ligerísimamente azulado */
        border: 1px solid #0078D7; /* El borde se vuelve azul corporativo */
        color: #005A9E; /* El texto se oscurece un poco en azul */
    }
    QPushButton#MenuCard:pressed {
        background-color: #E5F1FB;
        border: 1px solid #005499;
    }

    /* BOTÓN VERDE DE ACCIÓN PRINCIPAL ("Cargar Archivos", "Exportar") */
    QPushButton#BtnGreen {
        background-color: #6CB66C; /* Verde agradable */
        color: white;
        border: 1px solid #549A54;
        border-radius: 3px;
        font-weight: bold;
        padding: 8px;
    }
    QPushButton#BtnGreen:hover {
        background-color: #5CA55C;
        border: 1px solid #468446;
    }
    QPushButton#BtnGreen:pressed {
        background-color: #4A8C4A;
    }

    /* PESTAÑAS (TABS) */
    QTabWidget::pane {
        border: 1px solid #C0C0C0;
        background: #F0F2F5;
        top: -1px; /* Solapa el borde de la pestaña seleccionada */
    }
    QTabBar::tab {
        background: #E1E1E1;
        border: 1px solid #C0C0C0;
        padding: 6px 15px;
        margin-right: 2px;
        border-top-left-radius: 2px;
        border-top-right-radius: 2px;
    }
    QTabBar::tab:selected {
        background: #F0F2F5;
        border-bottom-color: #F0F2F5; /* Se fusiona con el fondo */
        font-weight: bold;
    }
    QTabBar::tab:hover:!selected {
        background: #ECECEC;
    }

    /* TÍTULOS GRANDES (Si conservas el menú principal) */
    QLabel#MainTitle {
        font-size: 24px;
        font-weight: bold;
        color: #333333;
    }
    
    /* CHECKBOXES */
    QCheckBox {
        spacing: 5px;
    }
    QCheckBox::indicator {
        width: 14px;
        height: 14px;
        border: 1px solid #ADADAD;
        background: #FFFFFF;
        border-radius: 2px;
    }
    QCheckBox::indicator:checked {
        background: #6CB66C;
        border: 1px solid #549A54;
    }
"""

class MainApp(QMainWindow):
    '''
    VENTANA PRINCIPAL (DASHBOARD)
    '''
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ultrafast Spectroscopy Analyzer")
        self.setMinimumSize(800, 400) 
        
        # Aplicamos directamente el nuevo estilo global
        self.setStyleSheet(STYLESHEET) 
        self.github_url = "https://github.com/AlejandroSerranoCapote/Ultrafast-Spectroscopy-Analyzer"
        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        main_layout.setContentsMargins(50, 40, 50, 30)
        main_layout.setSpacing(1)

        # 1. ENCABEZADO
        title = QLabel("SELECT ANALYSIS MODE")
        title.setObjectName("MainTitle")
        title.setAlignment(Qt.AlignCenter)
        
        subtitle = QLabel("Ultrafast Spectroscopy Processing Tools")
        subtitle.setAlignment(Qt.AlignCenter)

        main_layout.addWidget(title)
        main_layout.addWidget(subtitle)

        # 2. GRID DE BOTONES
        grid = QGridLayout()
        grid.setSpacing(20)

        # Textos
        txt_flups = "FLUPS ANALYZER"
        txt_tas   = "TAS ANALYZER"
        txt_fit   = "GLOBAL FIT"
        txt_xfel  = "2D MAPPER"    
        
        # Crear Botones
        self.btn_flups = self.create_card(txt_flups)
        self.btn_tas   = self.create_card(txt_tas)
        self.btn_fit   = self.create_card(txt_fit)
        self.btn_xfel  = self.create_card(txt_xfel)
        
        # Conectar
        self.btn_flups.clicked.connect(self.launch_flups)
        self.btn_tas.clicked.connect(self.launch_tas)
        self.btn_fit.clicked.connect(self.launch_global)
        self.btn_xfel.clicked.connect(self.launch_xfel)

        grid.addWidget(self.btn_flups, 0, 0)
        grid.addWidget(self.btn_tas, 0, 1)
        grid.addWidget(self.btn_fit, 1, 0)
        grid.addWidget(self.btn_xfel, 1, 1)

        main_layout.addLayout(grid)
        main_layout.addSpacing(20)

        # 3. FOOTER
        footer_layout = QVBoxLayout()
        footer_layout.setSpacing(10) 

        self.btn_github = QPushButton("View Source Code on GitHub")
        self.btn_github.setCursor(Qt.PointingHandCursor)
        self.btn_github.clicked.connect(self.open_github)

        # Centrar botón de GitHub
        h_center = QHBoxLayout()
        h_center.addStretch()
        h_center.addWidget(self.btn_github)
        h_center.addStretch()
        footer_layout.addLayout(h_center)

        # Label de descripción
        description = QLabel(
            "Welcome! This free and open-source software allows you to analyze "
            "ultrafast spectroscopy data directly from experiments such as "
            "<b>FLUPS</b> (Fluorescence Upconversion Spectroscopy) "
            ",<b>TAS</b> (Transient Absorption Spectroscopy) "
            "and <b>XTAS</b> (X-Ray Transient Absorption Spectroscopy) <br><br>"
            "For any questions or feedback, please contact:<br>"
            "<b>alejandro.serrano1610@gmail.com</b>"
        )
        description.setWordWrap(True)
        description.setAlignment(Qt.AlignCenter)
        
        footer_layout.addWidget(description)
        main_layout.addLayout(footer_layout)

    def create_card(self, text):
            btn = QPushButton(text)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setMinimumHeight(80) 
            
            btn.setObjectName("MenuCard") 
            
            return btn

    def open_github(self):
        # Asegúrate de tener self.github_url definido en tu __init__ si lo quitaste
        if hasattr(self, 'github_url'):
            QDesktopServices.openUrl(QUrl(self.github_url))

    def open_tool(self, tool_window):
        """
        Oculta el menú principal, abre la herramienta y configura
        que al cerrar la herramienta, el menú reaparezca.
        """
        self.current_tool = tool_window
        original_close = tool_window.closeEvent
        
        def on_close_tool(event):
            self.show()             
            original_close(event)  
            
        tool_window.closeEvent = on_close_tool
        tool_window.show()
        self.hide()
        
    def launch_xfel(self):
        window = XFELWindow() 
        self.open_tool(window)

    def launch_flups(self):
        window = FLUPSAnalyzer()
        self.open_tool(window)

    def launch_tas(self):
        window = TASAnalyzer()
        self.open_tool(window)

    def launch_global(self):
        window = GlobalFitPanel()
        self.open_tool(window)
        
class FLUPSAnalyzer(QMainWindow):

    def __init__(self):
            super().__init__()
            self.setWindowTitle("FLUPS Analyzer")
            
            screen = QApplication.primaryScreen()
            screen_geom = screen.availableGeometry() # Tamaño útil (sin barra de tareas)
            
            w_target = int(screen_geom.width() * 0.85)
            h_target = int(screen_geom.height() * 0.90)
            
            x_pos = (screen_geom.width() - w_target) // 2 + screen_geom.left()
            y_pos = screen_geom.top() + 35
            
            self.setGeometry(x_pos, y_pos, w_target, h_target)
            self.setMinimumSize(1000, 700)
        
            # estado
            self.WL = None
            self.TD = None
            self.data = None
            self.file_path = None
            self.data_corrected = None
            self.result_fit = None
            self.use_discrete_levels = True  # Cambia a False mapa continuo
            
            self.bg_cache = None
            self.cid_draw = None 
            self._is_drawing = False
    
            # widgets
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
                
            # elementos interactivos
            self.pcm = None
            self.cbar = None
            self.marker_map = None
            self.vline_map = None
            self.hline_map = None
            self.fit_line_artist = None
        
            self._init_small_plots()
        
            # Layout principal de la ventana
            layout = QVBoxLayout()
    
            # Layout Top (Botones)
            top_layout = QHBoxLayout()
            top_layout.addWidget(self.btn_load)
            top_layout.addWidget(self.label_status)
            top_layout.addWidget(self.btn_plot)
            top_layout.addWidget(self.btn_select)
            top_layout.addWidget(self.btn_fit)
            top_layout.addWidget(self.btn_show_corr)
            top_layout.addWidget(self.btn_remove_fringe)
            top_layout.addWidget(self.btn_global_fit)
            layout.addLayout(top_layout)
            
            # Añadir Canvas
            layout.addWidget(self.canvas)
    
            # ===================================================================
            # BLOQUE INFERIOR: CONTROLES CENTRADOS
            # ===================================================================
            
            # Este es el layout que guarda todos los controles (y donde TAS inyectará)
            self.bottom_controls_layout = QHBoxLayout()
            self.bottom_controls_layout.setSpacing(25) 
            
            # 1. --- Delay ---
            delay_layout = QVBoxLayout()
            delay_layout.setSpacing(5)
            delay_layout.addWidget(QLabel("Delay min (ps):"))
            self.xmin_edit = QLineEdit("-1")
            self.xmin_edit.setFixedWidth(50)
            delay_layout.addWidget(self.xmin_edit)
            
            delay_layout.addWidget(QLabel("Delay max (ps):"))
            self.xmax_edit = QLineEdit("3")
            self.xmax_edit.setFixedWidth(50)
            delay_layout.addWidget(self.xmax_edit)
            
            self.btn_apply_xlim = QPushButton("Apply X limits")
            self.btn_apply_xlim.setFixedWidth(120)
            self.btn_apply_xlim.clicked.connect(self.apply_x_limits)
            delay_layout.addWidget(self.btn_apply_xlim)
            delay_layout.addStretch() # Empuja arriba
            
            # 2. --- Wavelength ---
            wl_layout = QVBoxLayout()
            wl_layout.setSpacing(5)
            
            # λ min
            wl_min_layout = QHBoxLayout()
            wl_min_label = QLabel("λ min:")
            self.lbl_min_value = QLabel("400") 
            self.lbl_min_value.setCursor(Qt.PointingHandCursor)
            self.lbl_min_value.setToolTip("Haz clic para introducir valor exacto")
            self.lbl_min_value.installEventFilter(self) # Hacemos que la ventana escuche a este label
            
            self.slider_min = QSlider(Qt.Horizontal)
            self.slider_min.setMinimumWidth(200)
            self.slider_min.setMinimum(400)
            self.slider_min.setMaximum(800)
            self.slider_min.setValue(500)
            self.slider_min.valueChanged.connect(self.update_wl_range)
            wl_min_layout.addWidget(wl_min_label)
            wl_min_layout.addWidget(self.slider_min)
            wl_min_layout.addWidget(self.lbl_min_value)
            wl_layout.addLayout(wl_min_layout)
            
            # λ max
            wl_max_layout = QHBoxLayout()
            wl_max_label = QLabel("λ max:")
            self.lbl_max_value = QLabel("800") 
            self.lbl_max_value.setCursor(Qt.PointingHandCursor)
            self.lbl_max_value.setToolTip("Haz clic para introducir valor exacto")
            self.lbl_max_value.installEventFilter(self) # Hacemos que la ventana escuche a este label
            
            self.slider_max = QSlider(Qt.Horizontal)
            self.slider_max.setMinimumWidth(200)
            self.slider_max.setMinimum(400)
            self.slider_max.setMaximum(800)
            self.slider_max.setValue(700)
            self.slider_max.valueChanged.connect(self.update_wl_range)
            wl_max_layout.addWidget(wl_max_label)
            wl_max_layout.addWidget(self.slider_max)
            wl_max_layout.addWidget(self.lbl_max_value)
            wl_layout.addLayout(wl_max_layout)
            wl_layout.addStretch() # Empuja arriba
    
            # 3. --- Dial Levels ---
            dial_layout = QVBoxLayout()
            self.n_levels = 30
            self.dial_levels = QDial()
            self.dial_levels.setRange(2, 100)
            self.dial_levels.setValue(self.n_levels)
            self.dial_levels.setNotchesVisible(True)
            self.dial_levels.setWrapping(False)
            self.dial_levels.setFixedSize(80, 80)
            self.dial_levels.valueChanged.connect(self.update_n_levels)
            self.lbl_dial = QLabel(f"{self.n_levels}")
            self.lbl_dial.setAlignment(Qt.AlignCenter)
            dial_layout.addWidget(self.dial_levels, alignment=Qt.AlignCenter)
            dial_layout.addWidget(self.lbl_dial, alignment=Qt.AlignCenter)
            dial_layout.addStretch()
    
            # 4. --- Combo Box (Modelo t0) ---
            combo_layout = QVBoxLayout()
            lbl_model = QLabel("Chirp model fit ( t<sub>0</sub> ):")
            self.combo_model = QComboBox()
            self.combo_model.addItems(["Polynomial", "Non linear"])
            self.combo_model.setCurrentIndex(1)
            combo_layout.addWidget(lbl_model)
            combo_layout.addWidget(self.combo_model)
            combo_layout.addStretch()
    
        
            # 5. --- Escala Eje Y ---
            scale_layout = QVBoxLayout()
            scale_layout.setSpacing(5)
            
            scale_layout.addWidget(QLabel("Y-Axis Scale:"))
            self.combo_scale = QComboBox()
            self.combo_scale.addItems(["SymLog", "Linear"])
            self.combo_scale.setCurrentIndex(0) # SymLog por defecto
            scale_layout.addWidget(self.combo_scale)
            
            self.lbl_linthresh = QLabel("Linthresh (ps):")
            self.spin_linthresh = QDoubleSpinBox()
            self.spin_linthresh.setDecimals(2)
            self.spin_linthresh.setRange(0.01, 1000.0) # Rango amplio para jugar
            self.spin_linthresh.setValue(1.0) # Valor por defecto
            self.spin_linthresh.setSingleStep(0.5)
            
            scale_layout.addWidget(self.lbl_linthresh)
            scale_layout.addWidget(self.spin_linthresh)
            scale_layout.addStretch()
            
            # Conectar a la función que actualiza la gráfica instantáneamente
            self.combo_scale.currentIndexChanged.connect(self.apply_y_scale)
            self.spin_linthresh.valueChanged.connect(self.apply_y_scale)
    
            # --- Empaquetar todo en el layout de controles ---
            self.bottom_controls_layout.addLayout(delay_layout)
            self.bottom_controls_layout.addLayout(wl_layout)
            self.bottom_controls_layout.addLayout(dial_layout)
            
            # Insertamos el nuevo control aquí:
            self.bottom_controls_layout.addLayout(scale_layout) 
            
            self.bottom_controls_layout.addLayout(combo_layout)
        
            
            center_bottom_layout = QHBoxLayout()
            center_bottom_layout.addStretch() # Resorte izquierdo
            center_bottom_layout.addLayout(self.bottom_controls_layout)
            center_bottom_layout.addStretch() # Resorte derecho
            
            layout.addLayout(center_bottom_layout)

            self.bottom_controls_layout.setSpacing(60) 
           
            self.bottom_controls_layout.setContentsMargins(60, 10, 60, 0) 
            
           
            layout.addLayout(self.bottom_controls_layout)
            
            # Set central widget
            container = QWidget()
            container.setLayout(layout)
            self.setCentralWidget(container)
            
            # --- fit de colores de los ejes principales y colorbars ---
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
            """Captura el fondo para Blitting con protección anti-recursión."""
           
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
        """Aplica la escala Y seleccionada y actualiza el gráfico al instante."""
        is_symlog = self.combo_scale.currentText() == "SymLog"
       
        self.lbl_linthresh.setVisible(is_symlog)
        self.spin_linthresh.setVisible(is_symlog)
        
        # Si el mapa aún no existe, no hacemos nada
        if not hasattr(self, 'ax_map') or self.data is None:
            return

        # Aplicar la escala
        if is_symlog:
            # Lee el valor exacto del spinbox para fijar dónde empieza la parte logarítmica
            self.ax_map.set_yscale("symlog", linthresh=self.spin_linthresh.value())
            self.ax_map.set_ylabel("Delay (ps) - SymLog")
        else:
            self.ax_map.set_yscale("linear")
            self.ax_map.set_ylabel("Delay (ps) - Linear")
            
        # Redibujar solo lo necesario
        self.canvas.draw_idle()
    def draw_animated_artists(self):
        """Dibuja solo los elementos móviles."""
        # Mapa
        if self.vline_map: self.ax_map.draw_artist(self.vline_map)
        if self.hline_map: self.ax_map.draw_artist(self.hline_map)
        if self.marker_map: self.ax_map.draw_artist(self.marker_map)
        
        if self.cut_time_small: self.ax_time_small.draw_artist(self.cut_time_small)
        if self.vline_time_small: self.ax_time_small.draw_artist(self.vline_time_small)
        if self.cut_spec_small: self.ax_spec_small.draw_artist(self.cut_spec_small)

    def eventFilter(self, obj, event):
            """Intercepta eventos específicos de los widgets observados."""
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                if obj == self.lbl_min_value:
                    self.prompt_exact_wl_min()
                    return True # Indicamos que el evento ya ha sido procesado
                elif obj == self.lbl_max_value:
                    self.prompt_exact_wl_max()
                    return True
            
            # Dejar que el resto de eventos se procesen de forma normal
            return super().eventFilter(obj, event)

    def prompt_exact_wl_min(self):
            """Abre un diálogo para establecer el valor mínimo de λ de forma precisa."""
            if getattr(self, "WL", None) is None: 
                return # Evita errores si no hay datos cargados
                
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
        """Abre un diálogo para establecer el valor máximo de λ de forma precisa."""
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
        dlg = GlobalFitPanel(self)
        dlg.exec_()
    def _init_small_plots(self):

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
        """Aplica los límites del eje X (Delay) escritos por el usuario."""
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
        """Quita la franja de bombeo directamente sobre los datos actuales."""
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
    
        # índices de la franja
        posl1 = np.argmin(np.abs(self.WL - (sWl - wisWL / 2)))
        posl2 = np.argmin(np.abs(self.WL - (sWl + wisWL / 2)))
    
        # modificar los datos directamente
        data_target[posl1:posl2, :] = 1e-10
    
        # refrescar el mapa para ver el efecto
        if getattr(self, "showing_corrected", False):
            self.toggle_corrected_map()  # volver a mostrar mapa corregido
        else:
            self.plot_map()  # volver a mostrar mapa original
    
        QMessageBox.information(
            self, "Pump fringe removed",
            f"Fringe at {sWl} ± {wisWL/2} nm has been set to near-zero."
        )

    def load_file(self):
        """Carga archivo de datos y normaliza ΔA automáticamente."""
        # Seleccionar CSV o data.txt
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select CSV or Data File", "", 
            "CSV Files (*.csv);;Data Files (*.txt *.dat)"
        )
        if not file_path:
            return
    
        try:
            # Cargar datos
            if file_path.endswith(".csv"):
                data, wl, td = load_data(auto_path=file_path)
            else:
                wl_path, _ = QFileDialog.getOpenFileName(self, "Select Wavelength File", "", "Text Files (*.txt)")
                td_path, _ = QFileDialog.getOpenFileName(self, "Select Delay File", "", "Text Files (*.txt)")
                if not wl_path or not td_path:
                    QMessageBox.warning(self, "Files missing", "You must select both WL and TD files.")
                    return
                data, wl, td = load_data(data_path=file_path, wl_path=wl_path, td_path=td_path)
    
            # ---  ELIMINAR DUPLICADOS Y ORDENAR (FLUPS) ---
            wl, idx_wl = np.unique(wl, return_index=True)
            td, idx_td = np.unique(td, return_index=True)
            
            # Recortar la matriz de datos para que coincida con los índices limpios
            data = data[idx_wl, :][:, idx_td]
    
            # --- Normalización ---
            
            # =============================================================================
            #             NORMALIZACIÓN DATOS EN FLUPS
            # =============================================================================

            max_val = np.nanmax(np.abs(data))
            if max_val != 0:
                data = data / max_val
    
    
            self.WL, self.TD, self.data = wl, td, data
            self.file_path = file_path

            #  Guarda también la ruta y el directorio base del CSV
            self.csv_path = file_path
            self.base_dir = os.path.dirname(file_path)
            
            self.label_status.setText(f"Loaded : {os.path.basename(file_path)}")
            self.btn_plot.setEnabled(True)
            self.btn_select.setEnabled(True)
            self.btn_fit.setEnabled(True)
    
            # Actualizar sliders
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
        min_val = self.slider_min.value()
        max_val = self.slider_max.value()
        print(f"Aplicando λ min={min_val}, λ max={max_val}")

    def _plot_discrete_map(self, ax, WL, TD, data, n_levels=5, cmap='jet', shading='auto', vmin=None, vmax=None):
        """Dibuja mapa tipo contourf con pcolormesh discreto."""

        if vmin is None:
            vmin = np.nanmin(data)
        if vmax is None:
            vmax = np.nanmax(data)
    
        levels = np.linspace(vmin, vmax, n_levels)
        norm = BoundaryNorm(levels, ncolors=plt.get_cmap(cmap).N, clip=True)
    
        pcm = ax.pcolormesh(WL, TD, data.T, shading=shading, cmap=cmap, norm=norm)
        return pcm
    def update_n_levels(self, value):
        """Actualiza el número de niveles del mapa discreto y redibuja el mapa, respetando el rango visible."""
        self.n_levels = value
        self.lbl_dial.setText(f"{value} levels")  #  actualiza texto
    
        if self.data is None:
            return
    
        #  Determinar qué datos y WL usar (respetando el rango visible actual)
        if hasattr(self, "WL_visible") and self.WL_visible is not None:
            WL_used = self.WL_visible
            if getattr(self, "showing_corrected", False):
                # si estamos mostrando el mapa corregido
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
    
        #  Redibujar mapa directamente (sin resetear)
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
        
        #  Colorbar
        divider = make_axes_locatable(self.ax_map)
        cax = divider.append_axes("right", size="3%", pad=0.02)
        self.cbar = self.figure.colorbar(self.pcm, cax=cax, label="ΔA")
    
        #  Estilo visual coherente
        self.ax_map.set_facecolor("white")
        for spine in self.ax_map.spines.values():
            spine.set_color("black")
        self.ax_map.tick_params(colors="black")
        self.ax_map.xaxis.label.set_color("black")
        self.ax_map.yaxis.label.set_color("black")
        self.ax_map.title.set_color("black")
    
        self.canvas.draw_idle()


    def plot_map(self):
            """Dibuja el mapa principal configurado para Blitting (alta velocidad)."""
            if self.data is None: return
    
            # Limpieza estándar
            self.ax_map.clear()
            if self.cbar:
                try: self.cbar.remove()
                except: pass
                self.cbar = None
    
            # --- Determinar datos a pintar (respetando filtros) ---
            WL_plot = self.WL_visible if hasattr(self, "WL_visible") and self.WL_visible is not None else self.WL
            data_plot = self.data_visible if hasattr(self, "data_visible") and self.data_visible is not None else self.data
    
            # 1. Dibujar Mapa (Estático)
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
    
            # 2. Inicializar Elementos Dinámicos (animated=True)
            x0, y0 = WL_plot[0], self.TD[0]
            
            self.vline_map = self.ax_map.axvline(x0, color='k', ls='--', lw=1, animated=True, zorder=6)
            self.hline_map = self.ax_map.axhline(y0, color='k', ls='--', lw=1, animated=True, zorder=6)
            self.marker_map, = self.ax_map.plot([x0], [y0], 'wx', markersize=8, markeredgewidth=2, animated=True, zorder=7)
    
            # 3. Preparar subplots pequeños (IMPORTANTE: Fijar límites aquí)
            self.ax_time_small.clear()
            self.ax_spec_small.clear()
            
            # Inicializamos líneas animadas vacías o con el primer valor
            self.cut_time_small, = self.ax_time_small.plot(self.TD, data_plot[0, :], 'b-', lw=1.5, animated=True)
            self.vline_time_small = self.ax_time_small.axvline(y0, color='k', ls='--', lw=1, animated=True)
            
            self.cut_spec_small, = self.ax_spec_small.plot(WL_plot, data_plot[:, 0], 'r-', lw=1.5, animated=True)
    
            # --- FIJAR LÍMITES ESTÁTICOS ---
            vmin_g, vmax_g = np.nanmin(data_plot), np.nanmax(data_plot)
            margin = (vmax_g - vmin_g) * 0.05
            
            self.ax_time_small.set_xlim(self.TD.min(), self.TD.max())
            self.ax_time_small.set_ylim(vmin_g - margin, vmax_g + margin)
            self.ax_time_small.set_xlabel("Delay (ps)")
            self.ax_time_small.set_title("Kinetics (Preview)") # Título estático
    
            self.ax_spec_small.set_xlim(WL_plot.min(), WL_plot.max())
            self.ax_spec_small.set_ylim(vmin_g - margin, vmax_g + margin)
            self.ax_spec_small.set_xlabel("Wavelength (nm)")
            self.ax_spec_small.set_title("Spectra (Preview)") # Título estático
    
            # Conectar eventos
            if self.cid_click is None:
                self.cid_click = self.canvas.mpl_connect("button_press_event", self.on_click_map)
    
            # 4. Disparar el primer dibujado completo (Genera el bg_cache)
            self.canvas.draw()

    
    def update_wl_range(self):
            """
            Actualiza las variables de datos visibles según los sliders 
            y llama a plot_map para dibujar todo correctamente.
            """
            if getattr(self, "WL", None) is None or getattr(self, "data", None) is None:
                 # Actualizar textos a guiones si no hay datos
                if hasattr(self, "lbl_min_value"): self.lbl_min_value.setText("- nm")
                if hasattr(self, "lbl_max_value"): self.lbl_max_value.setText("- nm")
                return
    
            # 1. Obtener índices de los sliders
            wl_min_idx = int(self.slider_min.value())
            wl_max_idx = int(self.slider_max.value())
    
            # 2. Corregir cruces de índices
            if wl_min_idx >= wl_max_idx: 
                wl_max_idx = wl_min_idx + 1
            
            # Asegurar límites del array
            wl_min_idx = max(0, min(wl_min_idx, len(self.WL) - 1))
            wl_max_idx = max(0, min(wl_max_idx, len(self.WL) - 1))
    
            # 3. Actualizar Etiquetas de Texto (nm)
            try:
                self.lbl_min_value.setText(f"{self.WL[wl_min_idx]:.1f} nm")
                self.lbl_max_value.setText(f"{self.WL[wl_max_idx]:.1f} nm")
            except Exception:
                pass
    
            # 4. DEFINIR LOS DATOS VISIBLES (Estado Global de Visualización)
            source_data = self.data_corrected if getattr(self, "showing_corrected", False) else self.data
            
            # Cortamos los datos
            self.WL_visible = self.WL[wl_min_idx : wl_max_idx + 1]
            self.data_visible = source_data[wl_min_idx : wl_max_idx + 1, :]
    
            # 5. LLAMADA CENTRALIZADA
            self.plot_map()
            
    def enable_point_selection(self):
        self.clicked_points = []
        if self.cid_click is None:
            self.cid_click = self.canvas.mpl_connect("button_press_event", self.on_click_map)
        QMessageBox.information(self, "Mode: Select points",
                                "Click izquierdo: añadir punto\nClick derecho: borrar último punto.\nLuego pulsa 'Fit t₀'.")
    def update_small_cuts(self, x, y, WL_sel=None, data_sel=None):
            """Actualización completa tras un clic."""
            # Reutilizamos la lógica del movimiento simulando un evento
            # Esto asegura coherencia visual
            class MockEvent:
                pass
            evt = MockEvent()
            evt.xdata = x
            evt.ydata = y
            evt.inaxes = self.ax_map
            
            # Llamamos a on_move_map para pintar rápido
            self.on_move_map(evt)
            
            # Si fue un clic, aseguramos que se quede fijo (opcional)
            # self.canvas.draw_idle()
    
    def on_click_map(self, event):
        """Registrar puntos sobre el mapa (izq añade, derecha borra último) y actualizar cortes."""
        if event.inaxes != self.ax_map:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        if event.button == 1:  # left click -> añadir punto
            artist, = self.ax_map.plot(x, y, 'wo', markeredgecolor='k', markersize=6, zorder=6)
            self.clicked_points.append({'x': x, 'y': y, 'artist': artist})
        elif event.button == 3 and self.clicked_points:  # right click -> borrar último
            last = self.clicked_points.pop()
            try:
                last['artist'].remove()
            except Exception:
                pass

        # actualizar marker que sigue al cursor (opcional: mover el marker principal)
        if self.marker_map is None:
            self.marker_map, = self.ax_map.plot([x], [y], 'wx', markersize=8, markeredgewidth=2)
        else:
            self.marker_map.set_data([x], [y])


        # actualizar referencias visuales de la línea vertical
        if self.vline_map is None:
            # si no existe, crearla
            self.vline_map = self.ax_map.axvline(x, color='k', ls='--', lw=1)
        else:
            # si ya existe, solo actualizar su posición y asegurar que sea visible
            self.vline_map.set_xdata([x, x])
            self.vline_map.set_visible(True)
        
        # actualizar referencias visuales de la línea horizontal
        if self.hline_map is None:
            self.hline_map = self.ax_map.axhline(y, color='k', ls='--', lw=1)
        else:
            self.hline_map.set_ydata([y, y])
            self.hline_map.set_visible(True)

        # --- aquí está la diferencia: actualizar los subplots pequeños ---
        self.update_small_cuts(x, y)
        self.update_small_cuts(
            x, y,
            WL_sel=getattr(self, "WL_visible", None),
            data_sel=getattr(self, "data_visible", None)
        )
        self.canvas.draw_idle()

    def on_move_map(self, event):
            
            # Si no hay caché o no estamos en el eje, salir
            if self.bg_cache is None or self.data is None: 
                return
            if event.inaxes != self.ax_map: 
                return
    
            # 1. Restaurar fondo limpio (borra cursores anteriores instantáneamente)
            self.canvas.restore_region(self.bg_cache)
    
            # 2. Actualizar posiciones matemáticas (sin dibujar aún)
            x, y = event.xdata, event.ydata
            if x is None or y is None: return
            
            self._last_cursor_x = x
            self._last_cursor_y = y
            
            # Líneas del mapa
            self.vline_map.set_xdata([x, x])
            self.hline_map.set_ydata([y, y])
            self.marker_map.set_data([x], [y])
            
            # Calcular índices para los subplots
            # Usamos WL_visible si existe, si no WL completo
            cur_WL = self.WL_visible if hasattr(self, 'WL_visible') and self.WL_visible is not None else self.WL
            cur_data = self.data_visible if hasattr(self, 'data_visible') and self.data_visible is not None else self.data
            
            if cur_WL is not None and len(cur_WL) > 0:
                idx_wl = int(np.abs(cur_WL - x).argmin())
                idx_td = int(np.abs(self.TD - y).argmin())
    
                # Actualizar curvas pequeñas
                self.cut_time_small.set_data(self.TD, cur_data[idx_wl, :])
                self.vline_time_small.set_xdata([y, y])
                self.cut_spec_small.set_data(cur_WL, cur_data[:, idx_td])
    
                # Info en barra de estado
                val = cur_data[idx_wl, idx_td]
                self.label_status.setText(f"Cursor: {x:.1f} nm, {y:.2f} ps | Val: {val:.4e}")
    
            # 3. Dibujar SOLO lo animado y volcar a pantalla
            self.draw_animated_artists()
            self.canvas.blit(self.figure.bbox)

    def fit_t0_points(self):
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
        
        # Intentar el ajuste
        try:
            result = fit_t0(w_points, t0_points, self.WL, self.TD, self.data, mode=mode)
        except Exception as e:
            QMessageBox.critical(self, "Error de ajuste t₀", str(e))
            return

        self.result_fit = result
        self.data_corrected = result['corrected']

        # dibujar curva del fit sobre mapa principal
        if self.fit_line_artist is not None:
            try:
                self.fit_line_artist.remove()
            except Exception:
                pass
        self.fit_line_artist, = self.ax_map.plot(result['fit_x'], result['fit_y'], 'r-', lw=2, label="t₀ fit")
        self.ax_map.legend()
        self.canvas.draw_idle()

        # guardado automático (idéntico a tu comportamiento actual)
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


    def toggle_corrected_map(self):
            """Alterna entre mapa original y corregido usando el renderizado optimizado."""
            
            # 1. Validación de seguridad
            if self.data_corrected is None:
                QMessageBox.warning(self, "No corrected data", "Run 'Fit t₀' first.")
                return
    
            # 2. Alternar estado (flag booleano)
            self.showing_corrected = not getattr(self, "showing_corrected", False)
    
            # 3. Decidir la fuente de datos
            # Si showing_corrected es True, usamos los datos corregidos.
            # Si es False, usamos self.data (que es el base/original).
            source_data = self.data_corrected if self.showing_corrected else self.data
    
            # 4. Actualizar Textos
            if self.showing_corrected:
                self.btn_show_corr.setText("Show Base/Original Map")
                suffix = "(t₀ Corrected)"
            else:
                self.btn_show_corr.setText("Show Corrected Map")
                suffix = "(Base/Original)"
    
            # 5. Recalcular el slice visible (Respetando los Sliders)
            
            if hasattr(self, 'slider_min') and hasattr(self, 'slider_max'):
                wl_min_idx = int(self.slider_min.value())
                wl_max_idx = int(self.slider_max.value())
                
                # Protecciones de índice
                if wl_min_idx >= wl_max_idx: wl_max_idx = wl_min_idx + 1
                wl_min_idx = max(0, min(wl_min_idx, len(self.WL) - 1))
                wl_max_idx = max(0, min(wl_max_idx, len(self.WL) - 1))
                
                # Actualizamos las variables que usa plot_map
                self.WL_visible = self.WL[wl_min_idx:wl_max_idx+1]
                self.data_visible = source_data[wl_min_idx:wl_max_idx+1, :]
            else:
                # Fallback por si no hay sliders
                self.WL_visible = self.WL
                self.data_visible = source_data
                
            self.plot_map()
            
            # Actualizamos el título explícitamente para reflejar el estado
            tech_name = "TAS" if getattr(self, "is_TAS_mode", False) else "FLUPS"
            self.ax_map.set_title(f"ΔA Map ({tech_name}) {suffix}")
            
            # Un redraw final para asegurar que el título se actualice
            self.canvas.draw()
    
class TASAnalyzer(FLUPSAnalyzer):
    def __init__(self):
                super().__init__()
                self.setWindowTitle("TAS Analyzer")
                self.label_status.setText("No file loaded")
                
                # --- Datos ---
                self.medida = None
                self.solvente = None
                self.pump_mask = None  # nueva variable para eliminar pump
                self.TDSol = None
                self.WLSol = None
                self.is_TAS_mode = True
                self.use_discrete_levels = False 
                
    
                self.dial_levels.hide()
                self.lbl_dial.hide()
                
                # --- Inicializar variables para Blitting (optimización) ---
                self.bg_cache = None
                self.cid_draw = None
                self.cid_click = None
                self.cid_move = None
                
                # ===================================================================
                # SLIDERS Y ETIQUETAS CLICABLES (AM y SF)
                # ===================================================================
    
                # 1. Slider y Etiqueta para Amplitud (AM)
                self.slider_am = QSlider(Qt.Horizontal)
                self.slider_am.setMinimumWidth(200)
                self.slider_am.setMinimum(0)
                self.slider_am.setMaximum(200)
                self.slider_am.setValue(100)  # 100% por defecto
                
                self.lbl_am_value = QLabel("100 %")
                self.lbl_am_value.setCursor(Qt.PointingHandCursor)
                self.lbl_am_value.setToolTip("Haz clic para introducir valor exacto")
                self.lbl_am_value.installEventFilter(self)
                
                self.slider_am.valueChanged.connect(self.on_am_changed)
                
                # 2. Slider y Etiqueta para Shift temporal (SF)
                self.slider_sf = QSlider(Qt.Horizontal)
                self.slider_sf.setMinimumWidth(200)
                self.slider_sf.setMinimum(-20000)
                self.slider_sf.setMaximum(20000)
                self.slider_sf.setValue(0)
                
                self.lbl_sf_value = QLabel("0.000 ps")
                self.lbl_sf_value.setCursor(Qt.PointingHandCursor)
                self.lbl_sf_value.setToolTip("Haz clic para introducir valor exacto")
                self.lbl_sf_value.installEventFilter(self)
                
                self.slider_sf.valueChanged.connect(self.on_sf_changed)
                
                # ===================================================================
                # INYECCIÓN DE LAYOUT TAS EN EL CENTRO
                # ===================================================================
                
                tas_extra_layout = QVBoxLayout()
                tas_extra_layout.setSpacing(5)
        
                # Fila Amplitud
                amp_row = QHBoxLayout()
                amp_row.addWidget(QLabel("Amplitude (%):"))
                amp_row.addWidget(self.slider_am)
                amp_row.addWidget(self.lbl_am_value) # <--- Añadimos la etiqueta aquí
                tas_extra_layout.addLayout(amp_row)
        
                # Fila Shift
                shift_row = QHBoxLayout()
                shift_row.addWidget(QLabel("Shift (ps):"))
                shift_row.addWidget(self.slider_sf)
                shift_row.addWidget(self.lbl_sf_value) # <--- Añadimos la etiqueta aquí
                tas_extra_layout.addLayout(shift_row)
                
                tas_extra_layout.addStretch() # Empuja el bloque hacia arriba para alinear con el resto
        
                # Inyectamos este bloque en el layout principal inferior (índice 2 está entre WL y el Combo)
                if hasattr(self, 'bottom_controls_layout'):
                    self.bottom_controls_layout.insertLayout(2, tas_extra_layout)
         
                # === Checkbox para conversión automática de .dat  --> .csv ===
                self.chk_convert_dat = QCheckBox("Convert .dat → .csv (IMDEA DATA)")
                self.chk_convert_dat.setChecked(True)  # activado por defecto
        
                # Lo centramos y lo ponemos al fondo del todo
                chk_layout = QHBoxLayout()
                chk_layout.addStretch()
                chk_layout.addWidget(self.chk_convert_dat)
                chk_layout.addStretch()
                
                self.centralWidget().layout().addLayout(chk_layout)
                
    def on_am_changed(self, value):
        """Actualiza el texto de amplitud y recalcula el mapa."""
        self.lbl_am_value.setText(f"{value} %")
        self.update_am_sf()

    def on_sf_changed(self, value):
        """Actualiza el texto del shift y recalcula el mapa."""
        self.lbl_sf_value.setText(f"{value / 100.0:.3f} ps")
        self.update_am_sf()

    def prompt_exact_am(self):
        """Diálogo para introducir la amplitud exacta."""
        current_val = self.slider_am.value()
        val, ok = QInputDialog.getDouble(
            self, "Exact Amplitude", "Enter amplitude (%):", 
            value=current_val, decimals=0, min=0, max=200
        )
        if ok:
            self.slider_am.setValue(int(val))

    def prompt_exact_sf(self):
        """Diálogo para introducir el shift exacto."""
        current_val = self.slider_sf.value() / 100.0
        val, ok = QInputDialog.getDouble(
            self, "Exact Shift", "Enter shift (ps):", 
            value=current_val, decimals=3, min=-200.0, max=200.0
        )
        if ok:
            self.slider_sf.setValue(int(val * 100))

    def eventFilter(self, obj, event):
        """Intercepta los clics en las etiquetas de Amplitude y Shift."""
        if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            if obj == getattr(self, "lbl_am_value", None):
                self.prompt_exact_am()
                return True
            elif obj == getattr(self, "lbl_sf_value", None):
                self.prompt_exact_sf()
                return True
                
        # Pasamos el resto de eventos (como los de las WL) a la clase padre
        return super().eventFilter(obj, event)
    def switch_analyzer(self):
        """Cambia entre FLUPSAnalyzer y TASAnalyzer sin cerrar la nueva ventana."""
        try:
            target_cls_name = "FLUPSAnalyzer" if isinstance(self, TASAnalyzer) else "TASAnalyzer"
            
            if target_cls_name in globals() and callable(globals()[target_cls_name]):
                TargetCls = globals()[target_cls_name]
            else:
                raise NameError(f"{target_cls_name} not found")

            # Guarda la referencia en self (no variable local)
            self.new_window = TargetCls()
            self.new_window.show()

            # Cierra la ventana actual
            self.close()

        except Exception as e:
            QMessageBox.critical(self, "Switch error", f"Cannot switch analyzer:\n{e}")

    def convert_dat_to_csv(self, file_path):
        """Convierte un archivo .dat en un .csv con el formato TAS."""
        try:
            data = np.loadtxt(file_path)
    
            # wl = primera columna
            wl = data[:, 0]
    
            # t = primera fila (ps)
            t = data[0] * 1e-3

            # Reemplazar en la matriz
            data[:, 0] = wl
            data[0, :] = t
    
            # Crear ruta .csv
            csv_path = os.path.splitext(file_path)[0] + ".csv"
    
            # Guardar
            np.savetxt(csv_path, data, delimiter=",")
            return csv_path
    
        except Exception as e:
            QMessageBox.critical(self, "Conversion error", f"Cannot convert .dat:\n{e}")
            return None
    def get_base_dir(self):
        """
        Devuelve la carpeta donde se encuentra el CSV de medida.
        Crea subcarpetas 'fit' y 'plots' si no existen.
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
        if self.data is None:
            QMessageBox.warning(self, "No data", "Load TAS data first.")
            return
    
        sWl, ok1 = QInputDialog.getDouble(self, "Pump wavelength", "Pump wavelength (nm):", min=0.0)
        if not ok1: return
        wisWL, ok2 = QInputDialog.getDouble(self, "Width of scattering", "Width of pump scattering (nm):", min=0.0)
        if not ok2: return
    
        posl1 = np.argmin(np.abs(self.WL - (sWl - wisWL / 2)))
        posl2 = np.argmin(np.abs(self.WL - (sWl + wisWL / 2)))
    
        # crear o actualizar máscara
        if self.pump_mask is None:
            self.pump_mask = np.zeros_like(self.medida, dtype=bool)
        self.pump_mask[posl1:posl2, :] = True
    
        # aplicar máscara sobre self.data
        self.update_am_sf()
    
        QMessageBox.information(self, "Pump fringe removed",
                                f"Fringe at {sWl} ± {wisWL/2} nm will be zeroed.")

    # ------------------------------------------------------------------
    # CARGA DE ARCHIVOS
    # ------------------------------------------------------------------
    def load_file(self):
        """Carga los ficheros de medida y solvente TAS (CSV o TXT)."""
        
        # --- Seleccionar archivo de medida ---
        file_path_medida, _ = QFileDialog.getOpenFileName(
            self,
            "Select Measurement CSV",
            "",
            "CSV Files (*.csv);;Data Files (*.txt *.dat)"
        )
        if not file_path_medida or not os.path.exists(file_path_medida):
            self.label_status.setText(" No measurement file selected.")
            return
        # --- Conversión automática .dat → .csv si la opción está activada ---
        if self.chk_convert_dat.isChecked() and file_path_medida.lower().endswith(".dat"):
            new_path = self.convert_dat_to_csv(file_path_medida)
            if new_path:
                file_path_medida = new_path
        #  Guardar ruta del primer CSV leído
        self.file_path = file_path_medida
        
        #  Crear carpeta específica para esta medición
        base_dir = os.path.dirname(self.file_path)
        base_name = os.path.splitext(os.path.basename(self.file_path))[0]
        self.results_dir = os.path.join(base_dir, f"{base_name}_results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        raw = pd.read_csv(file_path_medida, header=None)
        raw = raw.apply(pd.to_numeric, errors="coerce").dropna(how="any")
        raw = raw.values.astype(float)
        
        temp_TD = raw[0, 1:]       # delay (ps) temporal
        temp_WL = raw[1:, 0]       # wavelength (nm) temporal
        temp_medida = raw[1:, 1:]  # Matriz 2D intensidad
        temp_medida[np.isnan(temp_medida)] = 0
        
        # --- ELIMINAR DUPLICADOS Y ORDENAR (MEDIDA) ---
        self.WL, idx_wl = np.unique(temp_WL, return_index=True)
        self.TD, idx_td = np.unique(temp_TD, return_index=True)
        # Recortar la matriz de datos para que coincida con los índices limpios
        self.medida = temp_medida[idx_wl, :][:, idx_td]
        
        # --- Seleccionar archivo de solvente ---
        file_path_solvente, _ = QFileDialog.getOpenFileName(
            self,
            "Select Solvent CSV",
            os.path.dirname(self.file_path),  # ✅ Abre el diálogo en la misma carpeta
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
        
        # ---  ELIMINAR DUPLICADOS Y ORDENAR (SOLVENTE) ---
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
        # --- Calcular mapa inicial ---
        self.label_status.setText(" TAS data loaded")
        self.update_am_sf()
        self.plot_map()
        
        # --- Definir ruta base para compatibilidad con FLUPSAnalyzer ---
        self.file_path = file_path_medida
        
        if hasattr(self, "btn_plot"):
            self.btn_plot.setEnabled(True)
        if hasattr(self, "btn_select"):
            self.btn_select.setEnabled(True)
        if hasattr(self, "btn_fit"):
            self.btn_fit.setEnabled(True)
        
        #  Mostrar solo el nombre del archivo cargado
        file_name = os.path.basename(file_path_medida)
        self.label_status.setText(f"TAS data loaded from: {file_name}")


    # En TASAnalyzer (reemplazar la versión actual)
    def fit_t0_points(self):
        if not getattr(self, "clicked_points", None) or len(self.clicked_points) < 2:
            QMessageBox.warning(self, "Not enough points", "Select at least 2 points on the map.")
            return
    
        w_points = np.array([p['x'] for p in self.clicked_points])
        t0_points = np.array([p['y'] for p in self.clicked_points])
    
        try:
            # Re-calcular la base (self.data) con el solvente/shift más reciente
            self.update_am_sf() 
            
            # Usar self.data (Base Data: solvente-corregida) para el fit
            result = fit_t0(w_points, t0_points, self.WL, self.TD, self.data)
        except Exception as e:
            QMessageBox.critical(self, "Fit error", str(e))
            return
    
        
        # --- Guardar datos corregidos globalmente ---
        self.result_fit = result
        self.data_corrected = result['corrected']
        
        self.plot_map(show_fit=True)
        self.btn_show_corr.setEnabled(True)
    
        # --- Crear carpeta de resultados junto al CSV y guardar ---
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
            """Actualiza los índices de recorte y refresca el mapa."""
            if self.medida is None:
                return
    
            # 1. Leer valores de los sliders
            # Aseguramos que son enteros (índices del array)
            s_min = int(self.slider_min.value())
            s_max = int(self.slider_max.value())
    
            # 2. Validar cruce (Min no puede ser >= Max)
            if s_min >= s_max:
                s_min = s_max - 1
                if s_min < 0: s_min = 0
                self.slider_min.blockSignals(True) # Evitar bucle infinito
                self.slider_min.setValue(s_min)
                self.slider_min.blockSignals(False)
    
            # 3. Guardar en las variables de clase
            self.idx_min = s_min
            self.idx_max = s_max
    
        # 4. Actualizar etiquetas de texto 
            try:
                self.lbl_min_value.setText(f"{self.WL[s_min]:.1f} nm")
                self.lbl_max_value.setText(f"{self.WL[s_max]:.1f} nm")
            except Exception:
                pass
    
            # 5. Redibujar
            self.plot_map()
    # ------------------------------------------------------------------
    # ACTUALIZACIÓN DE MAPA TRAS SLIDERS
        # ------------------------------------------------------------------
    # En TASAnalyzer (reemplazar la versión actual)
    def update_am_sf(self):
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
    
        # Base: medida - solvente
        base_data = self.medida - solvente_interp
    
        # Aplicar máscara si existe
        if self.pump_mask is not None:
            base_data[self.pump_mask] = 1e-10
        
        # Se elimina todo el bloque 'if hasattr(self, "data_corrected") ...' que causaba el doble cálculo.
        self.data = base_data 
    
        self.update_wl_range() 
    
        if hasattr(self, "global_fit_panel") and self.global_fit_panel is not None:
            self.global_fit_panel.update_from_parent()
    
        self._updating_am_sf = False
    # ------------------------------------------------------------------
    # DIBUJAR MAPA ΔA
    # ------------------------------------------------------------------
    def plot_map(self, show_fit=False):
            """Dibuja el mapa (SymLog en Y) con soporte para modo Corregido/Original."""
            
            # 1. Determinar qué datos usar (Base vs Corregidos)
            # Verificamos el flag que activa el botón toggle
            showing_corrected = getattr(self, "showing_corrected", False)
            
            if showing_corrected and hasattr(self, "data_corrected") and self.data_corrected is not None:
                source_data = self.data_corrected
                mode_suffix = "(t₀ Corrected)"
            else:
                # Si no hay flag o es False, usamos self.data (que ya tiene la resta de solvente)
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
                
            # --- Limpieza ---
            self.ax_map.clear()
            self.ax_time_small.clear()
            self.ax_spec_small.clear()
            
            # Reset de variables para evitar errores
            self.vline_map = None
            self.hline_map = None
            self.marker_map = None
            self.cut_time_small = None
            self.cut_spec_small = None
            
            if self.cbar:
                try: self.cbar.remove()
                except: pass
                self.cbar = None
    
            # --- 2. Recortes (Slicing) ---
            if not hasattr(self, 'idx_min'): self.idx_min = 0
            if not hasattr(self, 'idx_max'): self.idx_max = len(self.WL) - 1
    
            idx_start = self.idx_min
            idx_end = self.idx_max + 1
            
            wl_plot = self.WL[idx_start:idx_end]
            
            # Usamos source_data en vez de self.data
            data_plot = source_data[idx_start:idx_end, :]
            
            if len(wl_plot) < 2: return
    
            # --- 3. Calcular Límites Globales ---
            g_min = np.nanmin(data_plot)
            g_max = np.nanmax(data_plot)
                
            data_range = g_max - g_min
            if data_range == 0: data_range = 1.0
            y_lim_min = g_min - (0.1 * data_range)
            y_lim_max = g_max + (0.1 * data_range)
    
            # --- 4. Dibujar Mapa Principal ---
            self.pcm = self.ax_map.pcolormesh(
                wl_plot, self.TD, data_plot.T,
                shading="auto", cmap="jet",
            )
            
            self.apply_y_scale()
            self.ax_map.set_xlabel("Wavelength (nm)")
            self.ax_map.set_ylabel("Delay (ps) - SymLog")
            
            # Actualizamos el título dinámicamente según el modo
            self.ax_map.set_title(f"ΔA Map (TAS) {mode_suffix}")
            
            self.ax_map.set_xlim(wl_plot.min(), wl_plot.max())
            
            # Colorbar
            divider = make_axes_locatable(self.ax_map)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            self.cbar = self.figure.colorbar(self.pcm, cax=cax, label="ΔA")
            self.apply_y_scale()
    
            if saved_x is not None and saved_y is not None:
                # np.clip evita que el cursor se quede fuera de la pantalla si recortas demasiado con el slider
                mid_x = np.clip(saved_x, wl_plot.min(), wl_plot.max())
                mid_y = saved_y
            else:
                mid_x = np.median(wl_plot)
                mid_y = np.median(self.TD)
            
            self.vline_map = self.ax_map.axvline(mid_x, color="k", ls="--", lw=1, animated=True)
            self.hline_map = self.ax_map.axhline(mid_y, color="k", ls="--", lw=1, animated=True)
            self.marker_map, = self.ax_map.plot([mid_x], [mid_y], "wx", markersize=8, markeredgewidth=2, animated=True)
        
            # --- 6. Configurar Subplots Pequeños ---
            
            # A) CINÉTICA (Abajo-Izquierda)
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
    
            # B) ESPECTRO (Abajo-Derecha)
            y_cut_spec = data_plot[:, np.abs(self.TD - mid_y).argmin()]
            self.cut_spec_small, = self.ax_spec_small.plot(wl_plot, y_cut_spec, 'r-', animated=True)
            
            self.ax_spec_small.set_xlim(wl_plot.min(), wl_plot.max())
            self.ax_spec_small.set_ylim(y_lim_min, y_lim_max)
            self.ax_spec_small.set_title("Spectrum")
            self.ax_spec_small.set_xlabel("Wavelength (nm)")
    
            # --- 7. Eventos ---
            self.bg_cache = None
            if self.cid_draw is not None: self.canvas.mpl_disconnect(self.cid_draw)
            self.cid_draw = self.canvas.mpl_connect('draw_event', self.on_draw)
            
            if self.cid_click is None:
                self.cid_click = self.canvas.mpl_connect("button_press_event", self.on_click_map)
            if self.cid_move is None:
                self.cid_move = self.canvas.mpl_connect("motion_notify_event", self.on_move_map)
    
            self.canvas.draw()
    def on_draw(self, event):
            """Captura el fondo para blitting cuando se redibuja la figura completa."""
            if event is not None and event.canvas != self.canvas:
                return
            # Copiamos la región del canvas (sin las líneas animadas)
            self.bg_cache = self.canvas.copy_from_bbox(self.figure.bbox)
            
            # Aprovechamos para redibujar las líneas animadas una vez
            self.draw_animated_artists()
            
    def draw_animated_artists(self):
            """Ayuda para dibujar solo los elementos dinámicos."""
            # 1. Verificación de seguridad:
            # Si vline_map no existe o es None, no hacemos nada.
            # Esto evita el crash cuando la ventana se abre antes de cargar datos.
            vline = getattr(self, 'vline_map', None)
            if vline is None:
                return
    
            # 2. Dibujar elementos del Mapa
            # (Como ya comprobamos vline, asumimos que el resto se crearon junto a él)
            try:
                self.ax_map.draw_artist(self.vline_map)
                self.ax_map.draw_artist(self.hline_map)
                self.ax_map.draw_artist(self.marker_map)
                
                # 3. Dibujar elementos de los subplots
                # Verificamos también estos por seguridad
                if getattr(self, 'cut_time_small', None) is not None:
                    self.ax_time_small.draw_artist(self.cut_time_small)
                    self.ax_time_small.draw_artist(self.vline_time_small)
                
                if getattr(self, 'cut_spec_small', None) is not None:
                    self.ax_spec_small.draw_artist(self.cut_spec_small)
    
            except AttributeError:
                # Si algo falla internamente en matplotlib (ej. ventana cerrada), ignoramos
                pass
    def update_small_cuts(self, x, y, WL_sel=None, data_sel=None):
            """Actualización completa (lenta) para clicks o cambios de slider."""
  
            self.on_move_map(type('Event', (object,), {'xdata': x, 'ydata': y, 'inaxes': self.ax_map})())
            self.canvas.draw_idle() # Asegura que todo quede fijo

    # ------------------------------------------------------------------
    # EVENTO DE MOVIMIENTO DE RATÓN
    # ------------------------------------------------------------------
    def on_move_map(self, event):
            """Actualización ultra-rápida usando Blitting."""
            # 1. Validaciones básicas de ejes y datos
            if self.data is None or event.inaxes != self.ax_map:
                return
    
            # 2. --- CORRECCIÓN DEL ERROR ---
            # Verificamos si las líneas existen. Si vline_map es None,
            # significa que el gráfico se está limpiando o no se ha creado aún.
            # Usamos getattr por seguridad extra.
            if getattr(self, 'vline_map', None) is None:
                return
    
            # 3. Obtener coordenadas
            x, y = event.xdata, event.ydata
            if x is None or y is None: return
    
            # 4. Restaurar fondo limpio (borra las líneas anteriores)
            if self.bg_cache is not None:
                self.canvas.restore_region(self.bg_cache)
    
            # 5. Actualizar datos de las líneas (sin redibujar ejes)
        
            self.vline_map.set_xdata([x, x])
            self.hline_map.set_ydata([y, y])
            self.marker_map.set_data([x], [y])
            
            # --- Datos para cortes ---
            idx_wl = np.abs(self.WL - x).argmin()
            idx_td = np.abs(self.TD - y).argmin()
            
            # Validar índices (por si el ratón está fuera del rango de datos válidos)
            if idx_wl >= self.data.shape[0] or idx_td >= self.data.shape[1]:
                return
    
            # Actualizar curva Cinética
            y_time = self.data[idx_wl, :]
            self.cut_time_small.set_data(self.TD, y_time)
            self.vline_time_small.set_xdata([y, y])
            
            # Actualizar curva Espectro
            y_spec = self.data[:, idx_td]
            self.cut_spec_small.set_data(self.WL, y_spec)
    
            # 6. Dibujar los elementos animados
            self.draw_animated_artists()
    
            # 7. Blit
            self.canvas.blit(self.figure.bbox)
            
            # Barra de estado
            val = self.data[idx_wl, idx_td]
            self.label_status.setText(f"Cursor: {x:.1f} nm, {y:.2f} ps | ΔA: {val:.4e}")


if __name__ == "__main__":
    # 1. Crear la aplicación
    app = QApplication(sys.argv)
    
    
    app.setStyle("Fusion") 
    
    # 2. Aplicar hoja de estilos a toda la aplicación
    app.setStyleSheet(STYLESHEET) 

    # 3. Forzar a Windows a reconocer el icono
    import ctypes
    myappid = 'spectroscopy.analyzer.v1' 
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    # 4. Configurar el Icono Global
    icon_path = os.path.join(os.path.dirname(__file__), "icon.ico")
    
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    # 5. Lanzar la ventana principal
    window = MainApp()
    window.show()

    sys.exit(app.exec_())