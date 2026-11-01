# 1. Librerías estándar
import os
import re
import datetime

from spark.data.importer import load_universal_file
from spark.data.dataset import SpectroscopyDataset
from spark.core.models import DASModel, SASModel, OscillationModel, KMatrixModel
from spark.data.dataset import SpectroscopyDataset

# 2. PyQt5 (Interfaz Gráfica)
from PyQt5.QtCore import (
    Qt, QTimer, QPropertyAnimation, QThread, pyqtSignal, QEasingCurve,
    QRectF, QLineF, QPointF
)
from PyQt5.QtGui import (
    QIcon, QPixmap, QPainter, QColor, QKeySequence, QPen, QBrush, 
    QFont, QPolygonF
)
from PyQt5.QtWidgets import (
    QApplication, QWidget, QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QPushButton, QLabel, QMessageBox, QProgressBar, QTableWidget, 
    QTableWidgetItem, QHeaderView, QComboBox, QDoubleSpinBox, QSpinBox, 
    QGroupBox, QTabWidget, QInputDialog, QCheckBox, QLineEdit, QListView,
    QFileDialog, QScrollArea, QShortcut, QGraphicsOpacityEffect, QSplitter,
    QListWidget, QAbstractItemView, QListWidgetItem, QColorDialog,
    QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsTextItem, 
    QGraphicsLineItem, QGraphicsItem,QMenu
)

# 3. Datos y Matemáticas
import numpy as np
import spark.core.fitting.fit as fit
from spark.core.fitting.optimizer import FitWorker
from spark.gui.plotters.publication import PaperPlotterWindow, SASDASPlotterWindow
from spark.gui.plotters.explorers import Surface3DWindow, PlotViewerWindow, TraceExplorerWindow
from spark.gui.model_builder import ModelBuilderDialog,TransitionEdge
from spark.gui.dialogs import CompareSetupDialog, DiffMapDialog, ParameterIdentifiabilityDialog
from spark.gui.utils import ToastNotification
# 4. Matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas, 
    NavigationToolbar2QT as NavigationToolbar
)
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.widgets import Cursor
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
# ---------------------------------------------------------------------
# GUI Styling Constants
# ---------------------------------------------------------------------

BUTTON_STYLE = """
    QPushButton {
        background-color: #5A6268;   /* Gris azulado neutro profesional */
        color: white;                
        border: 1px solid #545B62;  
        border-radius: 4px;          
        padding: 6px 12px;           
        font-weight: bold;
        font-family: "Segoe UI";
        font-size: 9pt;              
    }
    QPushButton:hover {
        background-color: #6C757D;   /* Brillo al pasar el ratón */
        color: white;
        border: 1px solid #5A6268;   
    }
    QPushButton:pressed {
        background-color: #495057;  
        border: 1px solid #495057;
        color: white;
        padding-top: 7px;            
        padding-left: 13px;
    }
    QPushButton:disabled {
        background-color: #C0C4C8;   
        border: 1px solid #B4B9BE;
        color: #F8F9FA;              
    }
"""

DARK_THEME_STYLE = """
    QDialog, QWidget {
        color: #222222;            
        font-family: "Segoe UI", Arial, sans-serif;
        font-size: 9pt;              
    }
    
   
    QGroupBox {
        border: none;
        border-top: 1px solid #D0D0D0; /* Solo una línea separadora sutil arriba */
        margin-top: 18px;             
        padding-top: 15px;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0px 5px 0px 0px;
        font-weight: bold;
        font-size: 10pt;
        color: #3C5488; /* Un tono azul oscuro elegante (estilo Nature) */
    }

   
    QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {
        background-color: #F8F9FA; 
        border: 1px solid #CED4DA;
        border-radius: 4px;
        color: #212529;            
        padding: 4px 8px;                
        min-height: 22px; /* Forzamos a que todos tengan la misma altura */
    }
    QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QLineEdit:focus {
        border: 1px solid #80BDFF; 
        background-color: #FFFFFF;
    }
    QComboBox QAbstractItemView {
        background-color: #FFFFFF;
        color: #212529;
        selection-background-color: #0078D7; 
        selection-color: #FFFFFF;
        border: 1px solid #CED4DA;
    }
    QLabel, QCheckBox { color: #333333; }
    
    
    QProgressBar {
        border: 1px solid #CED4DA;
        border-radius: 4px;
        text-align: center;
        background-color: #E9ECEF; 
        color: #495057;
        max-height: 16px;
        font-size: 8pt;
        font-weight: bold;
    }
    QProgressBar::chunk {
        background-color: #28A745; 
        border-radius: 3px;
    }
    /* --- PESTAÑAS (TABS) PREMIUM --- */
    QTabWidget::pane { 
        border: 1px solid #CED4DA; 
        background-color: #FFFFFF; 
        border-radius: 4px;
    }
    QTabBar::tab { 
        background-color: #E9ECEF; 
        color: #495057;
        padding: 8px 20px; 
        min-width: 100px;  /* Forzamos un ancho mínimo para que no corte el texto */
        border: 1px solid #CED4DA; 
        border-bottom: none; 
        border-top-left-radius: 4px; 
        border-top-right-radius: 4px; 
        margin-right: 4px;
        font-family: "Segoe UI", Arial, sans-serif;
        font-size: 10pt;
        font-weight: bold; /* Al ser bold siempre, Qt calcula bien el tamaño */
    }
    QTabBar::tab:selected { 
        background-color: #FFFFFF;                         
        color: #0078D7; 
        border-bottom: 2px solid #FFFFFF; 
        padding-top: 10px; /* Efecto de solapa activa sin romper la geometría */
    }
    QTabBar::tab:hover:!selected {
        background-color: #DEE2E6;
    }
"""


        
class GlobalFitPanel(QDialog):
    """
    Global Fit Analysis Panel.
    
    Provides a comprehensive UI for loading kinetic data, applying pre-processing steps,
    setting up global fitting models (Parallel, Sequential, Oscillation), running SVD,
    executing the fit pipeline, and exploring the results and residuals.
    """
    def __init__(self, parent=None):
        """Initializes the Global Fit Panel UI, variables, and layouts."""
        super().__init__(parent)
        self.setWindowTitle("SPARK - Global Fit Analysis")
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinMaxButtonsHint)
        
        # --- AUTO-AJUSTE Y CENTRADO INTELIGENTE ---
        screen = QApplication.primaryScreen()
        screen_geom = screen.availableGeometry()
        
        # Calculamos un tamaño objetivo, asegurándonos de no exceder los márgenes de la pantalla
        w_target = min(1200, int(screen_geom.width() * 0.85))
        h_target = min(850, int(screen_geom.height() * 0.85))
        
        self.resize(w_target, h_target)
        self.setStyleSheet(DARK_THEME_STYLE + BUTTON_STYLE) # Apply Dark Theme

        # Centrar la ventana en la pantalla actual de forma nativa
        qr = self.frameGeometry()
        cp = screen_geom.center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

        # --- 1. Data Variables ---
        self.parent_app = parent
        self.datasets_pure_raw = [] # Lista de objetos SpectroscopyDataset puros (unidades nativas)
        self.datasets_raw = []     # Lista de objetos SpectroscopyDataset crudos
        self.datasets_proc = []    # Lista de objetos SpectroscopyDataset procesados
        self.filenames = []        # Nombres de los archivos para la leyenda
        self.base_dir = None
        
        if hasattr(parent, "save_dir") and parent.save_dir:
            self.base_dir = parent.save_dir
        elif hasattr(parent, "file_path") and parent.file_path:
            base_name = os.path.splitext(os.path.basename(parent.file_path))[0]
            self.base_dir = os.path.join(os.path.dirname(parent.file_path), f"{base_name}_Results")
            os.makedirs(self.base_dir, exist_ok=True)
        else:
            self.base_dir = os.getcwd()
    
        # --- 2. Fit Variables ---
        self.numExp = 2
        self.model_type = 'Parallel'
        self.tech = 'TAS'
        self.yscale = 'linear'
        
        # Placeholders for results
        self.fit_result = None
        self.fit_x = None
        self.As = None
        self.Bs = None
        self.errBs = None
        self.fitted_model_type = None
        
        # Rest of the fit variables
        self.fit_resid = None
        self.fit_fitres = None
        self.ci = None
        self.errAs = None
        self.errtaus = None
        self.ini = None
        self.limi = None
        self.lims = None
        self.batch_fit_results = {}
        
        # --- Historial Deshacer/Rehacer ---
        self.undo_stack = []
        self.redo_stack = []

        # Atajos de teclado globales para este panel
        self.shortcut_undo = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.shortcut_undo.activated.connect(self.undo)

        self.shortcut_redo = QShortcut(QKeySequence("Ctrl+Y"), self)
        self.shortcut_redo.activated.connect(self.redo)
        
        # --- 3. MAIN LAYOUT DESIGN ---
        main_layout = QHBoxLayout() 
        
        # --- A. Left Panel (Workflow Tabs) ---
        self.sidebar_tabs = QTabWidget()
        self.sidebar_tabs.setMinimumWidth(320)

        self.tab_data = QWidget()
        self.tab_model = QWidget()
        self.tab_export = QWidget()
        
        self.sidebar_tabs.addTab(self.tab_data, "1. Data")
        self.sidebar_tabs.addTab(self.tab_model, "2. Fit")
        self.sidebar_tabs.addTab(self.tab_export, "3. Export")
        
        self._init_sidebar_ui() 
        
        # --- B. Right Panel (Plots) ---
        self.right_area = QWidget()
        self.right_layout = QVBoxLayout(self.right_area)
        self._init_plots_ui() 
        
        # --- CREAR EL SPLITTER (Barra arrastrable) ---
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.sidebar_tabs)
        splitter.addWidget(self.right_area)
        
        # Le damos un tamaño inicial proporcional (ej: 370px izq, 830px derecha)
        splitter.setSizes([370, 830]) 
        
        # Añadimos el splitter al layout principal
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)
        
        # --- ANIMACIONES PREMIUM (EFECTO APPLE) ---
        # 1. Preparar el efecto de opacidad para el panel izquierdo
        self.sidebar_opacity = QGraphicsOpacityEffect(self.sidebar_tabs)
        self.sidebar_opacity.setOpacity(1.0)
        self.sidebar_tabs.setGraphicsEffect(self.sidebar_opacity)
        
        # 2. Configurar el motor de la animación
        self.sidebar_anim = QPropertyAnimation(self.sidebar_opacity, b"opacity")
        self.sidebar_anim.setDuration(200) # 200 milisegundos (súper rápido y sutil)
        self.sidebar_anim.setEasingCurve(QEasingCurve.OutQuad) # Curva de aceleración suave
        
        # 3. Conectar el clic de la pestaña a la animación
        self.sidebar_tabs.currentChanged.connect(self._animate_sidebar_tab)
        
        # --- IMPORTANT: INITIALIZE PLOTTING VARIABLES ---
        self.pcm_exp = None
        self.cbar_exp = None
        self.pcm_fit = None
        self.cbar_fit = None
        self.pcm_resid = None
        self.cbar_resid = None
        
    def _animate_sidebar_tab(self, index):
        """Dispara un fade-in suave al cambiar de pestaña en el panel izquierdo."""
        self.sidebar_anim.stop()
        
        
        self.sidebar_anim.setStartValue(0.85) 
        
        self.sidebar_anim.setEndValue(1.0)   
        self.sidebar_anim.start()
        
    def show_toast(self, message, duration=3000):
        """Muestra una notificación flotante no bloqueante en la ventana actual."""
        self.toast = ToastNotification(message, parent=self, duration=duration)
        self.toast.show_toast()
        
    def keyPressEvent(self, event):
        """Sobreescribe el comportamiento de la tecla Enter para evitar clics fantasma y mantener el Tab."""
        from PyQt5.QtWidgets import QApplication, QPushButton, QCheckBox, QAbstractSpinBox, QLineEdit
        from PyQt5.QtCore import Qt

        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            
            focus_widget = QApplication.focusWidget()
            
            if isinstance(focus_widget, QPushButton):
                focus_widget.click()
                return
                
            elif isinstance(focus_widget, QCheckBox):
                focus_widget.toggle()
                return
                
            elif isinstance(focus_widget, (QAbstractSpinBox, QLineEdit)):
                # ¡CORREGIDO! Ya no borramos el foco.
                # La caja ya ha guardado el número. Solo le decimos a la ventana
                # que acepte la tecla Enter y no haga nada más.
                event.accept()
                return
                
            return 
            
        super().keyPressEvent(event)
        
    def _init_sidebar_ui(self):
        """Sets up all the widgets of the left panel divided into workflow tabs."""
        layout_data = QVBoxLayout(self.tab_data)
        layout_model = QVBoxLayout(self.tab_model)
        layout_export = QVBoxLayout(self.tab_export)
        # ==========================================
        # TAB 1: DATA & PREPARATION
        # ==========================================
        
        # --- Group 1: Data Source ---
        gb_load = QGroupBox("Data Source")
        v_load = QVBoxLayout()
        
        self.label_status = QLabel("No data loaded")
        self.label_status.setStyleSheet("color: gray; font-style: italic; font-weight: bold;")
        v_load.addWidget(self.label_status)
        
        h_btns = QHBoxLayout()
        self.btn_load = QPushButton("Load file")
        self.btn_load.setAutoDefault(False)
        self.btn_load.clicked.connect(self.load_data) 
        h_btns.addWidget(self.btn_load)
    
        self.btn_parent = QPushButton("Use Parent Data")
        self.btn_parent.setAutoDefault(False)
        self.btn_parent.clicked.connect(self.use_parent_data) 
        h_btns.addWidget(self.btn_parent)
        v_load.addLayout(h_btns)
        
        # ---  Fila compartida para comparaciones ---
        h_compare = QHBoxLayout()
        
        self.btn_compare = QPushButton("Compare Kinetics")
        self.btn_compare.setAutoDefault(False)
        self.btn_compare.clicked.connect(self.compare_kinetics)
        h_compare.addWidget(self.btn_compare)
        
        self.btn_diff = QPushButton("Difference 2D Map")
        self.btn_diff.setAutoDefault(False)
        self.btn_diff.clicked.connect(self.show_difference_map)
        h_compare.addWidget(self.btn_diff)
        
        v_load.addLayout(h_compare)
        
        
        v_load.addWidget(QLabel("<b>Visualizing Dataset:</b>"))
        h_dataset = QHBoxLayout()
        self.combo_active_dataset = QComboBox()
        self.combo_active_dataset.setView(QListView())
        self.combo_active_dataset.currentIndexChanged.connect(self._on_active_dataset_changed)
        h_dataset.addWidget(self.combo_active_dataset, stretch=4)
        
        self.btn_remove_dataset = QPushButton("Del")
        self.btn_remove_dataset.setToolTip("Remove current dataset")
        self.btn_remove_dataset.setAutoDefault(False)
        self.btn_remove_dataset.setStyleSheet("background-color: #DC3545; color: white; font-weight: bold; min-width: 30px;")
        self.btn_remove_dataset.clicked.connect(self.remove_active_dataset)
        h_dataset.addWidget(self.btn_remove_dataset, stretch=1)
        v_load.addLayout(h_dataset)
        
        gb_load.setLayout(v_load)
        layout_data.addWidget(gb_load)

        # --- Group 1.5: Data Units (Auto-detected) ---
        self.gb_units = QGroupBox("Data Units")
        form_units = QFormLayout()
        
        self.combo_in_wl = QComboBox()
        self.combo_in_wl.addItems(["nm", "eV", "cm⁻¹"])
        self.combo_in_wl.currentTextChanged.connect(self.apply_canonical_conversion)
        form_units.addRow("Spectral axis:", self.combo_in_wl)

        
        self.combo_in_wl.setContextMenuPolicy(Qt.CustomContextMenu)
        self.combo_in_wl.customContextMenuRequested.connect(self._show_unit_override_menu)    
    
        self.combo_in_time = QComboBox()
        self.combo_in_time.addItems(["ps", "fs", "ns"])
        self.combo_in_time.currentTextChanged.connect(self.apply_canonical_conversion)
        form_units.addRow("Time axis:", self.combo_in_time)
        
        self.gb_units.setLayout(form_units)
        layout_data.addWidget(self.gb_units)


        # --- Group 2: Pre-processing ---
        gb_prep = QGroupBox("Pre-processing")
        gb_prep.setEnabled(False)
        self.gb_prep = gb_prep
        form_prep = QFormLayout()

        self.spin_bl = QSpinBox()
        self.spin_bl.setRange(0, 500)
        self.spin_bl.setValue(0)
        self.spin_bl.valueChanged.connect(lambda val: self._preview_data_processing())
        self.spin_bl.editingFinished.connect(self._preview_data_processing)
        form_prep.addRow("Baseline Pts:", self.spin_bl)

        self.spin_wl_min = QDoubleSpinBox(); self.spin_wl_min.setRange(0, 10000); self.spin_wl_min.setDecimals(2)
        self.spin_wl_max = QDoubleSpinBox(); self.spin_wl_max.setRange(0, 10000); self.spin_wl_max.setDecimals(2)
        self.spin_wl_min.editingFinished.connect(self._preview_data_processing) 
        self.spin_wl_max.editingFinished.connect(self._preview_data_processing)
 
        h_wl = QHBoxLayout()
        h_wl.addWidget(self.spin_wl_min)
        h_wl.addWidget(QLabel("to"))
        h_wl.addWidget(self.spin_wl_max)
        form_prep.addRow("WL Range (nm):", h_wl)

        self.line_exclude = QLineEdit()
        self.line_exclude.setPlaceholderText("e.g. 490-540, 600-615")
        self.line_exclude.editingFinished.connect(self._preview_data_processing)
        form_prep.addRow("Exclude WLs:", self.line_exclude)
    
        self.spin_t_min = QDoubleSpinBox(); self.spin_t_min.setRange(-100, 1e6); self.spin_t_min.setDecimals(3)
        self.spin_t_max = QDoubleSpinBox(); self.spin_t_max.setRange(-100, 1e6); self.spin_t_max.setDecimals(3)
        self.spin_t_min.editingFinished.connect(self._preview_data_processing) 
        self.spin_t_max.editingFinished.connect(self._preview_data_processing)
        
        h_time = QHBoxLayout()
        h_time.addWidget(self.spin_t_min)
        h_time.addWidget(QLabel("to"))
        h_time.addWidget(self.spin_t_max)
        form_prep.addRow("Time Range (ps):", h_time)
        
        self.spin_bin = QSpinBox()
        self.spin_bin.setRange(1, 10000)
        self.spin_bin.setValue(1)
        self.spin_bin.editingFinished.connect(self._preview_data_processing)
        form_prep.addRow("Binning (X axis):", self.spin_bin)
        
        # 1. Crear la etiqueta y el control para el Binning Temporal (Eje Y)
        self.lbl_time_binning = QLabel("Binning (Y axis):")
        self.spin_time_binning = QSpinBox()
        self.spin_time_binning.setMinimum(1)  # 1 significa "sin binning"
        self.spin_time_binning.setMaximum(1000) # Límite alto por si hay muchos tiempos
        self.spin_time_binning.setValue(1)    # Valor por defecto
        
        self.spin_time_binning.editingFinished.connect(self._preview_data_processing)
        form_prep.addRow("Binning (Y axis):", self.spin_time_binning)
        
        
        self.chk_zero_neg = QCheckBox("Set t < 0 to zero (background)")
        self.chk_zero_neg.setChecked(False) 
        self.chk_zero_neg.stateChanged.connect(lambda state: self._preview_data_processing())
        form_prep.addRow(self.chk_zero_neg)
        
        self.chk_norm_data = QCheckBox("Normalize Data Matrix (Max |ΔA| = 1)")
        self.chk_norm_data.setChecked(False)
        self.chk_norm_data.stateChanged.connect(lambda state: self._preview_data_processing())
        form_prep.addRow(self.chk_norm_data)
               
        gb_prep.setLayout(form_prep)
        layout_data.addWidget(gb_prep)

        # --- Group 3: Visualization  ---
        gb_vis = QGroupBox("Visualization")
        gb_vis.setEnabled(False)
        self.gb_vis = gb_vis
        form_vis = QFormLayout()
        self.btn_plot_3d = QPushButton("3D Map")
        self.btn_plot_3d.setAutoDefault(False)
        self.btn_plot_3d.clicked.connect(self.plot_3d_surface)
        form_vis.addRow(self.btn_plot_3d)
        
        self.combo_scale = QComboBox()
        self.combo_scale.addItems(["Linear", "SymLog"])
        self.combo_scale.currentTextChanged.connect(self._on_scale_changed)
        form_vis.addRow("Time Axis Scale:", self.combo_scale)

        # --- Selector de Paleta (Colormap) ---
        self.combo_cmap = QComboBox()
        self.combo_cmap.addItems(["jet", "viridis", "coolwarm", "bwr", "plasma"])
        self.combo_cmap.setCurrentText("jet")
        self.combo_cmap.currentTextChanged.connect(self._on_vis_setting_changed)
        form_vis.addRow("Color Palette:", self.combo_cmap)
        
        # --- Escala Simétrica ---
        self.chk_sym_cmap = QCheckBox("Symmetric Scale (Center at ΔA=0)")
        self.chk_sym_cmap.stateChanged.connect(self._on_vis_setting_changed)
        form_vis.addRow(self.chk_sym_cmap)
        
        gb_vis.setLayout(form_vis)
        layout_data.addWidget(gb_vis)
        
        # Añadimos Visualization al layout de la Pestaña 1
        layout_data.addWidget(gb_vis) 

        layout_data.addStretch() # Empuja todo hacia arriba en la primera pestaña
        
        # ==========================================
        # TAB 2: MODEL & FITTING
        # ==========================================
        
        # --- Group: SVD Analysis ---
        self.gb_svd = QGroupBox("SVD Analysis")
        v_svd = QVBoxLayout()
        
        self.btn_svd = QPushButton("Run SVD Analysis")
        self.btn_svd.setStyleSheet("background-color: #3C5488; color: white; font-weight: bold;")
        self.btn_svd.clicked.connect(self.run_svd)
        v_svd.addWidget(self.btn_svd)
        
        # Panel de diagnóstico integrado (ahora claro y alineado con la app)
        self.svd_info_box = QWidget()
        self.svd_info_box.setStyleSheet("""
            QWidget {
                background-color: #F8F9FA; /* Mismo fondo claro que los demás controles */
                border: 1px solid #CED4DA; /* Borde estándar gris sutil */
                border-radius: 4px;
            }
        """)
        info_layout = QVBoxLayout(self.svd_info_box)
        info_layout.setContentsMargins(12, 10, 12, 10)
        
        self.lbl_svd_title = QLabel("<b>Rank Comparison</b>")
        self.lbl_svd_title.setStyleSheet("color: #3C5488; font-size: 10pt; border: none; background: transparent;")
        info_layout.addWidget(self.lbl_svd_title)
        
        self.lbl_svd_rank = QLabel("Run SVD to see rank suggestion...")
        self.lbl_svd_rank.setStyleSheet("background: transparent; border: none; color: #333333;")
        self.lbl_svd_rank.setTextFormat(Qt.RichText)
        self.lbl_svd_rank.setWordWrap(True)
        info_layout.addWidget(self.lbl_svd_rank)
        
        self.svd_info_box.hide()
        v_svd.addWidget(self.svd_info_box)
        
        self.gb_svd.setLayout(v_svd)
        layout_model.addWidget(self.gb_svd)
        
        # --- Group: Model Settings ---
        self.gb_model = QGroupBox("Model Settings")
        form_model = QFormLayout()
        
        self.spin_numExp = QSpinBox()
        self.spin_numExp.setRange(1, 6)
        self.spin_numExp.setValue(2)
        form_model.addRow("Components:", self.spin_numExp)

        self.combo_model = QComboBox()
        self.combo_model.addItems([
                    "Parallel (DAS)", 
                    "Sequential (SAS)", 
                    "Stretched Exponential (KWW)", 
                    "Damped Oscillation", 
                    "Custom GUI Model"
                ])
        form_model.addRow("Model Type:", self.combo_model)
        
        self.chk_indep_t0 = QCheckBox("Fit independent t0 per wavelength")
        self.chk_indep_t0.setToolTip("Adds a nonlinear t0 parameter for each wavelength to the global fit.")
        form_model.addRow(self.chk_indep_t0)
        
        self.btn_build_model = QPushButton("Open Visual Model Builder")
        self.btn_build_model.setEnabled(False)
        self.btn_build_model.clicked.connect(self.open_visual_model_builder)
        form_model.addRow(self.btn_build_model)
        self.combo_model.currentTextChanged.connect(lambda text: self.btn_build_model.setEnabled("Custom GUI Model" in text))
        
        self.combo_tech = QComboBox()
        self.combo_tech.addItems(["FLUPS", "TAS", "TCSPC"])
        form_model.addRow("Technique:", self.combo_tech)
                
        self.chk_artifact = QCheckBox("Model Coherent Artifact")
        form_model.addRow(self.chk_artifact)
        
        self.combo_artifact_mode = QComboBox()
        self.combo_artifact_mode.addItems(["Raman + XPM (both)", "Raman only (φ₀)", "XPM only (φ₁ + φ₂)"])
        self.combo_artifact_mode.setEnabled(False) 
        self.combo_artifact_mode.setToolTip("Raman only: fits the instantaneous Raman/2PA signal (IRF shape).\n"
            "XPM only: fits cross-phase modulation (dispersive shape).\n"
            "Both: general case for broadband probe TAS.")
        form_model.addRow("Artifact type:", self.combo_artifact_mode)
        
        self.chk_artifact.stateChanged.connect(lambda state: self.combo_artifact_mode.setEnabled(state == Qt.Checked))
                
        self.chk_nnls = QCheckBox("Force Positive Spectra (NNLS)")
        form_model.addRow(self.chk_nnls)
        
        self.btn_edit_guess = QPushButton("Edit Initial Guesses")
        self.btn_edit_guess.clicked.connect(self._open_guess_editor_and_update)
        form_model.addRow(self.btn_edit_guess)
        self.gb_model.setLayout(form_model)
        layout_model.addWidget(self.gb_model)

        # --- Group: Workspace ---
        self.gb_work = QGroupBox("Workspace")
        h_work = QHBoxLayout()
        self.btn_save_proj = QPushButton("Save project")
        self.btn_save_proj.setStyleSheet("background-color: #3C5488; color: white;")
        self.btn_save_proj.clicked.connect(self.save_project)
        self.btn_load_proj = QPushButton("Load project")
        self.btn_load_proj.setStyleSheet("background-color: #E67E22; color: white;")
        self.btn_load_proj.clicked.connect(self.load_project)
        h_work.addWidget(self.btn_save_proj)
        h_work.addWidget(self.btn_load_proj)
        self.gb_work.setLayout(h_work)
        layout_model.addWidget(self.gb_work)

        # --- Acción Principal del Fit ---
        h_run = QHBoxLayout()
        self.btn_run = QPushButton("RUN FIT")
        self.btn_run.setStyleSheet("background-color: #10B981; border: 1px solid #059669; color: white; font-size: 10pt; font-weight: bold;")
        self.btn_run.setMinimumHeight(40)  
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self.run_fit_pipeline) 
        h_run.addWidget(self.btn_run, stretch=3) 
        
        self.btn_abort = QPushButton("ABORT")
        self.btn_abort.setStyleSheet("background-color: #EF4444; border: 1px solid #DC2626; color: white; font-weight: bold; font-size: 10pt;")
        self.btn_abort.setMinimumHeight(40)
        self.btn_abort.setEnabled(False)
        self.btn_abort.clicked.connect(self.abort_fit)
        h_run.addWidget(self.btn_abort, stretch=1) 
        layout_model.addLayout(h_run)
        
        self.btn_batch = QPushButton("RUN BATCH FIT (All Files)")
        self.btn_batch.setMinimumHeight(40)
        self.btn_batch.setEnabled(False)
        self.btn_batch.setStyleSheet("background-color: #0078D7; color: white; font-weight: bold;")
        self.btn_batch.clicked.connect(self.run_batch_pipeline)
        layout_model.addWidget(self.btn_batch)
        
        self.btn_show_das = QPushButton("Show Plots / Results")
        self.btn_show_das.setEnabled(False)
        self.btn_show_das.clicked.connect(self.plot_das_and_more) 
        layout_model.addWidget(self.btn_show_das)
        
        layout_model.addStretch() # Empujamos todo hacia arriba para que no haya huecos feos
        
     
        # ==========================================
        # TAB 3: EXPORT & TOOLS (LA NUEVA PESTAÑA)
        # ==========================================
        
        # --- Group 1: Export Spectrum at Delay ---
        gb_export = QGroupBox("Export Spectrum at Delay")
        gb_export.setEnabled(False) # Se activará cuando cargues datos
        self.gb_export = gb_export
        form_export = QFormLayout()

        h_export = QHBoxLayout()
        self.line_export_delay = QLineEdit()
        self.line_export_delay.setPlaceholderText("e.g. 2.5")
        h_export.addWidget(self.line_export_delay)
        h_export.addWidget(QLabel("ps"))
        form_export.addRow("Target Delay:", h_export)

        self.btn_export_spectrum = QPushButton("Export Spectrum (.txt)")
        self.btn_export_spectrum.setStyleSheet("background-color: #3C5488; color: white;")
        self.btn_export_spectrum.clicked.connect(self.export_spectrum_at_delay)
        form_export.addRow(self.btn_export_spectrum)

        gb_export.setLayout(form_export)
        layout_export.addWidget(gb_export)
        
        # --- Group 2: Advanced Plotters & Reports ---
        gb_tools = QGroupBox("Reports & Publication Plots")
        v_tools = QVBoxLayout()
        
        self.btn_export_pdf = QPushButton("PDF Report")
        self.btn_export_pdf.setEnabled(False) # Se activará cuando haya un Fit
        self.btn_export_pdf.setStyleSheet("background-color: #8B0000; color: white; font-weight: bold;") 
        self.btn_export_pdf.clicked.connect(self.export_pdf_report)
        v_tools.addWidget(self.btn_export_pdf)
        
        self.btn_standalone_plotter = QPushButton("Kinetics Plotter (Saved Traces)")
        self.btn_standalone_plotter.clicked.connect(self.open_standalone_plotter)
        self.btn_standalone_plotter.setStyleSheet("background-color: #3C5488; color: white; padding: 6px;")
        v_tools.addWidget(self.btn_standalone_plotter)     
        
        self.btn_sasdas_plotter = QPushButton("Spectra Plotter (Saved Traces)")
        self.btn_sasdas_plotter.clicked.connect(self.open_sasdas_plotter)
        self.btn_sasdas_plotter.setStyleSheet("background-color: #00A087; color: white; padding: 6px;")
        v_tools.addWidget(self.btn_sasdas_plotter)
        
        gb_tools.setLayout(v_tools)
        layout_export.addWidget(gb_tools)
        
        # Empuja todo hacia arriba para que quede bonito
        layout_export.addStretch()
    

    def _show_unit_override_menu(self, pos):
        """Despliega un menú oculto para forzar la unidad original del archivo si difiere de nm/ps."""
        menu = QMenu(self)
        
        title = menu.addAction("Assign original file units:")
        title.setEnabled(False)
        menu.addSeparator()
        
        action_nm = menu.addAction("Original file units: nm")
        action_ev = menu.addAction("Original file units: eV")
        action_cm = menu.addAction("Original file units: cm⁻¹")
        
        # Captura qué ha pinchado el usuario
        action = menu.exec_(self.combo_in_wl.mapToGlobal(pos))
        
        if not hasattr(self, 'datasets_pure_raw') or not self.datasets_pure_raw:
            return
            
        if action == action_nm: unit = "nm"
        elif action == action_ev: unit = "eV"
        elif action == action_cm: unit = "cm⁻¹"
        else: return
            
        # Modifica el "alma" del archivo en memoria
        for ds in self.datasets_pure_raw:
            ds.native_wl_unit = unit
            
        
        # Recalcula todo instantáneamente
        self.apply_canonical_conversion()

    def apply_canonical_conversion(self):
        """Convierte los datos desde su unidad original a la seleccionada en la vista."""
        if not hasattr(self, 'datasets_pure_raw') or not self.datasets_pure_raw:
            return
            
        target_wl_unit = self.combo_in_wl.currentText()
        target_time_unit = self.combo_in_time.currentText()
        
        self.datasets_raw = [] 
        import copy
        import numpy as np
        
        for ds_pure in self.datasets_pure_raw:
            ds = copy.deepcopy(ds_pure) 
            
            # Leemos las unidades reales con getattr por seguridad (por si el dataset no las tiene)
            orig_wl_unit = getattr(ds, 'native_wl_unit', 'nm')
            orig_time_unit = getattr(ds, 'native_time_unit', 'ps')
            
            # --- 1. CONVERSIÓN DE TIEMPO ---
            if orig_time_unit == "fs": time_ps = ds.td / 1000.0
            elif orig_time_unit == "ns": time_ps = ds.td * 1000.0
            else: time_ps = ds.td
                
            if target_time_unit == "fs": ds.td = time_ps * 1000.0
            elif target_time_unit == "ns": ds.td = time_ps / 1000.0
            else: ds.td = time_ps

            # --- 2. CONVERSIÓN DE ESPECTRO ---
            wl_safe = np.where(ds.wl == 0, 1e-10, ds.wl)
            
            # Pivote a nm
            if orig_wl_unit == "eV": wl_nm = 1239.84193 / wl_safe
            elif orig_wl_unit in ["cm⁻¹", "cm-1"]: wl_nm = 1e7 / wl_safe
            else: wl_nm = wl_safe
                
            # De nm al target de la interfaz
            if target_wl_unit == "eV": ds.wl = 1239.84193 / wl_nm
            elif target_wl_unit in ["cm⁻¹", "cm-1"]: ds.wl = 1e7 / wl_nm
            else: ds.wl = wl_nm
                
            # --- 3. REORDENACIÓN MATRICIAL ESTRICTA ---
            if len(ds.wl) > 1 and ds.wl[0] > ds.wl[-1]:
                ds.wl = ds.wl[::-1]
                if ds.data.shape[1] == len(ds.wl): 
                    ds.data = ds.data[:, ::-1] 
                elif ds.data.shape[0] == len(ds.wl): 
                    ds.data = ds.data[::-1, :] 
                    
            self.datasets_raw.append(ds)
            
        # --- 4. ACTUALIZACIÓN VISUAL ---
        idx = self.combo_active_dataset.currentIndex()
        if idx >= 0:
            self._on_active_dataset_changed(idx)
            
            if hasattr(self, 'WL') and hasattr(self, 'TD') and len(self.WL) > 0 and len(self.TD) > 0:
                self.current_wl = self.WL[len(self.WL) // 2]
                self.current_delay = self.TD[len(self.TD) // 2]
                
            self._update_ui_limits_from_data()
            
            if hasattr(self, 'WL') and len(self.WL) > 0:
                self.spin_wl_min.blockSignals(True)
                self.spin_wl_max.blockSignals(True)
                min_wl, max_wl = np.min(self.WL), np.max(self.WL)
                # Margen enorme para soportar números de Rayos X y cm⁻¹ sin que la GUI colapse
                self.spin_wl_min.setRange(min_wl - 50000, max_wl + 50000)
                self.spin_wl_max.setRange(min_wl - 50000, max_wl + 50000)
                self.spin_wl_min.setValue(min_wl)
                self.spin_wl_max.setValue(max_wl)
                self.spin_wl_min.blockSignals(False)
                self.spin_wl_max.blockSignals(False)

            if hasattr(self, 'TD') and len(self.TD) > 0:
                self.spin_t_min.blockSignals(True)
                self.spin_t_max.blockSignals(True)
                min_t, max_t = np.min(self.TD), np.max(self.TD)
                self.spin_t_min.setRange(min_t - 50000, max_t + 1e7)
                self.spin_t_max.setRange(min_t - 50000, max_t + 1e7)
                self.spin_t_min.setValue(min_t)
                self.spin_t_max.setValue(max_t)
                self.spin_t_min.blockSignals(False)
                self.spin_t_max.blockSignals(False)
                
            self._preview_data_processing()
    def export_spectrum_at_delay(self):
        """
        Exporta el espectro (ΔA vs Wavelength) al delay más cercano al introducido 
        por el usuario, como un fichero .txt. Si ya existe un resultado de fit, 
        incluye también el Fit y el Residual a ese mismo delay para comparación 
        directa en el mismo archivo.
        """
        if self.data_c is None:
            QMessageBox.warning(self, "No data", "Carga y aplica 'Preview' antes de exportar un espectro.")
            return
    
        try:
            target_delay = float(self.line_export_delay.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid input", "Introduce un valor numérico válido de delay (ps).")
            return
    
        Xs = getattr(self, '_wl_proc', self.WL)
        Ys = getattr(self, '_td_proc', self.TD)
        if Xs is None or Ys is None or len(Ys) == 0:
            QMessageBox.warning(self, "No data", "No hay ejes de Wavelength/Delay disponibles.")
            return
    
        idx_td = int(np.argmin(np.abs(Ys - target_delay)))
        real_delay = Ys[idx_td]
    
        columns = [Xs, self.data_c[:, idx_td]]
        headers = ["Wavelength(nm)", "Experimental_DeltaA"]
    
        # Si hay un fit calculado y es compatible con el dataset activo, lo añadimos
        if getattr(self, 'fit_fitres', None) is not None and self.fit_fitres.shape[1] == len(Ys):
            columns.append(self.fit_fitres[:, idx_td])
            headers.append("Fit_DeltaA")
        if getattr(self, 'fit_resid', None) is not None and self.fit_resid.shape[1] == len(Ys):
            columns.append(self.fit_resid[:, idx_td])
            headers.append("Residual")
    
        matrix = np.column_stack(columns)
    
        default_name = f"Spectrum_{real_delay:.3g}ps.txt"
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Spectrum at Delay",
            os.path.join(self.base_dir, default_name) if self.base_dir else default_name,
            "Text Files (*.txt)"
        )
        if not save_path:
            return
    
        header_str = (
            f"Requested delay: {target_delay} ps | Closest available: {real_delay:.6f} ps\n"
            + "\t".join(headers)
        )
        np.savetxt(save_path, matrix, fmt='%.6e', delimiter='\t', header=header_str, comments='')
    
        QMessageBox.information(
            self, "Saved",
            f"Espectro al delay más cercano ({real_delay:.4f} ps) guardado en:\n{save_path}"
        )
        
    def abort_fit(self):
        """Detiene el hilo del ajuste en curso de manera segura."""
        self._abort_fit = True
        self.btn_abort.setText("Aborting...")
        self.btn_abort.setEnabled(False)
        if hasattr(self, 'worker') and self.worker is not None and self.worker.isRunning():
            self.worker.is_aborted = True
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(3000, self._force_terminate_worker_if_running)

    def _force_terminate_worker_if_running(self):
        """Fuerza la finalización del hilo del fit si no responde al aborto cooperativo en 3 segundos."""
        if hasattr(self, 'worker') and self.worker is not None and self.worker.isRunning():
            try:
                self.worker.terminate()
                self.worker.wait(500)
            except Exception:
                pass
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("Aborted")
            self._cleanup_fit_ui()   
            
    def _open_identifiability_dialog(self):
        """Abre el diálogo de análisis de identificabilidad (profile likelihood)."""
        dlg = ParameterIdentifiabilityDialog(self, parent=self)
        dlg.exec_()
        
    def _on_vis_setting_changed(self):
        """Redibuja los lienzos cuando el usuario cambia el color o la simetría."""
        self._update_exp_canvas(use_processed=True)
        self._update_fit_canvas()
        self._update_resid_canvas()
        
    def open_standalone_plotter(self):
        """Lanza el módulo de gráficos de publicación de forma 100% independiente."""
        self.standalone_plotter = PaperPlotterWindow(self)
        self.standalone_plotter.show()    
        
    def open_sasdas_plotter(self):
        """Lanza el módulo de maquetación de espectros SAS/DAS de forma autónoma."""
        self.sasdas_plotter = SASDASPlotterWindow(self)
        self.sasdas_plotter.show()
    
    def save_project(self):
        """Empaqueta toda la UI, guesses, resultados y modelo en un estándar HDF5 universal."""
        import h5py
        import json
        
        path, _ = QFileDialog.getSaveFileName(self, "Save SPARK Project", "", "HDF5 Project Files (*.h5 *.hdf5)")
        if not path: return

        # 1. Recopilamos los metadatos de la interfaz y guesses como diccionarios
        ui_data = {
            'numExp': self.spin_numExp.value(),
            'model_idx': self.combo_model.currentIndex(),
            'tech_idx': self.combo_tech.currentIndex(),
            'artifact': self.chk_artifact.isChecked(),
            'nnls': self.chk_nnls.isChecked(),
            'bl_pts': self.spin_bl.value(),
            'wl_min': self.spin_wl_min.value(),
            'wl_max': self.spin_wl_max.value(),
            't_min': self.spin_t_min.value(),
            't_max': self.spin_t_max.value(),
            'exclude': self.line_exclude.text() if hasattr(self, 'line_exclude') else '',
            'binning': self.spin_bin.value() if hasattr(self, 'spin_bin') else 1,
            'zero_neg': self.chk_zero_neg.isChecked() if hasattr(self, 'chk_zero_neg') else False,
            'norm_data': self.chk_norm_data.isChecked() if hasattr(self, 'chk_norm_data') else False,
            'scale_idx': self.combo_scale.currentIndex() if hasattr(self, 'combo_scale') else 0,
            'cmap_idx': self.combo_cmap.currentIndex() if hasattr(self, 'combo_cmap') else 0,
            'sym_cmap': self.chk_sym_cmap.isChecked() if hasattr(self, 'chk_sym_cmap') else False,
            'fitted_model_type': getattr(self, 'fitted_model_type', self.model_type)
        }

        guesses_data = {
            'ini': getattr(self, 'ini', np.array([])).tolist(),
            'limi': getattr(self, 'limi', np.array([])).tolist(),
            'lims': getattr(self, 'lims', np.array([])).tolist(),
            'is_fixed': getattr(self, 'is_fixed', np.array([])).tolist()
        }

        visual_model_data = None
        if hasattr(self, 'model_builder_dlg') and self.model_builder_dlg is not None:
            if hasattr(self.model_builder_dlg, 'to_dict'):
                visual_model_data = self.model_builder_dlg.to_dict()
            else:
                canvas = self.model_builder_dlg.canvas
                nodes = [{'name': name, 'x': node.pos().x(), 'y': node.pos().y()} for name, node in canvas.nodes.items()]
                edges = []
                for item in canvas.scene.items():
                    if hasattr(item, 'source_node'): # Es un TransitionEdge
                        edges.append({
                            'source': item.source_node.name,
                            'target': item.target_node.name,
                            'type': item.param_type,
                            'label': item.label_name
                        })
                visual_model_data = {'nodes': nodes, 'edges': edges}

        # 2. Escribimos el archivo HDF5
        try:
            with h5py.File(path, 'w') as f:
                # --- METADATOS (JSON Strings en los atributos globales) ---
                f.attrs['ui'] = json.dumps(ui_data)
                f.attrs['guesses'] = json.dumps(guesses_data)
                if visual_model_data:
                    f.attrs['visual_model'] = json.dumps(visual_model_data)

                # --- DATOS EXPERIMENTALES (Matrices puras comprimidas) ---
                g_data = f.create_group('data')
                if getattr(self, 'data_raw', None) is not None:
                    g_data.create_dataset('data_raw', data=self.data_raw, compression='gzip')
                if getattr(self, 'WL', None) is not None:
                    g_data.create_dataset('WL', data=self.WL)
                if getattr(self, 'TD', None) is not None:
                    g_data.create_dataset('TD', data=self.TD)

                # --- RESULTADOS DEL AJUSTE ---
                if getattr(self, 'fit_fitres', None) is not None:
                    g_fit = f.create_group('fit_data')
                    g_fit.create_dataset('fit_x', data=self.fit_x)
                    g_fit.create_dataset('fit_fitres', data=self.fit_fitres, compression='gzip')
                    g_fit.create_dataset('fit_resid', data=self.fit_resid, compression='gzip')
                    g_fit.create_dataset('extracted_taus', data=self.extracted_taus)
                    g_fit.create_dataset('extracted_errtaus', data=self.extracted_errtaus)
                    
                    # --- GUARDADO DEL NUEVO MODELO ---
                    if self.model_type == "Stretched Exponential" and getattr(self, 'extracted_betas', None) is not None:
                        g_fit.create_dataset('betas', data=self.extracted_betas)
                        g_fit.create_dataset('err_betas', data=self.extracted_errbetas)
                    # ---------------------------------
                    
                    g_fit.create_dataset('As', data=self.As)
                    g_fit.create_dataset('errAs', data=self.errAs)
                    g_fit.create_dataset('ci', data=self.ci)

            self.show_toast("Project saved successfully (HDF5)")
        except Exception as e:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", f"Failed to save the project: {e}")

    def load_project(self):
            """Lee un archivo HDF5 y reconstruye toda la interfaz, datos, gráficos y el motor matemático."""
            import h5py
            import json
            from spark.data.dataset import SpectroscopyDataset
            from spark.core.fitting.global_fitter import GlobalFitter
            
            path, _ = QFileDialog.getOpenFileName(self, "Load SPARK Project", "", "HDF5 Project Files (*.h5 *.hdf5)")
            if not path: return
    
            self.base_dir = os.path.dirname(path)
            
            try:
                with h5py.File(path, 'r') as f:
                    # 1. Restaurar Datos Experimentales
                    if 'data' in f:
                        g_data = f['data']
                        if 'data_raw' in g_data: self.data_raw = g_data['data_raw'][:]
                        if 'WL' in g_data: self.WL = g_data['WL'][:]
                        if 'TD' in g_data: self.TD = g_data['TD'][:]
                        
                        ds = SpectroscopyDataset(self.data_raw, self.WL, self.TD)
                        self.datasets_pure_raw = [ds.copy()]
                        self.datasets_raw = [ds.copy()]
                        self.data_raw_list = [self.data_raw]
                        self.WL_list = [self.WL]
                        self.TD_list = [self.TD]
                        self.filenames = [os.path.basename(path)]
                        
                        self.combo_active_dataset.blockSignals(True)
                        self.combo_active_dataset.clear()
                        self.combo_active_dataset.addItems(self.filenames)
                        self.combo_active_dataset.setCurrentIndex(0)
                        self.combo_active_dataset.blockSignals(False)
                        
                        self.label_status.setText("Loaded 1 File from Project")
                        self._update_ui_limits_from_data()
    
                    # 2. Restaurar TODA la Interfaz ANTES de procesar
                    if 'ui' in f.attrs:
                        ui = json.loads(f.attrs['ui'])
                        self.spin_numExp.setValue(ui.get('numExp', 2))
                        self.combo_model.setCurrentIndex(ui.get('model_idx', 0))
                        self.combo_tech.setCurrentIndex(ui.get('tech_idx', 0))
                        self.chk_artifact.setChecked(ui.get('artifact', False))
                        self.chk_nnls.setChecked(ui.get('nnls', False))
                        
                        self.spin_bl.setValue(ui.get('bl_pts', 0))
                        self.spin_wl_min.setValue(ui.get('wl_min', 0))
                        self.spin_wl_max.setValue(ui.get('wl_max', 10000))
                        self.spin_t_min.setValue(ui.get('t_min', -100))
                        self.spin_t_max.setValue(ui.get('t_max', 1e6))
                        
                        if hasattr(self, 'line_exclude'): self.line_exclude.setText(ui.get('exclude', ''))
                        if hasattr(self, 'spin_bin'): self.spin_bin.setValue(ui.get('binning', 1))
                        if hasattr(self, 'chk_zero_neg'): self.chk_zero_neg.setChecked(ui.get('zero_neg', False))
                        if hasattr(self, 'chk_norm_data'): self.chk_norm_data.setChecked(ui.get('norm_data', False))
                        
                        if hasattr(self, 'combo_scale'): self.combo_scale.setCurrentIndex(ui.get('scale_idx', 0))
                        if hasattr(self, 'combo_cmap'): self.combo_cmap.setCurrentIndex(ui.get('cmap_idx', 0))
                        if hasattr(self, 'chk_sym_cmap'): self.chk_sym_cmap.setChecked(ui.get('sym_cmap', False))
                        
                        
                        self.numExp = self.spin_numExp.value()
                        model_str = self.combo_model.currentText()
                        if "Sequential" in model_str: self.model_type = "Sequential"
                        elif "Oscillation" in model_str: self.model_type = "Damped Oscillation"
                        elif "Stretched" in model_str: self.model_type = "Stretched Exponential"
                        elif "Custom GUI Model" in model_str: self.model_type = "Custom GUI Model"
                        else: self.model_type = "Parallel"
                        self.tech = self.combo_tech.currentText()
                        self.fitted_model_type = ui.get('fitted_model_type', self.model_type)
    
                    # 3. APLICAR EL PRE-PROCESADO
                    if hasattr(self, 'data_raw') and self.data_raw is not None:
                        self._preview_data_processing()
                        
                    # 4. Restaurar Guesses
                    if 'guesses' in f.attrs:
                        guesses = json.loads(f.attrs['guesses'])
                        if guesses.get('ini'):
                            self.ini = np.array(guesses['ini'])
                            self.limi = np.array(guesses['limi'])
                            self.lims = np.array(guesses['lims'])
                            self.is_fixed = np.array(guesses['is_fixed'], dtype=bool)
    
                    # 5. Restaurar Modelo Visual
                    if 'visual_model' in f.attrs:
                        try:
                            v_model = json.loads(f.attrs['visual_model'])
                            from spark.gui.model_builder import ModelBuilderDialog, TransitionEdge
                            if not hasattr(self, 'model_builder_dlg') or self.model_builder_dlg is None:
                                self.model_builder_dlg = ModelBuilderDialog(self)
                            
                            has_nodes = bool(v_model.get('nodes'))
                            has_script = bool(v_model.get('script', '').strip())
                            
                            if hasattr(self.model_builder_dlg, 'from_dict'):
                                self.model_builder_dlg.from_dict(v_model)
                            else:
                                canvas = self.model_builder_dlg.canvas
                                canvas.scene.clear()
                                canvas.nodes.clear()
                                self.model_builder_dlg.text_odes.clear()
                                for n in v_model.get('nodes', []):
                                    canvas.add_state(n['name'], x=n.get('x', 0), y=n.get('y', 0))
                                for e in v_model.get('edges', []):
                                    if e.get('source') in canvas.nodes and e.get('target') in canvas.nodes:
                                        src = canvas.nodes[e['source']]
                                        tgt = canvas.nodes[e['target']]
                                        edge = TransitionEdge(src, tgt, e.get('type', 'tau'), e.get('label', ''))
                                        canvas.scene.addItem(edge)

                            if has_nodes or has_script:
                                self.current_custom_model = self.model_builder_dlg.get_compiled_model(silent=True)
                            else:
                                self.current_custom_model = None
                        except Exception:
                            self.current_custom_model = None

                    # Salvaguarda: Si el proyecto guardó Custom GUI Model pero no hay modelo válido compilado:
                    if self.model_type == 'Custom GUI Model' and self.current_custom_model is None:
                        self.model_type = 'Parallel'
                        self.combo_model.blockSignals(True)
                        self.combo_model.setCurrentIndex(0)
                        self.combo_model.blockSignals(False)
                    
                    # 6. Restaurar Fit Results y Reconstruir el Motor
                    if 'fit_data' in f:
                        g_fit = f['fit_data']
                        self.fit_x = g_fit['fit_x'][:]
                        self.fit_fitres = g_fit['fit_fitres'][:]
                        self.fit_resid = g_fit['fit_resid'][:]
                        self.extracted_taus = g_fit['extracted_taus'][:]
                        self.extracted_errtaus = g_fit['extracted_errtaus'][:]
                        self.As = g_fit['As'][:]
                        self.errAs = g_fit['errAs'][:]
                        self.ci = g_fit['ci'][:]
                        
                        # --- LECTURA DEL NUEVO MODELO ---
                        if self.model_type == "Stretched Exponential":
                            if 'betas' in g_fit: self.extracted_betas = g_fit['betas'][:]
                            if 'err_betas' in g_fit: self.extracted_errbetas = g_fit['err_betas'][:]
                        # --------------------------------
                        
                        # --- RECONSTRUIR EL MOTOR PARA QUE FUNCIONEN LAS TRAZAS Y EL PROFILE LIKELIHOOD ---
                        
                        # Rescatamos los ejes ya recortados tras el paso 3
                        ds = SpectroscopyDataset(self.data_c, getattr(self, '_wl_proc', self.WL), getattr(self, '_td_proc', self.TD))
                        if hasattr(self, 'spin_time_binning'): ds.apply_time_binning(self.spin_time_binning.value())
                        
                        self.fitter = GlobalFitter(
                            dataset=ds,
                            model_type=self.model_type,
                            num_exp=self.numExp,
                            use_artifact=self.chk_artifact.isChecked(),
                            artifact_mode=self._get_artifact_mode(),
                            use_nnls=self.chk_nnls.isChecked(),
                            is_indep_t0=getattr(self, 'chk_indep_t0', None) and self.chk_indep_t0.isChecked(),
                            custom_model=getattr(self, 'current_custom_model', None)
                        )
                        # Inyectamos los resultados matemáticos puros en el motor
                        self.fitter.fit_x = self.fit_x
                        self.fitter.ci = self.ci
                        self.fitter.fitres = self.fit_fitres
                        self.fitter.resid = self.fit_resid
                        self.fitter.As = self.As
                        self.fitter.extracted_taus = self.extracted_taus
                        self.fitter.extracted_errtaus = self.extracted_errtaus
                        
                        if self.model_type == "Stretched Exponential":
                            self.fitter.extracted_betas = getattr(self, 'extracted_betas', None)
                            self.fitter.extracted_errbetas = getattr(self, 'extracted_errbetas', None)
                        
                        C_reconst = self.fitter.build_C_matrix(self.fit_x)
                        _, S_T_full = fit.eval_varpro_model(C_reconst, self.data_c.T, enforce_nonneg=self.chk_nnls.isChecked(), numExp=self.numExp)
                        
                        self.fitter.S_T_full = S_T_full
                        self.S_T_full = S_T_full
                        
                        if "Oscillation" in self.model_type:
                            self.Bs = S_T_full[self.numExp, :]
                            self.fitter.Bs = self.Bs
                            
                        self._update_fit_canvas()
                        self._update_resid_canvas()
                        
                        self.fitted_model_type = self.model_type
                        self.btn_show_das.setEnabled(True)
                        self.btn_export_pdf.setEnabled(True)
                        
                self.btn_run.setEnabled(True)
                self.btn_batch.setEnabled(True)
                self.show_toast("Project loaded successfully (HDF5)")
                
            except Exception as e:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.critical(self, "Error", f"Failed to load the HDF5 project.\n\nDetail: {e}")
        
    def run_svd(self):
        """Executes Singular Value Decomposition (SVD) and estimates the optimal rank."""
        if self.data_c is None:
            QMessageBox.warning(self, "Error", "Load the data before trying to run SVD analysis.")
            return
    
        try:
            U, s, Vh = np.linalg.svd(self.data_c, full_matrices=False)
            
            self.svd_U = U    
            self.svd_s = s    
            self.svd_V = Vh.T 
            
            # --- ALGORITMO INTELIGENTE DE RANGO SVD ---
            # Asumimos que la cola inferior de los valores singulares es ruido blanco puro
            if len(s) > 15:
                # Tomamos la mediana del último 30% como el "noise floor"
                noise_tail = s[int(len(s)*0.7):]
                noise_floor = np.median(noise_tail) if len(noise_tail) > 0 else s[-1]
                # Se considera señal real todo lo que sea 4 veces mayor que el ruido base
                self.svd_suggested_rank = np.sum(s > 4.0 * noise_floor)
            else:
                # Fallback para matrices muy pequeñas
                self.svd_suggested_rank = np.sum(s / s[0] > 0.015)
                
            # Limitamos el rango entre 1 y un máximo sensato (ej. 6) para cinéticas estándar
            self.svd_suggested_rank = max(1, min(int(self.svd_suggested_rank), 6))
    
            self._plot_svd_results()
            self.tabs.setCurrentWidget(self.tab_svd) 
            
        except Exception as e:
            print(f"SVD Error: {e}")
            
    def _create_svd_canvas(self, tab_widget):
        """
        Creates and embeds the matplotlib canvas for the SVD tab.
        """
        fig = plt.Figure(figsize=(7, 9))
        
        ax1 = fig.add_subplot(311) 
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
        canvas = FigureCanvas(fig)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(canvas)
        tab_widget.setLayout(layout)
        
        return canvas, (ax1, ax2, ax3)

    def _plot_svd_results(self):
        """Plots the SVD results with tight axes and updates the dynamic summary panel."""
        ax1, ax2, ax3 = self.ax_svd
        ax1.clear()
        ax2.clear()
        ax3.clear()
    
        n_comp = min(len(self.svd_s), 10) 
        n_mostrar = self.spin_numExp.value() 
        wl = getattr(self, '_wl_proc', self.WL)
        td = getattr(self, '_td_proc', self.TD)
        
        # --- Plot 1: Scree Plot ---
        ax1.semilogy(range(1, n_comp + 1), self.svd_s[:n_comp], 'o-', color='#DC3545', lw=1.5, markersize=5)
        ax1.axvline(self.svd_suggested_rank, color='#6C757D', linestyle='--', alpha=0.8, label=f'Suggested Rank ({self.svd_suggested_rank})')
        ax1.set_title("Singular Values (Scree Plot)", fontsize=11, fontweight='bold')
        ax1.set_ylabel("Eigenvalue (log)")
        ax1.set_xticks(range(1, n_comp + 1))
        ax1.grid(True, which="both", ls=":", alpha=0.5)
        ax1.legend(loc='upper right', frameon=True)
    
        # --- Plot 2: Spectral components ---
        cmap = plt.get_cmap("tab10")
        for i in range(min(n_mostrar, len(self.svd_s))):
            ax2.plot(wl, self.svd_U[:, i], color=cmap(i % 10), label=f"Comp {i+1}", lw=1.5)
        
        ax2.set_title(f"First {n_mostrar} Spectral Components", fontsize=11, fontweight='bold')
        ax2.set_xlabel("Wavelength (nm)") # <-- ¡AQUÍ ESTÁ LA ETIQUETA AÑADIDA!
        ax2.set_ylabel("Amplitude")
        ax2.axhline(0, color='black', lw=1, alpha=0.4)
        ax2.set_xlim(wl.min(), wl.max()) 
        ax2.grid(True, ls=":", alpha=0.5)
        ax2.legend(frameon=True, loc='upper right', fontsize=8)
        
        # --- Plot 3: Temporal components ---
        for i in range(min(n_mostrar, len(self.svd_s))):
            ax3.plot(td, self.svd_V[:, i], color=cmap(i % 10), label=f"Comp {i+1}", lw=1.5)
            
        ax3.set_title(f"First {n_mostrar} Temporal Components", fontsize=11, fontweight='bold')
        ax3.set_xlabel("Time Delay (ps)")
        ax3.set_ylabel("Amplitude")
        ax3.axhline(0, color='black', lw=1, alpha=0.4)
        ax3.set_xlim(td.min(), td.max()) 
        
        if hasattr(self, 'yscale') and self.yscale == 'symlog':
            ax3.set_xscale('symlog', linthresh=1.0)
            
        ax3.grid(True, which="both", ls=":", alpha=0.5)
        ax3.legend(frameon=True, loc='upper right', fontsize=8)
        
        self.canvas_svd.figure.tight_layout()
        self.canvas_svd.draw()
        
        # --- ACTUALIZAR PANEL DE ANÁLISIS SVD EN BARRA LATERAL (TEMA CLARO) ---
        current_model_components = self.spin_numExp.value()
        if self.model_type == "Custom GUI Model" and getattr(self, 'current_custom_model', None) is not None:
            current_model_components = len(self.current_custom_model.states)

        if current_model_components > self.svd_suggested_rank:
            color_bg = "#F8D7DA"      
            color_text = "#842029"    
            color_border = "#F5C2C7"
            msg = "Model has more components than SVD suggests. Risk of overfitting noise."
        elif current_model_components < self.svd_suggested_rank:
            color_bg = "#FFF3CD"      
            color_text = "#664D03"    
            color_border = "#FFECB5"
            msg = "SVD suggests more photoactive species than your current model accounts for."
        else:
            color_bg = "#D1E7DD"      
            color_text = "#0F5132"    
            color_border = "#BADBCC"
            msg = "Model complexity matches suggested experimental rank."

        html = f"""
        <table width="100%" style='margin-bottom: 8px;'>
            <tr>
                <td width="50%">
                    <span style='color: #6C757D; font-size: 9pt; font-weight: bold;'>SVD SUGGESTS:</span><br> 
                    <span style='font-size: 16pt; color: #0078D7; font-weight: bold;'>{self.svd_suggested_rank}</span>
                </td>
                <td width="50%">
                    <span style='color: #6C757D; font-size: 9pt; font-weight: bold;'>YOUR MODEL:</span><br> 
                    <span style='font-size: 16pt; color: #0078D7; font-weight: bold;'>{current_model_components}</span>
                </td>
            </tr>
        </table>
        <div style='color: {color_text}; font-size: 9pt; background-color: {color_bg}; border: 1px solid {color_border}; padding: 6px; border-radius: 4px;'>
            {msg}
        </div>
        """
        self.lbl_svd_rank.setText(html)
        self.svd_info_box.show()
        
    def _on_scale_changed(self, text):
        """
        Updates the scale parameter and replots the data canvases.

        Args:
            text (str): The selected scale ('Linear' or 'SymLog').
        """
        self.yscale = text.lower() # 'linear' or 'symlog'
        
        self._update_exp_canvas(use_processed=True)
        self._update_fit_canvas()
        self._update_resid_canvas()

    def open_visual_model_builder(self):
        """Abre la cuadrícula de diseño y guarda el modelo compilado en memoria."""
        
        # --- GUARDA ANTI-REENTRADA: evita que llamadas duplicadas se apilen ---
        if getattr(self, '_opening_model_builder', False):
            return
        self._opening_model_builder = True
        
        try:
            # 1. TRUCO DE PERSISTENCIA: Solo creamos la ventana si no existe aún en esta sesión
            if not hasattr(self, 'model_builder_dlg') or self.model_builder_dlg is None:
                self.model_builder_dlg = ModelBuilderDialog(self)
                
                # Opcional: Si quieres que el lienzo empiece 100% en blanco en vez de cargar
                # el triplete de ejemplo, descomenta las dos líneas siguientes:
                # self.model_builder_dlg.table.setRowCount(0)
                # self.model_builder_dlg.canvas.scene.clear()
    
            # 2. Mostramos la ventana que tenemos guardada en memoria
            if self.model_builder_dlg.exec_() == QDialog.Accepted:
                modelo_compilado = self.model_builder_dlg.get_compiled_model()
                
                # Si el modelo es None (el usuario lo dejó vacío o falló), detenemos el proceso aquí
                if modelo_compilado is None:
                    return 
                    
                self.current_custom_model = modelo_compilado
                
                # Sincronizar numExp con el número real de estados del modelo personalizado
                num_estados = len(self.current_custom_model.states)
                self.spin_numExp.blockSignals(True)
                self.spin_numExp.setValue(num_estados)
                self.spin_numExp.blockSignals(False)
                self.numExp = num_estados
                
                # Forzar regeneración de los guesses iniciales con las dimensiones del nuevo modelo
                self._generate_defaults(force_reset=True)
                
                # Mensaje de éxito
                num_estados = len(self.current_custom_model.states)
                num_params = len(self.current_custom_model.param_labels)
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.information(
                    self, "Model Compiled", 
                    f"¡Modelo cargado con éxito!\n"
                    f"• Estados Excitados detectados: {num_estados} {self.current_custom_model.states}\n"
                    f"• Parámetros cinéticos globales: {num_params} {self.current_custom_model.param_labels}\n"
                    f"Ya puedes editar sus valores iniciales o hacer clic en RUN FIT."
                )
        finally:
            self._opening_model_builder = False
        
    def _init_plots_ui(self):
            """Builds the right side widgets comprising the tabbed plotting areas."""
            l = self.right_layout
            
            # 1. Crear el contenedor de Pestañas
            self.tabs = QTabWidget()
            
            self.tab_exp = QWidget()
            self.tab_fit = QWidget()
            self.tab_resid = QWidget()
            self.tab_svd = QWidget() 
            
            self.tabs.addTab(self.tab_exp, "Raw Data")
            self.tabs.addTab(self.tab_fit, "Fit Result")
            self.tabs.addTab(self.tab_resid, "Residuals")
            self.tabs.addTab(self.tab_svd, "SVD")
            
            # 2. Crear los Lienzos interactivos (Gráficos)
            self.canvas_exp, self.ax_exp = self._create_canvas_for_tab(self.tab_exp)
            self.canvas_fit, self.ax_fit = self._create_canvas_for_tab(self.tab_fit)
            self.canvas_resid, self.ax_resid = self._create_canvas_for_tab(self.tab_resid)
            self.canvas_svd, self.ax_svd = self._create_svd_canvas(self.tab_svd)
            
            # Añadir las pestañas a la pantalla
            l.addWidget(self.tabs)
            
            # 3. Progress bar
            self.progress_bar = QProgressBar()
            self.progress_bar.setValue(0)
            self.progress_bar.setTextVisible(True)
            l.addWidget(self.progress_bar)
            
            # 4. Barra de estado inferior para el cursor
            self.lbl_cursor = QLabel("Cursor: Out of the 2D map")
            self.lbl_cursor.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.lbl_cursor.setStyleSheet("color: #6C757D; font-size: 9pt; font-family: Consolas, monospace; margin-top: 2px;")
            l.addWidget(self.lbl_cursor)
    def plot_3d_surface(self):
        """Plots the 3D surface representation of the current data matrix."""
        if self.data_c is None:
            QMessageBox.warning(self, "Sin datos", "Aplica 'Preview' antes de ver el 3D.")
            return
    
        # Take the actual data
        xs = getattr(self, '_wl_proc', self.WL)
        ys = getattr(self, '_td_proc', self.TD)
        zs = self.data_c
        scale = getattr(self, 'yscale', 'linear')
    
        # Create the window associated with the 3D plot
        self.pop_3d = Surface3DWindow(xs, ys, zs, scale, parent=self)
        self.pop_3d.show()    
    
    def _generate_defaults(self, force_reset=False):
         """
         Generates the initial parameter guesses for all models (including Stretched
         Exponential and Damped Oscillation) while preserving user-edited kinetic bounds.
         """
         numExp = self.spin_numExp.value()
         tech = self.combo_tech.currentText()
         model_str = self.combo_model.currentText()
    
         is_oscillation = "Oscillation" in model_str
         is_stretched = "Stretched" in model_str
    
         if self.data_c is not None:
             numWL = self.data_c.shape[0]
         elif self.WL is not None:
             numWL = len(self.WL)
         else:
             numWL = 1
    
         old_ini = self.ini.copy() if getattr(self, 'ini', None) is not None else None
         old_limi = self.limi.copy() if getattr(self, 'limi', None) is not None else None
         old_lims = self.lims.copy() if getattr(self, 'lims', None) is not None else None
         old_fixed = self.is_fixed.copy() if getattr(self, 'is_fixed', None) is not None else None
    
         taus_defaults = [0.5, 5.0, 50.0, 500.0, 2000.0, 5000.0]
         w_guess = 0.15 if tech == 'TAS' else (0.3 if tech == 'FLUPS' else 0.1)
    
         if "Custom GUI Model" in model_str: 
             if not hasattr(self, 'current_custom_model') or self.current_custom_model is None:
                 return False
    
             model = self.current_custom_model
             p_ini, p_low, p_upp = model.get_default_guesses_and_bounds()
             num_excitados = len(model.states)
    
             base_kin_params = 2 + len(p_ini)
             num_kin_params = self._get_num_kinetic_params()
             L = num_kin_params + numWL * num_excitados
    
             self.ini = np.zeros(L)
             self.limi = -np.inf * np.ones(L)
             self.lims = np.inf * np.ones(L)
    
             self.ini[0] = w_guess; self.limi[0] = 0.05; self.lims[0] = 2.0
             self.ini[1] = 0.0;     self.limi[1] = -5.0; self.lims[1] = 5.0
             self.ini[2:base_kin_params] = p_ini
             self.limi[2:base_kin_params] = p_low
             self.lims[2:base_kin_params] = p_upp
             self.ini[num_kin_params:] = 0.01
         else:
             if is_oscillation:
                 base_kin_params = 2 + numExp + 3
                 num_species = numExp + 1
             elif is_stretched:
                 base_kin_params = 2 + 2 * numExp
                 num_species = numExp
             else:
                 base_kin_params = 2 + numExp
                 num_species = numExp
    
             num_kin_params = self._get_num_kinetic_params()
             L = num_kin_params + numWL * num_species
             self.ini = np.zeros(L)
             self.limi = -np.inf * np.ones(L)
             self.lims = np.inf * np.ones(L)
    
             # 1. w y t0 global
             self.ini[0] = w_guess; self.limi[0] = 0.05; self.lims[0] = 2.0
             self.ini[1] = 0.0;     self.limi[1] = -5.0; self.lims[1] = 5.0
    
             # 2. Taus (posiciones 2 a 2 + numExp)
             for n in range(numExp):
                 self.ini[2 + n] = taus_defaults[n] if n < len(taus_defaults) else 1000.0 * (n + 1)
                 self.limi[2 + n] = 0.001
                 self.lims[2 + n] = 1e8
    
             # 3. Parámetros específicos del modelo
             if is_stretched:
                 for n in range(numExp):
                     idx_b = 2 + numExp + n
                     self.ini[idx_b] = 0.80   # Valor inicial para beta
                     self.limi[idx_b] = 0.10  # Límite inferior físico
                     self.lims[idx_b] = 2.00  # Permite estirado (<1) y comprimido (>1)
             elif is_oscillation:
                 idx_osc = 2 + numExp
                 self.ini[idx_osc]     = 0.5; self.limi[idx_osc]     = 0.0;    self.lims[idx_osc]     = 100.0  # alpha (damping)
                 self.ini[idx_osc + 1] = 2.0; self.limi[idx_osc + 1] = 0.001;  self.lims[idx_osc + 1] = 500.0  # omega (freq)
                 self.ini[idx_osc + 2] = 0.0; self.limi[idx_osc + 2] = -np.pi; self.lims[idx_osc + 2] = np.pi  # phi (phase)
    
             # 4. t0 independientes por longitud de onda (si está activo)
             if num_kin_params > base_kin_params:
                 time_axis = getattr(self, '_td_proc', getattr(self, 'TD', None))
                 for i in range(base_kin_params, num_kin_params):
                     idx_wl = i - base_kin_params
                     if getattr(self, 'data_c', None) is not None and time_axis is not None:
                         trace = self.data_c[idx_wl, :]
                         t0_guess = time_axis[np.argmax(np.abs(trace))]
                     else:
                         t0_guess = 0.0
                     self.ini[i] = np.clip(t0_guess, -4.99, 4.99)
                     self.limi[i] = -5.0
                     self.lims[i] = 5.0
    
             val_A = 1000.0 if tech == 'TCSPC' else (5.0 if tech == 'FLUPS' else 0.01)
             self.ini[num_kin_params:] = val_A
    
         if not hasattr(self, 'is_fixed') or len(self.is_fixed) != L:
             self.is_fixed = np.zeros(L, dtype=bool)
    
         if not force_reset and old_ini is not None:
             old_kin_len = len(old_ini)
             if old_limi is not None:
                 inf_indices = np.where(old_limi == -np.inf)[0]
                 if len(inf_indices) > 0:
                     old_kin_len = inf_indices[0]
    
             safe_len = min(num_kin_params, old_kin_len)
             if safe_len > 0:
                 self.ini[:safe_len] = old_ini[:safe_len]
                 self.limi[:safe_len] = old_limi[:safe_len]
                 self.lims[:safe_len] = old_lims[:safe_len]
             if old_fixed is not None:
                 safe_fixed_len = min(len(old_fixed), len(self.is_fixed), safe_len)
                 if safe_fixed_len > 0:
                     self.is_fixed[:safe_fixed_len] = old_fixed[:safe_fixed_len]
    
         return True    
    def _create_canvas_for_tab(self, tab_widget):
        """Crea un lienzo interactivo con 3 sub-paneles (Mapa, Espectro, Cinética)."""
        import matplotlib.gridspec as gridspec
        fig = plt.Figure(figsize=(7, 6))
        
        # --- LA MEJORA: Forzar a que el gráfico ocupe el 95% del espacio ---
        fig.subplots_adjust(left=0.07, right=0.95, top=0.95, bottom=0.08)
        
        # GridSpec 2x2: El mapa es grande, los perfiles son estrechos
        gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1.2], height_ratios=[1.2, 4], wspace=0.05, hspace=0.05)
        
        ax_spec = fig.add_subplot(gs[0, 0])
        ax_map = fig.add_subplot(gs[1, 0], sharex=ax_spec)
        ax_kin = fig.add_subplot(gs[1, 1], sharey=ax_map)
        
        # Ocultar etiquetas internas para un aspecto limpio
        ax_spec.tick_params(labelbottom=False)
        ax_kin.tick_params(labelleft=False)
        ax_spec.grid(True, alpha=0.3)
        ax_kin.grid(True, alpha=0.3)
        
        canvas = FigureCanvas(fig)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0) # Quitar márgenes de la pestaña Qt
        layout.addWidget(canvas)
        tab_widget.setLayout(layout)
        
        # Devolvemos un diccionario para controlar todos los ejes
        return canvas, {'map': ax_map, 'spec': ax_spec, 'kin': ax_kin}

    # --- Auxiliary methods to improve the user experience ---
        
    def update_from_parent(self):
        """Updates internal data from the parent application if it exists."""
        p = self.parent_app
        if p is None: 
            return
    
        incoming_data = None
        if getattr(p, "is_TAS_mode", False):
            if hasattr(p, "data_corrected") and p.data_corrected is not None:
                incoming_data = np.array(p.data_corrected, copy=True)
            elif hasattr(p, "data") and p.data is not None:
                incoming_data = np.array(p.data, copy=True)
        else:
            if hasattr(p, "data") and p.data is not None:
                incoming_data = np.array(p.data, copy=True)
    
        if incoming_data is None:
            return
    
        self.data_raw = incoming_data
        self.WL = getattr(p, "WL", None)
        self.TD = getattr(p, "TD", None)
    
        self.apply_baseline_correction()
        
    def apply_baseline_correction(self):
            """Performs a baseline correction based on the spinbox value and replots the data."""
            if self.data_raw is None:
                return
        
            n_pts = self.spin_bl.value()
            temp_data = self.data_raw.copy()
            
            if n_pts > 0:
                if temp_data.shape[1] >= n_pts:
                    # Calculate the baseline (average of the first n columns of time)
                    baseline = np.mean(temp_data[:, :n_pts], axis=1, keepdims=True)
                    temp_data = temp_data - baseline
                else:
                    print("Warning: Not enough points for baseline.")
        
            # --- NUEVO: Respetar la normalización "en vivo" ---
            if hasattr(self, 'chk_norm_data') and self.chk_norm_data.isChecked():
                max_abs_val = np.nanmax(np.abs(temp_data))
                if max_abs_val != 0:
                    temp_data = temp_data / max_abs_val
            # --------------------------------------------------
            
            self.data_c = temp_data                     
            self._update_exp_canvas()
        

    def _update_ui_limits_from_data(self):
        """Updates the internal SpinBox ranges based on the currently loaded data limits."""
        
        # --- Habilitar los paneles ahora que hay datos ---
        if hasattr(self, 'gb_prep'): self.gb_prep.setEnabled(True)
        if hasattr(self, 'gb_vis'): self.gb_vis.setEnabled(True)
        if hasattr(self, 'gb_export'): self.gb_export.setEnabled(True)
        # --------------------------------------------------------
        
        # Update wavelength limits if data exists
        if self.WL is not None and len(self.WL) > 0:
            self.spin_wl_min.setValue(np.min(self.WL))
            self.spin_wl_max.setValue(np.max(self.WL))
        
        # Update time/delay limits if data exists
        if self.TD is not None and len(self.TD) > 0:
            self.spin_t_min.setValue(np.min(self.TD))
            self.spin_t_max.setValue(np.max(self.TD))
        
        # Reset data_c to raw data upon loading and trigger plot
        self.data_c = self.data_raw.copy()
        
        # Immediately plot the raw data
        self._update_exp_canvas(use_processed=False)

    def use_parent_data(self):
        """Loads data from the main application window (if it exists)."""
        if self.parent_app is None: return
        
        # Check if parent has corrected data available
        if hasattr(self.parent_app, "data_corrected") and self.parent_app.data_corrected is not None:
            self.data_raw = np.array(self.parent_app.data_corrected, copy=True)
            self.WL = getattr(self.parent_app, "WL", None)
            self.TD = getattr(self.parent_app, "TD", None)
            
            ds = SpectroscopyDataset(self.data_raw, self.WL, self.TD)
            self.datasets_pure_raw = [ds.copy()]
            self.datasets_raw = [ds.copy()]
            parent_name = getattr(self.parent_app, "file_name", "Parent_Data") or "Parent_Data"
            self.filenames = [parent_name]
            
            self.combo_active_dataset.blockSignals(True)
            self.combo_active_dataset.clear()
            self.combo_active_dataset.addItems(self.filenames)
            self.combo_active_dataset.setCurrentIndex(0)
            self.combo_active_dataset.blockSignals(False)
            
            # Detect experimental technique
            if getattr(self.parent_app, "is_TAS_mode", False):
                self.combo_tech.setCurrentText("TAS")
            else:
                self.combo_tech.setCurrentText("FLUPS")
                
            # Refresh UI components and enable execution   
            self._update_ui_limits_from_data()
            self.btn_run.setEnabled(True)
            self.btn_batch.setEnabled(True)
            if hasattr(self, "gb_export"):
                self.gb_export.setEnabled(True)
            self.label_status.setText(f"Loaded from Parent: {len(self.WL)} WL, {len(self.TD)} TD")
            

    def load_data(self):
        """Carga múltiples archivos usando el lector universal y los convierte en Objetos."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select data files", "", 
            "All Data Files (*.npy *.txt *.csv *.dat);;Numpy Files (*.npy);;Text Files (*.txt *.csv *.dat);;All Files (*.*)"
        )
        
        if not file_paths:
            return
        
        self.base_dir = os.path.dirname(file_paths[0])
            
        for path in file_paths:
            try:
                data_c, WL, TD = load_universal_file(path, parent_window=self)
                if data_c is not None:
                    # Crear el objeto con los datos crudos y guardarlo
                    ds = SpectroscopyDataset(data_c, WL, TD)
                    if not hasattr(self, 'datasets_pure_raw'):
                        self.datasets_pure_raw = []
                    self.datasets_pure_raw.append(ds)
                    self.filenames.append(os.path.basename(path))
                else:
                    print(f"Canceled import for: {os.path.basename(path)}")
            except Exception as e:
                QMessageBox.critical(self, f"Error loading {os.path.basename(path)}", str(e))
                
        # Sincronizamos la UI usando el primer archivo cargado con éxito
        if hasattr(self, 'datasets_pure_raw') and self.datasets_pure_raw:
            
            # 1. Llenamos el combo_active_dataset en silencio para que tenga índice 0 disponible
            self.combo_active_dataset.blockSignals(True)
            self.combo_active_dataset.clear()
            self.combo_active_dataset.addItems(self.filenames)
            self.combo_active_dataset.setCurrentIndex(0)
            self.combo_active_dataset.blockSignals(False)
            
            # 2. Leemos las unidades nativas del primer archivo cargado (por defecto serán nm y ps)
            ds_first = self.datasets_pure_raw[0]
            native_wl = getattr(ds_first, 'native_wl_unit', 'nm')
            native_time = getattr(ds_first, 'native_time_unit', 'ps')
    
            # 3. Actualizamos los selectores de la interfaz para que coincidan (en silencio)
            self.combo_in_wl.blockSignals(True)
            self.combo_in_time.blockSignals(True)
            
            self.combo_in_wl.setCurrentText(native_wl)
            self.combo_in_time.setCurrentText(native_time)
            
            self.combo_in_wl.blockSignals(False)
            self.combo_in_time.blockSignals(False)
            
            # 4. Inicializamos la vista y los datos
            self.apply_canonical_conversion()
            
            # 5. Activamos los botones de control
            self.btn_run.setEnabled(True)
            self.btn_batch.setEnabled(True)
            if hasattr(self, 'gb_export'): 
                self.gb_export.setEnabled(True)
            
            # 6. Mostramos el estado
            self.label_status.setText(f"Loaded {len(self.filenames)} Files")
            
    def _on_active_dataset_changed(self, index):
        """Cambia dinámicamente el dataset visible leyendo de los objetos y restaura ajustes si existen."""
        if index < 0 or index >= len(self.datasets_raw):
            return
            
        ds_raw = self.datasets_raw[index]
        self.data_raw = ds_raw.data
        self.WL = ds_raw.wl
        self.TD = ds_raw.td
        
        use_proc = False
        if len(self.datasets_proc) > index:
            ds_proc = self.datasets_proc[index]
            self.data_c = ds_proc.data
            self._wl_proc = ds_proc.wl
            self._td_proc = ds_proc.td
            use_proc = True
        else:
            self.data_c = ds_raw.data
            self._wl_proc = ds_raw.wl
            self._td_proc = ds_raw.td
            
        self._update_exp_canvas(use_processed=use_proc)
        
        # Comprobar si tenemos resultados de ajuste guardados para este dataset
        filename = self.filenames[index] if index < len(self.filenames) else None
        cached = None
        if hasattr(self, 'batch_fit_results'):
            if filename and filename in self.batch_fit_results:
                cached = self.batch_fit_results[filename]
            elif index in self.batch_fit_results:
                cached = self.batch_fit_results[index]

        if cached is not None:
            self.fit_result = cached.get('fit_result')
            self.fit_x = cached.get('fit_x')
            self.fit_fitres = cached.get('fit_fitres')
            self.fit_resid = cached.get('fit_resid')
            self.As = cached.get('As')
            self.errAs = cached.get('errAs')
            self.Bs = cached.get('Bs')
            self.errBs = cached.get('errBs')
            self.ci = cached.get('ci')
            self.extracted_taus = cached.get('extracted_taus')
            self.extracted_errtaus = cached.get('extracted_errtaus')
            self.extracted_betas = cached.get('extracted_betas')
            self.extracted_errbetas = cached.get('extracted_errbetas')
            self.Artifact_Spectra = cached.get('Artifact_Spectra')
            self.fitted_model_type = cached.get('fitted_model_type', self.model_type)
            self.fitter = cached.get('fitter')
            self.fit_condition_number = cached.get('fit_condition_number')
            self.sloppiest_direction = cached.get('sloppiest_direction')
            self.param_correlation = cached.get('param_correlation')
            self.param_correlation_indices = cached.get('param_correlation_indices')
            self.S_T_full = cached.get('S_T_full')

            self._update_fit_canvas()
            self._update_resid_canvas()
            if hasattr(self, 'btn_show_das'):
                self.btn_show_das.setEnabled(True)
            if hasattr(self, 'btn_export_pdf'):
                self.btn_export_pdf.setEnabled(True)
        else:
            self._clear_fit_plots()
            
    def remove_active_dataset(self):
        """Borra de forma segura el dataset (objeto) activo de la memoria."""
        if not getattr(self, 'datasets_raw', None):
            return

        idx = self.combo_active_dataset.currentIndex()
        if idx < 0 or idx >= len(self.datasets_raw):
            return

        filename = self.filenames[idx] if idx < len(self.filenames) else f"Dataset {idx+1}"
        reply = QMessageBox.question(
            self, 'Remove Dataset',
            f"Are you sure you want to remove the dataset:\n\n'{filename}'?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        # 1. Eliminar de todas las colecciones de datos internas
        if hasattr(self, 'datasets_pure_raw') and len(self.datasets_pure_raw) > idx:
            self.datasets_pure_raw.pop(idx)

        if len(self.datasets_raw) > idx:
            self.datasets_raw.pop(idx)

        if len(self.datasets_proc) > idx:
            self.datasets_proc.pop(idx)

        if hasattr(self, 'data_c_list') and len(self.data_c_list) > idx:
            self.data_c_list.pop(idx)

        if hasattr(self, 'wl_proc_list') and len(self.wl_proc_list) > idx:
            self.wl_proc_list.pop(idx)

        if hasattr(self, 'td_proc_list') and len(self.td_proc_list) > idx:
            self.td_proc_list.pop(idx)

        if hasattr(self, 'data_raw_list') and len(self.data_raw_list) > idx:
            self.data_raw_list.pop(idx)

        if hasattr(self, 'WL_list') and len(self.WL_list) > idx:
            self.WL_list.pop(idx)

        if hasattr(self, 'TD_list') and len(self.TD_list) > idx:
            self.TD_list.pop(idx)

        if len(self.filenames) > idx:
            self.filenames.pop(idx)

        # 2. Actualizar el selector de la interfaz
        self.combo_active_dataset.blockSignals(True)
        self.combo_active_dataset.removeItem(idx)
        self.combo_active_dataset.blockSignals(False)

        if hasattr(self, 'batch_fit_results'):
            if filename in self.batch_fit_results:
                del self.batch_fit_results[filename]
            if idx in self.batch_fit_results:
                del self.batch_fit_results[idx]

        # 3. Limpiar siempre los gráficos y resultados del fit anterior
        self._clear_fit_plots()

        if len(self.datasets_raw) > 0:
            self.label_status.setText(f"Loaded {len(self.filenames)} Files")
            new_idx = max(0, min(idx, len(self.datasets_raw) - 1))
            self.combo_active_dataset.setCurrentIndex(new_idx)
            self._on_active_dataset_changed(new_idx)
            
        else:
            self.label_status.setText("No data loaded")
            self.datasets_pure_raw = []
            self.datasets_raw = []
            self.datasets_proc = []
            self.filenames = []
            if hasattr(self, 'data_c_list'): self.data_c_list = []
            if hasattr(self, 'wl_proc_list'): self.wl_proc_list = []
            if hasattr(self, 'td_proc_list'): self.td_proc_list = []
            if hasattr(self, 'data_raw_list'): self.data_raw_list = []
            if hasattr(self, 'WL_list'): self.WL_list = []
            if hasattr(self, 'TD_list'): self.TD_list = []
            self.data_raw = None
            self.data_c = None
            self.WL = None
            self.TD = None
            if hasattr(self, 'batch_fit_results'): self.batch_fit_results = {}
            if hasattr(self, '_wl_proc'): del self._wl_proc
            if hasattr(self, '_td_proc'): del self._td_proc

            if hasattr(self, 'ax_exp') and isinstance(self.ax_exp, dict):
                self.ax_exp['map'].clear()
                self.ax_exp['spec'].clear()
                self.ax_exp['kin'].clear()
                
            self._clear_colorbar_if_exists(getattr(self, 'cbar_exp', None))
            self.canvas_exp.draw_idle()

            self.lbl_cursor.setText("Cursor: Out of the 2D MAP")

            self.btn_run.setEnabled(False)
            self.btn_batch.setEnabled(False)
            
            if hasattr(self, 'gb_prep'): self.gb_prep.setEnabled(False)
            if hasattr(self, 'gb_vis'): self.gb_vis.setEnabled(False)
            if hasattr(self, 'gb_export'): self.gb_export.setEnabled(False)
            
            

    def _clear_fit_plots(self):
        """Limpia los lienzos de Fit y Residuales si el archivo actual no está ajustado."""
        self.fit_fitres = None
        self.fit_resid = None
        self.As = None
        self.errAs = None
        self.Bs = None
        self.errBs = None
        self.fitted_model_type = None
        self.fit_result = None
        self.fit_x = None
        self.ci = None
        self.extracted_taus = None
        self.extracted_errtaus = None
        self.extracted_betas = None
        self.extracted_errbetas = None
        self.Artifact_Spectra = None
        if hasattr(self, 'fitter'):
            self.fitter = None
        
        # Limpiar los sub-gráficos correctamente sin destruir el diccionario
        if hasattr(self, 'ax_fit') and isinstance(self.ax_fit, dict):
            self.ax_fit['map'].clear()
            self.ax_fit['spec'].clear()
            self.ax_fit['kin'].clear()
            
        if hasattr(self, 'ax_resid') and isinstance(self.ax_resid, dict):
            self.ax_resid['map'].clear()
            self.ax_resid['spec'].clear()
            self.ax_resid['kin'].clear()
            
        self._clear_colorbar_if_exists(getattr(self, 'cbar_fit', None))
        self._clear_colorbar_if_exists(getattr(self, 'cbar_resid', None))
        
        if hasattr(self, 'canvas_fit'):
            self.canvas_fit.draw_idle()
        if hasattr(self, 'canvas_resid'):
            self.canvas_resid.draw_idle()
            
        if hasattr(self, 'btn_show_das'):
            self.btn_show_das.setEnabled(False)
        if hasattr(self, 'btn_export_pdf'):
            self.btn_export_pdf.setEnabled(False)
        
    def _clear_colorbar_if_exists(self, cbar):
        """
        Removes the specified colorbar from the plot if it exists.

        Args:
            cbar: The colorbar object to remove.
        """
        try:
            if cbar is not None:
                cbar.remove()
        except Exception:
            # Silently fail if the colorbar cannot be removed (e.g., already deleted)
            pass
    
    def compare_kinetics(self):
            """Compara cinéticas de múltiples archivos para una lambda específica con personalización total."""
            if not getattr(self, 'data_c_list', None):
                QMessageBox.warning(self, "No data", "Load multiple .npy files first.")
                return
    
            use_proc = hasattr(self, 'wl_proc_list') and len(self.wl_proc_list) == len(self.data_c_list)
            wl_base = self.wl_proc_list[0] if use_proc else self.WL_list[0]
            text_default = f"{wl_base[len(wl_base)//2]:.1f}"
            
            dlg = CompareSetupDialog(wl_base.min(), wl_base.max(), text_default, self.filenames, self)
            if dlg.exec_() != QDialog.Accepted:
                return 
                
            # ===> Extraemos el nuevo array custom_colors <===
            target_wl, normalize, custom_title, custom_labels, ordered_indices, custom_colors = dlg.get_data()
            
            if target_wl is None:
                QMessageBox.critical(self, "Input Error", "Please enter a valid numeric wavelength.")
                return
    
            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(9, 6))
                
                title_str = custom_title
                if normalize and "Norm" not in title_str:
                    title_str += f" (Norm) @ ~{target_wl:.1f} nm"
                elif "nm" not in title_str:
                    title_str += f" @ ~{target_wl:.1f} nm"
                     
                ax.set_title(title_str, fontsize=14)
    
                # ===> Aplicamos los colores personalizados <===
                for ui_idx, original_idx in enumerate(ordered_indices):
                    wl_array = self.wl_proc_list[original_idx] if use_proc else self.WL_list[original_idx]
                    td_array = self.td_proc_list[original_idx] if use_proc else self.TD_list[original_idx]
                    data_matrix = self.data_c_list[original_idx]
                    
                    name = custom_labels[ui_idx]
                    plot_color = custom_colors[ui_idx]
    
                    idx = np.argmin(np.abs(wl_array - target_wl))
                    y_exp = data_matrix[idx, :].copy() 
    
                    if normalize:
                        max_val = np.nanmax(np.abs(y_exp))
                        if max_val != 0:
                            y_exp = y_exp / max_val
    
                    # Le pasamos el parámetro color a matplotlib
                    ax.plot(td_array, y_exp, 'o-', markersize=4, alpha=0.8, color=plot_color, label=name)
    
                ax.set_xscale('symlog', linthresh=1.0) 
                ax.set_xlabel("Time / ps (symlog scale)")
                ax.set_ylabel("Norm. ΔA" if normalize else "ΔA") 
                ax.grid(True, which="both", ls="-", alpha=0.3)
                ax.legend(frameon=True)
    
                plt.tight_layout()
                plt.show()
    
            except Exception as e:
                QMessageBox.critical(self, "Plot Error", f"Failed to plot comparison: {e}")
                
    def show_difference_map(self):
        """Calcula y muestra la resta matricial de dos datasets seleccionados."""
        if not getattr(self, 'data_c_list', None) or len(self.data_c_list) < 2:
            QMessageBox.warning(self, "Not enough data", "Please load at least 2 datasets to calculate a difference.")
            return
            
        dlg = DiffMapDialog(self.filenames, self)
        if dlg.exec_() != QDialog.Accepted:
            return
            
        idx_A, idx_B = dlg.get_indices()
        
        data_A = self.data_c_list[idx_A]
        data_B = self.data_c_list[idx_B]
        
        # Validar que las matrices tienen la misma forma
        if data_A.shape != data_B.shape:
            QMessageBox.critical(self, "Shape Mismatch", 
                                 f"Datasets have different shapes.\n"
                                 f"A: {data_A.shape}\nB: {data_B.shape}\n\n"
                                 f"Make sure both datasets have been processed with the exact same Wavelength/Time cropping.")
            return
            
        # 1. Calcular la diferencia matemática
        diff_data = data_A - data_B
        
        # 2. Extraer ejes
        use_proc = hasattr(self, 'wl_proc_list') and len(self.wl_proc_list) > idx_A
        wl = self.wl_proc_list[idx_A] if use_proc else self.WL_list[idx_A]
        td = self.td_proc_list[idx_A] if use_proc else self.TD_list[idx_A]
        
        # 3. Dibujar
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        
        # Escala de color divergente centrada en 0
        v_abs = np.nanmax(np.abs(diff_data))
        vmin, vmax = -v_abs, v_abs
        
        pcm = ax.pcolormesh(wl, td, diff_data.T, cmap='RdBu_r', shading='auto', vmin=vmin, vmax=vmax)
        ax.set_xlabel("Wavelength (nm)", fontweight='bold')
        ax.set_ylabel("Delay (ps)", fontweight='bold')
        
        title_A = os.path.splitext(self.filenames[idx_A])[0]
        title_B = os.path.splitext(self.filenames[idx_B])[0]
        ax.set_title(f"Difference Map\n[{title_A}]  -  [{title_B}]", fontsize=11, fontweight='bold')
        
        if hasattr(self, 'yscale') and self.yscale == 'symlog':
            ax.set_yscale('symlog', linthresh=2.0)
            
        fig.colorbar(pcm, ax=ax, label='ΔA (Difference)')
        fig.tight_layout()
        
        # Usamos tu visor de gráficos existente para mostrarlo de forma flotante
        self.diff_viewer = PlotViewerWindow(fig, title="Difference 2D Map", parent=self)
        self.diff_viewer.show()
        
    def _preview_data_processing(self):
        """Procesa los datos crudos usando nuestra arquitectura orientada a objetos."""
        if not getattr(self, 'datasets_raw', None):
            if getattr(self, 'data_raw', None) is None: return
            # Fallback por si los datos vienen inyectados del padre (FLUPS/TAS Analyzer)
            ds_raw = SpectroscopyDataset(self.data_raw, self.WL, self.TD)
            raw_list = [ds_raw]
        else:
            raw_list = self.datasets_raw

        self.datasets_proc = []

        for idx in range(len(raw_list)):
            QApplication.processEvents()
            if getattr(self, '_abort_fit', False):
                QMessageBox.warning(self, "Cancelado", "El procesamiento de datos fue cancelado.")
                return 
            
            ds = raw_list[idx].copy() # Clonamos el original para no alterarlo

            # Aplicar correcciones
            ds.apply_baseline_correction(self.spin_bl.value())
            
            w_min, w_max = self.spin_wl_min.value(), self.spin_wl_max.value()
            ds.crop_wavelength(min(w_min, w_max), max(w_min, w_max))
            
            if hasattr(self, 'line_exclude'):
                ds.exclude_wavelength_ranges(self.line_exclude.text().strip())
                
            t_min, t_max = self.spin_t_min.value(), self.spin_t_max.value()
            ds.crop_time(min(t_min, t_max), max(t_min, t_max))
            
            if hasattr(self, 'chk_zero_neg') and self.chk_zero_neg.isChecked():
                ds.zero_negative_times()
                
            ds.apply_binning(self.spin_bin.value())
            
            if hasattr(self, 'spin_time_binning'):
                ds.apply_time_binning(self.spin_time_binning.value())

                
            if hasattr(self, 'chk_norm_data') and self.chk_norm_data.isChecked():
                ds.normalize()

            self.datasets_proc.append(ds)
      
        # Sincronizar variables de retrocompatibilidad gráfica
        if self.datasets_proc:
            current_idx = self.combo_active_dataset.currentIndex()
            if current_idx < 0 or current_idx >= len(self.datasets_proc):
                current_idx = 0
                
            ds_active_raw = raw_list[current_idx]
            ds_active_proc = self.datasets_proc[current_idx]
            
            self.data_raw = ds_active_raw.data
            self.WL = ds_active_raw.wl
            self.TD = ds_active_raw.td
            
            self.data_c = ds_active_proc.data
            self._wl_proc = ds_active_proc.wl
            self._td_proc = ds_active_proc.td
            
            self.WL_list = [ds.wl for ds in raw_list]
            self.TD_list = [ds.td for ds in raw_list]
            self.data_c_list = [ds.data for ds in self.datasets_proc]
            self.wl_proc_list = [ds.wl for ds in self.datasets_proc]
            self.td_proc_list = [ds.td for ds in self.datasets_proc]
            
            self._update_exp_canvas(use_processed=True)
            self.label_status.setText(f"Processed {len(self.datasets_proc)} files")
            

    def _update_exp_canvas(self, use_processed=False):
        if self.data_c is None: return
        
        ax_map = self.ax_exp['map']
        ax_spec = self.ax_exp['spec']
        ax_kin = self.ax_exp['kin']
        
        ax_map.clear(); ax_spec.clear(); ax_kin.clear()
        self._clear_colorbar_if_exists(self.cbar_exp)
        
        if use_processed and hasattr(self, '_wl_proc'):
            Xs = self._wl_proc; Ys = self._td_proc
        else:
            Xs = self.WL; Ys = self.TD
            
        if Xs.shape[0] != self.data_c.shape[0] or Ys.shape[0] != self.data_c.shape[1]:
            Xs = np.arange(self.data_c.shape[0]); Ys = np.arange(self.data_c.shape[1])

        try:
            self.exp_vmin = np.nanmin(self.data_c)
            self.exp_vmax = np.nanmax(self.data_c)
            
           
            if hasattr(self, 'chk_sym_cmap') and self.chk_sym_cmap.isChecked():
                max_abs = max(abs(self.exp_vmin), abs(self.exp_vmax))
                self.exp_vmin = -max_abs
                self.exp_vmax = max_abs
                
            cmap_choice = getattr(self, 'combo_cmap', None).currentText() if hasattr(self, 'combo_cmap') else 'jet'
            
            # --- FIX INTELIGENTE: Dinámico según las dimensiones ---
            current_shading = 'nearest' if (len(Xs) == 1 or len(Ys) == 1) else 'auto'
            
            map_Xs, map_Ys, map_Z = Xs.copy(), Ys.copy(), self.data_c.T.copy()
            if len(Xs) == 1:
                map_Xs = np.array([map_Xs[0] - 5, map_Xs[0] + 5])
                map_Z = np.column_stack((map_Z, map_Z))
            if len(Ys) == 1:
                map_Ys = np.array([map_Ys[0] - 0.5, map_Ys[0] + 0.5])
                map_Z = np.vstack((map_Z, map_Z))
                
            self.pcm_exp = ax_map.pcolormesh(map_Xs, map_Ys, map_Z, shading=current_shading, cmap=cmap_choice, vmin=self.exp_vmin, vmax=self.exp_vmax)
            # ------------------------------------------------------
            
            ax_map.set_xlabel("Wavelength (nm)")
            ax_map.set_ylabel("Delay (ps)")
            
            if hasattr(self, 'yscale') and self.yscale == 'symlog':
                ax_map.set_yscale('symlog', linthresh=2)
            else:
                ax_map.set_yscale('linear')
            
            # --- INYECCIÓN DE LÍNEAS PARA EL EFECTO HOVER ---
            # self.ax_exp['line_spec'], = ax_spec.plot(Xs, self.data_c[:, 0], color='darkred', lw=1.5)
            # self.ax_exp['line_kin'], = ax_kin.plot(self.data_c[0, :], Ys, color='darkblue', lw=1.5)
            self.ax_exp['line_spec'], = ax_spec.plot(Xs, self.data_c[:, 0], color='darkred', lw=1.5, marker='.')
            self.ax_exp['line_kin'], = ax_kin.plot(self.data_c[0, :], Ys, color='darkblue', lw=1.5, marker='.')
            self.ax_exp['vline'] = ax_map.axvline(Xs[0], color='white', ls='--', lw=1, alpha=0.8)
            self.ax_exp['hline'] = ax_map.axhline(Ys[0], color='white', ls='--', lw=1, alpha=0.8)
            
            # Usamos los límites ensanchados (map_Xs) para que no crashee set_xlim
            ax_spec.set_xlim(map_Xs.min(), map_Xs.max()); ax_spec.set_ylim(self.exp_vmin, self.exp_vmax)
            ax_kin.set_xlim(self.exp_vmin, self.exp_vmax)
            
            
            
            divider = make_axes_locatable(ax_kin)
            cax = divider.append_axes("right", size="15%", pad=0.1)
            self.cbar_exp = self.canvas_exp.figure.colorbar(self.pcm_exp, cax=cax, label='$\Delta A$ / -')
            self.canvas_exp._bg_cache = None
            self.canvas_exp.draw_idle()
            
            # Conectar el sensor del ratón
            if not hasattr(self, 'cid_mouse_move'):
                self.cid_mouse_move = self.canvas_exp.mpl_connect('motion_notify_event', self.on_mouse_move)
                
        except Exception as e:
            print(f"Plotting error: {e}")
            
    def on_mouse_move(self, event):
        """Dynamic cross-sections at 60fps usando Blitting."""
        if event.inaxes is None:
            self.lbl_cursor.setText("Cursor: Out of the 2D MAP")
            return

        active_map_dict = None
        data_matrix = None
        canvas = event.canvas

        # Identificar qué lienzo estamos tocando
        if hasattr(self, 'ax_exp') and isinstance(self.ax_exp, dict) and event.inaxes == self.ax_exp.get('map'):
            active_map_dict = self.ax_exp
            data_matrix = self.data_c
        elif hasattr(self, 'ax_fit') and isinstance(self.ax_fit, dict) and event.inaxes == self.ax_fit.get('map'):
            active_map_dict = self.ax_fit
            data_matrix = getattr(self, 'fit_fitres', None)
        elif hasattr(self, 'ax_resid') and isinstance(self.ax_resid, dict) and event.inaxes == self.ax_resid.get('map'):
            active_map_dict = self.ax_resid
            data_matrix = getattr(self, 'fit_resid', None)

        if active_map_dict is None or data_matrix is None or 'line_spec' not in active_map_dict:
            self.lbl_cursor.setText("Cursor: Out of the 2D MAP")
            return

        x = event.xdata
        y = event.ydata
        if x is None or y is None: return

        Xs = getattr(self, '_wl_proc', self.WL)
        Ys = getattr(self, '_td_proc', self.TD)

        try:
            # Buscar el píxel más cercano
            idx_wl = (np.abs(Xs - x)).argmin()
            idx_td = (np.abs(Ys - y)).argmin()
            z_val = data_matrix[idx_wl, idx_td]

            self.lbl_cursor.setText(f"Cursor: λ = {x:.1f} nm  |  Delay = {y:.3f} ps  |  ΔA = {z_val:.3e}")

            # 1. Actualizar los datos matemáticos de las líneas
            active_map_dict['line_spec'].set_data(Xs, data_matrix[:, idx_td])
            active_map_dict['line_kin'].set_data(data_matrix[idx_wl, :], Ys)
            
            # Las líneas axvline/axhline esperan listas de 2 coordenadas
            active_map_dict['vline'].set_xdata([Xs[idx_wl], Xs[idx_wl]])
            active_map_dict['hline'].set_ydata([Ys[idx_td], Ys[idx_td]])

            # --- MAGIA DEL BLITTING (60 FPS) ---
            # Capturamos el tamaño actual de la ventana
            current_size = (canvas.figure.bbox.width, canvas.figure.bbox.height)
            
            # Si no hay caché, o si el usuario ha redimensionado la ventana, tomamos una "foto" limpia
            if getattr(canvas, '_bg_cache', None) is None or getattr(canvas, '_last_size', None) != current_size:
                # Escondemos los cursores temporalmente
                active_map_dict['line_spec'].set_visible(False)
                active_map_dict['line_kin'].set_visible(False)
                active_map_dict['vline'].set_visible(False)
                active_map_dict['hline'].set_visible(False)
                
                # Renderizamos todo lo estático (mapa, ejes, labels) - Lento, pero ocurre solo 1 vez
                canvas.draw()
                canvas._bg_cache = canvas.copy_from_bbox(canvas.figure.bbox)
                canvas._last_size = current_size
                
                # Volvemos a encender los cursores
                active_map_dict['line_spec'].set_visible(True)
                active_map_dict['line_kin'].set_visible(True)
                active_map_dict['vline'].set_visible(True)
                active_map_dict['hline'].set_visible(True)

            # 2. Restauramos la "foto" de fondo limpia al instante
            canvas.restore_region(canvas._bg_cache)

            # 3. Dibujamos solo los 4 vectores móviles encima de la foto (Ultra-rápido)
            active_map_dict['spec'].draw_artist(active_map_dict['line_spec'])
            active_map_dict['kin'].draw_artist(active_map_dict['line_kin'])
            active_map_dict['map'].draw_artist(active_map_dict['vline'])
            active_map_dict['map'].draw_artist(active_map_dict['hline'])

            # 4. Volcamos los píxeles a la pantalla
            canvas.blit(canvas.figure.bbox)

        except Exception:
            pass
        
    def _update_fit_canvas(self):
            if self.fit_fitres is None: return
            
            ax_map = self.ax_fit['map']
            ax_spec = self.ax_fit['spec']
            ax_kin = self.ax_fit['kin']
            
            ax_map.clear(); ax_spec.clear(); ax_kin.clear()
            self._clear_colorbar_if_exists(self.cbar_fit)
            
            Xs = getattr(self, '_wl_proc', self.WL); Ys = getattr(self, '_td_proc', self.TD)
            Z = self.fit_fitres.T 
            data_mat = self.fit_fitres
    
            if Xs is None or Xs.shape[0] != Z.shape[1]: Xs = np.arange(Z.shape[1])
            if Ys is None or Ys.shape[0] != Z.shape[0]: Ys = np.arange(Z.shape[0])
    
            try:
               
                # --- FIX INTELIGENTE: Dinámico según las dimensiones ---
                current_shading = 'nearest' if (len(Xs) == 1 or len(Ys) == 1) else 'auto'
                
                map_Xs, map_Ys, map_Z = Xs.copy(), Ys.copy(), Z.copy()
                if len(Xs) == 1:
                    map_Xs = np.array([map_Xs[0] - 5, map_Xs[0] + 5])
                    map_Z = np.column_stack((map_Z, map_Z))
                if len(Ys) == 1:
                    map_Ys = np.array([map_Ys[0] - 0.5, map_Ys[0] + 0.5])
                    map_Z = np.vstack((map_Z, map_Z))
                
                vmin = getattr(self, 'exp_vmin', np.nanmin(Z))
                vmax = getattr(self, 'exp_vmax', np.nanmax(Z))
    
                self.pcm_fit = ax_map.pcolormesh(map_Xs, map_Ys, map_Z, shading=current_shading, cmap='jet', vmin=vmin, vmax=vmax)
                # -------------------------------------------------------

                ax_map.set_xlabel("Wavelength (nm)"); ax_map.set_ylabel("Delay (ps)")
                
                if hasattr(self, 'yscale') and self.yscale == 'symlog':
                    ax_map.set_yscale('symlog', linthresh=2)
                else:
                    ax_map.set_yscale('linear')
                
                self.ax_fit['line_spec'], = ax_spec.plot(Xs, data_mat[:, 0], color='darkred', lw=1.5, marker='.')
                self.ax_fit['line_kin'], = ax_kin.plot(data_mat[0, :], Ys, color='darkblue', lw=1.5, marker='.')                    
                # self.ax_fit['line_spec'], = ax_spec.plot(Xs, data_mat[:, 0], color='darkred', lw=1.5)
                # self.ax_fit['line_kin'], = ax_kin.plot(data_mat[0, :], Ys, color='darkblue', lw=1.5)
                self.ax_fit['vline'] = ax_map.axvline(Xs[0], color='white', ls='--', lw=1, alpha=0.8)
                self.ax_fit['hline'] = ax_map.axhline(Ys[0], color='white', ls='--', lw=1, alpha=0.8)
    
                ax_spec.set_xlim(map_Xs.min(), map_Xs.max()); ax_spec.set_ylim(vmin, vmax)
                ax_kin.set_xlim(vmin, vmax)
    
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                divider = make_axes_locatable(ax_kin)
                cax = divider.append_axes("right", size="15%", pad=0.1)
                self.cbar_fit = self.canvas_fit.figure.colorbar(self.pcm_fit, cax=cax, label='$\Delta A$ / -')
                self.canvas_fit._bg_cache = None
                self.canvas_fit.draw_idle()
                
                if not hasattr(self, 'cid_mouse_move_fit'):
                    self.cid_mouse_move_fit = self.canvas_fit.mpl_connect('motion_notify_event', self.on_mouse_move)
            except Exception as e:
                print(f"Error painting Fit: {e}")        

    def _update_resid_canvas(self):
        if self.fit_resid is None: return
        
        ax_map = self.ax_resid['map']
        ax_spec = self.ax_resid['spec']
        ax_kin = self.ax_resid['kin']
        
        ax_map.clear(); ax_spec.clear(); ax_kin.clear()
        self._clear_colorbar_if_exists(self.cbar_resid)
        
        Xs = getattr(self, '_wl_proc', self.WL); Ys = getattr(self, '_td_proc', self.TD)
        Z = self.fit_resid.T
        data_mat = self.fit_resid

        if Xs is None or Xs.shape[0] != Z.shape[1]: Xs = np.arange(Z.shape[1])
        if Ys is None or Ys.shape[0] != Z.shape[0]: Ys = np.arange(Z.shape[0])

        try:
            # --- FIX INTELIGENTE: Dinámico según las dimensiones ---
            current_shading = 'nearest' if (len(Xs) == 1 or len(Ys) == 1) else 'auto'
            
            map_Xs, map_Ys, map_Z = Xs.copy(), Ys.copy(), Z.copy()
            if len(Xs) == 1:
                map_Xs = np.array([map_Xs[0] - 5, map_Xs[0] + 5])
                map_Z = np.column_stack((map_Z, map_Z))
            if len(Ys) == 1:
                map_Ys = np.array([map_Ys[0] - 0.5, map_Ys[0] + 0.5])
                map_Z = np.vstack((map_Z, map_Z))

            vals = Z.flatten()
            vmin = np.percentile(vals, 1); vmax = np.percentile(vals, 99)

            self.pcm_resid = ax_map.pcolormesh(map_Xs, map_Ys, map_Z, shading=current_shading, cmap='jet', vmin=vmin, vmax=vmax)
            # -------------------------------------------------------
            
            ax_map.set_xlabel("Wavelength (nm)"); ax_map.set_ylabel("Delay (ps)")
            
            if hasattr(self, 'yscale') and self.yscale == 'symlog':
                ax_map.set_yscale('symlog', linthresh=2)
            else:
                ax_map.set_yscale('linear')
                
            # self.ax_resid['line_spec'], = ax_spec.plot(Xs, data_mat[:, 0], color='purple', lw=1.5)
            # self.ax_resid['line_kin'], = ax_kin.plot(data_mat[0, :], Ys, color='green', lw=1.5)
            self.ax_resid['line_spec'], = ax_spec.plot(Xs, data_mat[:, 0], color='purple', lw=1.5, marker='.')
            self.ax_resid['line_kin'], = ax_kin.plot(data_mat[0, :], Ys, color='green', lw=1.5, marker='.')
            self.ax_resid['vline'] = ax_map.axvline(Xs[0], color='white', ls='--', lw=1, alpha=0.8)
            self.ax_resid['hline'] = ax_map.axhline(Ys[0], color='white', ls='--', lw=1, alpha=0.8)

            ax_spec.set_xlim(map_Xs.min(), map_Xs.max()); ax_spec.set_ylim(vmin, vmax)
            ax_kin.set_xlim(vmin, vmax)

            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax_kin)
            cax = divider.append_axes("right", size="15%", pad=0.1)
            self.cbar_resid = self.canvas_resid.figure.colorbar(self.pcm_resid, cax=cax, label='Residual')
            self.canvas_resid._bg_cache = None
            self.canvas_resid.draw_idle()
            
            if not hasattr(self, 'cid_mouse_move_resid'):
                self.cid_mouse_move_resid = self.canvas_resid.mpl_connect('motion_notify_event', self.on_mouse_move)
        except Exception as e:
            print(f"Error painting Resid: {e}")
# =============================================================================
# FIT PIPELINE
# =============================================================================
    def run_fit_pipeline(self):
        """Main execution pipeline: Preprocess, set model parameters, and run the optimization."""
        # 1. Preparamos los botones y la bandera
        self._abort_fit = False
        # Apagamos las otras pestañas para que el usuario no cambie de pantalla
        self.tab_data.setEnabled(False)
        if hasattr(self, 'tab_export'): self.tab_export.setEnabled(False)
        
        # Apagamos los paneles de configuración y el batch
        if hasattr(self, 'gb_model'): self.gb_model.setEnabled(False)
        if hasattr(self, 'gb_work'): self.gb_work.setEnabled(False)
        self.btn_batch.setEnabled(False)
        self.btn_run.setEnabled(False)
        
        self.btn_abort.setEnabled(True)
        self.btn_abort.setText("ABORT")
        
        try:
            if self.data_raw is None:
                QMessageBox.warning(self, "No data", "Load data first.")
                self._cleanup_fit_ui()
                return
            
            if self.data_c is None or self.data_c.size == 0:
                self._preview_data_processing()
            if self.data_c is None or self.data_c.size == 0:
                self._cleanup_fit_ui()
                return

            self.numExp = self.spin_numExp.value()
            self.tech = self.combo_tech.currentText()
        
            
            model_str = self.combo_model.currentText()
            
            if "Sequential" in model_str:
                self.model_type = "Sequential"
            elif "Stretched" in model_str:
                self.model_type = "Stretched Exponential"
            elif "Oscillation" in model_str:
                self.model_type = "Damped Oscillation"
            elif "Custom GUI Model" in model_str:
                self.model_type = "Custom GUI Model"
            else:
                self.model_type = "Parallel"

            numWL = self.data_c.shape[0] if self.data_c is not None else 0
            
            num_kin_params = self._get_num_kinetic_params()
            
            if self.model_type == "Custom GUI Model":
                num_species = len(self.current_custom_model.states)
            elif self.model_type == "Damped Oscillation":
                num_species = self.numExp + 1
            else:
                num_species = self.numExp
                
            L_needed = num_kin_params + numWL * num_species

            if self.ini is None or len(self.ini) != L_needed:
                self._generate_defaults()
                
            self._temp_fit_TD = getattr(self, '_td_proc', self.TD)
            self._temp_fit_WL = getattr(self, '_wl_proc', self.WL)
            
            self._run_least_squares_with_progress()
            
        except Exception as e:
            QMessageBox.critical(self, "Fit Setup Error", f"Error preparing fit:\n{str(e)}")
            self._cleanup_fit_ui()
            
    def _save_batch_summary(self, batch_outdir, summary_records):
        """Genera archivos de resumen consolidados (Batch_Summary.csv y Batch_Summary.txt) para toda la tanda."""
        if not summary_records:
            return

        import csv
        import datetime

        csv_path = os.path.join(batch_outdir, "Batch_Summary.csv")
        txt_path = os.path.join(batch_outdir, "Batch_Summary.txt")

        # 1. Determinar todos los nombres de parámetros presentes en los registros
        param_names = []
        for r in summary_records:
            for p_name in r.get('params', {}).keys():
                if p_name not in param_names:
                    param_names.append(p_name)

        # Cabeceras CSV
        headers = ["Dataset", "Status", "RMSD", "Condition_Number", "IRF_w (ps)", "IRF_w_err", "t0 (ps)", "t0_err"]
        for p_name in param_names:
            headers.extend([f"{p_name}", f"{p_name}_err"])

        # 2. Escribir CSV
        try:
            with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                for r in summary_records:
                    row = [
                        r['filename'],
                        r['status'],
                        f"{r['rmsd']:.4e}" if (r.get('rmsd') is not None and not np.isnan(r['rmsd'])) else "NaN",
                        f"{r['cond_num']:.2f}" if (r.get('cond_num') is not None and not np.isnan(r['cond_num'])) else "NaN",
                        f"{r['w']:.4f}" if (r.get('w') is not None and not np.isnan(r['w'])) else "NaN",
                        f"{r['w_err']:.4f}" if (r.get('w_err') is not None and not np.isnan(r['w_err'])) else "NaN",
                        f"{r['t0']:.4f}" if (r.get('t0') is not None and not np.isnan(r['t0'])) else "NaN",
                        f"{r['t0_err']:.4f}" if (r.get('t0_err') is not None and not np.isnan(r['t0_err'])) else "NaN",
                    ]
                    for p_name in param_names:
                        val_tuple = r.get('params', {}).get(p_name, (np.nan, np.nan))
                        val, err = val_tuple[0], val_tuple[1]
                        row.append(f"{val:.4e}" if (val is not None and not np.isnan(val)) else "NaN")
                        row.append(f"{err:.4e}" if (err is not None and not np.isnan(err)) else "NaN")
                    writer.writerow(row)
        except Exception as e:
            print(f"Error writing Batch_Summary.csv: {e}")

        # 3. Escribir Resumen TXT formateado
        try:
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            n_succ = sum(1 for r in summary_records if "Success" in r['status'])
            n_fail = len(summary_records) - n_succ

            with open(txt_path, mode="w", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write("SPARK GLOBAL FIT - BATCH EXECUTION SUMMARY\n")
                f.write(f"Generated: {now_str}\n")
                f.write(f"Model: {self.model_type} | Components: {self.numExp} | Tech: {self.tech}\n")
                f.write(f"Total Datasets: {len(summary_records)} | Succeeded: {n_succ} | Failed: {n_fail}\n")
                f.write("=" * 80 + "\n\n")

                col_w_dataset = max(15, max((len(r['filename']) for r in summary_records), default=15))
                fmt_header = f"{'Dataset':<{col_w_dataset}}  {'Status':<12}  {'RMSD':<10}  {'w (ps)':<16}  {'t0 (ps)':<16}"
                for p_name in param_names:
                    fmt_header += f"  {p_name:<20}"
                f.write(fmt_header + "\n")
                f.write("-" * max(80, len(fmt_header)) + "\n")

                for r in summary_records:
                    status_short = "SUCCESS" if "Success" in r['status'] else "FAILED"
                    rmsd_str = f"{r['rmsd']:.2e}" if (r.get('rmsd') is not None and not np.isnan(r['rmsd'])) else "N/A"
                    w_str = f"{r['w']:.3f}±{r['w_err']:.3f}" if (r.get('w') is not None and not np.isnan(r['w'])) else "N/A"
                    t0_str = f"{r['t0']:.3f}±{r['t0_err']:.3f}" if (r.get('t0') is not None and not np.isnan(r['t0'])) else "N/A"

                    line = f"{r['filename']:<{col_w_dataset}}  {status_short:<12}  {rmsd_str:<10}  {w_str:<16}  {t0_str:<16}"
                    for p_name in param_names:
                        val_tuple = r.get('params', {}).get(p_name, (np.nan, np.nan))
                        val, err = val_tuple[0], val_tuple[1]
                        if val is not None and not np.isnan(val):
                            p_str = f"{val:.3g}±{err:.3g}"
                        else:
                            p_str = "N/A"
                        line += f"  {p_str:<20}"
                    f.write(line + "\n")

                f.write("\n" + "=" * 80 + "\n")
        except Exception as e:
            print(f"Error writing Batch_Summary.txt: {e}")

    def run_batch_pipeline(self):
        """Ejecuta el ajuste para todos los archivos cargados de forma secuencial y genera reportes consolidados."""
        from PyQt5.QtCore import QEventLoop

        # 1. Validación de existencia de datos
        has_raw = hasattr(self, 'datasets_raw') and len(self.datasets_raw) > 0
        has_single = getattr(self, 'data_raw', None) is not None
        if not has_raw and not has_single:
            QMessageBox.warning(self, "No data", "Load data files first to run a batch fit.")
            return

        # 2. Asegurar que los datos están pre-procesados y sincronizados
        self._preview_data_processing()

        if not getattr(self, 'data_c_list', None) or len(self.data_c_list) == 0:
            QMessageBox.warning(self, "No data", "Could not process datasets for batch fit.")
            return

        # 3. Configuración inicial del modelo
        self.numExp = self.spin_numExp.value()
        self.tech = self.combo_tech.currentText()
        
        model_str = self.combo_model.currentText()
        if "Sequential" in model_str: 
            self.model_type = "Sequential"
        elif "Stretched" in model_str: 
            self.model_type = "Stretched Exponential"  
        elif "Oscillation" in model_str: 
            self.model_type = "Damped Oscillation"
        elif "Custom GUI Model" in model_str: 
            self.model_type = "Custom GUI Model"
        else: 
            self.model_type = "Parallel"

        # 4. Determinar cantidad de parámetros cinéticos base (no lineales)
        if self.model_type == "Custom GUI Model" and getattr(self, 'current_custom_model', None) is not None:
            base_kin = 2 + len(self.current_custom_model.param_labels)
            num_species = len(self.current_custom_model.states)
        elif self.model_type == "Damped Oscillation":
            base_kin = 2 + self.numExp + 3
            num_species = self.numExp + 1
        elif self.model_type == "Stretched Exponential":
            base_kin = 2 + 2 * self.numExp
            num_species = self.numExp
        else:
            base_kin = 2 + self.numExp
            num_species = self.numExp

        # Asegurar guesses iniciales válidos
        if self.ini is None or len(self.ini) < base_kin:
            self._generate_defaults()

        # Capturar guesses cinéticos, cotas y flags fijas del usuario
        kin_ini = self.ini[:base_kin].copy()
        kin_limi = self.limi[:base_kin].copy() if (hasattr(self, 'limi') and len(self.limi) >= base_kin) else -np.inf * np.ones(base_kin)
        kin_lims = self.lims[:base_kin].copy() if (hasattr(self, 'lims') and len(self.lims) >= base_kin) else np.inf * np.ones(base_kin)
        kin_is_fixed = self.is_fixed[:base_kin].copy() if (hasattr(self, 'is_fixed') and len(self.is_fixed) >= base_kin) else np.zeros(base_kin, dtype=bool)

        # 5. Configurar carpetas y estado de ejecución
        batch_outdir = os.path.join(self.base_dir, "Batch_Results")
        os.makedirs(batch_outdir, exist_ok=True)

        self.is_batch_running = True
        self._abort_fit = False

        # Bloquear botones durante la tanda completa
        self.tab_data.setEnabled(False)
        if hasattr(self, 'tab_export'): self.tab_export.setEnabled(False)
        if hasattr(self, 'gb_model'): self.gb_model.setEnabled(False)
        if hasattr(self, 'gb_work'): self.gb_work.setEnabled(False)
        self.btn_batch.setEnabled(False)
        self.btn_run.setEnabled(False)
        self.btn_abort.setEnabled(True)
        self.btn_abort.setText("ABORT")

        if not hasattr(self, 'batch_fit_results'):
            self.batch_fit_results = {}

        summary_records = []
        is_indep = getattr(self, 'chk_indep_t0', None) and self.chk_indep_t0.isChecked()

        try:
            total_files = len(self.data_c_list)
            for idx in range(total_files):
                QApplication.processEvents()
                if getattr(self, '_abort_fit', False):
                    raise InterruptedError("Batch Fit cancelado manualmente.")

                filename = self.filenames[idx] if idx < len(self.filenames) else f"Dataset_{idx+1}"
                base_name = os.path.splitext(filename)[0]

                self.label_status.setText(f"Batch Fitting: {idx+1}/{total_files} ({base_name})")

                # Sincronizar dataset activo en GUI
                self.combo_active_dataset.blockSignals(True)
                self.combo_active_dataset.setCurrentIndex(idx)
                self.combo_active_dataset.blockSignals(False)

                # Cargar datos específicos de esta iteración de forma completamente coherente
                self.data_c = self.data_c_list[idx]
                self._wl_proc = self.wl_proc_list[idx] if hasattr(self, 'wl_proc_list') else self.WL_list[idx]
                self._td_proc = self.td_proc_list[idx] if hasattr(self, 'td_proc_list') else self.TD_list[idx]
                self.WL = self.WL_list[idx] if hasattr(self, 'WL_list') and idx < len(self.WL_list) else self._wl_proc
                self.TD = self.TD_list[idx] if hasattr(self, 'TD_list') and idx < len(self.TD_list) else self._td_proc
                self._temp_fit_WL = self._wl_proc
                self._temp_fit_TD = self._td_proc
                if hasattr(self, 'datasets_raw') and idx < len(self.datasets_raw):
                    self.data_raw = self.datasets_raw[idx].data

                self._update_exp_canvas(use_processed=True)
                QApplication.processEvents()

                # Construir vector de parámetros de tamaño EXACTO para este dataset específico
                numWL_idx = self.data_c.shape[0]
                num_kin_idx = base_kin + numWL_idx if is_indep else base_kin
                L_idx = num_kin_idx + numWL_idx * num_species

                ini_idx = np.zeros(L_idx)
                limi_idx = -np.inf * np.ones(L_idx)
                lims_idx = np.inf * np.ones(L_idx)
                is_fixed_idx = np.zeros(L_idx, dtype=bool)

                # Cinética
                ini_idx[:base_kin] = kin_ini
                limi_idx[:base_kin] = kin_limi
                lims_idx[:base_kin] = kin_lims
                is_fixed_idx[:base_kin] = kin_is_fixed

                # t0 independiente por lambda si aplica
                if is_indep:
                    ini_idx[base_kin:num_kin_idx] = kin_ini[1]
                    limi_idx[base_kin:num_kin_idx] = -5.0
                    lims_idx[base_kin:num_kin_idx] = 5.0
                    is_fixed_idx[base_kin:num_kin_idx] = False

                # Amplitudes espectrales (fijas para VarPro)
                ini_idx[num_kin_idx:] = 0.01
                is_fixed_idx[num_kin_idx:] = True

                self.ini = ini_idx
                self.limi = limi_idx
                self.lims = lims_idx
                self.is_fixed = is_fixed_idx

                self.current_batch_outdir = os.path.join(batch_outdir, base_name)
                self.last_batch_error = None

                # Lanzar el ajuste
                self._run_least_squares_with_progress()

                loop = QEventLoop()
                self.worker.finished_success.connect(loop.quit)
                self.worker.finished_error.connect(loop.quit)
                loop.exec_()

                # Comprobar aborto de usuario
                if getattr(self, '_abort_fit', False) or (hasattr(self, 'worker') and getattr(self.worker, 'is_aborted', False)):
                    raise InterruptedError("Batch fit was canceled by user.")

                # Recoger registro de resumen para este dataset
                if getattr(self, 'last_batch_error', None) is not None:
                    summary_records.append({
                        'filename': filename,
                        'status': f"Failed: {self.last_batch_error.splitlines()[0]}",
                        'rmsd': np.nan,
                        'cond_num': np.nan,
                        'w': np.nan, 'w_err': np.nan,
                        't0': np.nan, 't0_err': np.nan,
                        'params': {}
                    })
                else:
                    rmsd = np.sqrt(np.mean(self.fit_resid**2)) if getattr(self, 'fit_resid', None) is not None else np.nan
                    cond_num = getattr(self, 'fit_condition_number', np.nan)
                    w_val = self.fit_x[0] if len(self.fit_x) > 0 else np.nan
                    w_err = self.ci[0] if (getattr(self, 'ci', None) is not None and len(self.ci) > 0) else 0.0
                    t0_val = self.fit_x[1] if len(self.fit_x) > 1 else np.nan
                    t0_err = self.ci[1] if (getattr(self, 'ci', None) is not None and len(self.ci) > 1) else 0.0

                    param_dict = {}
                    if self.model_type == "Custom GUI Model" and getattr(self, 'current_custom_model', None) is not None:
                        for p_i, p_label in enumerate(self.current_custom_model.param_labels):
                            p_val = self.extracted_taus[p_i] if (self.extracted_taus is not None and p_i < len(self.extracted_taus)) else np.nan
                            p_err = self.extracted_errtaus[p_i] if (self.extracted_errtaus is not None and p_i < len(self.extracted_errtaus)) else 0.0
                            param_dict[p_label] = (p_val, p_err)
                    else:
                        for n in range(self.numExp):
                            t_val = self.extracted_taus[n] if (self.extracted_taus is not None and n < len(self.extracted_taus)) else np.nan
                            t_err = self.extracted_errtaus[n] if (self.extracted_errtaus is not None and n < len(self.extracted_errtaus)) else 0.0
                            param_dict[f"tau_{n+1}"] = (t_val, t_err)
                            if self.model_type == "Stretched Exponential" and getattr(self, 'extracted_betas', None) is not None:
                                b_val = self.extracted_betas[n] if n < len(self.extracted_betas) else 1.0
                                b_err = self.extracted_errbetas[n] if (self.extracted_errbetas is not None and n < len(self.extracted_errbetas)) else 0.0
                                param_dict[f"beta_{n+1}"] = (b_val, b_err)

                        if self.model_type == "Damped Oscillation":
                            osc_offset = 2 + self.numExp
                            if len(self.fit_x) >= osc_offset + 3:
                                param_dict["alpha"] = (self.fit_x[osc_offset], self.ci[osc_offset] if len(self.ci) > osc_offset else 0.0)
                                param_dict["omega"] = (self.fit_x[osc_offset+1], self.ci[osc_offset+1] if len(self.ci) > osc_offset+1 else 0.0)
                                param_dict["phi"] = (self.fit_x[osc_offset+2], self.ci[osc_offset+2] if len(self.ci) > osc_offset+2 else 0.0)

                    summary_records.append({
                        'filename': filename,
                        'status': "Success",
                        'rmsd': rmsd,
                        'cond_num': cond_num,
                        'w': w_val, 'w_err': w_err,
                        't0': t0_val, 't0_err': t0_err,
                        'params': param_dict
                    })

            # Guardar archivos de resumen consolidados
            self._save_batch_summary(batch_outdir, summary_records)

            n_success = sum(1 for r in summary_records if "Success" in r['status'])
            n_failed = total_files - n_success
            status_msg = f"Batch Completed: {n_success}/{total_files} succeeded"
            if n_failed > 0:
                status_msg += f" ({n_failed} failed)"
            self.label_status.setText(status_msg)

            QMessageBox.information(
                self, "Batch Complete", 
                f"Batch execution finished!\n\n"
                f"• Successfully fitted: {n_success}/{total_files}\n"
                f"• Failed: {n_failed}\n\n"
                f"Summary report saved to:\n{os.path.join(batch_outdir, 'Batch_Summary.csv')}\n\n"
                f"Tip: Select any dataset in the dropdown to view its fit and DAS/SAS."
            )

        except InterruptedError:
            self.label_status.setText("Batch Fit Aborted")
            if summary_records:
                self._save_batch_summary(batch_outdir, summary_records)
            QMessageBox.warning(self, "Batch Aborted", "Batch fit was canceled by user.")
        except Exception as e:
            self.label_status.setText("Batch Error")
            QMessageBox.critical(self, "Batch Error", f"Error during batch fit: {str(e)}")
        finally:
            self.is_batch_running = False
            self._cleanup_fit_ui()
            
    # =============================================================================
    # SISTEMA DE DESHACER / REHACER (UNDO / REDO)
    # =============================================================================
    def save_state_to_history(self):
        """Guarda el estado actual de los parámetros en el historial."""
        if getattr(self, 'ini', None) is None: return
        
        # Guardamos una copia profunda (copia independiente) de los arrays
        state = {
            'ini': self.ini.copy(),
            'limi': self.limi.copy(),
            'lims': self.lims.copy(),
            'is_fixed': self.is_fixed.copy() if hasattr(self, 'is_fixed') else np.zeros(len(self.ini), dtype=bool)
        }
        
        self.undo_stack.append(state)
        # Límite de 20 pasos para no saturar la memoria RAM
        if len(self.undo_stack) > 20:
            self.undo_stack.pop(0)
            
        self.redo_stack.clear() # Al hacer un cambio nuevo, se borra el futuro alternativo

    def _apply_state(self, state):
        """Sobrescribe las variables actuales con un estado guardado."""
        self.ini = state['ini'].copy()
        self.limi = state['limi'].copy()
        self.lims = state['lims'].copy()
        self.is_fixed = state['is_fixed'].copy()

    def undo(self):
        """Ctrl+Z: Restaura el estado anterior."""
        if not self.undo_stack:
            return 
            
        # Guardar el estado actual en "rehacer" antes de viajar al pasado
        current_state = {
            'ini': self.ini.copy(), 'limi': self.limi.copy(),
            'lims': self.lims.copy(), 
            'is_fixed': self.is_fixed.copy() if hasattr(self, 'is_fixed') else np.zeros(len(self.ini), dtype=bool)
        }
        self.redo_stack.append(current_state)
        
        # Recuperar el estado anterior
        previous_state = self.undo_stack.pop()
        self._apply_state(previous_state)
        
        #self.label_status.setText("Undo applied (Ctrl+Z)")
        self.show_toast("↶ Undo applied", duration=1500)
        
    def redo(self):
        """Ctrl+Y: Rehace el estado deshecho."""
        if not self.redo_stack:
            return 
            
        # Guardar el estado actual en "deshacer" antes de viajar al futuro
        current_state = {
            'ini': self.ini.copy(), 'limi': self.limi.copy(),
            'lims': self.lims.copy(), 
            'is_fixed': self.is_fixed.copy() if hasattr(self, 'is_fixed') else np.zeros(len(self.ini), dtype=bool)
        }
        self.undo_stack.append(current_state)
        
        # Recuperar el estado siguiente
        next_state = self.redo_stack.pop()
        self._apply_state(next_state)
        
        #self.label_status.setText("Redo applied (Ctrl+Y)")
        self.show_toast("↷ Redo applied", duration=1500)
        
    def _open_guess_editor_and_update(self):
        """Opens a dialog to manually edit initial guesses (Only Non-linear Kinetic Params)."""
        numExp = self.spin_numExp.value()
        model_str = self.combo_model.currentText()
        is_oscillation = "Oscillation" in model_str
        is_custom = "Custom GUI Model" in model_str
        is_stretched = "Stretched" in model_str
        
        if self.data_c is not None: numWL = self.data_c.shape[0]
        elif self.WL is not None: numWL = len(self.WL)
        else: numWL = 1
            
        num_kin_params = self._get_num_kinetic_params()
        
        if is_custom:
            if not hasattr(self, 'current_custom_model') or self.current_custom_model is None:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Atención", "Aún no has diseñado ningún modelo. Abre el Visual Builder primero.")
                return
            model = self.current_custom_model
            num_states = len(model.states)
            L_needed = num_kin_params + numWL * num_states
        elif is_oscillation:
            L_needed = num_kin_params + numWL * (numExp + 1)
        else:
            L_needed = num_kin_params + numWL * numExp
            
        # Regenerate defaults if the vector size is inconsistent
        if self.ini is None or len(self.ini) != L_needed:
            self._generate_defaults()

        self.save_state_to_history()
        
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTableWidget, QHeaderView, QTableWidgetItem, QHBoxLayout, QPushButton, QMessageBox
        from PyQt5.QtCore import Qt, QPropertyAnimation, QEasingCurve
        
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Edit Initial Guesses - {model_str} (Kinetic Params Only)")
        dlg.resize(750, 450)
        
        # --- ANIMACIÓN DE FADE-IN PARA LA VENTANA ---
        dlg.setWindowOpacity(0.0) 
        self.dlg_anim = QPropertyAnimation(dlg, b"windowOpacity", dlg)
        self.dlg_anim.setDuration(250)
        self.dlg_anim.setStartValue(0.0)
        self.dlg_anim.setEndValue(1.0)
        self.dlg_anim.setEasingCurve(QEasingCurve.OutCubic)
        self.dlg_anim.start()
        
        v = QVBoxLayout()
        
        
        table = QTableWidget(num_kin_params, 5)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setHorizontalHeaderLabels(["Parameter", "Value", "Lower Bound", "Upper Bound", "Fix?"])
        
        if not hasattr(self, 'is_fixed') or len(self.is_fixed) != L_needed:
            self.is_fixed = np.zeros(L_needed, dtype=bool)
            
        
        base_kin = 2 + numExp
        if is_oscillation:
            base_kin = 2 + numExp + 3
        elif is_custom:
            base_kin = 2 + len(model.param_labels)
            
        for i in range(num_kin_params):
            label = f"{i}: "
            
            # 1. t0 independientes por longitud de onda
            if getattr(self, 'chk_indep_t0', None) and self.chk_indep_t0.isChecked() and i >= base_kin:
                idx_wl = i - base_kin
                curr_wl = self._wl_proc[idx_wl] if hasattr(self, '_wl_proc') else idx_wl
                label += f"t0 @ {curr_wl:.1f} nm"
            
            # 2. Custom Model (Tasas de transferencia, decaimientos...)
            elif is_custom:
                if i == 0: label += "w (IRF Width)"
                elif i == 1: label += "t0 (Global Time Zero)"
                elif i < base_kin:
                    idx_param = i - 2
                    p_type = "Branching Ratio" if "gamma" in model.param_labels[idx_param].lower() else "Tau (ps)"
                    label += f"[{p_type}] {model.param_labels[idx_param]}"
                    
            # 3. Oscillation Model
            elif is_oscillation:
                if i == 0: label += "w (IRF Width)"
                elif i == 1: label += "t0 (Global Time Zero)"
                elif i < 2 + numExp: label += f"τ{i-1} (Lifetime)"
                elif i == 2 + numExp: label += "α (Damping/Decay)"
                elif i == 2 + numExp + 1: label += "ω (Ang. Frequency)"
                elif i == 2 + numExp + 2: label += "φ (Phase)"
                
            elif is_stretched:
                if i == 0: label += "w (IRF Width)"
                elif i == 1: label += "t0 (Global Time Zero)"
                elif i < 2 + numExp: label += f"τ{i-1} (Characteristic Lifetime)"
                elif i < 2 + 2 * numExp: label += f"β{i - 1 - numExp} (Stretching Exponent)"
                                
            # 4. Standard Models (Paralelo, Secuencial)
            else:
                if i == 0: label += "w (FWHM (ps))"
                elif i == 1: label += "t0 (Time Zero)"
                elif i < 2 + numExp: label += f"τ{i-1} (Lifetime)"

            # Rellenar la fila en la tabla
            item_lbl = QTableWidgetItem(label)
            item_lbl.setFlags(item_lbl.flags() & ~Qt.ItemIsEditable) 
            table.setItem(i, 0, item_lbl)
            table.setItem(i, 1, QTableWidgetItem(f"{self.ini[i]:.6g}"))
            table.setItem(i, 2, QTableWidgetItem(f"{self.limi[i]:.6g}"))
            table.setItem(i, 3, QTableWidgetItem(f"{self.lims[i]:.6g}"))
            
            chk_item = QTableWidgetItem()
            chk_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            chk_item.setCheckState(Qt.Checked if self.is_fixed[i] else Qt.Unchecked)
            table.setItem(i, 4, chk_item)

        v.addWidget(table)
        
        # Botones de la ventana
        btns = QHBoxLayout()
        btn_reset = QPushButton("Reset to Defaults")
        
        def update_table_in_place():
            self.save_state_to_history()
            self._generate_defaults(force_reset=True)
            for i in range(num_kin_params):
                table.item(i, 1).setText(f"{self.ini[i]:.6g}")
                table.item(i, 2).setText(f"{self.limi[i]:.6g}")
                table.item(i, 3).setText(f"{self.lims[i]:.6g}")
                table.item(i, 4).setCheckState(Qt.Unchecked)
                self.is_fixed[i] = False
                
        btn_reset.clicked.connect(update_table_in_place)
        
        btn_load_fit = QPushButton("Use Last Fit")
        btn_load_fit.setStyleSheet("background-color: #E67E22; color: white;") 
        
        def load_fit_in_place():
            if getattr(self, 'fit_x', None) is None:
                QMessageBox.information(dlg, "No Fit Data", "No previous fit results available to load.")
                return
            
            self.save_state_to_history()
            safe_len = min(num_kin_params, len(self.fit_x))
            self.ini[:safe_len] = self.fit_x[:safe_len].copy()
            
            for i in range(safe_len):
                table.item(i, 1).setText(f"{self.ini[i]:.6g}")
                
        btn_load_fit.clicked.connect(load_fit_in_place)
        
        btn_ok = QPushButton("Save & Close")
        btn_ok.clicked.connect(dlg.accept)
        
        btns.addWidget(btn_reset)
        btns.addWidget(btn_load_fit) 
        btns.addWidget(btn_ok)
        v.addLayout(btns)
        
        dlg.setLayout(v)
        
        # Si el usuario guarda, leemos SOLO las filas cinéticas
        if dlg.exec_() == QDialog.Accepted:
            for i in range(num_kin_params):
                try:
                    self.ini[i] = float(table.item(i, 1).text())
                    self.limi[i] = float(table.item(i, 2).text())
                    self.lims[i] = float(table.item(i, 3).text())
                    self.is_fixed[i] = (table.item(i, 4).checkState() == Qt.Checked)
                except ValueError:
                    pass
                
    def _run_least_squares_with_progress(self):
        """Lanza el hilo de ajuste conectando los datos de la GUI con el motor independiente."""
        if not hasattr(self, 'is_fixed') or len(self.is_fixed) != len(self.ini):
            self.is_fixed = np.zeros(len(self.ini), dtype=bool)
            
        from spark.core.fitting.global_fitter import GlobalFitter
        from spark.data.dataset import SpectroscopyDataset
        
        dataset = SpectroscopyDataset(self.data_c, self._temp_fit_WL, self._temp_fit_TD)
        if hasattr(self, 'spin_time_binning'):
            dataset.apply_time_binning(self.spin_time_binning.value())

        use_art = getattr(self, 'chk_artifact', None) and self.chk_artifact.isChecked()
        art_mode = self._get_artifact_mode()
        use_nnls = getattr(self, 'chk_nnls', None) and self.chk_nnls.isChecked()
        is_indep = getattr(self, 'chk_indep_t0', None) and self.chk_indep_t0.isChecked()

        # 1. Instanciamos el MOTOR PURO
        num_effective_comp = len(self.current_custom_model.states) if (self.model_type == "Custom GUI Model" and getattr(self, 'current_custom_model', None) is not None) else self.numExp
        self.fitter = GlobalFitter(
            dataset=dataset, 
            model_type=self.model_type, 
            num_exp=num_effective_comp, 
            use_artifact=use_art, 
            artifact_mode=art_mode, 
            use_nnls=use_nnls, 
            is_indep_t0=is_indep,
            custom_model=getattr(self, 'current_custom_model', None)
        )
            
        num_kin = self.fitter.get_num_kinetic_params()
        self.is_fixed[num_kin:] = True
        
        self.save_state_to_history()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Iterating: %v%")
        
        # 2. Pasamos el constructor de matriz DEL MOTOR al Worker (adiós dependencia gráfica)
        
        enforce_order = self.model_type in ("Parallel", "Sequential", "Stretched Exponential")
        self.worker = FitWorker(
            dataset, self.fitter.build_C_matrix, self.ini, self.limi, self.lims, 
            self.is_fixed, num_effective_comp, use_nnls, enforce_tau_order=enforce_order
        )
        
        self.worker.progress_update.connect(self.progress_bar.setValue)
        self.worker.finished_success.connect(self._on_fit_success)
        self.worker.finished_error.connect(self._on_fit_error)
        self.worker.start()
        
    def _on_fit_success(self, result):
        """Recibe el FitResult de la API y actualiza la interfaz visual."""
        # Recuperamos la información del objeto devuelto por la API
        self.fit_result = result.scipy_res  # Necesario para que la GUI calcule las estadísticas luego
        self.fit_x = result.x
        
        A_base = self._get_num_kinetic_params()
        
        if self.model_type == "Custom GUI Model":
            num_species = len(self.current_custom_model.states)
        elif self.model_type == 'Damped Oscillation':
            num_species = self.numExp + 1
        else:
            num_species = self.numExp
            
        # Actualizamos el vector con las amplitudes encontradas de forma segura
        amp_flat = result.amplitudes[:num_species, :].T.flatten()
        if len(self.fit_x) == A_base + len(amp_flat):
            self.fit_x[A_base:] = amp_flat
        else:
            self.fit_x = np.concatenate([self.fit_x[:A_base], amp_flat])
        
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("Fit Completed")
        
        # Procedemos a los cálculos de errores (Jacobiano) y guardado de archivos
        self._postprocess_fit_and_save()
        if not getattr(self, 'is_batch_running', False):
            self._cleanup_fit_ui()

    def _on_fit_error(self, error_msg):
        """Se ejecuta si el QThread falla o es abortado."""
        if getattr(self, 'is_batch_running', False):
            self.last_batch_error = error_msg
            print(f"[Batch Fit Error]: {error_msg}")
        else:
            if error_msg == "Aborted":
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Aborted fit", "Manually aborted fit.")
            else:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.critical(self, "Fit Error", error_msg)
            self._cleanup_fit_ui()
            
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Idle")
        
    def _cleanup_fit_ui(self):
        """Restaura los botones de la interfaz tras el ajuste."""
        # Volvemos a encender todo
        self.tab_data.setEnabled(True)
        if hasattr(self, 'tab_export'): self.tab_export.setEnabled(True)
        
        if hasattr(self, 'gb_model'): self.gb_model.setEnabled(True)
        if hasattr(self, 'gb_work'): self.gb_work.setEnabled(True)
        
        self.btn_batch.setEnabled(True)
        self.btn_run.setEnabled(True)
        
        # Apagamos el ABORT
        self.btn_abort.setEnabled(False)
        self.btn_abort.setText("ABORT")
        
    def _get_num_kinetic_params(self):
        """Número de parámetros NO lineales (w, t0 global, taus + t0s locales)."""
        model_str = self.combo_model.currentText()
        
        if "Oscillation" in model_str or self.model_type == "Damped Oscillation":
            base = 2 + self.numExp + 3
        elif "Stretched" in model_str or self.model_type == "Stretched Exponential":
            base = 2 + 2 * self.numExp
        elif ("Custom GUI Model" in model_str or self.model_type == "Custom GUI Model") and getattr(self, 'current_custom_model', None) is not None:
            base = 2 + len(self.current_custom_model.param_labels)
        else:
            base = 2 + self.numExp
            
        # Si el usuario quiere un t0 por cada lambda, sumamos esa cantidad al vector
        if getattr(self, 'chk_indep_t0', None) and self.chk_indep_t0.isChecked():
            numWL = self.data_c.shape[0] if getattr(self, 'data_c', None) is not None else 0
            return base + numWL
            
        return base
    
    def _get_artifact_mode(self):
        
        """Devuelve el string de modo del artefacto coherente para fit.py."""
        if not (getattr(self, 'chk_artifact', None) and self.chk_artifact.isChecked()):
            return 'both'  # irrelevante si use_art=False, pero devolvemos un valor válido
        idx = self.combo_artifact_mode.currentIndex()
        
        return ['both', 'raman', 'xpm'][idx]
      
    def _postprocess_fit_and_save(self):
        """Usa el motor independiente para procesar los errores y sincronizar la GUI."""
        if self.fit_result is None or not hasattr(self, 'fitter'): return

        # 1. Determinar qué parámetros son libres con protección de longitud
        L_total = len(self.fit_x)
        free_indices = getattr(self, 'free_indices', None)
        if free_indices is None or len(free_indices) == 0:
            num_kin = self.fitter.get_num_kinetic_params()
            is_kinetic = np.zeros(L_total, dtype=bool)
            is_kinetic[:num_kin] = True
            if hasattr(self, 'is_fixed') and len(self.is_fixed) == L_total:
                free_indices = np.where(is_kinetic & ~self.is_fixed)[0]
            else:
                free_indices = np.where(is_kinetic)[0]

        # 2. EL MOTOR HACE TODO EL TRABAJO MATEMÁTICO PESADO EN LA SOMBRA
        self.fitter.process_fit_results(self.fit_result, self.fit_x, free_indices)

        # 3. Sincronizamos las variables extraídas por el motor para dibujarlas
        self.fit_fitres = self.fitter.fitres
        self.fit_resid = self.fitter.resid
        self.As = self.fitter.As
        self.errAs = self.fitter.errAs
        self.fitted_model_type = self.model_type
        if ("Oscillation" in self.model_type) and hasattr(self.fitter, 'Bs') and self.fitter.Bs is not None:
            self.Bs = self.fitter.Bs
            self.errBs = self.fitter.errBs
        else:
            self.Bs = None
            self.errBs = None
            
        self.ci = self.fitter.ci
        self.extracted_taus = self.fitter.extracted_taus
        self.extracted_errtaus = self.fitter.extracted_errtaus
        
        self.extracted_betas = getattr(self.fitter, 'extracted_betas', None)
        self.extracted_errbetas = getattr(self.fitter, 'extracted_errbetas', None)
        self.Artifact_Spectra = getattr(self.fitter, 'Artifact_Spectra', None)
        
        self.fit_condition_number = self.fitter.condition_number
        self.sloppiest_direction = self.fitter.sloppiest_direction
        self.param_correlation = self.fitter.param_correlation
        self.param_correlation_indices = self.fitter.param_correlation_indices
        self.S_T_full = self.fitter.S_T_full

        # 4. El motor exporta los archivos .txt y .npy
        outdir = self.current_batch_outdir if getattr(self, 'is_batch_running', False) else os.path.join(self.base_dir, "fit")
        self.fitter.save_results(outdir, self.ini)

        # 5. La interfaz se limita a refrescar los gráficos
        self._update_fit_canvas()
        self._update_resid_canvas()
        self.btn_show_das.setEnabled(True)
        self.btn_export_pdf.setEnabled(True)

        # 6. Cachear resultados de este dataset para navegación fluida
        curr_idx = self.combo_active_dataset.currentIndex()
        filename = self.filenames[curr_idx] if (curr_idx >= 0 and curr_idx < len(self.filenames)) else f"Dataset_{curr_idx+1}"
        if not hasattr(self, 'batch_fit_results'):
            self.batch_fit_results = {}
            
        cached_entry = {
            'fit_result': self.fit_result,
            'fit_x': self.fit_x.copy(),
            'fit_fitres': self.fit_fitres.copy(),
            'fit_resid': self.fit_resid.copy(),
            'As': self.As.copy(),
            'errAs': self.errAs.copy() if self.errAs is not None else None,
            'Bs': self.Bs.copy() if self.Bs is not None else None,
            'errBs': self.errBs.copy() if self.errBs is not None else None,
            'ci': self.ci.copy() if self.ci is not None else None,
            'extracted_taus': self.extracted_taus.copy() if self.extracted_taus is not None else None,
            'extracted_errtaus': self.extracted_errtaus.copy() if self.extracted_errtaus is not None else None,
            'extracted_betas': self.extracted_betas.copy() if self.extracted_betas is not None else None,
            'extracted_errbetas': self.extracted_errbetas.copy() if self.extracted_errbetas is not None else None,
            'Artifact_Spectra': self.Artifact_Spectra.copy() if self.Artifact_Spectra is not None else None,
            'fitted_model_type': self.fitted_model_type,
            'fitter': self.fitter,
            'fit_condition_number': self.fit_condition_number,
            'sloppiest_direction': self.sloppiest_direction,
            'param_correlation': self.param_correlation,
            'param_correlation_indices': self.param_correlation_indices,
            'S_T_full': self.S_T_full,
            'rmsd': np.sqrt(np.mean(self.fit_resid**2)) if self.fit_resid is not None else 0.0,
        }
        self.batch_fit_results[filename] = cached_entry
        if curr_idx >= 0:
            self.batch_fit_results[curr_idx] = cached_entry
        
        if not getattr(self, 'is_batch_running', False):
            self.show_results_summary()
            rmsd = np.sqrt(np.mean(self.fit_resid**2))
            self.show_toast(f"Fit completed successfully (RMSD: {rmsd:.2e})")
            
    def compute_profile_likelihood(self, param_idx, n_steps=15, confidence=0.95, span_sigma=6, progress_callback=None):
        """
        Delega el cálculo del profile likelihood al motor matemático independiente,
        pasándole callbacks para mantener la interfaz receptiva.
        """
        if not hasattr(self, 'fitter') or self.fitter is None:
            raise RuntimeError("Ejecuta un fit global primero antes de analizar la identificabilidad.")
            
        try:
            # Creamos una micro-función que lee la variable de aborto de la GUI
            def check_abort():
                return getattr(self, '_abort_fit', False)
                
            # Llamamos al motor pasándole los límites de la GUI y los callbacks
            result = self.fitter.run_profile_likelihood(
                param_idx=param_idx,
                ini=self.ini,
                limi=self.limi,
                lims=self.lims,
                is_fixed=getattr(self, 'is_fixed', np.zeros(len(self.ini), dtype=bool)),
                n_steps=n_steps,
                confidence=confidence,
                span_sigma=span_sigma,
                progress_callback=progress_callback,
                abort_callback=check_abort
            )
            return result
            
        finally:
            # Asegurarnos de apagar la señal de aborto al terminar
            self._abort_fit = False
            
    def _get_kinetic_param_label(self, i):
            """
            Devuelve una etiqueta legible para el parámetro cinético (NO lineal) de 
            índice i (w, t0, taus, parámetros de oscilación o de un modelo custom).
            Solo tiene sentido para i < num_kin_params; las amplitudes espectrales 
            no se etiquetan aquí.
            """
            model_str = self.combo_model.currentText()
            is_oscillation = "Oscillation" in model_str
            is_custom = "Custom GUI Model" in model_str
    
            # 1. Calculamos la longitud de la "base" cinética según el modelo elegido
            base_kin = 2 + self.numExp
            if is_oscillation:
                base_kin = 2 + self.numExp + 3
            elif is_custom and getattr(self, 'current_custom_model', None) is not None:
                base_kin = 2 + len(self.current_custom_model.param_labels)
    
            # 2. NUEVO: Si la opción de t0 por píxel está activa y el índice cae en esa zona
            if getattr(self, 'chk_indep_t0', None) and self.chk_indep_t0.isChecked():
                if base_kin <= i < self._get_num_kinetic_params():
                    idx_wl = i - base_kin
                    wl_val = self._wl_proc[idx_wl] if hasattr(self, '_wl_proc') else idx_wl
                    return f"t0 @ {wl_val:.1f} nm"
    
            # 3. Lógica ORIGINAL intacta para los parámetros base
            if is_custom and getattr(self, 'current_custom_model', None) is not None:
                model = self.current_custom_model
                if i == 0: return "w (IRF Width)"
                if i == 1: return "t0 (Global Time Zero)"
                
                idx_param = i - 2
                if 0 <= idx_param < len(model.param_labels):
                    p_name = model.param_labels[idx_param]
                    name_low = p_name.lower()
                    
                    # Asignación inteligente de etiqueta visual según el nombre
                    if "gamma" in name_low or "yield" in name_low:
                        p_type = "Branching Ratio"
                    elif name_low.startswith('k') or "rate" in name_low:
                        p_type = "Rate (ps⁻¹)"
                    elif name_low == 'a' or name_low.startswith('a_'):
                        p_type = "Amplitude"
                    else:
                        p_type = "Tau (ps)"
                        
                    return f"[{p_type}] {p_name}"
                    
                return f"param[{i}]"
    
            if is_oscillation:
                if i == 0: return "w (IRF Width)"
                if i == 1: return "t0 (Global Time Zero)"
                if i < 2 + self.numExp: return f"τ{i-1}"
                if i == 2 + self.numExp: return "α (Damping)"
                if i == 2 + self.numExp + 1: return "ω (Frequency)"
                if i == 2 + self.numExp + 2: return "φ (Phase)"
                return f"param[{i}]"
    
            if i == 0: return "w (IRF Width)"
            if i == 1: return "t0 (Global Time Zero)"
            if i < 2 + self.numExp: return f"τ{i-1}"
            
            return f"param[{i}]"
        
    def open_identifiability_dialog(self):
        """Abre el diálogo de análisis de identificabilidad (profile likelihood)."""
        from spark.gui.dialogs import ParameterIdentifiabilityDialog
        dlg = ParameterIdentifiabilityDialog(self, parent=self)
        dlg.exec_()
        
    def show_results_summary(self):
            """Displays a popup window detailing the final global parameters derived from the fit."""
            if self.fit_x is None: return
    
            dlg = QDialog(self)
            dlg.setWindowTitle("Fit Results Summary")
            dlg.resize(460, 420)
            layout = QVBoxLayout(dlg)
            
            table = QTableWidget()
            layout.addWidget(table)
            
            results = [
                ["w (IRF)", f"{self.fit_x[0]:.4f}"],
                ["t0", f"{self.fit_x[1]:.4f}"]
            ]
            
            if self.model_type == "Custom GUI Model":
                for i, label in enumerate(self.current_custom_model.param_labels):
                    val = self.extracted_taus[i]
                    error = self.extracted_errtaus[i] if self.extracted_errtaus is not None else 0.0
                    results.append([f"{label}", f"{val:.4f} ± {error:.4f}"])
                    
            elif self.model_type == "Stretched Exponential":
                from scipy.special import gamma
                for i in range(self.numExp):
                    val_tau = self.extracted_taus[i]
                    err_tau = self.extracted_errtaus[i] if self.extracted_errtaus is not None else 0.0
                    val_beta = self.extracted_betas[i]
                    err_beta = self.extracted_errbetas[i] if self.extracted_errbetas is not None else 0.0
                    
                    mean_tau = (val_tau / val_beta) * gamma(1.0 / val_beta)
                    
                    results.append([f"τ{i+1} (char.)", f"{val_tau:.2f} ± {err_tau:.2f} ps"])
                    results.append([f"β{i+1}", f"{val_beta:.3f} ± {err_beta:.3f}"])
                    results.append([f"⟨τ{i+1}⟩ (mean)", f"{mean_tau:.2f} ps"])
                    
            else:
                for i in range(self.numExp):
                    val = self.extracted_taus[i]
                    error = self.extracted_errtaus[i] if self.extracted_errtaus is not None else 0.0
                    results.append([f"τ{i+1}", f"{val:.2f} ± {error:.2f} ps"])
    
            table.setRowCount(len(results))
            table.setColumnCount(2)
            table.setHorizontalHeaderLabels(["Parameter", "Final Value"])
            table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            
            for i, (name, val) in enumerate(results):
                table.setItem(i, 0, QTableWidgetItem(name))
                table.setItem(i, 1, QTableWidgetItem(val))
                
            # --- Diagnóstico de identificabilidad ---
            gb_diag = QGroupBox("Parameter Identifiability")
            v_diag = QVBoxLayout(gb_diag)
    
            cond = getattr(self, 'fit_condition_number', None)
            if cond is not None and np.isfinite(cond):
                if cond < 1e2:
                    color, msg = "#10B981", "Well-conditioned"
                elif cond < 1e4:
                    color, msg = "#F59E0B", "Moderate correlation"
                else:
                    color, msg = "#EF4444", "Poorly identifiable"
                lbl_cond = QLabel(f"Condition number: {cond:.2e}  ({msg})")
                lbl_cond.setStyleSheet(f"color: {color}; font-weight: bold;")
            else:
                lbl_cond = QLabel("Condition number: N/A")
            v_diag.addWidget(lbl_cond)
    
            sloppy = getattr(self, 'sloppiest_direction', [])
            if cond is not None and cond > 1e3 and len(sloppy) >= 2:
                terms = ", ".join(
                    f"{self._get_kinetic_param_label(idx)} ({w:+.2f})" for idx, w in sloppy[:4]
                )
                lbl_sloppy = QLabel(f"Least determined combination: {terms}")
                lbl_sloppy.setWordWrap(True)
                lbl_sloppy.setStyleSheet("color: #6C757D; font-style: italic;")
                v_diag.addWidget(lbl_sloppy)
    
            btn_identifiability = QPushButton("Analyze Identifiability (Profile Likelihood)...")
            btn_identifiability.setStyleSheet("background-color: #3C5488; color: white;")
            btn_identifiability.clicked.connect(self.open_identifiability_dialog)
            v_diag.addWidget(btn_identifiability)
    
            layout.addWidget(gb_diag)
            
    
            btn_close = QPushButton("Close")
            btn_close.clicked.connect(dlg.accept)
            layout.addWidget(btn_close)
            dlg.exec_()
            
    def export_pdf_report(self):
        """Genera un reporte PDF vectorial (calidad publicación) con el resumen del ajuste global."""
        if self.fit_x is None or self.data_c is None:
            return

        outdir = self.current_batch_outdir if getattr(self, 'is_batch_running', False) else os.path.join(self.base_dir, "fit")
        os.makedirs(outdir, exist_ok=True)

        base_name = "Dataset"
        idx = self.combo_active_dataset.currentIndex()
        if idx >= 0 and idx < len(self.filenames):
            base_name = os.path.splitext(self.filenames[idx])[0]

        pdf_path = os.path.join(outdir, f"{base_name}_FitReport.pdf")

        # ------------------------------------------------------------------
        # Comprobación previa: si el PDF ya existe y está abierto en un lector
        # externo (Adobe/Edge/etc.), Windows bloquea el fichero y la escritura
        # falla con PermissionError. Lo detectamos ANTES de generar las páginas
        # (que son costosas) y usamos automáticamente un nombre alternativo
        # con marca de tiempo, en vez de perder el trabajo hecho.
        # ------------------------------------------------------------------
        def _first_writable_path(path):
            try:
                with open(path, 'ab'):
                    pass
                return path, False
            except PermissionError:
                root, ext = os.path.splitext(path)
                alt_path = f"{root}_{datetime.datetime.now().strftime('%H%M%S')}{ext}"
                return alt_path, True
            except OSError:
                # Otros problemas de ruta (permiso de carpeta, disco, etc.)
                return path, False

        pdf_path, used_fallback_name = _first_writable_path(pdf_path)
        if used_fallback_name:
            QMessageBox.information(
                self, "File in use",
                f"The report file appears to be open in another program (e.g. a PDF viewer).\n"
                f"Saving the new report as:\n{os.path.basename(pdf_path)}\n\n"
                f"Tip: close the PDF viewer before exporting to reuse the original filename."
            )

        QApplication.setOverrideCursor(Qt.WaitCursor)

        # ------------------------------------------------------------------
        # Estilo "publicación": tipografía y ejes consistentes en todo el PDF.
        # pdf.fonttype/ps.fonttype = 42 embebe el texto como TrueType real
        # (en vez de Type 3 bitmap): se ve nítido a cualquier zoom y queda
        # editable si alguien retoca la figura en Illustrator/Inkscape.
        # ------------------------------------------------------------------
        from matplotlib.patches import Rectangle

        pub_rc = {
            'font.family':       'sans-serif',
            'font.sans-serif':   ['Arial', 'Helvetica', 'DejaVu Sans'],
            'font.size':         10,
            'axes.titlesize':    12,
            'axes.titleweight':  'bold',
            'axes.labelsize':    10,
            'axes.linewidth':    1.1,
            'axes.edgecolor':    '#333333',
            'xtick.direction':   'in',
            'ytick.direction':   'in',
            'xtick.labelsize':   9,
            'ytick.labelsize':   9,
            'legend.frameon':    True,
            'legend.framealpha': 0.9,
            'legend.edgecolor':  '#B0B0B0',
            'figure.facecolor':  'white',
            'savefig.facecolor': 'white',
            'pdf.fonttype':      42,
            'ps.fonttype':       42,
        }

        HEADER_BLUE = '#1B2A4A'
        ACCENT_BLUE = '#3C5488'
        WARN_ROW = '#FBE3E3'
        GRID_KW = dict(linestyle=':', alpha=0.35, linewidth=0.7)

        # --- Diagnóstico de identificabilidad (condition number + dirección "sloppy") ---
        cond = getattr(self, 'fit_condition_number', np.inf)
        if cond is not None and np.isfinite(cond):
            if cond < 1e2:
                cond_color, cond_msg = '#10B981', 'Well-conditioned'
            elif cond < 1e4:
                cond_color, cond_msg = '#F59E0B', 'Moderate correlation'
            else:
                cond_color, cond_msg = '#EF4444', 'Poorly identifiable'
        else:
            cond_color, cond_msg = '#6C757D', 'N/A'

        sloppy = getattr(self, 'sloppiest_direction', [])
        sloppy_idx_set = set()
        sloppy_text = None
        if cond is not None and np.isfinite(cond) and cond > 1e3 and len(sloppy) >= 2:
            sloppy_idx_set = {i for i, w in sloppy if abs(w) > 0.15}
            terms = ", ".join(f"{self._get_kinetic_param_label(i)} ({w:+.2f})" for i, w in sloppy[:4])
            sloppy_text = f"Least determined combination: {terms}"

        have_corr = (getattr(self, 'param_correlation', None) is not None and
                     getattr(self, 'param_correlation_indices', None) is not None and
                     len(self.param_correlation_indices) >= 2)

        total_pages = 5 + (1 if have_corr else 0) 
        page_state = {'n': 1}

        def _add_footer(fig):
            """Pie de página discreto: nombre de archivo, fecha y nº de página (auto-incrementa)."""
            fig.text(0.02, 0.015, base_name, fontsize=8, color='#8A8A8A', ha='left')
            fig.text(0.98, 0.015, f"Page {page_state['n']} / {total_pages}", fontsize=8, color='#8A8A8A', ha='right')
            fig.text(0.5, 0.015, datetime.datetime.now().strftime('%Y-%m-%d'),
                     fontsize=8, color='#8A8A8A', ha='center')
            page_state['n'] += 1

        try:
            with plt.rc_context(pub_rc):
                with PdfPages(pdf_path) as pdf:

                    # ==========================================================
                    # PÁGINA 1 — PORTADA, DIAGNÓSTICO Y TABLA DE PARÁMETROS
                    # ==========================================================
                    fig1 = Figure(figsize=(8.27, 11.69))
                    FigureCanvasAgg(fig1)

                    fig1.add_artist(Rectangle((0, 0.93), 1, 0.07, transform=fig1.transFigure,
                                               facecolor=HEADER_BLUE, edgecolor='none', zorder=0))
                    fig1.text(0.5, 0.965, "Ultrafast Spectroscopy Analysis Report",
                              ha='center', va='center', fontsize=19, fontweight='bold', color='white')

                    fig1.text(0.5, 0.905, f"Dataset:  {base_name}", ha='center', fontsize=13, color='#222222')
                    fig1.text(0.5, 0.882, f"Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
                              ha='center', fontsize=9.5, color='gray')

                    # --- Bloque "Model Details" / "Fit Quality & Identifiability" ---
                    fig1.add_artist(Rectangle((0.08, 0.735), 0.84, 0.135, transform=fig1.transFigure,
                                               facecolor='#F2F4F8', edgecolor='#D9DEE7', linewidth=1.0, zorder=0))

                    fig1.text(0.11, 0.845, "Model Details", fontsize=11.5, fontweight='bold', color=ACCENT_BLUE)
                    fig1.text(0.11, 0.822, f"Type:  {self.model_type}", fontsize=10)
                    fig1.text(0.11, 0.800, f"Components:  {self.numExp}", fontsize=10)
                    use_art = getattr(self, 'chk_artifact', None) and self.chk_artifact.isChecked()
                    fig1.text(0.11, 0.778,
                              f"Coherent artifact:  {'Yes (' + self._get_artifact_mode() + ')' if use_art else 'No'}",
                              fontsize=10)
                    fig1.text(0.11, 0.756,
                              f"Technique:  {getattr(self, 'tech', 'N/A')}   |   "
                              f"NNLS (positive spectra):  {'Yes' if getattr(self, 'chk_nnls', None) and self.chk_nnls.isChecked() else 'No'}",
                              fontsize=9.5)

                    fig1.text(0.55, 0.845, "Fit Quality & Identifiability", fontsize=11.5, fontweight='bold', color=ACCENT_BLUE)
                    rmsd = np.sqrt(np.mean(self.fit_resid ** 2))
                    fig1.text(0.55, 0.822, f"RMSD:  {rmsd:.2e}", fontsize=10)
                    fig1.text(0.55, 0.800,
                              f"Condition number:  {cond:.2e}" if np.isfinite(cond) else "Condition number:  N/A",
                              fontsize=10)
                    fig1.text(0.55, 0.778, cond_msg, fontsize=10, fontweight='bold', color=cond_color)
                    if sloppy_text is not None:
                        fig1.text(0.55, 0.752, sloppy_text, fontsize=8, style='italic', color='#555555', wrap=True)

                    # --- Bloque "Data Pre-processing" (reproducibilidad) ---
                    fig1.add_artist(Rectangle((0.08, 0.60), 0.84, 0.115, transform=fig1.transFigure,
                                               facecolor='#F2F4F8', edgecolor='#D9DEE7', linewidth=1.0, zorder=0))
                    fig1.text(0.11, 0.688, "Data Pre-processing", fontsize=11.5, fontweight='bold', color=ACCENT_BLUE)

                    excl_str = self.line_exclude.text().strip() if hasattr(self, 'line_exclude') else ''
                    prep_left = [
                        f"Baseline points:  {self.spin_bl.value() if hasattr(self, 'spin_bl') else 'N/A'}",
                        f"Wavelength range:  {self.spin_wl_min.value():.1f} \u2013 {self.spin_wl_max.value():.1f} nm"
                        if hasattr(self, 'spin_wl_min') else "Wavelength range:  N/A",
                        f"Excluded WLs:  {excl_str if excl_str else 'None'}",
                    ]
                    prep_right = [
                        f"Time range:  {self.spin_t_min.value():.3g} \u2013 {self.spin_t_max.value():.3g} ps"
                        if hasattr(self, 'spin_t_min') else "Time range:  N/A",
                        f"Binning:  {self.spin_bin.value() if hasattr(self, 'spin_bin') else 1}   |   "
                        f"t<0 zeroed:  {'Yes' if hasattr(self, 'chk_zero_neg') and self.chk_zero_neg.isChecked() else 'No'}",
                        f"Normalized (max|\u0394A|=1):  {'Yes' if hasattr(self, 'chk_norm_data') and self.chk_norm_data.isChecked() else 'No'}",
                    ]
                    for k, line in enumerate(prep_left):
                        fig1.text(0.11, 0.665 - k * 0.022, line, fontsize=9.5)
                    for k, line in enumerate(prep_right):
                        fig1.text(0.55, 0.665 - k * 0.022, line, fontsize=9.5)

                    # --- Tabla de parámetros (CON INITIAL GUESS) ---
                    table_data = [["Parameter", "Initial Guess", "Final Value", "Error", "Unit"]]
                    table_data.append(["w (IRF FWHM)", f"{self.ini[0]:.4g}", f"{self.fit_x[0]:.4g}",
                                       f"{self.ci[0]:.4g}" if len(self.ci) > 0 else "N/A", "ps"])
                    table_data.append(["t0 (Time Zero)", f"{self.ini[1]:.4g}", f"{self.fit_x[1]:.4g}",
                                       f"{self.ci[1]:.4g}" if len(self.ci) > 1 else "N/A", "ps"])

                    if self.model_type == "Custom GUI Model":
                        for i, label in enumerate(self.current_custom_model.param_labels):
                            ini_val = self.ini[2+i]
                            val = self.extracted_taus[i]
                            err = self.extracted_errtaus[i] if self.extracted_errtaus is not None else 0.0
                            unit = "" if "gamma" in label.lower() else "ps"
                            table_data.append([label, f"{ini_val:.4g}", f"{val:.4g}", f"{err:.4g}", unit])
                    else:
                        for i in range(self.numExp):
                            ini_val = self.ini[2+i]
                            val = self.extracted_taus[i]
                            err = self.extracted_errtaus[i] if self.extracted_errtaus is not None else 0.0
                            table_data.append([f"tau_{i+1}", f"{ini_val:.4g}", f"{val:.4g}", f"{err:.4g}", "ps"])

                    # Hacemos la tabla un poco más ancha para que quepan las 5 columnas
                    ax_tab = fig1.add_axes([0.05, 0.16, 0.90, 0.415]) 
                    ax_tab.axis('off')
                    table = ax_tab.table(cellText=table_data, loc='center', cellLoc='center')
                    table.auto_set_font_size(False)
                    table.set_fontsize(10.5)
                    table.scale(1, 1.9)

                    # Estilizar la cabecera (5 columnas)
                    for j in range(5):
                        cell = table[(0, j)]
                        cell.set_facecolor(ACCENT_BLUE)
                        cell.get_text().set_color('white')
                        cell.set_text_props(weight='bold')

                    # Estilizar filas e identificar parámetros "resbaladizos"
                    any_highlighted = False
                    for i in range(1, len(table_data)):
                        param_idx = i - 1  
                        is_risky = param_idx in sloppy_idx_set
                        any_highlighted = any_highlighted or is_risky
                        row_color = WARN_ROW if is_risky else ('#F7F8FA' if i % 2 == 0 else 'white')
                        for j in range(5):
                            table[(i, j)].set_facecolor(row_color)
                            table[(i, j)].set_edgecolor('#D9DEE7')

                    if any_highlighted:
                        fig1.text(0.09, 0.145,
                                  "Highlighted rows: parameters contributing most to the least-determined "
                                  "combination (see Fit Quality & Identifiability above).",
                                  fontsize=8, style='italic', color='#B04545')

                    _add_footer(fig1)
                    pdf.savefig(fig1)

                    # ==========================================================
                    # PÁGINA 2 (opcional) — MATRIZ DE CORRELACIÓN DE PARÁMETROS
                    # ==========================================================
                    if have_corr:
                        fig_corr = Figure(figsize=(8.27, 8.27))
                        FigureCanvasAgg(fig_corr)
                        ax_c = fig_corr.add_axes([0.28, 0.12, 0.62, 0.62])

                        corr = self.param_correlation
                        corr_idxs = list(self.param_correlation_indices)
                        corr_labels = [self._get_kinetic_param_label(i) for i in corr_idxs]

                        im = ax_c.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
                        ax_c.set_xticks(range(len(corr_idxs)))
                        ax_c.set_xticklabels(corr_labels, rotation=45, ha='right', fontsize=8.5)
                        ax_c.set_yticks(range(len(corr_idxs)))
                        ax_c.set_yticklabels(corr_labels, fontsize=8.5)
                        ax_c.set_title("Kinetic Parameter Correlation Matrix", pad=12)

                        for i in range(len(corr_idxs)):
                            for j in range(len(corr_idxs)):
                                val = corr[i, j]
                                txt_color = 'white' if abs(val) > 0.6 else 'black'
                                ax_c.text(j, i, f"{val:.2f}", ha='center', va='center',
                                          fontsize=7.5, color=txt_color)

                        cax_c = fig_corr.add_axes([0.28, 0.075, 0.62, 0.02])
                        cb_c = fig_corr.colorbar(im, cax=cax_c, orientation='horizontal')
                        cb_c.set_label('Correlation coefficient')

                        fig_corr.text(0.5, 0.95,
                                      "Values near \u00b11 mean two parameters cannot be determined independently "
                                      "from the data (e.g. two very similar lifetimes).",
                                      ha='center', fontsize=9, style='italic', color='#555555', wrap=True)

                        _add_footer(fig_corr)
                        pdf.savefig(fig_corr)

                    # ==========================================================
                    # PÁGINA — MAPAS 2D COMPARATIVOS
                    # ==========================================================
                    fig2 = Figure(figsize=(11.69, 8.27))
                    FigureCanvasAgg(fig2)

                    wl = getattr(self, '_wl_proc', self.WL)
                    td = getattr(self, '_td_proc', self.TD)

                    # Más margen inferior: deja sitio de sobra para las etiquetas del eje X y
                    # la barra de color, evitando el solapamiento visto en versiones anteriores.
                    gs2 = gridspec.GridSpec(1, 3, figure=fig2, wspace=0.22,
                                            left=0.07, right=0.965, top=0.88, bottom=0.30)

                    ax1 = fig2.add_subplot(gs2[0, 0])
                    ax2 = fig2.add_subplot(gs2[0, 1], sharex=ax1, sharey=ax1)
                    ax3 = fig2.add_subplot(gs2[0, 2], sharex=ax1, sharey=ax1)

                    v_max = np.nanmax(self.data_c)
                    v_min = np.nanmin(self.data_c)
                    if hasattr(self, 'chk_sym_cmap') and self.chk_sym_cmap.isChecked():
                        max_abs = max(abs(v_min), abs(v_max))
                        v_min, v_max = -max_abs, max_abs

                    cmap = getattr(self, 'combo_cmap', None).currentText() if hasattr(self, 'combo_cmap') else 'jet'

                    m1 = ax1.pcolormesh(wl, td, self.data_c.T, cmap=cmap, shading='auto',
                                        vmin=v_min, vmax=v_max, rasterized=True)
                    m2 = ax2.pcolormesh(wl, td, self.fit_fitres.T, cmap=cmap, shading='auto',
                                        vmin=v_min, vmax=v_max, rasterized=True)

                    # Residuos: colormap divergente y límites simétricos centrados en 0 -- mucho
                    # más legible que reutilizar 'jet' para una magnitud con signo.
                    res_vals = self.fit_resid.flatten()
                    r_abs = np.nanpercentile(np.abs(res_vals), 99)
                    r_min, r_max = -r_abs, r_abs
                    m3 = ax3.pcolormesh(wl, td, self.fit_resid.T, cmap='RdBu_r', shading='auto',
                                        vmin=r_min, vmax=r_max, rasterized=True)

                    titles = ["1. Experimental Data", "2. Global Fit Model", "3. Residuals"]
                    for ax, title in zip([ax1, ax2, ax3], titles):
                        ax.set_title(title, pad=8)
                        ax.set_xlabel("Wavelength (nm)")
                        if hasattr(self, 'yscale') and self.yscale == 'symlog':
                            ax.set_yscale('symlog', linthresh=2)

                    ax1.set_ylabel("Delay (ps)")
                    plt.setp(ax2.get_yticklabels(), visible=False)
                    plt.setp(ax3.get_yticklabels(), visible=False)

                    # Barras de color horizontales, claramente separadas del eje X (nunca se tocan)
                    cax1 = fig2.add_axes([0.10, 0.115, 0.40, 0.028])
                    cax2 = fig2.add_axes([0.60, 0.115, 0.365, 0.028])

                    cb1 = fig2.colorbar(m2, cax=cax1, orientation='horizontal')
                    cb1.set_label(r'$\Delta A$  (Data / Fit)')
                    cb1.ax.tick_params(labelsize=9)

                    cb2 = fig2.colorbar(m3, cax=cax2, orientation='horizontal')
                    cb2.set_label('Residual')
                    cb2.ax.tick_params(labelsize=9)

                    fig2.suptitle("2D Maps Comparison", fontsize=15, fontweight='bold', y=0.965)
                    _add_footer(fig2)
                    pdf.savefig(fig2)

                    # ==========================================================
                    # PÁGINA — ESPECTROS SAS/DAS, ARTEFACTO E HISTOGRAMA
                    # ==========================================================
                    fig3 = Figure(figsize=(8.27, 11.69))
                    FigureCanvasAgg(fig3)

                    # Si hay artefacto usamos 3 filas, si no, dividimos la hoja en 2
                    if use_art:
                        ax_das = fig3.add_subplot(311)
                        ax_art = fig3.add_subplot(312)
                        ax_hist = fig3.add_subplot(313)
                    else:
                        ax_das = fig3.add_axes([0.12, 0.55, 0.80, 0.35])
                        ax_hist = fig3.add_axes([0.12, 0.08, 0.80, 0.35])
                    
                    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
                    markers = ['o', 's', '^', 'D', 'v', 'p']

                    num_curves = len(self.current_custom_model.states) if (self.model_type == "Custom GUI Model" and getattr(self, 'current_custom_model', None) is not None) else self.numExp
                    for n in range(num_curves):
                        color = colors[n % len(colors)]
                        marker = markers[n % len(markers)]

                        if self.model_type == "Custom GUI Model":
                            state_name = self.current_custom_model.states[n]
                            lbl = f"Species: {state_name}"
                            
                        elif self.model_type == "Stretched Exponential":
                            tau_val = self.extracted_taus[n]
                            err_tau = self.extracted_errtaus[n] if (self.extracted_errtaus is not None and not np.isnan(self.extracted_errtaus[n])) else 0.0
                            beta_val = getattr(self, 'extracted_betas', [1.0]*self.numExp)[n]
                            lbl = rf"$\tau_{{{n+1}}}$ = {tau_val:.2f} $\pm$ {err_tau:.2f} ps ($\beta$ = {beta_val:.2f})"
                        else:
                            tau_val = self.extracted_taus[n]
                            err_tau = self.extracted_errtaus[n] if (self.extracted_errtaus is not None and not np.isnan(self.extracted_errtaus[n])) else 0.0
                            lbl = rf"$\tau_{{{n+1}}}$ = {tau_val:.2f} $\pm$ {err_tau:.2f} ps"

                        if self.errAs is not None:
                            err_y = np.nan_to_num(self.errAs[n])
                            ax_das.errorbar(wl, self.As[n], yerr=err_y, label=lbl, color=color,
                                            fmt=f'-{marker}', markersize=5, capsize=3.5, elinewidth=1.1,
                                            markeredgewidth=0.8)
                        else:
                            ax_das.plot(wl, self.As[n], f'-{marker}', label=lbl, color=color, markersize=5)

                    ax_das.set_xlabel("Wavelength (nm)")
                    ax_das.set_ylabel(r"Amplitude ($\Delta A$)")
                    ax_das.set_title("Species / Decay Associated Spectra (SAS / DAS)")
                    ax_das.legend(fontsize=9.5, loc='best')
                    ax_das.axhline(0, color='k', linestyle='--', linewidth=1.0, alpha=0.5)
                    ax_das.grid(True, **GRID_KW)

                    # Dibujar Artefacto (Si existe)
                    if use_art and getattr(self, 'Artifact_Spectra', None) is not None:
                        art_mode = self._get_artifact_mode()
                        if art_mode == 'raman':
                            labels = [r"Raman ($\varphi_0$)"]
                        elif art_mode == 'xpm':
                            labels = [r"XPM ($\varphi_1$)", r"XPM ($\varphi_2$)"]
                        else:
                            labels = [r"Raman ($\varphi_0$)", r"XPM ($\varphi_1$)", r"XPM ($\varphi_2$)"]

                        art_colors = ['#6C757D', '#7B2CBF', '#8C564B']
                        for i, lbl in enumerate(labels):
                            ax_art.plot(wl, self.Artifact_Spectra[i], '-', color=art_colors[i], lw=2, label=lbl)

                        ax_art.set_xlabel("Wavelength (nm)")
                        ax_art.set_ylabel("Amplitude")
                        ax_art.set_title("Coherent Artifact Spectra")
                        ax_art.legend(fontsize=9.5)
                        ax_art.axhline(0, color='k', linestyle='--', linewidth=1.0, alpha=0.5)
                        ax_art.grid(True, **GRID_KW)
                        
                    # --- NUEVO: Histograma de Residuales ---
                    res_flat = self.fit_resid.flatten()
                    res_flat = res_flat[~np.isnan(res_flat)] # Por seguridad
                    
                    ax_hist.hist(res_flat, bins=100, density=True, color='#B0C4DE', edgecolor='#778899', alpha=0.7)
                    
                    # Dibujar curva de Gauss ideal
                    mu, std = np.mean(res_flat), np.std(res_flat)
                    x_pdf = np.linspace(np.min(res_flat), np.max(res_flat), 200)
                    y_pdf = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_pdf - mu) / std)**2)
                    ax_hist.plot(x_pdf, y_pdf, 'r-', lw=2, label=f'Ideal Gaussian\n$\mu$={mu:.2e}, $\sigma$={std:.2e}')
                    
                    ax_hist.set_title("Residuals Distribution (White Noise Check)", fontsize=11, fontweight='bold')
                    ax_hist.set_xlabel("Residual Value (ΔA)")
                    ax_hist.set_ylabel("Density")
                    ax_hist.legend(fontsize=9)
                    ax_hist.grid(True, **GRID_KW)

                    fig3.suptitle("Spectral Analysis & Residuals Diagnostics", fontsize=15, fontweight='bold', y=0.975)
                    if use_art:
                        fig3.tight_layout(rect=[0.03, 0.04, 0.97, 0.955], pad=2.5)
                    _add_footer(fig3)
                    pdf.savefig(fig3)
                    
                    
                    # ==========================================================
                    # PÁGINA — CINÉTICAS REPRESENTATIVAS
                    # ==========================================================
                    fig4 = Figure(figsize=(8.27, 11.69))
                    FigureCanvasAgg(fig4)

                    n_wls = len(wl)
                    indices_test = [int(n_wls * 0.2), int(n_wls * 0.4), int(n_wls * 0.6), int(n_wls * 0.8)]
                    axs_kin = [fig4.add_subplot(2, 2, i + 1) for i in range(4)]

                    for idx_ax, w_idx in enumerate(indices_test):
                        ax = axs_kin[idx_ax]
                        target_wl = wl[w_idx]
                        y_exp = self.data_c[w_idx, :]
                        y_fit = self.fit_fitres[w_idx, :]

                        ax.plot(td, y_exp, 'o', color='#1f77b4', markersize=3, alpha=0.55, label='Data')
                        ax.plot(td, y_fit, '-', color='#d62728', linewidth=1.8, label='Fit')

                        ax.set_title(f"Kinetics @ {target_wl:.1f} nm", fontsize=11)
                        ax.set_xlabel("Delay (ps)")
                        ax.set_ylabel(r"$\Delta A$")
                        ax.set_xscale('symlog', linthresh=1.0)
                        ax.grid(True, which="both", **GRID_KW)
                        ax.legend(fontsize=8.5)

                    fig4.suptitle("Representative Kinetic Traces (Data vs Fit)", fontsize=15, fontweight='bold', y=0.975)
                    fig4.tight_layout(rect=[0.04, 0.04, 0.97, 0.94], pad=2.0)
                    fig4.subplots_adjust(hspace=0.32, wspace=0.30)
                    _add_footer(fig4)
                    pdf.savefig(fig4)
                    
                    # ==========================================================
                    # PÁGINA 5 — SVD (SINGULAR VALUE DECOMPOSITION)
                    # ==========================================================
                    # Nos aseguramos de calcular el SVD si el usuario no hizo clic en el botón antes
                    if (not hasattr(self, 'svd_s') or self.svd_s is None or 
                        self.svd_U.shape[0] != len(wl) or self.svd_V.shape[0] != len(td)):
                        
                        U_svd, s_svd, Vh_svd = np.linalg.svd(self.data_c, full_matrices=False)
                        self.svd_U = U_svd
                        self.svd_s = s_svd
                        self.svd_V = Vh_svd.T
        

                    fig5 = Figure(figsize=(8.27, 11.69))
                    FigureCanvasAgg(fig5)
                    
                    gs_svd = gridspec.GridSpec(3, 1, figure=fig5, hspace=0.35, left=0.12, right=0.92, top=0.90, bottom=0.08)
                    ax_scree = fig5.add_subplot(gs_svd[0, 0])
                    ax_uspec = fig5.add_subplot(gs_svd[1, 0])
                    ax_vtkin = fig5.add_subplot(gs_svd[2, 0])

                    n_comp = min(len(self.svd_s), 10)
                    n_show = self.numExp 
                    
                    # 1. Scree Plot
                    ax_scree.semilogy(range(1, n_comp + 1), self.svd_s[:n_comp], 'o-', color='#d62728', markersize=6, lw=1.5)
                    ax_scree.axvline(n_show + 0.5, color='k', linestyle='--', alpha=0.5, label=f'Chosen Cutoff ({n_show} components)')
                    ax_scree.set_title("SVD: Singular Values (Scree Plot)", fontsize=11, fontweight='bold')
                    ax_scree.set_ylabel("Singular Value (Log Scale)")
                    ax_scree.set_xlabel("Component Index")
                    ax_scree.set_xticks(range(1, n_comp + 1))
                    ax_scree.grid(True, which="both", **GRID_KW)
                    ax_scree.legend()

                    # 2. Spectral Components
                    cmap_svd = plt.get_cmap("tab10")
                    for i in range(min(n_show + 1, len(self.svd_s))): 
                        c = cmap_svd(i % 10)
                        ls = '-' if i < n_show else ':'
                        lw = 2.0 if i < n_show else 1.2
                        lbl = f"Comp {i+1}" + (" (Noise?)" if i >= n_show else "")
                        ax_uspec.plot(wl, self.svd_U[:, i], color=c, linestyle=ls, linewidth=lw, label=lbl)
                    
                    ax_uspec.set_title(f"SVD: First {min(n_show + 1, len(self.svd_s))} Spectral Components (Left Singular Vectors)", fontsize=11, fontweight='bold')
                    ax_uspec.set_xlabel("Wavelength (nm)")
                    ax_uspec.set_ylabel("Amplitude")
                    ax_uspec.axhline(0, color='k', linestyle='-', alpha=0.3, lw=1)
                    ax_uspec.legend(fontsize=9, loc='best')
                    ax_uspec.grid(True, **GRID_KW)

                    # 3. Temporal Components
                    for i in range(min(n_show + 1, len(self.svd_s))):
                        c = cmap_svd(i % 10)
                        ls = '-' if i < n_show else ':'
                        lw = 2.0 if i < n_show else 1.2
                        lbl = f"Comp {i+1}" + (" (Noise?)" if i >= n_show else "")
                        ax_vtkin.plot(td, self.svd_V[:, i], color=c, linestyle=ls, linewidth=lw, label=lbl)
                    
                    ax_vtkin.set_title(f"SVD: First {min(n_show + 1, len(self.svd_s))} Temporal Components (Right Singular Vectors)", fontsize=11, fontweight='bold')
                    ax_vtkin.set_xlabel("Delay (ps)")
                    ax_vtkin.set_ylabel("Amplitude")
                    if hasattr(self, 'yscale') and self.yscale == 'symlog':
                        # Le ponemos linthresh 1.0 para que la visualización del SVD sea estándar
                        ax_vtkin.set_xscale('symlog', linthresh=1.0) 
                    ax_vtkin.axhline(0, color='k', linestyle='-', alpha=0.3, lw=1)
                    ax_vtkin.legend(fontsize=9, loc='best')
                    ax_vtkin.grid(True, which='both', **GRID_KW)

                    fig5.suptitle("Singular Value Decomposition (SVD) Analysis", fontsize=15, fontweight='bold', y=0.975)
                    _add_footer(fig5)
                    pdf.savefig(fig5)
                    

        except PermissionError:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(
                self, "File in use",
                f"Could not write the PDF file:\n{pdf_path}\n\n"
                f"It is most likely open in another program (a PDF viewer, cloud-sync app, etc.). "
                f"Close it and try again."
            )
            return
        except Exception as e:
            QApplication.restoreOverrideCursor()
            import traceback
            QMessageBox.critical(self, "PDF Error", f"Error generating PDF:\n{str(e)}\n\n{traceback.format_exc()}")
            return

        QApplication.restoreOverrideCursor()

        from PyQt5.QtGui import QDesktopServices
        from PyQt5.QtCore import QUrl

        reply = QMessageBox.question(self, "PDF Generated",
                                      f"Report successfully saved as:\n{pdf_path}\n\nDo you want to open it now?",
                                      QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if reply == QMessageBox.Yes:
            QDesktopServices.openUrl(QUrl.fromLocalFile(pdf_path))

    def plot_das_and_more(self):
        """Opens an external window to display DAS/SAS (Decay/Species Associated Spectra)."""
        if getattr(self, 'As', None) is None: return
        try:
            outdir = os.path.join(self.base_dir, "Plots") if getattr(self, 'base_dir', None) else os.path.abspath("Plots")
            os.makedirs(outdir, exist_ok=True)
            wl = getattr(self, '_wl_proc', self.WL)
            td = getattr(self, '_td_proc', self.TD)
            
            fitted_model = getattr(self, 'fitted_model_type', None) or self.model_type or "Parallel"
            self.fitted_model_type = fitted_model
            has_oscillation = ("Oscillation" in fitted_model) and (getattr(self, 'Bs', None) is not None)
            is_custom = fitted_model == "Custom GUI Model"
            
            fig_das = Figure(figsize=(14, 6) if has_oscillation else (8, 6))
            ax_das = fig_das.add_subplot(121) if has_oscillation else fig_das.add_subplot(111)
            ax_osc = fig_das.add_subplot(122) if has_oscillation else None
                        
            colors = ['b', 'r', 'g', 'orange', 'm', 'c']
            markers = ['o', 's', '^', 'D', 'v', 'p'] 
            
            if is_custom and getattr(self, 'current_custom_model', None) is not None:
                num_curves = min(len(self.current_custom_model.states), len(self.As))
            else:
                num_curves = min(self.numExp, len(self.As))
                if getattr(self, 'extracted_taus', None) is not None and len(self.extracted_taus) > 0:
                    num_curves = min(num_curves, len(self.extracted_taus))
            
            for n in range(num_curves):
                color = colors[n % len(colors)]
                marker = markers[n % len(markers)]
                
                # Asignación de leyenda correcta según el modelo
                if is_custom and getattr(self, 'current_custom_model', None) is not None and n < len(self.current_custom_model.states):
                    state_name = self.current_custom_model.states[n]
                    lbl = f"Species: {state_name}"
                    
                elif fitted_model == "Stretched Exponential":
                    tau_val = self.extracted_taus[n] if (getattr(self, 'extracted_taus', None) is not None and n < len(self.extracted_taus)) else 0.0
                    err_tau = self.extracted_errtaus[n] if (getattr(self, 'extracted_errtaus', None) is not None and n < len(self.extracted_errtaus) and not np.isnan(self.extracted_errtaus[n])) else 0.0
                    betas = getattr(self, 'extracted_betas', None)
                    beta_val = betas[n] if (betas is not None and n < len(betas)) else 1.0
                    lbl = rf"$\tau_{{{n+1}}}$ = {tau_val:.2f} $\pm$ {err_tau:.2f} ps ($\beta$ = {beta_val:.2f})"
                    
                else:
                    tau_val = self.extracted_taus[n] if (getattr(self, 'extracted_taus', None) is not None and n < len(self.extracted_taus)) else 0.0
                    err_tau = self.extracted_errtaus[n] if (getattr(self, 'extracted_errtaus', None) is not None and n < len(self.extracted_errtaus) and not np.isnan(self.extracted_errtaus[n])) else 0.0
                    lbl = rf"$\tau_{{{n+1}}}$ = {tau_val:.2f} $\pm$ {err_tau:.2f} ps"
    
                if getattr(self, 'errAs', None) is not None and n < len(self.errAs):
                    err_y = np.nan_to_num(self.errAs[n])
                    ax_das.errorbar(wl, self.As[n], yerr=err_y, label=lbl, color=color, fmt=f'-{marker}', markersize=5, capsize=4)
                else:
                    ax_das.plot(wl, self.As[n], f'-{marker}', label=lbl, color=color, markersize=5)
            
            ax_das.set_xlabel("Wavelength (nm)")
            if fitted_model in ["Sequential", "Custom GUI Model"]:
                ax_das.set_ylabel("SAS Amplitude (ΔA)")
                ax_das.set_title("Species Associated Spectra (SAS)")
            else:
                ax_das.set_ylabel("DAS Amplitude (ΔA)")
                ax_das.set_title("Decay Associated Spectra (DAS)")
            
            ax_das.legend(frameon=True)
            ax_das.axhline(0, color='k', linestyle='--', alpha=0.5)
            ax_das.grid(True, linestyle=':', alpha=0.4)
            
            if is_custom and getattr(self, 'current_custom_model', None) is not None:
                labels = getattr(self.current_custom_model, 'param_labels', [])
                taus = getattr(self, 'extracted_taus', [])
                errs = getattr(self, 'extracted_errtaus', [])
                params_list = []
                for idx_p, label in enumerate(labels):
                    v = taus[idx_p] if idx_p < len(taus) else 0.0
                    e = errs[idx_p] if idx_p < len(errs) else 0.0
                    params_list.append(f"{label} = {v:.2f} ± {e:.2f}")
                textstr = '\n'.join([r'$\mathbf{Kinetic\ Parameters:}$'] + params_list)
                props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
                ax_das.text(0.05, 0.95, textstr, transform=ax_das.transAxes, fontsize=10,
                            verticalalignment='top', bbox=props)
    
            if has_oscillation and ax_osc is not None:
                if hasattr(self, 'fit_x') and len(self.fit_x) >= 2 + self.numExp + 3:
                    alpha, omega, phi = self.fit_x[2 + self.numExp : 2 + self.numExp + 3]
                    title_osc = f"Oscillation Spectrum\nDamping α={alpha:.4f} | Freq ω={omega:.4f} | Phase φ={phi:.2f}"
                    ax_osc.plot(wl, self.Bs, color='black', linewidth=2, label='Oscillation Amplitude (B)')
                    if getattr(self, 'errBs', None) is not None:
                        ax_osc.fill_between(wl, self.Bs - self.errBs, self.Bs + self.errBs, color='black', alpha=0.1)
                    
                    ax_osc.set_xlabel("Wavelength (nm)")
                    ax_osc.set_ylabel("Oscillation Amplitude")
                    ax_osc.set_title(title_osc, color='darkblue')
                    ax_osc.axhline(0, color='k', linestyle='--', alpha=0.5)
                    ax_osc.grid(True, linestyle=':', alpha=0.4)
                    ax_osc.legend(frameon=True)
    
            fig_das.tight_layout()
            try: fig_das.savefig(os.path.join(outdir, "DAS_and_Oscillation.png" if has_oscillation else "DAS.png"), dpi=300)
            except: pass
    
            self.das_viewer = PlotViewerWindow(fig_das, title="DAS / SAS & Oscillation Spectra" if has_oscillation else "DAS / SAS Spectra", parent=self)
            self.das_viewer.show()
            
            fig_res = Figure()
            canvas_res = FigureCanvasAgg(fig_res)
            ax_res = fig_res.add_subplot(111)
            pcm = ax_res.pcolormesh(wl, td, self.fit_resid.T, cmap='jet', shading='auto')
            fig_res.colorbar(pcm, ax=ax_res, label='Residuals')
            ax_res.set_title("Residuals Map")
            ax_res.set_xlabel("Wavelength (nm)")
            ax_res.set_ylabel("Delay (ps)")
            if hasattr(self, 'yscale') and self.yscale == 'symlog':
                 ax_res.set_yscale('symlog', linthresh=2)
            fig_res.tight_layout()
            try: fig_res.savefig(os.path.join(outdir, "Residuals_Map.png"), dpi=300)
            except: pass
    
            self.trace_viewer = TraceExplorerWindow(self, outdir)
            self.trace_viewer.show()
        except Exception as e:
            from PyQt5.QtWidgets import QMessageBox
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Plot Error", f"An error occurred while generating plots:\n{e}")

