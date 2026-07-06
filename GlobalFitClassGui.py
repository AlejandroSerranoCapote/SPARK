import os
import re
import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QMessageBox, QProgressBar, QTableWidget, QTableWidgetItem,
    QHeaderView, QComboBox, QDoubleSpinBox, QSpinBox, QGroupBox, 
    QFormLayout, QWidget, QTabWidget, QApplication, QInputDialog,
    QCheckBox, QLineEdit, QListView,QFileDialog,QScrollArea
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QPixmap, QPainter, QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import fit
from matplotlib.widgets import Cursor
from matplotlib.ticker import FuncFormatter
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QMessageBox, QLineEdit, QCheckBox, QListWidget, 
    QAbstractItemView, QListWidgetItem, QColorDialog, QFormLayout
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PyQt5.QtGui import QIcon, QPixmap, QPainter, QColor
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView, 
    QComboBox, QTabWidget, QWidget, QInputDialog, QGraphicsView, 
    QGraphicsScene, QGraphicsRectItem, QGraphicsTextItem, QGraphicsLineItem, QGraphicsItem
)
from PyQt5.QtCore import Qt, QRectF, QLineF, QPointF
from PyQt5.QtGui import QPen, QBrush, QColor, QFont, QPolygonF
import matplotlib.gridspec as gridspec

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

        
class Surface3DWindow(QDialog):
    """
    Independent window to visualize the 3D plot without blocking the main application.
    """
    def __init__(self, xs, ys, zs, scale='linear', parent=None):
        """
        Initializes the 3D surface plotting window.

        Args:
            xs (numpy.ndarray): X-axis array (e.g., Wavelengths).
            ys (numpy.ndarray): Y-axis array (e.g., Time Delays).
            zs (numpy.ndarray): 2D Z-axis matrix (e.g., Transient Absorption data).
            scale (str, optional): The scale of the Y-axis ('linear' or 'symlog'). Defaults to 'linear'.
            parent (QWidget, optional): Parent widget. Defaults to None.
        """
        super().__init__(parent)
        self.setWindowTitle("3D Surface Preview")
        self.resize(800, 600)
        
        self.setStyleSheet(DARK_THEME_STYLE)
        self.setWindowModality(Qt.NonModal)

        layout = QVBoxLayout()
        
        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)
        
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setStyleSheet("QToolBar { background-color: transparent; border: none; }")
                
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.plot_data(xs, ys, zs, scale)

    def plot_data(self, xs, ys, zs, scale):
        """
        Renders the 3D surface plot onto the canvas.

        Args:
            xs (numpy.ndarray): X-axis array.
            ys (numpy.ndarray): Y-axis array.
            zs (numpy.ndarray): 2D Z-axis matrix.
            scale (str): The scale of the Y-axis ('linear' or 'symlog').
        """
        ax = self.fig.add_subplot(111, projection='3d')

        X, Y = np.meshgrid(xs, ys)
        Z = zs.T
        
        z_min = np.min(Z)
        
        Y_plot = Y
        y_axis_1d = ys
        
        if scale == 'symlog':
            linthresh = 1.0
            Y_plot = np.where(np.abs(Y) <= linthresh,
                              Y,
                              np.sign(Y) * (linthresh + np.log10(np.abs(Y) / linthresh)))
            y_axis_1d = Y_plot[:, 0] 
            
            ax.plot_surface(X, Y_plot, Z, cmap='jet', edgecolor='none', antialiased=True)
            ax.view_init(elev=30, azim=135)
            ax.contourf(X, Y_plot, Z, zdir='z', offset=z_min, cmap='jet', alpha=0.5)
            
            def symlog_ticks(val, pos):
                orig_val = val if np.abs(val) <= linthresh else np.sign(val) * linthresh * (10**(np.abs(val) - linthresh))
                if orig_val == 0: return "0"
                elif np.abs(orig_val) >= 10:
                    exponent = int(np.round(np.log10(np.abs(orig_val))))
                    sign = "-" if orig_val < 0 else ""
                    return f"{sign}$10^{{{exponent}}}$"
                else: return f"{orig_val:.0g}"
                    
            ax.yaxis.set_major_formatter(FuncFormatter(symlog_ticks))
            
        else:
            ax.plot_surface(X, Y, Z, cmap='jet', edgecolor='none', antialiased=True)
            ax.contourf(X, Y, Z, zdir='z', offset=z_min, cmap='jet', alpha=0.5)
            ax.view_init(elev=30, azim=-50, roll=-60)
            
        x_min = np.min(xs)
        y_max = np.max(Y_plot)
        
        x_min_pared = x_min - 20 
        y_max_pared = y_max + 0.5
        
        # 1. Spectra
        indices_tiempo = [len(ys)//10, len(ys)//4, len(ys)//2] 
        colores_espectros = ['red', 'orange', 'yellow'] 
        
        for i, idx_t in enumerate(indices_tiempo):
            espectro = Z[idx_t, :] 
            ax.plot(xs, espectro, zs=y_max_pared, zdir='y', color=colores_espectros[i%len(colores_espectros)], linewidth=1.5, alpha=0.8)

        # 2. Kinetics
        indices_onda = [len(xs)//4, len(xs)//2, 3*len(xs)//4]
        colores_cineticas = ['cyan', 'blue', 'magenta']
        
        for i, idx_w in enumerate(indices_onda):
            cinetica = Z[:, idx_w] 
            ax.plot(y_axis_1d, cinetica, zs=x_min_pared, zdir='x', color=colores_cineticas[i%len(colores_cineticas)], linewidth=1.5, alpha=0.8)

        ax.set_xlabel("Wavelength/Energy")
        ax.set_ylabel("Delay (ps)")
        ax.set_zlabel("Transient absorption")
        ax.set_zlim(bottom=z_min)
        
        # Clear panels (hide grid/panes for a cleaner look)
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.view_init(elev=25, azim=75)
            
        self.canvas.draw()

class PlotViewerWindow(QDialog):
    """Ventana independiente para visualizar gráficos SAS/DAS sin bloquear la app."""
    def __init__(self, fig, title="Plot", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(900, 600)
        self.setWindowModality(Qt.NonModal) # Esto hace que no bloquee la app principal
        self.setStyleSheet(DARK_THEME_STYLE)
        
        layout = QVBoxLayout(self)
        self.canvas = FigureCanvas(fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)


class TraceExplorerWindow(QDialog):
    """Explorador interactivo de cinéticas sin bloquear la interfaz."""
    def __init__(self, parent_panel, outdir):
        super().__init__(parent_panel)
        self.p = parent_panel
        self.outdir = outdir
        self.setWindowTitle("Interactive Trace Viewer")
        self.resize(1000, 550)
        self.setWindowModality(Qt.NonModal) # Clave para que flote libremente
        self.setStyleSheet(DARK_THEME_STYLE + BUTTON_STYLE)

        layout = QVBoxLayout(self)

        # Controles superiores
        ctrl_layout = QHBoxLayout()
        ctrl_layout.addWidget(QLabel("Wavelength (nm):"))
        
        self.wl_array = getattr(self.p, '_wl_proc', self.p.WL)
        self.td_array = getattr(self.p, '_td_proc', self.p.TD)
        
        # Usamos un ComboBox para moverse exactamente por los índices medidos
        self.combo_wl = QComboBox()
        self.combo_wl.addItems([f"{w:.1f}" for w in self.wl_array])
        self.combo_wl.setCurrentIndex(len(self.wl_array)//2)
        self.combo_wl.currentIndexChanged.connect(self.update_plot)
        ctrl_layout.addWidget(self.combo_wl)

        self.btn_save = QPushButton("Save Trace Data")
        self.btn_save.clicked.connect(self.save_trace)
        ctrl_layout.addWidget(self.btn_save)
        

        self.btn_paper_plot = QPushButton("Launch Paper Plotter")
        self.btn_paper_plot.clicked.connect(self.open_paper_plotter)
        self.btn_paper_plot.setStyleSheet("background-color: #3C5488; color: white;")
        ctrl_layout.addWidget(self.btn_paper_plot)
        
        layout.addLayout(ctrl_layout)
        # Lienzo (Canvas) de Matplotlib
        self.fig = plt.Figure(figsize=(12, 5))
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self.update_plot()

    def update_plot(self):
        self.fig.clear()
        idx = self.combo_wl.currentIndex()
        real_wl = self.wl_array[idx]
        
        y_exp = self.p.data_c[idx, :]
        td = self.td_array
        
        td_lin = np.linspace(td.min(), 1.0, 1000)
        td_log = np.geomspace(1.0, max(1.1, td.max()), 1000)
        td_smooth = np.unique(np.concatenate((td_lin, td_log)))
        
        # --- CÁLCULO DIRECTO POR VARPRO (SIN CHIRP) ---
        if hasattr(self.p, 'S_T_full'):
            use_art = getattr(self.p, 'chk_artifact', None) and self.p.chk_artifact.isChecked()
            
            if self.p.model_type == "Sequential":
                C_smooth = fit.get_concentration_matrix_sequential(self.p.fit_x, td_smooth, self.p.numExp, use_art)
            elif self.p.model_type == 'Damped Oscillation':
                C_smooth = fit.get_concentration_matrix_oscillation(self.p.fit_x, td_smooth, self.p.numExp, use_art)
            elif self.p.model_type == "Custom GUI Model":
                model = self.p.current_custom_model
                w, t0 = self.p.fit_x[0], self.p.fit_x[1]
                num_params_cineticos = len(model.param_labels)
                x_nl_params = self.p.fit_x[2:2+num_params_cineticos]
                C_smooth = model.get_concentration_matrix(x_nl_params, td_smooth, w, t0, use_art=use_art)
            else:
                C_smooth = fit.get_concentration_matrix_global(self.p.fit_x, td_smooth, self.p.numExp, use_art)
                
            y_fit_smooth = C_smooth @ self.p.S_T_full[:, idx]
        else:
            y_fit_smooth = np.zeros_like(td_smooth)
            print("Warning: S_T_full matrix not found.")

        ax1 = self.fig.add_subplot(121)
        ax2 = self.fig.add_subplot(122, sharey=ax1)
        self.fig.suptitle(f"Fit at {real_wl:.1f} nm", fontsize=14)

        # Plot Lineal
        ax1.plot(td, y_exp, 'bo', markersize=4, alpha=0.6, label='Data')
        ax1.plot(td_smooth, y_fit_smooth, 'r-', linewidth=2, label='Fit')
        ax1.set_xlabel("Time / ps")
        ax1.set_ylabel("ΔA")
        ax1.legend(frameon=True)
        ax1.grid(True, alpha=0.3)

        # Plot Semi-Log
        mask_pos_exp = td > 0
        mask_pos_smooth = td_smooth > 0
        if np.any(mask_pos_exp):
            ax2.plot(td[mask_pos_exp], y_exp[mask_pos_exp], 'bo', markersize=4, alpha=0.6)
            ax2.plot(td_smooth[mask_pos_smooth], y_fit_smooth[mask_pos_smooth], 'r-', linewidth=2)
            ax2.set_xscale('log')
            ax2.set_xlabel("Time / ps (log scale)")
            ax2.grid(True, which="both", ls="-", alpha=0.3)

        self.fig.tight_layout()
        self.canvas.draw()
    
    def save_trace(self):
        idx = self.combo_wl.currentIndex()
        real_wl = self.wl_array[idx]
        
        # 1. Guardar la imagen PNG (Lógica original)
        img_name = f"Trace_{real_wl:.1f}nm.png"
        self.fig.savefig(os.path.join(self.outdir, img_name), dpi=300)
        
        # 2. Extraer tiempos y datos experimentales
        td = self.td_array
        y_exp = self.p.data_c[idx, :]
        
        # 3. Recalcular el Fit para los puntos experimentales exactos (td)
        if hasattr(self.p, 'S_T_full'):
            use_art = getattr(self.p, 'chk_artifact', None) and self.p.chk_artifact.isChecked()
            
            if self.p.model_type == "Sequential":
                C_exp = fit.get_concentration_matrix_sequential(self.p.fit_x, td, self.p.numExp, use_art)
            elif self.p.model_type == 'Damped Oscillation':
                C_exp = fit.get_concentration_matrix_oscillation(self.p.fit_x, td, self.p.numExp, use_art)
            elif self.p.model_type == "Custom GUI Model":
                model = self.p.current_custom_model
                w, t0 = self.p.fit_x[0], self.p.fit_x[1]
                num_params_cineticos = len(model.param_labels)
                x_nl_params = self.p.fit_x[2:2+num_params_cineticos]
                C_exp = model.get_concentration_matrix(x_nl_params, td, w, t0, use_art=use_art)
            else:
                C_exp = fit.get_concentration_matrix_global(self.p.fit_x, td, self.p.numExp, use_art)
                
            y_fit_exp = C_exp @ self.p.S_T_full[:, idx]
        else:
            y_fit_exp = np.zeros_like(td)
            
        # 4. Empaquetar las 3 columnas y exportar a .txt
        txt_name = f"Trace_{real_wl:.1f}nm.txt"
        txt_path = os.path.join(self.outdir, txt_name)
        
        # Unimos las columnas en una matriz (Time_Delay, Experimental, Fit)
        matriz_guardar = np.column_stack((td, y_exp, y_fit_exp))
        
        # Añadimos una cabecera limpia aclarando qué es cada columna
        cabecera = f"Wavelength: {real_wl:.1f} nm\nTime_Delay(ps)\tExperimental_Data\tFit_Data"
        
        np.savetxt(txt_path, matriz_guardar, fmt='%.6e', delimiter='\t', header=cabecera)
        
        # Mensaje de confirmación actualizado
        QMessageBox.information(self, "Saved", f"Trace PNG and TXT data saved successfully in:\n{self.outdir}")
        
    def open_paper_plotter(self):
        """Abre la ventana interactiva de Drag & Drop para gráficos de papel."""
        self.plotter_win = PaperPlotterWindow(self)
        self.plotter_win.show()            

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QCheckBox, QPushButton, QColorDialog, QScrollArea, QWidget, QGridLayout

class CompareSetupDialog(QDialog):
    """
    Dialog to setup parameters for comparing kinetics across multiple datasets.
    Allows user to pick target wavelength, normalization, titles, labels, and colors.
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
        """Opens a color picker and updates the button's stored color."""
        color = QColorDialog.getColor()
        if color.isValid():
            hex_color = color.name()
            btn.setStyleSheet(f"background-color: {hex_color}; border: 1px solid gray; width: 25px; height: 15px;")
            btn.setProperty("color_val", hex_color)

    def get_data(self):
        """Returns the tuple expected by the compare_kinetics function."""
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
    
class PaperPlotterWindow(QDialog):
    """
    Advanced Drag & Drop Publication-Quality Plotter for Kinetics Traces.
    Works 100% autonomously. Allows users to customize dimensions, 
    palettes, and crop the Y-axis (ΔA) with high precision.
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
        layout = QVBoxLayout(self)
        top_layout = QHBoxLayout()
        
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
        top_layout.addWidget(self.drop_label, 3)
        
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
        
        self.btn_export_fig = QPushButton("Export Figure (600 DPI)")
        self.btn_export_fig.clicked.connect(self.export_figure)
        self.btn_export_fig.setStyleSheet("background-color: #4A8C4A; color: white; font-weight: bold;")
        ctrl_form.addRow(self.btn_export_fig)
        
        top_layout.addWidget(ctrl_group, 2)
        layout.addLayout(top_layout)
        
        self.fig = Figure(figsize=(self.spin_width.value(), self.spin_height.value()), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        self.ax = self.fig.add_subplot(111)
        self.setup_paper_style()
        
    def toggle_y_inputs(self):
        """Activa o desactiva las cajas de Crop de la señal según el modo elegido."""
        auto_mode = self.chk_auto_y.isChecked()
        self.spin_ymin.setEnabled(not auto_mode)
        self.spin_ymax.setEnabled(not auto_mode)
        self.replotted()

    def update_fig_size(self):
        w = self.spin_width.value()
        h = self.spin_height.value()
        self.fig.set_size_inches(w, h)
        self.canvas.draw_idle() 
        
    def setup_paper_style(self):
        self.ax.clear()
        self.ax.tick_params(direction='in', top=True, right=True, labelsize=11, width=1.2, length=6)
        for spine in self.ax.spines.values():
            spine.set_linewidth(1.2)
        self.ax.set_xlabel("Time Delay / ps", fontsize=13, fontname="Arial", fontweight='bold')
        self.ax.set_ylabel("ΔA (a.u.)", fontsize=13, fontname="Arial", fontweight='bold')
        self.ax.grid(False)
        
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            
    def dropEvent(self, event):
        files_added = 0
        for url in event.mimeData().urls():
            file_path = str(url.toLocalFile())
            if file_path.lower().endswith('.txt'):
                if self.parse_trace_file(file_path):
                    files_added += 1
        if files_added > 0:
            self.file_data.sort(key=lambda x: x['wl'])
            self.replotted()
            
    def parse_trace_file(self, path):
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
                
            td = raw_data[:, 0]
            y_exp = raw_data[:, 1]
            y_fit = raw_data[:, 2] if raw_data.shape[1] >= 3 else None
            
            self.file_data.append({
                'wl': wl_rounded,
                'td': td,
                'exp': y_exp,
                'fit': y_fit,
                'filename': filename
            })
            return True
        except Exception as e:
            print(f"Error parseando: {e}")
            return False
            
    def replotted(self):
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
            td = data['td']
            y_exp = data['exp']
            y_fit = data['fit']
            wl = data['wl']
            
            label_text = f"{wl} nm" if wl > 0 else data['filename']
            
            max_td_found = max(max_td_found, np.max(td))
            min_td_found = min(min_td_found, np.min(td))
            
            if norm_indep:
                max_val = max(np.max(np.abs(y_exp)), 1e-10)
                y_exp = y_exp / max_val
                if y_fit is not None:
                    y_fit = y_fit / max_val
            
            self.ax.plot(td, y_exp, 'o', color=color, markersize=4, alpha=0.4, 
                         markeredgewidth=1.0, label=label_text)
            
            if y_fit is not None:
                self.ax.plot(td, y_fit, '-', color=color, linewidth=2.0)
            
        if is_symlog:
            self.ax.set_xscale('symlog', linthresh=self.spin_thresh.value())
            
        if hide_negatives:
            self.ax.set_xlim(0.0, max_td_found)
        else:
            self.ax.set_xlim(min_td_found, max_td_found)
            
        # --- APLICACIÓN DEL CROP PERSONALIZADO DEL EJE Y ---
        if not self.chk_auto_y.isChecked():
            self.ax.set_ylim(self.spin_ymin.value(), self.spin_ymax.value())
        # ---------------------------------------------------
            
        self.ax.legend(frameon=True, framealpha=0.0, edgecolor='none', fontsize=10, loc='best')
        self.fig.tight_layout()
        self.canvas.draw_idle()
        
    def clear_data(self):
        self.file_data = []
        self.replotted()
        
    def export_figure(self):
        if not self.file_data:
            return
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Publication Figure", "", 
            "PDF Vector Graphic (*.pdf);;PNG High-Resolution Image (*.png);;TIFF Image (*.tiff)"
        )
        if save_path:
            self.update_fig_size()
            self.fig.savefig(save_path, dpi=600, bbox_inches='tight')
            QMessageBox.information(self, "Export Successful", f"Gráfico exportado a alta resolución con las dimensiones seleccionadas.")            

from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QComboBox, QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel


import math
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox, QLineEdit, QMessageBox
from PyQt5.QtCore import Qt, QRectF, QLineF, QPointF
from PyQt5.QtGui import QPen, QBrush, QColor, QFont, QPolygonF, QPainter

class StateNode(QGraphicsRectItem):
    """Caja gráfica que representa un estado físico (S1*, 3CT, etc.)"""
    def __init__(self, name, x, y):
        super().__init__(-40, -20, 80, 40) # Rectángulo centrado
        self.name = name
        self.setPos(x, y)
        self.setBrush(QBrush(QColor("#6CB66C"))) # Color base
        self.setPen(QPen(Qt.black, 1.5))
        
        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setFlag(QGraphicsItem.ItemIsSelectable)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges)
        
        # Texto interior
        self.text = QGraphicsTextItem(name, self)
        self.text.setFont(QFont("Arial", 10, QFont.Bold))
        self.text.setDefaultTextColor(Qt.white)
        # Centrar texto
        br = self.text.boundingRect()
        self.text.setPos(-br.width()/2, -br.height()/2)
        
        self.edges = [] 

    def add_edge(self, edge):
        self.edges.append(edge)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            for edge in self.edges:
                edge.update_position()
        return super().itemChange(change, value)


class TransitionEdge(QGraphicsLineItem):
    """Flecha gráfica inteligente que conecta dos estados con dirección."""
    def __init__(self, source_node, target_node, param_type, label):
        super().__init__()
        self.source_node = source_node
        self.target_node = target_node
        self.param_type = param_type
        self.label_name = label
        
        # Estilo de la línea
        self.setPen(QPen(QColor("#2B2B2B"), 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        self.setZValue(-1) # La flecha pasa por debajo de las cajas
        self.setFlag(QGraphicsItem.ItemIsSelectable)
        
        # Etiqueta de texto de la flecha
        self.text = QGraphicsTextItem(f"{label} ({param_type})", self)
        self.text.setFont(QFont("Arial", 9, QFont.Bold))
        self.text.setDefaultTextColor(QColor("#0078D7"))
        
        self.source_node.add_edge(self)
        self.target_node.add_edge(self)
        self.update_position()

    def update_position(self):
        line = QLineF(self.source_node.pos(), self.target_node.pos())
        self.setLine(line)
        
        # Posicionar el texto encima del centro de la línea
        center = line.center()
        self.text.setPos(center.x() - self.text.boundingRect().width()/2, 
                         center.y() - self.text.boundingRect().height() - 10)

    def paint(self, painter, option, widget=None):
        """Sobreescribimos el dibujado para añadir una punta de flecha en el centro."""
        # 1. Dibujar la línea normal
        painter.setPen(self.pen())
        painter.drawLine(self.line())
        
        # 2. Dibujar la punta de flecha en el centro
        line = self.line()
        if line.length() > 0:
            center = line.center()
            # Ángulo de la línea (en Qt, el eje Y crece hacia abajo)
            angle = math.atan2(line.dy(), line.dx())
            arrow_size = 12
            
            # Calcular los vértices de la flecha usando trigonometría
            p1 = center - QPointF(math.cos(angle - math.pi / 6) * arrow_size,
                                  math.sin(angle - math.pi / 6) * arrow_size)
            p2 = center - QPointF(math.cos(angle + math.pi / 6) * arrow_size,
                                  math.sin(angle + math.pi / 6) * arrow_size)
            
            arrow_head = QPolygonF([center, p1, p2])
            painter.setBrush(QBrush(self.pen().color()))
            painter.drawPolygon(arrow_head)


class KineticCanvas(QGraphicsView):
    """Lienzo interactivo (Diagrama de Jablonski / Grotrian)"""
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.scene.setSceneRect(0, 0, 800, 600)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)
        
        self.nodes = {} 
        self.linking_mode = False
        self.link_source = None
        
    def add_state(self, name, x=None, y=None):
        if name in self.nodes:
            QMessageBox.warning(self, "Warning", f"State '{name}' already exists.")
            return
        offset = len(self.nodes) * 60
        pos_x = x if x is not None else 150 + offset
        pos_y = y if y is not None else 100 + offset
        
        node = StateNode(name, pos_x, pos_y)
        self.scene.addItem(node)
        self.nodes[name] = node

    def create_connection(self, source, target):
        dialog = QDialog()
        dialog.setWindowTitle("New Pathway / Transition")
        layout = QVBoxLayout(dialog)
        
        layout.addWidget(QLabel(f"From: <b>{source.name}</b> ➔ To: <b>{target.name}</b>"))
        
        combo_type = QComboBox()
        combo_type.addItems(["tau", "gamma"])
        layout.addWidget(QLabel("Parameter type:"))
        layout.addWidget(combo_type)
        
        input_label = QLineEdit()
        input_label.setPlaceholderText("E.g.: tau1, tau2, gamma...")
        layout.addWidget(QLabel("Variable name (label):"))
        layout.addWidget(input_label)
        
        btn_ok = QPushButton("Connect")
        btn_ok.clicked.connect(dialog.accept)
        layout.addWidget(btn_ok)
        
        if dialog.exec_() == QDialog.Accepted and input_label.text().strip():
            edge = TransitionEdge(source, target, combo_type.currentText(), input_label.text().strip())
            self.scene.addItem(edge)

    def mousePressEvent(self, event):
        item = self.itemAt(event.pos())
        
        # FIX: Redirigir el clic a la caja padre si se hace en el texto
        if item and hasattr(item, 'parentItem') and isinstance(item.parentItem(), StateNode):
            item = item.parentItem()
            
        if self.linking_mode:
            if isinstance(item, StateNode):
                if self.link_source is None:
                    self.link_source = item
                    item.setBrush(QBrush(QColor("#0078D7"))) 
                else:
                    if self.link_source != item:
                        self.create_connection(self.link_source, item)
                    self.link_source.setBrush(QBrush(QColor("#6CB66C")))
                    self.link_source = None
                    self.linking_mode = False
            elif self.link_source is not None:
                self.link_source.setBrush(QBrush(QColor("#6CB66C")))
                self.link_source = None
                self.linking_mode = False
                
        super().mousePressEvent(event)

    def delete_selected(self):
        """Elimina de forma segura los nodos o flechas seleccionadas."""
        for item in self.scene.selectedItems():
            if hasattr(item, 'parentItem') and item.parentItem() is not None:
                item = item.parentItem()
                
            if isinstance(item, StateNode):
                for edge in list(item.edges):
                    if edge in self.scene.items():
                        self.scene.removeItem(edge)
                    if edge in edge.source_node.edges:
                        edge.source_node.edges.remove(edge)
                    if edge in edge.target_node.edges:
                        edge.target_node.edges.remove(edge)
                if item.name in self.nodes:
                    del self.nodes[item.name]
                if item in self.scene.items():
                    self.scene.removeItem(item)
                    
            elif isinstance(item, TransitionEdge):
                if item in item.source_node.edges:
                    item.source_node.edges.remove(item)
                if item in item.target_node.edges:
                    item.target_node.edges.remove(item)
                if item in self.scene.items():
                    self.scene.removeItem(item)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete or event.key() == Qt.Key_Backspace:
            self.delete_selected()
        super().keyPressEvent(event)
        
class ModelBuilderDialog(QDialog):
    """
    Ventana interactiva Dual: Modo Tabla y Modo Visual (Canvas)
    para construir modelos cinéticos in-situ.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Advanced Kinetic Model Builder")
        self.resize(800, 600)
        
        if parent and hasattr(parent, 'styleSheet'):
            self.setStyleSheet(parent.styleSheet())
            
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout(self)
        
        # --- CREAR LAS PESTAÑAS ---
        self.tabs = QTabWidget()
        self.tab_table = QWidget()
        self.tab_canvas = QWidget()
        
        self.tabs.addTab(self.tab_canvas, "Visual Mode (Grotrian diagrams)")
        self.tabs.addTab(self.tab_table, "Classic Mode") # Typo corregido aquí
        
        layout.addWidget(self.tabs)
        
        # ==========================================
        # CONFIGURAR PESTAÑA 1: MODO VISUAL (CANVAS)
        # ==========================================
        canvas_layout = QVBoxLayout(self.tab_canvas)
        
        self.canvas = KineticCanvas()
        
        toolbar_layout = QHBoxLayout()
        
        self.btn_add_node = QPushButton("Add state")
        self.btn_add_node.clicked.connect(self.prompt_add_state)
        
        self.btn_link_nodes = QPushButton("Connect states")
        self.btn_link_nodes.clicked.connect(self.activate_linking_mode)
        self.btn_link_nodes.setStyleSheet("background-color: #0078D7; color: white;")
        
        self.btn_reset_canvas = QPushButton("Reboot scheme")
        self.btn_reset_canvas.setStyleSheet("background-color: #DC3545; color: white; font-weight: bold;")
        self.btn_reset_canvas.clicked.connect(self.reset_all)
        
        self.btn_delete_item = QPushButton("Erase selected")
        self.btn_delete_item.setStyleSheet("background-color: #E67E22; color: white; font-weight: bold;")
        self.btn_delete_item.clicked.connect(self.canvas.delete_selected)
        
        toolbar_layout.addWidget(self.btn_add_node)
        toolbar_layout.addWidget(self.btn_link_nodes)
        toolbar_layout.addWidget(self.btn_reset_canvas)
        toolbar_layout.addWidget(self.btn_delete_item)
        
        canvas_layout.addLayout(toolbar_layout)
        canvas_layout.addWidget(self.canvas)
        
        lbl_inst = QLabel("<i>Drag the boxes to organize your model. Select an item and press 'Delete' or the orange button to remove it.</i>")
        canvas_layout.addWidget(lbl_inst)
        
        # ==========================================
        # CONFIGURAR PESTAÑA 2: MODO TABLA CLÁSICA
        # ==========================================
        table_layout = QVBoxLayout(self.tab_table)
        
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Source State", "Target State", "Parameter Type", "Parameter Name"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table_layout.addWidget(self.table)
        
        t_btns = QHBoxLayout()
        btn_add_row = QPushButton("Add Row")
        btn_add_row.clicked.connect(lambda: self.add_table_row("", "", "tau", ""))
        btn_rem_row = QPushButton("Remove Row")
        btn_rem_row.clicked.connect(self.remove_table_row)
        t_btns.addWidget(btn_add_row)
        t_btns.addWidget(btn_rem_row)
        table_layout.addLayout(t_btns)
        
        # ==========================================
        # BOTONES GLOBALES DE ACEPTAR / CANCELAR
        # ==========================================
        actions_layout = QHBoxLayout()
        self.btn_compile = QPushButton("Load & compile kinetic model")
        self.btn_compile.setObjectName("BtnGreen") 
        self.btn_compile.clicked.connect(self.accept)
        
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        
        actions_layout.addWidget(self.btn_compile)
        actions_layout.addWidget(self.btn_cancel)
        layout.addLayout(actions_layout)

        self.load_example_table()

    # --- Funciones del Modo Canvas ---
    def reset_all(self):
        from PyQt5.QtWidgets import QMessageBox
        
        respuesta = QMessageBox.question(
            self, "Confirm Reset", 
            "Are you sure you want to delete all states and start from scratch?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if respuesta == QMessageBox.Yes:
            self.canvas.scene.clear()
            self.canvas.nodes.clear()
            self.canvas.linking_mode = False
            self.canvas.link_source = None
            self.table.setRowCount(0)
            
    def prompt_add_state(self):
        name, ok = QInputDialog.getText(self, "Add State", "State name (e.g., S1, 3CT, S0):")
        if ok and name.strip():
            self.canvas.add_state(name.strip())
            
    def activate_linking_mode(self):
        self.canvas.linking_mode = True
        QMessageBox.information(self, "Connection Mode", "Click on the SOURCE state and then on the TARGET state.")

    # --- Funciones del Modo Tabla ---
    def add_table_row(self, src, tgt, p_type, label):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(src))
        self.table.setItem(row, 1, QTableWidgetItem(tgt))
        
        combo = QComboBox()
        combo.addItems(["tau", "gamma"])
        if p_type == "gamma": combo.setCurrentIndex(1)
        self.table.setCellWidget(row, 2, combo)
        self.table.setItem(row, 3, QTableWidgetItem(label))
        
    def remove_table_row(self):
        curr_row = self.table.currentRow()
        if curr_row >= 0: self.table.removeRow(curr_row)
            
    def load_example_table(self):
        self.add_table_row("S1*", "S1", "tau", "tau1")
        self.add_table_row("S1", "3CT*", "tau", "tau2")
        self.add_table_row("S1", "3CT*", "gamma", "gamma")

    # --- MOTOR DE COMPILACIÓN INTELIGENTE ---
    def get_compiled_model(self):
        try:
            from fit import KMatrixModel
        except ImportError as e:
            QMessageBox.critical(self, "Critical Error", f"Could not import KMatrixModel from fit.py.\nDetail: {e}")
            return None

        model = KMatrixModel("Custom GUI Model")
        
        if self.tabs.currentIndex() == 0: 
            edges_found = False
            for item in self.canvas.scene.items():
                if isinstance(item, TransitionEdge):
                    edges_found = True
                    src = item.source_node.name
                    tgt = item.target_node.name
                    p_type = item.param_type
                    label = item.label_name
                    model.add_transition(src, tgt, param_type=p_type, label=label)
            
            if not edges_found:
                QMessageBox.warning(self, "Empty Canvas", "You haven't drawn any connections in the visual mode.")
                return None
                
        else:
            if self.table.rowCount() == 0:
                return None
            for row in range(self.table.rowCount()):
                src = self.table.item(row, 0).text().strip()
                tgt = self.table.item(row, 1).text().strip()
                combo = self.table.cellWidget(row, 2)
                p_type = "tau" if combo.currentIndex() == 0 else "gamma"
                label = self.table.item(row, 3).text().strip()
                
                if src and tgt and label:
                    model.add_transition(src, tgt, param_type=p_type, label=label)
                    
        return model
    
class ParameterIdentifiabilityDialog(QDialog):
    """
    Diálogo para analizar la identificabilidad de un parámetro cinético 
    mediante perfiles de verosimilitud (profile likelihood), en vez de 
    confiar únicamente en el error basado en la covarianza.
    """
    def __init__(self, panel, parent=None):
        super().__init__(parent)
        self.panel = panel  # Referencia al GlobalFitPanel (motor de cálculo)
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
        num_kin = self.panel._get_num_kinetic_params()
        self.combo_param.clear()
        for i in range(num_kin):
            label = self.panel._get_kinetic_param_label(i)
            self.combo_param.addItem(f"[{i}] {label}", userData=i)

    def cancel_analysis(self):
        self.panel._abort_fit = True
        self.lbl_result.setText("Cancelling...")

    def run_analysis(self):
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
        self.setWindowTitle("Global Fit Analysis")
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
        self.data_c_list = []      # Lista para guardar los datos procesados
        self.data_raw_list = []    # Lista para guardar las matrices crudas
        self.TD_list = []          # Lista para los ejes de tiempo
        self.WL_list = []          # Lista para los ejes de longitud de onda
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
        
        # Rest of the fit variables
        self.fit_resid = None
        self.fit_fitres = None
        self.ci = None
        self.errAs = None
        self.errtaus = None
        self.ini = None
        self.limi = None
        self.lims = None

        # --- 3. MAIN LAYOUT DESIGN ---
        main_layout = QHBoxLayout() 
        
        # --- A. Left Panel (Workflow Tabs) ---
        self.sidebar_tabs = QTabWidget()
        self.sidebar_tabs.setFixedWidth(370) # Un pelín más ancho para que respiren los botones

        self.tab_data = QWidget()
        self.tab_model = QWidget()

        self.sidebar_tabs.addTab(self.tab_data, "1. Data")
        self.sidebar_tabs.addTab(self.tab_model, "2. Fit")

        self._init_sidebar_ui() 
        
        main_layout.addWidget(self.sidebar_tabs)
    
        # --- B. Right Panel (Plots) ---
        self.right_area = QWidget()
        self.right_layout = QVBoxLayout(self.right_area)
        
        self._init_plots_ui() 
        
        main_layout.addWidget(self.right_area)
    
        self.setLayout(main_layout)
    
        # --- IMPORTANT: INITIALIZE PLOTTING VARIABLES ---
        self.pcm_exp = None
        self.cbar_exp = None
        self.pcm_fit = None
        self.cbar_fit = None
        self.pcm_resid = None
        self.cbar_resid = None

    def _init_sidebar_ui(self):
        """Sets up all the widgets of the left panel divided into workflow tabs."""
        layout_data = QVBoxLayout(self.tab_data)
        layout_model = QVBoxLayout(self.tab_model)
        
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
        self.btn_load = QPushButton("Load .npy")
        self.btn_load.clicked.connect(self.load_data) 
        h_btns.addWidget(self.btn_load)
    
        self.btn_parent = QPushButton("Use Parent Data")
        self.btn_parent.clicked.connect(self.use_parent_data) 
        h_btns.addWidget(self.btn_parent)
        v_load.addLayout(h_btns)
        
        self.btn_compare = QPushButton("Compare Kinetics")
        self.btn_compare.clicked.connect(self.compare_kinetics)
        v_load.addWidget(self.btn_compare)
        
        v_load.addWidget(QLabel("<b>Visualizing Dataset:</b>"))
        h_dataset = QHBoxLayout()
        self.combo_active_dataset = QComboBox()
        self.combo_active_dataset.setView(QListView())
        self.combo_active_dataset.currentIndexChanged.connect(self._on_active_dataset_changed)
        h_dataset.addWidget(self.combo_active_dataset, stretch=4)
        
        self.btn_remove_dataset = QPushButton("Del")
        self.btn_remove_dataset.setToolTip("Remove current dataset")
        self.btn_remove_dataset.setStyleSheet("background-color: #DC3545; color: white; font-weight: bold; max-width: 30px;")
        self.btn_remove_dataset.clicked.connect(self.remove_active_dataset)
        h_dataset.addWidget(self.btn_remove_dataset, stretch=1)
        v_load.addLayout(h_dataset)
        
        gb_load.setLayout(v_load)
        layout_data.addWidget(gb_load)

        # --- Group 2: Pre-processing ---
        gb_prep = QGroupBox("Pre-processing")
        form_prep = QFormLayout()

        self.spin_bl = QSpinBox()
        self.spin_bl.setRange(0, 500)
        self.spin_bl.setValue(0)
        self.spin_bl.valueChanged.connect(lambda val: self._preview_data_processing())
        form_prep.addRow("Baseline Pts:", self.spin_bl)

        self.spin_wl_min = QDoubleSpinBox(); self.spin_wl_min.setRange(0, 10000); self.spin_wl_min.setDecimals(2)
        self.spin_wl_max = QDoubleSpinBox(); self.spin_wl_max.setRange(0, 10000); self.spin_wl_max.setDecimals(2)
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
        h_time = QHBoxLayout()
        h_time.addWidget(self.spin_t_min)
        h_time.addWidget(QLabel("to"))
        h_time.addWidget(self.spin_t_max)
        form_prep.addRow("Time Range (ps):", h_time)
        
        self.spin_bin = QSpinBox()
        self.spin_bin.setRange(1, 50)
        self.spin_bin.setValue(1)
        form_prep.addRow("Binning:", self.spin_bin)
        
        self.chk_zero_neg = QCheckBox("Set t < 0 to zero (background)")
        self.chk_zero_neg.setChecked(False) 
        form_prep.addRow(self.chk_zero_neg)
        
        self.chk_norm_data = QCheckBox("Normalize Data Matrix (Max |ΔA| = 1)")
        self.chk_norm_data.setChecked(False)
        self.chk_norm_data.stateChanged.connect(lambda state: self._preview_data_processing())
        form_prep.addRow(self.chk_norm_data)
        
        self.btn_preview = QPushButton("Apply and Preview")
        self.btn_preview.setStyleSheet("background-color: #28A745; border: 1px solid #218838; color: white;") 
        self.btn_preview.clicked.connect(self._preview_data_processing) 
        form_prep.addRow(self.btn_preview)

        gb_prep.setLayout(form_prep)
        layout_data.addWidget(gb_prep)

        # --- Group 3: Visualization  ---
        gb_vis = QGroupBox("Visualization")
        form_vis = QFormLayout()
        self.btn_plot_3d = QPushButton("3D Map")
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

        # --- Group 3b: Export Spectrum at Delay ---
        gb_export = QGroupBox("Export Spectrum at Delay")
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
        layout_data.addWidget(gb_export)
        
        layout_data.addStretch() # Empuja todo hacia arriba en la primera pestaña

        # ==========================================
        # TAB 2: MODEL & FITTING
        # ==========================================
        
        # --- Group 4: Model Settings ---
        gb_model = QGroupBox("Model Settings")
        form_model = QFormLayout()
        
        self.btn_svd = QPushButton("Run SVD Analysis")
        self.btn_svd.clicked.connect(self.run_svd)
        form_model.addRow(self.btn_svd)
        
        self.spin_numExp = QSpinBox()
        self.spin_numExp.setRange(1, 6)
        self.spin_numExp.setValue(2)
        form_model.addRow("Components:", self.spin_numExp)

        self.combo_model = QComboBox()
        self.combo_model.addItems(["Parallel (DAS)", "Sequential (SAS)", "Damped Oscillation", "Custom GUI Model"])
        form_model.addRow("Model Type:", self.combo_model)
        
        self.btn_build_model = QPushButton("Open Visual Model Builder")
        self.btn_build_model.setEnabled(False)
        self.btn_build_model.clicked.connect(self.open_visual_model_builder)
        form_model.addRow(self.btn_build_model)
        self.combo_model.currentTextChanged.connect(lambda text: self.btn_build_model.setEnabled("Custom GUI Model" in text))
        
        self.combo_tech = QComboBox()
        self.combo_tech.addItems(["FLUPS", "TAS", "TCSPC"])
        form_model.addRow("Technique:", self.combo_tech)
        
        self.chk_artifact = QCheckBox("Model Coherent Artifact (XPM/Raman)")
        form_model.addRow(self.chk_artifact)
        
        self.chk_nnls = QCheckBox("Force Positive Spectra (NNLS)")
        form_model.addRow(self.chk_nnls)
        
        self.btn_edit_guess = QPushButton("Edit Initial Guesses")
        self.btn_edit_guess.clicked.connect(self._open_guess_editor_and_update)
        form_model.addRow(self.btn_edit_guess)
        gb_model.setLayout(form_model)
        layout_model.addWidget(gb_model)

        # --- Group 5: Workspace ---
        gb_work = QGroupBox("Workspace")
        h_work = QHBoxLayout()
        self.btn_save_proj = QPushButton("Save project")
        self.btn_save_proj.setStyleSheet("background-color: #3C5488; color: white;")
        self.btn_save_proj.clicked.connect(self.save_project)
        self.btn_load_proj = QPushButton("Load project")
        self.btn_load_proj.setStyleSheet("background-color: #E67E22; color: white;")
        self.btn_load_proj.clicked.connect(self.load_project)
        h_work.addWidget(self.btn_save_proj)
        h_work.addWidget(self.btn_load_proj)
        gb_work.setLayout(h_work)
        layout_model.addWidget(gb_work)

        # --- Acción Principal del Fit ---
        h_run = QHBoxLayout()
        
        self.btn_run = QPushButton("RUN FIT")
        self.btn_run.setStyleSheet("background-color: #10B981; border: 1px solid #059669; color: white; font-size: 10pt; font-weight: bold;")
        self.btn_run.setFixedHeight(40)  
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self.run_fit_pipeline) 
        h_run.addWidget(self.btn_run, stretch=3) 
        
        self.btn_abort = QPushButton("ABORT")
        self.btn_abort.setStyleSheet("background-color: #EF4444; border: 1px solid #DC2626; color: white; font-weight: bold; font-size: 10pt;")
        self.btn_abort.setFixedHeight(40)
        self.btn_abort.setEnabled(False)
        self.btn_abort.clicked.connect(self.abort_fit)
        h_run.addWidget(self.btn_abort, stretch=1) 
        
        layout_model.addLayout(h_run)
        
        self.btn_batch = QPushButton("RUN BATCH FIT (All Files)")
        self.btn_batch.setFixedHeight(40)
        self.btn_batch.setEnabled(False)
        self.btn_batch.setStyleSheet("background-color: #0078D7; color: white; font-weight: bold;")
        self.btn_batch.clicked.connect(self.run_batch_pipeline)
        layout_model.addWidget(self.btn_batch)
        
        self.btn_show_das = QPushButton("Show Plots / Results")
        self.btn_show_das.setEnabled(False)
        self.btn_show_das.clicked.connect(self.plot_das_and_more) 
        layout_model.addWidget(self.btn_show_das)

        layout_model.addStretch() # Empuja los botones extra hacia abajo en la segunda pestaña
        
        # --- Plotters Independientes ---
        self.btn_standalone_plotter = QPushButton("Open Paper Plotter (Saved Traces)")
        self.btn_standalone_plotter.clicked.connect(self.open_standalone_plotter)
        self.btn_standalone_plotter.setStyleSheet("background-color: #3C5488; color: white; padding: 6px;")
        layout_model.addWidget(self.btn_standalone_plotter)     
        
        self.btn_sasdas_plotter = QPushButton("Open SAS/DAS Plotter (Spectra)")
        self.btn_sasdas_plotter.clicked.connect(self.open_sasdas_plotter)
        self.btn_sasdas_plotter.setStyleSheet("background-color: #00A087; color: white; padding: 6px;")
        layout_model.addWidget(self.btn_sasdas_plotter)
        
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
        """Activa la bandera de cancelación para detener el ajuste en curso."""
        self._abort_fit = True
        self.btn_abort.setText("Aborting...")
        self.btn_abort.setEnabled(False)    
    def open_identifiability_dialog(self):
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
        """Empaqueta toda la UI, guesses y el modelo visual en un archivo .proj"""
        import pickle
        
        path, _ = QFileDialog.getSaveFileName(self, "Save Kinetic Project", "", "Project Files (*.proj)")
        if not path: return

        # 1. Guardar estado de la interfaz y variables de VarPro
        proj_data = {
            'ui': {
                'numExp': self.spin_numExp.value(),
                'model_idx': self.combo_model.currentIndex(),
                'tech_idx': self.combo_tech.currentIndex(),
                'artifact': self.chk_artifact.isChecked(),
                'nnls': self.chk_nnls.isChecked()
            },
            'guesses': {
                'ini': self.ini,
                'limi': self.limi,
                'lims': self.lims,
                'is_fixed': getattr(self, 'is_fixed', None)
            },
            'visual_model': None
        }

        # 2. Guardar el modelo gráfico (Cajas y Flechas) si existe
        if hasattr(self, 'model_builder_dlg') and self.model_builder_dlg is not None:
            canvas = self.model_builder_dlg.canvas
            # Guardamos las coordenadas X e Y exactas de cada caja
            nodes = [{'name': name, 'x': node.pos().x(), 'y': node.pos().y()} for name, node in canvas.nodes.items()]
            edges = []
            for item in canvas.scene.items():
                if isinstance(item, TransitionEdge):
                    edges.append({
                        'source': item.source_node.name,
                        'target': item.target_node.name,
                        'type': item.param_type,
                        'label': item.label_name
                    })
            proj_data['visual_model'] = {'nodes': nodes, 'edges': edges}

        # Escribir a disco
        try:
            with open(path, 'wb') as f:
                pickle.dump(proj_data, f)
            QMessageBox.information(self, "Éxito", "¡Proyecto guardado correctamente!\nAhora puedes compartir este archivo .proj")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Fallo al guardar el proyecto: {e}")

    def load_project(self):
        """Lee un archivo .proj y reconstruye la interfaz y el lienzo gráfico."""
        import pickle
        
        path, _ = QFileDialog.getOpenFileName(self, "Load Kinetic Project", "", "Project Files (*.proj)")
        if not path: return

        try:
            with open(path, 'rb') as f:
                proj_data = pickle.load(f)

            # 1. Restaurar Interfaz
            ui = proj_data['ui']
            self.spin_numExp.setValue(ui['numExp'])
            self.combo_model.setCurrentIndex(ui['model_idx'])
            self.combo_tech.setCurrentIndex(ui['tech_idx'])
            self.chk_artifact.setChecked(ui['artifact'])
            self.chk_nnls.setChecked(ui['nnls'])

            # 2. Restaurar Guesses (Valores iniciales y límites)
            guesses = proj_data['guesses']
            self.ini = guesses['ini']
            self.limi = guesses['limi']
            self.lims = guesses['lims']
            if guesses['is_fixed'] is not None:
                self.is_fixed = guesses['is_fixed']

            # 3. Restaurar Modelo Visual (Magia)
            v_model = proj_data['visual_model']
            if v_model is not None:
                # Si el usuario no había abierto el builder en esta sesión, lo creamos en la sombra
                if not hasattr(self, 'model_builder_dlg') or self.model_builder_dlg is None:
                    self.model_builder_dlg = ModelBuilderDialog(self)
                
                canvas = self.model_builder_dlg.canvas
                canvas.scene.clear()
                canvas.nodes.clear()
                self.model_builder_dlg.table.setRowCount(0)

                # Recreamos las cajas en sus coordenadas exactas
                for n in v_model['nodes']:
                    canvas.add_state(n['name'], x=n['x'], y=n['y'])
                
                # Recreamos las flechas
                for e in v_model['edges']:
                    src = canvas.nodes[e['source']]
                    tgt = canvas.nodes[e['target']]
                    edge = TransitionEdge(src, tgt, e['type'], e['label'])
                    canvas.scene.addItem(edge)

                # Re-compilamos la matemática por debajo
                self.current_custom_model = self.model_builder_dlg.get_compiled_model()

            QMessageBox.information(self, "Éxito", "¡Proyecto cargado con éxito!\n\nAsegúrate de tener un Dataset (.npy) cargado y ya puedes hacer clic en RUN FIT o abrir el Model Builder para ver tu esquema.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo cargar el proyecto. Puede estar corrupto o ser de una versión antigua.\n\nDetalle: {e}")    
    def run_svd(self):
        """Executes Singular Value Decomposition (SVD) on the active dataset to identify components."""
        if self.data_c is None:
            QMessageBox.warning(self, "Error", "Load the data before trying to run SVD analysis.")
            return
    
        # 1. Run SVD
        # data_c must be [WL x TD]
        try:
            U, s, Vh = np.linalg.svd(self.data_c, full_matrices=False)
            
            self.svd_U = U    # Spectral vectors (species)
            self.svd_s = s    # Weight of the species
            self.svd_V = Vh.T # Temporal vectors (kinetics)
    
            self._plot_svd_results()
            self.tabs.setCurrentWidget(self.tab_svd) 
            
        except Exception as e:
            print(f"SVD Error: {e}")
            
    def _create_svd_canvas(self, tab_widget):
            """
            Creates and embeds the matplotlib canvas for the SVD tab.
            """
            fig = plt.Figure(figsize=(6, 10)) # Hacemos la figura más alta
            # ax1: Scree Plot, ax2: Espectros, ax3: Cinéticas
            ax1 = fig.add_subplot(311) 
            ax2 = fig.add_subplot(312)
            ax3 = fig.add_subplot(313)
            canvas = FigureCanvas(fig)
            
            layout = QVBoxLayout()
            layout.addWidget(canvas)
            tab_widget.setLayout(layout)
            return canvas, (ax1, ax2, ax3)

    def _plot_svd_results(self):
        """Plots the singular values (Scree Plot) and principal components."""
        ax1, ax2, ax3 = self.ax_svd
        ax1.clear()
        ax2.clear()
        ax3.clear()
    
        # --- Plot 1: Scree Plot (Log scale) ---
        n_comp = min(len(self.svd_s), 10) # Ver el top 10
        ax1.semilogy(range(1, n_comp + 1), self.svd_s[:n_comp], 'o-', color='red')
        ax1.set_title("Singular Values (Scree Plot)")
        ax1.set_ylabel("Eigenvalue (log)")
        ax1.grid(True, which="both", ls="-", alpha=0.2)
    
        # Extraer los ejes reales (procesados si existen)
        wl = getattr(self, '_wl_proc', self.WL)
        td = getattr(self, '_td_proc', self.TD)
        n_mostrar = self.spin_numExp.value() 
        
        # --- Plot 2: Spectral components ---
        for i in range(min(n_mostrar, len(self.svd_s))):
            ax2.plot(wl, self.svd_U[:, i], label=f"Comp {i+1}")
        
        ax2.set_title(f"First {n_mostrar} Spectral Components")
        ax2.set_ylabel("Amplitude")
        ax2.axhline(0, color='black', lw=1, alpha=0.5)
        ax2.legend(frameon=True)
        
        # --- Plot 3: Temporal components ---
        for i in range(min(n_mostrar, len(self.svd_s))):
            ax3.plot(td, self.svd_V[:, i], label=f"Comp {i+1}")
            
        ax3.set_title(f"First {n_mostrar} Temporal Components")
        ax3.set_xlabel("Time Delay / ps")
        ax3.set_ylabel("Amplitude")
        ax3.axhline(0, color='black', lw=1, alpha=0.5)
        
        # Aplicamos escala logarítmica si el usuario la tiene seleccionada en la interfaz
        if hasattr(self, 'yscale') and self.yscale == 'symlog':
            ax3.set_xscale('symlog', linthresh=1.0)
            
        ax3.legend(frameon=True)
        
        # Ajustamos los márgenes para que los títulos y ejes no se solapen
        self.canvas_svd.figure.tight_layout()
        self.canvas_svd.draw()
  
  
    def _on_scale_changed(self, text):
        """
        Updates the scale parameter and replots the data canvases.

        Args:
            text (str): The selected scale ('Linear' or 'SymLog').
        """
        self.yscale = text.lower() # 'linear' or 'symlog'
        
        self._update_exp_canvas()
        self._update_fit_canvas()
        self._update_resid_canvas()
        
    def open_visual_model_builder(self):
        """Abre la cuadrícula de diseño y guarda el modelo compilado en memoria."""
        
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
            
            # Forzar regeneración de los guesses iniciales con las dimensiones del nuevo modelo
            self._generate_defaults()
            
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
        Generates the initial parameter guesses. 
        If force_reset=True, overwrites everything. Otherwise, it safely preserves 
        existing kinetic bounds when changing datasets or models.
        """
        numExp = self.spin_numExp.value()
        tech = self.combo_tech.currentText()
        model_str = self.combo_model.currentText()
        
        is_oscillation = "Oscillation" in model_str

        if self.data_c is not None:
            numWL = self.data_c.shape[0]
        elif self.WL is not None:
            numWL = len(self.WL)
        else:
            numWL = 1
            
        # === 1. RESPALDO DE SEGURIDAD ===
        old_ini = self.ini.copy() if getattr(self, 'ini', None) is not None else None
        old_limi = self.limi.copy() if getattr(self, 'limi', None) is not None else None
        old_lims = self.lims.copy() if getattr(self, 'lims', None) is not None else None
        old_fixed = self.is_fixed.copy() if getattr(self, 'is_fixed', None) is not None else None

        taus_defaults = [0.5, 5.0, 50.0, 500.0, 2000.0, 5000.0]
        w_guess = 0.15 if tech == 'TAS' else (0.3 if tech == 'FLUPS' else 0.1)
        
        num_kin_params = 0

        # === BLOQUE DEL MODELO CUSTOM ===
        if "Custom GUI Model" in model_str: 
            if not hasattr(self, 'current_custom_model') or self.current_custom_model is None:
                return False

            model = self.current_custom_model
            p_ini, p_low, p_upp = model.get_default_guesses_and_bounds()
            num_excitados = len(model.states)
            
            num_kin_params = 2 + len(p_ini)
            L = num_kin_params + numWL * num_excitados
            
            self.ini = np.zeros(L)
            self.limi = -np.inf * np.ones(L)
            self.lims = np.inf * np.ones(L)
            
            self.ini[0] = 0.15; self.limi[0] = 0.05; self.lims[0] = 2.0  # w
            self.ini[1] = 0.0;  self.limi[1] = -5.0; self.lims[1] = 5.0  # t0
            
            self.ini[2:num_kin_params] = p_ini
            self.limi[2:num_kin_params] = p_low
            self.lims[2:num_kin_params] = p_upp
            
            self.ini[num_kin_params:] = 0.01
            
        # === BLOQUE DE MODELOS ESTÁNDAR ===
        else:
            if is_oscillation:
                 L = (2 + numExp + 3) + numWL * (numExp + 1)
                 num_kin_params = 2 + numExp + 3
                 self.ini = np.zeros(L); self.limi = -np.inf * np.ones(L); self.lims = np.inf * np.ones(L)
                 self.ini[0] = w_guess; self.limi[0] = 0.05; self.lims[0] = 2.0
                 self.ini[1] = 0.0;     self.limi[1] = -5.0; self.lims[1] = 5.0
                 base_tau = 2
                 for n in range(numExp):
                     val_t = taus_defaults[n] if n < len(taus_defaults) else 1000.0*(n+1)
                     self.ini[base_tau + n] = val_t
                     self.limi[base_tau + n] = 0.001
                     self.lims[base_tau + n] = 1e8
                 idx_osc = base_tau + numExp
                 self.ini[idx_osc] = 0.1; self.limi[idx_osc] = 0.0; self.lims[idx_osc] = 100.0
                 self.ini[idx_osc+1] = 1.0; self.limi[idx_osc+1] = 0.0; self.lims[idx_osc+1] = 500.0
                 self.ini[idx_osc+2] = 0.0; self.limi[idx_osc+2] = -np.pi; self.lims[idx_osc+2] = np.pi
                 val_A = 1000.0 if tech == 'TCSPC' else (5.0 if tech == 'FLUPS' else 0.01)
                 self.ini[num_kin_params:] = val_A 
                 
            else:
                L = 2 + numExp + numWL*numExp
                num_kin_params = 2 + numExp
                self.ini = np.zeros(L); self.limi = -np.inf * np.ones(L); self.lims = np.inf * np.ones(L)
                self.ini[0] = w_guess; self.limi[0] = 0.05; self.lims[0] = 2.0
                self.ini[1] = 0.0;     self.limi[1] = -5.0; self.lims[1] = 5.0
                base_tau = 2
                for n in range(numExp):
                    self.ini[base_tau + n] = taus_defaults[n] if n < len(taus_defaults) else 1000.0*(n+1)
                    self.limi[base_tau + n] = 0.001; self.lims[base_tau + n] = 1e8
                val_A = 1000.0 if tech == 'TCSPC' else (5.0 if tech == 'FLUPS' else 0.01)
                self.ini[num_kin_params:] = val_A
                


        # === 2. RESTAURAR LOS VALORES PREVIOS SIN ROMPER NADA ===
        if not force_reset and old_ini is not None:
            # MAGIA: Averiguar exactamente cuántos parámetros cinéticos tenía el modelo antiguo
            # buscando el primer límite -infinito (que siempre pertenece a las amplitudes y nunca a Taus/T0).
            old_kin_len = len(old_ini)
            if old_limi is not None:
                inf_indices = np.where(old_limi == -np.inf)[0]
                if len(inf_indices) > 0:
                    old_kin_len = inf_indices[0]
            
            # Solo restauramos el solapamiento válido entre la cinética vieja y la nueva
            safe_len = min(num_kin_params, old_kin_len)
            
            if safe_len > 0:
                self.ini[:safe_len] = old_ini[:safe_len]
                self.limi[:safe_len] = old_limi[:safe_len]
                self.lims[:safe_len] = old_lims[:safe_len]
            
            if not hasattr(self, 'is_fixed') or len(self.is_fixed) != L:
                self.is_fixed = np.zeros(L, dtype=bool)
                
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
            
            # Detect experimental technique
            if getattr(self.parent_app, "is_TAS_mode", False):
                self.combo_tech.setCurrentText("TAS")
            else:
                self.combo_tech.setCurrentText("FLUPS")
                
            # Refresh UI components and enable execution   
            self._update_ui_limits_from_data()
            self.btn_run.setEnabled(True)
            self.btn_batch.setEnabled(True)
            self.label_status.setText(f"Loaded from Parent: {len(self.WL)} WL, {len(self.TD)} TD")

    def load_data(self):
        """Carga múltiples archivos .npy para compararlos."""
  
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select .npy files", "", "Numpy Files (*.npy)"
        )
        
        
        if not file_paths:
            return
        
        self.base_dir = os.path.dirname(file_paths[0])
            
        for path in file_paths:
            try:
                # Cargamos el archivo .npy directamente usando numpy
                # allow_pickle=True y .item() extraen el diccionario directamente
                loaded_dict = np.load(path, allow_pickle=True).item()
                
                # Extraemos las matrices usando las claves exactas de tu script original
                raw_data = loaded_dict['data_c']
                WL = loaded_dict['WL']
                TD = loaded_dict['TD']
                
                self.data_raw_list.append(raw_data.copy())
                self.data_c_list.append(raw_data.copy()) # Por defecto igual a crudo
                self.TD_list.append(TD)
                self.WL_list.append(WL)
                self.filenames.append(os.path.basename(path))
                
            except Exception as e:
                QMessageBox.critical(self, f"Error loading {os.path.basename(path)}", str(e))
                
        # Sincronizamos la UI usando el primer archivo cargado como base
        if self.WL_list:
            self.WL = self.WL_list[0]
            self.TD = self.TD_list[0]
            self.data_raw = self.data_raw_list[0]
            self.data_c = self.data_c_list[0]
            
            self._update_ui_limits_from_data()
            self.btn_run.setEnabled(True)
            self.btn_batch.setEnabled(True)
            
            # --- NUEVO: Poblar el combo box con los archivos cargados ---
            self.combo_active_dataset.blockSignals(True) # Bloqueamos eventos temporales
            self.combo_active_dataset.clear()
            self.combo_active_dataset.addItems(self.filenames)
            self.combo_active_dataset.setCurrentIndex(0)
            self.combo_active_dataset.blockSignals(False)
            
            self.label_status.setText(f"Loaded {len(self.filenames)} Files")
    def _on_active_dataset_changed(self, index):
        """Cambia dinámicamente el dataset visible y carga sus ajustes si existen."""
        if index < 0 or index >= len(self.data_c_list):
            return
            
        # 1. Actualizar los punteros a las matrices del archivo seleccionado
        self.data_raw = self.data_raw_list[index]
        self.data_c = self.data_c_list[index]
        self.WL = self.WL_list[index]
        self.TD = self.TD_list[index]
        
        # Comprobar si ya pasaron por el pre-procesamiento (recortes/binning)
        use_proc = False
        if hasattr(self, 'wl_proc_list') and len(self.wl_proc_list) > index:
            self._wl_proc = self.wl_proc_list[index]
            self._td_proc = self.td_proc_list[index]
            use_proc = True
            
        # 2. Re-dibujar el mapa experimental correspondiente
        self._update_exp_canvas(use_processed=use_proc)
        
        # 3. Lógica inteligente: Intentar buscar si este archivo ya tiene un ajuste guardado
        filename = self.filenames[index]
        base_name = os.path.splitext(filename)[0]
        
        # Ruta donde el Batch guarda el ajuste de esta molécula específica
        batch_fit_file = os.path.join(self.base_dir, "Batch_Results", base_name, "GFitResults.npy")
        # Ruta del ajuste único estándar
        standard_fit_file = os.path.join(self.base_dir, "fit", "GFitResults.npy")
        
        fit_path = None
        if os.path.exists(batch_fit_file):
            fit_path = batch_fit_file
        elif index == 0 and os.path.exists(standard_fit_file):
            fit_path = standard_fit_file
            
        if fit_path and os.path.exists(fit_path):
            try:
                # Cargamos el Fit histórico de esa molécula
                fit_data = np.load(fit_path, allow_pickle=True).item()
                self.fit_fitres = fit_data["fitres"]
                self.fit_resid = fit_data["resid"]
                self.extracted_taus = fit_data["taus"]
                self.extracted_errtaus = fit_data["err_taus"]
                self.As = fit_data["As"]
                self.errAs = fit_data["errAs"]
                self.numExp = len(self.extracted_taus)
                
                # Sincronizamos el spinbox de componentes por si acaso varió
                self.spin_numExp.setValue(self.numExp)
                
                # Re-dibujamos las pestañas de ajuste y residuales con los datos correctos
                self._update_fit_canvas()
                self._update_resid_canvas()
                self.btn_show_das.setEnabled(True)
            except Exception as e:
                print(f"Error cargando el Fit de {base_name}: {e}")
                self._clear_fit_plots()
        else:
            # Si la molécula seleccionada aún no se ha ajustado, limpiamos las pantallas del Fit
            self._clear_fit_plots()
    
    def remove_active_dataset(self):
        """Safely removes the currently selected dataset from memory without restarting."""
        if not getattr(self, 'data_c_list', None):
            return

        idx = self.combo_active_dataset.currentIndex()
        if idx < 0 or idx >= len(self.data_c_list):
            return

        filename = self.filenames[idx]
        reply = QMessageBox.question(
            self, 'Remove Dataset',
            f"Are you sure you want to remove the dataset:\n\n'{filename}'?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        # 1. Purgar el archivo de todas las listas de la memoria
        self.data_raw_list.pop(idx)
        self.data_c_list.pop(idx)
        self.WL_list.pop(idx)
        self.TD_list.pop(idx)
        self.filenames.pop(idx)

        # Si ya se habían preprocesado, borrarlos de ahí también
        if hasattr(self, 'wl_proc_list') and len(self.wl_proc_list) > idx:
            self.wl_proc_list.pop(idx)
            self.td_proc_list.pop(idx)

        # 2. Actualizar el ComboBox sin disparar la actualización gráfica aún
        self.combo_active_dataset.blockSignals(True)
        self.combo_active_dataset.removeItem(idx)
        self.combo_active_dataset.blockSignals(False)

        # 3. Decidir qué mostrar ahora en la pantalla
        if len(self.data_c_list) > 0:
            # Si quedan archivos, saltamos al archivo anterior (o al primero)
            self.label_status.setText(f"Loaded {len(self.filenames)} Files")
            new_idx = max(0, idx - 1)
            self.combo_active_dataset.setCurrentIndex(new_idx)
            self._on_active_dataset_changed(new_idx)
            
        else:
            # 4. SUPERVIVENCIA: Si hemos borrado el ÚLTIMO archivo, resetear todo visualmente
            self.label_status.setText("No data loaded")
            self.data_raw = None
            self.data_c = None
            self.WL = None
            self.TD = None
            if hasattr(self, '_wl_proc'): del self._wl_proc
            if hasattr(self, '_td_proc'): del self._td_proc

            # Limpiar los lienzos usando la estructura de diccionario (GridSpec)
            if hasattr(self, 'ax_exp') and isinstance(self.ax_exp, dict):
                self.ax_exp['map'].clear()
                self.ax_exp['spec'].clear()
                self.ax_exp['kin'].clear()
                
            self._clear_colorbar_if_exists(getattr(self, 'cbar_exp', None))
            self.canvas_exp.draw_idle()

            self._clear_fit_plots()
            self.lbl_cursor.setText("Cursor: Out of the 2D MAP")

            # Bloquear botones para evitar crasheos
            self.btn_run.setEnabled(False)
            self.btn_batch.setEnabled(False)
            

    def _clear_fit_plots(self):
        """Limpia los lienzos de Fit y Residuales si el archivo actual no está ajustado."""
        self.fit_fitres = None
        self.fit_resid = None
        self.As = None
        
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
    def _preview_data_processing(self):
        """
        Procesa los datos crudos para TODOS los archivos cargados aplicando: 
        Baseline -> Wavelength Crop -> Time Crop -> Binning.
        """
        if getattr(self, 'data_raw_list', None) is None or not self.data_raw_list:
            if self.data_raw is None: return
            raw_list = [self.data_raw]
            wl_list = [self.WL]
            td_list = [self.TD]
        else:
            raw_list = self.data_raw_list
            wl_list = self.WL_list
            td_list = self.TD_list

        self.data_c_list = []
        self.wl_proc_list = []
        self.td_proc_list = []

        # Aplicar el procesamiento a cada archivo cargado
        for idx in range(len(raw_list)):
            # --- 1. FRENO PARA EL BINNING ---
            QApplication.processEvents()
            if getattr(self, '_abort_fit', False):
                QMessageBox.warning(self, "Cancelado", "El procesamiento de datos fue cancelado.")
                return # Salimos de la función inmediatamente
            # --------------------------------
            
            temp_data = raw_list[idx].copy()
            temp_WL = wl_list[idx].copy()
            temp_TD = td_list[idx].copy()

            # 1. Baseline Correction
            n_pts = self.spin_bl.value()
            if n_pts > 0 and temp_data.shape[1] >= n_pts:
                baseline = np.mean(temp_data[:, :n_pts], axis=1, keepdims=True)
                temp_data = temp_data - baseline
                
            # 2. Wavelength Cropping
            w_min = self.spin_wl_min.value()
            w_max = self.spin_wl_max.value()
            mask_w = (temp_WL >= min(w_min, w_max)) & (temp_WL <= max(w_min, w_max))
            
            # Exclusiones específicas
            if hasattr(self, 'line_exclude'):
                exclude_str = self.line_exclude.text().strip()
                if exclude_str:
                    mask_exclude = np.zeros_like(temp_WL, dtype=bool)
                    ranges = exclude_str.split(',')
                    for r in ranges:
                        try:
                            parts = r.split('-')
                            if len(parts) == 2:
                                c_min = float(parts[0].strip())
                                c_max = float(parts[1].strip())
                                mask_exclude |= (temp_WL >= min(c_min, c_max)) & (temp_WL <= max(c_min, c_max))
                        except ValueError:
                            pass 
                    mask_w &= (~mask_exclude)
            
            if np.any(mask_w):
                temp_data = temp_data[mask_w, :]
                temp_WL = temp_WL[mask_w]
                
            # 3. Time Cropping
            t_min = self.spin_t_min.value()
            t_max = self.spin_t_max.value()
            mask_t = (temp_TD >= min(t_min, t_max)) & (temp_TD <= max(t_min, t_max))
            
            if np.any(mask_t):
                temp_data = temp_data[:, mask_t]
                temp_TD = temp_TD[mask_t]
                
            if hasattr(self, 'chk_zero_neg') and self.chk_zero_neg.isChecked():
                mask_neg = temp_TD < 0
                if np.any(mask_neg):
                    temp_data[:, mask_neg] = 0.0

            # 4. Binning
            b_size = self.spin_bin.value()
            if b_size > 1:
                n_wl = temp_data.shape[0]
                new_len = n_wl // b_size
                if new_len > 0:
                    temp_data = temp_data[:new_len*b_size, :]
                    temp_data = temp_data.reshape(new_len, b_size, temp_data.shape[1]).mean(axis=1)
                    temp_WL = temp_WL[:new_len*b_size]
                    temp_WL = temp_WL.reshape(new_len, b_size).mean(axis=1)

            
            # --- 5. NORMALIZACIÓN ---
            if hasattr(self, 'chk_norm_data') and self.chk_norm_data.isChecked():
                # Forzamos conversión a float por si los conteos son enteros
                temp_data = temp_data.astype(float) 
                max_abs_val = np.nanmax(np.abs(temp_data))
                if max_abs_val != 0:
                    temp_data = temp_data / max_abs_val
            # ------------------------
            
            # Guardar el resultado procesado
            self.data_c_list.append(temp_data)
            self.wl_proc_list.append(temp_WL)
            self.td_proc_list.append(temp_TD)

    
                    
        # Actualizar variables del modelo global usando el primer archivo
        if self.data_c_list:
            self.data_c = self.data_c_list[0]
            self._wl_proc = self.wl_proc_list[0]
            self._td_proc = self.td_proc_list[0]
            
            self._update_exp_canvas(use_processed=True)
            self.label_status.setText(f"Processed {len(self.data_c_list)} files")
            
            try:
                
                outdir = os.path.join(self.base_dir, "Plots")
                os.makedirs(outdir, exist_ok=True)
                
                # Iteramos sobre todos los datasets que se acaban de procesar
                for i in range(len(self.data_c_list)):
                    z_data = self.data_c_list[i]
                    x_data = self.wl_proc_list[i]
                    y_data = self.td_proc_list[i]
                    
                    # Extraer el nombre original sin la extensión .npy
                    if hasattr(self, 'filenames') and i < len(self.filenames):
                        base_name = os.path.splitext(self.filenames[i])[0]
                    else:
                        base_name = f"Dataset_{i+1}"
                        
                    # 1. Crear figura "desconectada" de la GUI para evitar pantallazos
                    fig_temp = Figure(figsize=(6, 4))
                    canvas_temp = FigureCanvasAgg(fig_temp) # Motor de renderizado en segundo plano
                    ax_temp = fig_temp.add_subplot(111)
                    
                    # 2. Calcular límites reales para no cortar la señal
                    vmin_val = np.nanmin(z_data)
                    vmax_val = np.nanmax(z_data)
                    
                    # 3. Dibujar el mapa
                    pcm = ax_temp.pcolormesh(x_data, y_data, z_data.T, 
                                             shading='auto', cmap='jet', 
                                             vmin=vmin_val, vmax=vmax_val)
                    
                    ax_temp.set_title(f"Processed: {base_name}", fontsize=10)
                    ax_temp.set_xlabel("Wavelength (nm)")
                    ax_temp.set_ylabel("Delay (ps)")
                    
                    # Aplicar escala (SymLog o Lineal) según la interfaz
                    if hasattr(self, 'yscale') and self.yscale == 'symlog':
                        ax_temp.set_yscale('symlog', linthresh=2) 
                    else:
                        ax_temp.set_yscale('linear')
                        
                    fig_temp.colorbar(pcm, ax=ax_temp, label='$\Delta A$ / -')
                    fig_temp.tight_layout()
                    
                    # 4. Guardar imagen (Ya no hace falta plt.close() porque no usa pyplot)
                    filepath = os.path.join(outdir, f"Map_Processed_{base_name}.png")
                    fig_temp.savefig(filepath, dpi=300)
                
                print(f"Éxito: Se han exportado {len(self.data_c_list)} mapas procesados a la carpeta /Plots/")
                
            except Exception as e:
                print(f"Error durante el guardado masivo de los mapas procesados: {e}")
                

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
            
            self.pcm_exp = ax_map.pcolormesh(Xs, Ys, self.data_c.T, shading="auto", cmap=cmap_choice, vmin=self.exp_vmin, vmax=self.exp_vmax)
            ax_map.set_xlabel("Wavelength (nm)")
            ax_map.set_ylabel("Delay (ps)")
            
            if hasattr(self, 'yscale') and self.yscale == 'symlog':
                ax_map.set_yscale('symlog', linthresh=2)
            else:
                ax_map.set_yscale('linear')
            
            # --- INYECCIÓN DE LÍNEAS PARA EL EFECTO HOVER ---
            self.ax_exp['line_spec'], = ax_spec.plot(Xs, self.data_c[:, 0], color='darkred', lw=1.5)
            self.ax_exp['line_kin'], = ax_kin.plot(self.data_c[0, :], Ys, color='darkblue', lw=1.5)
            self.ax_exp['vline'] = ax_map.axvline(Xs[0], color='white', ls='--', lw=1, alpha=0.8)
            self.ax_exp['hline'] = ax_map.axhline(Ys[0], color='white', ls='--', lw=1, alpha=0.8)
            
            ax_spec.set_xlim(Xs.min(), Xs.max()); ax_spec.set_ylim(self.exp_vmin, self.exp_vmax)
            ax_kin.set_xlim(self.exp_vmin, self.exp_vmax)
            
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax_kin)
            cax = divider.append_axes("right", size="15%", pad=0.1)
            self.cbar_exp = self.canvas_exp.figure.colorbar(self.pcm_exp, cax=cax, label='$\Delta A$ / -')
            
            self.canvas_exp.draw_idle()
            
            # Conectar el sensor del ratón
            if not hasattr(self, 'cid_mouse_move'):
                self.cid_mouse_move = self.canvas_exp.mpl_connect('motion_notify_event', self.on_mouse_move)
                
        except Exception as e:
            print(f"Plotting error: {e}")
            
    def on_mouse_move(self, event):
        """Dynamic cross-sections: actualiza instantáneamente los cortes 1D al pasar el ratón."""
        
        # 1. PROTECCIÓN: Si el ratón no está sobre ningún gráfico, salimos inmediatamente
        if event.inaxes is None:
            self.lbl_cursor.setText("Cursor: Out of the 2D MAP")
            return

        active_map_dict = None
        data_matrix = None
        
        # 2. PROTECCIÓN: Usamos .get('map') en lugar de ['map'] para evitar KeyErrors
        if hasattr(self, 'ax_exp') and isinstance(self.ax_exp, dict) and event.inaxes == self.ax_exp.get('map'):
            active_map_dict = self.ax_exp
            data_matrix = self.data_c
        elif hasattr(self, 'ax_fit') and isinstance(self.ax_fit, dict) and event.inaxes == self.ax_fit.get('map'):
            active_map_dict = self.ax_fit
            data_matrix = getattr(self, 'fit_fitres', None)
        elif hasattr(self, 'ax_resid') and isinstance(self.ax_resid, dict) and event.inaxes == self.ax_resid.get('map'):
            active_map_dict = self.ax_resid
            data_matrix = getattr(self, 'fit_resid', None)
            
        # 3. PROTECCIÓN: Si faltan las líneas 1D porque se están borrando, ignoramos el evento
        if active_map_dict is None or data_matrix is None or 'line_spec' not in active_map_dict:
            self.lbl_cursor.setText("Cursor: Out of the 2D MAP")
            return
            
        x = event.xdata
        y = event.ydata
        if x is None or y is None: return
        
        Xs = getattr(self, '_wl_proc', self.WL)
        Ys = getattr(self, '_td_proc', self.TD)
        
        try:
            # Buscar el píxel más cercano a donde está el ratón
            idx_wl = (np.abs(Xs - x)).argmin()
            idx_td = (np.abs(Ys - y)).argmin()
            z_val = data_matrix[idx_wl, idx_td]
            
            # Actualizar el texto superior
            self.lbl_cursor.setText(f"Cursor: λ = {x:.1f} nm  |  Delay = {y:.3f} ps  |  ΔA = {z_val:.3e}")
            
            # --- LA MAGIA: Actualizar las líneas 1D ---
            active_map_dict['line_spec'].set_data(Xs, data_matrix[:, idx_td])
            active_map_dict['line_kin'].set_data(data_matrix[idx_wl, :], Ys)
            
            # Mover la cruz blanca sobre el mapa
            active_map_dict['vline'].set_xdata([Xs[idx_wl]])
            active_map_dict['hline'].set_ydata([Ys[idx_td]])
            
            # Refrescar solo el lienzo que estamos tocando
            event.canvas.draw_idle()
            
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
                if Z.shape[0] < 2 or Z.shape[1] < 2: return
                vmin = getattr(self, 'exp_vmin', np.nanmin(Z))
                vmax = getattr(self, 'exp_vmax', np.nanmax(Z))
    
                self.pcm_fit = ax_map.pcolormesh(Xs, Ys, Z, shading='auto', cmap='jet', vmin=vmin, vmax=vmax)
                ax_map.set_xlabel("Wavelength (nm)"); ax_map.set_ylabel("Delay (ps)")
                
                if hasattr(self, 'yscale') and self.yscale == 'symlog':
                    ax_map.set_yscale('symlog', linthresh=2)
                else:
                    ax_map.set_yscale('linear')
                    
                self.ax_fit['line_spec'], = ax_spec.plot(Xs, data_mat[:, 0], color='darkred', lw=1.5)
                self.ax_fit['line_kin'], = ax_kin.plot(data_mat[0, :], Ys, color='darkblue', lw=1.5)
                self.ax_fit['vline'] = ax_map.axvline(Xs[0], color='white', ls='--', lw=1, alpha=0.8)
                self.ax_fit['hline'] = ax_map.axhline(Ys[0], color='white', ls='--', lw=1, alpha=0.8)
    
                ax_spec.set_xlim(Xs.min(), Xs.max()); ax_spec.set_ylim(vmin, vmax)
                ax_kin.set_xlim(vmin, vmax)
    
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                divider = make_axes_locatable(ax_kin)
                cax = divider.append_axes("right", size="15%", pad=0.1)
                self.cbar_fit = self.canvas_fit.figure.colorbar(self.pcm_fit, cax=cax, label='$\Delta A$ / -')
                
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
            if Z.shape[0] < 2 or Z.shape[1] < 2: return
            vals = Z.flatten()
            vmin = np.percentile(vals, 1); vmax = np.percentile(vals, 99)

            self.pcm_resid = ax_map.pcolormesh(Xs, Ys, Z, shading='auto', cmap='jet', vmin=vmin, vmax=vmax)
            ax_map.set_xlabel("Wavelength (nm)"); ax_map.set_ylabel("Delay (ps)")
            
            if hasattr(self, 'yscale') and self.yscale == 'symlog':
                ax_map.set_yscale('symlog', linthresh=2)
            else:
                ax_map.set_yscale('linear')
                
            self.ax_resid['line_spec'], = ax_spec.plot(Xs, data_mat[:, 0], color='purple', lw=1.5)
            self.ax_resid['line_kin'], = ax_kin.plot(data_mat[0, :], Ys, color='green', lw=1.5)
            self.ax_resid['vline'] = ax_map.axvline(Xs[0], color='white', ls='--', lw=1, alpha=0.8)
            self.ax_resid['hline'] = ax_map.axhline(Ys[0], color='white', ls='--', lw=1, alpha=0.8)

            ax_spec.set_xlim(Xs.min(), Xs.max()); ax_spec.set_ylim(vmin, vmax)
            ax_kin.set_xlim(vmin, vmax)

            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax_kin)
            cax = divider.append_axes("right", size="15%", pad=0.1)
            self.cbar_resid = self.canvas_resid.figure.colorbar(self.pcm_resid, cax=cax, label='Residual')
            
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
        self.btn_run.setEnabled(False)
        self.btn_abort.setEnabled(True)
        self.btn_abort.setText("ABORT")
        
        try:
            if self.data_raw is None:
                QMessageBox.warning(self, "No data", "Load data first.")
                return
            
            self._preview_data_processing()
            if self.data_c is None or self.data_c.size == 0: return

            self.numExp = self.spin_numExp.value()
            self.tech = self.combo_tech.currentText()
        
            
            model_str = self.combo_model.currentText()
            if "Sequential" in model_str:
                self.model_type = "Sequential"
            elif "Oscillation" in model_str:
                self.model_type = "Damped Oscillation"
            elif "Custom GUI Model" in model_str:
                self.model_type = "Custom GUI Model"
                if not hasattr(self, 'current_custom_model') or self.current_custom_model is None:
                    QMessageBox.warning(self, "Error", "Abre el Visual Model Builder y compila tu modelo primero.")
                    return
                # FIJAMOS EL TAMAÑO A LOS ESTADOS QUE HAS DIBUJADO
                self.numExp = len(self.current_custom_model.states)
            else:
                self.model_type = "Parallel"

            numWL = self.data_c.shape[0] if self.data_c is not None else 0
            
            if self.model_type == "Damped Oscillation":
                 L_needed = (2 + self.numExp + 3) + numWL * (self.numExp + 1)
            elif self.model_type == "Custom GUI Model":
                 num_kin_params = len(self.current_custom_model.param_labels)
                 L_needed = 2 + num_kin_params + numWL * self.numExp
            else:             
                L_needed = 2 + self.numExp + numWL*self.numExp

            if self.ini is None or len(self.ini) != L_needed:
                self._generate_defaults()
                
            self._temp_fit_TD = getattr(self, '_td_proc', self.TD)
            self._temp_fit_WL = getattr(self, '_wl_proc', self.WL)
            
            self._run_least_squares_with_progress()
            self._postprocess_fit_and_save()
            
        except InterruptedError:
            QMessageBox.warning(self, "Aborted fit","Manually aborted fit")
            
        finally:
            # Cuando termine (o si falla/aborta), restauramos los botones
            self.btn_run.setEnabled(True)
            self.btn_abort.setEnabled(False)
            self.btn_abort.setText("ABORT")
            
    def run_batch_pipeline(self):
        """Ejecuta el ajuste para todos los archivos cargados de forma secuencial."""
        if not getattr(self, 'data_c_list', None):
            QMessageBox.warning(self, "No data", "Load multiple files first to run a batch fit.")
            return

        # 1. Aplicar pre-procesamiento si no se ha hecho
        self._preview_data_processing()
        
        # 2. Configuración inicial del modelo
        self.numExp = self.spin_numExp.value()
        self.tech = self.combo_tech.currentText()
        
        model_str = self.combo_model.currentText()
        if "Sequential" in model_str: self.model_type = "Sequential"
        elif "Oscillation" in model_str: self.model_type = "Damped Oscillation"
        else: self.model_type = "Parallel"
        
        # Asegurar que tenemos guesses iniciales
        if self.ini is None:
            self._generate_defaults()
            
        # Guardar los guesses originales para reiniciar en cada iteración
        original_ini = self.ini.copy()
        
        # Carpeta maestra para el Batch
        batch_outdir = os.path.join(self.base_dir, "Batch_Results")
        os.makedirs(batch_outdir, exist_ok=True)
        
        # Bandera para evitar que los popups bloqueen el bucle
        self.is_batch_running = True 
        
        self._abort_fit = False
        self.btn_abort.setEnabled(True)
        self.btn_abort.setText("ABORT")
        
        try:
            for idx in range(len(self.data_c_list)):
                QApplication.processEvents()
                if getattr(self, '_abort_fit', False):
                    raise InterruptedError("Batch Fit cancelado manualmente.")
                    
                filename = self.filenames[idx] if idx < len(self.filenames) else f"Dataset_{idx+1}"
                base_name = os.path.splitext(filename)[0]
                
                self.label_status.setText(f"Batch Fitting: {idx+1}/{len(self.data_c_list)} ({base_name})")
                
                self.combo_active_dataset.blockSignals(True)
                self.combo_active_dataset.setCurrentIndex(idx)
                self.combo_active_dataset.blockSignals(False)
                
                QApplication.processEvents()
                
                # Cargar datos específicos de esta iteración
                self.data_c = self.data_c_list[idx]
                self._temp_fit_WL = self.wl_proc_list[idx] if hasattr(self, 'wl_proc_list') else self.WL_list[idx]
                self._temp_fit_TD = self.td_proc_list[idx] if hasattr(self, 'td_proc_list') else self.TD_list[idx]
                
                # Restaurar guesses y definir carpeta de salida única
                self.ini = original_ini.copy()
                self.current_batch_outdir = os.path.join(batch_outdir, base_name)
                
                # Ejecutar el núcleo matemático
                self._run_least_squares_with_progress()
                self._postprocess_fit_and_save()
                
            self.label_status.setText(f"Batch Completed: {len(self.data_c_list)} files.")
            QMessageBox.information(self, "Batch Complete", f"Successfully fitted {len(self.data_c_list)} datasets.\nResults saved in: {batch_outdir}")
            
        except Exception as e:
            QMessageBox.critical(self, "Batch Error", f"Error during batch fit: {str(e)}")
        finally:
            self.is_batch_running = False
            self.btn_abort.setEnabled(False)
            
    def _open_guess_editor_and_update(self):
            """Opens a dialog to manually edit initial guesses, bounds, and fixed parameters."""
            numExp = self.spin_numExp.value()
            model_str = self.combo_model.currentText()
            is_oscillation = "Oscillation" in model_str
            is_custom = "Custom GUI Model" in model_str
            
            # Determine number of wavelengths for parameter indexing
            if self.data_c is not None: numWL = self.data_c.shape[0]
            elif self.WL is not None: numWL = len(self.WL)
            else: numWL = 1
                
            # Calculate expected vector length based on the selected model
            if is_custom:
                if not hasattr(self, 'current_custom_model') or self.current_custom_model is None:
                    QMessageBox.warning(self, "Atención", "Aún no has diseñado ningún modelo. Abre el Visual Builder primero.")
                    return
                model = self.current_custom_model
                num_kin_params = len(model.param_labels)
                num_states = len(model.states)
                L_needed = 2 + num_kin_params + numWL * num_states
            elif is_oscillation:
                L_needed = 2 + numExp + 3 + numWL * (numExp + 1)
            else:
                L_needed = 2 + numExp + numWL * numExp
                
            # Regenerate defaults if the vector size is inconsistent
            if self.ini is None or len(self.ini) != L_needed:
                self._generate_defaults()
    
            L = len(self.ini)
            dlg = QDialog(self)
            dlg.setWindowTitle(f"Edit Initial Guesses - {model_str}")
            dlg.resize(800, 600)
            v = QVBoxLayout()
            
            # Initialize Table Widget
            table = QTableWidget(L, 5)
            table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            table.setHorizontalHeaderLabels(["Parameter", "Value", "Lower Bound", "Upper Bound", "Fix?"])
            
            if not hasattr(self, 'is_fixed') or len(self.is_fixed) != L:
                self.is_fixed = np.zeros(L, dtype=bool)
                
            for i in range(L):
                label = f"{i}: "
                
                # --- Labeling logic para el Modelo Visual Interactivo ---
                if is_custom:
                    if i == 0: label += "w (IRF Width)"
                    elif i == 1: label += "t0 (Time Zero)"
                    elif i < 2 + num_kin_params:
                        idx_param = i - 2
                        p_type = "Branching Ratio" if "gamma" in model.param_labels[idx_param].lower() else "Tau (ps)"
                        label += f"[{p_type}] {model.param_labels[idx_param]}"
                    else:
                        local_idx = i - (2 + num_kin_params)
                        wl_idx = local_idx // num_states
                        p_idx = local_idx % num_states
                        curr_wl = self._wl_proc[wl_idx] if hasattr(self, '_wl_proc') else wl_idx
                        state_name = model.states[p_idx]
                        label += f"Amplitude ({state_name}) @ {curr_wl:.1f}nm"
                        
                # Labeling logic for Damped Oscillation model
                elif is_oscillation:
                    if i == 0: label += "w (IRF Width)"
                    elif i == 1: label += "t0 (Time Zero)"
                    elif i < 2 + numExp: label += f"τ{i-1} (Lifetime)"
                    elif i == 2 + numExp: label += "α (Damping/Decay)"
                    elif i == 2 + numExp + 1: label += "ω (Ang. Frequency)"
                    elif i == 2 + numExp + 2: label += "φ (Phase)"
                    else:
                        local_idx = i - (2 + numExp + 3)
                        wl_idx = local_idx // (numExp + 1)
                        p_idx = local_idx % (numExp + 1)
                        curr_wl = self._wl_proc[wl_idx] if hasattr(self, '_wl_proc') else wl_idx
                        if p_idx < numExp: label += f"A{p_idx+1} (Amp) @ {curr_wl:.1f}nm"
                        else: label += f"B (Osc. Amp) @ {curr_wl:.1f}nm"
                        
                # Labeling logic for standard (Parallel/Sequential) and Chirp models
                else:
                    if i == 0: label += "w (FWHM (ps))"
                    elif i == 1: label += "t0 (Time Zero)"
                    elif i < 2 + numExp: label += f"τ{i-1} (Lifetime)"
                    else:
                        local_idx = i - (2 + numExp)
                        wl_idx = local_idx // numExp
                        p_idx = local_idx % numExp
                        label += f"A{p_idx+1} @ WL {wl_idx}"
    
                # Populate table row
                item_lbl = QTableWidgetItem(label)
                # FIX 1: Manera segura de quitar la edición
                item_lbl.setFlags(item_lbl.flags() & ~Qt.ItemIsEditable) 
                table.setItem(i, 0, item_lbl)
                table.setItem(i, 1, QTableWidgetItem(str(self.ini[i])))
                table.setItem(i, 2, QTableWidgetItem(str(self.limi[i])))
                table.setItem(i, 3, QTableWidgetItem(str(self.lims[i])))
                
                # Checkbox for fixing parameters during optimization
                chk_item = QTableWidgetItem()
                chk_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                chk_item.setCheckState(Qt.Checked if self.is_fixed[i] else Qt.Unchecked)
                table.setItem(i, 4, chk_item)
    
            v.addWidget(table)
            
            # Dialog Buttons
            btns = QHBoxLayout()
            btn_reset = QPushButton("Reset to Defaults")
            
            # FIX 2: Actualización segura en memoria sin cerrar la ventana
            def update_table_in_place():
                self._generate_defaults(force_reset=True)
                for i in range(L):
                    table.item(i, 1).setText(str(self.ini[i]))
                    table.item(i, 2).setText(str(self.limi[i]))
                    table.item(i, 3).setText(str(self.lims[i]))
                    table.item(i, 4).setCheckState(Qt.Unchecked)
                    self.is_fixed[i] = False
                    
            btn_reset.clicked.connect(update_table_in_place)
            
            btn_ok = QPushButton("Save & Close")
            btn_ok.clicked.connect(dlg.accept)
            btns.addWidget(btn_reset)
            btns.addWidget(btn_ok)
            v.addLayout(btns)
            
            dlg.setLayout(v)
            
            # If user clicks "Save & Close", update internal values from table
            if dlg.exec_() == QDialog.Accepted:
                for i in range(L):
                    # FIX 3: Captura de errores por si el usuario teclea letras en vez de números
                    try:
                        self.ini[i] = float(table.item(i, 1).text())
                        self.limi[i] = float(table.item(i, 2).text())
                        self.lims[i] = float(table.item(i, 3).text())
                        self.is_fixed[i] = (table.item(i, 4).checkState() == Qt.Checked)
                    except ValueError:
                        pass # Ignora basura silenciosamente y mantiene el valor anterior                
    def _run_least_squares_with_progress(self):
        """Executes the least squares optimization using pure VarPro."""
        TD = self._temp_fit_TD
        WL = self._temp_fit_WL
        numWL = len(WL)
        data_flat = self.data_c.T.flatten()
        
        if not hasattr(self, 'is_fixed') or len(self.is_fixed) != len(self.ini):
            self.is_fixed = np.zeros(len(self.ini), dtype=bool)
        
        num_kin_params = self._get_num_kinetic_params()
        
        # Solo los parámetros NO lineales pueden ser "libres" para el optimizador.
        # Las amplitudes nunca deben entrar aquí, aunque el usuario no las haya
        # marcado como "Fix": VarPro las recalcula internamente en cada evaluación.
        is_kinetic = np.zeros(len(self.ini), dtype=bool)
        is_kinetic[:num_kin_params] = True
        
        free_indices = np.where(is_kinetic & ~self.is_fixed)[0]
        self.free_indices = free_indices  # Se reutiliza en el postproceso (errores)
        
        x0_free = self.ini[free_indices]
        low_free = self.limi[free_indices]
        upp_free = self.lims[free_indices]
        
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Iterating: %v")
        self.iter_count = 0 
    
        def residuals(p_free):
            self.iter_count += 1
            if self.iter_count % 10 == 0:
                val = (self.iter_count // 10) % 101
                self.progress_bar.setValue(val)
                QApplication.processEvents()
            if getattr(self, '_abort_fit', False):
                raise InterruptedError("Fit cancelado manualmente.")
                
            x_full = self.ini.copy()
            x_full[free_indices] = p_free
            
            use_art = getattr(self, 'chk_artifact', None) and self.chk_artifact.isChecked() 
            if self.model_type == "Sequential":
                C = fit.get_concentration_matrix_sequential(x_full, TD, self.numExp, use_art)
            elif self.model_type == 'Damped Oscillation':
                C = fit.get_concentration_matrix_oscillation(x_full, TD, self.numExp, use_art)
            elif self.model_type == "Custom GUI Model":
                model = self.current_custom_model
                w, t0 = x_full[0], x_full[1]
                num_params_cineticos = len(model.param_labels)
                x_nl_params = x_full[2:2+num_params_cineticos]
                C = model.get_concentration_matrix(x_nl_params, TD, w, t0, use_art=use_art)
            else: 
                C = fit.get_concentration_matrix_global(x_full, TD, self.numExp, use_art)
                
            use_nnls = getattr(self, 'chk_nnls', None) and self.chk_nnls.isChecked()
            F, _ = fit.eval_varpro_model(C, self.data_c.T, enforce_nonneg=use_nnls, numExp=self.numExp)
            
            return F.flatten() - data_flat
    
        try:
            res = least_squares(
                fun=residuals, x0=x0_free, bounds=(low_free, upp_free),
                method='trf', x_scale='jac', loss='soft_l1',      
                ftol=1e-8, xtol=1e-8, verbose=0
                )
            self.fit_result = res
            self.fit_x = self.ini.copy()
            self.fit_x[free_indices] = res.x
            
            # --- PUENTE MÁGICO PARA LA GUI ---
            if self.model_type == "Sequential":
                C = fit.get_concentration_matrix_sequential(self.fit_x, TD, self.numExp)
                A_base = 2 + self.numExp
            elif self.model_type == 'Damped Oscillation':
                C = fit.get_concentration_matrix_oscillation(self.fit_x, TD, self.numExp)
                A_base = 2 + self.numExp + 3
            elif self.model_type == "Custom GUI Model":
                model = self.current_custom_model
                w, t0 = self.fit_x[0], self.fit_x[1]
                num_params_cineticos = len(model.param_labels)
                x_nl_params = self.fit_x[2:2+num_params_cineticos]
                C = model.get_concentration_matrix(x_nl_params, TD, w, t0, use_art=False)
                A_base = 2 + num_params_cineticos
            else:
                C = fit.get_concentration_matrix_global(self.fit_x, TD, self.numExp)
                A_base = 2 + self.numExp
                
            _, S_T = fit.eval_varpro_model(C, self.data_c.T)
            self.fit_x[A_base:] = S_T.T.flatten()
            self.progress_bar.setValue(100)
            self.progress_bar.setFormat("Fit Completed")
            
        except Exception as e:
            self.progress_bar.setValue(0)
            raise e
    def _get_num_kinetic_params(self):
        """
        Número de parámetros NO lineales (w, t0, taus / parámetros cinéticos)
        al principio del vector de parámetros. El resto son amplitudes 
        espectrales por longitud de onda, que VarPro resuelve internamente vía 
        mínimos cuadrados lineales en cada evaluación de residuales: nunca deben 
        pasarse al optimizador no lineal como parámetros libres.
        """
        if self.model_type == "Damped Oscillation":
            return 2 + self.numExp + 3
        elif self.model_type == "Custom GUI Model":
            return 2 + len(self.current_custom_model.param_labels)
        else:
            return 2 + self.numExp
           
    def _postprocess_fit_and_save(self):
            """Calculates statistics, extracts spectra with errors, and saves files to the /fit/ directory."""
            if self.fit_result is None: return
    
            x = self.fit_x
            TD = getattr(self, '_temp_fit_TD', self.TD)
            WL = getattr(self, '_temp_fit_WL', self.WL)
            if TD is None or WL is None: return
    
            numWL = len(WL)
            numExp = self.numExp
            use_art = getattr(self, 'chk_artifact', None) and self.chk_artifact.isChecked()
            
            
            if self.model_type == "Sequential":
                C = fit.get_concentration_matrix_sequential(x, TD, numExp, use_art)
            elif self.model_type == 'Damped Oscillation':
                C = fit.get_concentration_matrix_oscillation(x, TD, numExp, use_art)
            elif self.model_type == "Custom GUI Model":
                model = self.current_custom_model
                w, t0 = x[0], x[1]
                num_params_cineticos = len(model.param_labels)
                x_nl_params = x[2:2+num_params_cineticos]
                C = model.get_concentration_matrix(x_nl_params, TD, w, t0, use_art=use_art)
            else:
                C = fit.get_concentration_matrix_global(x, TD, numExp, use_art)
            
            use_nnls = getattr(self, 'chk_nnls', None) and self.chk_nnls.isChecked()
            F_mat, S_T_full = fit.eval_varpro_model(C, self.data_c.T, enforce_nonneg=use_nnls, numExp=numExp) 
            self.S_T_full = S_T_full
            if "Oscillation" in self.model_type:
                self.As = S_T_full[:numExp, :]
                self.Bs = S_T_full[numExp, :]
            else:
                self.As = S_T_full[:numExp, :]
                    


            fitres = F_mat.T 
            resid = self.data_c - fitres
            self.fit_fitres = fitres
            self.fit_resid = resid
    
            L_total = len(x)
            self.ci = np.zeros(L_total)
            self.param_correlation = None
            self.param_correlation_indices = None
            
            try:
                free_indices = getattr(self, 'free_indices', None)
                if free_indices is None or len(free_indices) == 0:
                    # Fallback de seguridad (p.ej. tras cargar un proyecto antiguo)
                    num_kin_fallback = self._get_num_kinetic_params()
                    is_kinetic_fb = np.zeros(L_total, dtype=bool)
                    is_kinetic_fb[:num_kin_fallback] = True
                    free_indices = np.where(is_kinetic_fb & ~self.is_fixed)[0]
            
                J = self.fit_result.jac
            
                if J is not None and J.size > 0 and len(free_indices) > 0:
                    U, s, Vh = np.linalg.svd(J, full_matrices=False)
                    tol = np.finfo(float).eps * max(J.shape) * s[0]
                    s_inv = np.zeros_like(s)
                    s_inv[s > tol] = 1.0 / s[s > tol]
                    cov_free = (Vh.T * (s_inv**2)) @ Vh
            
                    s_nonzero = s[s > tol]
                    if len(s_nonzero) > 0:
                        self.fit_condition_number = s_nonzero[0] / s_nonzero[-1]
                    else:
                        self.fit_condition_number = np.inf
                     
                    # Dirección menos determinada del espacio de parámetros (la "sloppy direction")
                    if len(s_nonzero) > 0:
                        idx_sloppiest = np.argmin(s)
                        direction = Vh[idx_sloppiest, :]  # combinación lineal de los parámetros libres
                        self.sloppiest_direction = dict(zip(free_indices, direction))
    
                    # Nº de parámetros lineales (amplitudes) realmente ajustados por VarPro,
                    # para restarlos también de los grados de libertad.
                    if self.model_type == "Custom GUI Model":
                        num_species = len(self.current_custom_model.states)
                    elif "Oscillation" in self.model_type:
                        num_species = numExp + 1
                    else:
                        num_species = numExp
                    num_linear_params = numWL * num_species
            
                    total_fitted_params = len(free_indices) + num_linear_params
                    dof = resid.size - total_fitted_params
            
                    if dof > 0:
                        mse = np.sum(resid**2) / dof
                        cov_free_scaled = cov_free * mse
                        var_free = np.diagonal(cov_free_scaled)
                        err_free = np.sqrt(np.maximum(var_free, 0))
                        self.ci[free_indices] = err_free
            
                        # Matriz de correlación entre parámetros cinéticos: valores
                        # cercanos a ±1 indican que dos parámetros no son identificables
                        # de forma independiente (p.ej. dos taus muy próximos entre sí).
                        d = np.sqrt(np.maximum(np.diagonal(cov_free_scaled), 1e-300))
                        self.param_correlation = cov_free_scaled / np.outer(d, d)
                        self.param_correlation_indices = free_indices
                        
                    # --- Diagnóstico de identificabilidad: número de condición y dirección menos determinada ---
                    s_nonzero = s[s > tol]
                    if len(s_nonzero) > 0:
                        self.fit_condition_number = float(s_nonzero[0] / s_nonzero[-1])
                        idx_sloppiest = int(np.argmin(s))
                        direction = Vh[idx_sloppiest, :]
                        contributions = sorted(
                            zip(free_indices.tolist(), direction.tolist()),
                            key=lambda t: abs(t[1]), reverse=True
                        )
                        # Solo guardamos contribuciones no despreciables a esa dirección
                        self.sloppiest_direction = [(idx, w) for idx, w in contributions if abs(w) > 0.15]
                    else:
                        self.fit_condition_number = np.inf
                        self.sloppiest_direction = []
                        
            except Exception as e:
                print(f"CRITICAL ERROR calculating covariance: {e}")
                
            idx_tau = 2
            
            if self.model_type == "Custom GUI Model":
                num_kinetic_params = len(self.current_custom_model.param_labels)
                end_tau = idx_tau + num_kinetic_params
            else:
                end_tau = idx_tau + numExp
                
            if end_tau <= len(x):
                self.extracted_taus = x[idx_tau : end_tau]
                self.extracted_errtaus = self.ci[idx_tau : end_tau]
            else:
                self.extracted_taus = np.zeros(end_tau - idx_tau)
                self.extracted_errtaus = np.zeros(end_tau - idx_tau)
    
            self.As = np.zeros((numExp, numWL))
            self.errAs = np.zeros((numExp, numWL))
            self.Bs = None      
            self.errBs = None
            
            try:
            
                pseudo_inv_C = np.linalg.pinv(C.T @ C)
                diag_cov = np.diagonal(pseudo_inv_C) 
                dof_linear = resid.shape[1] - C.shape[1]
                mse_per_wl = np.sum(resid**2, axis=1) / dof_linear if dof_linear > 0 else np.zeros(numWL)
                err_S_T_full = np.sqrt(np.maximum(np.outer(diag_cov, mse_per_wl), 0))
                
                if "Oscillation" in self.model_type:
                    self.As = self.S_T_full[:numExp, :]
                    self.errAs = err_S_T_full[:numExp, :]
                    self.Bs = self.S_T_full[numExp, :]
                    self.errBs = err_S_T_full[numExp, :]
                else:
                    self.As = self.S_T_full[:numExp, :]
                    self.errAs = err_S_T_full[:numExp, :]

            except Exception as e:
                pass
    
            outdir = self.current_batch_outdir if getattr(self, 'is_batch_running', False) else os.path.join(self.base_dir, "fit")
            os.makedirs(outdir, exist_ok=True)
            
            # --- GUARDADO DE ARCHIVOS DE RESULTADOS ---
            
            # 1. Paquete completo de resultados (consumido por _on_active_dataset_changed)
            results_dict = {
                'fitres': self.fit_fitres,
                'resid': self.fit_resid,
                'taus': self.extracted_taus,
                'err_taus': self.extracted_errtaus,
                'As': self.As,
                'errAs': self.errAs,
                'x': x,
                'ci': self.ci,
                'model_type': self.model_type,
                'numExp': numExp,
            }
            np.save(os.path.join(outdir, "GFitResults.npy"), results_dict)
            
            # 2. Ejes
            np.savetxt(os.path.join(outdir, "WL.txt"), WL, fmt='%.6f', header='Wavelength (nm)', comments='')
            np.savetxt(os.path.join(outdir, "TD.txt"), TD, fmt='%.6f', header='Delay (ps)', comments='')
            
            # 3. Amplitudes (DAS/SAS) en el formato que espera SASDASPlotterWindow.parse_spectra_file:
            #    líneas "# tauN=valor+-error" seguidas de columnas Wavelength / AN / AN_err
            tau_comment_lines = []
            for n in range(numExp):
                tau_val = self.extracted_taus[n] if n < len(self.extracted_taus) else np.nan
                err_val = self.extracted_errtaus[n] if (self.extracted_errtaus is not None and n < len(self.extracted_errtaus)) else 0.0
                tau_comment_lines.append(f"tau{n+1}={tau_val:.6g}+-{err_val:.6g}")
            
            col_headers = ["Wavelength"]
            amp_columns = [WL]
            for n in range(numExp):
                col_headers.append(f"A{n+1}")
                col_headers.append(f"A{n+1}_err")
                amp_columns.append(self.As[n])
                amp_columns.append(self.errAs[n])
            
            amp_matrix = np.column_stack(amp_columns)
            
            with open(os.path.join(outdir, "Amplitudes.txt"), 'w') as f:
                for line in tau_comment_lines:
                    f.write(f"# {line}\n")
                f.write("\t".join(col_headers) + "\n")
                np.savetxt(f, amp_matrix, fmt='%.6e', delimiter='\t')
            
            self._update_fit_canvas()
            self._update_resid_canvas()
            self.btn_show_das.setEnabled(True)
            if not getattr(self, 'is_batch_running', False):
                self.show_results_summary()
                rmsd = np.sqrt(np.mean(resid**2))
                QMessageBox.information(self, "Fit Complete", f"Optimization finished successfully.\nRMSD: {rmsd:.2e}")
                
    def compute_profile_likelihood(self, param_idx, n_steps=15, confidence=0.95, span_sigma=6, progress_callback=None):
        """
        Calcula el intervalo de verosimilitud-perfil (profile likelihood) para 
        el parámetro cinético `param_idx`.
    
        A diferencia del error basado en covarianza (que asume que el chi-cuadrado 
        se comporta como una parábola simétrica alrededor del óptimo), este método 
        fija el parámetro en una rejilla de valores y REAJUSTA todos los demás 
        parámetros cinéticos libres en cada punto. El intervalo de confianza es la 
        región donde el chi-cuadrado no empeora más de lo que el azar explicaría 
        al nivel de confianza dado (test de razón de verosimilitudes, 1 g.d.l.).
        """
        from scipy.stats import chi2 as chi2_dist
    
        if self.fit_x is None or self.fit_result is None:
            raise RuntimeError("Ejecuta un fit antes de calcular el perfil de verosimilitud.")
    
        best_val = self.fit_x[param_idx]
        best_err = self.ci[param_idx] if self.ci[param_idx] > 0 else abs(best_val) * 0.1 + 1e-6
        grid = np.linspace(best_val - span_sigma * best_err, best_val + span_sigma * best_err, n_steps)
    
        chi2_values = []
        original_is_fixed = self.is_fixed.copy()
        original_fit_x = self.fit_x.copy()
    
        try:
            for k, val in enumerate(grid):
                if getattr(self, '_abort_fit', False):
                    raise InterruptedError("Identifiability analysis cancelled.")
    
                self.ini = original_fit_x.copy()
                self.ini[param_idx] = val
                self.is_fixed = original_is_fixed.copy()
                self.is_fixed[param_idx] = True
    
                self._run_least_squares_with_progress()
                chi2_values.append(float(np.sum(self.fit_result.fun ** 2)))
    
                if progress_callback is not None:
                    progress_callback(k + 1, len(grid))
        finally:
            # Restauramos siempre el ajuste óptimo original. Soltamos la bandera 
            # de aborto momentáneamente para que esta última re-optimización 
            # pueda completarse sin cortarse a mitad de camino.
            self._abort_fit = False
            self.is_fixed = original_is_fixed
            self.ini = original_fit_x.copy()
            self._run_least_squares_with_progress()
    
        chi2_values = np.array(chi2_values)
        if len(chi2_values) == 0:
            return None
    
        chi2_min = chi2_values.min()
        delta_threshold = chi2_dist.ppf(confidence, df=1)
    
        within = grid[(chi2_values - chi2_min) <= delta_threshold]
        lower_bound = float(within.min()) if within.size > 0 else None
        upper_bound = float(within.max()) if within.size > 0 else None
    
        return {
            'grid': grid, 'chi2': chi2_values,
            'delta_threshold': delta_threshold, 'chi2_min': chi2_min,
            'lower_bound': lower_bound, 'upper_bound': upper_bound,
            'best_value': best_val, 'label': self._get_kinetic_param_label(param_idx),
        }
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
    
        if is_custom and getattr(self, 'current_custom_model', None) is not None:
            model = self.current_custom_model
            if i == 0: return "w (IRF Width)"
            if i == 1: return "t0 (Time Zero)"
            idx_param = i - 2
            if 0 <= idx_param < len(model.param_labels):
                return model.param_labels[idx_param]
            return f"param[{i}]"
    
        if is_oscillation:
            if i == 0: return "w (IRF Width)"
            if i == 1: return "t0 (Time Zero)"
            if i < 2 + self.numExp: return f"τ{i-1}"
            if i == 2 + self.numExp: return "α (Damping)"
            if i == 2 + self.numExp + 1: return "ω (Frequency)"
            if i == 2 + self.numExp + 2: return "φ (Phase)"
            return f"param[{i}]"
    
        if i == 0: return "w (IRF Width)"
        if i == 1: return "t0 (Time Zero)"
        if i < 2 + self.numExp: return f"τ{i-1}"
        return f"param[{i}]"
    
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
            
        
    def plot_das_and_more(self):
                """Opens an external window to display DAS/SAS (Decay/Species Associated Spectra)."""
                if self.As is None: return
                outdir = os.path.join(self.base_dir, "Plots")
                os.makedirs(outdir, exist_ok=True)
                wl = getattr(self, '_wl_proc', self.WL)
                td = getattr(self, '_td_proc', self.TD)
                
                has_oscillation = hasattr(self, 'Bs') and self.Bs is not None
                is_custom = self.model_type == "Custom GUI Model"
                
                fig_das = Figure(figsize=(14, 6) if has_oscillation else (8, 6))
                ax_das = fig_das.add_subplot(121) if has_oscillation else fig_das.add_subplot(111)
                ax_osc = fig_das.add_subplot(122) if has_oscillation else None
                            
                colors = ['b', 'r', 'g', 'orange', 'm', 'c']
                markers = ['o', 's', '^', 'D', 'v', 'p'] 
                
                for n in range(self.numExp):
                    color = colors[n % len(colors)]
                    marker = markers[n % len(markers)]
                    
                    # Asignación de leyenda correcta según el modelo
                    if is_custom:
                        state_name = self.current_custom_model.states[n]
                        lbl = f"Species: {state_name}"
                    else:
                        tau_val = self.extracted_taus[n]
                        err_tau = self.extracted_errtaus[n] if (self.extracted_errtaus is not None and not np.isnan(self.extracted_errtaus[n])) else 0.0
                        lbl = f"$\\tau_{n+1}$ = {tau_val:.2f} ± {err_tau:.2f} ps"
        
                    if self.errAs is not None:
                        err_y = np.nan_to_num(self.errAs[n])
                        ax_das.errorbar(wl, self.As[n], yerr=err_y, label=lbl, color=color, fmt=f'-{marker}', markersize=5, capsize=4)
                    else:
                        ax_das.plot(wl, self.As[n], f'-{marker}', label=lbl, color=color, markersize=5)
                
                ax_das.set_xlabel("Wavelength (nm)")
                if self.model_type in ["Sequential", "Custom GUI Model"]:
                    ax_das.set_ylabel("SAS Amplitude (ΔA)")
                    ax_das.set_title("Species Associated Spectra (SAS)")
                else:
                    ax_das.set_ylabel("DAS Amplitude (ΔA)")
                    ax_das.set_title("Decay Associated Spectra (DAS)")
                
                ax_das.legend(frameon=True)
                ax_das.axhline(0, color='k', linestyle='--', alpha=0.5)
                ax_das.grid(True, linestyle=':', alpha=0.4)
                
                # NUEVO: Caja de texto con los Taus si es modelo visual Custom
                if is_custom:
                    textstr = '\n'.join((
                        r'$\mathbf{Kinetic\ Parameters:}$',
                        *[f"{label} = {val:.2f} ± {err:.2f}" for label, val, err in zip(self.current_custom_model.param_labels, self.extracted_taus, self.extracted_errtaus)]
                    ))
                    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
                    ax_das.text(0.05, 0.95, textstr, transform=ax_das.transAxes, fontsize=10,
                                verticalalignment='top', bbox=props)
        
                if has_oscillation and ax_osc is not None:
                    alpha, omega, phi = self.fit_x[2 + self.numExp : 2 + self.numExp + 3]
                    title_osc = f"Oscillation Spectrum\nDamping α={alpha:.4f} | Freq ω={omega:.4f} | Phase φ={phi:.2f}"
                    ax_osc.plot(wl, self.Bs, color='black', linewidth=2, label='Oscillation Amplitude (B)')
                    if self.errBs is not None:
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
        
                self.das_viewer = PlotViewerWindow(fig_das, title="DAS / SAS Spectra", parent=self)
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
                fig_res.savefig(os.path.join(outdir, "Residuals_Map.png"), dpi=300)
        
                self.trace_viewer = TraceExplorerWindow(self, outdir)
                self.trace_viewer.show()


class SASDASPlotterWindow(QDialog):
    """
    Advanced Drag & Drop Publication-Quality Plotter for SAS/DAS Spectra.
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
        layout = QVBoxLayout(self)
        top_layout = QHBoxLayout()
        
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
        top_layout.addWidget(self.drop_label, 3)
        
        ctrl_group = QGroupBox("Plots")
        ctrl_form = QFormLayout(ctrl_group)
        
        # Selector de paletas cromáticas
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
        ctrl_form.addRow("Colour palette:", self.combo_palette)
        
        # Controles de puntos y errorbars
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
        
        # Dimensiones de la figura en pulgadas
        self.spin_width = QDoubleSpinBox()
        self.spin_width.setRange(3.0, 15.0)
        self.spin_width.setValue(6.5)  
        self.spin_width.setSingleStep(0.5)
        self.spin_width.setSuffix(" in (Width)")
        self.spin_width.valueChanged.connect(self.update_fig_size)
        ctrl_form.addRow("Figure width:", self.spin_width)
        
        self.spin_height = QDoubleSpinBox()
        self.spin_height.setRange(2.0, 10.0)
        self.spin_height.setValue(4.5)  
        self.spin_height.setSingleStep(0.5)
        self.spin_height.setSuffix(" in (Height)")
        self.spin_height.valueChanged.connect(self.update_fig_size)
        ctrl_form.addRow("Figure height:", self.spin_height)
        
        # Herramienta de Crop manual ejes X e Y
        self.chk_auto_axes = QCheckBox("Auto limits axis")
        self.chk_auto_axes.setChecked(True)
        self.chk_auto_axes.stateChanged.connect(self.toggle_axes_inputs)
        ctrl_form.addRow(self.chk_auto_axes)
        
        self.spin_xmin = QSpinBox()
        self.spin_xmin.setRange(200, 1500)
        self.spin_xmin.setValue(300)
        self.spin_xmin.setSuffix(" nm (X Min)")
        self.spin_xmin.setEnabled(False)
        self.spin_xmin.valueChanged.connect(self.replotted)
        ctrl_form.addRow("Crop X Min:", self.spin_xmin)
        
        self.spin_xmax = QSpinBox()
        self.spin_xmax.setRange(200, 1500)
        self.spin_xmax.setValue(800)
        self.spin_xmax.setSuffix(" nm (X Max)")
        self.spin_xmax.setEnabled(False)
        self.spin_xmax.valueChanged.connect(self.replotted)
        ctrl_form.addRow("Crop X Max:", self.spin_xmax)
        
        self.spin_ymin = QDoubleSpinBox()
        self.spin_ymin.setRange(-10.0, 10.0)
        self.spin_ymin.setValue(-0.10)
        self.spin_ymin.setSingleStep(0.01)
        self.spin_ymin.setDecimals(3)
        self.spin_ymin.setEnabled(False)
        self.spin_ymin.valueChanged.connect(self.replotted)
        ctrl_form.addRow("Crop Y Min:", self.spin_ymin)
        
        self.spin_ymax = QDoubleSpinBox()
        self.spin_ymax.setRange(-10.0, 10.0)
        self.spin_ymax.setValue(1.0)
        self.spin_ymax.setSingleStep(0.05)
        self.spin_ymax.setDecimals(3)
        self.spin_ymax.setEnabled(False)
        self.spin_ymax.valueChanged.connect(self.replotted)
        ctrl_form.addRow("Crop Y Max:", self.spin_ymax)
        
        self.btn_clear = QPushButton("Clean plot")
        self.btn_clear.clicked.connect(self.clear_data)
        ctrl_form.addRow(self.btn_clear)
        
        self.btn_export_fig = QPushButton("Export spectra (600 DPI)")
        self.btn_export_fig.clicked.connect(self.export_figure)
        self.btn_export_fig.setStyleSheet("background-color: #4A8C4A; color: white; font-weight: bold;")
        ctrl_form.addRow(self.btn_export_fig)
        
        top_layout.addWidget(ctrl_group, 2)
        layout.addLayout(top_layout)
        
        self.fig = Figure(figsize=(self.spin_width.value(), self.spin_height.value()), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        self.ax = self.fig.add_subplot(111)
        self.setup_paper_style()
        
    def toggle_axes_inputs(self):
        state = not self.chk_auto_axes.isChecked()
        self.spin_xmin.setEnabled(state)
        self.spin_xmax.setEnabled(state)
        self.spin_ymin.setEnabled(state)
        self.spin_ymax.setEnabled(state)
        self.replotted()

    def update_fig_size(self):
        self.fig.set_size_inches(self.spin_width.value(), self.spin_height.value())
        self.canvas.draw_idle() 
        
    def setup_paper_style(self):
        self.ax.clear()
        self.ax.tick_params(direction='in', top=True, right=True, labelsize=11, width=1.2, length=6)
        for spine in self.ax.spines.values():
            spine.set_linewidth(1.2)
        self.ax.set_xlabel("Wavelength / nm", fontsize=13, fontname="Arial", fontweight='bold')
        self.ax.set_ylabel("Amplitude / a.u.", fontsize=13, fontname="Arial", fontweight='bold')
        self.ax.axhline(0, color='#7F7F7F', linestyle='--', linewidth=1.0, zorder=1)
        self.ax.grid(False)
        
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            
    def dropEvent(self, event):
        files_added = 0
        for url in event.mimeData().urls():
            file_path = str(url.toLocalFile())
            if file_path.lower().endswith('.txt'):
                if self.parse_spectra_file(file_path):
                    files_added += 1
        if files_added > 0:
            self.replotted()
            
    def parse_spectra_file(self, path):
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
                
                # --- MODIFICADO: REDONDEO Y VISUALIZACIÓN A 2 DECIMALES (:.2f) ---
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
            print(f"Error procesando espectro: {e}")
            return False
            
    def replotted(self):
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
            
            self.ax.errorbar(wl, amp, yerr=err, fmt='-', marker=marker_style, color=color, 
                             linewidth=2.0, markersize=ms, capsize=cap, elinewidth=1.2, 
                             markeredgewidth=1.0, alpha=0.9, label=data['label'], zorder=4)
            
        if not self.chk_auto_axes.isChecked():
            self.ax.set_xlim(self.spin_xmin.value(), self.spin_xmax.value())
            self.ax.set_ylim(self.spin_ymin.value(), self.spin_ymax.value())
            
        self.ax.legend(frameon=True, framealpha=0.0, edgecolor='none', fontsize=10, loc='best')
        self.fig.tight_layout()
        self.canvas.draw_idle()
        
    def clear_data(self):
        self.spectra_data = []
        self.replotted()
        
    def export_figure(self):
        if not self.spectra_data:
            return
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Publication Figure", "", 
            "PDF Vector Graphic (*.pdf);;PNG High-Resolution Image (*.png);;TIFF Image (*.tiff)"
        )
        if save_path:
            self.update_fig_size()
            self.fig.savefig(save_path, dpi=600, bbox_inches='tight')
            QMessageBox.information(self, "Export Successful", f"Espectro exportado con éxito a 600 DPI.")
