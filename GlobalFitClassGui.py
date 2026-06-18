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
# ---------------------------------------------------------------------
# GUI Styling Constants
# ---------------------------------------------------------------------

BUTTON_STYLE = """
    QPushButton {
        background-color: #6CB66C;   
        color: white;                
        border: 1px solid #549A54;  
        border-radius: 4px;          
        padding: 6px 12px;           
        font-weight: bold;
        font-family: "Segoe UI";
        font-size: 9pt;              
    }
    QPushButton:hover {
        background-color: #5CA55C;   
        color: white;
        border: 1px solid #468446;   
    }
    QPushButton:pressed {
        background-color: #4A8C4A;  
        border: 1px solid #4A8C4A;
        color: white;
        padding-top: 7px;            
        padding-left: 13px;
    }
    QPushButton:disabled {
        background-color: #A0C8A0;   
        border: 1px solid #A0C8A0;
        color: #F0F0F0;              
    }
"""

DARK_THEME_STYLE = """
    QDialog, QWidget {
        color: #222222;            
        font-family: "Segoe UI";
        font-size: 8pt;              
    }
    QGroupBox {
        border: 1px solid #C0C0C0;
        border-radius: 5px;
        margin-top: 8px;             
        padding-top: 10px;
        font-weight: bold;
        color: #000000;            
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 3px;
    }
    QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {
        background-color: #FFFFFF; 
        border: 1px solid #C0C0C0;
        border-radius: 4px;
        color: #000000;            
        padding: 2px;                
        min-height: 18px;            
    }
    QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QLineEdit:focus {
        border: 1px solid #0078D7; 
    }
    QComboBox QAbstractItemView {
        background-color: #FFFFFF;
        color: #000000;
        selection-background-color: #E5F1FB; 
        selection-color: #000000;
        border: 1px solid #C0C0C0;
    }
    QLabel { color: #222222; }
    QCheckBox { color: #222222; }
    
    QProgressBar {
        border: 1px solid #C0C0C0;
        border-radius: 5px;
        text-align: center;
        background-color: #FFFFFF; 
        color: #222222;
        max-height: 15px;            
    }
    QProgressBar::chunk {
        background-color: #6CB66C; 
        border-radius: 3px;
        width: 10px;
        margin: 0.5px;
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
        
        # Generar eje de tiempo suave para el Fit
        td_lin = np.linspace(td.min(), 1.0, 1000)
        td_log = np.geomspace(1.0, max(1.1, td.max()), 1000)
        td_smooth = np.unique(np.concatenate((td_lin, td_log)))
        
        # Evaluar el modelo
        if self.p.t0_choice == 'No' and hasattr(self.p, 'S_T_full'):
            # NUEVO: Reconstrucción fiel usando VarPro y el Artefacto
            use_art = getattr(self.p, 'chk_artifact', None) and self.p.chk_artifact.isChecked()
            
            if self.p.model_type == "Sequential":
                C_smooth = fit.get_concentration_matrix_sequential(self.p.fit_x, td_smooth, self.p.numExp, use_art)
            elif self.p.model_type == 'Damped Oscillation':
                C_smooth = fit.get_concentration_matrix_oscillation(self.p.fit_x, td_smooth, self.p.numExp, use_art)
            else:
                C_smooth = fit.get_concentration_matrix_global(self.p.fit_x, td_smooth, self.p.numExp, use_art)
                
            # Multiplicamos la cinética suave por TODAS las amplitudes de esta longitud de onda
            y_fit_smooth = C_smooth @ self.p.S_T_full[:, idx]
            
        else:
            # MODO CLÁSICO (Chirp o compatibilidad)
            if self.p.model_type == "Sequential":
                F_mat_smooth = fit.eval_sequential_model(self.p.fit_x, td_smooth, self.p.numExp, len(self.wl_array), self.p.t0_choice)
            elif self.p.model_type == 'Damped Oscillation':
                F_mat_smooth = fit.eval_oscillation_model(self.p.fit_x, td_smooth, self.p.numExp, len(self.wl_array), self.p.t0_choice)
            else:
                F_mat_smooth = fit.eval_global_model(self.p.fit_x, td_smooth, self.p.numExp, len(self.wl_array), self.p.t0_choice)
                
            y_fit_smooth = F_mat_smooth.T[idx, :]

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
            if self.p.t0_choice == 'No' and hasattr(self.p, 'S_T_full'):
                use_art = getattr(self.p, 'chk_artifact', None) and self.p.chk_artifact.isChecked()
                
                if self.p.model_type == "Sequential":
                    C_exp = fit.get_concentration_matrix_sequential(self.p.fit_x, td, self.p.numExp, use_art)
                elif self.p.model_type == 'Damped Oscillation':
                    C_exp = fit.get_concentration_matrix_oscillation(self.p.fit_x, td, self.p.numExp, use_art)
                else:
                    C_exp = fit.get_concentration_matrix_global(self.p.fit_x, td, self.p.numExp, use_art)
                    
                y_fit_exp = C_exp @ self.p.S_T_full[:, idx]
                
            else:
                if self.p.model_type == "Sequential":
                    F_mat_exp = fit.eval_sequential_model(self.p.fit_x, td, self.p.numExp, len(self.wl_array), self.p.t0_choice)
                elif self.p.model_type == 'Damped Oscillation':
                    F_mat_exp = fit.eval_oscillation_model(self.p.fit_x, td, self.p.numExp, len(self.wl_array), self.p.t0_choice)
                else:
                    F_mat_exp = fit.eval_global_model(self.p.fit_x, td, self.p.numExp, len(self.wl_array), self.p.t0_choice)
                    
                y_fit_exp = F_mat_exp.T[idx, :]
                
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
    Funciona de forma 100% autónoma. Permite al usuario personalizar 
    las dimensiones, paletas y cropear el Eje Y (ΔA) con precisión milimétrica.
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
        
        self.drop_label = QLabel("📥 ARRASTRA AQUÍ TUS .TXT DE CINÉTICAS (CON O SIN AJUSTE)")
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
        
        ctrl_group = QGroupBox("Estética de Publicación")
        ctrl_form = QFormLayout(ctrl_group)
        
        # Selector de paleta de colores
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
        ctrl_form.addRow("Paleta de Colores:", self.combo_palette)
        
        # Controles de dimensiones de la figura
        self.spin_width = QDoubleSpinBox()
        self.spin_width.setRange(3.0, 15.0)
        self.spin_width.setValue(7.0)  
        self.spin_width.setSingleStep(0.5)
        self.spin_width.setSuffix(" in (Ancho)")
        self.spin_width.valueChanged.connect(self.update_fig_size)
        ctrl_form.addRow("Ancho Figura:", self.spin_width)
        
        self.spin_height = QDoubleSpinBox()
        self.spin_height.setRange(2.0, 10.0)
        self.spin_height.setValue(4.5)  
        self.spin_height.setSingleStep(0.5)
        self.spin_height.setSuffix(" in (Alto)")
        self.spin_height.valueChanged.connect(self.update_fig_size)
        ctrl_form.addRow("Alto Figura:", self.spin_height)
        
        self.combo_scale = QComboBox()
        self.combo_scale.addItems(["Linear", "SymLog"])
        self.combo_scale.currentIndexChanged.connect(self.replotted)
        ctrl_form.addRow("Escala Eje X:", self.combo_scale)
        
        self.spin_thresh = QDoubleSpinBox()
        self.spin_thresh.setRange(0.01, 10.0)
        self.spin_thresh.setValue(1.0)
        self.spin_thresh.setSingleStep(0.5)
        self.spin_thresh.valueChanged.connect(self.replotted)
        ctrl_form.addRow("Linthresh (ps):", self.spin_thresh)
        
        self.chk_norm = QCheckBox("Normalizar amplitudes individuales")
        self.chk_norm.stateChanged.connect(self.replotted)
        ctrl_form.addRow(self.chk_norm)
        
        self.chk_no_negatives = QCheckBox("Omitir visualmente tiempos < 0")
        self.chk_no_negatives.setChecked(True) 
        self.chk_no_negatives.stateChanged.connect(self.replotted)
        ctrl_form.addRow(self.chk_no_negatives)

        # --- NUEVO: HERRAMIENTA MANUAL DE CROP PARA EL EJE Y (ΔA) ---
        self.chk_auto_y = QCheckBox("Eje Y (ΔA) Automático")
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
        ctrl_form.addRow("Crop Y Min:", self.spin_ymin)
        
        self.spin_ymax = QDoubleSpinBox()
        self.spin_ymax.setRange(-10.0, 10.0)
        self.spin_ymax.setValue(0.20)
        self.spin_ymax.setSingleStep(0.005)
        self.spin_ymax.setDecimals(3)
        self.spin_ymax.setEnabled(False)
        self.spin_ymax.valueChanged.connect(self.replotted)
        ctrl_form.addRow("Crop Y Max:", self.spin_ymax)
        # -------------------------------------------------------------
        
        self.btn_clear = QPushButton("🗑️ Limpiar Gráfico")
        self.btn_clear.clicked.connect(self.clear_data)
        ctrl_form.addRow(self.btn_clear)
        
        self.btn_export_fig = QPushButton("💾 Exportar Figura (600 DPI)")
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
        self.t0_choice = 'No'
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
        self.t0s = None
        self.errt0s = None
        self.errtaus = None
        self.ini = None
        self.limi = None
        self.lims = None

        # --- 3. MAIN LAYOUT DESIGN ---
        main_layout = QHBoxLayout() 
        
        # --- A. Left Panel (Sidebar) CON SCROLL ---
        # 1. Creamos el área de Scroll
        self.sidebar_scroll = QScrollArea()
        self.sidebar_scroll.setFixedWidth(360) # Un poco más ancho para dejar espacio a la barra
        self.sidebar_scroll.setWidgetResizable(True) # Esto hace que se adapte al tamaño
        self.sidebar_scroll.setFrameShape(QScrollArea.NoFrame) # Sin borde feo
        
        # 2. Creamos el Widget contenedor que irá DENTRO del scroll
        self.sidebar_widget = QWidget()
        self.sidebar_layout = QVBoxLayout(self.sidebar_widget)
        self.sidebar_layout.setContentsMargins(5, 5, 5, 5)
        
        # 3. Metemos el widget en el scroll y el scroll en el layout principal
        self.sidebar_scroll.setWidget(self.sidebar_widget)
        
        self._init_sidebar_ui() # Esto llenará el self.sidebar_layout como siempre
        
        main_layout.addWidget(self.sidebar_scroll)
    
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
        """Sets up all the widgets of the left panel (controls and settings)."""
        l = self.sidebar_layout
        
        # --- Group 1: Data Loading ---
        gb_load = QGroupBox("1. Data Source")
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
        self.combo_active_dataset = QComboBox()
        self.combo_active_dataset.setView(QListView())
        
        self.combo_active_dataset.currentIndexChanged.connect(self._on_active_dataset_changed)
        v_load.addWidget(self.combo_active_dataset)
        
        gb_load.setLayout(v_load)
        l.addWidget(gb_load)

        # --- Group 2: Pre-processing ---
        gb_prep = QGroupBox("2. Pre-processing")
        form_prep = QFormLayout()

        # Baseline
        self.spin_bl = QSpinBox()
        self.spin_bl.setRange(0, 500)
        self.spin_bl.setValue(0)
        self.spin_bl.valueChanged.connect(lambda val: self._preview_data_processing())
        form_prep.addRow("Baseline Pts:", self.spin_bl)

        # WL Ranges
        self.spin_wl_min = QDoubleSpinBox(); self.spin_wl_min.setRange(0, 10000); 
        self.spin_wl_max = QDoubleSpinBox(); self.spin_wl_max.setRange(0, 10000); 
        self.spin_wl_max.setDecimals(6)    
        self.spin_wl_max.setSingleStep(0.5)
        self.spin_wl_min.setDecimals(6)
        self.spin_wl_min.setSingleStep(0.1)
        
        form_prep.addRow("Min WL (nm):", self.spin_wl_min)
        form_prep.addRow("Max WL (nm):", self.spin_wl_max)

        self.line_exclude = QLineEdit()
        self.line_exclude.setPlaceholderText("e.g. 490-540, 600-615")
        
        self.line_exclude.editingFinished.connect(self._preview_data_processing)
        
        form_prep.addRow("Exclude WLs:", self.line_exclude)
    
        # Time Ranges
        self.spin_t_min = QDoubleSpinBox(); self.spin_t_min.setRange(-100, 1e6); self.spin_t_min.setDecimals(3)
        self.spin_t_max = QDoubleSpinBox(); self.spin_t_max.setRange(-100, 1e6); self.spin_t_max.setDecimals(3)
        form_prep.addRow("Min Time (ps):", self.spin_t_min)
        form_prep.addRow("Max Time (ps):", self.spin_t_max)
        
        # Binning
        self.spin_bin = QSpinBox()
        self.spin_bin.setRange(1, 50)
        self.spin_bin.setValue(1)
        form_prep.addRow("Binning:", self.spin_bin)
        
        self.chk_zero_neg = QCheckBox("Set t < 0 to zero (background)")
        self.chk_zero_neg.setChecked(False) 
        form_prep.addRow(self.chk_zero_neg)
        
        # ---  Botón de Normalización ---
        self.chk_norm_data = QCheckBox("Normalize Data Matrix (Max |ΔA| = 1)")
        self.chk_norm_data.setChecked(False)
        
        # Conectamos el clic directamente al motor de pre-procesamiento
        self.chk_norm_data.stateChanged.connect(lambda state: self._preview_data_processing())
        
        form_prep.addRow(self.chk_norm_data)
        # -------------------------------------------
        # Preview Button
        self.btn_preview = QPushButton("Apply and Preview")
        self.btn_preview.clicked.connect(self._preview_data_processing) 
        form_prep.addRow(self.btn_preview)

        gb_prep.setLayout(form_prep)
        l.addWidget(gb_prep)

        # --- Group 3: Model Settings ---
        gb_model = QGroupBox("3. Model Settings")
        form_model = QFormLayout()
        
        self.btn_svd = QPushButton("Run SVD Analysis")
        self.btn_svd.clicked.connect(self.run_svd)
        form_model.addRow(self.btn_svd)
        
        gb_vis = QGroupBox("4. Visualization")
        form_vis = QFormLayout()
        
        self.btn_plot_3d = QPushButton("3D Map")
        self.btn_plot_3d.clicked.connect(self.plot_3d_surface)
        form_vis.addRow(self.btn_plot_3d)
        
        self.combo_scale = QComboBox()
        self.combo_scale.setView(QListView())
        self.combo_scale.setMaxVisibleItems(10)
        self.combo_scale.addItems(["Linear", "SymLog"])
        self.combo_scale.currentTextChanged.connect(self._on_scale_changed) # Connect function
        form_vis.addRow("Time Axis Scale:", self.combo_scale)
        
        gb_vis.setLayout(form_vis)
        l.addWidget(gb_vis)
        
        # Num Exponentials
        self.spin_numExp = QSpinBox()
        self.spin_numExp.setRange(1, 6)
        self.spin_numExp.setValue(2)
        form_model.addRow("Components:", self.spin_numExp)

        # Model type
        self.combo_model = QComboBox()
        self.combo_model.setView(QListView())
        self.combo_model.setMaxVisibleItems(10)
        
        self.combo_model.addItems(["Parallel (DAS)", "Sequential (SAS)", "Damped Oscillation"])
        
        form_model.addRow("Model Type:", self.combo_model)

        # Technique
        self.combo_tech = QComboBox()
        self.combo_tech.setView(QListView())
        self.combo_tech.setMaxVisibleItems(10)
        self.combo_tech.addItems(["FLUPS", "TAS", "TCSPC"])
        form_model.addRow("Technique:", self.combo_tech)

        # Chirp
        self.chk_chirp = QCheckBox("Fit Independent t0 (Chirp)")
        form_model.addRow(self.chk_chirp)
        
        self.chk_artifact = QCheckBox("Model Coherent Artifact (XPM/Raman)")
        form_model.addRow(self.chk_artifact)
        
        # ---Checkbox para Forzar Positivos ---
        self.chk_nnls = QCheckBox("Force Positive Spectra (NNLS)")
        form_model.addRow(self.chk_nnls)
        
        # Initial Guesses
        self.btn_edit_guess = QPushButton("Edit Initial Guesses")
        self.btn_edit_guess.clicked.connect(self._open_guess_editor_and_update)
        form_model.addRow(self.btn_edit_guess)
        gb_model.setLayout(form_model)
        l.addWidget(gb_model)

        self.btn_run = QPushButton("RUN FIT")
        self.btn_preview = QPushButton("Apply and Preview")
        self.btn_run.setFixedHeight(40)  
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self.run_fit_pipeline) 
        l.addWidget(self.btn_run)
        
        self.btn_batch = QPushButton("RUN BATCH FIT (All Files)")
        self.btn_batch.setFixedHeight(40)
        self.btn_batch.setEnabled(False)
        self.btn_batch.setStyleSheet("background-color: #0078D7; color: white; font-weight: bold;") # Azul para diferenciarlo
        self.btn_batch.clicked.connect(self.run_batch_pipeline)
        l.addWidget(self.btn_batch)
        
        self.btn_show_das = QPushButton("Show Plots / Results")
        self.btn_show_das.setEnabled(False)
        self.btn_show_das.clicked.connect(self.plot_das_and_more) 
        l.addWidget(self.btn_show_das)

        l.addStretch()
        
        l.addSpacing(15)
        self.btn_standalone_plotter = QPushButton("📊 Open Paper Plotter (Saved Traces)")
        self.btn_standalone_plotter.clicked.connect(self.open_standalone_plotter)
        self.btn_standalone_plotter.setStyleSheet("""
            QPushButton {
                background-color: #3C5488; 
                color: white; 
                font-weight: bold; 
                padding: 10px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #4A66A0; }
        """)
        l.addWidget(self.btn_standalone_plotter)     
        
        self.btn_sasdas_plotter = QPushButton("Open SAS/DAS Plotter (Spectra)")
        self.btn_sasdas_plotter.clicked.connect(self.open_sasdas_plotter)
        self.btn_sasdas_plotter.setStyleSheet("background-color: #00A087; color: white; font-weight: bold; padding: 8px; border-radius: 4px;")
        l.addWidget(self.btn_sasdas_plotter)
        
    def open_standalone_plotter(self):
        """Lanza el módulo de gráficos de publicación de forma 100% independiente."""
        self.standalone_plotter = PaperPlotterWindow(self)
        self.standalone_plotter.show()    
        
    def open_sasdas_plotter(self):
        """Lanza el módulo de maquetación de espectros SAS/DAS de forma autónoma."""
        self.sasdas_plotter = SASDASPlotterWindow(self)
        self.sasdas_plotter.show()
        
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

        Args:
            tab_widget (QWidget): The tab container.

        Returns:
            tuple: (canvas, (ax1, ax2))
        """
        fig = plt.Figure(figsize=(5, 8))
        # ax1: Scree Plot, ax2: First spectral components
        ax1 = fig.add_subplot(211) 
        ax2 = fig.add_subplot(212)
        canvas = FigureCanvas(fig)
        
        layout = QVBoxLayout()
        layout.addWidget(canvas)
        tab_widget.setLayout(layout)
        return canvas, (ax1, ax2)

    def _plot_svd_results(self):
        """Plots the singular values (Scree Plot) and principal spectral components."""
        ax1, ax2 = self.ax_svd
        ax1.clear()
        ax2.clear()
    
        # --- Plot 1: Scree Plot (Log scale) ---
        n_comp = min(len(self.svd_s), 10) # See top 10
        ax1.semilogy(range(1, n_comp + 1), self.svd_s[:n_comp], 'o-', color='red')
        ax1.set_title("Singular Values (Scree Plot)")
        ax1.set_ylabel("Eigenvalue (log)")
        ax1.set_xlabel("Component Number")
        ax1.grid(True, which="both", ls="-", alpha=0.2)
    
        # 2. Spectral components
        wl = getattr(self, '_wl_proc', self.WL)
       
        n_mostrar = self.spin_numExp.value() 
        
        for i in range(min(n_mostrar, len(self.svd_s))):
            ax2.plot(wl, self.svd_U[:, i], label=f"Comp {i+1}")
        
        ax2.set_title(f"First {n_mostrar} Spectral Components")
        ax2.set_xlabel("Wavelength (nm)")
        ax2.axhline(0, color='black', lw=1, alpha=0.5)
        ax2.legend(frameon=True)
        
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

    def _init_plots_ui(self):
        """Builds the right side widgets comprising the tabbed plotting areas."""
        l = self.right_layout
        
        self.lbl_cursor = QLabel("Cursor: Out of the 2D map")
        self.lbl_cursor.setStyleSheet("font-weight: bold; color: #0078D7; font-size: 10pt;")
        l.addWidget(self.lbl_cursor)            
        # Tabs
        self.tabs = QTabWidget()
        
        self.tabs.setStyleSheet("""
                    QTabWidget::pane { 
                        border: 1px solid #999; 
                        background: white; 
                    }
                    QTabBar::tab { 
                        background: #e0e0e0; 
                        color: black;
                        padding: 8px 20px; 
                        border: 1px solid #bbb; 
                        border-bottom: none; 
                        border-top-left-radius: 4px; 
                        border-top-right-radius: 4px; 
                        margin-right: 2px;
                    }
                    QTabBar::tab:selected { 
                        background: #ffffff;                         
                        border-bottom: 1px solid #ffffff; 
                    }
                    QTabBar::tab:hover {
                        background: #d0d0d0;
                    }
                """)
        
        self.tab_exp = QWidget()
        self.tab_fit = QWidget()
        self.tab_resid = QWidget()
        self.tab_svd = QWidget() 
        
        self.tabs.addTab(self.tab_exp, "Experimental")
        self.tabs.addTab(self.tab_fit, "Fit Reconstructed")
        self.tabs.addTab(self.tab_resid, "Residuals")
        self.tabs.addTab(self.tab_svd, "SVD Diagnosis")
        
        # Create Canvas
        self.canvas_exp, self.ax_exp = self._create_canvas_for_tab(self.tab_exp)
        self.canvas_fit, self.ax_fit = self._create_canvas_for_tab(self.tab_fit)
        self.canvas_resid, self.ax_resid = self._create_canvas_for_tab(self.tab_resid)
        self.canvas_svd, self.ax_svd = self._create_svd_canvas(self.tab_svd)
        
        l.addWidget(self.tabs)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        l.addWidget(self.progress_bar)
        
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
        
    def _generate_defaults(self):
        """
        Generates the initial parameter guesses based on the chosen experimental technique and model.
        
        Returns:
            bool: True if defaults were generated successfully, False otherwise.
        """
        
        numExp = self.spin_numExp.value()
        t0_choice = 'Yes' if self.chk_chirp.isChecked() else 'No'
        tech = self.combo_tech.currentText()
        model_str = self.combo_model.currentText()
        
        is_oscillation = "Oscillation" in model_str

        
        if self.data_c is not None:
            numWL = self.data_c.shape[0]
        elif self.WL is not None:
            numWL = len(self.WL)
        else:
            QMessageBox.warning(self, "Warning", "Load data first to generate guesses.")
            return False
    

        if is_oscillation:
            L = (2 + numExp + 3) + numWL * (numExp + 1)
        elif t0_choice == 'Yes':
            L = 1 + numExp + numWL*(numExp+1)
        else:
            L = 2 + numExp + numWL*numExp
            
        self.ini = np.zeros(L)
        self.limi = -np.inf * np.ones(L)
        self.lims = np.inf * np.ones(L)
    
        taus_defaults = [0.5, 5.0, 50.0, 500.0, 2000.0, 5000.0]
        w_guess = 0.15 if tech == 'TAS' else (0.3 if tech == 'FLUPS' else 0.1)

        if is_oscillation:
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
             start_local = idx_osc + 3
             val_A = 1000.0 if tech == 'TCSPC' else (5.0 if tech == 'FLUPS' else 0.01)
             self.ini[start_local:] = val_A 
             
        elif t0_choice == 'No':
            self.ini[0] = w_guess; self.limi[0] = 0.05; self.lims[0] = 2.0
            self.ini[1] = 0.0;     self.limi[1] = -5.0; self.lims[1] = 5.0
            base_tau = 2
            for n in range(numExp):
                self.ini[base_tau + n] = taus_defaults[n] if n < len(taus_defaults) else 1000.0*(n+1)
                self.limi[base_tau + n] = 0.001; self.lims[base_tau + n] = 1e8
            start_A = base_tau + numExp
            val_A = 1000.0 if tech == 'TCSPC' else (5.0 if tech == 'FLUPS' else 0.01)
            self.ini[start_A:] = val_A
            
        else:
            self.ini[0] = w_guess; self.limi[0] = 0.05; self.lims[0] = 2.0
            for n in range(numExp):
                self.ini[1+n] = taus_defaults[n] if n < len(taus_defaults) else 100.0
                self.limi[1+n] = 0.001; self.lims[1+n] = 1e8
            base_idx = 1 + numExp
            params_per_wl = 1 + numExp
            val_A = 1000.0 if tech == 'TCSPC' else 0.1
            self.ini[base_idx:] = val_A
            self.ini[base_idx::params_per_wl] = 0.0
            self.limi[base_idx::params_per_wl] = -5.0
            self.lims[base_idx::params_per_wl] = 5.0
            
        return True

    def _create_canvas_for_tab(self, tab_widget):
        """
        Helper method to initialize a matplotlib canvas inside a specific tab.

        Args:
            tab_widget (QWidget): The tab container.

        Returns:
            tuple: (canvas, ax)
        """
        fig = plt.Figure(figsize=(5,4))
        ax = fig.add_subplot(111)
        canvas = FigureCanvas(fig)
        
        layout = QVBoxLayout()
        layout.addWidget(canvas)
        tab_widget.setLayout(layout)
        
        return canvas, ax

    # --- Auxiliary methods to improve the user experience ---
        
    def update_from_parent(self):
        """Updates internal data from the parent application if it exists."""
        p = self.parent_app
        if p is None: return
             
        if getattr(p, "is_TAS_mode", False):
             if hasattr(p, "data_corrected") and p.data_corrected is not None:
                 incoming_data = np.array(p.data_corrected, copy=True)
         
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
        
        self.base_dir = os.path.dirname(file_paths[0])
        
        if not file_paths:
            return
            
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

    def _clear_fit_plots(self):
        """Limpia los lienzos de Fit y Residuales si el archivo actual no está ajustado."""
        self.fit_fitres = None
        self.fit_resid = None
        self.As = None
        self.ax_fit.clear()
        self.ax_resid.clear()
        self._clear_colorbar_if_exists(self.cbar_fit)
        self._clear_colorbar_if_exists(self.cbar_resid)
        self.canvas_fit.draw_idle()
        self.canvas_resid.draw_idle()
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
            """
            Plots the experimental map on the first tab canvas, 
            with support for dynamic scaling and Linear/SymLog axes.

            Args:
                use_processed (bool, optional): If True, plots processed data. Defaults to False.
            """
            if self.data_c is None: return
            
            self.ax_exp.clear()
            self._clear_colorbar_if_exists(self.cbar_exp)
            
            # Select which axes to use based on data processing state
            if use_processed and hasattr(self, '_wl_proc'):
                Xs = self._wl_proc
                Ys = self._td_proc
                Title = "Experimental (Processed)"
            else:
                Xs = self.WL
                Ys = self.TD
                Title = "Experimental (Raw)"
                
            # Axis protection: Fallback to index-based axes if shapes mismatch
            if Xs.shape[0] != self.data_c.shape[0] or Ys.shape[0] != self.data_c.shape[1]:
                Xs = np.arange(self.data_c.shape[0])
                Ys = np.arange(self.data_c.shape[1])
    
        
            try:
                # Reemplaza la línea antigua de np.percentile por esto:
                # Usamos el mínimo y máximo real para que no se corte la escala al normalizar
                self.exp_vmin = np.nanmin(self.data_c)
                self.exp_vmax = np.nanmax(self.data_c)
                
                # Render the 2D map
                self.pcm_exp = self.ax_exp.pcolormesh(Xs, Ys, self.data_c.T, 
                                                      shading="auto", cmap='jet', 
                                                      vmin=self.exp_vmin, vmax=self.exp_vmax)
                            
                # Set axis labels
                self.ax_exp.set_xlabel("Wavelength (nm)")
                self.ax_exp.set_ylabel("Delay (ps)")
                
                # --- APPLY Y-AXIS SCALE (CONDITIONAL) ---
                if hasattr(self, 'yscale') and self.yscale == 'symlog':
                    self.ax_exp.set_yscale('symlog', linthresh=2)
                else:
                    self.ax_exp.set_yscale('linear')
                
                # Colorbar management using Axes Divider for proper alignment
                divider = make_axes_locatable(self.ax_exp)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                self.cbar_exp = self.canvas_exp.figure.colorbar(self.pcm_exp, cax=cax, label='$\Delta A$ / -')
                
                self.canvas_exp.draw_idle()
                
                # Initialize interactive cursor (white dashed lines)
                self.cursor_exp = Cursor(self.ax_exp, useblit=True, color='white', linewidth=1, linestyle='--')
                
                # Connect mouse motion event to the handler if not already connected
                if not hasattr(self, 'cid_mouse_move'):
                    self.cid_mouse_move = self.canvas_exp.mpl_connect('motion_notify_event', self.on_mouse_move)
                
            except Exception as e:
                print(f"Plotting error: {e}")
                
    def on_mouse_move(self, event):
        """Updates the status label with real-time mouse coordinates and data values on the map."""
        # Ensure the mouse is within the main plot area
        if event.inaxes == self.ax_exp:
            x = event.xdata
            y = event.ydata
            
            if x is None or y is None:
                return
                
            # Attempt to retrieve the specific ΔA value at the cursor position
            if self.data_c is not None and hasattr(self, '_wl_proc') and hasattr(self, '_td_proc'):
                try:
                    # Find the closest indices using absolute difference (nearest neighbor)
                    idx_wl = (np.abs(self._wl_proc - x)).argmin()
                    idx_td = (np.abs(self._td_proc - y)).argmin()
                    z_val = self.data_c[idx_wl, idx_td]
                    
                    self.lbl_cursor.setText(f"Cursor: λ = {x:.1f} nm  |  Delay = {y:.3f} ps  |  ΔA = {z_val:.3e}")
                except Exception:
                    # Fallback if there is a temporary mismatch in array indexing
                    self.lbl_cursor.setText(f"Cursor: λ = {x:.1f} nm  |  Delay = {y:.3f} ps")
            else:
                self.lbl_cursor.setText(f"Cursor: λ = {x:.1f} nm  |  Delay = {y:.3f} ps")
        else:
            # Clear or notify when the cursor leaves the plot boundaries
            self.lbl_cursor.setText("Cursor: Out of the 2D MAP")
            
    def _update_fit_canvas(self):
            """Plots the reconstructed fit map onto the appropriate tab canvas."""
            if self.fit_fitres is None: return
    
            self.ax_fit.clear()
            self._clear_colorbar_if_exists(self.cbar_fit)
            
            # Use processed axes if available, fallback to raw WL/TD
            Xs = getattr(self, '_wl_proc', self.WL)
            Ys = getattr(self, '_td_proc', self.TD)
            Z = self.fit_fitres.T 
    
            # Dimension safety check: Fallback to index-based ranges if shapes don't match data
            if Xs is None or Xs.shape[0] != Z.shape[1]: Xs = np.arange(Z.shape[1])
            if Ys is None or Ys.shape[0] != Z.shape[0]: Ys = np.arange(Z.shape[0])
    
            try:
                if Z.shape[0] < 2 or Z.shape[1] < 2: return
    
                # Forzamos los mismos límites de color (Z) que el Experimental
                vmin = getattr(self, 'exp_vmin', np.nanmin(Z))
                vmax = getattr(self, 'exp_vmax', np.nanmax(Z))
    
                self.pcm_fit = self.ax_fit.pcolormesh(Xs, Ys, Z, shading='auto', cmap='jet', 
                                                      vmin=vmin, vmax=vmax)
                
                self.ax_fit.set_title("Fit Reconstructed")
                self.ax_fit.set_xlabel("Wavelength (nm)")
                self.ax_fit.set_ylabel("Delay (ps)")
                
                # --- APPLY Y-AXIS SCALE (CONDITIONAL) ---
                if hasattr(self, 'yscale') and self.yscale == 'symlog':
                    self.ax_fit.set_yscale('symlog', linthresh=2)
                else:
                    self.ax_fit.set_yscale('linear')
                    
                # Sincronizar límites del eje Y exactamente con el Experimental
                if hasattr(self, 'ax_exp'):
                    self.ax_fit.set_ylim(self.ax_exp.get_ylim())
                    self.ax_fit.set_xlim(self.ax_exp.get_xlim())
                
                # Align colorbar using Axes Divider to match the plot height
                divider = make_axes_locatable(self.ax_fit)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                self.cbar_fit = self.canvas_fit.figure.colorbar(self.pcm_fit, cax=cax, label='Transient absorption / -')
                self.canvas_fit.draw()
            except Exception as e:
                print(f"Error painting Fit: {e}")
        
    def _update_resid_canvas(self):
            """Plots the residuals map (Difference between Experimental and Fit) onto the canvas."""
            if self.fit_resid is None: return
    
            self.ax_resid.clear()
            self._clear_colorbar_if_exists(self.cbar_resid)
            
            # Select processed axes if available; otherwise, use raw data axes
            Xs = getattr(self, '_wl_proc', self.WL)
            Ys = getattr(self, '_td_proc', self.TD)
            Z = self.fit_resid.T
    
            # Axis consistency check: Revert to indices if coordinates mismatch data shape
            if Xs is None or Xs.shape[0] != Z.shape[1]: Xs = np.arange(Z.shape[1])
            if Ys is None or Ys.shape[0] != Z.shape[0]: Ys = np.arange(Z.shape[0])
    
            try:
                # Ensure the data array is valid for 2D plotting
                if Z.shape[0] < 2 or Z.shape[1] < 2: return
                
                # Dynamic contrast adjustment using the 1st and 99th percentiles
                vals = Z.flatten()
                vmin = np.percentile(vals, 1)
                vmax = np.percentile(vals, 99)
    
                self.pcm_resid = self.ax_resid.pcolormesh(Xs, Ys, Z, shading='auto', cmap='jet',
                                                          vmin=vmin, vmax=vmax)
                
                self.ax_resid.set_title("Residuals")
                self.ax_resid.set_xlabel("Wavelength (nm)")
                self.ax_resid.set_ylabel("Delay (ps)")
                
                # --- APPLY Y-AXIS SCALE (CONDITIONAL) ---
                if hasattr(self, 'yscale') and self.yscale == 'symlog':
                    self.ax_resid.set_yscale('symlog', linthresh=2)
                    
                else:
                    self.ax_resid.set_yscale('linear')
                if hasattr(self, 'ax_exp'):
                    self.ax_resid.set_ylim(self.ax_exp.get_ylim()) 
                    self.ax_resid.set_xlim(self.ax_exp.get_xlim()) 
                # Create and align colorbar for the residuals plot
                divider = make_axes_locatable(self.ax_resid)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                self.cbar_resid = self.canvas_resid.figure.colorbar(self.pcm_resid, cax=cax, label='Residual')
                self.canvas_resid.draw()
            except Exception as e:
                print(f"Error painting Resid: {e}")
                
# =============================================================================
# FIT PIPELINE
# =============================================================================
    def run_fit_pipeline(self):
        """Main execution pipeline: Preprocess, set model parameters, and run the optimization."""
        try:
            # 1. Validation: Ensure data is loaded
            if self.data_raw is None:
                QMessageBox.warning(self, "No data", "Load data first.")
                return
            
            # 2. Pre-processing: Apply crops, baseline, and binning
            self._preview_data_processing()
            if self.data_c is None or self.data_c.size == 0: return

            # 3. Parameter setup from UI
            self.numExp = self.spin_numExp.value()
            self.tech = self.combo_tech.currentText()
            self.t0_choice = 'Yes' if self.chk_chirp.isChecked() else 'No'
            
            # Identify Kinetic Model type
            model_str = self.combo_model.currentText()
            if "Sequential" in model_str:
                self.model_type = "Sequential"
            elif "Oscillation" in model_str:
                self.model_type = "Damped Oscillation"
            else:
                self.model_type = "Parallel"

            if self.data_c is not None:
                numWL = self.data_c.shape[0]
            else:
                numWL = 0
            
            # 4. Determine parameter vector length (L_needed) based on model selection
            # This logic ensures the initial guess vector matches the mathematical model

            if self.model_type == "Damped Oscillation":
                # Formula for Oscillation model parameters
                 if self.t0_choice == 'Yes':
                     L_needed = (2 + self.numExp + 3) + numWL * (self.numExp + 1)
                 else:
                     L_needed = (2 + self.numExp + 3) + numWL * (self.numExp + 1)
            elif self.t0_choice == 'Yes': 
                # Formula with chirp/t0 correction
                L_needed = 1 + self.numExp + numWL*(self.numExp+1)
            else:             
                # Standard parallel/sequential formula
                L_needed = 2 + self.numExp + numWL*self.numExp
            # 5. Initialization check: Reset guesses if dimensions have changed
            if self.ini is None or len(self.ini) != L_needed:
                print(f"Size mismatch (Vector: {len(self.ini) if self.ini is not None else 0}, Needed: {L_needed}). Regenerating defaults...")
                self._generate_defaults()
            else:
                print("Using existing guesses.")
                
            # Store current axes for the fitting session
            self._temp_fit_TD = getattr(self, '_td_proc', self.TD)
            self._temp_fit_WL = getattr(self, '_wl_proc', self.WL)
            
            # 6. Execute Optimization and Post-processing
            self._run_least_squares_with_progress()
            self._postprocess_fit_and_save()

        except Exception as e:
            # Error handling with full traceback for debugging
            QMessageBox.critical(self, "Fit Error", str(e))
            import traceback
            traceback.print_exc()
            
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
        self.t0_choice = 'Yes' if self.chk_chirp.isChecked() else 'No'
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
        
        try:
            for idx in range(len(self.data_c_list)):
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
            
    def _open_guess_editor_and_update(self):
        """Opens a dialog to manually edit initial guesses, bounds, and fixed parameters."""
        numExp = self.spin_numExp.value()
        is_chirp = self.chk_chirp.isChecked()
        model_str = self.combo_model.currentText()
        is_oscillation = "Oscillation" in model_str
        
        # Determine number of wavelengths for parameter indexing
        if self.data_c is not None: numWL = self.data_c.shape[0]
        elif self.WL is not None: numWL = len(self.WL)
        else: numWL = 1
            
        # Calculate expected vector length based on the selected model
        if is_oscillation:
            L_needed = 2 + numExp + 3 + numWL * (numExp + 1)
        elif is_chirp:
            L_needed = 1 + numExp + numWL * (numExp + 1)
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
            
            # Labeling logic for Damped Oscillation model
            if is_oscillation:
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
                if not is_chirp:
                    if i == 0: label += "w (FWHM (ps))"
                    elif i == 1: label += "t0 (Time Zero)"
                    elif i < 2 + numExp: label += f"τ{i-1} (Lifetime)"
                    else:
                        local_idx = i - (2 + numExp)
                        wl_idx = local_idx // numExp
                        p_idx = local_idx % numExp
                        label += f"A{p_idx+1} @ WL {wl_idx}"
                else:
                    if i == 0: label += "w (FWHM (ps) /2.355)"
                    elif i < 1 + numExp: label += f"τ{i} (Lifetime)"
                    else: label += "Local (t0 or Amp)"

            # Populate table row
            item_lbl = QTableWidgetItem(label)
            item_lbl.setFlags(item_lbl.flags() ^ Qt.ItemIsEditable)
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
        # Recursive call to reopen with fresh defaults
        btn_reset.clicked.connect(lambda: [self._generate_defaults(), dlg.accept(), self._open_guess_editor_and_update()])
        btn_ok = QPushButton("Save & Close")
        btn_ok.clicked.connect(dlg.accept)
        btns.addWidget(btn_reset); btns.addWidget(btn_ok)
        v.addLayout(btns)
        
        dlg.setLayout(v)
        # If user clicks "Save & Close", update internal values from table
        if dlg.exec_() == QDialog.Accepted:
            for i in range(L):
                self.ini[i] = float(table.item(i, 1).text())
                self.limi[i] = float(table.item(i, 2).text())
                self.lims[i] = float(table.item(i, 3).text())
                self.is_fixed[i] = (table.item(i, 4).checkState() == Qt.Checked)

    def _run_least_squares_with_progress(self):
            """Executes the least squares optimization while updating the UI progress bar."""
            TD = self._temp_fit_TD
            WL = self._temp_fit_WL
            numWL = len(WL)
            data_flat = self.data_c.T.flatten()
            
            # Ensure 'is_fixed' mask exists and matches parameter size
            if not hasattr(self, 'is_fixed') or len(self.is_fixed) != len(self.ini):
                self.is_fixed = np.zeros(len(self.ini), dtype=bool)
                
            # Filter parameters: separate free (optimizable) parameters from fixed ones
            free_indices = np.where(~self.is_fixed)[0]
            x0_free = self.ini[free_indices]
            low_free = self.limi[free_indices]
            upp_free = self.lims[free_indices]
        
            # --- PROGRESS BAR LOGIC ---
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("Iterating: %v")
            self.iter_count = 0 
        
            def residuals(p_free):
                """Objective function for least_squares: returns the difference between model and data."""
                self.iter_count += 1
                
                if self.iter_count % 10 == 0:
                    val = (self.iter_count // 10) % 101
                    self.progress_bar.setValue(val)
                    QApplication.processEvents()
        
                x_full = self.ini.copy()
                x_full[free_indices] = p_free
                
                # ==========================================
                # 1. MOTOR VARPRO (Alta Velocidad y Precisión)
                # ==========================================
                if self.t0_choice == 'No':
                    use_art = self.chk_artifact.isChecked() # <--- LEEMOS LA UI
                    if self.model_type == "Sequential":
                        C = fit.get_concentration_matrix_sequential(x_full, TD, self.numExp, use_art)
                    elif self.model_type == 'Damped Oscillation':
                        C = fit.get_concentration_matrix_oscillation(x_full, TD, self.numExp, use_art)
                    else: # Parallel
                        C = fit.get_concentration_matrix_global(x_full, TD, self.numExp, use_art)
                        
                    use_nnls = getattr(self, 'chk_nnls', None) and self.chk_nnls.isChecked()
                    F, _ = fit.eval_varpro_model(C, self.data_c.T, enforce_nonneg=use_nnls, numExp=self.numExp)
                    
                # ==========================================
                # 2. MOTOR CLÁSICO (Para Chirp / t0 variable)
                # ==========================================
                else:
                    if self.model_type == "Sequential":
                        F = fit.eval_sequential_model(x_full, TD, self.numExp, numWL, self.t0_choice)
                    elif self.model_type == 'Damped Oscillation':
                        F = fit.eval_oscillation_model(x_full, TD, self.numExp, numWL, self.t0_choice)
                    else:
                        F = fit.eval_global_model(x_full, TD, self.numExp, numWL, self.t0_choice)
                
                return F.flatten() - data_flat
        
            try:
                # Ejecutar el optimizador
                res = least_squares(
                    fun=residuals,
                    x0=x0_free,
                    bounds=(low_free, upp_free),
                    method='trf',
                    x_scale='jac',       
                    loss='soft_l1',      
                    ftol=1e-8,           
                    xtol=1e-8,
                    verbose=0
                    )
                # Guardar resultados
                self.fit_result = res
                self.fit_x = self.ini.copy()
                self.fit_x[free_indices] = res.x
                
                # =========================================================
                # EL "PUENTE MÁGICO": Reconstruir Amplitudes para la GUI
                # =========================================================
                if self.t0_choice == 'No':
                    # Recalculamos la matriz C con los parámetros finales óptimos
                    if self.model_type == "Sequential":
                        C = fit.get_concentration_matrix_sequential(self.fit_x, TD, self.numExp)
                    elif self.model_type == 'Damped Oscillation':
                        C = fit.get_concentration_matrix_oscillation(self.fit_x, TD, self.numExp)
                    else:
                        C = fit.get_concentration_matrix_global(self.fit_x, TD, self.numExp)
                        
                    # Extraemos las amplitudes óptimas definitivas (S_T)
                    _, S_T = fit.eval_varpro_model(C, self.data_c.T)
                    
                    # Calculamos dónde empiezan las amplitudes en el vector fit_x
                    A_base = 2 + self.numExp
                    if self.model_type == 'Damped Oscillation':
                        A_base += 3
                        
                    # Inyectamos las amplitudes en formato plano.
                    # Tu código original extraía: all_Amps = x[A_base:].reshape(numWL, numExp).T
                    # La operación inversa exacta es S_T.T.flatten()
                    self.fit_x[A_base:] = S_T.T.flatten()
    
                # Finalizar la barra de progreso
                self.progress_bar.setValue(100)
                self.progress_bar.setFormat("Fit Completed")
                
            except Exception as e:
                self.progress_bar.setValue(0)
                raise e
                
    def _postprocess_fit_and_save(self):
            """Calculates statistics, extracts spectra with errors, and saves files to the /fit/ directory."""
            if self.fit_result is None:
                return
    
            x = self.fit_x
            TD = getattr(self, '_temp_fit_TD', self.TD)
            WL = getattr(self, '_temp_fit_WL', self.WL)
            
            if TD is None or WL is None:
                print("Error: No se encontraron los ejes (TD/WL) del ajuste.")
                return
    
            numWL = len(WL)
            numExp = self.numExp
    
            # --- 1. Reconstruct Fit Matrix and Residuals ---
            use_art = getattr(self, 'chk_artifact', None) and self.chk_artifact.isChecked()
            
            if self.t0_choice == 'No':
                # Reconstrucción 100% exacta usando VarPro
                if self.model_type == "Sequential":
                    C = fit.get_concentration_matrix_sequential(x, TD, numExp, use_art)
                elif self.model_type == 'Damped Oscillation':
                    C = fit.get_concentration_matrix_oscillation(x, TD, numExp, use_art)
                else:
                    C = fit.get_concentration_matrix_global(x, TD, numExp, use_art)
                
                # Extraemos las amplitudes limpias directamente desde aquí
                use_nnls = getattr(self, 'chk_nnls', None) and self.chk_nnls.isChecked()
                F_mat, S_T_full = fit.eval_varpro_model(C, self.data_c.T, enforce_nonneg=use_nnls, numExp=numExp) 
                self.S_T_full = S_T_full
                # Guardamos los DAS reales e ignoramos las amplitudes del artefacto para el plot
                if "Oscillation" in self.model_type:
                    self.As = S_T_full[:numExp, :]
                    self.Bs = S_T_full[numExp, :]
                else:
                    self.As = S_T_full[:numExp, :]
                    
            else:
                # Reconstrucción Clásica (Chirp)
                if self.model_type == "Sequential":
                    F_mat = fit.eval_sequential_model(x, TD, numExp, numWL, self.t0_choice)
                elif self.model_type == 'Damped Oscillation':
                    F_mat = fit.eval_oscillation_model(x, TD, numExp, numWL, self.t0_choice)
                else:
                    F_mat = fit.eval_global_model(x, TD, numExp, numWL, self.t0_choice)

            fitres = F_mat.T if self.t0_choice == 'Yes' else F_mat.T 
            resid = self.data_c - fitres
            
            self.fit_fitres = fitres
            self.fit_resid = resid
    
            # --- 2. Robust Error Calculation (Confidence Intervals) ---
            L_total = len(x)
            self.ci = np.zeros(L_total) # Default to 0
            
            try:
                # Ensure is_fixed mask exists and matches parameter count
                if not hasattr(self, 'is_fixed') or len(self.is_fixed) != L_total:
                    self.is_fixed = np.zeros(L_total, dtype=bool)
                
                free_indices = np.where(~self.is_fixed)[0]
                J = self.fit_result.jac 
                
                # Proceed only if there are free parameters and a valid Jacobian
                if J is not None and J.size > 0 and len(free_indices) > 0:
                    
                    # -------------------------------------------------------------
                    # NUEVO: Cálculo de covarianza robusto mediante SVD directo
                    # -------------------------------------------------------------
                    U, s, Vh = np.linalg.svd(J, full_matrices=False)
                    
                    # Tolerancia dinámica: filtra valores singulares demasiado pequeños
                    # que causarían divisiones por cero o inestabilidad numérica.
                    tol = np.finfo(float).eps * max(J.shape) * s[0]
                    s_inv = np.zeros_like(s)
                    s_inv[s > tol] = 1.0 / s[s > tol]
                    
                    # Matriz de Covarianza: V * (1/S^2) * V^T
                    cov_free = (Vh.T * (s_inv**2)) @ Vh
                    # -------------------------------------------------------------
                    
                    # Degrees of Freedom (DoF)
                    dof = resid.size - len(free_indices)
                    if dof > 0:
                        mse = np.sum(resid**2) / dof
                        # Parameter Variance = Diagonal of Covariance * MSE
                        var_free = np.diagonal(cov_free) * mse
                        
                        # Prevent negative roots due to small numerical precision errors
                        err_free = np.sqrt(np.maximum(var_free, 0))
                        
                        # Map calculated errors back to their respective positions
                        self.ci[free_indices] = err_free
                    else:
                        print("Warning: Degrees of freedom <= 0. Cannot compute errors.")
    
            except Exception as e:             
                print(f"CRITICAL ERROR calculating covariance: {e}")
    
            # --- 3. Extract Lifetimes (Taus) and their Errors ---
            idx_tau = 1 if self.t0_choice == 'Yes' else 2
            
            # Index protection in case of model changes
            end_tau = idx_tau + numExp
            if end_tau <= len(x):
                self.extracted_taus = x[idx_tau : end_tau]
                self.extracted_errtaus = self.ci[idx_tau : end_tau]
            else:
                self.extracted_taus = np.zeros(numExp)
                self.extracted_errtaus = np.zeros(numExp)
    
        # --- 4. Extract Amplitudes and their Errors ---
            self.As = np.zeros((numExp, numWL))
            self.errAs = np.zeros((numExp, numWL))
            self.Bs = None      # To store Oscillation Amplitude Spectrum
            self.errBs = None
            
            try:
                if self.t0_choice == 'No':
                    # -----------------------------------------------------------------
                    # FIX VARPRO: Cálculo analítico de errores para parámetros lineales
                    # -----------------------------------------------------------------
                    # 'C' y 'self.S_T_full' ya fueron calculados en el paso 1 de esta función
                    
                    # 1. Pseudo-inversa de (C^T * C)
                    pseudo_inv_C = np.linalg.pinv(C.T @ C)
                    diag_cov = np.diagonal(pseudo_inv_C) # Shape: (n_species,)
                    
                    # 2. Calcular el Mean Squared Error (MSE) por longitud de onda
                    # resid tiene shape (numWL, numTD). Evaluamos la varianza en el tiempo.
                    dof_linear = resid.shape[1] - C.shape[1]
                    if dof_linear > 0:
                        mse_per_wl = np.sum(resid**2, axis=1) / dof_linear # Shape: (numWL,)
                    else:
                        mse_per_wl = np.zeros(numWL)
                        
                    # 3. Matriz de errores (Propagación: covarianza * MSE)
                    # Resultado: Matriz de shape (n_species, numWL)
                    err_S_T_full = np.sqrt(np.maximum(np.outer(diag_cov, mse_per_wl), 0))
                    
                    if "Oscillation" in self.model_type:
                        self.As = self.S_T_full[:numExp, :]
                        self.errAs = err_S_T_full[:numExp, :]
                        
                        self.Bs = self.S_T_full[numExp, :]
                        self.errBs = err_S_T_full[numExp, :]
                        self.t0s = np.full(numWL, x[1])
                    else:
                        # Modelo Estándar (Parallel/Sequential)
                        self.As = self.S_T_full[:numExp, :]
                        self.errAs = err_S_T_full[:numExp, :]
                        self.t0s = np.full(numWL, x[1])
    
                else:
                    # Reconstrucción Clásica (Chirp) donde el optimizador sí ve las amplitudes
                    if "Oscillation" in self.model_type:
                        base_A = 2 + numExp + 3
                        params_per_wl = numExp + 1 
                        
                        all_local = x[base_A:]
                        all_local_err = self.ci[base_A:]
                        
                        mat_local = all_local.reshape(numWL, params_per_wl)
                        mat_err = all_local_err.reshape(numWL, params_per_wl)
                        
                        self.As = mat_local[:, :numExp].T
                        self.errAs = mat_err[:, :numExp].T
                        
                        self.Bs = mat_local[:, numExp]
                        self.errBs = mat_err[:, numExp]
                        self.t0s = np.full(numWL, x[1])
                    else:
                        base_A = 2 + numExp
                        self.As = x[base_A:].reshape(numWL, numExp).T
                        self.errAs = self.ci[base_A:].reshape(numWL, numExp).T
                        self.t0s = np.full(numWL, x[1])
                        
            except Exception as e:
                print(f"Error extrayendo amplitudes o calculando errores: {e}")
    
            # --- 5. Export Results ---
            if getattr(self, 'is_batch_running', False) and hasattr(self, 'current_batch_outdir'):
                        outdir = self.current_batch_outdir
            else:
                outdir = os.path.join(self.base_dir, "fit")
                        
            os.makedirs(outdir, exist_ok=True)
    
            try:
                # Save full results as a binary NumPy file
                np.save(os.path.join(outdir, "GFitResults.npy"), {
                    "taus": self.extracted_taus,
                    "err_taus": self.extracted_errtaus,
                    "As": self.As,
                    "errAs": self.errAs,
                    "WL": WL,
                    "TD": TD,
                    "fitres": fitres,
                    "resid": resid
                })
                
                # Export plain text axes for accessibility
                np.savetxt(os.path.join(outdir, "WL.txt"), WL, fmt='%.6f', header="Wavelength (nm)")
                np.savetxt(os.path.join(outdir, "TD.txt"), TD, fmt='%.6f', header="Time Delay (ps)")
                
                # Write amplitudes and errors to a tab-separated text file
                with open(os.path.join(outdir, "Amplitudes.txt"), 'w') as f:
                                    # Inyectamos las constantes de tiempo globales en forma de comentario (#) para no romper la tabla
                                    f.write("# === Global Fit Lifetimes (Taus) & Errors Summary ===\n")
                                    tau_meta = []
                                    for k in range(numExp):
                                        t_val = self.extracted_taus[k] if (hasattr(self, 'extracted_taus') and k < len(self.extracted_taus)) else 0.0
                                        t_err = self.extracted_errtaus[k] if (hasattr(self, 'extracted_errtaus') and k < len(self.extracted_errtaus)) else 0.0
                                        tau_meta.append(f"tau{k+1}={t_val:.4f}+-{t_err:.4f}")
                                    f.write("# " + ", ".join(tau_meta) + "\n")
                                    f.write("# ====================================================\n")
                                    
                                    # Escritura normal de las columnas de amplitudes
                                    header_list = [f"A{i+1}\tErrA{i+1}" for i in range(numExp)]
                                    f.write("WL(nm)\t" + "\t".join(header_list) + "\n")
                                    for i in range(numWL):
                                        line_data = [f"{WL[i]:.2f}"]
                                        for j in range(numExp):
                                            val = self.As[j, i] if j < self.As.shape[0] else 0
                                            err = self.errAs[j, i] if j < self.errAs.shape[0] else 0
                                            line_data.append(f"{val:.6e}")
                                            line_data.append(f"{err:.6e}")
                                        f.write("\t".join(line_data) + "\n")
                print(f"Results successfully exported to: {outdir}")
    
            except Exception as e:
                print(f"Error saving output files: {e}")     
    
            # Update visualization and summary
            self._update_fit_canvas()
            self._update_resid_canvas()
            self.btn_show_das.setEnabled(True)
    
            if not getattr(self, 'is_batch_running', False):
                self.show_results_summary()
                rmsd = np.sqrt(np.mean(resid**2))
                QMessageBox.information(self, "Fit Complete", 
                                        f"Optimization finished successfully.\nRMSD: {rmsd:.2e}\nData saved in /fit/")
                
            self.show_results_summary()
            
            # Final completion notice with RMSD
            rmsd = np.sqrt(np.mean(resid**2))
            QMessageBox.information(self, "Fit Complete", 
                                    f"Optimization finished successfully.\nRMSD: {rmsd:.2e}\nData saved in /fit/")
    
    def show_results_summary(self):
        """Displays a popup window detailing the final global parameters derived from the fit."""
        if self.fit_x is None: return

        # Initialize the Summary Dialog
        dlg = QDialog(self)
        dlg.setWindowTitle("Fit Results Summary")
        dlg.resize(400, 300)
        layout = QVBoxLayout(dlg)
        
        # Initialize the table to display parameters
        table = QTableWidget()
        layout.addWidget(table)
        
        # Prepare data based on the kinetic model results
        results = []
        # Index 0: Instrument Response Function (IRF) width
        results.append(["w (IRF)", f"{self.fit_x[0]:.4f}"])
        # Index 1: Time zero offset
        results.append(["t0", f"{self.fit_x[1]:.4f}"])
        
        # Append extracted lifetimes (Taus) with their respective errors
        for i in range(self.numExp):
            val = self.extracted_taus[i]
            error = self.extracted_errtaus[i] if self.extracted_errtaus is not None else 0.0
            results.append([f"τ{i+1}", f"{val:.2f} ± {error:.2f} ps"])

        # Configure table dimensions and headers
        table.setRowCount(len(results))
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Parameter", "Final Value"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # Populate the table with results
        for i, (name, val) in enumerate(results):
            table.setItem(i, 0, QTableWidgetItem(name))
            table.setItem(i, 1, QTableWidgetItem(val))
            
        # Add a close button to dismiss the dialog
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dlg.accept)
        layout.addWidget(btn_close)
        
        dlg.exec_()
    
    
    def plot_das_and_more(self):
            """
            Opens an external window to display DAS/SAS (Decay/Species Associated Spectra).
            If an oscillation model is used, it separates the plots into two distinct panels.
            """
            if self.As is None: return
    
            # Define output directory for plots
            outdir = os.path.join(self.base_dir, "Plots")
            os.makedirs(outdir, exist_ok=True)
            
            wl = getattr(self, '_wl_proc', self.WL)
            td = getattr(self, '_td_proc', self.TD)
            
            # --- DETECT OSCILLATION COMPONENT ---
            has_oscillation = hasattr(self, 'Bs') and self.Bs is not None
            
            # Figure configuration: 2 panels if oscillation exists, 1 otherwise
            fig_das = Figure(figsize=(14, 6) if has_oscillation else (8, 6))
            
            if has_oscillation:
                        ax_das = fig_das.add_subplot(121)
                        ax_osc = fig_das.add_subplot(122)
            else:
                        ax_das = fig_das.add_subplot(111)
                        ax_osc = None
                        
            # --- 1. PLOT DAS/SAS (Exponential Components) ---
            colors = ['b', 'r', 'g', 'orange', 'm', 'c']
            markers = ['o', 's', '^', 'D', 'v', 'p'] 
            
            for n in range(self.numExp):
                tau_val = self.extracted_taus[n]
                # Validate error existence and check for NaNs
                if self.extracted_errtaus is not None and n < len(self.extracted_errtaus):
                     err_tau = self.extracted_errtaus[n]
                     if np.isnan(err_tau): err_tau = 0.0
                else:
                     err_tau = 0.0
    
                lbl = f"$\\tau_{n+1}$ = {tau_val:.2f} ± {err_tau:.2f} ps"
                color = colors[n % len(colors)]
                marker = markers[n % len(markers)] # Assign marker
    
                if self.errAs is not None:
                    # Clean NaNs in Y-error to prevent Matplotlib issues
                    err_y = np.nan_to_num(self.errAs[n])
                    
                    # Plot line with markers and error bars (caps included)
                    ax_das.errorbar(wl, self.As[n], yerr=err_y, label=lbl, 
                                    color=color, fmt=f'-{marker}', markersize=5, 
                                    capsize=4, capthick=1.5, linewidth=1.5, elinewidth=1.5)
                else:
                    # Fallback plot without error bars
                    ax_das.plot(wl, self.As[n], f'-{marker}', label=lbl, color=color, 
                                markersize=5, linewidth=1.5)
            
            ax_das.set_xlabel("Wavelength (nm)")
            if self.model_type == "Sequential":
                ax_das.set_ylabel("SAS (Concentration)")
                ax_das.set_title("Species Associated Spectra (SAS)")
            else:
                ax_das.set_ylabel("DAS Amplitude (ΔA)")
                ax_das.set_title("Decay Associated Spectra (DAS)")
            
            ax_das.legend(frameon=True)
            ax_das.axhline(0, color='k', linestyle='--', alpha=0.5)
            ax_das.grid(True, linestyle=':', alpha=0.4)
    
            # --- 2. PLOT OSCILLATION COMPONENT (If applicable) ---
            if has_oscillation and ax_osc is not None:
                # Retrieve physical parameters from fit_x vector
                # Indices based on: [w, t0, tau1..n, alpha, omega, phi, ...]
                alpha = self.fit_x[2 + self.numExp]
                omega = self.fit_x[2 + self.numExp + 1]
                phi   = self.fit_x[2 + self.numExp + 2]
                
                # Descriptive title for the oscillation panel
                title_osc = (f"Oscillation Spectrum\n"
                             f"Damping α={alpha:.4f} | Freq ω={omega:.4f} | Phase φ={phi:.2f}")
                
                # Plot B-Spectrum (Oscillation Amplitude)
                ax_osc.plot(wl, self.Bs, color='black', linewidth=2, label='Oscillation Amplitude (B)')
                
                # Plot B-Spectrum error as a shaded area (fill_between)
                if self.errBs is not None:
                    ax_osc.fill_between(wl, self.Bs - self.errBs, self.Bs + self.errBs, color='black', alpha=0.1)
                
                ax_osc.set_xlabel("Wavelength (nm)")
                ax_osc.set_ylabel("Oscillation Amplitude")
                ax_osc.set_title(title_osc, color='darkblue')
                ax_osc.axhline(0, color='k', linestyle='--', alpha=0.5)
                ax_osc.grid(True, linestyle=':', alpha=0.4)
                ax_osc.legend(frameon=True)
    
            fig_das.tight_layout()
    
            # Save the figure silently
            savename = "DAS_and_Oscillation.png" if has_oscillation else "DAS.png"
            try:
                fig_das.savefig(os.path.join(outdir, savename), dpi=300)
                print(f"Plot saved to {outdir}")
            except Exception as e:
                print(f"Error saving DAS plot: {e}")
    
            # --- LA MAGIA: Usar tu ventana personalizada para que no desaparezca ---
            self.das_viewer = PlotViewerWindow(fig_das, title="DAS / SAS Spectra", parent=self)
            self.das_viewer.show()
            
            # --- 3. RESIDUALS MAP EXPORT (También en modo silencioso) ---
            fig_res = Figure()
            canvas_res = FigureCanvasAgg(fig_res)
            ax_res = fig_res.add_subplot(111)
            
            pcm = ax_res.pcolormesh(wl, td, self.fit_resid.T, cmap='jet', shading='auto')
            fig_res.colorbar(pcm, ax=ax_res, label='Residuals')
            ax_res.set_title("Residuals Map")
            ax_res.set_xlabel("Wavelength (nm)")
            ax_res.set_ylabel("Delay (ps)")
            if hasattr(self, 'yscale') and self.yscale == 'symlog':
                 ax_res.set_yscale('symlog', linthresh=2) # Consistente con el resto
            fig_res.tight_layout()
            
            fig_res.savefig(os.path.join(outdir, "Residuals_Map.png"), dpi=300)
    
            # Mostramos el Trace Explorer
            self.trace_viewer = TraceExplorerWindow(self, outdir)
            self.trace_viewer.show()
            


class SASDASPlotterWindow(QDialog):
    """
    Advanced Drag & Drop Publication-Quality Plotter for SAS/DAS Spectra.
    Carga archivos multi-columna emparejando amplitudes con sus errores de forma clásica 
    e incluye controles dinámicos para modificar el tamaño de los puntos, las tapitas de error (capsize)
    y formatea la leyenda con precisión de dos decimales (τ = X.XX ± Y.YY ps).
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Publication-Quality SAS/DAS Plotter (Spectra Mode)")
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
        
        self.drop_label = QLabel("📥 DEJA CAER AQUÍ TUS ARCHIVOS DE ESPECTROS (DAS / SAS / AMPLITUDES)")
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
        
        ctrl_group = QGroupBox("Estética del Espectro")
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
        ctrl_form.addRow("Paleta de Colores:", self.combo_palette)
        
        # Controles de puntos y errorbars
        self.spin_ms = QSpinBox()
        self.spin_ms.setRange(0, 15)
        self.spin_ms.setValue(4)  
        self.spin_ms.setSuffix(" px (Puntos)")
        self.spin_ms.valueChanged.connect(self.replotted)
        ctrl_form.addRow("Tamaño Puntos:", self.spin_ms)
        
        self.spin_cap = QDoubleSpinBox()
        self.spin_cap.setRange(0.0, 15.0)
        self.spin_cap.setValue(3.0)  
        self.spin_cap.setSingleStep(0.5)
        self.spin_cap.setSuffix(" pt (Tapita)")
        self.spin_cap.valueChanged.connect(self.replotted)
        ctrl_form.addRow("Ancho Tapita:", self.spin_cap)
        
        # Dimensiones de la figura en pulgadas
        self.spin_width = QDoubleSpinBox()
        self.spin_width.setRange(3.0, 15.0)
        self.spin_width.setValue(6.5)  
        self.spin_width.setSingleStep(0.5)
        self.spin_width.setSuffix(" in (Ancho)")
        self.spin_width.valueChanged.connect(self.update_fig_size)
        ctrl_form.addRow("Ancho Figura:", self.spin_width)
        
        self.spin_height = QDoubleSpinBox()
        self.spin_height.setRange(2.0, 10.0)
        self.spin_height.setValue(4.5)  
        self.spin_height.setSingleStep(0.5)
        self.spin_height.setSuffix(" in (Alto)")
        self.spin_height.valueChanged.connect(self.update_fig_size)
        ctrl_form.addRow("Alto Figura:", self.spin_height)
        
        # Herramienta de Crop manual ejes X e Y
        self.chk_auto_axes = QCheckBox("Límites de Ejes Automáticos")
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
        
        self.btn_clear = QPushButton("🗑️ Limpiar Gráfico")
        self.btn_clear.clicked.connect(self.clear_data)
        ctrl_form.addRow(self.btn_clear)
        
        self.btn_export_fig = QPushButton("💾 Exportar Espectro (600 DPI)")
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
