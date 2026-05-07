import os
import re
import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QMessageBox, QProgressBar, QTableWidget, QTableWidgetItem,
    QHeaderView, QComboBox, QDoubleSpinBox, QSpinBox, QGroupBox, 
    QFormLayout, QWidget, QTabWidget, QApplication, QInputDialog,
    QCheckBox, QLineEdit, QListView
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
        
        screen = QApplication.primaryScreen()
        screen_geom = screen.availableGeometry()
        
        w_target = int(screen_geom.width() * 0.8)
        h_target = int(screen_geom.height() * 0.65)
        
        x_pos = (screen_geom.width() - w_target) // 2 + screen_geom.left()
        y_pos = screen_geom.top() + 50
        
        self.setGeometry(x_pos, y_pos, w_target, h_target)
        self.setStyleSheet(DARK_THEME_STYLE + BUTTON_STYLE) # Apply Dark Theme

        # --- 1. Data Variables ---
        self.parent_app = parent
        self.data_c = None   
        self.data_raw = None 
        self.TD = None       
        self.WL = None       
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
        
        # --- A. Left Panel (Sidebar) ---
        self.sidebar = QWidget()
        self.sidebar.setFixedWidth(340) 
        self.sidebar_layout = QVBoxLayout(self.sidebar)
        self.sidebar_layout.setContentsMargins(5, 5, 5, 5)
        
        self._init_sidebar_ui() 
        
        main_layout.addWidget(self.sidebar)
    
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
        gb_load.setLayout(v_load)
        l.addWidget(gb_load)

        # --- Group 2: Pre-processing ---
        gb_prep = QGroupBox("2. Pre-processing")
        form_prep = QFormLayout()

        # Baseline
        self.spin_bl = QSpinBox()
        self.spin_bl.setRange(0, 500)
        self.spin_bl.setValue(0)
        self.spin_bl.valueChanged.connect(self.apply_baseline_correction) 
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
        
        self.btn_show_das = QPushButton("Show Plots / Results")
        self.btn_show_das.setEnabled(False)
        self.btn_show_das.clicked.connect(self.plot_das_and_more) 
        l.addWidget(self.btn_show_das)

        l.addStretch()
                
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
        ax2.set_xlabel("Energy / Wavelength")
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
                # assuming a shape [NumWL, NumTD] or [NumTD,NumWL] 
                baseline = np.mean(temp_data[:, :n_pts], axis=1, keepdims=True)
                temp_data = temp_data - baseline
            else:
                print("Warning: Not enough points for baseline.")
    
        
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
            self.label_status.setText(f"Loaded from Parent: {len(self.WL)} WL, {len(self.TD)} TD")

    def load_data(self):
        """Loads data from .npy files utilizing the external 'fit' module."""
        try:
            # Unpack data from the external fit module loader
            raw_data, TD, WL, base_dir = fit.load_npy(self)
            
            self.data_raw = raw_data.copy()
            self.TD = TD
            self.WL = WL
            self.base_dir = base_dir
            
            # Synchronize UI with the new dataset
            self._update_ui_limits_from_data()
            self.btn_run.setEnabled(True)
            self.label_status.setText(f"Loaded File: {len(self.WL)} WL, {len(self.TD)} TD")
            
        except Exception as e:
            # Display a critical message box if the loading process fails
            QMessageBox.critical(self, "Error loading", str(e))

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
    

    def _preview_data_processing(self):
        """
        Processes raw data by applying: Baseline -> Wavelength Crop -> Time Crop -> Binning.
        Stores the resulting processed matrix in self.data_c for fitting purposes.
        """
        if self.data_raw is None: return
        
        temp_data = self.data_raw.copy()
        temp_WL = self.WL.copy()
        temp_TD = self.TD.copy()

        # 1. Baseline Correction
        n_pts = self.spin_bl.value()
        if n_pts > 0 and temp_data.shape[1] >= n_pts:
            # Assuming shape (WL, TD) -> axis 1 is time
            # Calculate mean of the first n points to subtract as background
            baseline = np.mean(temp_data[:, :n_pts], axis=1, keepdims=True)
            temp_data = temp_data - baseline
            
        # 2. Wavelength Cropping
        w_min = self.spin_wl_min.value()
        w_max = self.spin_wl_max.value()
        mask_w = (temp_WL >= min(w_min, w_max)) & (temp_WL <= max(w_min, w_max))
        
        # --- MULTI-CROP PROCESSOR ---
        if hasattr(self, 'line_exclude'):
            exclude_str = self.line_exclude.text().strip()
            if exclude_str:
                # Initialize an empty exclusion mask (all False)
                mask_exclude = np.zeros_like(temp_WL, dtype=bool)
                
                # Split by commas to handle multiple ranges
                ranges = exclude_str.split(',')
                for r in ranges:
                    try:
                        # Split by hyphen to define start-end
                        parts = r.split('-')
                        if len(parts) == 2:
                            c_min = float(parts[0].strip())
                            c_max = float(parts[1].strip())
                            
                            # Accumulate exclusion zones using bitwise OR (|)
                            mask_exclude |= (temp_WL >= min(c_min, c_max)) & (temp_WL <= max(c_min, c_max))
                    except ValueError:
                        # Ignore malformed input without crashing
                        pass 
                
                # Apply general exclusion: Keep previous mask AND NOT excluded regions
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
            
        # Optional: Zero-out negative time delays post-baseline
        if hasattr(self, 'chk_zero_neg') and self.chk_zero_neg.isChecked():
                    mask_neg = temp_TD < 0
                    if np.any(mask_neg):
                        temp_data[:, mask_neg] = 0.0  # The real background post-baseline is 0
        # 4. Binning (Simple averaging)
        b_size = self.spin_bin.value()
        if b_size > 1:
            # Spectral (WL) axis binning
            n_wl = temp_data.shape[0]
            new_len = n_wl // b_size
            if new_len > 0:
                # Trim excess and perform reshape + mean
                temp_data = temp_data[:new_len*b_size, :]
                temp_data = temp_data.reshape(new_len, b_size, temp_data.shape[1]).mean(axis=1)
                temp_WL = temp_WL[:new_len*b_size]
                temp_WL = temp_WL.reshape(new_len, b_size).mean(axis=1)

        # SAVE PROCESSED RESULT
        self.data_c = temp_data
        
        # Store processed versions for accurate plotting
        self._wl_proc = temp_WL
        self._td_proc = temp_TD
        
        # Update UI and plot
        self._update_exp_canvas(use_processed=True)
        self.label_status.setText(f"Data Ready: {len(temp_WL)} WL, {len(temp_TD)} TD")
        
        # --- AUTOMATIC 2D MAP EXPORT ---
        try:
            if self.base_dir:
                outdir = os.path.join(self.base_dir, "Plots")
                os.makedirs(outdir, exist_ok=True)
                
                save_path = os.path.join(outdir, "Experimental_Processed_Preview.png")
                
                # Extract figure from the experimental canvas and save
                self.canvas_exp.figure.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"2D Map saved to: {save_path}")
                
        except Exception as e:
            print(f"Error saving automatic preview: {e}")
            
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
                # Dynamic color scaling using 1st and 99th percentiles for better contrast
                vals = self.data_c.flatten()
                vmin = np.percentile(vals, 1) 
                vmax = np.percentile(vals, 99)
                
                # Render the 2D map (transpose data_c to match WL vs TD orientation)
                self.pcm_exp = self.ax_exp.pcolormesh(Xs, Ys, self.data_c.T, 
                                                      shading="auto", cmap='jet', 
                                                      vmin=vmin, vmax=vmax)
                
                # Set axis labels
                self.ax_exp.set_xlabel("Energy / Wavelength")
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
                # Basic validation to ensure there is enough data to plot a mesh
                if Z.shape[0] < 2 or Z.shape[1] < 2: return
    
                # Contrast enhancement by clipping the colorbar to the 1st and 99th percentiles
                vals = Z.flatten()
                vmin = np.percentile(vals, 1)  
                vmax = np.percentile(vals, 99) 
    
                self.pcm_fit = self.ax_fit.pcolormesh(Xs, Ys, Z, shading='auto', cmap='jet', 
                                                      vmin=vmin, vmax=vmax)
                
                self.ax_fit.set_title("Fit Reconstructed")
                self.ax_fit.set_xlabel("Energy (eV)")
                self.ax_fit.set_ylabel("Delay (ps)")
                
                # --- APPLY Y-AXIS SCALE (CONDITIONAL) ---
                if hasattr(self, 'yscale') and self.yscale == 'symlog':
                    self.ax_fit.set_yscale('symlog', linthresh=1.0)
                else:
                    self.ax_fit.set_yscale('linear')
                
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
                self.ax_resid.set_xlabel("Energy (eV)")
                self.ax_resid.set_ylabel("Delay (ps)")
                
                # --- APPLY Y-AXIS SCALE (CONDITIONAL) ---
                if hasattr(self, 'yscale') and self.yscale == 'symlog':
                    self.ax_resid.set_yscale('symlog', linthresh=1.0)
                else:
                    self.ax_resid.set_yscale('linear')
                
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
        self.progress_bar.setFormat("Iterating: %v") # Display current iteration count
        self.iter_count = 0 
    
        def residuals(p_free):
            """Objective function for least_squares: returns the difference between model and data."""
            # 1. Increment counter on every model evaluation
            self.iter_count += 1
            
            # 2. Update progress bar every 10 evaluations to avoid UI overhead
            if self.iter_count % 10 == 0:
                # Cycle the bar (0-100) since the total number of iterations is unknown
                val = (self.iter_count // 10) % 101
                self.progress_bar.setValue(val)
                
                # Force the UI to process events to prevent freezing and update graphics
                QApplication.processEvents()
    
            # Reconstruct the full parameter vector (injecting free params into original array)
            x_full = self.ini.copy()
            x_full[free_indices] = p_free
            
            # Select and evaluate the specific kinetic model
            if self.model_type == "Sequential":
                F = fit.eval_sequential_model(x_full, TD, self.numExp, numWL, self.t0_choice)
            elif self.model_type == 'Damped Oscillation':
                F = fit.eval_oscillation_model(x_full, TD, self.numExp, numWL, self.t0_choice)
          
            else:
                F = fit.eval_global_model(x_full, TD, self.numExp, numWL, self.t0_choice)
            
            return F.flatten() - data_flat
    
        try:
            # Run the Levenberg-Marquardt or TRF algorithm
            res = least_squares(
                fun=residuals,
                x0=x0_free,
                bounds=(low_free, upp_free),
                method='trf',
                verbose=0
            )
            
            # Store the results and reconstruct the final parameter vector
            self.fit_result = res
            self.fit_x = self.ini.copy()
            self.fit_x[free_indices] = res.x
            
            # Finalize progress bar status
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
            if self.model_type == "Sequential":
                F_mat = fit.eval_sequential_model(x, TD, numExp, numWL, self.t0_choice)
            elif self.model_type == 'Damped Oscillation':
                F_mat = fit.eval_oscillation_model(x, TD, numExp, numWL, self.t0_choice)
            else:
                F_mat = fit.eval_global_model(x, TD, numExp, numWL, self.t0_choice)
                
            fitres = F_mat.T 
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
                    # Use Pseudo-Inverse (pinv) to prevent crashes from singular matrices
                    # cov_free = inv(J.T * J) * MSE
                    H = J.T @ J
                    cov_free = np.linalg.pinv(H) 
                    
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
                    if "Oscillation" in self.model_type:
                        # Structure: [w, t0, taus..., alpha, omega, phi, ...]
                        base_A = 2 + numExp + 3
                        params_per_wl = numExp + 1 
                        
                        # Extract all local params (A's + B)
                        all_local = x[base_A:]
                        all_local_err = self.ci[base_A:]
                        
                        # Reshape to (numWL, params_per_wl)
                        mat_local = all_local.reshape(numWL, params_per_wl)
                        mat_err = all_local_err.reshape(numWL, params_per_wl)
                        
                        # Separate A's (Decays) and B (Oscillation)
                        self.As = mat_local[:, :numExp].T
                        self.errAs = mat_err[:, :numExp].T
                        
                        # The last column is B (Oscillation Amplitude)
                        self.Bs = mat_local[:, numExp]
                        self.errBs = mat_err[:, numExp]
                        
                        self.t0s = np.full(numWL, x[1])
    
                    else:
                        # Standard Model: Extract amplitudes for each lifetime
                        base_A = 2 + numExp
                        self.As = x[base_A:].reshape(numWL, numExp).T
                        self.errAs = self.ci[base_A:].reshape(numWL, numExp).T
                        self.t0s = np.full(numWL, x[1])
                else:
                    pass # Chirp-specific logic could be added here
                    
            except Exception as e:
                print(f"Error extrayendo amplitudes: {e}")
    
            # --- 5. Export Results ---
            base_dir = self.base_dir
            outdir = os.path.join(base_dir, "fit")
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
            if has_oscillation:
                fig_das, (ax_das, ax_osc) = plt.subplots(1, 2, figsize=(14, 6))
            else:
                fig_das, ax_das = plt.subplots(figsize=(8, 6))
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
    
            # Save the figure
            savename = "DAS_and_Oscillation.png" if has_oscillation else "DAS.png"
            try:
                fig_das.savefig(os.path.join(outdir, savename), dpi=300)
                print(f"Plot saved to {outdir}")
            except Exception as e:
                print(f"Error saving DAS plot: {e}")
    
            fig_das.show()
            
            # --- 3. RESIDUALS MAP EXPORT ---
            fig_res, ax_res = plt.subplots()
            pcm = ax_res.pcolormesh(wl, td, self.fit_resid.T, cmap='jet', shading='auto')
            fig_res.colorbar(pcm, ax=ax_res, label='Residuals')
            ax_res.set_title("Residuals Map")
            ax_res.set_xlabel("Wavelength / Energy")
            ax_res.set_ylabel("Delay (ps)")
            if hasattr(self, 'yscale') and self.yscale == 'symlog':
                 ax_res.set_yscale('symlog', linthresh=1.0)
            fig_res.tight_layout()
            fig_res.savefig(os.path.join(outdir, "Residuals_Map.png"), dpi=300)
            plt.close(fig_res)
    
            # --- 4. INTERACTIVE TRACE VIEWER ---
            cont = True
            while cont:
                text_default = f"{wl[len(wl)//2]:.1f}"
                wl_str, ok = QInputDialog.getText(self, "Check Trace", 
                                                  f"Enter wavelength nm ({wl.min():.1f}-{wl.max():.1f}):", 
                                                  text=text_default)
                if not ok: break
    
                try:
                    target_wl = float(wl_str)
                    idx = np.argmin(np.abs(wl - target_wl))
                    real_wl = wl[idx]
    
                    y_exp = self.data_c[idx, :]
                    
                    # Create a hybrid time axis for curve smoothing
                    # Linear for early delays (rise time), Logarithmic for late decays
                    td_lin = np.linspace(td.min(), 1.0, 1000)
                    td_log = np.geomspace(1.0, td.max(), 1000)
                    td_smooth = np.unique(np.concatenate((td_lin, td_log)))
                    
                    # Re-evaluate the kinetic model using the smooth time axis
                    if self.model_type == "Sequential":
                        F_mat_smooth = fit.eval_sequential_model(self.fit_x, td_smooth, self.numExp, len(wl), self.t0_choice)
                    elif self.model_type == 'Damped Oscillation':
                        F_mat_smooth = fit.eval_oscillation_model(self.fit_x, td_smooth, self.numExp, len(wl), self.t0_choice)
                    else:
                        F_mat_smooth = fit.eval_global_model(self.fit_x, td_smooth, self.numExp, len(wl), self.t0_choice)
                        
                    # Extract smooth fit trace
                    y_fit_smooth = F_mat_smooth.T[idx, :] 

                    fig_trace, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
                    fig_trace.suptitle(f"Fit at {real_wl:.1f} nm", fontsize=14)
    
                    # Linear Plot (Early delays)
                    ax1.plot(td, y_exp, 'bo', markersize=4, alpha=0.6, label='Data')
                    ax1.plot(td_smooth, y_fit_smooth, 'r-', linewidth=2, label='Fit')
                    ax1.set_xlabel("Time / ps")
                    ax1.set_ylabel("ΔA")
                    ax1.legend(frameon=True)
                    ax1.grid(True, alpha=0.3)
    
                    # Semi-log Plot (Full decay)
                    mask_pos_exp = td > 0
                    mask_pos_smooth = td_smooth > 0
                    if np.any(mask_pos_exp):
                        ax2.plot(td[mask_pos_exp], y_exp[mask_pos_exp], 'bo', markersize=4, alpha=0.6)
                        ax2.plot(td_smooth[mask_pos_smooth], y_fit_smooth[mask_pos_smooth], 'r-', linewidth=2)
                        ax2.set_xscale('log')
                        ax2.set_xlabel("Time / ps (log scale)")
                        ax2.grid(True, which="both", ls="-", alpha=0.3)
    
                    plt.tight_layout()
                    plt.show(block=True) 
    
                    # Export individual trace data?
                    resp = QMessageBox.question(self, "Save Trace?",
                                                f"Do you want to save the trace files for {real_wl:.1f} nm",
                                                QMessageBox.Yes | QMessageBox.No)
    
                    if resp == QMessageBox.Yes:
                        img_name = f"Trace_{real_wl:.1f}nm.png"
                        fig_trace.savefig(os.path.join(outdir, img_name), dpi=300)
    
                        txt_name = f"Fit_{real_wl:.1f}nm.txt"
                        txt_path = os.path.join(outdir, txt_name)
                        
                        # Export data as tab-separated columns, handling mismatched lengths with empty strings
                        max_len = max(len(td), len(td_smooth))
                        
                        with open(txt_path, 'w') as f:
                            f.write("TD_exp(ps)\tExp(A)\tTD_fit(ps)\tFit_smooth(A)\n")
                            for i in range(max_len):
                                val_td = f"{td[i]:.6e}" if i < len(td) else ""
                                val_exp = f"{y_exp[i]:.6e}" if i < len(y_exp) else ""
                                val_td_s = f"{td_smooth[i]:.6e}" if i < len(td_smooth) else ""
                                val_fit_s = f"{y_fit_smooth[i]:.6e}" if i < len(y_fit_smooth) else ""
                                f.write(f"{val_td}\t{val_exp}\t{val_td_s}\t{val_fit_s}\n")
                    plt.close(fig_trace)
    
                except Exception as e:
                    QMessageBox.critical(self, "Trace Error", f"Failed to process trace: {e}")
    
                if QMessageBox.question(self, "Continue?", "View another wavelength trace?",
                                        QMessageBox.Yes|QMessageBox.No) == QMessageBox.No:
                    cont = False