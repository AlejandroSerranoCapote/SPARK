import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.ticker import FuncFormatter

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
    QPushButton, QMessageBox, QCheckBox
)

import spark.core.fitting.fit as fit
# We import the Paper Plotter for the button that launches it from the Trace Explorer
from spark.gui.plotters.publication import PaperPlotterWindow

BUTTON_STYLE = """
    QPushButton {
        background-color: #5A6268;   /* Professional neutral blue-gray */
        color: white;                
        border: 1px solid #545B62;  
        border-radius: 4px;          
        padding: 6px 12px;           
        font-weight: bold;
        font-family: "Segoe UI";
        font-size: 9pt;              
    }
    QPushButton:hover {
        background-color: #6C757D;   /* Hover glow */
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
        border-top: 1px solid #D0D0D0; /* Subtle top separator line */
        margin-top: 18px;             
        padding-top: 15px;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0px 5px 0px 0px;
        font-weight: bold;
        font-size: 10pt;
        color: #3C5488; /* Elegant dark blue tone (Nature style) */
    }

   
    QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {
        background-color: #F8F9FA; 
        border: 1px solid #CED4DA;
        border-radius: 4px;
        color: #212529;            
        padding: 4px 8px;                
        min-height: 22px; /* Force uniform height */
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
    /* --- PREMIUM TABS --- */
    QTabWidget::pane { 
        border: 1px solid #CED4DA; 
        background-color: #FFFFFF; 
        border-radius: 4px;
    }
    QTabBar::tab { 
        background-color: #E9ECEF; 
        color: #495057;
        padding: 8px 20px; 
        min-width: 100px;  /* Force minimum width to avoid cutting text */
        border: 1px solid #CED4DA; 
        border-bottom: none; 
        border-top-left-radius: 4px; 
        border-top-right-radius: 4px; 
        margin-right: 4px;
        font-family: "Segoe UI", Arial, sans-serif;
        font-size: 10pt;
        font-weight: bold; /* Always bold ensures correct geometry calculation by Qt */
    }
    QTabBar::tab:selected { 
        background-color: #FFFFFF;                         
        color: #0078D7; 
        border-bottom: 2px solid #FFFFFF; 
        padding-top: 10px; /* Active tab effect without breaking geometry */
    }
    QTabBar::tab:hover:!selected {
        background-color: #DEE2E6;
    }
"""


class Surface3DWindow(QDialog):
    """
    Independent window to visualize the 3D plot without blocking the main application.

    Provides interactive 3D rotation, zooming, and projection of 2D data onto 
    a 3D surface using Matplotlib's Qt5Agg backend.

    Parameters
    ----------
    xs : numpy.ndarray
        X-axis array (e.g., Wavelengths).
    ys : numpy.ndarray
        Y-axis array (e.g., Time Delays).
    zs : numpy.ndarray
        2D Z-axis matrix (e.g., Transient Absorption data).
    scale : str, optional
        The scale of the Y-axis ('linear' or 'symlog'). Defaults to 'linear'.
    parent : QWidget, optional
        Parent widget. Defaults to None.
    """
    def __init__(self, xs, ys, zs, scale='linear', parent=None):
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

        Draws the main surface, projections (shadows) on the Z floor, and 
        extracted representative slices for both spectra and kinetics on the walls.

        Parameters
        ----------
        xs : numpy.ndarray
            X-axis array.
        ys : numpy.ndarray
            Y-axis array.
        zs : numpy.ndarray
            2D Z-axis matrix.
        scale : str
            The scale of the Y-axis ('linear' or 'symlog').
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
    """
    Independent window to visualize SAS/DAS charts without blocking the application.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The Matplotlib figure object to be embedded.
    title : str, optional
        The window title. Defaults to "Plot".
    parent : QWidget, optional
        Parent widget. Defaults to None.
    """
    def __init__(self, fig, title="Plot", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.fig = fig
        self.resize(900, 600)
        self.setWindowModality(Qt.NonModal) # Ensures it doesn't block the main app
        self.setStyleSheet(DARK_THEME_STYLE)
        
        layout = QVBoxLayout(self)
        self.canvas = FigureCanvas(fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)


class TraceExplorerWindow(QDialog):
    """
    Interactive kinetics trace explorer without blocking the interface.

    Allows the user to select specific wavelengths, visualize their kinetic trace
    alongside the calculated VarPro fit, and export the data to PNG and TXT.

    Parameters
    ----------
    parent_panel : QWidget
        The parent GUI panel containing the fitted data attributes (e.g., fit_x, S_T_full).
    outdir : str
        The output directory path where exported trace files will be saved.
    """
    def __init__(self, parent_panel, outdir):
        super().__init__(parent_panel)
        self.p = parent_panel
        self.outdir = outdir
        self.setWindowTitle("Interactive Trace Viewer")
        self.resize(1000, 550)
        self.setWindowModality(Qt.NonModal) # Key to float freely
        self.setStyleSheet(DARK_THEME_STYLE + BUTTON_STYLE)

        layout = QVBoxLayout(self)

        # Top Controls
        ctrl_layout = QHBoxLayout()
        ctrl_layout.addWidget(QLabel("Wavelength (nm):"))
        
        self.wl_array = getattr(self.p, '_wl_proc', self.p.WL)
        self.td_array = getattr(self.p, '_td_proc', self.p.TD)
        
        # We use a ComboBox to navigate exactly through the measured indices
        self.combo_wl = QComboBox()
        self.combo_wl.addItems([f"{w:.1f}" for w in self.wl_array])
        self.combo_wl.setCurrentIndex(len(self.wl_array)//2)
        self.combo_wl.currentIndexChanged.connect(self.update_plot)
        ctrl_layout.addWidget(self.combo_wl)

        self.chk_show_components = QCheckBox("Show Components")
        self.chk_show_components.setChecked(True)
        self.chk_show_components.toggled.connect(self.update_plot)
        ctrl_layout.addWidget(self.chk_show_components)

        self.btn_save = QPushButton("Save Trace Data")
        self.btn_save.clicked.connect(self.save_trace)
        ctrl_layout.addWidget(self.btn_save)

        self.btn_paper_plot = QPushButton("Launch Paper Plotter")
        self.btn_paper_plot.clicked.connect(self.open_paper_plotter)
        self.btn_paper_plot.setStyleSheet("background-color: #3C5488; color: white;")
        ctrl_layout.addWidget(self.btn_paper_plot)
        
        layout.addLayout(ctrl_layout)
        # Matplotlib Canvas
        self.fig = plt.Figure(figsize=(12, 5))
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self.update_plot()

    def _extract_components_data(self, idx, td_smooth, x_local, use_art, art_mode):
        """
        Calculates individual component contributions (species, decay channels, oscillations)
        at the specified wavelength channel.
        """
        components_data = []
        y_art_smooth = None
        y_fit_smooth = np.zeros_like(td_smooth)
        
        if not hasattr(self.p, 'S_T_full'):
            return y_fit_smooth, y_art_smooth, components_data

        if self.p.model_type == "Sequential":
            C_smooth = fit.get_concentration_matrix_sequential(x_local, td_smooth, self.p.numExp, use_art, art_mode)
        elif self.p.model_type == "Stretched Exponential":
            C_smooth = fit.get_concentration_matrix_stretched(x_local, td_smooth, self.p.numExp, use_art, art_mode)
        elif self.p.model_type == 'Damped Oscillation':
            C_smooth = fit.get_concentration_matrix_oscillation(x_local, td_smooth, self.p.numExp, use_art, art_mode)
        elif self.p.model_type == "Custom GUI Model" and getattr(self.p, 'current_custom_model', None) is not None:
            model = self.p.current_custom_model
            w, t0 = x_local[0], x_local[1]
            num_params_cineticos = len(model.param_labels)
            x_nl_params = x_local[2:2+num_params_cineticos]
            C_smooth = model.get_concentration_matrix(x_nl_params, td_smooth, w, t0, use_art, art_mode)
        else:
            C_smooth = fit.get_concentration_matrix_global(x_local, td_smooth, self.p.numExp, use_art, art_mode)

        y_fit_smooth = C_smooth @ self.p.S_T_full[:, idx]

        # 1. Coherent artifact contribution
        num_art_bases = 0
        if use_art:
            from spark.core.fitting.fit import get_coherent_artifact
            num_art_bases = 1 if art_mode == 'raman' else (2 if art_mode == 'xpm' else 3)
            w_local, t0_local = x_local[0], x_local[1]
            C_art_smooth = get_coherent_artifact(td_smooth, t0_local, w_local, mode=art_mode)
            S_T_art = self.p.S_T_full[-num_art_bases:, idx]
            y_art_smooth = C_art_smooth @ S_T_art

        # 2. Individual physical components contribution
        comp_palette = [
            '#1f77b4',  # Blue
            '#2ca02c',  # Green
            '#ff7f0e',  # Orange
            '#9467bd',  # Purple
            '#8c564b',  # Brown
            '#e377c2',  # Pink
            '#17becf',  # Cyan
            '#bcbd22',  # Olive / Yellow-green
            '#4363d8',  # Indigo
            '#3cb44b',  # Lime green
        ]

        num_phys_comp = C_smooth.shape[1] - num_art_bases
        for k in range(num_phys_comp):
            y_comp_k = C_smooth[:, k] * self.p.S_T_full[k, idx]

            if self.p.model_type == "Custom GUI Model":
                if hasattr(self.p, 'current_custom_model') and self.p.current_custom_model and k < len(self.p.current_custom_model.states):
                    lbl_k = f"Species: {self.p.current_custom_model.states[k]}"
                else:
                    lbl_k = f"Species {k+1}"
            elif self.p.model_type == 'Damped Oscillation' and k == self.p.numExp:
                lbl_k = "Oscillation"
            elif self.p.model_type == "Stretched Exponential":
                tau_val = self.p.extracted_taus[k] if (getattr(self.p, 'extracted_taus', None) is not None and k < len(self.p.extracted_taus)) else None
                beta_val = getattr(self.p, 'extracted_betas', None)
                if tau_val is not None:
                    if beta_val is not None and k < len(beta_val):
                        lbl_k = rf"Comp {k+1} ($\tau$ = {tau_val:.2f} ps, $\beta$ = {beta_val[k]:.2f})"
                    else:
                        lbl_k = rf"Comp {k+1} ($\tau$ = {tau_val:.2f} ps)"
                else:
                    lbl_k = f"Comp {k+1}"
            else:
                tau_val = self.p.extracted_taus[k] if (getattr(self.p, 'extracted_taus', None) is not None and k < len(self.p.extracted_taus)) else None
                if tau_val is not None:
                    lbl_k = rf"Comp {k+1} ($\tau$ = {tau_val:.2f} ps)"
                else:
                    lbl_k = f"Comp {k+1}"

            col_k = comp_palette[k % len(comp_palette)]
            components_data.append((lbl_k, y_comp_k, col_k))

        return y_fit_smooth, y_art_smooth, components_data

    def update_plot(self):
        """
        Clears the canvas and renders the linear and semi-logarithmic 
        kinetics plots for the currently selected wavelength, showing total fit,
        individual components, and coherent artifact.
        """
        self.fig.clear()
        idx = self.combo_wl.currentIndex()
        real_wl = self.wl_array[idx]
        
        y_exp = self.p.data_c[idx, :]
        td = self.td_array
        
        td_lin = np.linspace(td.min(), 1.0, 1000)
        td_log = np.geomspace(1.0, max(1.1, td.max()), 1000)
        td_smooth = np.unique(np.concatenate((td_lin, td_log)))
        
        # --- Local x vector with correct t0 ---
        x_local = self.p.fit_x.copy()
        is_indep_t0 = getattr(self.p, 'chk_indep_t0', None) and self.p.chk_indep_t0.isChecked()
        if is_indep_t0:
            base_kin = self.p._get_num_kinetic_params() - len(self.wl_array)
            x_local[1] = self.p.fit_x[base_kin + idx]
        
        use_art = getattr(self.p, 'chk_artifact', None) and self.p.chk_artifact.isChecked()
        art_mode = self.p._get_artifact_mode()
        
        y_fit_smooth, y_art_smooth, components_data = self._extract_components_data(
            idx, td_smooth, x_local, use_art, art_mode
        )

        ax1 = self.fig.add_subplot(121)
        ax2 = self.fig.add_subplot(122, sharey=ax1)
        self.fig.suptitle(f"Fit at {real_wl:.1f} nm", fontsize=14, fontweight="bold")

        show_comps = getattr(self, 'chk_show_components', None) is None or self.chk_show_components.isChecked()

        # --- Linear Plot (ax1) ---
        ax1.plot(td, y_exp, 'o', color='#1E293B', markersize=3.8, alpha=0.55, label='Data')
        ax1.plot(td_smooth, y_fit_smooth, color='#DC2626', linestyle='-', linewidth=2.0, label='Total Fit')

        if show_comps:
            for lbl_k, y_comp_k, col_k in components_data:
                ax1.plot(td_smooth, y_comp_k, linestyle='--', color=col_k, linewidth=1.5, alpha=0.9, label=lbl_k)

        if y_art_smooth is not None:
            ax1.plot(td_smooth, y_art_smooth, linestyle=':', color='gray', linewidth=1.8, label='Artifact')
            
        ax1.set_xlabel("Time / ps", fontweight="bold")
        ax1.set_ylabel("$\Delta A$", fontweight="bold")
        ax1.legend(frameon=True, fontsize=8, loc='best')
        ax1.grid(True, linestyle=":", alpha=0.4)

        # --- Semi-Log Plot (ax2) ---
        ax2.plot(td, y_exp, 'o', color='#1E293B', markersize=3.8, alpha=0.55)
        ax2.plot(td_smooth, y_fit_smooth, color='#DC2626', linestyle='-', linewidth=2.0)

        if show_comps:
            for lbl_k, y_comp_k, col_k in components_data:
                ax2.plot(td_smooth, y_comp_k, linestyle='--', color=col_k, linewidth=1.5, alpha=0.9)

        if y_art_smooth is not None:
            ax2.plot(td_smooth, y_art_smooth, linestyle=':', color='gray', linewidth=1.8)
            
        ax2.set_xscale('symlog', linthresh=1.0)
        ax2.set_xlabel("Time / ps (symlog scale)", fontweight="bold")
        ax2.grid(True, which="both", ls=":", alpha=0.4)

        self.fig.tight_layout()
        self.canvas.draw()
    
    def save_trace(self):
        idx = self.combo_wl.currentIndex()
        real_wl = self.wl_array[idx]
        
        # 1. Save High-Res PNG
        img_name = f"Trace_{real_wl:.1f}nm.png"
        self.fig.savefig(os.path.join(self.outdir, img_name), dpi=300)
        
        # 2. Extract experimental data
        td_exp = self.td_array
        y_exp = self.p.data_c[idx, :]
        
        # 3. Create high-resolution smooth time axis
        td_lin = np.linspace(td_exp.min(), 1.0, 1000)
        td_log = np.geomspace(1.0, max(1.1, td_exp.max()), 1000)
        td_smooth = np.unique(np.concatenate((td_lin, td_log)))
        
        x_local = self.p.fit_x.copy()
        is_indep_t0 = getattr(self.p, 'chk_indep_t0', None) and self.p.chk_indep_t0.isChecked()
        if is_indep_t0:
            base_kin = self.p._get_num_kinetic_params() - len(self.wl_array)
            x_local[1] = self.p.fit_x[base_kin + idx]
            
        use_art = getattr(self.p, 'chk_artifact', None) and self.p.chk_artifact.isChecked()
        art_mode = self.p._get_artifact_mode()
        
        y_fit_smooth, y_art_smooth, components_data = self._extract_components_data(
            idx, td_smooth, x_local, use_art, art_mode
        )
            
        # 4. Pad shorter columns with NaNs for OriginLab / Excel
        max_len = len(td_smooth) 
        td_exp_pad = np.pad(td_exp, (0, max_len - len(td_exp)), constant_values=np.nan)
        y_exp_pad = np.pad(y_exp, (0, max_len - len(y_exp)), constant_values=np.nan)
        
        txt_name = f"Trace_{real_wl:.1f}nm.txt"
        txt_path = os.path.join(self.outdir, txt_name)
        
        cols_to_stack = [td_exp_pad, y_exp_pad, td_smooth, y_fit_smooth]
        header_cols = ["Time_Exp(ps)", "Data_Exp", "Time_Fit_Smooth(ps)", "Fit_Total_Smooth"]
        
        for lbl_k, y_comp_k, _ in components_data:
            import re
            clean_name = re.sub(r'[^a-zA-Z0-9_]+', '_', lbl_k).strip('_')
            cols_to_stack.append(y_comp_k)
            header_cols.append(clean_name)
            
        if y_art_smooth is not None:
            cols_to_stack.append(y_art_smooth)
            header_cols.append("Artifact_Smooth")
            
        matriz_guardar = np.column_stack(cols_to_stack)
        cabecera = f"Wavelength: {real_wl:.1f} nm\n" + "\t".join(header_cols)
            
        np.savetxt(txt_path, matriz_guardar, fmt='%.6e', delimiter='\t', header=cabecera)
        QMessageBox.information(self, "Saved", f"High-Resolution trace data saved successfully in:\n{self.outdir}")

    def open_paper_plotter(self):
        """Opens the interactive Drag & Drop window for paper plots."""
        self.plotter_win = PaperPlotterWindow(self)
        self.plotter_win.show()
