import os
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from PyQt5.QtWidgets import (QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
                             QFileDialog, QMessageBox, QCheckBox, QSlider, QGroupBox, QInputDialog)
from PyQt5.QtCore import Qt, QEvent
from spark.core.chirp import fit_t0, apply_t0_correction_nonlinear, apply_t0_correction_poly, t0_model
from mpl_toolkits.axes_grid1 import make_axes_locatable
# Importamos a su clase "padre"
from spark.gui.flups import FLUPSAnalyzer
from spark.data.dataset import SpectroscopyDataset

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
        self.setWindowTitle("SPARK - TAS Analyzer")
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
 
        # Feature desactivada: la conversión ad-hoc de .dat a .csv (IMDEA) está deshabilitada.
        # El lector universal / pd.read_csv procesa directamente archivos .dat, .txt o .csv sin duplicar archivos.
        self.chk_convert_dat = None
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
        Loads both the measurement data and the corresponding solvent data,
        creating SpectroscopyDataset objects for both.
        """
        # --- Select measurement file ---
        file_path_medida, _ = QFileDialog.getOpenFileName(
            self, "Select Measurement CSV", "", "CSV Files (*.csv);;Data Files (*.txt *.dat)"
        )
        if not file_path_medida or not os.path.exists(file_path_medida):
            self.label_status.setText(" No measurement file selected.")
            return
            
        self.file_path = file_path_medida
        base_dir = os.path.dirname(self.file_path)
        base_name = os.path.splitext(os.path.basename(self.file_path))[0]
        self.results_dir = os.path.join(base_dir, f"{base_name}_results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        raw = pd.read_csv(file_path_medida, header=None, sep=None, engine='python').apply(pd.to_numeric, errors="coerce").dropna(how="any").values.astype(float)
        temp_TD, temp_WL, temp_medida = raw[0, 1:], raw[1:, 0], raw[1:, 1:]
        temp_medida[np.isnan(temp_medida)] = 0
        
        # Clean duplicates and sort
        wl, idx_wl = np.unique(temp_WL, return_index=True)
        td, idx_td = np.unique(temp_TD, return_index=True)
        medida_clean = temp_medida[idx_wl, :][:, idx_td]
        
        # --- 1. OBJETO DE MEDIDA ---
        self.dataset_medida = SpectroscopyDataset(medida_clean, wl, td)
        # Retrocompatibilidad gráfica
        self.medida = self.dataset_medida.data 
        self.WL = self.dataset_medida.wl
        self.TD = self.dataset_medida.td
        
        # --- Select solvent file ---
        file_path_solvente, _ = QFileDialog.getOpenFileName(
            self, "Select Solvent CSV", os.path.dirname(self.file_path), "CSV Files (*.csv);;Data Files (*.txt *.dat)"
        )
        
        if not file_path_solvente or not os.path.exists(file_path_solvente):
            self.label_status.setText(" No solvent file selected.")
            return
            
        rawSol = pd.read_csv(file_path_solvente, header=None, sep=None, engine='python').apply(pd.to_numeric, errors="coerce").dropna(how="any").values.astype(float)
        temp_TDSol, temp_WLSol, temp_solvente = rawSol[0, 1:], rawSol[1:, 0], rawSol[1:, 1:]
        temp_solvente[np.isnan(temp_solvente)] = 0
        
        wl_sol, idx_wl_sol = np.unique(temp_WLSol, return_index=True)
        td_sol, idx_td_sol = np.unique(temp_TDSol, return_index=True)
        solvente_clean = temp_solvente[idx_wl_sol, :][:, idx_td_sol]
        
        # --- 2. OBJETO DE SOLVENTE ---
        self.dataset_solvente = SpectroscopyDataset(solvente_clean, wl_sol, td_sol)
        # Retrocompatibilidad
        self.solvente = self.dataset_solvente.data
        self.WLSol = self.dataset_solvente.wl
        self.TDSol = self.dataset_solvente.td
        
        # Sincronizar Sliders UI
        nwl = len(self.dataset_medida.wl)
        self.slider_min.setRange(0, nwl - 1)
        self.slider_max.setRange(0, nwl - 1)
        self.slider_min.setValue(0)
        self.slider_max.setValue(nwl - 1)
        
        self.idx_min, self.idx_max = 0, nwl - 1
        
        try: self.slider_min.valueChanged.disconnect()
        except: pass
        try: self.slider_max.valueChanged.disconnect()
        except: pass
        self.slider_min.valueChanged.connect(self.update_wl_range)
        self.slider_max.valueChanged.connect(self.update_wl_range)
        
        # Reset de estados
        self.result_fit = None
        self.dataset_corrected = None
        self.data_corrected = None
        self.showing_corrected = False
        self.fit_line_artist = None
        self.clicked_points = []
        if hasattr(self, 'btn_show_corr'):
            self.btn_show_corr.setEnabled(False)
            self.btn_show_corr.setText("Show Corrected Map")
        
        # Generar el mapa inicial restando medida - solvente
        self.label_status.setText(" TAS data loaded")
        self.update_am_sf()
        self.plot_map()
        
        if hasattr(self, "btn_plot"): self.btn_plot.setEnabled(True)
        if hasattr(self, "btn_select"): self.btn_select.setEnabled(True)
        if hasattr(self, "btn_fit"): self.btn_fit.setEnabled(True)
        if hasattr(self, "btn_auto_chirp"): self.btn_auto_chirp.setEnabled(True)
        if hasattr(self, "btn_manual_chirp"): self.btn_manual_chirp.setEnabled(True)
        
        self.label_status.setText(f"TAS data loaded from: {os.path.basename(file_path_medida)}")

  
    def fit_t0_points(self):
        """
        Fits selected time-zero points and saves the corrected matrix.
        Overrides the base FLUPS method to ensure the solvent-subtracted base data is used.
        """
        if not getattr(self, "clicked_points", None) or len(self.clicked_points) < 2:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Not enough points", "Select at least 2 points on the map.")
            return
    
        w_points = np.array([p['x'] for p in self.clicked_points])
        t0_points = np.array([p['y'] for p in self.clicked_points])
        
        
        texto_modelo = self.combo_model.currentText()
        if texto_modelo == "Polynomial": mode = 'poly'
        elif texto_modelo == "Non linear": mode = 'nonlinear'
        else: mode = 'auto'
    
        try:
            self.update_am_sf() 
          
            from spark.core.chirp import fit_t0
            result = fit_t0(w_points, t0_points, self.WL, self.TD, self.data, mode=mode)
        except Exception as e:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Fit error", str(e))
            return
    
        self.result_fit = result
        self.data_corrected = result['corrected']
        
        self.plot_map(show_fit=True)
        self.btn_show_corr.setEnabled(True)
    
        base_dir = os.path.dirname(self.file_path)
        base_name = os.path.splitext(os.path.basename(self.file_path))[0]
        
        self.save_dir = os.path.join(base_dir, f"{base_name}_results")
        os.makedirs(self.save_dir, exist_ok=True)
    
        data_corr = np.copy(self.data_corrected)
        WL, TD = self.WL, self.TD
        t0_lambda, popt, method = result['t0_lambda'], result['popt'], result['method']
    
        np.save(os.path.join(self.save_dir, f"{base_name}_treated_data.npy"),
                {'data_c': data_corr, 'WL': WL, 'TD': TD})
        np.savetxt(os.path.join(self.save_dir, f"{base_name}_WL.txt"), WL, fmt='%.6f', header='Wavelength (nm)', comments='')
        np.savetxt(os.path.join(self.save_dir, f"{base_name}_TD.txt"), TD, fmt='%.6f', header='Delay (ps)', comments='')
        np.savetxt(os.path.join(self.save_dir, f"{base_name}_kin.txt"), data_corr.T, fmt='%.6e', delimiter='\t')
        np.savetxt(os.path.join(self.save_dir, f"{base_name}_spec.txt"), data_corr, fmt='%.6e', delimiter='\t')
        np.savetxt(os.path.join(self.save_dir, f"{base_name}_t0_fit.txt"), np.column_stack((WL, t0_lambda)), fmt='%.6f', header='Wavelength (nm)\t t0 (ps)', comments='')
                    
        with open(os.path.join(self.save_dir, f"{base_name}_fit_params.txt"), 'w') as f:
            f.write(f"Fit method: {method}\nFit parameters:\n")
            names = ['c4', 'c3', 'c2', 'c1', 'c0'] if method.startswith('poly') else ['a', 'b', 'c', 'd']
            for name, val in zip(names, popt): f.write(f"  {name} = {val:.6g}\n")
    
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.information(self, "Files saved", f" Results saved in:\n{self.save_dir}")
        QMessageBox.information(self, "t₀ Fit Result", f"Fit completed using {method} model.\nParameters: {np.round(popt,4)}")        

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
        Reads directly from the SpectroscopyDataset objects.
        """
        if not hasattr(self, 'dataset_medida') or not hasattr(self, 'dataset_solvente'):
            return
    
        if hasattr(self, "_updating_am_sf") and self._updating_am_sf:
            return
        self._updating_am_sf = True
    
        am = self.slider_am.value() / 100.0
        sf = self.slider_sf.value() / 100.0
        
        interpSol = RegularGridInterpolator(
            (self.dataset_solvente.wl, self.dataset_solvente.td),
            self.dataset_solvente.data,
            bounds_error=False,
            fill_value=0
        )
    
        WL_grid, TD_grid = np.meshgrid(self.dataset_medida.wl, self.dataset_medida.td, indexing="ij")
        points = np.column_stack([WL_grid.ravel(), (TD_grid - sf).ravel()])
        solvente_interp = interpSol(points).reshape(len(self.dataset_medida.wl), len(self.dataset_medida.td)) * am
    
        # Base: measurement - solvent
        base_data = self.dataset_medida.data - solvente_interp
    
        # Apply mask if it exists
        if self.pump_mask is not None:
            base_data[self.pump_mask] = 1e-10
        
        # --- 3. OBJETO FINAL (COMBINADO) ---
        self.dataset = SpectroscopyDataset(base_data, self.dataset_medida.wl, self.dataset_medida.td)
        self.data = self.dataset.data  # Retrocompatibilidad gráfica
    
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
    
       
        if not showing_corrected and getattr(self, 'result_fit', None) is not None:
            fit_x = self.result_fit.get('fit_x')
            fit_y = self.result_fit.get('fit_y')
            if fit_x is not None and fit_y is not None:
                metodo = self.result_fit.get('method', 't₀ fit')
                if getattr(self, 'fit_line_artist', None) is not None:
                    try: self.fit_line_artist.remove()
                    except: pass
                
                current_xlim = self.ax_map.get_xlim()
                self.fit_line_artist, = self.ax_map.plot(fit_x, fit_y, 'r-', lw=2, label=f"Fit: {metodo}")
                self.ax_map.legend()
                self.ax_map.set_xlim(current_xlim)
        # ---------------------------------------------------------------------

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
        """
       
        showing_corrected = getattr(self, "showing_corrected", False)
        active_data = self.data_corrected if showing_corrected and getattr(self, "data_corrected", None) is not None else self.data
        
        if active_data is None or event.inaxes != self.ax_map:
            return
    
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
        
        
        idx_start = getattr(self, 'idx_min', 0)
        idx_end = getattr(self, 'idx_max', len(self.WL) - 1) + 1
        
        wl_plot = self.WL[idx_start:idx_end]
        data_plot = active_data[idx_start:idx_end, :]
        
        idx_wl = np.abs(wl_plot - x).argmin()
        idx_td = np.abs(self.TD - y).argmin()
        
        if idx_wl >= data_plot.shape[0] or idx_td >= data_plot.shape[1]:
            return
    
        # Update Kinetics curve
        y_time = data_plot[idx_wl, :]
        self.cut_time_small.set_data(self.TD, y_time)
        self.vline_time_small.set_xdata([y, y])
        
        # Update Spectrum curve
        y_spec = data_plot[:, idx_td]
        self.cut_spec_small.set_data(wl_plot, y_spec)
    
        # 6. Draw the animated elements
        self.draw_animated_artists()
    
        # 7. Blit
        self.canvas.blit(self.figure.bbox)
        
        # Status bar
        val = data_plot[idx_wl, idx_td]
        self.label_status.setText(f"Cursor: {x:.1f} nm, {y:.2f} ps | ΔA: {val:.4e}")

