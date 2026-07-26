# -*- coding: utf-8 -*-
import numpy as np
import os
from PyQt5.QtWidgets import QFileDialog
from scipy import special as _special
from scipy.linalg import expm
from scipy.optimize import lsq_linear
def load_npy(parent=None, normalize_per_wl=True):
    """
        Opens a dialog to load a treated data file (.npy).
        
        Parameters
        ----------
        parent : QWidget, optional
            The parent widget for the dialog. Defaults to None.
        normalize_per_wl : bool, optional
            (Currently unused) Flag to normalize per wavelength.
    
        Returns
        -------
        tuple
            A tuple containing:
            - data_c (numpy.ndarray): Data matrix.
            - TD (numpy.ndarray): Time delay vector.
            - WL (numpy.ndarray): Wavele
            ngth vector.
            - base_dir (str): Directory of the selected file.
    
        Raises
        ------
        ValueError
            If the user cancels the file selection.
    """
    file_path, _ = QFileDialog.getOpenFileName(parent, "Select treated data file", "", "NumPy files (*.npy)")
    if not file_path:
        raise ValueError("No file selected")
    
    data = np.load(file_path, allow_pickle=True).item()
    data_c = data['data_c'].astype(float) 
    
    WL = data['WL'].flatten()
    TD = data['TD'].flatten()
    base_dir = os.path.dirname(file_path)
    
    return data_c, TD, WL, base_dir



def crop_spectrum(data_c, WL, WLmin, WLmax):
    """
        Crops the spectral data to a specific wavelength range.
    
        Parameters
        ----------
        data_c : numpy.ndarray
            Original data matrix (Times x Wavelengths).
        WL : numpy.ndarray
            Wavelength vector.
        WLmin : float
            Lower wavelength limit.
        WLmax : float
            Upper wavelength limit.
    
        Returns
        -------
        tuple
            Cropped data matrix and cropped wavelength vector.
    """
    mask = (WL >= WLmin) & (WL <= WLmax)
    return data_c[:, mask], WL[mask] 

def crop_kinetics(data_c, TD, TDmin, TDmax):
    """
        Crops the kinetics to a specific time range.
    
        Parameters
        ----------
        data_c : numpy.ndarray
            Original data matrix (Times x Wavelengths).
        TD : numpy.ndarray
            Time delay vector.
        TDmin : float
            Lower time limit.
        TDmax : float
            Upper time limit.
    
        Returns
        -------
        tuple
            Cropped data matrix and cropped time vector.
    """
    mask = (TD >= TDmin) & (TD <= TDmax)
    return data_c[:, mask], TD[mask] 

def binning(data_c, WL, bin_size):
    """
        Bins adjacent wavelength channels to improve the signal-to-noise ratio.
    
        Parameters
        ----------
        data_c : numpy.ndarray
            Original data matrix.
        WL : numpy.ndarray
            Wavelength vector.
        bin_size : int
            Number of channels to bin together.
    
        Returns
        -------
        tuple
            Averaged data matrix and averaged wavelength vector.
    """
    numWL = len(WL) // bin_size
    datacAVG = np.zeros((numWL, data_c.shape[1]))
    WLAVG = np.zeros(numWL)
    for i in range(numWL):
        datacAVG[i, :] = np.mean(data_c[i*bin_size:(i+1)*bin_size, :], axis=0)
        WLAVG[i] = np.mean(WL[i*bin_size:(i+1)*bin_size])
    return datacAVG, WLAVG

import numpy as np
import scipy.special as _special

def convolved_exp_vectorized(t, t0, taus, w):
    """
    Calculates a sum of exponential decays convolved with a Gaussian IRF.
    VERSIÓN HÍBRIDA ESTABLE CON BROADCASTING CORREGIDO.
    """
    if t.ndim == 1:
        t = t[:, np.newaxis]
        
    taus = np.asarray(taus)
    if taus.ndim == 1:
        taus = taus[np.newaxis, :]
    
    sigma = w / (2 * np.sqrt(2 * np.log(2)))
    
    tau_safe = np.maximum(taus, 1e-12)
    sigma_safe = np.maximum(sigma, 1e-12)
    
    t_diff = t - t0
    
    # 'x' tendrá tamaño (N_tiempos, M_taus) debido al broadcasting
    x = (sigma_safe**2 - tau_safe * t_diff) / (np.sqrt(2) * sigma_safe * tau_safe)
    
    out = np.zeros_like(x)
    

    t_diff_full = np.broadcast_to(t_diff, x.shape)
    tau_full = np.broadcast_to(tau_safe, x.shape)
    
    # RÉGIMEN 1: Tiempos tempranos (x >= 0)
    mask_pos = x >= 0
    if np.any(mask_pos):
        exponent_erfcx = - (t_diff_full[mask_pos]**2) / (2 * sigma_safe**2)
        out[mask_pos] = 0.5 * np.exp(exponent_erfcx) * _special.erfcx(x[mask_pos])
        
    # RÉGIMEN 2: Tiempos tardíos (x < 0)
    mask_neg = ~mask_pos
    if np.any(mask_neg):
        arg1 = (sigma_safe**2 - 2 * tau_full[mask_neg] * t_diff_full[mask_neg]) / (2 * tau_full[mask_neg]**2)
        out[mask_neg] = 0.5 * np.exp(arg1) * _special.erfc(x[mask_neg])
        
    return out
# =============================================================================
# MODEL EVALUATION FUNCTIONS
# =============================================================================

def get_sequential_populations(t, t0, w, taus):
    """ 
        Calculates the populations for a sequential model (A -> B -> C...).
        
        Uses a dynamic Bateman equations generator to support any number 
        of exponential components.
    
        Parameters
        ----------
        t : numpy.ndarray
            Time vector.
        t0 : float
            Time zero.
        w : float
            Width of the IRF.
        taus : list
            List of lifetimes for each sequential species.
    
        Returns
        -------
        list of numpy.ndarray
            List where each element is the population over time for the corresponding species.
    """
    k = 1.0 / np.asarray(taus) # Rates (k = 1/tau)
    pops = []
    
    # Get all basic exponentials at once
    E_matrix = convolved_exp_vectorized(t, t0, taus, w)
    E = [E_matrix[:, i] for i in range(len(taus))]
    
    num_species = len(taus)
    
    for i in range(num_species):
        if i == 0:
            # Species 1 is always just the first exponential decay
            pops.append(E[0])
        else:
            # For Species i (where i > 0), calculate the Bateman equation dynamically
            
            # 1. Product of all previous rates: k_0 * k_1 * ... * k_{i-1}
            rate_prod = np.prod(k[:i])
            
            # 2. Sum over all exponentials up to i
            species_pop = np.zeros_like(E[0])
            for j in range(i + 1):
                # Calculate the denominator product: prod(k_m - k_j) for m != j
                denom = 1.0
                for m in range(i + 1):
                    if m != j:
                        diff = k[m] - k[j]
                        # Safety for degenerate rates (if two taus are too similar)
                        if abs(diff) < 1e-12: 
                            diff = 1e-12 if diff >= 0 else -1e-12
                        denom *= diff
                
                # Add the term to the sum
                species_pop += E[j] / denom
            
            # 3. Final population for species i
            pops.append(rate_prod * species_pop)

    return pops



def damped_oscillation(t, t0, alpha, omega, phi, w):
    """
        Calculates a damped oscillation with a smooth step (approximating IRF convolution).
    
        Equation used:
        $S(t) = 0.5 \cdot (1 + \text{erf}((t-t0)/(\sqrt{2}w))) \cdot \exp(-\alpha(t-t0)) \cdot \sin(\omega(t-t0) + \phi)$
    
        Parameters
        ----------
        t : numpy.ndarray
            Time vector.
        t0 : float
            Time zero (start of the oscillation).
        alpha : float
            Damping rate.
        omega : float
            Angular frequency of the oscillation.
        phi : float
            Initial phase (in radians).
        w : float
            Width of the IRF (controls the smoothness of the onset).
    
        Returns
        -------
        numpy.ndarray
            Vector with the damped oscillatory signal.
    """
    t_shifted = t - t0
    
    # 1. Safety Mask: Prevent exp() overflow for very negative times.
    safe_mask = t_shifted > -6 * w
    
    osc = np.zeros_like(t_shifted)
    
    # Only calculate where it is numerically safe
    ts_safe = t_shifted[safe_mask]
    
    # Smooth Step (Simulates convolution with Gaussian IRF)
    # Using erf here is standard for step smoothing
    step = 0.5 * (1 + _special.erf(ts_safe / (np.sqrt(2) * w)))
    
    # Damped Sine
    decay = np.exp(-alpha * ts_safe)
    sine = np.sin(omega * ts_safe + phi)
    
    osc[safe_mask] = step * decay * sine
    
    return osc



# =============================================================================
# NUEVO MOTOR VARPRO: Generadores de Matrices de Concentración (C)
# =============================================================================
def get_concentration_matrix_global(x_nl, t, numExp, use_art=False, artifact_mode='both'):
    w, t0 = x_nl[0], x_nl[1]
    taus = x_nl[2:2+numExp]
    C = convolved_exp_vectorized(t, t0, taus, w)
    if use_art: 
        C = np.hstack([C, get_coherent_artifact(t, t0, w, mode=artifact_mode)])
    return C

def get_concentration_matrix_sequential(x_nl, t, numExp, use_art=False, artifact_mode='both'):
    w, t0 = x_nl[0], x_nl[1]
    taus = x_nl[2:2+numExp]
    pops_list = get_sequential_populations(t, t0, w, taus)
    C = np.column_stack(pops_list)
    if use_art: 
        C = np.hstack([C, get_coherent_artifact(t, t0, w, mode=artifact_mode)])
    return C

def get_concentration_matrix_oscillation(x_nl, t, numExp, use_art=False, artifact_mode='both'):
    w, t0 = x_nl[0], x_nl[1]
    taus = x_nl[2:2+numExp]
    alpha, omega, phi = x_nl[2+numExp], x_nl[2+numExp+1], x_nl[2+numExp+2]
    basis_exp = convolved_exp_vectorized(t, t0, taus, w)
    basis_osc = damped_oscillation(t, t0, alpha, omega, phi, w).reshape(-1, 1)
    C = np.hstack([basis_exp, basis_osc])
    if use_art: 
        C = np.hstack([C, get_coherent_artifact(t, t0, w, mode=artifact_mode)])
    return C


def eval_varpro_model(C, data_c_T, enforce_nonneg=False, numExp=None):
    """
    Ejecuta la Proyección Variable.
    Si enforce_nonneg es True, fuerza a que las amplitudes de las especies (SAS)
    sean >= 0, pero permite que el artefacto coherente fluya libremente.
    """
    if enforce_nonneg and numExp is not None:
        S_T = np.zeros((C.shape[1], data_c_T.shape[1]))
        
        # Límites: [0, infinito] para las especies exponenciales
        # [-infinito, infinito] para el artefacto coherente o la oscilación
        lb = np.full(C.shape[1], -np.inf)
        lb[:numExp] = 0.0 
        
        # Lazo súper optimizado usando el método Bounded Variables (BVLS)
        for i in range(data_c_T.shape[1]):
            res = lsq_linear(C, data_c_T[:, i], bounds=(lb, np.inf), method='bvls')
            S_T[:, i] = res.x
            
        F = C @ S_T
        return F, S_T
    else:
        # Mínimos cuadrados lineales estándar (Sin límites)
        S_T, _, _, _ = np.linalg.lstsq(C, data_c_T, rcond=None)
        F = C @ S_T
        return F, S_T

def get_coherent_artifact(t, t0, w, mode='both'):
    """
    Genera las bases matemáticas para el artefacto coherente.
    
    mode : str
        'raman'  → solo φ₀ (IRF gaussiana). Absorbe Raman espontáneo 
                   y absorción de 2 fotones.
        'xpm'    → solo φ₁ y φ₂ (1ª y 2ª derivada de la IRF). Absorbe 
                   Cross-Phase Modulation y efectos dispersivos.
        'both'   → las tres bases (defecto). Caso general TAS broadband.
    """
    if t.ndim == 1: 
        t = t[:, np.newaxis]
    sigma = max(w / (2 * np.sqrt(2 * np.log(2))), 1e-12)
    t_diff = t - t0

    irf    = np.exp(-0.5 * (t_diff / sigma)**2)
    irf_d1 = -(t_diff / sigma**2) * irf
    irf_d2 = ((t_diff**2 - sigma**2) / sigma**4) * irf

    # Normalización para estabilidad numérica
    def _norm(x):
        return x / (np.max(np.abs(x)) + 1e-12)

    irf    = _norm(irf)
    irf_d1 = _norm(irf_d1)
    irf_d2 = _norm(irf_d2)

    if mode == 'raman':
        return irf                              # 1 base
    elif mode == 'xpm':
        return np.hstack([irf_d1, irf_d2])     # 2 bases
    else:  # 'both'
        return np.hstack([irf, irf_d1, irf_d2]) # 3 bases (comportamiento anterior)

# =============================================================================
# CONSTRUCTOR AUTOMÁTICO DE MODELOS CINÉTICOS (K-MATRIX MANAGER)
# =============================================================================

class KMatrixModel:
    def __init__(self, name="Modelo Personalizado"):
        self.name = name
        self.states = []        # Lista de nombres de estados excitados
        self.transitions = []   # Lista de tuplas: (origen, destino, tipo_param, label_param)
        self.param_labels = []  # Nombres ordenados de los parámetros no lineales
        
    def add_state(self, state_name):
        """Añade un estado físico al sistema (ej. 'S1*', 'S1', '3CT')"""
        if state_name not in self.states:
            self.states.append(state_name)
            
    def add_transition(self, source, target, param_type="tau", label=""):
        """Añade una transferencia de población entre dos estados."""
        self.add_state(source)
        if target != "S0": # S0 es el estado fundamental
            self.add_state(target)
        self.transitions.append((source, target, param_type, label))
        
    def build_parameter_list(self):
        """Analiza las transiciones y extrae la lista única de parámetros a optimizar."""
        labels = []
        for src, tgt, p_type, label in self.transitions:
            if label and label not in labels:
                labels.append(label)
        self.param_labels = labels
        return self.param_labels

    def get_default_guesses_and_bounds(self):
        """Generar automáticamente los valores iniciales y límites escalonados."""
        self.build_parameter_list()
        ini, low, upp = [], [], []
        
        # Valores iniciales escalonados para ayudar al optimizador a ordenar las taus
        tau_defaults = [1.0, 15.0, 200.0, 1500.0, 5000.0]
        tau_idx = 0
        
        for label in self.param_labels:
            p_type = next(t[2] for t in self.transitions if t[3] == label)
            if p_type == "tau":
                val = tau_defaults[tau_idx] if tau_idx < len(tau_defaults) else 10.0
                ini.append(val)
                low.append(0.001)     
                upp.append(1e8)       
                tau_idx += 1
            elif p_type == "gamma":
                ini.append(0.5)       
                low.append(0.0)       
                upp.append(1.0)       
                
        return np.array(ini), np.array(low), np.array(upp)


    def get_concentration_matrix(self, x_nl_params, t, w, t0, use_art=False, artifact_mode='both'):
            """Toma los parámetros, construye la matriz K, la diagonaliza y devuelve poblaciones."""
            N = len(self.states)
            K = np.zeros((N, N))
            
            p_dict = dict(zip(self.param_labels, x_nl_params))
            
            tau_totals = {}
            for src, tgt, p_type, label in self.transitions:
                if p_type == "tau":
                    tau_totals[src] = p_dict[label]
    
            for src, tgt, p_type, label in self.transitions:
                idx_src = self.states.index(src)
                val_param = p_dict[label]
                
                if p_type == "tau":
                    k_val = 1.0 / max(val_param, 1e-12)
                    K[idx_src, idx_src] -= k_val 
                    if tgt != "S0":
                        idx_tgt = self.states.index(tgt)
                        K[idx_tgt, idx_src] += k_val 
                        
                elif p_type == "gamma":
                    tau_total = tau_totals.get(src, 1.0)
                    k_total = 1.0 / max(tau_total, 1e-12)
                    k_via = val_param * k_total
                    
                    if tgt != "S0":
                        idx_tgt = self.states.index(tgt)
                        K[idx_tgt, idx_src] += k_via
                        
            # Perturbación determinista para evitar autovalores degenerados
            diag_vals = np.diagonal(K).copy()
            eps_base = 1e-9
            perturbacion = eps_base * np.arange(N)
            np.fill_diagonal(K, diag_vals * (1 + perturbacion))
            
            eigenvalues, V = np.linalg.eig(K)   
            
            try:
                V_inv = np.linalg.inv(V)
            except np.linalg.LinAlgError:
                V_inv = np.linalg.pinv(V)
    
            targets = [tgt for src, tgt, p_type, label in self.transitions]
            roots = [s for s in self.states if s not in targets]
            
            P0 = np.zeros(N)
            if roots:
                # --- Poblar TODOS los estados iniciales ---
                for root_state in roots:
                    idx_p0 = self.states.index(root_state)
                    P0[idx_p0] = 1.0
            else:
                idx_p0 = 0
                P0[idx_p0] = 1.0
            
            c = V_inv @ P0
            
            eigenvalues = np.real(eigenvalues)
            taus_eff = np.zeros_like(eigenvalues)
            
            for i, ev in enumerate(eigenvalues):
                if ev > -1e-12:
                    taus_eff[i] = 1e8
                else:
                    taus_eff[i] = -1.0 / ev
                    
            E_matrix = convolved_exp_vectorized(t, t0, taus_eff, w)
            
            pops = []
            for i in range(N):
                state_pop = np.zeros_like(t)
                for j in range(N):
                    peso = np.real(V[i, j] * c[j])
                    state_pop += peso * E_matrix[:, j]
                pops.append(state_pop)
                
            C = np.column_stack(pops)
            
            if use_art: 
                C = np.hstack([C, get_coherent_artifact(t, t0, w, mode=artifact_mode)])
                
            return C
