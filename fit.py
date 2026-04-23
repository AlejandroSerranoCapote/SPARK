# -*- coding: utf-8 -*-
import numpy as np
import os
from PyQt5.QtWidgets import QFileDialog
from scipy import special as _special
from scipy.linalg import expm

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

def convolved_exp_vectorized(t, t0, taus, w):
    """
        Calculates a sum of exponential decays convolved with a Gaussian IRF.
        
        HYBRID Version: Vectorized but using 'erf' (faster than erfc).
        Optimized for speed using NumPy broadcasting.
    
        Parameters
        ----------
        t : numpy.ndarray
            Time delay vector.
        t0 : float
            Time zero (center of the IRF).
        taus : list or numpy.ndarray
            List of exponential lifetimes.
        w : float
            Width of the Instrument Response Function (IRF, related to FWHM).
    
        Returns
        -------
        numpy.ndarray
            Matrix of convolved exponential shapes evaluated at t.
    """
    # t -> (N, 1)
    if t.ndim == 1:
        t = t[:, np.newaxis]
        
    # taus -> (1, M)
    taus = np.asarray(taus)
    if taus.ndim == 1:
        taus = taus[np.newaxis, :]
    
    # Constantes pequeñas para evitar división por cero
    tau_safe = np.maximum(taus, 1e-12)
    w_safe = np.maximum(w, 1e-12)
    
    # arg1 = (w^2 - 2*tau*(t-t0)) / (2*tau^2)
    # Factorizamos para reducir operaciones:
    t_diff = t - t0
    w2 = w_safe**2
    
    # Operación matricial (Broadcasting)
    arg1 = (w2 - 2 * tau_safe * t_diff) / (2 * tau_safe**2)
    arg2 = (w2 - tau_safe * t_diff) / (np.sqrt(2) * w_safe * tau_safe)

    # Mantenemos el clip en 700 que funciona bien con float64
    arg1 = np.clip(arg1, -700, 700)
    
    # Fórmula: 0.5 * exp(arg1) * (1 - erf(arg2))
    return 0.5 * np.exp(arg1) * (1 - _special.erf(arg2))

# =============================================================================
# MODEL EVALUATION FUNCTIONS
# =============================================================================

def eval_global_model(x, t, numExp, numWL, t0_choice_str):
    """
        Evaluates the global parallel model (DAS - Decay Associated Spectra).
    
        Calculates the 2D data surface (Time x Wavelength) based on 
        shared lifetimes and independent amplitudes per wavelength.
    
        Parameters
        ----------
        x : list or numpy.ndarray
            Array of fitting parameters (w, t0, taus, amplitudes).
        t : numpy.ndarray
            Time delay vector.
        numExp : int
            Number of exponential components.
        numWL : int
            Number of wavelengths to fit.
        t0_choice_str : str
            'Yes' for fitting with chirp correction (variable t0 per wavelength),
            any other value for standard global fitting (single shared t0).
    
        Returns
        -------
        numpy.ndarray
            Simulated data matrix (Time x Wavelength).
    """
    F = np.zeros((len(t), numWL))
    
    if t0_choice_str == 'Yes': # CHIRP MODE
        w = x[0]
        taus = x[1:1+numExp] # Array de todos los Taus
        base_idx = 1 + numExp
        
        for j in range(numWL):
            idx = base_idx + j*(numExp+1)
            t0 = x[idx]
            Amps = x[idx+1 : idx+1+numExp] # Array de Amplitudes (shape: numExp,)
            
            # 1. Calculamos TODAS las bases exponenciales de golpe para este t0
            # Devuelve matriz (Tiempo x numExp) directamente.
            bases = convolved_exp_vectorized(t, t0, taus, w)
            
            # 2. Multiplicamos matricialmente Bases @ Amplitudes
            # (N_t, N_exp) @ (N_exp,) -> (N_t,)  (Vector plano)
            F[:, j] = bases @ Amps

    else: # GLOBAL FIT (Optimización Máxima)
        w = x[0]
        t0 = x[1]
        taus = x[2:2+numExp]
        A_base = 2 + numExp
        
        # Devuelve directamente la matriz (Tiempo x numExp)
        basis_functions = convolved_exp_vectorized(t, t0, taus, w)
            
        # Extraer Amplitudes
        all_Amps = x[A_base:].reshape(numWL, numExp).T 
        
        # Multiplicación
        F = basis_functions @ all_Amps
        
    return F

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

def eval_sequential_model(x, t, numExp, numWL, t0_choice_str):
    """
        Evaluates the sequential model (SAS - Species Associated Spectra).
    
        Parameters
        ----------
        x : list
            Parameter array.
        t : numpy.ndarray
            Time vector.
        numExp : int
            Number of species in the cascade.
        numWL : int
            Number of wavelengths.
        t0_choice_str : str
            'Yes' for chirp correction mode.
    
        Returns
        -------
        numpy.ndarray
            Simulated data matrix.
    """
    F = np.zeros((len(t), numWL))
    
    if t0_choice_str == 'Yes': 
        w = x[0]
        taus = x[1:1+numExp]
        base_idx = 1 + numExp
        
        for j in range(numWL):
            idx = base_idx + j*(numExp+1)
            t0 = x[idx]
            sas_coeffs = x[idx+1 : idx+1+numExp] 
            
            pops_list = get_sequential_populations(t, t0, w, taus)
            
            # Manual dot product for the single WL
            kinetics = np.zeros_like(t)
            for n in range(numExp):
                kinetics += sas_coeffs[n] * pops_list[n]
            
            F[:, j] = kinetics

    else: 
        w = x[0]
        t0 = x[1]
        taus = x[2:2+numExp]
        A_base = 2 + numExp
        
        # 1. Basis Functions (Time x Species)
        pops_list = get_sequential_populations(t, t0, w, taus)
        basis_functions = np.column_stack(pops_list) 
        
        # 2. Amplitudes / SAS (Species x WL)
        all_SAS = x[A_base:].reshape(numWL, numExp).T 
        
        # 3. Matrix Multiplication
        F = basis_functions @ all_SAS
        
    return F

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

def eval_oscillation_model(x, t, numExp, numWL, t0_choice_str):
    """
        Evaluates a model combining parallel exponential decays and a damped oscillation.
    
        Parameters
        ----------
        x : list
            Model parameters (w, t0, taus, alpha, omega, phi, local amplitudes).
        t : numpy.ndarray
            Time vector.
        numExp : int
            Number of exponential decays.
        numWL : int
            Number of wavelengths.
        t0_choice_str : str
            Must be 'No' (Chirp is not yet implemented for this model).
    
        Returns
        -------
        numpy.ndarray
            Simulated data matrix.
    
        Raises
        ------
        NotImplementedError
            If attempted to use with `t0_choice_str == 'Yes'`.
    """
    F = np.zeros((len(t), numWL))
    
    if t0_choice_str == 'Yes':
        raise NotImplementedError("Chirp not implemented for Oscillation model yet.")

    # --- Global Parameters ---
    w = x[0]
    t0 = x[1]
    taus = x[2:2+numExp]
    
    # Oscillation parameters
    alpha = x[2+numExp]
    omega = x[2+numExp+1]
    phi   = x[2+numExp+2]
    
    A_base = 2 + numExp + 3 
    
    # 1. Decay Bases (Vectorized) -> Matrix (Time x Exp)
    basis_exp = convolved_exp_vectorized(t, t0, taus, w)
        
    # 2. Oscillation Basis -> Matrix (Time x 1)
    basis_osc = damped_oscillation(t, t0, alpha, omega, phi, w).reshape(-1, 1)
    
    # 3. Stack all bases: [T x (numExp + 1)]
    all_bases = np.hstack([basis_exp, basis_osc])
    
    # 4. Extract Local Amplitudes [A1...An, B] per wavelength
    num_local_params = numExp + 1
    all_amps = x[A_base:].reshape(numWL, num_local_params).T
    
    # 5. Matrix Multiplication
    F = all_bases @ all_amps
    
    return F



