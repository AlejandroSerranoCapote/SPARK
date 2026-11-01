# -*- coding: utf-8 -*-
import numpy as np
import os
from scipy import special as _special
from scipy.linalg import expm
from scipy.optimize import lsq_linear


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
        A tuple containing:
        - data_c_cropped (numpy.ndarray): Cropped data matrix.
        - WL_cropped (numpy.ndarray): Cropped wavelength vector.
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
        A tuple containing:
        - data_c_cropped (numpy.ndarray): Cropped data matrix.
        - TD_cropped (numpy.ndarray): Cropped time vector.
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
        A tuple containing:
        - datacAVG (numpy.ndarray): Averaged data matrix.
        - WLAVG (numpy.ndarray): Averaged wavelength vector.
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
    Stable hybrid version with corrected broadcasting.

    Parameters
    ----------
    t : numpy.ndarray
        Time vector.
    t0 : float
        Time zero (center of the IRF).
    taus : array_like
        List or array of lifetimes.
    w : float
        Width (FWHM) of the IRF.

    Returns
    -------
    numpy.ndarray
        Matrix where each column represents the convolved exponential decay 
        for a given lifetime.
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
    
    # 'x' will have size (N_times, M_taus) due to broadcasting
    x = (sigma_safe**2 - tau_safe * t_diff) / (np.sqrt(2) * sigma_safe * tau_safe)
    
    out = np.zeros_like(x)
    

    t_diff_full = np.broadcast_to(t_diff, x.shape)
    tau_full = np.broadcast_to(tau_safe, x.shape)
    
    # REGIME 1: Early times (x >= 0)
    mask_pos = x >= 0
    if np.any(mask_pos):
        exponent_erfcx = - (t_diff_full[mask_pos]**2) / (2 * sigma_safe**2)
        out[mask_pos] = 0.5 * np.exp(exponent_erfcx) * _special.erfcx(x[mask_pos])
        
    # REGIME 2: Late times (x < 0)
    mask_neg = ~mask_pos
    if np.any(mask_neg):
        arg1 = (sigma_safe**2 - 2 * tau_full[mask_neg] * t_diff_full[mask_neg]) / (2 * tau_full[mask_neg]**2)
        out[mask_neg] = 0.5 * np.exp(arg1) * _special.erfc(x[mask_neg])
        
    return out


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
    taus : list or array_like
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


def get_concentration_matrix_global(x_nl, t, numExp, use_art=False, artifact_mode='both'):
    """
    Generates the parallel (global) concentration matrix C.

    Parameters
    ----------
    x_nl : array_like
        Non-linear parameters array (w, t0, taus...).
    t : numpy.ndarray
        Time vector.
    numExp : int
        Number of exponential components.
    use_art : bool, optional
        Whether to include coherent artifact components. Defaults to False.
    artifact_mode : str, optional
        Mode for the artifact ('raman', 'xpm', or 'both'). Defaults to 'both'.

    Returns
    -------
    numpy.ndarray
        The concentration matrix C.
    """
    w, t0 = x_nl[0], x_nl[1]
    taus = x_nl[2:2+numExp]
    C = convolved_exp_vectorized(t, t0, taus, w)
    if use_art: 
        C = np.hstack([C, get_coherent_artifact(t, t0, w, mode=artifact_mode)])
    return C

def get_concentration_matrix_sequential(x_nl, t, numExp, use_art=False, artifact_mode='both'):
    """
    Generates the sequential concentration matrix C.

    Parameters
    ----------
    x_nl : array_like
        Non-linear parameters array (w, t0, taus...).
    t : numpy.ndarray
        Time vector.
    numExp : int
        Number of exponential components.
    use_art : bool, optional
        Whether to include coherent artifact components. Defaults to False.
    artifact_mode : str, optional
        Mode for the artifact ('raman', 'xpm', or 'both'). Defaults to 'both'.

    Returns
    -------
    numpy.ndarray
        The concentration matrix C.
    """
    w, t0 = x_nl[0], x_nl[1]
    taus = x_nl[2:2+numExp]
    pops_list = get_sequential_populations(t, t0, w, taus)
    C = np.column_stack(pops_list)
    if use_art: 
        C = np.hstack([C, get_coherent_artifact(t, t0, w, mode=artifact_mode)])
    return C

def get_concentration_matrix_oscillation(x_nl, t, numExp, use_art=False, artifact_mode='both'):
    """
    Generates the concentration matrix C including a damped oscillation component.

    Parameters
    ----------
    x_nl : array_like
        Non-linear parameters array (w, t0, taus..., alpha, omega, phi).
    t : numpy.ndarray
        Time vector.
    numExp : int
        Number of exponential components.
    use_art : bool, optional
        Whether to include coherent artifact components. Defaults to False.
    artifact_mode : str, optional
        Mode for the artifact ('raman', 'xpm', or 'both'). Defaults to 'both'.

    Returns
    -------
    numpy.ndarray
        The concentration matrix C.
    """
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
    Executes the Variable Projection (VarPro) algorithm.
    
    Allows C to be a single 2D matrix (global t0, ultra-fast path) 
    or a list of 2D matrices (one per wavelength, for independent t0).

    Parameters
    ----------
    C : numpy.ndarray or list of numpy.ndarray
        Concentration matrix or list of concentration matrices.
    data_c_T : numpy.ndarray
        Transposed data matrix.
    enforce_nonneg : bool, optional
        Whether to enforce non-negative least squares. Defaults to False.
    numExp : int, optional
        Number of exponential components (required if enforce_nonneg is True). Defaults to None.

    Returns
    -------
    tuple
        A tuple containing:
        - F (numpy.ndarray): Evaluated matrix model.
        - S_T (numpy.ndarray): Spectra (amplitudes) matrix.
    """
    is_multi_C = isinstance(C, list)
    
   
    if not is_multi_C and not enforce_nonneg:
        S_T, _, _, _ = np.linalg.lstsq(C, data_c_T, rcond=None)
        F = C @ S_T
        return F, S_T
        
    
    num_bases = C[0].shape[1] if is_multi_C else C.shape[1]
    num_wl = data_c_T.shape[1]
    
    S_T = np.zeros((num_bases, num_wl))
    F = np.zeros_like(data_c_T)
    
    if enforce_nonneg and numExp is not None:
        lb = np.full(num_bases, -np.inf)
        lb[:numExp] = 0.0 
        
        for i in range(num_wl):
            C_i = C[i] if is_multi_C else C
            res = lsq_linear(C_i, data_c_T[:, i], bounds=(lb, np.inf), method='bvls')
            S_T[:, i] = res.x
            F[:, i] = C_i @ S_T[:, i]
    else:
        for i in range(num_wl):
            C_i = C[i] if is_multi_C else C
            S_T[:, i], _, _, _ = np.linalg.lstsq(C_i, data_c_T[:, i], rcond=None)
            F[:, i] = C_i @ S_T[:, i]
            
    return F, S_T

def get_coherent_artifact(t, t0, w, mode='both'):
    """
    Generates the mathematical bases for the coherent artifact.
    
    Parameters
    ----------
    t : numpy.ndarray
        Time vector.
    t0 : float
        Time zero.
    w : float
        Width of the IRF.
    mode : str, optional
        - 'raman' : only phi0 (Gaussian IRF). Absorbs spontaneous Raman 
                    and 2-photon absorption.
        - 'xpm'   : only phi1 and phi2 (1st and 2nd derivative of the IRF). Absorbs 
                    Cross-Phase Modulation and dispersive effects.
        - 'both'  : all three bases (default). General broadband TAS case.

    Returns
    -------
    numpy.ndarray
        Array containing the generated artifact bases.
    """
    if t.ndim == 1: 
        t = t[:, np.newaxis]
    sigma = max(w / (2 * np.sqrt(2 * np.log(2))), 1e-12)
    t_diff = t - t0

    irf    = np.exp(-0.5 * (t_diff / sigma)**2)
    irf_d1 = -(t_diff / sigma**2) * irf
    irf_d2 = ((t_diff**2 - sigma**2) / sigma**4) * irf

    # Normalization for numerical stability
    def _norm(x):
        return x / (np.max(np.abs(x)) + 1e-12)

    irf    = _norm(irf)
    irf_d1 = _norm(irf_d1)
    irf_d2 = _norm(irf_d2)

    if mode == 'raman':
        return irf                              
    elif mode == 'xpm':
        return np.hstack([irf_d1, irf_d2])    
    else:  # 'both'
        return np.hstack([irf, irf_d1, irf_d2]) 
    
    

_GL_NODES_32, _GL_WEIGHTS_32 = np.polynomial.legendre.leggauss(32)
_XI_32 = 0.5 * (_GL_NODES_32 + 1.0)
_W_32 = 0.5 * _GL_WEIGHTS_32


def convolved_stretched_exp_vectorized(t, t0, taus, betas, w):
    """
    Calculates a sum of stretched exponential decays exp(-(t/tau)^beta)
    convolved with a Gaussian IRF using a vectorized 32-node Gauss-Legendre
    quadrature with quadratic smoothing over the active IRF support.

    Parameters
    ----------
    t : numpy.ndarray
        Time vector (1D or 2D column).
    t0 : float
        Time zero (center of the IRF).
    taus : array_like
        Array of characteristic lifetimes (tau_i > 0).
    betas : array_like
        Array of stretching exponents (0 < beta_i <= 2).
    w : float
        Width (FWHM) of the Gaussian IRF.

    Returns
    -------
    numpy.ndarray
        Matrix of shape (N_times, N_components) with each convolved stretched decay.
    """
    t_arr = np.asarray(t, dtype=float).ravel()
    taus_arr = np.maximum(np.asarray(taus, dtype=float).ravel(), 1e-12)
    betas_arr = np.clip(np.asarray(betas, dtype=float).ravel(), 0.05, 5.0)
    sigma = max(w / (2.0 * np.sqrt(2.0 * np.log(2.0))), 1e-12)

    t_diff = t_arr[:, np.newaxis] - t0  # (N_t, 1)
    n_t = len(t_arr)
    n_c = len(taus_arr)

    # Ventana donde la IRF gaussiana es no nula (u >= 0)
    a = np.maximum(0.0, t_diff - 6.0 * sigma)
    b = np.maximum(0.0, t_diff + 6.0 * sigma)

    # Cota superior donde exp(-(u/tau)^beta) < exp(-35) ~ 6e-16
    u_max_decay = taus_arr[np.newaxis, :] * (35.0 ** (1.0 / betas_arr[np.newaxis, :]))
    b_eff = np.minimum(b, np.maximum(a, u_max_decay))  # (N_t, N_c)
    a_eff = np.broadcast_to(a, (n_t, n_c))
    span = b_eff - a_eff

    # Transformación u = a_eff + span * xi^2 (elimina la singularidad en u=0 para beta < 1)
    xi = _XI_32[np.newaxis, np.newaxis, :]
    w_q = _W_32[np.newaxis, np.newaxis, :]

    u = a_eff[:, :, np.newaxis] + span[:, :, np.newaxis] * (xi ** 2)
    jac = span[:, :, np.newaxis] * (2.0 * xi)

    gauss = (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(
        -0.5 * ((t_diff[:, :, np.newaxis] - u) / sigma) ** 2
    )
    stretch = np.exp(
        -((u / taus_arr[np.newaxis, :, np.newaxis]) ** betas_arr[np.newaxis, :, np.newaxis])
    )

    return np.sum(w_q * gauss * stretch * jac, axis=2)


def get_concentration_matrix_stretched(x_nl, t, numExp, use_art=False, artifact_mode='both'):
    """
    Generates the stretched exponential (KWW) concentration matrix C.

    Parameters
    ----------
    x_nl : array_like
        Non-linear parameters array [w, t0, tau_1..N, beta_1..N].
    t : numpy.ndarray
        Time vector.
    numExp : int
        Number of stretched exponential components.
    use_art : bool, optional
        Whether to include coherent artifact components. Defaults to False.
    artifact_mode : str, optional
        Mode for the artifact ('raman', 'xpm', or 'both'). Defaults to 'both'.

    Returns
    -------
    numpy.ndarray
        The concentration matrix C.
    """
    w, t0 = x_nl[0], x_nl[1]
    taus = x_nl[2 : 2 + numExp]
    betas = x_nl[2 + numExp : 2 + 2 * numExp]
    C = convolved_stretched_exp_vectorized(t, t0, taus, betas, w)
    if use_art:
        C = np.hstack([C, get_coherent_artifact(t, t0, w, mode=artifact_mode)])
    return C