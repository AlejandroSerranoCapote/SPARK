import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


def eV_a_nm(E_eV):
    """
    Converts energy in electron-volts (eV) to wavelength in nanometers (nm).
    
    Parameters
    ----------
    E_eV : numpy.ndarray or float
        Energy in eV.

    Returns
    -------
    numpy.ndarray or float
        Corresponding wavelength in nm.
    """
    E_eV_safe = np.where(E_eV == 0, 1, E_eV)
    return 1239.841984 / E_eV_safe


def t0_model(w, a, b, c, d):
    """
    Proposed non-linear physical model for time-zero (t0) group-velocity dispersion correction.
    
    Formula: t0 = a * sqrt((b*w^2 - 1) / (c*w^2 - 1)) + d
    
    Parameters
    ----------
    w : array_like
        Wavelength points (nm).
    a : float
        Fitting parameter 'a'.
    b : float
        Fitting parameter 'b'.
    c : float
        Fitting parameter 'c'.
    d : float
        Fitting parameter 'd'.

    Returns
    -------
    numpy.ndarray
        Computed t0 values. Returns NaN where the expression is mathematically invalid.
    """
    w = np.asarray(w, dtype=float)
    num = b * w**2 - 1.0
    den = c * w**2 - 1.0
    
    out = np.full_like(w, np.nan, dtype=float)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = num / den
        valid = (den != 0) & (ratio >= 0)
        out[valid] = a * np.sqrt(ratio[valid]) + d
        
    return out


def apply_t0_correction_poly(popt, WL, TD, data):
    """
    Applies a polynomial time-zero (t0) dispersion correction to the 2D dataset matrix.
    
    Parameters
    ----------
    popt : array_like
        Polynomial coefficients (must be of length 5, degree 4).
    WL : numpy.ndarray
        Wavelength array.
    TD : numpy.ndarray
        Time Delay array.
    data : numpy.ndarray
        2D array of the uncorrected experimental matrix.

    Returns
    -------
    tuple
        A tuple containing:
        - corrected (numpy.ndarray): The t0-corrected 2D data matrix.
        - t0_lambda (numpy.ndarray): The calculated t0 offset vector for each wavelength.
    
    Raises
    ------
    ValueError
        If the polynomial coefficients array does not have a length of exactly 5.
    """
    coeffs = np.asarray(popt)
    if coeffs.size != 5:
        raise ValueError("Polynomial coefficients must have length exactly 5.")
        
    t0_lambda = np.polyval(coeffs, WL)
    corrected = np.zeros_like(data)
    
    for i, wl in enumerate(WL):
        delay_corr = TD - t0_lambda[i]
        # Use 0.0 to fill out-of-bounds shifted data to prevent extrapolation noise
        f = interp1d(delay_corr, data[i, :], kind='linear', bounds_error=False, fill_value=0.0)
        corrected[i, :] = f(TD)
        
    return corrected, t0_lambda


def apply_t0_correction_nonlinear(popt, WL, TD, data):
    """
    Applies a non-linear time-zero (t0) dispersion correction using the `t0_model`.
    
    Parameters
    ----------
    popt : list or array_like
        Optimized parameters [a, b, c, d] for the non-linear model.
    WL : numpy.ndarray
        Wavelength array.
    TD : numpy.ndarray
        Time Delay array.
    data : numpy.ndarray
        2D array of the uncorrected experimental matrix.

    Returns
    -------
    tuple
        A tuple containing:
        - corrected (numpy.ndarray): The t0-corrected 2D data matrix.
        - t0_lambda (numpy.ndarray): The calculated t0 offset vector for each wavelength.
    """
    t0_lambda = t0_model(WL, *popt)
    corrected = data.copy()
    
    for i, t0_val in enumerate(t0_lambda):
        if np.isfinite(t0_val):
            delay_corr = TD - t0_val
            f = interp1d(delay_corr, data[i, :], kind='linear', bounds_error=False, fill_value=0.0)
            corrected[i, :] = f(TD)
        else:
            corrected[i, :] = data[i, :]
            
    return corrected, t0_lambda


def fit_t0(w_points, t0_points, WL, TD, data, min_points_nonlinear=4, mode='auto'):
    """
    Performs non-linear or polynomial least-squares fitting of time-zero (t0) dispersion 
    based on user-selected calibration points.

    Parameters
    ----------
    w_points : array_like
        Wavelengths (nm) of selected calibration points.
    t0_points : array_like
        Corresponding time delays (ps) of selected calibration points.
    WL : numpy.ndarray
        Full wavelength axis array.
    TD : numpy.ndarray
        Full time delay axis array.
    data : numpy.ndarray
        Full 2D experimental dataset matrix.
    min_points_nonlinear : int, optional
        Minimum points required to attempt non-linear fitting. Defaults to 4.
    mode : str, optional
        Strategy mode for the fitting process: 'auto', 'nonlinear', or 'poly'. 
        Defaults to 'auto'.

    Returns
    -------
    dict
        Optimization results containing:
        - 'method': The applied fitting strategy.
        - 'popt': Optimal parameters.
        - 'fit_x': High-resolution x-axis array for plotting the curve.
        - 'fit_y': High-resolution y-axis array for plotting the curve.
        - 'corrected': The completely corrected 2D data matrix.
        - 't0_lambda': The extracted t0 correction vector.
        
    Raises
    ------
    ValueError
        If fewer than 2 data points are provided.
    RuntimeError
        If the forced non-linear mode fails to converge.
    """
    w = np.asarray(w_points, dtype=float)
    t0 = np.asarray(t0_points, dtype=float)

    if w.size < 2:
        raise ValueError("At least 2 points are required for time-zero fitting.")

    fit_x = np.linspace(np.min(w), np.max(w), 400)

    # --- 1. Forced Mode: Polynomial ---
    if mode == 'poly':
        deg = min(4, max(1, w.size - 1)) if w.size < 5 else 4
        coeffs = np.polyfit(w, t0, deg)
        if coeffs.size < 5:
            coeffs = np.concatenate([np.zeros(5 - coeffs.size), coeffs])
            
        fit_y = np.polyval(coeffs, fit_x)
        corrected, t0_lambda = apply_t0_correction_poly(coeffs, WL, TD, data)
        
        return {
            'method': f'poly{deg}',
            'popt': coeffs,
            'fit_x': fit_x,
            'fit_y': fit_y,
            'corrected': corrected,
            't0_lambda': t0_lambda
        }

    # --- 2. Forced Mode: Non-Linear ---
    if mode == 'nonlinear':
        try:
            wmin = np.min(w)
            a0 = (np.nanmax(t0) - np.nanmin(t0)) / 2.0 if np.isfinite(t0).any() else 0.0
            d0 = np.nanmedian(t0)
            min_required = 1.0 / (wmin**2) if wmin != 0 else 1e-8
            b0 = min_required * 1.1
            c0 = min_required * 1.2
            
            p0 = [a0, b0, c0, d0]
            bounds = ([-np.inf, min_required, min_required, -np.inf],
                      [np.inf, np.inf, np.inf, np.inf])
                      
            popt, _ = curve_fit(t0_model, w, t0, p0=p0, bounds=bounds, maxfev=20000, method="trf")
            fit_y = t0_model(fit_x, *popt)
            corrected, t0_lambda = apply_t0_correction_nonlinear(popt, WL, TD, data)
            
            return {
                'method': 'nonlinear',
                'popt': popt,
                'fit_x': fit_x,
                'fit_y': fit_y,
                'corrected': corrected,
                't0_lambda': t0_lambda
            }
        except Exception as e:
            raise RuntimeError(f"Non-linear chirp fit failed: {e}")

    # --- 3. Auto Mode Strategy ---
    try_nl = (w.size >= min_points_nonlinear)
    if try_nl:
        try:
            wmin = np.min(w)
            a0 = (np.nanmax(t0) - np.nanmin(t0)) / 2.0 if np.isfinite(t0).any() else 0.0
            d0 = np.nanmedian(t0)
            min_required = 1.0 / (wmin**2) if wmin != 0 else 1e-8
            b0 = min_required * 1.1
            c0 = min_required * 1.2
            
            p0 = [a0, b0, c0, d0]
            bounds = ([-np.inf, min_required, min_required, -np.inf],
                      [np.inf, np.inf, np.inf, np.inf])
                      
            popt, _ = curve_fit(t0_model, w, t0, p0=p0, bounds=bounds, maxfev=20000, method="trf")
            fit_y = t0_model(fit_x, *popt)
            
            if np.all(np.isfinite(fit_y)):
                corrected, t0_lambda = apply_t0_correction_nonlinear(popt, WL, TD, data)
                return {
                    'method': 'nonlinear',
                    'popt': popt,
                    'fit_x': fit_x,
                    'fit_y': fit_y,
                    'corrected': corrected,
                    't0_lambda': t0_lambda
                }
        except Exception:
            pass  # Fallback to polynomial if non-linear curve fitting fails

    # --- Polynomial Fallback ---
    deg = min(3, max(1, w.size - 1)) if w.size < 5 else 4
    coeffs = np.polyfit(w, t0, deg)
    if coeffs.size < 5:
        coeffs = np.concatenate([np.zeros(5 - coeffs.size), coeffs])
        
    fit_y = np.polyval(coeffs, fit_x)
    corrected, t0_lambda = apply_t0_correction_poly(coeffs, WL, TD, data)
    
    return {
        'method': f'poly{deg}',
        'popt': coeffs,
        'fit_x': fit_x,
        'fit_y': fit_y,
        'corrected': corrected,
        't0_lambda': t0_lambda
    }
