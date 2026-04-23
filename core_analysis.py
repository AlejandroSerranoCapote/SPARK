# core_analysis.py
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


def read_csv_file(path):
    """
    Reads and cleans data from a single CSV file.

    It assumes the first column contains Wavelengths (WL) and the headers of the 
    subsequent columns represent Time Delays (TD).
    
    Args:
        path (str): The file path to the CSV file.

    Returns:
        tuple: A tuple containing:
            - WL (numpy.ndarray): 1D array of Wavelengths.
            - TD (numpy.ndarray): 1D array of Time Delays.
            - data (numpy.ndarray): 2D array of the main data matrix with shape (n_wl, n_td).
    """
    df = pd.read_csv(path)
    
    # Extract Wavelengths (WL) from the first column, coercing errors to NaN
    WL_raw = pd.to_numeric(df.iloc[:, 0], errors='coerce')
    valid_rows = WL_raw.notna()
    WL = WL_raw[valid_rows].to_numpy()
    
    TD = []
    valid_cols = []
    
    # Extract Time Delays (TD) from the column headers
    for col in df.columns[1:]:
        try:
            TD.append(float(col))
            valid_cols.append(col)
        except Exception:
            continue
            
    TD = np.array(TD)
    
    # Extract the main data matrix, fill NaNs with 0, and convert to numpy array
    data = df.loc[valid_rows, valid_cols].apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy()

    return WL, TD, data


def load_from_paths(data_path, wl_path, td_path):
    """
    Loads data from three separate files: main data matrix, wavelengths, and time delays.
    
    This function automatically detects the orientation of the data matrix and 
    transposes it if necessary to match the sizes of the WL and TD arrays. 
    If dimensions do not match perfectly, it pads with zeros or truncates as needed.
    
    Args:
        data_path (str): Path to the 2D data matrix file.
        wl_path (str): Path to the 1D wavelength array file.
        td_path (str): Path to the 1D time delay array file.

    Returns:
        tuple: A tuple containing:
            - data_arr (numpy.ndarray): 2D array with shape (n_wl, n_td).
            - wl (numpy.ndarray): 1D array of Wavelengths.
            - td (numpy.ndarray): 1D array of Time Delays.
    """
    
    # ---- Load files ----
    try:
        # Attempt to read data using pandas for flexibility with delimiters
        df = pd.read_csv(data_path, header=None, sep=None, engine='python')
        data = df.values
    except Exception:
        # Fallback to numpy loadtxt if pandas fails
        data = np.loadtxt(data_path)
    
    wl = np.loadtxt(wl_path)
    td = np.loadtxt(td_path)
    
    # Replace NaNs with 0 to prevent downstream calculation errors
    data = np.nan_to_num(data, nan=0.0)
    wl = np.nan_to_num(wl, nan=0.0)
    td = np.nan_to_num(td, nan=0.0)
    
    # ---- Detect Matrix Orientation ----
    nwl, ntd = wl.size, td.size
    
    if data.shape == (nwl, ntd):
        data_arr = data.copy()
    elif data.shape == (ntd, nwl):
        data_arr = data.T.copy()
    else:
        # Attempt reshaping if the total number of elements matches
        if data.size == nwl * ntd:
            data_arr = data.reshape((nwl, ntd))
        else:
            # Pad with zeros or truncate if dimensions are completely off
            data_arr = np.zeros((nwl, ntd))
            r = min(data.shape[0], nwl)
            c = min(data.shape[1], ntd)
            data_arr[:r, :c] = data[:r, :c]
            print(f"Warning: data shape {data.shape} does not match (n_wl, n_td). Padded/truncated to {(nwl, ntd)}.")
    
    return data_arr, wl, td


def load_data(auto_path=None, data_path=None, wl_path=None, td_path=None):
    """
    Master function to load data either from a single comprehensive CSV or from three separate files.
    
    Args:
        auto_path (str, optional): Path to a single CSV containing WL, TD, and data.
        data_path (str, optional): Path to the 2D data matrix file.
        wl_path (str, optional): Path to the 1D wavelength array file.
        td_path (str, optional): Path to the 1D time delay array file.

    Returns:
        tuple: data (2D), WL (1D), TD (1D)

    Raises:
        ValueError: If no valid files are provided or reading fails entirely.
    """
    import os

    # Attempt to load from a single combined CSV first
    if auto_path is not None and os.path.isfile(auto_path):
        try:
            WL, TD, data = read_csv_file(auto_path)
            return data, WL, TD
        except Exception:
            pass  # Fallback to the three-file method if single CSV parsing fails

    # Attempt to load from three separate files
    if data_path and wl_path and td_path:
        data, WL, TD = load_from_paths(data_path, wl_path, td_path)
        return data, WL, TD

    raise ValueError("No valid files provided, or files could not be read.")


# ---------------------------------------------------------------------
# Models and Correction Functions
# ---------------------------------------------------------------------

def eV_a_nm(E_eV):
    """
    Converts energy in electron-volts (eV) to wavelength in nanometers (nm).
    
    Args:
        E_eV (numpy.ndarray or float): Energy in eV.

    Returns:
        numpy.ndarray or float: Corresponding wavelength in nm.
    """
    # Prevent division by zero by temporarily replacing 0 with 1
    E_eV_safe = np.where(E_eV == 0, 1, E_eV)
    return 1239.841984 / E_eV_safe


def t0_model(w, a, b, c, d):
    """
    Proposed non-linear model for time-zero (t0) dispersion correction.
    
    Formula: t0 = a * sqrt((b*w^2 - 1) / (c*w^2 - 1)) + d
    
    Args:
        w (array-like): Wavelength points.
        a, b, c, d (float): Fitting parameters.

    Returns:
        numpy.ndarray: Computed t0 values. Returns NaN where the expression is mathematically invalid.
    """
    w = np.asarray(w, dtype=float)
    num = b * w**2 - 1.0
    den = c * w**2 - 1.0
    
    out = np.full_like(w, np.nan, dtype=float)
    
    # Suppress warnings for expected invalid operations (like negative roots or division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = num / den
        valid = (den != 0) & (ratio >= 0)
        out[valid] = a * np.sqrt(ratio[valid]) + d
        
    return out


def apply_t0_correction_poly(popt, WL, TD, data):
    """
    Applies a polynomial time-zero (t0) correction to the dataset.
    
    Args:
        popt (array-like): Polynomial coefficients (must be of length 5, degree 4).
        WL (numpy.ndarray): Wavelength array.
        TD (numpy.ndarray): Time Delay array.
        data (numpy.ndarray): 2D array of the main data matrix.

    Returns:
        tuple: A tuple containing:
            - corrected (numpy.ndarray): The t0-corrected 2D data matrix.
            - t0_lambda (numpy.ndarray): The calculated t0 offset for each wavelength.
    """
    coeffs = np.asarray(popt)
    if coeffs.size != 5:
        raise ValueError("Polynomial coefficients must have length 5.")
        
    t0_lambda = np.polyval(coeffs, WL)
    corrected = np.zeros_like(data)
    
    for i, wl in enumerate(WL):
        delay_corr = TD - t0_lambda[i]
        
        # SOLUTION: Strictly use 0.0 to fill out-of-bounds data to prevent extrapolation artifacts
        f = interp1d(delay_corr, data[i, :], kind='linear', bounds_error=False, fill_value=0.0)
        corrected[i, :] = f(TD)
        
    return corrected, t0_lambda


def apply_t0_correction_nonlinear(popt, WL, TD, data):
    """
    Applies a non-linear time-zero (t0) correction to the dataset using `t0_model`.
    
    Args:
        popt (list or array-like): Optimized parameters [a, b, c, d] for the non-linear model.
        WL (numpy.ndarray): Wavelength array.
        TD (numpy.ndarray): Time Delay array.
        data (numpy.ndarray): 2D array of the main data matrix.

    Returns:
        tuple: A tuple containing:
            - corrected (numpy.ndarray): The t0-corrected 2D data matrix.
            - t0_lambda (numpy.ndarray): The calculated t0 offset for each wavelength.
    """
    t0_lambda = t0_model(WL, *popt)
    corrected = data.copy()
    
    for i, t0_val in enumerate(t0_lambda):
        if np.isfinite(t0_val):
            delay_corr = TD - t0_val
            
            # SOLUTION: Strictly use 0.0 to fill out-of-bounds data to prevent extrapolation artifacts
            f = interp1d(delay_corr, data[i, :], kind='linear', bounds_error=False, fill_value=0.0)
            corrected[i, :] = f(TD)
        else:
            # If the calculated t0 is not finite, leave the data uncorrected for that wavelength
            corrected[i, :] = data[i, :]
            
    return corrected, t0_lambda


def fit_t0(w_points, t0_points, WL, TD, data, min_points_nonlinear=4, mode='auto'):
    """
    Fits time-zero (t0) dispersion based on user-selected points (w_points, t0_points).
    
    The function attempts to fit the non-linear model (`t0_model`) if there are enough points.
    If the non-linear fit fails or lacks sufficient points, it falls back to a polynomial 
    fit of up to degree 4.

    Args:
        w_points (array-like): Wavelengths (nm) of the chosen data points.
        t0_points (array-like): Corresponding delays (ps) of the chosen data points.
        WL (numpy.ndarray): Full wavelength array (from read_csv_file or load_from_paths).
        TD (numpy.ndarray): Full time delay array.
        data (numpy.ndarray): Full 2D data matrix.
        min_points_nonlinear (int): Minimum number of points required to attempt the non-linear model. Defaults to 4.
        mode (str): Fitting mode strategy. 'auto' (default), 'nonlinear' (force non-linear), or 'poly' (force polynomial).

    Returns:
        dict: A dictionary containing the fitting results:
            - 'method': String indicating the method used ('nonlinear' or 'polyX').
            - 'popt': The optimized coefficients/parameters.
            - 'fit_x': High-resolution X array used for plotting the fit line.
            - 'fit_y': High-resolution Y array representing the fit curve.
            - 'corrected': The newly corrected 2D data matrix.
            - 't0_lambda': The calculated t0 vector for every wavelength in WL.
    """
    w = np.asarray(w_points, dtype=float)
    t0 = np.asarray(t0_points, dtype=float)

    if w.size < 2:
        raise ValueError("At least 2 points are required for fitting (ideally >= 4 for the non-linear model).")

    # High-resolution array strictly for smooth plotting of the fit curve
    fit_x = np.linspace(np.min(w), np.max(w), 400)

    # =======================================================
    # --- Forced Mode: Polynomial ---------------------------
    # =======================================================
    if mode == 'poly':
        if w.size < 5:
            deg = min(4, max(1, w.size - 1))
        else:
            deg = 4
            
        coeffs = np.polyfit(w, t0, deg)
        
        # Ensure coefficients array is strictly length 5 (pad with leading zeros)
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

    # =======================================================
    # --- Forced Mode: Non-Linear ---------------------------
    # =======================================================
    if mode == 'nonlinear':
        try:
            wmin = np.min(w)
            # Initial parameter estimates (p0)
            a0 = (np.nanmax(t0) - np.nanmin(t0)) / 2.0 if np.isfinite(t0).any() else 0.0
            d0 = np.nanmedian(t0)
            min_required = 1.0 / (wmin**2) if wmin != 0 else 1e-8
            b0 = min_required * 1.1
            c0 = min_required * 1.2
            
            p0 = [a0, b0, c0, d0]
            
            # Bound constraints to ensure stability
            bounds = ([-np.inf, min_required, min_required, -np.inf],
                      [np.inf, np.inf, np.inf, np.inf])
                      
            popt, pcov = curve_fit(
                t0_model, w, t0,
                p0=p0, bounds=bounds,
                maxfev=20000, method="trf"
            )
            
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
            raise RuntimeError(f"Non-linear fit failed: {e}")

    # =======================================================
    # --- Auto Mode (Original Behavior) ---------------------
    # =======================================================
    try_nl = (w.size >= min_points_nonlinear)
    
    if try_nl:
        try:
            wmin = np.min(w)
            # Initial parameter estimates (p0)
            a0 = (np.nanmax(t0) - np.nanmin(t0)) / 2.0 if np.isfinite(t0).any() else 0.0
            d0 = np.nanmedian(t0)
            min_required = 1.0 / (wmin**2) if wmin != 0 else 1e-8
            b0 = min_required * 1.1
            c0 = min_required * 1.2
            
            p0 = [a0, b0, c0, d0]
            
            # Bound constraints to ensure stability
            bounds = ([-np.inf, min_required, min_required, -np.inf],
                      [np.inf, np.inf, np.inf, np.inf])
                      
            popt, pcov = curve_fit(
                t0_model, w, t0,
                p0=p0, bounds=bounds,
                maxfev=20000, method="trf"
            )
            
            fit_y = t0_model(fit_x, *popt)
            
            # Check if the curve is stable before accepting it
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
            pass  # If the non-linear fit fails, proceed to polynomial fallback

    # --- Polynomial Fallback ---
    # Determine the degree based on the number of available points
    if w.size < 5:
        deg = min(3, max(1, w.size - 1))
    else:
        deg = 4

    coeffs = np.polyfit(w, t0, deg)
    
    # Ensure coefficients array is strictly length 5
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



