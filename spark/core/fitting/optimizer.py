import numpy as np
from scipy.optimize import least_squares
from PyQt5.QtCore import QThread, pyqtSignal
from spark.core.fitting.fit import eval_varpro_model

class FitResult:
    """
    Object returned by our API with the ready results.

    Parameters
    ----------
    x_final : numpy.ndarray
        The final optimized non-linear parameters.
    fitres : numpy.ndarray
        The reconstructed fitted data matrix.
    resid : numpy.ndarray
        The residual matrix (Experimental Data - Fitted Data).
    S_T : numpy.ndarray
        The linear amplitudes matrix (e.g., DAS/SAS spectra).
    scipy_res : scipy.optimize.OptimizeResult
        The full optimization result object from SciPy.
    """
    def __init__(self, x_final, fitres, resid, S_T, scipy_res):
        self.x = x_final
        self.fitres = fitres
        self.resid = resid
        self.amplitudes = S_T
        self.scipy_res = scipy_res  
        
def fit(dataset, build_c_func, ini, limi, lims, is_fixed, num_exp, use_nnls=False, enforce_tau_order=True, progress_cb=None, abort_cb=None):
    """
    THE MAIN FUNCTION OF SPARK.
    Equivalent to: result = fit(dataset, model)

    Parameters
    ----------
    dataset : SpectroscopyDataset
        The dataset object containing the data, time delays, and wavelengths.
    build_c_func : callable
        Function to build the concentration matrix C.
    ini : numpy.ndarray
        Initial guess for the non-linear parameters.
    limi : numpy.ndarray
        Lower bounds for the non-linear parameters.
    lims : numpy.ndarray
        Upper bounds for the non-linear parameters.
    is_fixed : numpy.ndarray or list
        Boolean mask indicating which parameters are fixed.
    num_exp : int
        Number of exponential components.
    use_nnls : bool, optional
        Whether to use non-negative least squares. Defaults to False.
    enforce_tau_order : bool, optional
        Whether to enforce tau ordering (tau1 < tau2 < ...). Defaults to True.
    progress_cb : callable, optional
        Callback function to report progress. Defaults to None.
    abort_cb : callable, optional
        Callback function to check for an abort signal. Defaults to None.

    Returns
    -------
    FitResult
        An object containing the optimized parameters, fit matrix, and residuals.
    """
    free_indices = np.where(~np.array(is_fixed))[0]
    x0_free = ini[free_indices]
    bounds = (limi[free_indices], lims[free_indices])
    data_flat = dataset.data.T.flatten()
    iter_count = [0]

    def residuals(p_free):
        if abort_cb and abort_cb(): raise InterruptedError()
        iter_count[0] += 1
        if progress_cb and iter_count[0] % 5 == 0: progress_cb((iter_count[0] // 5) % 100)
        
        x_full = ini.copy()
        x_full[free_indices] = p_free
        
        C = build_c_func(x_full, dataset.td, dataset.wl)
        F, _ = eval_varpro_model(C, dataset.data.T, enforce_nonneg=use_nnls, numExp=num_exp)
        
        # 1. We calculate the real residual (Experimental Data vs Model)
        res_array = F.flatten() - data_flat

        # 2. PENALTY INJECTION (Soft Constraint)
        if enforce_tau_order and num_exp > 1:
            # We extract the times (assuming they are in positions 2 to 2+num_exp)
            taus = x_full[2 : 2 + num_exp]
            penalty = np.zeros(num_exp - 1)
            
            for i in range(num_exp - 1):
                if taus[i] >= taus[i+1]:
                    # If they try to cross, we trigger the error by multiplying by 100,000
                    # This forces the SciPy Jacobian to push the values back
                    penalty[i] = (taus[i] - taus[i+1]) * 1e5
                    
            # We add the penalty to the end of the residuals vector
            return np.concatenate([res_array, penalty])
            
        return res_array

    
    res = least_squares(
        fun=residuals, x0=x0_free, bounds=bounds,
        method='trf', x_scale='jac', loss='soft_l1',      
        ftol=1e-8, xtol=1e-8, verbose=0
    )

    x_final = ini.copy()
    x_final[free_indices] = res.x
    
    C_final = build_c_func(x_final, dataset.td, dataset.wl)
    F_final, S_T_final = eval_varpro_model(C_final, dataset.data.T, enforce_nonneg=use_nnls, numExp=num_exp)
    
    return FitResult(x_final, F_final.T, dataset.data - F_final.T, S_T_final, res)


class FitWorker(QThread):
    """
    QThread worker to execute the global fit without freezing the GUI.

    Signals
    -------
    progress_update : pyqtSignal(int)
        Emits the current progress percentage.
    finished_success : pyqtSignal(object)
        Emits the FitResult object upon successful completion.
    finished_error : pyqtSignal(str)
        Emits an error message string if the fit fails or is aborted.
    """
    progress_update = pyqtSignal(int)
    finished_success = pyqtSignal(object) 
    finished_error = pyqtSignal(str)

    def __init__(self, dataset, build_c_func, ini, limi, lims, is_fixed, num_exp, use_nnls, enforce_tau_order=True):
        super().__init__()
        self.dataset = dataset
        self.build_c_func = build_c_func
        self.ini = ini
        self.limi = limi
        self.lims = lims
        self.is_fixed = is_fixed
        self.num_exp = num_exp
        self.use_nnls = use_nnls
        self.enforce_tau_order = enforce_tau_order
        self.is_aborted = False

    def run(self):
        """Executes the fitting process in a separate thread."""
        try:
            result = fit(
                self.dataset, self.build_c_func, self.ini, self.limi, self.lims, 
                self.is_fixed, self.num_exp, self.use_nnls, self.enforce_tau_order,
                progress_cb=self.progress_update.emit,
                abort_cb=lambda: self.is_aborted
            )
            self.finished_success.emit(result)

        except InterruptedError:
            self.finished_error.emit("Aborted")
        except Exception as e:
            import traceback
            self.finished_error.emit(f"Critical error in the fit: {str(e)}\n{traceback.format_exc()}")