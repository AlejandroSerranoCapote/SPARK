import os
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2_dist
import spark.core.fitting.fit as fit

class GlobalFitter:
    """
    Independent engine for global optimization and error calculation for SPARK.
    It does not contain any dependency on PyQt5 or graphical user interfaces.

    Parameters
    ----------
    dataset : SpectroscopyDataset
        The core dataset containing data, wl, and td matrices.
    model_type : str
        The kinetic model to apply (e.g., "Sequential", "Damped Oscillation", "Custom GUI Model").
    num_exp : int
        Number of exponential decay components.
    use_artifact : bool, optional
        Whether to model the coherent artifact. Defaults to False.
    artifact_mode : str, optional
        The artifact base strategy ('raman', 'xpm', or 'both'). Defaults to 'both'.
    use_nnls : bool, optional
        If True, restricts the extracted amplitudes to be strictly non-negative. Defaults to False.
    is_indep_t0 : bool, optional
        If True, fits an independent time-zero (t0) for each wavelength. Defaults to False.
    custom_model : object, optional
        A user-defined custom kinetic model. Defaults to None.
    """
    def __init__(self, dataset, model_type, num_exp, use_artifact=False, 
                 artifact_mode='both', use_nnls=False, is_indep_t0=False, custom_model=None):
        
        # 1. Experimental Data
        self.dataset = dataset
        self.data_c = dataset.data
        self.WL = dataset.wl
        self.TD = dataset.td
        
        # 2. Model Configuration
        self.model_type = model_type
        self.num_exp = num_exp
        self.use_artifact = use_artifact
        self.artifact_mode = artifact_mode
        self.use_nnls = use_nnls
        self.is_indep_t0 = is_indep_t0
        self.custom_model = custom_model
        
        # 3. Result Containers
        self.fit_result = None
        self.fit_x = None
        self.fitres = None
        self.resid = None
        
        self.As = None
        self.errAs = None
        self.Bs = None
        self.errBs = None
        
        self.ci = None
        self.extracted_taus = None
        self.extracted_errtaus = None
        
        # 4. Identifiability Diagnostics
        self.condition_number = np.inf
        self.sloppiest_direction = []
        self.param_correlation = None
        self.param_correlation_indices = None
        self.S_T_full = None
    
    def get_num_kinetic_params(self):
        if self.model_type == "Damped Oscillation":
            base = 2 + self.num_exp + 3
        elif self.model_type == "Stretched Exponential":
            base = 2 + 2 * self.num_exp
        elif self.model_type == "Custom GUI Model" and self.custom_model is not None:
            base = 2 + len(self.custom_model.param_labels)
        else:
            base = 2 + self.num_exp
            
        if self.is_indep_t0:
            numWL = self.data_c.shape[0] if self.data_c is not None else 0
            return base + numWL
            
        return base

    def get_kinetic_param_label(self, i):
        base_kin = 2 + self.num_exp
        if self.model_type == "Damped Oscillation":
            base_kin = 2 + self.num_exp + 3
        elif self.model_type == "Stretched Exponential":
            base_kin = 2 + 2 * self.num_exp
        elif self.model_type == "Custom GUI Model" and self.custom_model is not None:
            base_kin = 2 + len(self.custom_model.param_labels)

        if self.is_indep_t0 and base_kin <= i < self.get_num_kinetic_params():
            idx_wl = i - base_kin
            wl_val = self.WL[idx_wl] if self.WL is not None else idx_wl
            return f"t0 @ {wl_val:.1f} nm"

        if self.model_type == "Custom GUI Model" and self.custom_model is not None:
            if i == 0: return "w (IRF Width)"
            if i == 1: return "t0 (Global Time Zero)"
            idx_param = i - 2
            if 0 <= idx_param < len(self.custom_model.param_labels):
                return self.custom_model.param_labels[idx_param]
            return f"param[{i}]"

        if self.model_type == "Damped Oscillation":
            if i == 0: return "w (IRF Width)"
            if i == 1: return "t0 (Global Time Zero)"
            if i < 2 + self.num_exp: return f"tau_{i-1}"
            if i == 2 + self.num_exp: return "alpha (Damping)"
            if i == 2 + self.num_exp + 1: return "omega (Frequency)"
            if i == 2 + self.num_exp + 2: return "phi (Phase)"
            return f"param[{i}]"

        if self.model_type == "Stretched Exponential":
            if i == 0: return "w (IRF Width)"
            if i == 1: return "t0 (Global Time Zero)"
            if i < 2 + self.num_exp: return f"tau_{i-1}"
            if i < 2 + 2 * self.num_exp: return f"beta_{i - 1 - self.num_exp}"
            return f"param[{i}]"

        if i == 0: return "w (IRF Width)"
        if i == 1: return "t0 (Global Time Zero)"
        if i < 2 + self.num_exp: return f"tau_{i-1}"
        return f"param[{i}]"

    def build_C_matrix(self, x_target, TD=None, WL=None):
        from spark.core.models import DASModel, SASModel, OscillationModel, StretchedModel
        
        current_TD = TD if TD is not None else self.TD
        current_WL = WL if WL is not None else self.WL

        if self.model_type == "Sequential":
            model = SASModel(self.num_exp, self.use_artifact, self.artifact_mode)
        elif self.model_type == "Damped Oscillation":
            model = OscillationModel(self.num_exp, self.use_artifact, self.artifact_mode)
        elif self.model_type == "Stretched Exponential":
            model = StretchedModel(self.num_exp, self.use_artifact, self.artifact_mode)
        elif self.model_type == "Custom GUI Model" and self.custom_model is not None:
            model = self.custom_model
        else:
            model = DASModel(self.num_exp, self.use_artifact, self.artifact_mode)

        def build_single(x_loc):
            if self.model_type == "Custom GUI Model" and model is not None and hasattr(model, 'param_labels'):
                w, t0_val = x_loc[0], x_loc[1]
                num_params_cineticos = len(model.param_labels)
                x_nl_params = x_loc[2:2+num_params_cineticos]
                return model.get_concentration_matrix(x_nl_params, current_TD, w, t0_val, self.use_artifact, self.artifact_mode)
            else:
                return model.get_concentration_matrix(x_loc, current_TD)

        if self.is_indep_t0:
            base_kin = self.get_num_kinetic_params() - len(current_WL)
            C_list = []
            for i in range(len(current_WL)):
                x_local = x_target.copy()
                x_local[1] = x_target[base_kin + i] 
                C_list.append(build_single(x_local))
            return C_list
        else:
            return build_single(x_target)
        
    def process_fit_results(self, scipy_res, full_x, free_indices):
        """
        Receives raw optimizer results and calculates spectra, covariance matrix, 
        amplitude errors, and identifiability diagnostics.

        Parameters
        ----------
        scipy_res : scipy.optimize.OptimizeResult
            The object returned by the least_squares optimizer.
        full_x : numpy.ndarray
            The full array of optimized parameters (both fixed and free).
        free_indices : numpy.ndarray or list
            The indices of the parameters that were allowed to float during optimization.
        """
        self.fit_result = scipy_res
        self.fit_x = full_x
        numWL = len(self.WL)
        
        # 1. Reconstruct Final Matrices
        C = self.build_C_matrix(self.fit_x)
        F_mat, S_T_full = fit.eval_varpro_model(C, self.data_c.T, enforce_nonneg=self.use_nnls, numExp=self.num_exp)
        
        self.S_T_full = S_T_full
        self.fitres = F_mat.T 
        self.resid = self.data_c - self.fitres
        
        # 2. Jacobian Analysis and Covariance Matrix
        L_total = len(self.fit_x)
        self.ci = np.zeros(L_total)
        
        J = scipy_res.jac
        if J is not None and J.size > 0 and len(free_indices) > 0:
            U, s, Vh = np.linalg.svd(J, full_matrices=False)
            tol = np.finfo(float).eps * max(J.shape) * s[0]
            s_inv = np.zeros_like(s)
            s_inv[s > tol] = 1.0 / s[s > tol]
            cov_free = (Vh.T * (s_inv**2)) @ Vh

            s_nonzero = s[s > tol]
            if len(s_nonzero) > 0:
                self.condition_number = float(s_nonzero[0] / s_nonzero[-1])
                idx_sloppiest = int(np.argmin(s))
                direction = Vh[idx_sloppiest, :]
                contributions = sorted(zip(free_indices.tolist(), direction.tolist()), key=lambda t: abs(t[1]), reverse=True)
                self.sloppiest_direction = [(idx, w) for idx, w in contributions if abs(w) > 0.15]
            
            # Degrees of freedom
            num_species = len(self.custom_model.states) if (self.model_type == "Custom GUI Model" and self.custom_model is not None) else (self.num_exp + 1 if "Oscillation" in self.model_type else self.num_exp)
            num_linear_params = numWL * num_species
            total_fitted_params = len(free_indices) + num_linear_params
            dof = self.resid.size - total_fitted_params

            if dof > 0:
                mse = np.sum(self.resid**2) / dof
                cov_free_scaled = cov_free * mse
                var_free = np.diagonal(cov_free_scaled)
                self.ci[free_indices] = np.sqrt(np.maximum(var_free, 0))

                d = np.sqrt(np.maximum(np.diagonal(cov_free_scaled), 1e-300))
                self.param_correlation = cov_free_scaled / np.outer(d, d)
                self.param_correlation_indices = free_indices

        # 3. Extract Taus and their Errors
        idx_tau = 2
        end_tau = idx_tau + (len(self.custom_model.param_labels) if (self.model_type == "Custom GUI Model" and self.custom_model is not None) else self.num_exp)
        
        if end_tau <= len(self.fit_x):
            self.extracted_taus = self.fit_x[idx_tau : end_tau]
            self.extracted_errtaus = self.ci[idx_tau : end_tau]
        else:
            self.extracted_taus = np.zeros(end_tau - idx_tau)
            self.extracted_errtaus = np.zeros(end_tau - idx_tau)

        # 4. Calculation of LINEAR errors (Amplitudes of DAS/SAS spectra)
        self.As = np.zeros((self.num_exp, numWL))
        self.errAs = np.zeros((self.num_exp, numWL))
        
        try:
            if isinstance(C, list):
                err_S_T_list = []
                for i in range(numWL):
                    pseudo_inv_C = np.linalg.pinv(C[i].T @ C[i])
                    diag_cov = np.diagonal(pseudo_inv_C)
                    dof_linear = self.resid.shape[1] - C[i].shape[1]
                    mse = np.sum(self.resid[i, :]**2) / dof_linear if dof_linear > 0 else 0
                    err_S_T_list.append(np.sqrt(np.maximum(diag_cov * mse, 0)))
                err_S_T_full = np.column_stack(err_S_T_list)
            else:
                pseudo_inv_C = np.linalg.pinv(C.T @ C)
                diag_cov = np.diagonal(pseudo_inv_C) 
                dof_linear = self.resid.shape[1] - C.shape[1]
                mse_per_wl = np.sum(self.resid**2, axis=1) / dof_linear if dof_linear > 0 else np.zeros(numWL)
                err_S_T_full = np.sqrt(np.maximum(np.outer(diag_cov, mse_per_wl), 0))
            
            if "Oscillation" in self.model_type:
                self.As = self.S_T_full[:self.num_exp, :]
                self.errAs = err_S_T_full[:self.num_exp, :]
                self.Bs = self.S_T_full[self.num_exp, :]
                self.errBs = err_S_T_full[self.num_exp, :]
            else:
                self.As = self.S_T_full[:self.num_exp, :]
                self.errAs = err_S_T_full[:self.num_exp, :]
                self.Bs = None
                self.errBs = None
                
            if self.model_type == "Stretched Exponential":
                b_start = 2 + self.num_exp
                b_end = b_start + self.num_exp
                self.extracted_betas = self.fit_x[b_start:b_end]
                self.extracted_errbetas = self.ci[b_start:b_end]
            else:
                self.extracted_betas = None
                self.extracted_errbetas = None
                
            self.err_S_T_full = err_S_T_full # We save it to export the artifact later

        except Exception as e:
            print(f"Error calculando errores lineales en Fitter: {e}")
            
        if self.use_artifact:
            n_art = 1 if self.artifact_mode == 'raman' else (2 if self.artifact_mode == 'xpm' else 3)
            self.Artifact_Spectra = self.S_T_full[-n_art:, :]
        else:
            self.Artifact_Spectra = None

    def save_results(self, outdir, ini_guesses):
        """
        Exports all results, matrices, and spectra to disk.

        Parameters
        ----------
        outdir : str
            The output directory path where files will be saved.
        ini_guesses : numpy.ndarray
            The initial guesses used for the optimization.
        """
        os.makedirs(outdir, exist_ok=True)
        
        # 1. Save complete variables in binary format
        results_dict = {
            'fitres': self.fitres, 'resid': self.resid,
            'taus': self.extracted_taus, 'err_taus': self.extracted_errtaus,
            'As': self.As, 'errAs': self.errAs,
            'x': self.fit_x, 'ci': self.ci,
            'model_type': self.model_type, 'numExp': self.num_exp,
        }
        if self.model_type == "Custom GUI Model" and self.custom_model is not None:
            results_dict['param_labels'] = self.custom_model.param_labels
            results_dict['states'] = self.custom_model.states
        
        # ---> INYECCIÓN 1: Guardar betas en el diccionario .npy <---
        if self.model_type == "Stretched Exponential" and getattr(self, 'extracted_betas', None) is not None:
            results_dict['betas'] = self.extracted_betas
            results_dict['err_betas'] = self.extracted_errbetas
        # -----------------------------------------------------------
        
        np.save(os.path.join(outdir, "GFitResults.npy"), results_dict)
        
        # 2. Experimental, fit, and residual 2D matrices
        header_line = "Wavelength\t" + "\t".join([f"{t:.6g}" for t in self.TD])
        np.savetxt(os.path.join(outdir, "Corrected_data.txt"), np.column_stack((self.WL, self.data_c)), fmt='%.6e', delimiter='\t', header=header_line, comments='')
        np.savetxt(os.path.join(outdir, "Fitted_data.txt"), np.column_stack((self.WL, self.fitres)), fmt='%.6e', delimiter='\t', header=header_line, comments='')
        np.savetxt(os.path.join(outdir, "Residuals.txt"), np.column_stack((self.WL, self.resid)), fmt='%.6e', delimiter='\t', header=header_line, comments='')
        
        np.savetxt(os.path.join(outdir, "WL.txt"), self.WL, fmt='%.6f', header='Wavelength (nm)', comments='')
        np.savetxt(os.path.join(outdir, "TD.txt"), self.TD, fmt='%.6f', header='Delay (ps)', comments='')
        
        # 3. Save Amplitudes (SAS/DAS) and metadata
        tau_comment_lines = [
            f"irf_fwhm={self.fit_x[0]:.6g}+-{self.ci[0] if len(self.ci)>0 else 0:.6g}",
            f"t0={self.fit_x[1]:.6g}+-{self.ci[1] if len(self.ci)>1 else 0:.6g}"
        ]
        
        if self.model_type == "Custom GUI Model" and self.custom_model is not None:
            for i, label in enumerate(self.custom_model.param_labels):
                t_val = self.extracted_taus[i] if i < len(self.extracted_taus) else np.nan
                e_val = self.extracted_errtaus[i] if (self.extracted_errtaus is not None and i < len(self.extracted_errtaus)) else 0.0
                tau_comment_lines.append(f"{label}={t_val:.6g}+-{e_val:.6g}")
        else:
            for n in range(self.num_exp):
                t_val = self.extracted_taus[n] if n < len(self.extracted_taus) else np.nan
                e_val = self.extracted_errtaus[n] if (self.extracted_errtaus is not None and n < len(self.extracted_errtaus)) else 0.0
                tau_comment_lines.append(f"tau{n+1}={t_val:.6g}+-{e_val:.6g}")
            
        # ---> INYECCIÓN 2: Escribir betas en la cabecera de Amplitudes.txt <---
        if self.model_type == "Stretched Exponential" and getattr(self, 'extracted_betas', None) is not None:
            for n in range(self.num_exp):
                b_val = self.extracted_betas[n]
                eb_val = self.extracted_errbetas[n] if self.extracted_errbetas is not None else 0.0
                tau_comment_lines.append(f"beta{n+1}={b_val:.6g}+-{eb_val:.6g}")
        # ----------------------------------------------------------------------
            
        if self.use_artifact:
            tau_comment_lines.append("coherent_artifact=True")
            tau_comment_lines.append(f"artifact_mode={self.artifact_mode}")
            
        col_headers = ["Wavelength"]
        amp_columns = [self.WL]
        
        if self.model_type == "Custom GUI Model" and self.custom_model is not None:
            for n in range(len(self.custom_model.states)):
                s_name = self.custom_model.states[n]
                col_headers.extend([f"A_{s_name}", f"A_{s_name}_err"])
                amp_columns.extend([self.As[n], self.errAs[n]])
        else:
            for n in range(self.num_exp):
                col_headers.extend([f"A{n+1}", f"A{n+1}_err"])
                amp_columns.extend([self.As[n], self.errAs[n]])
            
        if self.use_artifact:
            if self.artifact_mode == 'raman': labels = ["Raman_phi0"]
            elif self.artifact_mode == 'xpm': labels = ["XPM_phi1", "XPM_phi2"]
            else: labels = ["Raman_phi0", "XPM_phi1", "XPM_phi2"]
            
            art_amps = self.S_T_full[-len(labels):, :]
            art_errs = getattr(self, 'err_S_T_full', np.zeros_like(art_amps))
            # Ajuste de seguridad: asegurar que extraemos bien los errores del artefacto
            if art_errs.shape[0] >= len(labels):
                art_errs = art_errs[-len(labels):, :]
            else:
                art_errs = np.zeros_like(art_amps)
            
            for i, label in enumerate(labels):
                col_headers.extend([label, f"{label}_err"])
                amp_columns.extend([art_amps[i], art_errs[i]])

        with open(os.path.join(outdir, "Amplitudes.txt"), 'w') as f:
            for line in tau_comment_lines: f.write(f"# {line}\n")
            f.write("\t".join(col_headers) + "\n")
            np.savetxt(f, np.column_stack(amp_columns), fmt='%.6e', delimiter='\t')
            
    def run_profile_likelihood(self, param_idx, ini, limi, lims, is_fixed, 
                               n_steps=15, confidence=0.95, span_sigma=6, 
                               progress_callback=None, abort_callback=None):
        """
        Calculates the profile likelihood interval by fixing the parameter on a grid 
        and refitting the remaining free parameters.

        Parameters
        ----------
        param_idx : int
            Index of the parameter to compute the profile likelihood for.
        ini : numpy.ndarray
            Initial guess values.
        limi : numpy.ndarray
            Lower bounds array.
        lims : numpy.ndarray
            Upper bounds array.
        is_fixed : numpy.ndarray or list
            Boolean mask of globally fixed parameters.
        n_steps : int, optional
            Number of points in the grid. Defaults to 15.
        confidence : float, optional
            Confidence level (e.g., 0.95 for 95%). Defaults to 0.95.
        span_sigma : float, optional
            Number of standard deviations to span on each side of the optimum. Defaults to 6.
        progress_callback : callable, optional
            Callback function to report progress (step, total_steps). Defaults to None.
        abort_callback : callable, optional
            Callback function to check if the process should be interrupted. Defaults to None.

        Returns
        -------
        dict or None
            A dictionary containing the profile likelihood results ('grid', 'chi2', 
            'delta_threshold', 'lower_bound', 'upper_bound', etc.) or None if empty.
        
        Raises
        ------
        RuntimeError
            If the function is called before executing a global fit.
        InterruptedError
            If the abort_callback signals to cancel the operation.
        """
        if self.fit_x is None or self.fit_result is None:
            raise RuntimeError("Ejecuta un fit antes de calcular el perfil de verosimilitud.")

        best_val = self.fit_x[param_idx]
        best_err = self.ci[param_idx] if self.ci[param_idx] > 0 else abs(best_val) * 0.1 + 1e-6
        grid = np.linspace(best_val - span_sigma * best_err, best_val + span_sigma * best_err, n_steps)

        chi2_values = []
        data_flat = self.data_c.T.flatten()
        
        # We start from the optimal parameters of the last successful fit
        original_ini = self.fit_x.copy()
        original_is_fixed = is_fixed.copy()

        for k, val in enumerate(grid):
            # Check if the GUI ordered to abort the process
            if abort_callback is not None and abort_callback():
                raise InterruptedError("Identifiability analysis cancelled.")

            current_ini = original_ini.copy()
            current_ini[param_idx] = val
            
            current_fixed = original_is_fixed.copy()
            current_fixed[param_idx] = True # We pin the parameter being analyzed
            
            num_kin_params = self.get_num_kinetic_params()
            is_kinetic = np.zeros(len(current_ini), dtype=bool)
            is_kinetic[:num_kin_params] = True
            
            free_indices = np.where(is_kinetic & ~current_fixed)[0]
            
            # Temporary residuals function for SciPy
            def sync_residuals(p_free):
                x_full = current_ini.copy()
                x_full[free_indices] = p_free
                C = self.build_C_matrix(x_full) # Uses the internal engine constructor
                F, _ = fit.eval_varpro_model(C, self.data_c.T, enforce_nonneg=self.use_nnls, numExp=self.num_exp)
                return F.flatten() - data_flat

            if len(free_indices) > 0:
                x0_free = current_ini[free_indices]
                low_free = limi[free_indices]
                upp_free = lims[free_indices]
                
                # Relax ftol and xtol to 1e-6 for faster evaluation
                res = least_squares(
                    fun=sync_residuals, x0=x0_free, bounds=(low_free, upp_free),
                    method='trf', x_scale='jac', loss='soft_l1',      
                    ftol=1e-6, xtol=1e-6, verbose=0 
                )
                chi2_values.append(float(np.sum(res.fun ** 2)))
            else:
                res_eval = sync_residuals([])
                chi2_values.append(float(np.sum(res_eval ** 2)))

            # Notify progress if there is a connected callback
            if progress_callback is not None:
                progress_callback(k + 1, n_steps)

        chi2_values = np.array(chi2_values)
        if len(chi2_values) == 0:
            return None

        chi2_min = chi2_values.min()
        delta_threshold = chi2_dist.ppf(confidence, df=1)

        within = grid[(chi2_values - chi2_min) <= delta_threshold]
        lower_bound = float(within.min()) if within.size > 0 else None
        upper_bound = float(within.max()) if within.size > 0 else None

        return {
            'grid': grid, 'chi2': chi2_values,
            'delta_threshold': delta_threshold, 'chi2_min': chi2_min,
            'lower_bound': lower_bound, 'upper_bound': upper_bound,
            'best_value': best_val, 'label': self.get_kinetic_param_label(param_idx),
        }
