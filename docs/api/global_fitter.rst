Global Fitter API (`spark.core.fitting.global_fitter`)
=====================================================

.. currentmodule:: spark.core.fitting.global_fitter

The ``GlobalFitter`` class orchestrates the mathematical construction of kinetic concentration matrices, Variable Projection linear solves, covariance analysis, and profile likelihood calculations.

Class Definition
----------------

.. py:class:: GlobalFitter(dataset, model_type, num_exp, use_artifact=False, artifact_mode='both', use_nnls=False, is_indep_t0=False, custom_model=None)

   Mathematical coordinator for global variable projection fitting.

   :param SpectroscopyDataset dataset: The target experimental dataset.
   :param str model_type: Kinetic model identifier (``"Parallel"``, ``"Sequential"``, ``"Stretched Exponential"``, ``"Damped Oscillation"``, or ``"Custom GUI Model"``).
   :param int num_exp: Number of exponential decay components (1 to 6).
   :param bool use_artifact: If True, appends coherent artifact basis functions to $C$.
   :param str artifact_mode: ``'both'`` (Raman + XPM), ``'raman'`` ($\phi_0$), or ``'xpm'`` ($\phi_1 + \phi_2$).
   :param bool use_nnls: If True, enforces non-negative spectra via BVLS linear solver.
   :param bool is_indep_t0: If True, fits independent time-zero $t_0(\lambda)$ per wavelength.
   :param KMatrixModel custom_model: Custom compiled kinetic model instance (if using custom model).

Attributes
----------

.. py:attribute:: condition_number
   :type: float

   Condition number $\kappa = s_{\max} / s_{\min}$ of the Jacobian matrix evaluated at the optimum.

.. py:attribute:: sloppiest_direction
   :type: str

   Text description identifying the linear combination of parameters with the weakest constraint.

.. py:attribute:: param_correlation
   :type: numpy.ndarray

   Square normalized correlation matrix $(N_{\text{free}} \times N_{\text{free}})$ of non-linear parameters.

.. py:attribute:: extracted_taus
   :type: list of float

   Fitted lifetime values $[\tau_1, \tau_2, \dots]$ in picoseconds.

.. py:attribute:: extracted_errtaus
   :type: list of float

   Asymptotic standard errors $[\Delta\tau_1, \Delta\tau_2, \dots]$ derived from the covariance matrix.

Methods
-------

.. py:method:: GlobalFitter.get_num_kinetic_params()

   Calculates total count of non-linear parameters based on model type and settings.

   :return: Integer count.
   :rtype: int

.. py:method:: GlobalFitter.get_kinetic_param_label(i)

   Returns human-readable text label for parameter index $i$ (e.g. ``"tau_1"``, ``"w (IRF Width)"``).

   :param int i: Parameter index.
   :return: Formatted label string.
   :rtype: str

.. py:method:: GlobalFitter.build_C_matrix(x_target, TD=None, WL=None)

   Evaluates the concentration matrix $C(t; \boldsymbol{\theta}_{\text{nl}})$ for candidate parameters ``x_target``.

   :param numpy.ndarray x_target: Vector of non-linear parameters.
   :param numpy.ndarray TD: Optional delay time grid.
   :param numpy.ndarray WL: Optional wavelength grid.
   :return: 2D concentration matrix $(N_t \times K)$ or list of matrices if ``is_indep_t0`` is True.
   :rtype: numpy.ndarray or list

.. py:method:: GlobalFitter.process_fit_results(scipy_res, full_x, free_indices)

   Performs full post-optimization covariance analysis:
   * Evaluates final fitted data matrix ($D_{\text{fit}} = C S^T$) and residual matrix ($R$).
   * Computes SVD of the Jacobian matrix to extract parameter standard errors, condition numbers, and correlation coefficients.
   * Extracts linear amplitude spectra ($A_k(\lambda)$) with per-wavelength uncertainty bounds.

   :param scipy.optimize.OptimizeResult scipy_res: Raw result object from TRF optimizer.
   :param numpy.ndarray full_x: Complete vector of fitted parameters.
   :param numpy.ndarray free_indices: Indices of non-fixed parameters.

.. py:method:: GlobalFitter.save_results(outdir, ini_guesses)

   Exports all analysis outputs to disk: ``Amplitudes.txt``, ``Corrected_data.txt``, ``Fitted_data.txt``, ``Residuals.txt``, ``WL.txt``, ``TD.txt``, and ``GFitResults.npy``.

   :param str outdir: Output destination directory path.
   :param numpy.ndarray ini_guesses: Initial starting guesses used for the fit.

.. py:method:: GlobalFitter.run_profile_likelihood(param_idx, ini, limi, lims, is_fixed, n_steps=15, confidence=0.95, span_sigma=6, progress_callback=None, abort_callback=None)

   Computes non-linear $\chi^2$ profile likelihood curve for parameter index ``param_idx``.

   :param int param_idx: Index of parameter to profile.
   :param numpy.ndarray ini: Initial parameter vector.
   :param numpy.ndarray limi: Lower bounds array.
   :param numpy.ndarray lims: Upper bounds array.
   :param numpy.ndarray is_fixed: Boolean parameter lock mask.
   :param int n_steps: Number of evaluation points along the profile.
   :param float confidence: Statistical confidence level (0.68, 0.90, 0.95, 0.99).
   :param float span_sigma: Profile span width in asymptotic standard error units.
   :return: Dictionary containing grid coordinates, $\chi^2$ values, statistical threshold, and bounds.
   :rtype: dict or None
