Optimizer API (`spark.core.fitting.optimizer`)
==============================================

.. currentmodule:: spark.core.fitting.optimizer

The optimizer module provides the main non-linear least-squares execution function and a background Qt worker thread.

Classes
-------

.. py:class:: FitResult(x_final, fitres, resid, S_T, scipy_res)

   Container holding results of a completed Variable Projection fit.

   :param numpy.ndarray x_final: Final optimized non-linear parameter vector.
   :param numpy.ndarray fitres: Reconstructed 2D fitted data matrix ($N_\lambda \times N_t$).
   :param numpy.ndarray resid: 2D residual matrix ($D - D_{\text{fit}}$).
   :param numpy.ndarray S_T: Matrix of linear amplitudes / species-associated spectra ($K \times N_\lambda$).
   :param scipy.optimize.OptimizeResult scipy_res: The underlying SciPy optimizer result object.

.. py:class:: FitWorker(dataset, build_c_func, ini, limi, lims, is_fixed, num_exp, use_nnls, enforce_tau_order=True)

   Qt ``QThread`` worker executing the non-linear optimization pipeline in a background thread to prevent GUI freezing.

   **Signals**:

   * ``progress_update(int)``: Emitted periodically with completion percentage.
   * ``finished_success(object)``: Emitted upon successful convergence with the resulting :class:`FitResult`.
   * ``finished_error(str)``: Emitted if optimization encounters a numerical error or user abort.

   .. py:method:: abort()

      Sets internal abort flag to terminate iteration gracefully.

Functions
---------

.. py:function:: fit(dataset, build_c_func, ini, limi, lims, is_fixed, num_exp, use_nnls=False, enforce_tau_order=True, progress_cb=None, abort_cb=None)

   Core Variable Projection non-linear optimization routine.

   Constructs the objective residual function by projecting out linear amplitudes, evaluates soft-$L_1$ loss metrics, adds tau-ordering soft barrier penalties, and minimizes residuals via SciPy's Trust-Region Reflective algorithm.

   :param SpectroscopyDataset dataset: Experimental dataset.
   :param callable build_c_func: Function ``build_c_func(x, td, wl)`` returning the concentration matrix $C$.
   :param numpy.ndarray ini: Array of initial parameter guesses.
   :param numpy.ndarray limi: Lower parameter bounds.
   :param numpy.ndarray lims: Upper parameter bounds.
   :param numpy.ndarray is_fixed: Boolean mask indicating locked parameters.
   :param int num_exp: Number of exponential components.
   :param bool use_nnls: If True, solves linear spectra using non-negative least squares.
   :param bool enforce_tau_order: If True, applies barrier penalty to enforce $\tau_1 < \tau_2 < \dots < \tau_K$.
   :param callable progress_cb: Optional callback ``fn(percent)``.
   :param callable abort_cb: Optional callback ``fn() -> bool``.
   :return: Completed fit results.
   :rtype: FitResult
