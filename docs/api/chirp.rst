Chirp API (`spark.core.chirp`)
================================

.. currentmodule:: spark.core.chirp

The chirp module provides functions for fitting and correcting Group Velocity Dispersion (GVD) / time-zero variations across wavelength channels.

Functions
---------

.. py:function:: eV_a_nm(E_eV)

   Converts photon energy in electron-volts (eV) to wavelength in nanometers (nm).

   .. math::

      \lambda = \frac{1239.841984}{E}

   :param float or numpy.ndarray E_eV: Energy value(s) in eV.
   :return: Wavelength(s) in nm.
   :rtype: float or numpy.ndarray

.. py:function:: t0_model(w, a, b, c, d)

   Sellmeier-derived non-linear dispersion model for time-zero variation across wavelength $w$:

   .. math::

      t_0(w) = a \sqrt{\frac{b\,w^2 - 1}{c\,w^2 - 1}} + d

   :param numpy.ndarray w: Wavelength array.
   :param float a: Amplitude scaling factor.
   :param float b: Numerator dispersion parameter.
   :param float c: Denominator dispersion parameter.
   :param float d: Constant temporal offset.
   :return: Array of fitted time-zero values.
   :rtype: numpy.ndarray

.. py:function:: fit_t0(w_points, t0_points, WL, TD, data, min_points_nonlinear=4, mode='auto')

   Fits a dispersion curve to discrete experimental $(\lambda_i, t_{0,i})$ calibration points.

   :param list w_points: Wavelength coordinates of calibration points.
   :param list t0_points: Delay time coordinates of calibration points.
   :param numpy.ndarray WL: Complete experimental wavelength axis.
   :param numpy.ndarray TD: Complete experimental delay time axis.
   :param numpy.ndarray data: Experimental 2D data matrix.
   :param int min_points_nonlinear: Minimum points required to attempt non-linear Sellmeier fit (default: 4).
   :param str mode: Fitting strategy: ``'auto'`` (tries non-linear, falls back to polynomial), ``'nonlinear'``, or ``'poly'``.
   :return: Dictionary containing keys ``'method'``, ``'popt'``, ``'fit_x'``, ``'fit_y'``, ``'corrected'``, and ``'t0_lambda'``.
   :rtype: dict

.. py:function:: apply_t0_correction_poly(popt, WL, TD, data)

   Applies polynomial chirp correction by interpolating each wavelength channel along the delay time axis.

   :param numpy.ndarray popt: 5 polynomial coefficients ($c_4, c_3, c_2, c_1, c_0$).
   :param numpy.ndarray WL: Wavelength axis.
   :param numpy.ndarray TD: Delay time axis.
   :param numpy.ndarray data: 2D data matrix.
   :return: Tuple ``(corrected_data, t0_per_wavelength)``.
   :rtype: tuple of numpy.ndarray

.. py:function:: apply_t0_correction_nonlinear(popt, WL, TD, data)

   Applies Sellmeier non-linear model chirp correction by interpolating each wavelength channel along the delay time axis.

   :param numpy.ndarray popt: 4 parameters ``[a, b, c, d]``.
   :param numpy.ndarray WL: Wavelength axis.
   :param numpy.ndarray TD: Delay time axis.
   :param numpy.ndarray data: 2D data matrix.
   :return: Tuple ``(corrected_data, t0_per_wavelength)``.
   :rtype: tuple of numpy.ndarray
