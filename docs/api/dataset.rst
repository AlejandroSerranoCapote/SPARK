Dataset API (`spark.data.dataset`)
===================================

.. currentmodule:: spark.data.dataset

The ``SpectroscopyDataset`` class encapsulates the 2D experimental spectroscopy matrix alongside its calibrated wavelength and delay time coordinate vectors.

Class Definition
----------------

.. py:class:: SpectroscopyDataset(data_matrix, wl, td)

   Core data container representing time-resolved spectroscopy measurements.

   :param numpy.ndarray data_matrix: 2D array of experimental signal values (typically $\Delta A$ or luminescence intensity), of shape $(N_\lambda \times N_t)$.
   :param numpy.ndarray wl: 1D array of calibrated wavelength or energy coordinates of length $N_\lambda$.
   :param numpy.ndarray td: 1D array of calibrated delay time coordinates of length $N_t$.

   **Attributes**:

   .. py:attribute:: data
      :type: numpy.ndarray

      The 2D float64 data matrix of shape $(N_\lambda, N_t)$.

   .. py:attribute:: wl
      :type: numpy.ndarray

      1D float64 array of wavelength coordinates.

   .. py:attribute:: td
      :type: numpy.ndarray

      1D float64 array of delay time coordinates.

   .. py:attribute:: native_wl_unit
      :type: str

      Wavelength unit string (default: ``"nm"``).

   .. py:attribute:: native_time_unit
      :type: str

      Time delay unit string (default: ``"ps"``).

Methods
-------

.. py:method:: SpectroscopyDataset.copy()

   Creates an independent deep copy of the dataset.

   :return: A new ``SpectroscopyDataset`` instance with identical arrays and units.
   :rtype: SpectroscopyDataset

.. py:method:: SpectroscopyDataset.get_shape()

   Returns the dimensions of the internal data matrix.

   :return: Tuple ``(n_wavelengths, n_time_delays)``.
   :rtype: tuple of int

.. py:method:: SpectroscopyDataset.apply_baseline_correction(n_pts)

   Subtracts the average signal of the first ``n_pts`` time points from each wavelength channel (in-place).

   :param int n_pts: Number of early pre-time-zero delay points to average.

.. py:method:: SpectroscopyDataset.normalize()

   Normalizes the data matrix by dividing by the global maximum absolute value $\max |D|$ (in-place).

.. py:method:: SpectroscopyDataset.crop_time(t_min, t_max)

   Truncates the delay time axis and data matrix to the interval $[t_{\min}, t_{\max}]$ (in-place).

   :param float t_min: Minimum delay time to retain.
   :param float t_max: Maximum delay time to retain.

.. py:method:: SpectroscopyDataset.crop_wavelength(wl_min, wl_max)

   Truncates the wavelength axis and data matrix to the interval $[\lambda_{\min}, \lambda_{\max}]$ (in-place).

   :param float wl_min: Minimum wavelength to retain.
   :param float wl_max: Maximum wavelength to retain.

.. py:method:: SpectroscopyDataset.exclude_wavelength_ranges(exclude_str)

   Parses comma-separated spectral exclusion intervals and excises those wavelength channels from the dataset (in-place).

   :param str exclude_str: String such as ``"490-540, 600-615"``.

.. py:method:: SpectroscopyDataset.zero_negative_times()

   Sets all matrix values at delay times $t < 0$ strictly to $0.0$ (in-place).

.. py:method:: SpectroscopyDataset.apply_binning(b_size)

   Performs spectral channel binning by averaging adjacent wavelength pixels in blocks of ``b_size`` (in-place).

   :param int b_size: Number of adjacent wavelength channels to average.

.. py:method:: SpectroscopyDataset.apply_time_binning(n_pts)

   Performs temporal binning by averaging adjacent time delay points in blocks of ``n_pts`` (in-place).

   :param int n_pts: Number of adjacent time delay points to average.
