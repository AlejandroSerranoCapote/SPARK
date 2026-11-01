Diagnostics API (`spark.core.diagnostics`)
==============================================

.. currentmodule:: spark.core.diagnostics

The diagnostics module provides automated reporting functions and Matplotlib plotting routines for evaluating fit results.

Functions
---------

.. py:function:: analyze(result, param_names=None)

   Prints a formatted console report summarizing optimizer convergence status, final cost, number of function evaluations, and optimized parameter values.

   :param FitResult result: Output object from optimizer.
   :param list param_names: Optional custom parameter labels.

.. py:function:: plot_spectra(dataset, target_times)

   Plots 1D spectral slices at the specified target delay times.

   :param SpectroscopyDataset dataset: Experimental dataset.
   :param list target_times: List of delay times (in ps) to extract and plot.

.. py:function:: plot_kinetics_fit(dataset, result, target_wavelengths, use_symlog=True)

   Generates a publication plot comparing experimental data points against global fit curves at specified target wavelengths.

   :param SpectroscopyDataset dataset: Experimental dataset.
   :param FitResult result: Optimization result.
   :param list target_wavelengths: List of wavelength values (in nm) to display.
   :param bool use_symlog: If True, uses symmetric logarithmic scale for the delay axis.

.. py:function:: plot_associated_spectra(dataset, result, num_species=None)

   Plots Decay-Associated Spectra (DAS) or Species-Associated Spectra (SAS) along with coherent artifact spectra if present.

   :param SpectroscopyDataset dataset: Experimental dataset.
   :param FitResult result: Optimization result.
   :param int num_species: Optional number of species to plot.

.. py:function:: plot_single_map(dataset, title="2D Map", use_symlog=True)

   Renders a single 2D false-color pcolormesh heatmap centered at zero with symmetric color limits.

   :param SpectroscopyDataset dataset: Experimental dataset.
   :param str title: Plot title.
   :param bool use_symlog: Whether to apply symlog scaling to delay axis.

.. py:function:: plot_fit_maps(dataset, result, use_symlog=True, vmin=None, vmax=None)

   Creates a 3-panel comparative display:
   1. Pre-processed experimental data matrix.
   2. Reconstructed global fit matrix.
   3. Residual difference matrix ($D - D_{\text{fit}}$).

   :param SpectroscopyDataset dataset: Experimental dataset.
   :param FitResult result: Optimization result.
   :param bool use_symlog: Symmetric log delay scaling.
   :param float vmin: Optional minimum colorbar limit.
   :param float vmax: Optional maximum colorbar limit.
