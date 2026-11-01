Tutorial 4: Headless Python Scripting & Automation
===================================================

SPARK's architecture separates the computational core from the PyQt5 graphical interface. Any dataset can be loaded, pre-processed, globally fitted, and exported via standalone Python scripts or Jupyter Notebooks without launching the GUI.

.. contents:: Table of Contents
   :local:
   :depth: 2

Complete Runnable Script Example
--------------------------------

The following script demonstrates a complete automated workflow: loading a 2D dataset, pre-processing, running a 2-component VarPro global fit, and saving publication-ready plots.

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from spark.data.dataset import SpectroscopyDataset
   from spark.data.importer import load_data
   from spark.core.fitting.global_fitter import GlobalFitter
   from spark.core.fitting.optimizer import fit

   # -------------------------------------------------------------
   # Step 1: Load Experimental Data
   # -------------------------------------------------------------
   # load_data automatically handles single CSVs or separate WL/TD files
   data_matrix, wl_axis, td_axis = load_data(auto_path='my_experiment.csv')

   # Wrap in SpectroscopyDataset container
   dataset = SpectroscopyDataset(data_matrix, wl_axis, td_axis)
   print(f"Loaded dataset: {dataset.get_shape()} (wavelengths x delays)")

   # -------------------------------------------------------------
   # Step 2: Pre-Processing Pipeline
   # -------------------------------------------------------------
   # Subtract baseline using first 10 time points (pre-time-zero)
   dataset.apply_baseline_correction(n_pts=10)

   # Crop spectral and temporal boundaries
   dataset.crop_wavelength(wl_min=450.0, wl_max=750.0)
   dataset.crop_time(t_min=-1.0, t_max=1500.0)

   # Exclude pump scatter band
   dataset.exclude_wavelength_ranges("525-535")

   # Normalize to max |signal| = 1.0
   dataset.normalize()

   # -------------------------------------------------------------
   # Step 3: Instantiate Global Fitter
   # -------------------------------------------------------------
   # Choose model: "Parallel" (DAS), "Sequential" (SAS), etc.
   num_exp = 2
   fitter = GlobalFitter(
       dataset=dataset,
       model_type="Parallel",
       num_exp=num_exp,
       use_artifact=True,       # Fit coherent artifact (Raman + XPM)
       artifact_mode="both",
       use_nnls=False           # Standard VarPro unconstrained linear solve
   )

   # -------------------------------------------------------------
   # Step 4: Configure Guesses, Bounds, and Constraints
   # -------------------------------------------------------------
   # Parameter layout: [w, t0, tau_1, tau_2]
   ini_guesses = np.array([0.10, 0.00,  1.5, 120.0])  # Starting values
   lower_bounds = np.array([0.01, -0.5, 0.05,   5.0])  # Box lower bounds
   upper_bounds = np.array([0.50,  0.5, 10.0, 5000.0]) # Box upper bounds
   is_fixed     = np.array([False, False, False, False]) # Parameter lock mask

   # -------------------------------------------------------------
   # Step 5: Execute VarPro Global Optimization
   # -------------------------------------------------------------
   print("Running Global Variable Projection Optimization...")
   fit_result = fit(
       dataset=dataset,
       build_c_func=fitter.build_C_matrix,
       ini=ini_guesses,
       limi=lower_bounds,
       lims=upper_bounds,
       is_fixed=is_fixed,
       num_exp=num_exp,
       use_nnls=False,
       enforce_tau_order=True   # Prevents tau crossing during optimization
   )

   # -------------------------------------------------------------
   # Step 6: Process Results & Covariance Analysis
   # -------------------------------------------------------------
   free_indices = np.where(~is_fixed)[0]
   fitter.process_fit_results(
       scipy_res=fit_result.scipy_res,
       full_x=fit_result.x,
       free_indices=free_indices
   )

   # Print summary report
   print("\nFit Optimization Succeeded!")
   print(f"Fitted IRF Width (FWHM): {fit_result.x[0]:.4f} ps")
   print(f"Fitted Time-Zero (t0):   {fit_result.x[1]:.4f} ps")
   for i, (tau, err) in enumerate(zip(fitter.extracted_taus, fitter.extracted_errtaus), 1):
       print(f"Tau {i}: {tau:.3f} +/- {err:.3f} ps")

   print(f"Jacobian Condition Number: {fitter.condition_number:.1f}")

   # -------------------------------------------------------------
   # Step 7: Save Results & Publication Figures
   # -------------------------------------------------------------
   fitter.save_results(outdir='fit_output/', ini_guesses=ini_guesses)
   print("Saved ASCII tables and GFitResults.npy to fit_output/")

Advantages of Scripting
-----------------------

* **Reproducibility**: Parameter values, crop limits, and model choices can be committed to Git repositories alongside laboratory notebooks.
* **High-Throughput Batch Processing**: Fit hundreds of automated datasets overnight on laboratory computing servers without user intervention.
* **Jupyter Notebook Integration**: Combine live code, interactive plots, and mathematical descriptions for lab meeting presentations or thesis documentation.
