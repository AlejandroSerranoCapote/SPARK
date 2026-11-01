Reports & Data Export
======================

SPARK provides automated, multi-page publication reports and standardized ASCII data export, ensuring complete transparency and reproducibility for research papers and supporting information.

.. contents:: Table of Contents
   :local:
   :depth: 2

Automated Multi-Page PDF Report
-------------------------------

Clicking **Export PDF Report** in `GlobalFitPanel` compiles a comprehensive vector PDF document via Matplotlib's `PdfPages` backend.

Structure of the Generated PDF Report:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Page 1: Executive Summary & Parameter Table**:
  - **Header Banner**: Project name, date, software version (v2.0.0), and analyst metadata.
  - **Model Specification Box**: Model topology (Parallel/Sequential/Custom), number of components, IRF settings, coherent artifact status, and NNLS constraint flag.
  - **Fit Quality & Identifiability Box**: Root Mean Square Deviation (RMSD), condition number ($\kappa$), and mathematical identifiability verdict (Well-Conditioned / Correlated / Ill-Conditioned).
  - **Pre-Processing Record**: Summary of baseline subtraction, spectral/temporal crop bounds, and binning factors.
  - **Fitted Parameters Table**: Lists each kinetic parameter ($w$, $t_0$, $\tau_1 \dots \tau_K$, $\beta_i$) with its Initial Guess, Final Optimized Value, Standard Error, and Physical Units. Parameters suffering from poor identifiability are highlighted in amber.

* **Page 2: Parameter Correlation Matrix**:
  - An annotated heatmap displaying Pearson correlation coefficients ($R_{ij}$) between all pairs of non-linear parameters.
  - Uses the symmetric diverging ``RdBu_r`` palette ($-1.0$ dark blue, $0.0$ white, $+1.0$ dark red).

* **Page 3: 2D Triplet Maps**:
  - Three synchronized heatmaps displayed side-by-side:
    1. Pre-processed Experimental Data $\Delta A(\lambda, t)$.
    2. Global VarPro Reconstructed Fit.
    3. Residual Map ($D - D_{\text{fit}}$).
  - Uses symmetric zero-centered color scaling with symlog delay axis.

* **Page 4: Associated Spectra & Residuals Distribution**:
  - Top: Decay-Associated Spectra (DAS) or Species-Associated Spectra (SAS) with $\pm 1\sigma$ error bars.
  - Middle: Coherent artifact spectrum (Raman, XPM) if enabled.
  - Bottom: Histogram of residuals compared against an ideal Gaussian distribution to verify statistical normality.

* **Page 5: Representative Multi-Wavelength Kinetics**:
  - Time-resolved decay curves extracted at four representative positions across the probe spectrum (20%, 40%, 60%, and 80% of the spectral range).
  - Experimental points overlaid with continuous fit lines on dual linear/symlog time axes.

* **Page 6: Singular Value Decomposition (SVD)**:
  - Scree plot of singular values ($\sigma_k$).
  - Spectral singular vectors ($\mathbf{u}_k$) across wavelength.
  - Temporal singular vectors ($\mathbf{v}_k$) across delay time.

Standard ASCII Data Export
--------------------------

When saving fit results, SPARK creates a dedicated folder containing plain text files ready for import into Origin, MATLAB, Python, or Igor Pro:

* **``Amplitudes.txt``**:
  - Header comments recording IRF width ($w$), $t_0$, all fitted lifetimes with uncertainty bounds ($\tau_i \pm \Delta\tau_i$), and model type.
  - Data columns: ``Wavelength | A1 | A1_err | A2 | A2_err | ... [Artifact columns]``.
* **``Corrected_data.txt``**: The 2D matrix of pre-processed experimental data.
* **``Fitted_data.txt``**: The 2D matrix evaluated from the fitted model.
* **``Residuals.txt``**: The 2D difference matrix ($D - D_{\text{fit}}$).
* **``WL.txt``**: 1D vector of wavelength coordinates.
* **``TD.txt``**: 1D vector of delay time coordinates.
* **``GFitResults.npy``**: Serialized NumPy dictionary containing the complete numerical workspace.

HDF5 Project Files (``.h5``)
----------------------------

SPARK projects can be saved at any stage via **Workspace** → **Save Project (HDF5)**:
* Stores all raw and pre-processed data matrices.
* Stores the full configuration of the model, initial guesses, bounds, and locked flags.
* Stores fitted matrices, linear spectra, and covariance matrices.
* Loading an ``.h5`` file restores the entire session instantly, allowing you to re-examine plots, tweak guesses, or run additional diagnostic analyses at any time.
