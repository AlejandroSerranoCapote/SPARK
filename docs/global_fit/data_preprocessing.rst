Data Import & Pre-Processing
=============================

Before performing global kinetic fitting, experimental data must be validated, cleaned of optical artifacts, and aligned. SPARK provides a comprehensive pre-processing suite implemented in ``spark/data/dataset.py`` and ``spark/data/importer.py``.

.. contents:: Table of Contents
   :local:
   :depth: 2

Universal File Importer & Spreadsheet GUI
-----------------------------------------

SPARK accepts spectroscopy data in a wide variety of formats:
* **HDF5 Projects** (``.h5``): Contains full raw data, pre-processing settings, fitted models, and covariance matrices.
* **NumPy Dictionaries** (``.npy``): Containing array keys ``'data_c'``, ``'WL'``, ``'TD'``.
* **Delimited Text & Spreadsheets** (``.csv``, ``.txt``, ``.dat``): Tab-, comma-, or whitespace-delimited files.

Interactive Spreadsheet Dialog (`DataImporterDialog`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When loading an unfamiliar or irregularly structured text file, SPARK opens an interactive spreadsheet dialog:

.. code-block:: text

   ┌────────────────────────────────────────────────────────────────────────┐
   │  SPARK Data Importer                                                   │
   ├────────────────────────────────────────────────────────────────────────┤
   │  [ Auto-Detect Layout ]  [ Reset ]                                     │
   │  [ Assign Wavelength (Green) ]  [ Assign Time Delay (Blue) ]           │
   │  [ Assign Data Matrix (Orange) ]                                       │
   │ ────────────────────────────────────────────────────────────────────── │
   │   Col 0 (WL) │ Col 1 (t=-1.0) │ Col 2 (t=0.0) │ Col 3 (t=1.0) ...      │
   │   450.0 nm   │   0.00012      │   0.00341     │   0.00892              │
   │   452.0 nm   │  -0.00004      │   0.00310     │   0.00840              │
   │   454.0 nm   │   0.00002      │   0.00289     │   0.00788              │
   │ ────────────────────────────────────────────────────────────────────── │
   │  Status: WL=350 pts (Green) | TD=120 pts (Blue) | DATA=350x120 (Orange)│
   │  [ Confirm and Load Data ]                                             │
   └────────────────────────────────────────────────────────────────────────┘

1. **Auto-Detect**: Automatically scans for numeric boundaries, skipping metadata headers or measurement footers.
2. **Interactive Selection**: Highlight cells with your mouse and click the corresponding role button:
   - Green: Wavelength axis coordinates.
   - Blue: Time delay axis coordinates.
   - Orange: Data matrix values ($\Delta A$, fluorescence counts).
3. **Orientation Check**: If the user highlights transposed rows/columns, SPARK detects the dimensional inversion and automatically transposes the matrix upon confirmation.

Multi-Dataset Management
------------------------

SPARK supports loading multiple experimental datasets simultaneously within a single session:
* **Active Dataset Dropdown**: Switch between different excitation powers, solvent conditions, or temperatures with one click.
* **Compare Kinetics (`CompareSetupDialog`)**: Select a target wavelength (e.g. 520 nm) and plot the temporal decay curves across all loaded datasets on a single axis with custom color pickers and legends.
* **Difference Map (`DiffMapDialog`)**: Select two datasets $A$ and $B$ to compute and display a 2D subtraction map ($A - B$) for isolating subtle photoproduct signatures.

Data Units Conversion
---------------------

The left panel includes unit converters:
* **Spectral Units**: ``nm`` (nanometers), ``eV`` (electron-volts), or ``cm⁻¹`` (wavenumbers).
* **Temporal Units**: ``ps`` (picoseconds), ``fs`` (femtoseconds), or ``ns`` (nanoseconds).
Conversions are applied dynamically without modifying original raw data on disk.

The Pre-Processing Pipeline
---------------------------

Pre-processing transformations are executed non-destructively on a working copy (``SpectroscopyDataset``):

1. Baseline Correction (`spin_bl`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Subtracts the average signal of the first $N$ delay points (pre-time-zero baseline) from each wavelength trace:

.. math::

   D_{\text{corr}}(\lambda_i, t_j) = D(\lambda_i, t_j) - \frac{1}{N_{\text{bl}}} \sum_{m=1}^{N_{\text{bl}}} D(\lambda_i, t_m)

2. Spectral & Temporal Cropping
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Isolate the physical region of interest using ``Wavelength Min/Max`` and ``Time Min/Max`` spin boxes. Eliminates noisy edge detector pixels or mechanical stage run-out delays.

3. Wavelength Exclusion Band (`line_exclude`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Enter comma-separated ranges (e.g. ``"495-535, 790-810"``) to excise laser pump scatter or probe filter cut-offs. The excluded channels are removed from the global fitting matrix, preventing distortion of the residual sum of squares.

4. Spatial & Temporal Binning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* **Spectral Binning (`spin_bin`)**: Averages $K$ adjacent wavelength pixels together. Useful for improving signal-to-noise ratio in low-light spectroscopy.
* **Temporal Binning (`spin_time_binning`)**: Averages adjacent time delay points along the kinetics axis.

5. Zero Negative Times (`chk_zero_neg`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Sets all data points where $t < 0$ strictly to $0.0$, ideal for systems with clean step-like excitation and negligible pre-pump thermal signals.

6. Normalization (`chk_norm_data`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Scales the entire matrix such that $\max |D| = 1.0$. This ensures numerical stability during least-squares optimization regardless of whether raw input was recorded in OD, mOD, or counts.
