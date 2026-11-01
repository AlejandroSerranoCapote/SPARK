Changelog & Release Notes
==========================

All notable changes to the **SPARK** software suite are documented here.

Version 2.0.0 (Current Release)
-------------------------------

Release Date: November 2026

Major Features & Architecture Overhaul
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* **Unified v2.0.0 Architecture**: Standardized version identifier across the launcher dashboard, GUI panels, PyInstaller specs, and documentation.
* **Variable Projection (VarPro) Engine**: Fully decoupled ``GlobalFitter`` and ``optimizer.fit`` from PyQt5, enabling headless scripted execution and multithreaded GUI operations.
* **Visual Model Builder & ODE Compiler**:
  - Drag-and-drop Grotrian state diagrams on Qt Graphics Scene.
  - Symbolic SymPy ODE equation editor with live LaTeX rendering.
  - Hybrid compiler automatically selecting analytical K-matrix diagonalization for linear systems and stiff BDF numerical integration for non-linear systems.
  - Pre-configured kinetic presets: Exciton-Exciton Annihilation (EEA), Triplet-Triplet Annihilation (TTA), Reversible Kinetics, Bimolecular Recombination, and Exciton-Polaron models.
* **Uncertainty & Identifiability Suite**:
  - SVD Jacobian analysis with Condition Number health indexing.
  - Automated "sloppiest parameter direction" identification.
  - Parameter correlation matrix heatmap in publication reports.
  - Profile Likelihood non-linear $\chi^2$ confidence profiling dialog.
* **Publication Plotting Ecosystem**:
  - Standalone 600 DPI kinetics designer (``PaperPlotterWindow``) with scientific palettes (Nature, Viridis, Plasma).
  - Standalone 600 DPI spectra designer (``SASDASPlotterWindow``) with error bar controls.
  - Multi-page vector PDF publication report compiler.
* **Interactive Trace Explorer**: Added component decomposition checkbox displaying individual kinetic component curves in distinct colors on dual linear/symlog time axes.
* **Universal Data Importer**: Spreadsheet mapping GUI (``DataImporterDialog``) with auto-detection of delimited files (CSV, TXT, DAT, NPY).

Bug Fixes & Refinements
^^^^^^^^^^^^^^^^^^^^^^^
* **LaTeX Greek Legend Rendering**: Fixed string escaping in Matplotlib legends; raw f-strings (``rf"$\tau_{...}$"`` and ``$\pm$``) now render proper Greek $\tau$ and $\beta$ symbols instead of literal "au" or "eta".
* **HDF5 Project Load & Show Plots**:
  - Resolved `UnboundLocalError: local variable 'spectroscopydataset' referenced before assignment` during HDF5 project loading.
  - Restored proper initialization of `self.fitted_model_type` during project loading, preventing crashes upon clicking **Show Plots**.
* **Coherent Artifact Normalization**: Standardized Gaussian IRF basis functions ($\phi_0, \phi_1, \phi_2$) normalized to unity.
* **Standalone Windows Installer**: Packaged full application payload into Tkinter installer wizard (`SPARK_Setup.exe`) with desktop/Start Menu shortcuts and clean uninstaller.
