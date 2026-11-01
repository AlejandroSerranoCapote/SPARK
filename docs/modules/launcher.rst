Main Launcher Dashboard
======================

The SPARK Launcher is the central access portal. It organizes the processing suite into modular workstations while preserving a cohesive design aesthetic.

.. contents:: Sections
   :local:
   :depth: 2

Overview & Architecture
-----------------------

Implemented in ``spark/main.py``, the launcher is managed by ``MainApp(QMainWindow)``. It features a two-panel responsive design styled with modern dark and navy palettes:

* **Left Identity Panel** (300 px fixed width):
  - Displays the golden **S.P.A.R.K.** brand banner.
  - Subtitle: *"Spectroscopy Pump-probe Analysis and Research Kit"*.
  - Version indicator: **v2.0.0**.
  - Direct link button to the official GitHub repository.

* **Right Workspace Panel**:
  - Divided into two functional categories:
    1. **Primary Processing Modules**: For raw experimental data pre-treatment.
    2. **Advanced Analysis & Visualization**: For global mathematical fitting and multidimensional dataset assembly.

Card Modules
------------

1. **FLUPS Analyzer** (Green accent):
   - Opens the Fluorescence Upconversion analysis workstation.
   - Designed for single-matrix data inspection, wavelength/time cropping, and dispersion chirp correction.

2. **TAS Analyzer** (Orange accent):
   - Opens the Transient Absorption Spectroscopy module.
   - Incorporates real-time solvent baseline subtraction alongside the standard FLUPS pre-treatment suite.

3. **Global Fit** (Blue accent):
   - Opens the advanced Global Variable Projection (VarPro) workstation (`GlobalFitPanel`).
   - Supports multi-dataset comparison, custom kinetic ODE modeling, SVD analysis, and publication reporting.

4. **2D Mapper** (Purple accent):
   - Opens the time-scan assembler (`AppWindow` / `XFELProcessor`).
   - Converts collections of individual `.npy` or `.txt` delay/energy traces into a unified 2D matrix.

Seamless Window Management
--------------------------

When a user clicks on any module card:
1. The launcher window hides smoothly to reduce screen clutter.
2. The chosen tool window opens in maximized or default geometry.
3. When the user closes the active tool window, SPARK's ``closeEvent`` interceptor automatically re-displays the launcher hub.
