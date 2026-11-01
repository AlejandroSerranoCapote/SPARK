SPARK Processing Modules
========================

SPARK is organized into specialized experimental processing modules, each tailored to specific spectroscopic measurement geometries and pre-treatment workflows.

.. toctree::
   :maxdepth: 2

   launcher
   flups
   tas
   mapper_2d

Module Summary
--------------

* **Main Launcher**: The centralized dashboard providing one-click access, navigation, and theme management across all modules.
* **FLUPS Analyzer**: Built for Fluorescence Upconversion and broad-band fluorescence datasets. Implements interactive 60 FPS exploration, pump scatter masking, and rigorous Group Velocity Dispersion (GVD / chirp) calibration.
* **TAS Analyzer**: Specialized for Transient Absorption Spectroscopy. Extends FLUPS with automated solvent background subtraction using 2D bivariate spline interpolation over arbitrary delay grids and relative amplitudes.
* **2D Mapper (XFEL)**: Converts collections of individual 1D energy or delay scans (e.g. from X-ray Free Electron Laser beamlines or time-resolved single-wavelength detectors) into standardized 2D spectral-temporal data matrices.
