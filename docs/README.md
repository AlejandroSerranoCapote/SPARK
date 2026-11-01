# SPARK Documentation (Read the Docs Source)

This directory contains the complete source documentation for the **SPARK** (*Spectroscopy Pump-probe Analysis and Research Kit*) software suite, configured for deployment on **[Read the Docs](https://readthedocs.org)** via **Sphinx**.

## Structure

```text
spark/
├── .readthedocs.yaml                  # Read the Docs configuration file
└── docs/
    ├── conf.py                        # Sphinx build configuration
    ├── requirements.txt               # Documentation dependencies
    ├── index.rst                      # Master documentation landing page & TOC
    ├── installation.rst               # Setup wizard, portable binary, source install
    ├── quickstart.rst                 # 5-minute hands-on tutorial with p1.h5
    ├── changelog.rst                  # Version 2.0.0 release notes and bug fixes
    │
    ├── modules/                       # Experimental pre-treatment modules
    │   ├── index.rst
    │   ├── launcher.rst               # Main application dashboard
    │   ├── flups.rst                  # Fluorescence Upconversion (chirp & blitting)
    │   ├── tas.rst                    # Transient Absorption (solvent subtraction)
    │   └── mapper_2d.rst              # 2D map assembly from XFEL/time-scan files
    │
    ├── global_fit/                    # Global analysis & VarPro engine
    │   ├── index.rst                  # Variable Projection mathematics
    │   ├── data_preprocessing.rst     # Universal spreadsheet importer & cropping
    │   ├── svd.rst                    # SVD analysis & rank estimation
    │   ├── kinetic_models.rst         # DAS, SAS, KWW, Damped Oscillation, Artifacts
    │   └── solver_and_identifiability.rst # TRF optimizer, condition number, profiling
    │
    ├── equation_editor/               # Custom kinetic models & ODE compiler
    │   ├── index.rst                  # Hybrid compiler architecture
    │   ├── visual_builder.rst         # Interactive Grotrian canvas
    │   ├── ode_syntax.rst             # Pythonic ODE syntax rules
    │   ├── irf_convolution.rst        # Gaussian IRF formulation & normalization
    │   └── presets.rst                # EEA, TTA, Reversible, Bimolecular templates
    │
    ├── visualization_and_export/      # Scientific plotting & reporting
    │   ├── index.rst
    │   ├── explorers.rst              # Trace Explorer (decomposed curves) & 3D surface
    │   ├── publication_tools.rst      # PaperPlotter & SASDASPlotter (600 DPI)
    │   └── reports.rst                # Multi-page PDF reports & ASCII exports
    │
    ├── examples/                      # Illustrated step-by-step tutorials
    │   ├── index.rst
    │   ├── example_1_hdf5.rst         # Loading p1.h5, DAS analysis, component traces
    │   ├── example_2_tas_solvent.rst  # Solvent subtraction & chirp calibration
    │   ├── example_3_custom_model.rst # Scripting an Annihilation model
    │   └── example_4_scripting.rst    # Headless Python scripting & automation
    │
    ├── api/                           # Developer API reference
    │   ├── index.rst
    │   ├── dataset.rst                # SpectroscopyDataset class
    │   ├── importer.rst               # Universal file importer & spreadsheet GUI
    │   ├── global_fitter.rst          # GlobalFitter class & VarPro matrix generator
    │   ├── optimizer.rst              # fit() function & FitWorker thread
    │   ├── models.rst                 # DASModel, SASModel, KMatrixModel
    │   ├── chirp.rst                  # Sellmeier non-linear & polynomial chirp
    │   └── diagnostics.rst            # Reporting & diagnostic plotting
    │
    └── _static/
        └── images/                    # High-resolution illustrative figures
            ├── triplet_maps_fit.png   # 2D experimental, fit, and residuals maps
            ├── das_spectra_example.png# DAS spectra with tau values
            ├── trace_explorer_example.png # Trace Explorer with component curves
            └── kinetics_slices_fit.png# Multi-wavelength kinetic slices
```

## Building Locally

To build HTML documentation locally:

```bash
cd docs
pip install -r requirements.txt
sphinx-build -b html . _build/html
```

Then open `_build/html/index.html` in your web browser.

## Deploying to Read the Docs

1. Link your GitHub repository to [Read the Docs](https://readthedocs.org).
2. Read the Docs will automatically detect `.readthedocs.yaml` in the root folder.
3. Every commit or release tag pushed to the repository triggers an automated build.
