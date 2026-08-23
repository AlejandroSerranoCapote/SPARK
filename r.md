# SPARK: Spectroscopy Pump-Probe Analysis and Research Kit

[![PyPI version](https://img.shields.io/pypi/v/spark-spectroscopy.svg)](https://pypi.org/project/spark-spectroscopy/)
[![Documentation Status](https://readthedocs.org/projects/ultrafast-spectroscopy-analyzer/badge/?version=latest)](https://ultrafast-spectroscopy-analyzer.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**SPARK** is an open-source Python suite for the comprehensive processing, analysis, and kinetic modeling of ultrafast spectroscopy data (TAS, FLUPS, and XTAS). It combines an intuitive Graphical User Interface (GUI) with a high-performance mathematical backend, designed to streamline data handling from standard table-top setups as well as Synchrotron and XFEL facilities.

##  Core Capabilities

- **Advanced Global Fitting:** Powered by a Variable Projection (VarPro) engine using Trust Region Reflective optimization. Fits Parallel (DAS), Sequential (SAS), and Damped Oscillation models seamlessly.
- **Visual Kinetic Modeler:** Build complex Jablonski/Grotrian kinetic schemes visually. Draw states and transitions to automatically generate and diagonalize custom K-matrices.
- **Robust Error Analysis:** Computes true confidence intervals via **Profile Likelihood** mapping, going beyond standard covariance-based error estimation to ensure parameter identifiability.
- **Artifacts & Pre-processing:** Automated chirp correction (polynomial and non-linear physical models), baseline corrections, and coherent artifact handling (Raman/XPM bases).
- **XFEL & Timescan Ready:** Native support for batch-processing X-ray Free-Electron Laser and synchrotron `.npy` kinetic traces into fully calibrated 2D maps.
- **Publication-Ready Plotting:** An integrated drag-and-drop plotting tool to customize and export high-resolution (600 DPI) vector figures ready for scientific journals.

##  Quick Start

### Installation
Install SPARK directly via PyPI. We recommend using a virtual environment (like conda or venv) to ensure a clean installation of all UI and scientific dependencies.

```bash
pip install spark-spectroscopy
```

### Launching the Application
Once installed, launch the main SPARK dashboard directly from your terminal:

```bash
spark-gui
```
*(Note: If command-line entry points are not configured, you can launch the app by cloning the repository and running `python main.py`).*

## 📖 Documentation
For detailed installation instructions, user tutorials, and the complete API reference, please visit the [Official Documentation on Read the Docs](https://ultrafast-spectroscopy-analyzer.readthedocs.io/).

## 📄 License
SPARK is distributed under the MIT License. See `LICENSE` for more information.
