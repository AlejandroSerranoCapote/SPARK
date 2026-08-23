.. SPARK documentation master file

Welcome to SPARK: Spectroscopy Pump-Probe Analysis and Research Kit
===================================================================

.. image:: _static/icon.png
   :alt: SPARK Logo
   :align: center
   :width: 150px

**SPARK** is an open-source Python suite designed for the processing and analysis of ultrafast spectroscopy data. It provides a comprehensive environment for handling transient absorption (**TAS**) and fluorescence up-conversion (**FLUPS**) measurements, featuring an integrated graphical user interface and a robust computational backend.

While it supports standard pre-processing tasks such as chirp correction and map generation, SPARK's primary analytical focus is its **Global Fitting Engine**. This module implements Variable Projection (VarPro) algorithms to handle parallel and sequential kinetic models, as well as user-defined systems through custom rate matrices.

Core Capabilities
-----------------

* **Global Fitting (VarPro):** Separation of non-linear parameters (lifetimes, IRF width, etc.) from linear amplitude spectra using Trust Region Reflective optimization.
* **Kinetic Models:** Support for Decay Associated Spectra (DAS), Species Associated Spectra (SAS), damped oscillations, and custom K-matrix models.
* **Statistical Diagnostics:** Confidence interval estimation via profile likelihood mapping and condition number evaluation.
* **Data Pre-processing:** Automated and manual data importers, baseline correction, and coherent artifact handling (Raman and XPM bases).
* **Publication Plotting:** Integrated tools for customizing and exporting high-resolution vector figures.

Getting Started
---------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   installation
   usage
   tutorials

Technical Reference
-------------------

.. toctree::
   :maxdepth: 2
   :caption: API Documentation:

   api/index
