FLUPS Analyzer (Fluorescence Upconversion)
===========================================

The **FLUPS Analyzer** module (implemented in ``spark/gui/flups.py``) is designed for the inspection, calibration, and pre-processing of time-resolved luminescence and fluorescence upconversion data.

.. contents:: Table of Contents
   :local:
   :depth: 2

Layout & User Interface
-----------------------

The FLUPS interface is divided into an interactive control sidebar on the left and a 4-quadrant dynamic Matplotlib visualization canvas on the right:

* **Top-Right Main Canvas**: 2D false-color heatmap showing intensity as a function of wavelength ($\lambda$, horizontal) and delay time ($t$, vertical).
* **Bottom-Left Marginal Canvas**: Temporal decay trace at the current cursor wavelength.
* **Bottom-Right Marginal Canvas**: Emission spectrum slice at the current cursor delay time.

High-Performance Blitting Engine
--------------------------------

Standard Matplotlib canvas redraws can take 100–300 ms per frame, causing sluggish interactivity. FLUPS solves this with an optimized **blitting architecture**:

1. When the 2D heatmap is plotted or resized, `on_draw` captures and caches the rendered background pixel buffer.
2. As the user moves their mouse over the 2D map, `on_move_map` restores the cached pixel buffer and redraws *only* the dynamic crosshair lines and marginal 1D trace curves.
3. This achieves fluid **60 frames per second** interactivity even on massive datasets ($1000 \times 1000$ points).

Data Cropping & Filtering
-------------------------

* **Delay Axis Crop**: Set ``Delay Min`` and ``Delay Max`` via spin boxes to exclude noisy pre-time-zero data or long delays where signal decays into baseline noise.
* **Wavelength Axis Crop**: Adjust dual sliders or click labels to enter numerical limits ($\lambda_{\min}, \lambda_{\max}$) to isolate the region of interest.
* **Pump Fringe Removal**: Click **Remove Pump Fringe** to mask parasitic laser scatter at the excitation wavelength, replacing that spectral band with interpolated or zeroed values.

t₀ Chirp (Group Velocity Dispersion) Correction
-----------------------------------------------

In broad-band ultrafast spectroscopy, the probe pulse experiences Group Velocity Dispersion (GVD) when passing through optics, cuvette windows, and solvent, causing blue wavelengths to arrive at a different physical time than red wavelengths (chirp).

FLUPS offers four distinct calibration strategies:

1. Interactive Point Selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Click **Select t₀ Points** to activate click-capture mode on the 2D map.
- Click along the rising edge of the signal across multiple wavelength channels (at least 4 points recommended).
- Markers appear on the canvas marking the chosen $( \lambda_i, t_{0, i} )$ pairs.

2. Polynomial Dispersion Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Select polynomial degree (3, 4, 5, or 6) from the dropdown and click **Fit t₀ Curve**. SPARK fits:

.. math::

   t_0(\lambda) = \sum_{k=0}^{N} c_k \, \lambda^k

The fitted curve is overlaid onto the 2D map.

3. Physical Non-Linear Dispersion Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Select **Nonlinear Model** and click **Fit t₀ Curve**. SPARK uses the Sellmeier-derived non-linear dispersion equation:

.. math::

   t_0(\lambda) = a \sqrt{\frac{b\,\lambda^2 - 1}{c\,\lambda^2 - 1}} + d

where $a, b, c, d$ are parameters fitted via SciPy Trust-Region Reflective least squares (`curve_fit`). This provides superior extrapolation in spectral regions where experimental calibration points are sparse.

4. Automated Chirp Detection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Click **Auto-Detect Chirp**: SPARK computes the cross-correlation of early rising edges across all wavelength channels, automatically extracting time-zero coordinates and fitting the optimal dispersion curve without requiring manual clicking.

5. Manual Parameter Entry
^^^^^^^^^^^^^^^^^^^^^^^^^
Click **Enter Manually** to open `ManualChirpDialog`. Users can directly paste comma-separated coefficients or load them from a previously saved `_fit_params.txt` file.

Applying and Previewing Chirp Correction
----------------------------------------

* Click **Show Corrected Map**: SPARK shifts each wavelength trace along the delay axis by $-t_0(\lambda)$ using linear interpolation (`scipy.interpolate.interp1d`).
* The corrected 2D map is redrawn with an aligned, horizontal time-zero across all wavelengths.

Hand-off to Global Fit
----------------------

Click the **Global Fit** button at the bottom of the sidebar:
1. FLUPS automatically saves the calibrated, cropped, and chirp-corrected data matrix to temporary files (`_treated_data.npy`, `_WL.txt`, `_TD.txt`, `_fit_params.txt`).
2. Opens the `GlobalFitPanel` dialog with the treated data pre-loaded and ready for kinetic model fitting.
