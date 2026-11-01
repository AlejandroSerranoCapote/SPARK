Tutorial 2: TAS Pre-processing & Solvent Subtraction
======================================================

In this tutorial, we process a raw Transient Absorption measurement contaminated by solvent cross-phase modulation and cuvette window artifacts, subtract the solvent signal with sub-picosecond accuracy, and transfer the treated data to the Global Fit engine.

.. contents:: Table of Contents
   :local:
   :depth: 2

Goal of this Tutorial
---------------------
* Load a paired sample-and-solvent transient absorption measurement.
* Use the real-time interpolation engine to cancel solvent Raman and cross-phase modulation.
* Mask pump scatter fringes.
* Perform chirp correction and hand-off to Global Fit.

Step 1: Open the TAS Analyzer
-----------------------------

From the SPARK Launcher, click the orange **TAS ANALYZER** card.

Step 2: Load Sample & Solvent Files
-----------------------------------

1. Click **Load File** in the top-left **Data Source** group box.
2. Select your experimental sample file: ``sample_data.csv``.
3. SPARK immediately prompts for the solvent blank file: select ``solvent_blank.csv``.
4. Both datasets load into memory. The 2D false-color map renders the raw sample data.

Step 3: Align & Subtract the Solvent
------------------------------------

Navigate to the **TAS Adjustments** section in the left sidebar:

1. **Observe Early Times**: Move your cursor on the 2D map to $t \approx 0\text{ ps}$. In the bottom-right spectrum slice, notice sharp solvent Raman peaks and bipolar derivative oscillations from cross-phase modulation (XPM).
2. **Fine-Tune Amplitude (`slider_am`)**:
   - Slowly slide the amplitude slider between 80% and 120%.
   - As amplitude increases, the solvent peaks in the marginal spectrum decrease.
   - Click the label to enter an exact value (e.g. ``98.5%``).
3. **Fine-Tune Temporal Shift (`slider_sf`)**:
   - Small differences in optical path length (e.g. $100\ \mu\text{m}$ difference in cuvette wall thickness) introduce a temporal delay $\Delta t$ between sample and solvent.
   - Adjust the shift slider in sub-picosecond increments (e.g. ``+0.045 ps``).
   - Watch the bipolar XPM oscillations disappear as the temporal overlap reaches optimum alignment.

Step 4: Mask Pump Scatter
-------------------------

If your laser pump pulse (e.g. at 400 nm) scatters into the spectrometer:

1. Click **Remove Pump Fringe**.
2. A dialog prompts for the scatter wavelength range.
3. Enter ``395 - 405`` and click **Apply**.
4. The saturated scatter band is masked, preventing distortion of the colormap dynamic range.

Step 5: Calibrate Chirp (Dispersion Correction)
----------------------------------------------

With solvent artifacts eliminated, the probe chirp is clearly visible as a curved rising edge:

1. Click **Select t₀ Points**.
2. Click 5–6 points along the rising edge of the signal across the spectrum (e.g., at 450 nm, 500 nm, 550 nm, 600 nm, 650 nm, 700 nm).
3. Select ``Nonlinear Model`` from the model dropdown.
4. Click **Fit t₀ Curve**: SPARK fits the physical dispersion curve and overlays it onto the data.
5. Click **Show Corrected Map**: The data matrix is resampled along the delay axis, aligning time-zero into a flat horizontal line across all wavelengths.

Step 6: Send to Global Fit
--------------------------

1. Click the green **Global Fit** button at the bottom of the sidebar.
2. SPARK packages the corrected data and opens `GlobalFitPanel`.
3. You are now ready to perform kinetic model fitting on pristine, artifact-free transient absorption data.
