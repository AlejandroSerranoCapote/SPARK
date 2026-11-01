TAS Analyzer (Transient Absorption Spectroscopy)
===================================================

The **TAS Analyzer** module (implemented in ``spark/gui/tas.py``) extends ``FLUPSAnalyzer`` to address the specific experimental challenges of Transient Absorption Spectroscopy, most notably solvent background and cuvette cross-phase modulation subtraction.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview & Inheritance
-----------------------

In Transient Absorption measurements, solvent responses (Raman scattering, two-photon absorption, cross-phase modulation) often overlap with early-time solute dynamics. To recover the pristine solute signal, an independent solvent measurement must be subtracted with precise temporal and amplitude alignment.

``TASAnalyzer`` subclasses ``FLUPSAnalyzer``, inheriting its high-performance blitting canvas, cropping controls, and chirp correction engine, while introducing a dedicated **TAS Adjustments** control panel.

Dual-Dataset Loading Workflow
-----------------------------

When clicking **Load File** in TAS Analyzer:
1. SPARK prompts for the **Sample Measurement** data file (CSV, TXT, or DAT format).
2. Immediately following, SPARK prompts for the corresponding **Pure Solvent / Blank** data file.
3. Both matrices are parsed into internal memory grids.

Real-Time Solvent Subtraction Engine
------------------------------------

The sidebar features two primary controls under the **TAS Adjustments** group:

1. **Amplitude Slider (`slider_am`)**:
   - Range: 0% to 200% (default: 100%).
   - Controls the scaling factor $\alpha$ of the solvent signal.
2. **Temporal Shift Slider (`slider_sf`)**:
   - Range: -200 ps to +200 ps in sub-picosecond increments.
   - Adjusts the relative delay shift $\Delta t$ between sample and solvent.

.. tip::
   Click directly on the numeric labels next to the sliders to open a direct numerical input dialog for high-precision entry (e.g., entering exactly ``0.942`` for amplitude or ``-0.035 ps`` for temporal shift).

Mathematical Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Because sample and solvent datasets may be acquired with slightly differing delay time samplings or mechanical delay stage drifts, SPARK uses `scipy.interpolate.RegularGridInterpolator` to evaluate the solvent data continuously:

.. math::

   \Delta A_{\text{pure}}(\lambda, t) = \Delta A_{\text{sample}}(\lambda, t) - \alpha \cdot \mathcal{I}_{\text{solvent}}(\lambda, t + \Delta t)

As the user adjusts either slider, the subtraction is recomputed and the 2D map is redrawn in real time.

Pump Scatter & Fringe Masking
-----------------------------

Near the pump wavelength, intense Rayleigh and Raman scatter can saturate detector pixels:
- Click **Remove Pump Fringe** to specify a wavelength band $[\lambda_{\text{start}}, \lambda_{\text{stop}}]$.
- SPARK masks this region, preventing high-amplitude scatter spikes from biasing subsequent chirp fitting or global kinetic analysis.

Hot-Swapping Between TAS and FLUPS
----------------------------------

If an experiment does not require solvent subtraction, users can switch analyzers on the fly using ``switch_analyzer()``, preserving loaded dataset grids without restarting the software.
