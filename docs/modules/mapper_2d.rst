2D Mapper / XFEL Assembly
==========================

The **2D Mapper** module (implemented in ``spark/gui/maps_from_timescans.py``) is designed for experimental setups where 2D spectroscopy maps are not recorded as single images, but assembled from sequences of individual 1D time scans across multiple discrete excitation or probe energies.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

In synchrotron facilities, X-ray Free Electron Laser (XFEL) beamlines, or single-channel monochromator setups, an experiment often records hundreds of separate files—each containing a delay scan at one specific photon energy or wavelength.

The 2D Mapper utility automates the extraction, unit conversion, alignment, and assembly of these individual scans into a unified 2D spectroscopy dataset matrix $(N_{\text{energy}} \times N_{\text{time}})$.

Architecture
------------

The tool comprises two components:
1. ``XFELProcessor`` (Backend): Reads file dictionaries, validates key structures, reconciles time unit scales, and interpolates data onto a common delay grid.
2. ``AppWindow`` (GUI): An interface for configuring dictionary keys, supplying energy axes, batch-loading scan files, and visualizing the assembled map.

Configuration & Key Mapping
---------------------------

Different data acquisition systems store time-resolved arrays under varying dictionary keys. The configuration panel allows mapping custom dictionary field names:

* **Time Key** (default: ``'time'``): Delay axis vector.
* **Excited State Key** (default: ``'signal'`` or ``'es'``): Signal with pump pulse ON.
* **Ground State Key** (default: ``'reference'`` or ``'gs'``): Signal with pump pulse OFF.
* **Direct Calculation Key** (default: ``'delta'``): Pre-computed differential signal ($\Delta I / I$ or $\Delta A$).
* **Time Scale**: Select ``ps`` (picoseconds) or ``fs`` (femtoseconds).

Automatic Unit Detection
------------------------

Click **Check Units**:
* ``XFELProcessor.analyze_units()`` scans the numerical range of the time vectors across all loaded files.
* If maximum delay values exceed $10^3$, SPARK automatically alerts the user that data is in femtoseconds and scales it to picoseconds for consistency with standard kinetic models.

Energy Vector Input
-------------------

The assembled 2D map requires an energy or wavelength coordinate for each loaded scan file:
* **Manual Input**: Enter comma-separated values (e.g. ``520.0, 525.0, 530.0, 535.0, ...``).
* **Import from File**: Click **Import from TXT** to load a column of calibrated energies or wavelengths from an external ASCII file.

Count Validation
^^^^^^^^^^^^^^^^
SPARK's ``validate_counts()`` dynamically monitors the file list and energy vector:
- If count of files matches count of energy points, a green status indicator confirms readiness.
- If there is a mismatch (e.g., 48 files loaded but 50 energy points provided), the **Generate Map** button is locked, preventing misaligned coordinates.

Assembling and Exporting the 2D Map
-----------------------------------

1. Click **Generate Map**:
   - ``XFELProcessor.process()`` extracts traces from each file.
   - Interpolates all scans onto a common delay axis.
   - Stacks traces into a 2D matrix ($N_{\text{energy}} \times N_{\text{time}}$).
   - Displays the assembled 2D heatmap on the embedded Matplotlib canvas.
2. Click **Save Map**:
   - Saves the assembled dataset as a standardized `.npy` dictionary:
     .. code-block:: python

        {
            'data': data_matrix,        # shape: (n_energy, n_delays)
            'wavelengths': energy_arr,   # 1D array
            'delays': delay_arr          # 1D array
        }
   - This file can be imported directly into the **Global Fit** workstation.
