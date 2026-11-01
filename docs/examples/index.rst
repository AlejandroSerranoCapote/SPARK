Step-by-Step Tutorials & Examples
==================================

These hands-on tutorials walk you through realistic spectroscopy workflows using SPARK, from raw data loading to publication figures and Python automation.

.. toctree::
   :maxdepth: 2

   example_1_hdf5
   example_2_tas_solvent
   example_3_custom_model
   example_4_scripting

Tutorial Overview
-----------------

1. **Tutorial 1: Loading an HDF5 Project & Analyzing DAS**:
   - Work with the included ``p1.h5`` dataset.
   - Inspect the 2D fit and residuals map.
   - Deconstruct multi-exponential kinetics inside the Trace Explorer.
2. **Tutorial 2: TAS Pre-processing & Solvent Subtraction**:
   - Load sample and solvent matrices.
   - Calibrate the sub-picosecond temporal delay shift ($\Delta t$) and amplitude scaling factor ($\alpha$).
   - Export treated data into Global Fit.
3. **Tutorial 3: Defining a Custom Annihilation Model in the Equation Editor**:
   - Model excitation-density dependent kinetics ($A \to \text{GS}$ with competing bimolecular annihilation $A + A \to \text{GS}$).
   - Compile using the hybrid symbolic-numeric engine.
   - Fit experimental power-dependent traces.
4. **Tutorial 4: Headless Python Scripting**:
   - Use ``SpectroscopyDataset`` and ``GlobalFitter`` directly in Python scripts or Jupyter Notebooks without launching the graphical interface.
