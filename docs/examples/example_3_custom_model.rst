Tutorial 3: Custom Annihilation Model in Equation Editor
=========================================================

In this tutorial, we construct a custom non-linear photochemical model describing exciton dynamics in an organic semiconductor film with competing monomolecular decay, bimolecular exciton-exciton annihilation (EEA), and charge-transfer polaron formation.

.. contents:: Table of Contents
   :local:
   :depth: 2

Goal of this Tutorial
---------------------
* Build a multi-state reaction model containing both first-order transitions and second-order non-linear terms.
* Use the Equation Editor and verify the model via symbolic LaTeX preview.
* Compile and fit the model against experimental data.

Step 1: The Photochemical Mechanism
-----------------------------------

Upon pulsed laser excitation, singlet excitons ($S_1$) are generated at $t = 0$:

1. **Monomolecular Decay**: $S_1 \xrightarrow{k_1} S_0$ (spontaneous radiative/non-radiative decay).
2. **Exciton-Exciton Annihilation (EEA)**: $S_1 + S_1 \xrightarrow{\gamma_{\text{ann}}} S_1 + S_0$ (density-dependent quenching).
3. **Charge Separation**: $S_1 \xrightarrow{k_{\text{cs}}} P$ (formation of polarons or charge-transfer states).
4. **Polaron Recombination**: $P \xrightarrow{k_{\text{rec}}} S_0$ (decay of polarons to ground state).

Differential Equations:
^^^^^^^^^^^^^^^^^^^^^^^
.. math::

   \frac{d[S_1]}{dt} = -(k_1 + k_{\text{cs}}) [S_1] - \gamma_{\text{ann}} [S_1]^2

   \frac{d[P]}{dt} = +k_{\text{cs}} [S_1] - k_{\text{rec}} [P]

Step 2: Open the Model Builder
------------------------------

1. In `GlobalFitPanel`, navigate to the **Fit** tab.
2. In the Model dropdown, select **Custom GUI Model**.
3. Click the **Open Visual Model Builder** button (`btn_build_model`).

Step 3: Script the Model in the Equation Editor
----------------------------------------------

Switch to the **Equation Editor** tab:

1. **Set Initial States (`txt_initial_states`)**:
   .. code-block:: text

      S1

2. **Set Dark States (`txt_dark_states`)**:
   .. code-block:: text

      S0

3. **Enter Differential Equations (`txt_equations`)**:
   .. code-block:: text

      # Singlet exciton dynamics with bimolecular annihilation
      dS1/dt = -(k1 + k_cs)*S1 - gamma*S1**2

      # Charge-separated polaron dynamics
      dP/dt  =  k_cs*S1 - k_rec*P

Step 4: Validate & Preview LaTeX
--------------------------------

Click the **Preview LaTeX** button:
* SymPy parses the mathematical tokens and verifies that all variables and derivatives are well-formed.
* The LaTeX preview canvas renders the system of equations in clear mathematical typography.
* A green status message confirms: *"Syntax Valid: Non-linear system detected (will use BDF numerical integrator convolved with Gaussian IRF)"*.

Step 5: Compile & Apply
-----------------------

1. Click **Compile and Apply** at the bottom of the dialog.
2. The dialog closes and returns to `GlobalFitPanel`.
3. In the **Fit** tab, click **Edit Initial Guesses**:
   - The table displays your custom parameters: ``w``, ``t0``, ``k1``, ``k_cs``, ``gamma``, ``k_rec``.
   - Enter physically reasonable starting estimates:
     * ``k1``: $0.005\text{ ps}^{-1}$ ($\tau \approx 200\text{ ps}$)
     * ``k_cs``: $0.1\text{ ps}^{-1}$ ($\tau_{\text{cs}} \approx 10\text{ ps}$)
     * ``gamma``: $0.02$
     * ``k_rec``: $0.001\text{ ps}^{-1}$ ($\tau_{\text{rec}} \approx 1000\text{ ps}$)
4. Click **RUN FIT**:
   - SPARK integrates the non-linear ODE system for each candidate parameter set using the BDF integrator.
   - Computes species-associated spectra $S_{S_1}(\lambda)$ and $S_{P}(\lambda)$.
   - Displays the fitted 2D map and individual species concentration profiles.
