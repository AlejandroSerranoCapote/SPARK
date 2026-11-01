Optimization, Identifiability & Solvers
=========================================

Fitting complex multiexponential models to experimental data requires robust numerical optimization and rigorous uncertainty quantification. SPARK implements advanced diagnostics to verify that fitted parameters are physically meaningful and mathematically identifiable.

.. contents:: Table of Contents
   :local:
   :depth: 2

The Optimization Engine
-----------------------

Global fitting in SPARK is coordinated by ``GlobalFitter`` (``spark/core/fitting/global_fitter.py``) and executed by the multithreaded worker ``FitWorker`` (``spark/core/fitting/optimizer.py``):

1. **Trust-Region Reflective (TRF) Algorithm**:
   - Uses SciPy's bounded nonlinear least-squares solver (`scipy.optimize.least_squares`).
   - Strictly enforces box bounds on physical parameters (e.g., $\tau > 0$, $w > 0$, $0 < \beta \le 1$).
2. **Robust Loss Function (`loss='soft_l1'`)**:
   - Standard least-squares ($L_2$ norm) is vulnerable to optical spikes or detector cosmic rays.
   - SPARK employs the soft-$L_1$ loss metric:

     .. math::

        \rho(z) = 2 \left( \sqrt{1 + z} - 1 \right), \quad z = \frac{r^2}{\sigma^2}

   - Exhibits quadratic behavior near the minimum for smooth convergence, while transitioning to linear penalization for large outliers.
3. **Tau-Ordering Soft Penalty**:
   - In unconstrained multi-component models, permutation symmetry allows lifetimes to cross (e.g., $\tau_1$ and $\tau_2$ swapping roles during iteration).
   - SPARK enforces monotonic ordering ($\tau_1 < \tau_2 < \dots < \tau_K$) by applying a soft quadratic barrier penalty whenever $\tau_i \ge \tau_{i+1}$:

     .. math::

        P_{\text{order}} = 10^5 \cdot \max(0, \, \tau_i - \tau_{i+1})

4. **Non-Blocking Multithreading (`FitWorker`)**:
   - Fits run inside a dedicated Qt background thread (`QThread`).
   - The user interface remains fully responsive, featuring a live progress bar and an **ABORT** button to safely interrupt runaway fits.

Initial Guess & Bound Editor
----------------------------

Clicking **Edit Initial Guesses** opens the parameter table:

* **Parameter Values**: Manually adjust starting values for $w$, $t_0$, $\tau_i$, $\beta_i$.
* **Bounds**: Configure lower and upper boundaries to prevent the solver from entering non-physical parameter spaces.
* **Fix Checkbox**: Lock known experimental parameters (e.g., fixing the IRF width $w = 0.12\text{ ps}$ determined from an independent cross-correlation measurement).
* **"Use Last Fit" Button**: Loads the results of the previous optimization run as starting guesses, speeding up iterative refinements.
* **Undo / Redo Stack**: Use ``Ctrl+Z`` and ``Ctrl+Y`` to step backward and forward through up to 20 historical parameter configurations.

Parameter Identifiability Analysis
----------------------------------

A fit may achieve a low residual sum of squares even if individual parameters are physically ill-determined due to strong correlations. SPARK audits parameter identifiability via Singular Value Decomposition of the Jacobian matrix:

.. math::

   J = U_J \, \Sigma_J \, V_J^T

Condition Number Metric ($\kappa$)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The condition number is the ratio of the largest to smallest singular value of the Jacobian:

.. math::

   \kappa = \frac{s_{\max}}{s_{\min}}

SPARK provides a color-coded health indicator:
* **Green ($\kappa < 100$)**: *Well-Conditioned*. All kinetic parameters are independently determined with narrow, trustworthy confidence intervals.
* **Amber ($100 \le \kappa < 1000$)**: *Moderately Correlated*. Moderate parameter coupling exists; consider fixing known parameters or narrowing bounds.
* **Red ($\kappa \ge 1000$)**: *Ill-Conditioned / Over-Parameterized*. The model contains too many components or highly correlated rates that cannot be uniquely resolved by the data.

The "Sloppiest Direction"
^^^^^^^^^^^^^^^^^^^^^^^^^
The right singular vector $\mathbf{v}_{\min}$ corresponding to the smallest singular value $s_{\min}$ identifies the exact linear combination of parameters along which the objective function is flattest ("sloppiest"). SPARK inspects this vector and alerts the user to which specific parameters are co-varying (e.g., *"Strong mutual compensation between $\tau_2$ and $\tau_3$"*).

Parameter Correlation Matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Computed from the normalized covariance matrix $C_{\text{cov}} = (J^T J)^{-1}$:

.. math::

   R_{ij} = \frac{C_{ij}}{\sqrt{C_{ii} C_{jj}}}

Correlation coefficients approaching $\pm 1.0$ highlight parameter pairs that cannot be resolved independently.

Profile Likelihood Analysis
---------------------------

For rigorous non-asymmetric confidence intervals, open **Parameter Identifiability Dialog**:

1. Select a target parameter (e.g. $\tau_2$).
2. Choose a confidence level: **68%** ($1\sigma$), **90%**, **95%** ($2\sigma$), or **99%** ($3\sigma$).
3. Click **Run Profile Likelihood**:
   - SPARK steps the target parameter across an evaluation grid (e.g., $\pm 6$ standard errors).
   - At each grid point, the parameter is held fixed while all *other* free parameters are re-optimized to their conditional minimum.
   - A plot of the profile $\chi^2(\theta)$ is rendered with the statistical threshold line:

     .. math::

        \chi^2_{\text{threshold}} = \chi^2_{\min} + \chi^2_{1, 1-\alpha}

   - **Identifiable**: The profile forms a distinct bowl crossing the threshold on both sides, yielding exact upper and lower non-linear confidence limits.
   - **Practically Non-Identifiable**: The curve flattens out on one or both sides without crossing the threshold, demonstrating that the parameter can approach infinity without degrading fit quality.

Batch Fitting Across Multiple Datasets
--------------------------------------

When investigating series of measurements (e.g., concentration series, excitation power dependencies, temperature series):

1. Load all datasets into the Multi-Dataset Manager.
2. Configure the desired kinetic model, initial guesses, and bounds on the primary dataset.
3. Click **RUN BATCH FIT**:
   - SPARK fits each dataset sequentially using an asynchronous event loop.
   - Progress is displayed per dataset.
   - Automatically saves individual fit matrices and generates ``Batch_Summary.csv`` and ``Batch_Summary.txt`` summarizing RMSD, condition numbers, IRF widths, time-zeros, and all fitted lifetimes with uncertainty intervals across the entire sample series.
