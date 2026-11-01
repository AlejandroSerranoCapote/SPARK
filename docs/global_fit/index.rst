Global Analysis & Variable Projection Engine
==============================================

The **Global Fit** workstation (implemented in ``spark/gui/global_fit_class_gui.py`` and powered by ``spark/core/fitting/global_fitter.py``) is the core analytical engine of SPARK.

.. toctree::
   :maxdepth: 2

   data_preprocessing
   svd
   kinetic_models
   solver_and_identifiability

The Variable Projection (VarPro) Philosophy
-------------------------------------------

In time-resolved spectroscopy, an experimental dataset $D(\lambda, t)$ recorded across $N_\lambda$ wavelengths and $N_t$ time delays is modeled as a linear combination of $K$ spectral species whose populations evolve according to a set of non-linear kinetic parameters $\boldsymbol{\theta}_{\text{nl}}$ (lifetimes $\tau_k$, instrument response function width $w$, time-zero $t_0$):

.. math::

   D(\lambda_i, t_j) \approx \sum_{k=1}^{K} c_k(t_j; \boldsymbol{\theta}_{\text{nl}}) \cdot S_k(\lambda_i)

In matrix notation:

.. math::

   D = C(\boldsymbol{\theta}_{\text{nl}}) \cdot S^T

Where:
* $D \in \mathbb{R}^{N_\lambda \times N_t}$ is the experimental data matrix.
* $C(\boldsymbol{\theta}_{\text{nl}}) \in \mathbb{R}^{N_t \times K}$ is the concentration matrix describing population dynamics convolved with the experimental Instrument Response Function (IRF).
* $S \in \mathbb{R}^{N_\lambda \times K}$ is the matrix of species-associated or decay-associated spectra.

Why Standard Non-Linear Optimization Fails
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If one attempts to fit both $\boldsymbol{\theta}_{\text{nl}}$ and $S$ simultaneously using a generic optimizer (e.g., Nelder-Mead or standard Levenberg-Marquardt), the optimization space contains:

.. math::

   N_{\text{total}} = N_{\text{nl}} + (K \times N_\lambda)

For a typical experiment with $K = 4$ components and $N_\lambda = 500$ wavelengths, the optimizer must search a **2000-dimensional** parameter space. This causes severe slow-down, frequent convergence into local non-physical minima, and extreme sensitivity to starting values.

The VarPro Solution
^^^^^^^^^^^^^^^^^^^
The **Variable Projection** algorithm (Golub & Pereyra, 1973) exploits the fact that for any fixed guess of the non-linear parameters $\boldsymbol{\theta}_{\text{nl}}$, the concentration matrix $C$ is known. Therefore, the optimal linear spectra $S^T$ can be computed **analytically in closed form** via linear least squares:

.. math::

   S^T = C^+ \cdot D^T = (C^T C)^{-1} C^T D^T

Substituting this expression back into the objective function yields the *projected* residual:

.. math::

   r(\boldsymbol{\theta}_{\text{nl}}) = \left( I - C C^+ \right) D^T

The non-linear optimizer now searches **only** over $\boldsymbol{\theta}_{\text{nl}}$ (typically 4–8 parameters instead of 2000). This reduces the parameter space dimensionality by over 99%, making fits orders of magnitude faster and virtually immune to linear local minima.
