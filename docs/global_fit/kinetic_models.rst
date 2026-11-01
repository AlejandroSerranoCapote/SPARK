Built-in Kinetic Models
=======================

SPARK includes a comprehensive suite of mathematical kinetic models implemented in ``spark/core/models.py`` and ``spark/core/fitting/fit.py``. Each model constructs the concentration matrix $C(t; \boldsymbol{\theta}_{\text{nl}})$ that feeds into the Variable Projection engine.

.. contents:: Table of Contents
   :local:
   :depth: 2

Gaussian Instrument Response Function (IRF) Convolution
-------------------------------------------------------

Ultrafast laser pulses have finite temporal duration. SPARK models the Instrument Response Function as a normalized Gaussian centered at time-zero $t_0$ with Full Width at Half Maximum (FWHM) $w$:

.. math::

   G(t - t_0; w) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left( -\frac{(t - t_0)^2}{2\sigma^2} \right), \quad \text{where } \sigma = \frac{w}{2\sqrt{2\ln 2}}

Analytical Convolution & Numerical Stability
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The convolution of a Gaussian IRF with an exponential decay $\theta(t) \exp(-k t)$ yields an analytical solution involving the complementary error function $\operatorname{erfc}$:

.. math::

   c(t) = \frac{1}{2} \exp\left( -k (t - t_0) + \frac{k^2 \sigma^2}{2} \right) \operatorname{erfc}\left( \frac{k\sigma^2 - (t - t_0)}{\sqrt{2}\sigma} \right)

.. note::
   **Numerical Overflow Prevention**: When $(t - t_0) \ll 0$, the exponential term $\exp(k^2\sigma^2/2 - k(t-t_0))$ grows exponentially while $\operatorname{erfc}$ approaches zero, causing standard floating-point algorithms to encounter ``NaN`` or ``Inf`` overflow.
   SPARK solves this by partitioning the calculation:
   * For early times ($x \ge 0$), it employs the scaled complementary error function `scipy.special.erfcx(x)`:

     .. math::

        \operatorname{erfcx}(x) = \exp(x^2) \operatorname{erfc}(x)

   * This guarantees machine-precision stability across all time scales from femtoseconds to milliseconds.

1. Parallel Model (Decay-Associated Spectra / DAS)
-------------------------------------------------

Selected via ``Parallel`` in the Model dropdown.
Assumes all excited species are populated at $t = t_0$ and decay independently to the ground state in parallel:

.. math::

   D(\lambda, t) = \sum_{k=1}^{K} A_k(\lambda) \cdot \left[ G(t) \ast \exp(-t / \tau_k) \right]

* **Parameters**: $w$, $t_0$, and lifetimes $\tau_1, \tau_2, \dots, \tau_K$.
* **Interpretation**:
  - $A_k(\lambda) > 0$: The signal at $\lambda$ decays with lifetime $\tau_k$.
  - $A_k(\lambda) < 0$: The signal at $\lambda$ rises (grows in) with lifetime $\tau_k$.
* Ideal for initial survey analysis where reaction mechanisms are not yet established.

2. Sequential Model (Species-Associated Spectra / SAS)
---------------------------------------------------

Selected via ``Sequential`` in the Model dropdown.
Models a unidirectional cascading reaction:

.. math::

   1 \xrightarrow{\tau_1} 2 \xrightarrow{\tau_2} 3 \xrightarrow{\tau_3} \dots \xrightarrow{\tau_K} \text{Ground State}

* **Concentration Dynamics**: Evaluated using the generalized analytical **Bateman equations** convolved with the Gaussian IRF.
* **Degeneracy Protection**: If two lifetimes approach identical values ($|\tau_i - \tau_j| < 10^{-12}$), SPARK introduces an infinitesimal perturbation to prevent division by zero while preserving physical continuity.
* **Interpretation**: The resulting spectra $S_k(\lambda)$ represent the true, unmixed absorption/emission spectra of each intermediate species in the cascade.

3. Stretched Exponential Model (KWW)
------------------------------------

Selected via ``Stretched Exponential`` in the Model dropdown.
Widely used for heterogeneous environments, disordered semiconductors, quantum dot films, polymers, and solar cells exhibiting distributed trap states:

.. math::

   I(t) \propto \exp\left( -\left(\frac{t}{\tau}\right)^\beta \right), \quad 0 < \beta \le 1

* **Parameters**: $w$, $t_0$, lifetimes $\tau_1 \dots \tau_K$, and dispersion stretching factors $\beta_1 \dots \beta_K$.
* **Quadrature Algorithm**: Because convolved stretched exponentials possess no closed-form analytical expression, SPARK evaluates the convolution integral numerically using **32-node Gauss-Legendre quadrature**:

  .. math::

     C(t) = \int_{a}^{b} G(t - u) \exp\left( -(u/\tau)^\beta \right) du

* **Singularity Removal**: When $\beta < 1$, the derivative of the stretched exponential has a branch singularity at $u = 0$. SPARK applies a quadratic coordinate transformation:

  .. math::

     u = a + \Delta \cdot \xi^2, \quad du = 2\Delta \xi \, d\xi

  The term $\xi$ cancels the singular behavior, yielding stable, smooth gradients for the TRF optimizer.
* **Mean Lifetime Metric**: In addition to $\tau$, SPARK computes the physical mean lifetime:

  .. math::

     \langle \tau \rangle = \frac{\tau}{\beta} \, \Gamma\left(\frac{1}{\beta}\right)

4. Damped Oscillation Model
---------------------------

Selected via ``Damped Oscillation`` in the Model dropdown.
Designed for studying coherent phonon wavepackets, vibrational coherences, or polaron formation coupled with electronic population decay:

.. math::

   D(\lambda, t) = \sum_{k=1}^{K} A_k(\lambda) c_k(t) + B(\lambda) \cdot S_{\text{osc}}(t)

Where the oscillatory kinetic component is defined as:

.. math::

   S_{\text{osc}}(t) = \frac{1}{2} \left[ 1 + \operatorname{erf}\left( \frac{t - t_0}{\sqrt{2}w} \right) \right] \exp(-\alpha (t - t_0)) \sin(\omega (t - t_0) + \phi)

* **Parameters**: $\alpha$ (damping rate / dephasing), $\omega$ (oscillation frequency in rad/ps), and $\phi$ (initial phase).
* Yields the electronic DAS along with the vibrational amplitude spectrum $B(\lambda)$.

5. Independent $t_0(\lambda)$ Dispersion Fitting
------------------------------------------------

Checked via ``Independent t₀ per Wavelength`` (`chk_indep_t0`):
* Instead of assuming a single global time-zero for all wavelengths, SPARK assigns an independent parameter $t_0(\lambda_i)$ to each spectral channel.
* Particularly valuable when data exhibits residual chirp that cannot be parameterized by a simple 4th-degree polynomial.
* The VarPro engine dynamically partitions the linear solve into per-wavelength matrix projections.

6. Coherent Artifact Basis Decomposition
----------------------------------------

Checked via ``Fit Coherent Artifact`` (`chk_artifact`):
Near $t = 0$, intense overlap between pump and probe pulses in the sample generates solvent Raman scattering, two-photon absorption, and cross-phase modulation (XPM). SPARK appends an analytical orthogonal basis to the concentration matrix:

* **$\phi_0(t)$ (Raman / 2PA)**: Direct normalized Gaussian IRF $G(t - t_0)$.
* **$\phi_1(t)$ (XPM Derivative)**: First derivative $G'(t - t_0)$, capturing symmetric spectral broadening shifts.
* **$\phi_2(t)$ (Dispersive XPM)**: Second derivative $G''(t - t_0)$, capturing asymmetric spectral lensing.

Users can choose between:
* ``Raman + XPM`` (all 3 basis functions)
* ``Raman Only`` ($\phi_0$ only)
* ``XPM Only`` ($\phi_1$ and $\phi_2$)

7. Non-Negative Least Squares (NNLS) Constraint
-----------------------------------------------

Checked via ``Enforce Non-Negative Spectra`` (`chk_nnls`):
* In standard VarPro, the linear solve uses unconstrained least-squares ($\min \|C S^T - D\|_2$), which can occasionally produce non-physical negative spectral lobes.
* When NNLS is enabled, SPARK solves the linear system using Bounded-Variable Least Squares (`scipy.optimize.lsq_linear`), enforcing:

  .. math::

     S_k(\lambda) \ge 0 \quad \forall \lambda

* Coherent artifact columns remain unconstrained ($-\infty < S_{\text{art}} < \infty$) because XPM signals inherently possess positive and negative derivative oscillations.
