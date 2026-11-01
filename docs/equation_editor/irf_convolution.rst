Gaussian IRF & Convolution Mechanics
======================================

A fundamental requirement in ultrafast spectroscopy is accounting for the finite temporal duration of the excitation laser pulse. This chapter explains how SPARK formulates and integrates the Instrument Response Function (**IRF**) within custom kinetic models.

.. contents:: Table of Contents
   :local:
   :depth: 2

Mathematical Formulation of $G(t)$
----------------------------------

In experimental measurements, the excitation pulse envelope is modeled as a normalized Gaussian function centered at time-zero $t_0$ with Full Width at Half Maximum (FWHM) $w$.

The Gaussian function is parameterized as:

.. math::

   G(t) = \sqrt{\frac{4 \ln 2}{\pi \, w^2}} \, \exp\left( -\frac{4 \ln 2 \, (t - t_0)^2}{w^2} \right)

Equivalently, expressed in terms of the Gaussian standard deviation $\sigma = \frac{w}{2\sqrt{2\ln 2}}$:

.. math::

   G(t) = \frac{1}{\sqrt{2\pi}\,\sigma} \, \exp\left( -\frac{(t - t_0)^2}{2\sigma^2} \right)

Proof of Normalization to Unity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A physical excitation pulse shape must be normalized such that its total integrated area across all time is strictly equal to $1.0$:

.. math::

   \int_{-\infty}^{+\infty} G(t) \, dt = \sqrt{\frac{4\ln 2}{\pi w^2}} \int_{-\infty}^{+\infty} \exp\left( -\frac{4\ln 2 \, u^2}{w^2} \right) du = 1.0

This normalization ensures that when integrating the kinetic generation term:

.. math::

   \frac{d[S_1]}{dt} = G(t) - k [S_1]

the total number of excited states generated across the entire pulse envelope is exactly equal to $1$ (in normalized population units).

Scale Factors & Pump Intensity: $a \cdot G(t)$
----------------------------------------------

In certain photochemical models (such as excitation-density dependent annihilation or multiphoton processes), the initial excitation population is proportional to the incident pump laser fluence.

When writing rate equations:

.. math::

   \frac{dA}{dt} = a \cdot G(t) - k_1 A - \gamma_{\text{ann}} A^2

* $G(t)$ supplies the normalized temporal pulse profile ($\int G(t) dt = 1$).
* The prefactor $a$ serves as a **dimensionless scaling factor** representing the initial excitation density (photons absorbed per unit volume or fractional ground-state depletion).
* In linear models, the absolute scaling factor $a$ is absorbed directly into the linear spectral amplitude matrix $S(\lambda)$ by the Variable Projection theorem, so explicitly fitting $a$ is unnecessary.
* In non-linear models (such as bimolecular recombination or exciton annihilation), $a$ determines the competition between first-order decay ($k_1 A$) and second-order annihilation ($\gamma A^2$), making $a$ a critical non-linear parameter.

Analytical Convolution vs Numerical Integration
-----------------------------------------------

SPARK handles IRF convolution differently depending on whether the kinetic system is linear or non-linear:

Linear Systems (K-Matrix Mode)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If the system of differential equations is strictly linear (i.e. $\frac{d\mathbf{c}}{dt} = K \mathbf{c}$):
1. SPARK diagonalizes the rate matrix: $K = V \Lambda V^{-1}$.
2. The population of each physical state is expressed as a linear combination of pure exponential decay modes:

   .. math::

      c_i(t) = \sum_m V_{im} \exp(\lambda_m t)

3. Each mode is convolved **analytically** with $G(t)$ using the complementary error function ($\operatorname{erfcx}$) formulation:

   .. math::

      \widetilde{c}_i(t) = \sum_m V_{im} \left[ G(t) \ast \exp(\lambda_m t) \right]

4. **Result**: Zero numerical integration error, instant computation (sub-millisecond), and perfect gradient evaluation for the optimizer.

Non-Linear Systems (Numerical BDF Mode)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If the rate equations contain state-state products (e.g., $A \cdot B$, $A^2$):
1. Analytical diagonalization is impossible.
2. The normalized Gaussian pulse $G(t)$ is inserted directly as an explicit source term into the right-hand side of the ODE system:

   .. math::

      \frac{dA}{dt} = G(t) - k_1 A - k_{\text{ann}} A^2

3. The system is integrated numerically from $t_{\min}$ to $t_{\max}$ using SciPy's stiff **Backward Differentiation Formula (BDF)** algorithm with adaptive time stepping.
4. The adaptive solver places dense evaluation points across the Gaussian peak (where $G(t)$ varies rapidly), automatically widening time steps at long delay times where only smooth exponential decays remain.
