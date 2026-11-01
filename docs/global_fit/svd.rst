SVD Analysis & Rank Estimation
==============================

Before selecting a kinetic model, one must estimate how many distinguishable physical states or transition pathways exist in the data. SPARK integrates Singular Value Decomposition (**SVD**) to perform objective, model-free rank estimation.

.. contents:: Table of Contents
   :local:
   :depth: 2

Mathematical Background
-----------------------

Singular Value Decomposition factors the 2D experimental data matrix $D \in \mathbb{R}^{N_\lambda \times N_t}$ into the product of three matrices:

.. math::

   D = U \, \Sigma \, V^T = \sum_{k=1}^{R} \sigma_k \, \mathbf{u}_k \, \mathbf{v}_k^T

Where:
* $\Sigma = \operatorname{diag}(\sigma_1, \sigma_2, \dots, \sigma_R)$ is a diagonal matrix containing the **singular values** sorted in descending order ($\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_R \ge 0$). Each singular value is proportional to the variance captured by that component.
* $U = [\mathbf{u}_1, \mathbf{u}_2, \dots]$ contains the orthonormal **spectral basis vectors** (wavelength-dependent shapes).
* $V = [\mathbf{v}_1, \mathbf{v}_2, \dots]$ contains the orthonormal **temporal basis vectors** (kinetics evolution).
* $R = \min(N_\lambda, N_t)$ is the maximum theoretical rank.

Running SVD in SPARK
--------------------

In the **Fit** tab of `GlobalFitPanel`, click **Run SVD**:

1. SPARK computes the economy SVD on the pre-processed data matrix.
2. The right display switches to the dedicated **SVD** tab containing a 3-panel visualization:
   - **Top Panel: Scree Plot**: Shows singular values $\sigma_k$ plotted against component index $k$ (on both linear and logarithmic scales).
   - **Bottom-Left Panel: Spectral Components**: Plots the first 5 columns of $U$ across wavelength.
   - **Bottom-Right Panel: Temporal Components**: Plots the first 5 columns of $V$ across delay time.
3. The **SVD Info Box** prints the numeric singular values, their relative percentage of total variance, and the cumulative variance:

   .. code-block:: text

      SVD Analysis Results:
      ----------------------------------------
      Comp 1:  σ = 142.31  (78.4% variance)  [Cumulative:  78.4%]
      Comp 2:  σ =  34.12  (18.8% variance)  [Cumulative:  97.2%]
      Comp 3:  σ =   4.25  ( 2.3% variance)  [Cumulative:  99.5%]
      Comp 4:  σ =   0.58  ( 0.3% variance)  [Cumulative:  99.8%]
      Comp 5:  σ =   0.21  ( 0.1% variance)  [Cumulative:  99.9%]
      ----------------------------------------
      Recommended physical components: 2 or 3

Interpreting the SVD Output
---------------------------

The "Elbow" in the Scree Plot
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Physical spectroscopic processes produce singular values that stand significantly above the noise floor. A sharp downward bend or "elbow" in the scree curve separates real physical dynamics from random experimental noise. In the example above, components 1 and 2 capture 97.2% of the variance, while component 4 drops into the noise floor.

Inspecting Basis Vectors
^^^^^^^^^^^^^^^^^^^^^^^^
* **Signal Components**: Show smooth, continuous spectral features in $\mathbf{u}_k$ and monotonic or multi-exponential decays in $\mathbf{v}_k$.
* **Noise Components**: Show high-frequency, stochastic fluctuations across both wavelength and delay time.
* If a component displays structured derivative-like oscillations near $t_0$, this indicates an optical **coherent artifact** (solvent cross-phase modulation) rather than an electronic state decay.
