Models API (`spark.core.models`)
=================================

.. currentmodule:: spark.core.models

The models module defines the kinetic model classes that generate the concentration matrix $C(t)$ for VarPro fitting.

Classes
-------

.. py:class:: BaseModel(num_components, use_artifact=False, artifact_mode='both')

   Abstract base class for all kinetic models.

   :param int num_components: Number of kinetic species/components.
   :param bool use_artifact: Whether to append coherent artifact basis functions.
   :param str artifact_mode: Artifact mode (``'both'``, ``'raman'``, or ``'xpm'``).

   .. py:method:: get_concentration_matrix(x_nl, t)

      Abstract method returning $C(t) \in \mathbb{R}^{N_t \times K}$.

.. py:class:: DASModel(num_components, use_artifact=False, artifact_mode='both')

   Parallel decay model (Decay-Associated Spectra).

   Parameter layout in ``x_nl``: ``[w, t0, tau_1, ..., tau_N]``.

.. py:class:: SASModel(num_components, use_artifact=False, artifact_mode='both')

   Sequential cascading model (Species-Associated Spectra) evaluated via Bateman equations.

   Parameter layout in ``x_nl``: ``[w, t0, tau_1, ..., tau_N]``.

.. py:class:: StretchedModel(num_components, use_artifact=False, artifact_mode='both')

   Stretched exponential model (Kohlrausch–Williams–Watts / KWW) evaluated via 32-node Gauss-Legendre quadrature.

   Parameter layout in ``x_nl``: ``[w, t0, tau_1, ..., tau_N, beta_1, ..., beta_N]``.

.. py:class:: OscillationModel(num_components, use_artifact=False, artifact_mode='both')

   Coupled exponential decay and damped sinusoidal wavepacket model.

   Parameter layout in ``x_nl``: ``[w, t0, tau_1, ..., tau_N, alpha, omega, phi]``.

.. py:class:: KMatrixModel(name="Custom Model")

   User-defined kinetic reaction network model compiled from the Visual Model Builder or Equation Editor.

   .. py:method:: add_state(state_name)

      Registers a new named physical state.

      :param str state_name: State identifier.

   .. py:method:: add_transition(source, target, param_type="tau", label="")

      Adds a directed transition between two states.

      :param str source: Source state name.
      :param str target: Target state name (or ``"S0"`` for ground-state relaxation).
      :param str param_type: Parameter type (``"tau"`` for lifetime, ``"gamma"`` for branching yield).
      :param str label: Parameter display label.

   .. py:method:: build_parameter_list()

      Extracts the unique parameter labels from all registered transitions.

      :return: List of parameter label strings.
      :rtype: list of str

   .. py:method:: get_default_guesses_and_bounds()

      Generates heuristic initial guesses, lower bounds, and upper bounds for all registered parameters.

      :return: Tuple ``(ini, low, upp)``.
      :rtype: tuple of numpy.ndarray

   .. py:method:: get_concentration_matrix(x_nl_params, t, w, t0, use_art=False, artifact_mode='both')

      Constructs the K-matrix, calculates eigenvalues/eigenvectors, and produces the convolved concentration matrix.

      :param numpy.ndarray x_nl_params: Parameter values.
      :param numpy.ndarray t: Delay time axis.
      :param float w: IRF FWHM.
      :param float t0: Time-zero offset.
      :return: Concentration matrix $(N_t \times K)$.
      :rtype: numpy.ndarray
