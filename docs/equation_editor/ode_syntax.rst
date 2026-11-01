ODE Scripting & Syntax Guide
==============================

The **Equation Editor** tab of ``ModelBuilderDialog`` allows you to specify custom kinetic differential equations directly using standard mathematical notation.

.. contents:: Table of Contents
   :local:
   :depth: 2

Script Syntax Rules
-------------------

The ODE script format is clean, transparent, and Pythonic. Each line defines the time derivative of one physical state:

.. code-block:: text

   d<StateName>/dt = <algebraic rate expression>

Example Script:
^^^^^^^^^^^^^^^
.. code-block:: text

   dA/dt = -k1*A + k_back*B
   dB/dt =  k1*A - k_back*B - k2*B
   dC/dt =  k2*B

Syntax Rules & Conventions:
* **Time Derivatives**: The left-hand side must always follow the form ``d<State>/dt =``.
* **State Names**: Use concise, alphanumeric names (e.g., ``A``, ``B``, ``S1``, ``P_plus``). Avoid punctuation.
* **Parameters**: Any identifier that is not a state name or reserved constant is treated as an optimizable kinetic parameter (e.g., ``k1``, ``k_et``, ``tau_isc``, ``gamma``).
* **Mathematical Operators**: Standard Python/SymPy operators are supported:
  - Addition (``+``) and Subtraction (``-``).
  - Multiplication (``*``) and Division (``/``).
  - Powers (``**``).
* **Comments**: Prefix comments with ``#`` (ignored during parsing).

Configuring Initial & Dark States
---------------------------------

Above the equation script area are two dedicated configuration lines:

1. **Initial States (`txt_initial_states`)**:
   - Comma-separated list of states that possess population at $t = 0$.
   - Example: ``S1`` (single excitation) or ``A, B`` (mixed excitation).
   - If left empty, SPARK automatically detects root states (states that do not appear on the production side of any differential equation).

2. **Dark States (`txt_dark_states`)**:
   - Comma-separated list of states that do not contribute to transient absorption or emission spectra.
   - Example: ``S0, GroundState, SolventTrap``.
   - Dark state concentrations are tracked during kinetic evolution but excluded from the $S(\lambda)$ spectral amplitude matrix.

Live LaTeX Preview & Syntax Validation
--------------------------------------

Click the **Preview LaTeX** button:
1. SPARK sends the equation script to **SymPy** for symbolic tokenization and semantic analysis.
2. If syntax errors exist (such as unclosed parentheses, undefined variables, or invalid symbols), an informative error dialog highlights the exact line and issue.
3. If syntax is valid, SymPy generates standard LaTeX expressions and renders them onto the embedded Matplotlib canvas:

   .. math::

      \frac{d[A]}{dt} = -k_1 [A] + k_{\text{back}} [B]

      \frac{d[B]}{dt} = k_1 [A] - (k_{\text{back}} + k_2) [B]

This visual confirmation guarantees that the compiler has interpreted your kinetic model exactly as intended.
