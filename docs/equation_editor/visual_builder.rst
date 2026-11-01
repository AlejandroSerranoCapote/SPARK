Visual Model Builder (Grotrian Mode)
=====================================

The **Visual Mode** of ``ModelBuilderDialog`` allows you to build photochemical and photophysical reaction schemes graphically, similar to drawing a Jablonski or Grotrian energy level diagram.

.. contents:: Table of Contents
   :local:
   :depth: 2

The Interactive Canvas (`KineticCanvas`)
----------------------------------------

The visual builder uses an interactive Qt graphics scene (`KineticCanvas`):

.. code-block:: text

   ┌────────────────────────────────────────────────────────────────────────┐
   │ [ Add State ]  [ Connect States ]  [ Erase Selected ]  [ Reboot Canvas ]│
   ├───────────────────────────────────────────────────────┬────────────────┤
   │                                                       │ Inspector      │
   │      ┌────────────┐                                   │ ────────────── │
   │      │    S1*     │ ◄── Initial State (Green border)  │ State: S1*     │
   │      └─────┬──────┘                                   │ [X] Initial    │
   │            │ k_IC (τ1)                                │ [ ] Dark       │
   │            ▼                                          │                │
   │      ┌────────────┐                                   │ Transition     │
   │      │    S1      │                                   │ ────────────── │
   │      └─────┬──────┘                                   │ Type: Lifetime │
   │      k_ISC │                                          │ Label: tau_1   │
   │            ▼                                          │                │
   │      ┌────────────┐                                   │ Live ODE:      │
   │      │    T1      │                                   │ dS1*/dt=-k1*S1*│
   │      └────────────┘                                   │                │
   │                                                       │ [Send to ODE]  │
   └───────────────────────────────────────────────────────┴────────────────┘

Creating and Editing States (`StateNode`)
-----------------------------------------

1. Click **Add State** on the toolbar. A new rectangular state node appears on the canvas.
2. Drag the node to position it on your energy level diagram.
3. In the **Properties Inspector** on the right, customize:
   - **Name**: Give the state an identifier (e.g., ``S1``, ``S2``, ``T1``, ``ChargeSeparated``).
   - **Initial State (`is_initial`)**: Check this box if the state is directly populated by the laser excitation pulse at $t = 0$. Initial states are highlighted with a distinct **green border**. Root states without incoming transitions are automatically treated as initial states.
   - **Dark State (`is_dark`)**: Check this box if the state is optically transparent or non-absorbing in the spectral probe window (e.g., the electronic ground state $S_0$ or an unobserved trap). Dark states are rendered in **gray** and are excluded from the linear spectral amplitude matrix, preventing rank inflation.

Drawing Transitions (`TransitionEdge`)
--------------------------------------

1. Click **Connect States** to toggle the arrow connection tool.
2. Click on the **Source State**, then click on the **Target State**.
3. A directed transition arrow appears connecting the two states.
4. In the Inspector, specify:
   - **Parameter Type**:
     * ``Lifetime (tau)``: Rates are parameterized as $k = 1/\tau$.
     * ``Rate (k)``: Direct rate constant.
     * ``Branching Ratio (gamma)``: Fractional yield $\gamma \in [0, 1]$.
   - **Label**: Custom parameter name that will appear in fit summaries and reports (e.g., ``tau_ET`` for electron transfer).
5. Ground State Decay: To represent relaxation back to the unexcited ground state, draw an arrow to a designated ground state node (e.g., ``S0`` or ``GS``) marked as **Dark**.

Live Equation Generation
------------------------

As you add states, delete elements, or re-route arrows, SPARK's graph topology analyzer dynamically writes the corresponding differential equations into the **Live Equations** box:

.. math::

   \frac{d[\text{S1}]}{dt} = -\left(k_{\text{IC}} + k_{\text{ISC}}\right) [\text{S1}], \quad \frac{d[\text{T1}]}{dt} = +k_{\text{ISC}} [\text{S1}] - k_{\text{phos}} [\text{T1}]

Click **Send to Equation Editor** to transfer the auto-generated model into Tab 2 for symbolic inspection or customization.
