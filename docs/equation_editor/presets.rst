Built-in Model Presets & Templates
===================================

The Equation Editor includes six pre-configured reaction templates accessible from the **Presets** dropdown. These templates illustrate how to formulate diverse photophysical and photochemical processes.

.. contents:: Table of Contents
   :local:
   :depth: 2

Preset 1: Exciton-Exciton Annihilation (EEA)
--------------------------------------------

* **Physical Context**: High-fluence excitation of conjugated polymers, perovskite thin films, 2D transition metal dichalcogenides (TMDs), or molecular aggregates where high exciton densities lead to mutual collision and annihilation ($S_1 + S_1 \to S_n + S_0 \to S_1 + S_0$).
* **Equations**:

  .. math::

     \frac{d[N]}{dt} = -k_1 [N] - \gamma_{\text{ann}} [N]^2

* **Script**:

  .. code-block:: text

     Initial: N
     Dark:
     dN/dt = -k1*N - gamma*N**2

* **Parameters**:
  - ``k1``: Monomolecular decay rate ($1/\tau_{\text{spontaneous}}$).
  - ``gamma``: Bimolecular annihilation rate coefficient.

Preset 2: Sequential Cascade ($A \to B \to C \to \text{GS}$)
------------------------------------------------------------

* **Physical Context**: Cascading energy transfer, vibrational cooling on the excited state surface followed by internal conversion and ground state recovery.
* **Equations**:

  .. math::

     \frac{dA}{dt} = -k_1 A, \quad \frac{dB}{dt} = k_1 A - k_2 B, \quad \frac{dC}{dt} = k_2 B - k_3 C

* **Script**:

  .. code-block:: text

     Initial: A
     Dark: GS
     dA/dt = -k1*A
     dB/dt =  k1*A - k2*B
     dC/dt =  k2*B - k3*C

* **Parameters**: Rates $k_1, k_2, k_3$ or lifetimes $\tau_1, \tau_2, \tau_3$.

Preset 3: Triplet-Triplet Annihilation (TTA) & Upconversion
-----------------------------------------------------------

* **Physical Context**: Photochemical upconversion or delayed fluorescence in organic light-emitting diodes (OLEDs) and sensitizer-emitter pairs ($T_1 + T_1 \to S_1 + S_0$).
* **Equations**:

  .. math::

     \frac{d[T]}{dt} = -k_T [T] - 2 \gamma_{\text{TTA}} [T]^2, \quad \frac{d[S]}{dt} = \gamma_{\text{TTA}} [T]^2 - k_S [S]

* **Script**:

  .. code-block:: text

     Initial: T
     Dark:
     dT/dt = -kT*T - 2*gamma_tta*T**2
     dS/dt = gamma_tta*T**2 - kS*S

Preset 4: Bimolecular Free Carrier Recombination
------------------------------------------------

* **Physical Context**: Electron-hole recombination in photovoltaic absorbers (e.g. lead halide perovskites, organic bulk heterojunctions) with competing trap-assisted (Shockley-Read-Hall) recombination.
* **Equations**:

  .. math::

     \frac{dn}{dt} = -k_{\text{trap}} n - \beta_{\text{rad}} n^2 - \gamma_{\text{Auger}} n^3

* **Script**:

  .. code-block:: text

     Initial: n
     Dark:
     dn/dt = -k_trap*n - beta_rad*n**2 - gamma_auger*n**3

Preset 5: Reversible Equilibrium ($A \rightleftharpoons B$)
-----------------------------------------------------------

* **Physical Context**: Excited-state proton transfer (ESPT), conformational isomerization, or exciplex formation where forward and reverse reactions occur on comparable timescales.
* **Equations**:

  .. math::

     \frac{dA}{dt} = -k_{\text{forward}} A + k_{\text{reverse}} B - k_{\text{decay}, A} A

     \frac{dB}{dt} =  k_{\text{forward}} A - k_{\text{reverse}} B - k_{\text{decay}, B} B

* **Script**:

  .. code-block:: text

     Initial: A
     Dark:
     dA/dt = -k_fwd*A + k_rev*B - k_dA*A
     dB/dt =  k_fwd*A - k_rev*B - k_dB*B

* **Parameters**: Forward rate $k_{\text{fwd}}$, reverse rate $k_{\text{rev}}$, and uncoupled decay rates $k_{dA}, k_{dB}$.

Preset 6: Exciton-Polaron Model
-------------------------------

* **Physical Context**: Organic photovoltaics where neutral singlet excitons undergo charge separation at donor-acceptor interfaces into polaron pairs, followed by recombination or dissociation.
* **Equations**:

  .. math::

     \frac{d[X]}{dt} = -(k_{\text{diss}} + k_{\text{rad}}) [X], \quad \frac{d[P]}{dt} = k_{\text{diss}} [X] - k_{\text{rec}} [P]

* **Script**:

  .. code-block:: text

     Initial: X
     Dark:
     dX/dt = -(k_diss + k_rad)*X
     dP/dt =  k_diss*X - k_rec*P
