Custom Kinetic Models & Equation Editor
========================================

The **Visual Model Builder & Equation Editor** (implemented in ``spark/gui/model_builder.py``) empowers researchers to define custom reaction mechanisms that go beyond parallel (DAS) and sequential (SAS) schemes.

.. toctree::
   :maxdepth: 2

   visual_builder
   ode_syntax
   irf_convolution
   presets

Overview & Dual-Mode Architecture
---------------------------------

SPARK provides two synchronized interfaces for model design:

1. **Visual Drag-and-Drop Canvas (Grotrian Mode)**:
   - Construct kinetic states as graphical nodes.
   - Draw transition arrows connecting states with associated rate constants ($k$) or lifetimes ($\tau$).
   - Topology changes instantly generate corresponding differential equations in real time.

2. **Symbolic ODE Script Editor (Equation Mode)**:
   - Write reaction rate equations directly in clean, Pythonic syntax (e.g. ``dA/dt = -k1*A + k2*B``).
   - Render live mathematical LaTeX previews powered by SymPy.
   - Choose from pre-configured reaction templates (annihilation, bimolecular recombination, reversible equilibria).

Hybrid Compilation Pipeline
---------------------------

When you click **Compile and Apply**, SPARK executes an intelligent **hybrid compiler**:

.. code-block:: text

   ┌────────────────────────────────────────────────────────┐
   │             User Input: Graph or ODE Script            │
   └───────────────────────────┬────────────────────────────┘
                               │
                               ▼
   ┌────────────────────────────────────────────────────────┐
   │               SymPy Symbolic Expression Parser         │
   └───────────────────────────┬────────────────────────────┘
                               │
               Is the system strictly linear?
               (No state-state products like A*B)
                               │
             ┌─────────────────┴─────────────────┐
             │ YES                               │ NO
             ▼                                   ▼
   ┌───────────────────────────┐       ┌───────────────────────────┐
   │   K-Matrix Eigenvalue     │       │   Backward Differentiation│
   │      Decomposition        │       │      Formula (BDF)        │
   │  (Analytical / Ultra-Fast)│       │   (Numerical Integration) │
   └─────────────┬─────────────┘       └─────────────┬─────────────┘
                 │                                   │
                 └─────────────────┬─────────────────┘
                                   │
                                   ▼
   ┌────────────────────────────────────────────────────────┐
   │         Convolution with Gaussian IRF: G(t)            │
   └───────────────────────────┬────────────────────────────┘
                               │
                               ▼
   ┌────────────────────────────────────────────────────────┐
   │         VarPro Optimization & SAS Extraction           │
   └────────────────────────────────────────────────────────┘

* **Linear Systems**: Evaluated analytically via matrix diagonalization ($K = V \Lambda V^{-1}$), achieving execution speeds comparable to built-in DAS/SAS models.
* **Non-Linear Systems**: Evaluated using adaptive step-size stiff numerical integrators (SciPy's BDF algorithm).
