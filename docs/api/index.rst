API & Developer Reference
=========================

This section provides complete technical documentation for SPARK's core modules, classes, algorithms, and public functions.

.. toctree::
   :maxdepth: 2

   dataset
   importer
   global_fitter
   optimizer
   models
   chirp
   diagnostics

Architecture Overview
---------------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Module Path
     - Description
   * - ``spark.data.dataset``
     - In-memory 2D matrix container (``SpectroscopyDataset``) with slicing, binning, and units.
   * - ``spark.data.importer``
     - Universal format loaders, spreadsheet mapping GUI (``DataImporterDialog``), pandas dispatchers.
   * - ``spark.core.fitting.global_fitter``
     - Mathematical manager for VarPro matrix construction, Jacobian SVD, condition metrics, and profile likelihood.
   * - ``spark.core.fitting.optimizer``
     - Non-linear least squares engine (TRF, soft-L1 loss, order constraints) and background ``QThread`` worker.
   * - ``spark.core.models``
     - Kinetic models: Parallel DAS, Sequential SAS, Stretched Exponential, Damped Oscillation, and Custom K-Matrix.
   * - ``spark.core.chirp``
     - Group velocity dispersion fitting (polynomial, nonlinear Sellmeier model, auto-detection).
   * - ``spark.core.diagnostics``
     - Console reporting and Matplotlib diagnostic plotting utilities.
