"""
Computational backend for ultrafast spectroscopy analysis.
"""

from .chirp import fit_t0, apply_t0_correction_nonlinear, apply_t0_correction_poly, t0_model
from .models import *
from .fitting import GlobalFitter
from .fitting.optimizer import fit, FitWorker, FitResult
from .diagnostics import *  # <--- Añade esto para importar todo lo de diagnostics

__all__ = [
    "fit_t0",
    "apply_t0_correction_nonlinear",
    "apply_t0_correction_poly",
    "t0_model",
    "GlobalFitter",
    "fit",
]
