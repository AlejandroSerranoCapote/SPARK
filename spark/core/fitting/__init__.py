"""
Fitting and non-linear optimization submodule.
"""

from . import fit as _fit_module
from .fit import *
from .global_fitter import GlobalFitter
from .optimizer import fit, FitWorker, FitResult

# Forward all kinetic basis and concentration matrix functions to fit
# so fit is BOTH callable as fit(...) AND exposes all fit.py algorithms
for _attr_name, _attr_val in _fit_module.__dict__.items():
    if not _attr_name.startswith('_'):
        setattr(fit, _attr_name, _attr_val)

__all__ = ["GlobalFitter", "fit", "FitWorker", "FitResult"]
