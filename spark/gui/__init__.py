"""
Graphical User Interface module based on PyQt5.
"""

from .flups import FLUPSAnalyzer
from .tas import TASAnalyzer
from .global_fit_class_gui import GlobalFitPanel
from .maps_from_timescans import AppWindow

__all__ = ["FLUPSAnalyzer", "TASAnalyzer", "GlobalFitPanel", "AppWindow"]