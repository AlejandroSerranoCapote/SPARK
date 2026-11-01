"""
Data management and I/O module for SPARK.
"""

from .dataset import SpectroscopyDataset
from .importer import load_data, load_universal_file  

__all__ = ["SpectroscopyDataset", "load_data", "load_universal_file"]  