"""
    fishplotpy: A Python implementation for visualizing clonal evolution dynamics.

    This package provides tools to create "fish plots" (also known as Muller plots)
    that visualize the temporal changes in the frequencies of clones within a
    population, often used in cancer genomics.
"""

__version__ = "1.0"

from .data import FishPlotData
from .plot import fishplot, draw_legend

__all__ = [
    "FishPlotData",
    "fishplot",
    "draw_legend",
    "__version__"
]

print(f"fishplotpy=={__version__}")
