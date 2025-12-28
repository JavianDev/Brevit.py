"""
Brevit.py - A high-performance Python library for semantically compressing
and optimizing data before sending it to a Large Language Model (LLM).
"""

from .brevit import (
    BrevitClient,
    BrevitConfig,
    JsonOptimizationMode,
    TextOptimizationMode,
    ImageOptimizationMode,
    ITextOptimizer,
    IImageOptimizer,
    DefaultTextOptimizer,
    DefaultImageOptimizer,
)

__version__ = "0.1.0"
__all__ = [
    "BrevitClient",
    "BrevitConfig",
    "JsonOptimizationMode",
    "TextOptimizationMode",
    "ImageOptimizationMode",
    "ITextOptimizer",
    "IImageOptimizer",
    "DefaultTextOptimizer",
    "DefaultImageOptimizer",
]

