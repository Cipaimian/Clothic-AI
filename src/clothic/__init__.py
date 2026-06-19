"""Clothic AI - Clothing Vision.

An explainable, real-time campus outfit-compliance system built as a
*Visual Attribute Recognition + Policy Reasoning Engine*.

The neural perception layer only detects observable visual attributes
(garment type, sleeve length, hemline vs. knee, skin-exposure ratios, ...).
A transparent, JSON-configurable rule engine maps those attributes to a
compliance decision with a written, citable explanation. The models never
learn the value judgement of "modesty" -- policy does.
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
