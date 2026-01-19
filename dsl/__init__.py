"""
DSL (Double-Supervised Learning) Framework for Python

This is a Python implementation of the DSL framework, which is a method for estimating
regression models with partially labeled data. The framework combines supervised machine
learning with econometric methods to handle situations where only a subset of observations
have labels.
"""

__version__ = "0.2.0"

from .dsl import (
    DSLResult,
    PowerDSLResult,
    dsl,
    plot_power,
    power_dsl,
    summary,
    summary_power,
)

__all__ = [
    "__version__",
    "dsl",
    "power_dsl",
    "summary",
    "summary_power",
    "plot_power",
    "DSLResult",
    "PowerDSLResult",
]
