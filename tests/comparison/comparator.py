#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Python-R comparison framework with configurable tolerances.
Provides automated comparison of DSL results between implementations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ComparisonConfig:
    """Configuration for comparison tolerances."""

    # Absolute tolerances
    coef_abs_tol: float = 0.01  # Coefficients absolute tolerance
    se_abs_tol: float = 0.01  # Standard errors absolute tolerance
    pvalue_abs_tol: float = 0.05  # P-values absolute tolerance

    # Relative tolerances (as fractions)
    coef_rel_tol: float = 0.05  # 5% relative tolerance for coefficients
    se_rel_tol: float = 0.10  # 10% relative tolerance for SEs
    pvalue_rel_tol: float = 0.20  # 20% relative tolerance for p-values

    # Use relative tolerance when values exceed these thresholds
    coef_rel_threshold: float = 0.1
    se_rel_threshold: float = 0.1
    pvalue_rel_threshold: float = 0.01


@dataclass
class ComparisonMetric:
    """Single metric comparison result."""

    name: str
    variable: str
    python_value: float
    r_value: float
    abs_diff: float
    rel_diff: float
    abs_tol: float
    rel_tol: float
    passed: bool
    message: str


@dataclass
class ComparisonResult:
    """Complete comparison result."""

    name: str
    config: ComparisonConfig
    coefficient_comparisons: List[ComparisonMetric] = field(default_factory=list)
    se_comparisons: List[ComparisonMetric] = field(default_factory=list)
    pvalue_comparisons: List[ComparisonMetric] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        """Check if all comparisons passed."""
        all_comps = (
            self.coefficient_comparisons + self.se_comparisons + self.pvalue_comparisons
        )
        return all(comp.passed for comp in all_comps)

    @property
    def n_comparisons(self) -> int:
        """Total number of comparisons."""
        return (
            len(self.coefficient_comparisons)
            + len(self.se_comparisons)
            + len(self.pvalue_comparisons)
        )

    @property
    def n_passed(self) -> int:
        """Number of passed comparisons."""
        all_comps = (
            self.coefficient_comparisons + self.se_comparisons + self.pvalue_comparisons
        )
        return sum(1 for comp in all_comps if comp.passed)

    @property
    def n_failed(self) -> int:
        """Number of failed comparisons."""
        return self.n_comparisons - self.n_passed

    def get_failures(self) -> List[ComparisonMetric]:
        """Get list of failed comparisons."""
        all_comps = (
            self.coefficient_comparisons + self.se_comparisons + self.pvalue_comparisons
        )
        return [comp for comp in all_comps if not comp.passed]

    def summary(self) -> str:
        """Generate summary report."""
        lines = [
            f"\n{'=' * 60}",
            f"Comparison Summary: {self.name}",
            f"{'=' * 60}",
            f"Total Comparisons: {self.n_comparisons}",
            f"Passed: {self.n_passed} ({100 * self.n_passed / self.n_comparisons:.1f}%)",
            f"Failed: {self.n_failed} ({100 * self.n_failed / self.n_comparisons:.1f}%)",
            f"",
            f"Configuration:",
            f"  Coefficient Tolerance: abs={self.config.coef_abs_tol}, rel={self.config.coef_rel_tol}",
            f"  SE Tolerance: abs={self.config.se_abs_tol}, rel={self.config.se_rel_tol}",
            f"  P-value Tolerance: abs={self.config.pvalue_abs_tol}, rel={self.config.pvalue_rel_tol}",
        ]

        if self.n_failed > 0:
            lines.append(f"\n{'=' * 60}")
            lines.append("FAILED COMPARISONS:")
            lines.append(f"{'=' * 60}")
            for comp in self.get_failures():
                lines.append(f"\n{comp.message}")

        return "\n".join(lines)


def compare_value(
    name: str,
    variable: str,
    py_val: float,
    r_val: float,
    abs_tol: float,
    rel_tol: float,
    rel_threshold: float,
) -> ComparisonMetric:
    """Compare a single value between Python and R implementations.

    Args:
        name: Metric name (e.g., "Coefficient", "Standard Error")
        variable: Variable name
        py_val: Python value
        r_val: R reference value
        abs_tol: Absolute tolerance
        rel_tol: Relative tolerance
        rel_threshold: Threshold for using relative tolerance

    Returns:
        ComparisonMetric object
    """
    abs_diff = abs(py_val - r_val)
    rel_diff = abs_diff / abs(r_val) if r_val != 0 else abs_diff

    # Determine which tolerance to use
    if abs(r_val) > rel_threshold:
        # Use relative tolerance for larger values
        passed = rel_diff <= rel_tol
        message = (
            f"{name} '{variable}': "
            f"Python={py_val:.6f}, R={r_val:.6f}, "
            f"RelDiff={rel_diff:.4f} (tol={rel_tol})"
        )
    else:
        # Use absolute tolerance for smaller values
        passed = abs_diff <= abs_tol
        message = (
            f"{name} '{variable}': "
            f"Python={py_val:.6f}, R={r_val:.6f}, "
            f"AbsDiff={abs_diff:.6f} (tol={abs_tol})"
        )

    if not passed:
        message = "❌ FAILED: " + message
    else:
        message = "✓ PASSED: " + message

    return ComparisonMetric(
        name=name,
        variable=variable,
        python_value=py_val,
        r_value=r_val,
        abs_diff=abs_diff,
        rel_diff=rel_diff,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        passed=passed,
        message=message,
    )


def compare_implementations(
    python_coefs: Dict[str, float],
    python_ses: Dict[str, float],
    python_pvalues: Dict[str, float],
    r_coefs: Dict[str, float],
    r_ses: Dict[str, float],
    r_pvalues: Dict[str, float],
    config: Optional[ComparisonConfig] = None,
    name: str = "Python-R Comparison",
) -> ComparisonResult:
    """Compare Python and R implementations with configurable tolerances.

    Args:
        python_coefs: Python coefficients dict {var_name: value}
        python_ses: Python standard errors dict {var_name: value}
        python_pvalues: Python p-values dict {var_name: value}
        r_coefs: R coefficients dict {var_name: value}
        r_ses: R standard errors dict {var_name: value}
        r_pvalues: R p-values dict {var_name: value}
        config: ComparisonConfig object (uses defaults if None)
        name: Name for this comparison

    Returns:
        ComparisonResult object with detailed comparison results
    """
    if config is None:
        config = ComparisonConfig()

    result = ComparisonResult(name=name, config=config)

    # Compare coefficients
    for var in python_coefs.keys():
        if var in r_coefs:
            comp = compare_value(
                name="Coefficient",
                variable=var,
                py_val=python_coefs[var],
                r_val=r_coefs[var],
                abs_tol=config.coef_abs_tol,
                rel_tol=config.coef_rel_tol,
                rel_threshold=config.coef_rel_threshold,
            )
            result.coefficient_comparisons.append(comp)

    # Compare standard errors
    for var in python_ses.keys():
        if var in r_ses:
            comp = compare_value(
                name="Standard Error",
                variable=var,
                py_val=python_ses[var],
                r_val=r_ses[var],
                abs_tol=config.se_abs_tol,
                rel_tol=config.se_rel_tol,
                rel_threshold=config.se_rel_threshold,
            )
            result.se_comparisons.append(comp)

    # Compare p-values
    for var in python_pvalues.keys():
        if var in r_pvalues:
            comp = compare_value(
                name="P-value",
                variable=var,
                py_val=python_pvalues[var],
                r_val=r_pvalues[var],
                abs_tol=config.pvalue_abs_tol,
                rel_tol=config.pvalue_rel_tol,
                rel_threshold=config.pvalue_rel_threshold,
            )
            result.pvalue_comparisons.append(comp)

    return result


def assert_implementations_match(
    comparison_result: ComparisonResult, raise_on_failure: bool = True
) -> bool:
    """Assert that Python and R implementations match within tolerances.

    Args:
        comparison_result: ComparisonResult object
        raise_on_failure: If True, raise AssertionError on failure

    Returns:
        True if all comparisons passed

    Raises:
        AssertionError: If any comparison failed and raise_on_failure is True
    """
    if not comparison_result.all_passed:
        summary = comparison_result.summary()
        if raise_on_failure:
            raise AssertionError(f"Implementation comparison failed:\n{summary}")
        return False
    return True
