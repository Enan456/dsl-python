#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
R reference results loader and parser.
Loads reference results from R output files for comparison testing.
"""

import os
import re
from typing import Dict, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class ReferenceResult:
    """Container for R reference results."""

    name: str
    coefficients: Dict[str, float]
    standard_errors: Dict[str, float]
    p_values: Dict[str, float]
    ci_lower: Optional[Dict[str, float]] = None
    ci_upper: Optional[Dict[str, float]] = None
    model: Optional[str] = None
    n_obs: Optional[int] = None


def parse_r_output_panchen(file_path: str) -> ReferenceResult:
    """Parse R output file for PanChen dataset results.

    Args:
        file_path: Path to R output text file

    Returns:
        ReferenceResult object with parsed coefficients, SEs, and p-values
    """
    with open(file_path, "r") as f:
        content = f.read()

    # Extract coefficient table
    coef_section = re.search(
        r"Coefficients:\s*=+\s*(.*?)\s*---", content, re.DOTALL
    )
    if not coef_section:
        raise ValueError("Could not find coefficients section in R output")

    coef_text = coef_section.group(1)
    lines = coef_text.strip().split("\n")

    coefficients = {}
    standard_errors = {}
    p_values = {}
    ci_lower = {}
    ci_upper = {}

    # Skip header line
    data_lines = [line for line in lines[1:] if line.strip()]

    for line in data_lines:
        # Parse line: varname estimate stderr ci_lower ci_upper pvalue [stars]
        parts = line.split()
        if len(parts) < 6:
            continue

        var_name = parts[0]
        estimate = float(parts[1])
        stderr = float(parts[2])
        ci_low = float(parts[3])
        ci_high = float(parts[4])
        pval = float(parts[5])

        coefficients[var_name] = estimate
        standard_errors[var_name] = stderr
        p_values[var_name] = pval
        ci_lower[var_name] = ci_low
        ci_upper[var_name] = ci_high

    # Extract model info
    model_match = re.search(r"Model:\s+(\w+)", content)
    model = model_match.group(1) if model_match else None

    # Extract number of observations
    n_obs_match = re.search(r"Number of Labeled Observations:\s+(\d+)", content)
    n_obs = int(n_obs_match.group(1)) if n_obs_match else None

    return ReferenceResult(
        name="panchen",
        coefficients=coefficients,
        standard_errors=standard_errors,
        p_values=p_values,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        model=model,
        n_obs=n_obs,
    )


def load_panchen_r_reference() -> ReferenceResult:
    """Load PanChen R reference results from the standard location.

    Returns:
        ReferenceResult object with PanChen R results
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    file_path = os.path.join(
        project_root, "PanChen_test", "target_r_output_panchen.txt"
    )

    return parse_r_output_panchen(file_path)
