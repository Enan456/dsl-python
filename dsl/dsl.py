"""
Core DSL (Double-Supervised Learning) module
"""

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

from .helpers.dsl_general import dsl_general
from .helpers.estimate import estimate_power


@dataclass
class DSLResult:
    """Results from DSL estimation."""

    coefficients: np.ndarray
    standard_errors: np.ndarray
    vcov: np.ndarray
    objective: float
    success: bool
    message: str
    niter: int
    model: str
    labeled_size: int
    total_size: int
    predicted_values: Optional[np.ndarray] = None
    residuals: Optional[np.ndarray] = None

    def __getitem__(self, key):
        """Allow indexing of DSLResult object."""
        if key == 0:
            return self.coefficients
        elif key == 1:
            return self.standard_errors
        elif key == 2:
            return self.vcov
        else:
            raise IndexError("DSLResult index out of range")


@dataclass
class PowerDSLResult:
    """Results from DSL power analysis."""

    power: np.ndarray
    predicted_se: np.ndarray
    critical_value: float
    alpha: float
    dsl_out: Optional[DSLResult] = None


def dsl(
    X: np.ndarray,
    y: np.ndarray,
    labeled_ind: np.ndarray,
    sample_prob: np.ndarray,
    model: str = "logit",
    method: str = "linear",
) -> DSLResult:
    """
    Estimate DSL model.

    Parameters
    ----------
    X : np.ndarray
        Design matrix
    y : np.ndarray
        Response variable
    labeled_ind : np.ndarray
        Labeled indicator
    sample_prob : np.ndarray
        Sampling probability
    model : str, optional
        Model type, by default "logit"
    method : str, optional
        Method for estimation, by default "linear"

    Returns
    -------
    DSLResult
        Object containing estimation results
    """
    # Determine model type for dsl_general
    model_internal = "logit"  # Default
    if method == "linear":
        model_internal = "lm"
    elif method == "logistic":
        model_internal = "logit"
    elif method == "fixed_effects":
        model_internal = "felm"
        # Note: FELM requires additional fe_Y, fe_X args not handled here yet
    # Keep original model name for result object
    model_name_for_result = model

    # Estimate parameters using the general function
    par, info = dsl_general(
        y,  # Pass y directly (will be flattened inside if needed)
        X,
        y,  # Pass y directly
        X,
        labeled_ind,
        sample_prob,
        model=model_internal,  # Use determined internal model type
    )

    # Note: dsl_vcov might be redundant if vcov is already computed in dsl_general
    # vcov = dsl_vcov(X, par, info["standard_errors"], model_internal)
    vcov = info["vcov"]  # Use vcov from dsl_general info dict

    # Populate and return DSLResult object
    return DSLResult(
        coefficients=par,
        standard_errors=info["standard_errors"],
        vcov=vcov,
        objective=info["objective"],
        success=info["convergence"],
        message=info["message"],
        niter=info["iterations"],
        model=model_name_for_result,  # Use original model name
        labeled_size=int(np.sum(labeled_ind)),
        total_size=X.shape[0],
        # predicted_values and residuals are not calculated here yet
    )


def power_dsl(
    formula: str,
    data: pd.DataFrame,
    labeled_ind: np.ndarray,
    sample_prob: Optional[np.ndarray] = None,
    model: str = "lm",
    fe: Optional[str] = None,
    method: str = "linear",
    n_samples: Optional[int] = None,
    alpha: float = 0.05,
    dsl_out: Optional[DSLResult] = None,
    **kwargs,
) -> PowerDSLResult:
    """
    Perform DSL power analysis.

    Parameters
    ----------
    formula : str
        Model formula
    data : pd.DataFrame
        Data frame
    labeled_ind : np.ndarray
        Labeled indicator
    sample_prob : Optional[np.ndarray], optional
        Sampling probability, by default None
    model : str, optional
        Model type, by default "lm"
    fe : Optional[str], optional
        Fixed effects variable, by default None
    method : str, optional
        Supervised learning method, by default "linear"
    n_samples : Optional[int], optional
        Number of samples for power analysis, by default None
    alpha : float, optional
        Significance level, by default 0.05
    dsl_out : Optional[DSLResult], optional
        DSL estimation results, by default None
    **kwargs : dict
        Additional arguments for the estimator

    Returns
    -------
    PowerDSLResult
        DSL power analysis results
    """
    # Estimate DSL model if not provided
    if dsl_out is None:
        dsl_out = dsl(
            data.values,
            data.values[:, 0],
            data.values[:, 0],
            data.values[:, 1],
            model,
            method,
        )

    # Parse formula
    from patsy import dmatrices

    _, X = dmatrices(formula, data, return_type="dataframe")
    X = X.values

    # Set default number of samples
    if n_samples is None:
        n_samples = len(data)

    # Estimate power
    power_results = estimate_power(
        X,
        dsl_out.coefficients,
        dsl_out.standard_errors,
        n_samples,
        alpha,
    )

    # Return results
    return PowerDSLResult(
        power=power_results["power"],
        predicted_se=power_results["predicted_se"],
        critical_value=power_results["critical_value"],
        alpha=power_results["alpha"],
        dsl_out=dsl_out,
    )


def summary(result: DSLResult) -> pd.DataFrame:
    """
    Summarize DSL estimation results.

    Parameters
    ----------
    result : DSLResult
        DSL estimation results

    Returns
    -------
    pd.DataFrame
        Summary table
    """
    # Create summary table
    summary = pd.DataFrame(
        {
            "Estimate": result.coefficients,
            "Std. Error": result.standard_errors,
            "t value": result.coefficients / result.standard_errors,
            "Pr(>|t|)": 2
            * (
                1
                - stats.t.cdf(
                    np.abs(result.coefficients / result.standard_errors),
                    len(result.residuals) - len(result.coefficients),
                )
            ),
        }
    )

    return summary


def summary_power(result: PowerDSLResult) -> pd.DataFrame:
    """
    Summarize DSL power analysis results.

    Parameters
    ----------
    result : PowerDSLResult
        DSL power analysis results

    Returns
    -------
    pd.DataFrame
        Summary table
    """
    # Create summary table
    summary = pd.DataFrame(
        {
            "Power": result.power,
            "Predicted SE": result.predicted_se,
        }
    )

    return summary


def plot_power(
    result: PowerDSLResult,
    coefficients: Optional[Union[str, List[str]]] = None,
) -> None:
    """
    Plot DSL power analysis results.

    Parameters
    ----------
    result : PowerDSLResult
        DSL power analysis results
    coefficients : Optional[Union[str, List[str]]], optional
        Coefficients to plot, by default None
    """
    import matplotlib.pyplot as plt

    # Get coefficient names
    if result.dsl_out is not None:
        from patsy import dmatrices

        _, X = dmatrices(
            result.dsl_out.formula, result.dsl_out.data, return_type="dataframe"
        )
        coef_names = X.columns
    else:
        coef_names = [f"beta_{i}" for i in range(len(result.power))]

    # Select coefficients to plot
    if coefficients is None:
        coefficients = coef_names
    elif isinstance(coefficients, str):
        coefficients = [coefficients]

    # Create plot
    plt.figure(figsize=(10, 6))
    for coef in coefficients:
        idx = coef_names.index(coef)
        plt.plot(
            result.predicted_se[idx],
            result.power[idx],
            label=coef,
            marker="o",
        )

    plt.xlabel("Predicted Standard Error")
    plt.ylabel("Power")
    plt.title("DSL Power Analysis")
    plt.legend()
    plt.grid(True)
    plt.show()
