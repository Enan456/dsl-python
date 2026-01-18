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
        elif key == 3:
            return self.objective
        elif key == 4:
            return self.success
        elif key == "coefficients":
            return self.coefficients
        elif key == "standard_errors":
            return self.standard_errors
        elif key == "vcov":
            return self.vcov
        elif key == "objective":
            return self.objective
        elif key == "success":
            return self.success
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

    def __getitem__(self, key):
        """Allow dictionary-style access to PowerDSLResult attributes."""
        if key == "power":
            return self.power
        elif key == "predicted_se":
            return self.predicted_se
        elif key == "critical_value":
            return self.critical_value
        elif key == "alpha":
            return self.alpha
        elif key == "dsl_out":
            return self.dsl_out
        else:
            raise KeyError(f"Invalid key: {key}")


def dsl(
    # New formula-based interface
    model: str = "lm",
    formula: Optional[str] = None,
    data: Optional[pd.DataFrame] = None,
    predicted_var: Optional[List[str]] = None,
    prediction: Optional[str] = None,
    labeled_ind: Optional[np.ndarray] = None,
    sample_prob: Optional[np.ndarray] = None,
    sl_method: Optional[str] = None,
    feature: Optional[List[str]] = None,
    family: Optional[str] = None,
    cross_fit: Optional[int] = None,
    sample_split: Optional[int] = None,
    seed: Optional[int] = None,
    # Legacy array-based interface
    X: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    method: str = "linear",
    # Fixed effects
    fe: Optional[str] = None,
) -> DSLResult:
    """
    Estimate DSL model.

    Supports two interfaces:
    1. Formula-based interface (R-compatible): formula, data, labeled_ind, sample_prob
    2. Array-based interface: X, y, labeled_ind, sample_prob

    Parameters
    ----------
    model : str, optional
        Model type ("lm", "logit", "felm"), by default "lm"
    formula : str, optional
        Model formula (e.g., "y ~ x1 + x2 + x3")
    data : pd.DataFrame, optional
        Data frame containing variables
    predicted_var : List[str], optional
        List of predicted variable names
    prediction : str, optional
        Column name for predictions
    labeled_ind : np.ndarray, optional
        Labeled indicator (1 for labeled, 0 for unlabeled)
    sample_prob : np.ndarray, optional
        Sampling probability for each observation
    sl_method : str, optional
        Supervised learning method (not used in current implementation)
    feature : List[str], optional
        Feature names for supervised learning (not used in current implementation)
    family : str, optional
        Distribution family (not used in current implementation)
    cross_fit : int, optional
        Number of cross-fitting folds (not used in current implementation)
    sample_split : int, optional
        Number of sample splits (not used in current implementation)
    seed : int, optional
        Random seed
    X : np.ndarray, optional
        Design matrix (for array-based interface)
    y : np.ndarray, optional
        Response variable (for array-based interface)
    method : str, optional
        Method for estimation ("linear", "logistic", "fixed_effects")
    fe : str, optional
        Fixed effects specification

    Returns
    -------
    DSLResult
        Object containing estimation results
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Determine which interface is being used
    using_formula_interface = formula is not None and data is not None

    if using_formula_interface:
        # Formula-based interface
        from patsy import dmatrices

        # Parse fixed effects if model is "felm"
        fe_vars = None
        formula_main = formula
        if model == "felm" and "|" in formula:
            parts = formula.split("|")
            formula_main = parts[0].strip()
            fe_vars = [v.strip() for v in parts[1].split("+")]

        # Extract X and y from formula
        y_df, X_df = dmatrices(formula_main, data, return_type="dataframe")
        X = X_df.values
        y = y_df.values.flatten()

        # Get labeled indicator
        if labeled_ind is None:
            labeled_ind = np.ones(len(data))

        # Get sample probability
        if sample_prob is None:
            # Calculate sample probability from labeled indicator
            n_labeled = int(np.sum(labeled_ind))
            sample_prob = np.full(len(data), n_labeled / len(data))

        # Get predictions if provided
        if prediction is not None and prediction in data.columns:
            y_pred = data[prediction].values
            X_pred = X.copy()
        else:
            # Use same values as original
            y_pred = y.copy()
            X_pred = X.copy()

        # Handle fixed effects
        fe_Y = None
        fe_X = None
        if model == "felm" and fe_vars is not None:
            # Create fixed effect dummies
            fe_data = pd.get_dummies(data[fe_vars], drop_first=True)
            fe_Y = fe_data.values
            fe_X = fe_data.values

    else:
        # Array-based interface
        if X is None or y is None:
            raise ValueError(
                "Either provide formula+data or X+y arrays"
            )

        # Ensure y is flattened
        y = np.asarray(y).flatten()
        X = np.asarray(X)

        if labeled_ind is None:
            labeled_ind = np.ones(X.shape[0])

        if sample_prob is None:
            n_labeled = int(np.sum(labeled_ind))
            sample_prob = np.full(X.shape[0], n_labeled / X.shape[0])

        # Use same values for predicted
        y_pred = y.copy()
        X_pred = X.copy()

        fe_Y = None
        fe_X = None

    # Determine model type for dsl_general
    if model in ["lm", "linear"]:
        model_internal = "lm"
    elif model in ["logit", "logistic"]:
        model_internal = "logit"
    elif model == "felm":
        model_internal = "felm"
    else:
        # Use method parameter for backward compatibility
        if method == "linear":
            model_internal = "lm"
        elif method == "logistic":
            model_internal = "logit"
        elif method == "fixed_effects":
            model_internal = "felm"
        else:
            model_internal = "lm"

    # Ensure arrays are proper numpy arrays
    labeled_ind = np.asarray(labeled_ind).flatten()
    sample_prob = np.asarray(sample_prob).flatten()

    # Estimate parameters using the general function
    if model_internal == "felm":
        par, info = dsl_general(
            y,
            X,
            y_pred,
            X_pred,
            labeled_ind,
            sample_prob,
            model=model_internal,
            fe_Y=fe_Y,
            fe_X=fe_X,
        )
    else:
        par, info = dsl_general(
            y,
            X,
            y_pred,
            X_pred,
            labeled_ind,
            sample_prob,
            model=model_internal,
        )

    vcov = info["vcov"]

    # Calculate predicted values and residuals
    if model_internal == "lm":
        predicted_values = X @ par
        residuals = y - predicted_values
    elif model_internal == "logit":
        logits = X @ par
        predicted_values = 1 / (1 + np.exp(-logits))
        residuals = y - predicted_values
    else:
        predicted_values = X @ par[:X.shape[1]]
        residuals = y - predicted_values

    # Populate and return DSLResult object
    return DSLResult(
        coefficients=par,
        standard_errors=info["standard_errors"],
        vcov=vcov,
        objective=info["objective"],
        success=info["convergence"],
        message=info["message"],
        niter=info["iterations"],
        model=model,
        labeled_size=int(np.sum(labeled_ind)),
        total_size=X.shape[0],
        predicted_values=predicted_values,
        residuals=residuals,
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
            model=model,
            formula=formula,
            data=data,
            labeled_ind=labeled_ind,
            sample_prob=sample_prob,
            **kwargs,
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
    n_coef = len(result.coefficients)

    # Calculate degrees of freedom
    if result.residuals is not None:
        df = len(result.residuals) - n_coef
    else:
        df = result.total_size - n_coef

    # Calculate t-statistics and p-values
    t_values = result.coefficients / result.standard_errors
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_values), df))

    # Create summary table
    summary_df = pd.DataFrame(
        {
            "Estimate": result.coefficients,
            "Std. Error": result.standard_errors,
            "t value": t_values,
            "Pr(>|t|)": p_values,
        }
    )

    return summary_df


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
    summary_df = pd.DataFrame(
        {
            "Power": result.power,
            "Predicted SE": result.predicted_se,
        }
    )

    return summary_df


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

    # Get number of coefficients
    n_coef = len(result.power)
    coef_names = [f"beta_{i}" for i in range(n_coef)]

    # Select coefficients to plot
    if coefficients is None:
        coefficients = coef_names
    elif isinstance(coefficients, str):
        coefficients = [coefficients]

    # Create plot
    plt.figure(figsize=(10, 6))
    for i, coef in enumerate(coefficients):
        if coef in coef_names:
            idx = coef_names.index(coef)
        else:
            idx = i
        plt.bar(coef, result.power[idx] if idx < len(result.power) else 0)

    plt.xlabel("Coefficient")
    plt.ylabel("Power")
    plt.title("DSL Power Analysis")
    plt.ylim(0, 1)
    plt.grid(True, axis='y')
    plt.show()
