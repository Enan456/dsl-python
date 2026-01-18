"""
Estimation helper functions for DSL
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error


def estimate_supervised(
    y: np.ndarray,
    X: np.ndarray,
    method: str = "linear",
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate a supervised learning model.

    Parameters
    ----------
    y : np.ndarray
        Outcome variable
    X : np.ndarray
        Feature matrix
    method : str, optional
        Supervised learning method, by default "linear"
    **kwargs : dict
        Additional arguments for the estimator

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Predicted values and standard errors
    """
    if method == "linear":
        # Add small regularization for numerical stability
        reg_param = 1e-8  # Match R's regularization
        X_dot_X = X.T @ X

        # Use QR decomposition for stable inverse
        Q, R = np.linalg.qr(X_dot_X + reg_param * np.eye(X_dot_X.shape[0]))
        X_dot_X_inv = np.linalg.solve(R.T @ R, R.T @ Q.T)

        beta = X_dot_X_inv @ X.T @ y
        y_pred = X @ beta
        residuals = y - y_pred
        sigma2 = np.sum(residuals**2) / (len(y) - X.shape[1])
        se = np.sqrt(np.diag(sigma2 * X_dot_X_inv))
        return y_pred, se
    elif method == "logistic":
        # Initialize parameters
        beta = np.zeros(X.shape[1])
        max_iter = 100
        tol = 1e-10  # Match R's tolerance
        eps = 1e-10  # Small constant for numerical stability
        reg_param = 1e-8  # Match R's regularization

        # R-like convergence criteria
        rel_tol = 1e-8
        abs_tol = 1e-10
        step_tol = 1e-10

        for i in range(max_iter):
            # Calculate probabilities using log-sum-exp trick
            z = X @ beta
            max_z = np.maximum(0, z)
            log_p = z - max_z - np.log1p(np.exp(z - max_z))
            p = np.exp(log_p)
            p = np.clip(p, eps, 1 - eps)

            # Calculate gradient and Hessian
            W = np.diag(p * (1 - p))
            gradient = X.T @ (y - p)
            hessian = X.T @ W @ X + reg_param * np.eye(X.shape[1])

            # Use QR decomposition for stable solve
            try:
                Q, R = np.linalg.qr(hessian)
                delta = np.linalg.solve(R.T @ R, R.T @ Q.T @ gradient)
            except np.linalg.LinAlgError:
                # Fallback to regularized solve if QR fails
                delta = np.linalg.solve(
                    hessian + reg_param * np.eye(hessian.shape[0]), gradient
                )

            beta_new = beta + delta

            # R-like convergence checks
            rel_change = np.max(np.abs(delta / (np.abs(beta) + eps)))
            abs_change = np.max(np.abs(delta))
            step_size = np.sqrt(np.sum(delta**2))

            if rel_change < rel_tol and abs_change < abs_tol and step_size < step_tol:
                break

            beta = beta_new

        # Calculate final probabilities
        z = X @ beta
        max_z = np.maximum(0, z)
        log_p = z - max_z - np.log1p(np.exp(z - max_z))
        p = np.exp(log_p)
        p = np.clip(p, eps, 1 - eps)

        # Calculate standard errors using stable methods
        W = np.diag(p * (1 - p))
        hessian = X.T @ W @ X + reg_param * np.eye(X.shape[1])

        # Use QR decomposition for stable inverse
        Q, R = np.linalg.qr(hessian)
        hessian_inv = np.linalg.solve(R.T @ R, R.T @ Q.T)
        se = np.sqrt(np.diag(hessian_inv))

        return p, se
    else:
        raise ValueError(f"Unknown method: {method}")


def estimate_fixed_effects(
    y: np.ndarray,
    X: np.ndarray,
    fe: np.ndarray,
    method: str = "linear",
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate a fixed effects model.

    Parameters
    ----------
    y : np.ndarray
        Outcome variable
    X : np.ndarray
        Feature matrix
    fe : np.ndarray
        Fixed effects variable (can be 1D or 2D)
    method : str, optional
        Supervised learning method, by default "linear"
    **kwargs : dict
        Additional arguments for the estimator

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Predicted values, standard errors, and fixed effects
    """
    # Demean the data
    y_demeaned = y.copy()
    X_demeaned = X.copy()
    fe_pred = np.zeros_like(y)

    # Ensure fe is 2D for consistent processing
    if fe.ndim == 1:
        fe_2d = fe.reshape(-1, 1)
    else:
        fe_2d = fe

    # Get unique fixed effects
    unique_fe = np.unique(fe_2d, axis=0)

    # Demean within each fixed effect group
    for fe_val in unique_fe:
        # Compare each row of fe_2d with fe_val
        mask = (fe_2d == fe_val).all(axis=1)
        y_mean = np.mean(y[mask])
        X_mean = np.mean(X[mask], axis=0)
        y_demeaned[mask] = y[mask] - y_mean
        X_demeaned[mask] = X[mask] - X_mean
        fe_pred[mask] = y_mean

    # Estimate model on demeaned data
    y_pred, se = estimate_supervised(y_demeaned, X_demeaned, method, **kwargs)

    # Add back fixed effects
    y_pred += fe_pred

    return y_pred, se, fe_pred


def estimate_power(
    X: np.ndarray,
    coefficients: np.ndarray,
    standard_errors: np.ndarray,
    n_samples: int,
    alpha: float = 0.05,
) -> Dict[str, np.ndarray]:
    """
    Estimate power for DSL model.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    coefficients : np.ndarray
        Estimated coefficients
    standard_errors : np.ndarray
        Estimated standard errors
    n_samples : int
        Number of samples for power analysis
    alpha : float, optional
        Significance level, by default 0.05

    Returns
    -------
    Dict[str, np.ndarray]
        Power analysis results
    """
    from scipy import stats

    # Compute predicted standard errors
    X_dot_X_inv = np.linalg.inv(X.T @ X)
    predicted_se = np.sqrt(np.diag(X_dot_X_inv)) * np.sqrt(n_samples)

    # Compute critical values
    critical_value = stats.norm.ppf(1 - alpha / 2)

    # Compute power
    power = (
        1
        - stats.norm.cdf(critical_value - np.abs(coefficients) / predicted_se)
        + stats.norm.cdf(-critical_value - np.abs(coefficients) / predicted_se)
    )

    return {
        "power": power,
        "predicted_se": predicted_se,
        "critical_value": critical_value,
        "alpha": alpha,
    }


def available_method(print_out: bool = True) -> List[str]:
    """
    Get available supervised learning methods.

    Parameters
    ----------
    print_out : bool, optional
        Whether to print the methods, by default True

    Returns
    -------
    List[str]
        List of available methods
    """
    methods = ["linear", "logistic", "random_forest"]
    if print_out:
        print("Available methods:", methods)
    return methods


def fit_model(
    outcome: str,
    labeled: str,
    covariates: List[str],
    data: pd.DataFrame,
    method: str = "random_forest",
    sample_prob: Optional[str] = None,
    family: str = "gaussian",
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Fit a supervised learning model.

    Parameters
    ----------
    outcome : str
        Name of outcome variable in data
    labeled : str
        Name of labeled indicator in data
    covariates : List[str]
        List of covariate names in data
    data : pd.DataFrame
        Data frame containing all variables
    method : str, optional
        Method to use for fitting, by default "random_forest"
    sample_prob : Optional[str], optional
        Name of sampling probability variable in data, by default None
    family : str, optional
        Family for GLM, by default "gaussian"
    **kwargs : Any
        Additional arguments passed to the estimator

    Returns
    -------
    Dict[str, Any]
        Dictionary containing fitted model and metadata
    """
    # Get labeled data
    labeled_data = data[data[labeled] == 1]

    # Prepare X and y
    X = labeled_data[covariates].values
    y = labeled_data[outcome].values

    # Initialize model based on method
    if method == "random_forest":
        model = RandomForestRegressor(**kwargs)
    elif method == "linear":
        model = LinearRegression(**kwargs)
    elif method == "logistic":
        model = LogisticRegression(**kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Fit model
    model.fit(X, y)

    # Return model and metadata
    return {
        "method": method,
        "model": model,
        "family": family,
        "covariates": covariates,
        "outcome": outcome,
    }


def fit_test(
    fit_out: Dict[str, Any],
    outcome: str,
    labeled: str,
    covariates: List[str],
    data: pd.DataFrame,
    method: str = "random_forest",
    family: str = "gaussian",
) -> Tuple[np.ndarray, float]:
    """
    Test a fitted model on new data.

    Parameters
    ----------
    fit_out : Dict[str, Any]
        Output from fit_model
    outcome : str
        Name of outcome variable in data
    labeled : str
        Name of labeled indicator in data
    covariates : List[str]
        List of covariate names in data
    data : pd.DataFrame
        Data frame containing all variables
    method : str, optional
        Method used for fitting, by default "random_forest"
    family : str, optional
        Family for GLM, by default "gaussian"

    Returns
    -------
    Tuple[np.ndarray, float]
        Predicted values and RMSE
    """
    # Get test data
    test_data = data[data[labeled] == 1]

    # Prepare X and y
    X = test_data[covariates].values
    y = test_data[outcome].values

    # Get model
    model = fit_out["model"]

    # Make predictions
    y_pred = model.predict(X)

    # Compute RMSE
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    return y_pred, rmse
