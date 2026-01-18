"""
General DSL helper functions
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm  # Import statsmodels
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


def validate_inputs(
    Y: np.ndarray,
    X: np.ndarray,
    labeled_ind: np.ndarray,
    sample_prob: np.ndarray,
    model: str = "lm",
) -> None:
    """
    Validate inputs for DSL estimation.

    Parameters
    ----------
    Y : np.ndarray
        Response variable
    X : np.ndarray
        Design matrix
    labeled_ind : np.ndarray
        Labeled indicator (1 for labeled, 0 for unlabeled)
    sample_prob : np.ndarray
        Sampling probability for each observation
    model : str
        Model type ("lm", "logit", "felm")

    Raises
    ------
    ValueError
        If any input validation fails
    """
    n_obs = X.shape[0]
    n_features = X.shape[1] if X.ndim > 1 else 1

    # Check for empty inputs
    if n_obs == 0:
        raise ValueError("Input arrays cannot be empty")

    # Check dimensions match
    Y_flat = np.asarray(Y).flatten()
    if len(Y_flat) != n_obs:
        raise ValueError(
            f"Y length ({len(Y_flat)}) must match X rows ({n_obs})"
        )

    if len(labeled_ind) != n_obs:
        raise ValueError(
            f"labeled_ind length ({len(labeled_ind)}) must match X rows ({n_obs})"
        )

    if len(sample_prob) != n_obs:
        raise ValueError(
            f"sample_prob length ({len(sample_prob)}) must match X rows ({n_obs})"
        )

    # Check for NaN/inf in X
    if np.any(~np.isfinite(X)):
        nan_count = np.sum(np.isnan(X))
        inf_count = np.sum(np.isinf(X))
        raise ValueError(
            f"X contains {nan_count} NaN and {inf_count} infinite values. "
            "Please handle missing/infinite data before calling DSL."
        )

    # Check for NaN/inf in Y
    if np.any(~np.isfinite(Y_flat)):
        nan_count = np.sum(np.isnan(Y_flat))
        inf_count = np.sum(np.isinf(Y_flat))
        raise ValueError(
            f"Y contains {nan_count} NaN and {inf_count} infinite values. "
            "Please handle missing/infinite data before calling DSL."
        )

    # Validate labeled_ind is binary
    unique_labels = np.unique(labeled_ind)
    if not np.all(np.isin(unique_labels, [0, 1])):
        raise ValueError(
            f"labeled_ind must contain only 0 and 1. "
            f"Found unique values: {unique_labels}"
        )

    # Check at least some labeled observations
    n_labeled = int(np.sum(labeled_ind))
    if n_labeled == 0:
        raise ValueError(
            "No labeled observations (labeled_ind contains no 1s). "
            "DSL requires at least some labeled data."
        )

    # Check enough labeled for estimation
    if n_labeled < n_features:
        raise ValueError(
            f"Not enough labeled observations ({n_labeled}) to estimate "
            f"{n_features} parameters. Need at least {n_features} labeled points."
        )

    # Validate sample_prob is in valid range
    if np.any(sample_prob <= 0):
        raise ValueError(
            "sample_prob must be strictly positive. "
            f"Found {np.sum(sample_prob <= 0)} values <= 0."
        )

    if np.any(sample_prob > 1):
        raise ValueError(
            "sample_prob must be <= 1. "
            f"Found {np.sum(sample_prob > 1)} values > 1."
        )

    # Check for very small sample_prob (numerical issues)
    if np.any(sample_prob < 1e-10):
        logger.warning(
            f"sample_prob contains {np.sum(sample_prob < 1e-10)} very small values "
            "(< 1e-10). This may cause numerical instability."
        )

    # Validate model type
    valid_models = ["lm", "logit", "felm", "linear", "logistic"]
    if model not in valid_models:
        raise ValueError(
            f"Unknown model type: '{model}'. "
            f"Valid options are: {valid_models}"
        )

    # For logistic regression, check Y is binary
    if model in ["logit", "logistic"]:
        unique_y = np.unique(Y_flat[labeled_ind == 1])
        if not np.all(np.isin(unique_y, [0, 1])):
            raise ValueError(
                f"For logistic regression, Y must be binary (0 or 1). "
                f"Found unique values in labeled data: {unique_y}"
            )


def _stable_sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Numerically stable sigmoid function.

    Uses the log-sum-exp trick to avoid overflow/underflow for large |z|.
    For z > 0: sigmoid(z) = 1 / (1 + exp(-z))
    For z <= 0: sigmoid(z) = exp(z) / (1 + exp(z))

    Parameters
    ----------
    z : np.ndarray
        Input values (linear predictor)

    Returns
    -------
    np.ndarray
        Sigmoid values in range (0, 1)
    """
    # Use different formulas for positive and negative z to avoid overflow
    result = np.zeros_like(z, dtype=np.float64)
    pos_mask = z >= 0
    neg_mask = ~pos_mask

    # For z >= 0: 1 / (1 + exp(-z))
    result[pos_mask] = 1.0 / (1.0 + np.exp(-z[pos_mask]))

    # For z < 0: exp(z) / (1 + exp(z)) - avoids exp(large positive) overflow
    exp_z = np.exp(z[neg_mask])
    result[neg_mask] = exp_z / (1.0 + exp_z)

    return result


def dsl_general(
    Y_orig: np.ndarray,
    X_orig: np.ndarray,
    Y_pred: np.ndarray,
    X_pred: np.ndarray,
    labeled_ind: np.ndarray,
    sample_prob_use: np.ndarray,
    model: str = "lm",
    fe_Y: Optional[np.ndarray] = None,
    fe_X: Optional[np.ndarray] = None,
    moment_fn: Optional[callable] = None,
    jac_fn: Optional[callable] = None,
    use_ipw: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    General DSL estimation function.

    Parameters
    ----------
    Y_orig : np.ndarray
        Original outcome
    X_orig : np.ndarray
        Original features
    Y_pred : np.ndarray
        Predicted outcome
    X_pred : np.ndarray
        Predicted features
    labeled_ind : np.ndarray
        Labeled indicator
    sample_prob_use : np.ndarray
        Sampling probability
    model : str, optional
        Model type, by default "lm"
    fe_Y : Optional[np.ndarray], optional
        Fixed effects outcome, by default None
    fe_X : Optional[np.ndarray], optional
        Fixed effects features, by default None
    moment_fn : Optional[callable], optional
        Moment function, by default None
    jac_fn : Optional[callable], optional
        Jacobian function, by default None
    use_ipw : bool, optional
        If True, use IPW weighting instead of doubly-robust.
        This should be True when predictions are incorporated into X.
        Default is False.

    Returns
    -------
    Tuple[np.ndarray, Dict[str, Any]]
        Estimated parameters and additional information
    """
    from .moment import (
        felm_dsl_Jacobian,
        felm_dsl_moment_base,
        lm_dsl_Jacobian,
        lm_dsl_moment_base,
        logit_dsl_Jacobian,
        logit_dsl_moment_base,
    )

    # Validate inputs before proceeding
    validate_inputs(Y_orig, X_orig, labeled_ind, sample_prob_use, model)

    if moment_fn is None:
        if model == "lm":
            moment_fn = lm_dsl_moment_base
        elif model == "logit":
            moment_fn = logit_dsl_moment_base
        elif model == "felm":
            moment_fn = felm_dsl_moment_base

    if jac_fn is None:
        if model == "lm":
            jac_fn = lm_dsl_Jacobian
        elif model == "logit":
            jac_fn = logit_dsl_Jacobian
        elif model == "felm":
            jac_fn = felm_dsl_Jacobian

    # Standardize data to make estimation easier
    X_orig_use = np.ones_like(X_orig)
    X_pred_use = np.ones_like(X_pred)
    scale_cov = np.where(np.std(X_pred, axis=0) > 0)[0]

    # Check if first column is intercept
    with_intercept = np.all(X_orig[:, 0] == 1)
    mean_X = np.zeros(X_pred.shape[1])
    sd_X = np.ones(X_pred.shape[1])

    if with_intercept:
        # For models with intercept
        for j in scale_cov:
            X_pred_use[:, j] = (X_pred[:, j] - np.mean(X_pred[:, j])) / np.std(
                X_pred[:, j]
            )
            X_orig_use[:, j] = (X_orig[:, j] - np.mean(X_pred[:, j])) / np.std(
                X_pred[:, j]
            )
            mean_X[j] = np.mean(X_pred[:, j])
            sd_X[j] = np.std(X_pred[:, j])
        mean_X = mean_X[1:]  # Remove first element (intercept)
        sd_X = sd_X[1:]  # Remove first element (intercept)
    else:
        # For models without intercept
        for j in scale_cov:
            X_pred_use[:, j] = X_pred[:, j] / np.std(X_pred[:, j])
            X_orig_use[:, j] = X_orig[:, j] / np.std(X_pred[:, j])
            sd_X[j] = np.std(X_pred[:, j])

    # Initialize parameters using statsmodels for better starting point
    X_labeled = X_orig_use[labeled_ind == 1]
    y_labeled = Y_orig[labeled_ind == 1].flatten()

    try:
        if model == "logit":
            model_sm = sm.Logit(y_labeled, X_labeled)
            result_sm = model_sm.fit(disp=0, method='bfgs', maxiter=100)
            par_init = result_sm.params.copy()
            logger.info("Initialized parameters using statsmodels Logit")
        elif model in ["lm", "felm"]:
            # Use OLS for linear model initialization
            model_sm = sm.OLS(y_labeled, X_labeled)
            result_sm = model_sm.fit()
            par_init = result_sm.params.copy()
            logger.info("Initialized parameters using statsmodels OLS")
        else:
            par_init = np.zeros(X_orig.shape[1])
    except Exception as e:
        logger.warning(f"Statsmodels initialization failed ({str(e)}), using zeros")
        par_init = np.zeros(X_orig.shape[1])

    # Ensure par_init is finite
    if not np.all(np.isfinite(par_init)):
        logger.warning("Non-finite initial parameters, resetting to zeros")
        par_init = np.zeros(X_orig.shape[1])

    # Define objective and gradient functions
    def objective(par):
        if model == "felm":
            # FELM needs special handling, assuming moment_fn handles FE
            m = moment_fn(
                par,
                labeled_ind,
                sample_prob_use,
                Y_orig.flatten(),  # Ensure flattened Y
                X_orig_use,
                Y_pred.flatten(),  # Ensure flattened Y
                X_pred_use,
                fe_Y,
                fe_X,
            )
        else:
            m = moment_fn(
                par,
                labeled_ind,
                sample_prob_use,
                Y_orig.flatten(),  # Ensure flattened Y
                X_orig_use,
                Y_pred.flatten(),  # Ensure flattened Y
                X_pred_use,
                use_ipw=use_ipw,
            )
        # Objective: sum of squared mean moments
        mean_m = np.mean(m, axis=0)
        return np.sum(mean_m**2)

    def gradient(par):
        # Calculate average Jacobian J (k x k)
        if model == "felm":
            # Assuming jac_fn for felm returns the correct (k x k) average J
            J = jac_fn(
                par,
                labeled_ind,
                sample_prob_use,
                Y_orig.flatten(),
                X_orig_use,
                Y_pred.flatten(),
                X_pred_use,
                model=model,
                fe_Y=fe_Y,  # Pass FE info if needed by jac_fn
                fe_X=fe_X,
            )
            # Recalculate moments m (n x k)
            m = moment_fn(
                par,
                labeled_ind,
                sample_prob_use,
                Y_orig.flatten(),
                X_orig_use,
                Y_pred.flatten(),
                X_pred_use,
                fe_Y,
                fe_X,
            )
        else:
            J = jac_fn(
                par,
                labeled_ind,
                sample_prob_use,
                Y_orig.flatten(),
                X_orig_use,
                Y_pred.flatten(),
                X_pred_use,
                model=model,
            )
            # Recalculate moments m (n x k)
            m = moment_fn(
                par,
                labeled_ind,
                sample_prob_use,
                Y_orig.flatten(),
                X_orig_use,
                Y_pred.flatten(),
                X_pred_use,
                use_ipw=use_ipw,
            )

        # Gradient: 2 * J.T @ mean(m)
        mean_m = np.mean(m, axis=0)
        # Ensure mean_m is (k, 1) for matmul
        grad = 2 * J.T @ mean_m.reshape(-1, 1)
        return grad.flatten()  # Optimizer expects flattened gradient

    # Add numerical stability check
    def check_numerical_stability(par):
        if not np.all(np.isfinite(par)):
            logger.warning("Non-finite parameters detected")
            return False
        if np.any(np.abs(par) > 1e10):
            logger.warning("Parameters too large detected")
            return False
        return True

    # Add optimization monitoring callback
    def callback(xk):
        if not check_numerical_stability(xk):
            logger.warning("Numerical stability check failed in callback")
        obj_val = objective(xk)
        grad_norm = np.linalg.norm(gradient(xk))
        logger.debug(
            f"Current objective: {obj_val:.6f}, gradient norm: {grad_norm:.6f}"
        )

    # Set up optimization options with improved settings
    optim_options = {
        "method": "BFGS",
        "jac": gradient,
        "options": {
            "gtol": 1e-5,  # Relaxed tolerance for better convergence
            "maxiter": 1000,
            "disp": False,  # Suppress optimization progress output
        },
        "callback": callback,
    }

    # Track best result across optimization attempts
    best_result = None
    best_objective = np.inf

    # Run optimization with error handling
    try:
        result = minimize(objective, par_init, **optim_options)
        if result.fun < best_objective:
            best_result = result
            best_objective = result.fun

        if not result.success:
            logger.warning(f"BFGS optimization did not fully converge: {result.message}")
            logger.info("Attempting fallback optimization with L-BFGS-B")

            # Try L-BFGS-B as fallback with bounds
            lbfgsb_options = {
                "method": "L-BFGS-B",
                "jac": gradient,
                "options": {
                    "gtol": 1e-5,
                    "maxiter": 1000,
                    "disp": False,
                },
            }
            result_lbfgsb = minimize(objective, par_init, **lbfgsb_options)
            if result_lbfgsb.fun < best_objective:
                best_result = result_lbfgsb
                best_objective = result_lbfgsb.fun

            if not result_lbfgsb.success:
                logger.warning(f"L-BFGS-B also did not fully converge: {result_lbfgsb.message}")

                # Try Nelder-Mead as last resort (no gradient needed)
                logger.info("Attempting fallback optimization with Nelder-Mead")
                nm_options = {
                    "method": "Nelder-Mead",
                    "options": {
                        "xatol": 1e-5,
                        "fatol": 1e-5,
                        "maxiter": 2000,
                        "disp": False,
                    },
                }
                result_nm = minimize(objective, best_result.x, **nm_options)
                if result_nm.fun < best_objective:
                    best_result = result_nm
                    best_objective = result_nm.fun

    except Exception as e:
        logger.error(f"Optimization error: {str(e)}")
        # If we have any result, use it; otherwise use initial parameters
        if best_result is None:
            logger.warning("Using initial parameters due to optimization failure")
            # Create a mock result object
            class MockResult:
                def __init__(self, x, fun, success, nit, message):
                    self.x = x
                    self.fun = fun
                    self.success = success
                    self.nit = nit
                    self.message = message
            best_result = MockResult(
                par_init, objective(par_init), False, 0,
                f"Optimization failed with error: {str(e)}"
            )

    # Use the best result found
    result = best_result

    # Log final status
    if not result.success:
        logger.warning(
            f"Optimization did not fully converge. "
            f"Returning best parameters found with objective value: {result.fun:.6e}"
        )
    else:
        logger.info(f"Optimization converged successfully with objective value: {result.fun:.6e}")

    # Get final parameters
    par_opt_scaled = result.x

    # Compute sandwich variance estimator
    n_obs = X_orig.shape[0]
    n_features = X_orig.shape[1]

    # Recalculate Jacobian and Moments at optimal parameters (result.x)
    if model == "felm":
        J = jac_fn(
            par_opt_scaled,
            labeled_ind,
            sample_prob_use,
            Y_orig.flatten(),
            X_orig_use,
            Y_pred.flatten(),
            X_pred_use,
            model=model,
            fe_Y=fe_Y,
            fe_X=fe_X,
        )
        m = moment_fn(
            par_opt_scaled,
            labeled_ind,
            sample_prob_use,
            Y_orig.flatten(),
            X_orig_use,
            Y_pred.flatten(),
            X_pred_use,
            fe_Y,
            fe_X,
        )
    else:
        J = jac_fn(
            par_opt_scaled,
            labeled_ind,
            sample_prob_use,
            Y_orig.flatten(),
            X_orig_use,
            Y_pred.flatten(),
            X_pred_use,
            model=model,
        )
        m = moment_fn(
            par_opt_scaled,
            labeled_ind,
            sample_prob_use,
            Y_orig.flatten(),
            X_orig_use,
            Y_pred.flatten(),
            X_pred_use,
            use_ipw=use_ipw,
        )

    # J should be (k, k), m should be (n, k)
    if J.shape != (n_features, n_features):
        raise ValueError(
            f"Jacobian shape mismatch: Expected ({n_features},{n_features}), got {J.shape}"
        )
    if m.shape[1] != n_features:
        raise ValueError(
            f"Moment shape mismatch: Expected (*, {n_features}), got {m.shape}"
        )

    # Compute sandwich variance estimator using helper function
    vcov_scaled = compute_sandwich_var(J, m, n_obs)

    # Rescale parameters back to original scale
    par_final = np.copy(result.x)  # Start with optimized scaled params
    if with_intercept:
        if n_features > 1:  # Check if there are non-intercept features
            # Unscale non-intercept parameters: par_orig = par_scaled / sd
            par_orig_non_intercept = par_final[1:] / sd_X
            # Unscale intercept: b0_orig = b0_s - np.sum(b_orig * mean)
            par_orig_intercept = par_final[0] - np.sum(par_orig_non_intercept * mean_X)
            # Update final parameters
            par_final[0] = par_orig_intercept
            par_final[1:] = par_orig_non_intercept
        # If only intercept, par_final[0] is already in original scale
    else:
        # Unscale all parameters: par_orig = par_scaled / sd
        par_final = par_final / sd_X

    # Create the rescaling Jacobian D = d(par_orig)/d(par_scaled)
    D_rescale_jacobian = np.identity(n_features)
    if with_intercept:
        if n_features > 1:
            # d(b1_orig)/d(b1_s) = 1/sd
            inv_sd_X = 1.0 / sd_X
            np.fill_diagonal(D_rescale_jacobian[1:, 1:], inv_sd_X)
            # d(b0_orig)/d(b1_s) = -mean/sd
            D_rescale_jacobian[0, 1:] = -mean_X / sd_X
        # d(b0_orig)/d(b0_s) = 1 (already set by identity)
    else:
        # d(b_orig)/d(b_s) = 1/sd
        inv_sd_X = 1.0 / sd_X
        np.fill_diagonal(D_rescale_jacobian, inv_sd_X)

    # Rescale variance: vcov_orig = D @ vcov_scaled @ D.T
    vcov_final = D_rescale_jacobian @ vcov_scaled @ D_rescale_jacobian.T

    # Return results
    return par_final, {
        "vcov": vcov_final,
        "standard_errors": np.sqrt(np.diag(vcov_final)),
        "convergence": result.success,
        "iterations": result.nit,
        "objective": result.fun,  # Objective value from optimizer
        "message": result.message,
    }


def dsl_predict(X: np.ndarray, coef: np.ndarray, model: str = "linear") -> np.ndarray:
    """
    Predict using DSL estimated coefficients.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    coef : np.ndarray
        Estimated coefficients from DSL estimation
    model : str, optional
        Model type: "linear", "lm", "felm" for linear regression,
        "logit" or "logistic" for logistic regression. Default is "linear".

    Returns
    -------
    np.ndarray
        Predicted values:
        - For linear models: X @ coef (linear predictor)
        - For logistic models: predicted probabilities P(Y=1|X)
    """
    if model in ["linear", "lm", "felm"]:
        return X @ coef
    elif model in ["logit", "logistic"]:
        # Use numerically stable sigmoid computation
        z = X @ coef
        return _stable_sigmoid(z)
    else:
        raise ValueError(f"Unknown model type: {model}. Use 'linear', 'lm', 'felm', 'logit', or 'logistic'.")


def dsl_residuals(
    Y: np.ndarray,
    X: np.ndarray,
    par: np.ndarray,
    model: str = "lm",
) -> np.ndarray:
    """
    Compute residuals using DSL estimates.

    Parameters
    ----------
    Y : np.ndarray
        Outcomes
    X : np.ndarray
        Features
    par : np.ndarray
        Estimated parameters
    model : str, optional
        Model type, by default "lm"

    Returns
    -------
    np.ndarray
        Residuals
    """
    if model == "lm":
        return Y - X @ par
    elif model == "logit":
        return Y - _stable_sigmoid(X @ par)
    else:
        raise ValueError(f"Unknown model type: {model}")


def dsl_vcov(
    X: np.ndarray,
    par: np.ndarray,
    se: np.ndarray,
    model: str = "lm",
) -> np.ndarray:
    """
    Compute variance-covariance matrix using DSL estimates.

    Parameters
    ----------
    X : np.ndarray
        Features
    par : np.ndarray
        Estimated parameters
    se : np.ndarray
        Standard errors
    model : str, optional
        Model type, by default "lm"

    Returns
    -------
    np.ndarray
        Variance-covariance matrix
    """
    # Compute predicted values
    if model == "lm":
        pred = X @ par
    elif model == "logit":
        pred = _stable_sigmoid(X @ par)
    else:
        raise ValueError(f"Unknown model type: {model}")

    # Compute weights
    if model == "lm":
        w = np.ones_like(pred)
    elif model == "logit":
        w = pred * (1 - pred)

    # Compute variance-covariance matrix
    X_w = X * w[:, np.newaxis]
    V = np.linalg.inv(X_w.T @ X_w)
    V = V * (se**2)[:, np.newaxis]

    return V


def dsl_general_Jacobian(
    par: np.ndarray,
    labeled_ind: np.ndarray,
    sample_prob_use: np.ndarray,
    Y_orig: np.ndarray,
    X_orig: np.ndarray,
    Y_pred: np.ndarray,
    X_pred: np.ndarray,
    model: str = "lm",
    fe_Y: Optional[np.ndarray] = None,
    fe_X: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute the average Jacobian matrix (k x k) for DSL estimation.

    Parameters
    ----------
    par : np.ndarray
        Parameters at which to evaluate the Jacobian
    labeled_ind : np.ndarray
        Labeled indicator
    sample_prob_use : np.ndarray
        Sampling probability
    Y_orig : np.ndarray
        Original outcome
    X_orig : np.ndarray
        Original features
    Y_pred : np.ndarray
        Predicted outcome
    X_pred : np.ndarray
        Predicted features
    model : str, optional
        Model type, by default "lm"
    fe_Y : Optional[np.ndarray], optional
        Fixed effects outcome, by default None
    fe_X : Optional[np.ndarray], optional
        Fixed effects features, by default None

    Returns
    -------
    np.ndarray
        Jacobian matrix
    """
    # Import moment functions
    from .moment import felm_dsl_Jacobian, lm_dsl_Jacobian, logit_dsl_Jacobian

    # Select appropriate Jacobian function
    if model == "lm":
        jac_fn = lm_dsl_Jacobian
        J = jac_fn(
            par, labeled_ind, sample_prob_use, Y_orig, X_orig, Y_pred, X_pred, model=model
        )
    elif model == "logit":
        jac_fn = logit_dsl_Jacobian
        J = jac_fn(
            par, labeled_ind, sample_prob_use, Y_orig, X_orig, Y_pred, X_pred, model=model
        )
    elif model == "felm":
        jac_fn = felm_dsl_Jacobian
        J = jac_fn(
            par, labeled_ind, sample_prob_use, Y_orig, X_orig, Y_pred, X_pred,
            model=model, fe_Y=fe_Y, fe_X=fe_X
        )
    else:
        raise ValueError(f"Unknown model type: {model}")

    return J


def dsl_general_moment(
    par: np.ndarray,
    labeled_ind: np.ndarray,
    sample_prob_use: np.ndarray,
    Y_orig: np.ndarray,
    X_orig: np.ndarray,
    Y_pred: np.ndarray,
    X_pred: np.ndarray,
    fe_Y: Optional[np.ndarray] = None,
    fe_X: Optional[np.ndarray] = None,
    model: str = "lm",
    tol: float = 1e-5,
) -> float:
    """
    Compute the general DSL GMM objective function value.

    Parameters
    ----------
    par : np.ndarray
        Parameters at which to evaluate the moment
    labeled_ind : np.ndarray
        Labeled indicator
    sample_prob_use : np.ndarray
        Sampling probability
    Y_orig : np.ndarray
        Original outcome
    X_orig : np.ndarray
        Original features
    Y_pred : np.ndarray
        Predicted outcome
    X_pred : np.ndarray
        Predicted features
    fe_Y : Optional[np.ndarray], optional
        Fixed effects outcome, by default None
    fe_X : Optional[np.ndarray], optional
        Fixed effects features, by default None
    model : str, optional
        Model type, by default "lm"
    tol : float, optional
        Tolerance for numerical stability, by default 1e-5

    Returns
    -------
    float
        Value of the moment function
    """
    # Import moment functions
    from .moment import felm_dsl_moment_base, lm_dsl_moment_base, logit_dsl_moment_base

    # Select appropriate moment function
    if model == "lm":
        moment_fn = lm_dsl_moment_base
    elif model == "logit":
        moment_fn = logit_dsl_moment_base
    elif model == "felm":
        moment_fn = felm_dsl_moment_base
    else:
        raise ValueError(f"Unknown model type: {model}")

    # Compute moment
    if model == "felm":
        m = moment_fn(
            par,
            labeled_ind,
            sample_prob_use,
            Y_orig,
            X_orig,
            Y_pred,
            X_pred,
            fe_Y,
            fe_X,
        )
    else:
        m = moment_fn(
            par,
            labeled_ind,
            sample_prob_use,
            Y_orig,
            X_orig,
            Y_pred,
            X_pred,
        )

    # Return sum of squared moments
    return float(np.sum(m**2) + tol)


def dsl_general_moment_base_decomp(
    par: np.ndarray,
    labeled_ind: np.ndarray,
    sample_prob_use: np.ndarray,
    Y_orig: np.ndarray,
    X_orig: np.ndarray,
    Y_pred: np.ndarray,
    X_pred: np.ndarray,
    model: str = "lm",
    clustered: bool = False,
    cluster: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the base moment decomposition for DSL estimation.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (main_1, main_23) components where Meat = main_1 + main_23
    """
    # Import moment functions for m_orig and m_pred
    from .moment import (
        lm_dsl_moment_orig,
        lm_dsl_moment_pred,
        logit_dsl_moment_orig,
        logit_dsl_moment_pred,
    )

    # Select appropriate originals/predicted moment functions
    if model == "lm":
        m_orig_fn = lm_dsl_moment_orig
        m_pred_fn = lm_dsl_moment_pred
    elif model == "logit":
        m_orig_fn = logit_dsl_moment_orig
        m_pred_fn = logit_dsl_moment_pred
    elif model == "felm":
        # Use the same decomposition form as lm for felm
        m_orig_fn = lm_dsl_moment_orig
        m_pred_fn = lm_dsl_moment_pred
    else:
        raise ValueError(f"Unknown model type: {model}")

    # Compute components
    m_orig = m_orig_fn(par, labeled_ind, sample_prob_use, Y_orig, X_orig, Y_pred, X_pred)
    m_pred = m_pred_fn(par, labeled_ind, sample_prob_use, Y_orig, X_orig, Y_pred, X_pred)

    n = X_orig.shape[0]
    diff = m_orig - m_pred

    if not clustered:
        # Non-clustered
        D1 = np.diag(labeled_ind / (sample_prob_use ** 2))
        D2 = np.diag(labeled_ind / sample_prob_use)
        main_1 = (diff.T @ D1 @ diff) / n
        main_2 = (m_pred.T @ m_pred) / n
        main_3 = ((m_pred.T @ D2 @ diff) + (diff.T @ D2 @ m_pred)) / n
    else:
        if cluster is None:
            raise ValueError("cluster must be provided when clustered=True")
        # Weighted difference by r/pi
        w = (labeled_ind / sample_prob_use).reshape(-1, 1)
        diff_w = diff * w
        # Sum within clusters
        uniq = np.unique(cluster)
        s_pred = np.stack([m_pred[cluster == g].sum(axis=0) for g in uniq], axis=0)
        s_diff = np.stack([diff_w[cluster == g].sum(axis=0) for g in uniq], axis=0)
        main_1 = (s_diff.T @ s_diff) / n
        main_2 = (s_pred.T @ s_pred) / n
        main_3_c0 = (s_pred.T @ s_diff) / n
        main_3 = main_3_c0 + main_3_c0.T

    return main_1, main_2 + main_3


def dsl_general_moment_est(
    model: str,
    formula: str,
    labeled: str,
    sample_prob: str,
    predicted_var: List[str],
    data_orig: pd.DataFrame,
    data_pred: pd.DataFrame,
    index: Optional[List[str]] = None,
    fixed_effect: Optional[str] = None,
    clustered: bool = False,
    cluster: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Estimate DSL model using moment conditions (Original Interface - Deprecated?).
    This function seems to use a different approach than dsl_general
    and might be outdated or serve a different purpose (e.g., decomposition).

    Parameters
    ----------
    model : str
        Model type ("lm", "logit", or "felm")
    formula : str
        Formula for the model
    labeled : str
        Name of labeled indicator column
    sample_prob : str
        Name of sampling probability column
    predicted_var : List[str]
        List of predicted variable names
    data_orig : pd.DataFrame
        Original data
    data_pred : pd.DataFrame
        Predicted data
    index : Optional[List[str]], optional
        List of index variables for fixed effects, by default None
    fixed_effect : Optional[str], optional
        Type of fixed effects, by default None
    clustered : bool, optional
        Whether to use clustered standard errors, by default False
    cluster : Optional[str], optional
        Name of cluster variable, by default None

    Returns
    -------
    Dict[str, Any]
        Dictionary containing estimation results
    """
    # Import patsy for formula parsing
    import patsy

    # Parse formula
    y_orig, X_orig = patsy.dmatrices(formula, data_orig, return_type="dataframe")
    y_pred, X_pred = patsy.dmatrices(formula, data_pred, return_type="dataframe")

    # Get labeled indicator and sampling probability
    labeled_ind = data_orig[labeled].values
    sample_prob_use = data_orig[sample_prob].values

    # Get fixed effects if needed
    fe_Y = None
    fe_X = None
    if fixed_effect is not None and index is not None:
        fe_Y = data_orig[index].values
        fe_X = data_pred[index].values

    # Get cluster variable if needed
    cluster_var = None
    if clustered and cluster is not None:
        cluster_var = data_orig[cluster].values

    # Initial parameter estimate
    par_init = np.zeros(X_orig.shape[1])

    # Estimate model
    par_est, info = dsl_general(
        Y_orig=y_orig.values.flatten(),
        X_orig=X_orig.values,
        Y_pred=y_pred.values.flatten(),
        X_pred=X_pred.values,
        labeled_ind=labeled_ind,
        sample_prob_use=sample_prob_use,
        model=model,
        fe_Y=fe_Y,
        fe_X=fe_X,
    )

    # Compute moment decomposition
    main_1, main_23 = dsl_general_moment_base_decomp(
        par=par_est,
        labeled_ind=labeled_ind,
        sample_prob_use=sample_prob_use,
        Y_orig=y_orig.values.flatten(),
        X_orig=X_orig.values,
        Y_pred=y_pred.values.flatten(),
        X_pred=X_pred.values,
        model=model,
        clustered=clustered,
        cluster=cluster_var,
    )

    # Compute Jacobian
    J = dsl_general_Jacobian(
        par=par_est,
        labeled_ind=labeled_ind,
        sample_prob_use=sample_prob_use,
        Y_orig=y_orig.values.flatten(),
        X_orig=X_orig.values,
        Y_pred=y_pred.values.flatten(),
        X_pred=X_pred.values,
        model=model,
    )

    # Compute variance-covariance matrices
    # Note: J here is likely the (n x k) Jacobian from the moment function,
    # not the (k x k) average Jacobian used in dsl_general.
    D = np.linalg.inv(J.T @ J)
    Meat = main_1 + main_23
    vcov = D @ Meat @ D
    vcov0 = D @ main_1 @ D

    # Get column names for output
    coef_names = X_orig.columns

    # Return results
    return {
        "coefficients": pd.Series(par_est, index=coef_names),
        "standard_errors": pd.Series(np.sqrt(np.diag(vcov)), index=coef_names),
        "vcov": pd.DataFrame(vcov, index=coef_names, columns=coef_names),
        "Meat": pd.DataFrame(Meat, index=coef_names, columns=coef_names),
        "Meat_decomp": {
            "main_1": pd.DataFrame(main_1, index=coef_names, columns=coef_names),
            "main_23": pd.DataFrame(main_23, index=coef_names, columns=coef_names),
        },
        "J": pd.DataFrame(J),  # J here might be (n x k)
        "D": pd.DataFrame(D, index=coef_names, columns=coef_names),
        "vcov0": pd.DataFrame(vcov0, index=coef_names, columns=coef_names),
    }


def stable_inverse(A: np.ndarray) -> np.ndarray:
    """
    Compute stable matrix inverse using QR decomposition.
    """
    Q, R = np.linalg.qr(A)
    return np.linalg.solve(R.T @ R, R.T @ Q.T)


def compute_sandwich_var(J, m, n_obs):
    """
    Compute sandwich variance estimator (scaled version).
    Assumes J is (k,k) average Jacobian and m is (n,k) moments.
    vcov = (J^-1) @ Meat @ (J^-1) / n
    Meat = (1/n) * sum(m_i @ m_i.T) (should be k x k)
    """
    n_features = J.shape[0]

    # Compute J inverse (Bread)
    try:
        bread = np.linalg.inv(J)
    except np.linalg.LinAlgError:
        print("Warning: Jacobian is singular, using pseudo-inverse.")
        bread = np.linalg.pinv(J)

    # Compute Meat = (1/n) * sum(m_i @ m_i.T)
    # m has shape (n, k)
    if m.shape[0] != n_obs or m.shape[1] != n_features:
        raise ValueError(
            f"Moment shape mismatch in vcov: Expected ({n_obs},{n_features}), "
            f"got {m.shape}"
        )

    meat = (m.T @ m) / n_obs  # Efficient calculation: (k,n) @ (n,k) -> (k,k)

    # Compute variance-covariance matrix (scaled)
    vcov_scaled = (bread @ meat @ bread.T) / n_obs

    return vcov_scaled
