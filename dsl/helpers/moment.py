"""
Moment estimation helper functions for DSL
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def lm_dsl_moment_base(
    par: np.ndarray,
    labeled_ind: np.ndarray,
    sample_prob_use: np.ndarray,
    Y_orig: np.ndarray,
    X_orig: np.ndarray,
    Y_pred: np.ndarray,
    X_pred: np.ndarray,
) -> np.ndarray:
    """
    Base moment function for linear regression.
    Returns moments with shape (n_obs, n_features).

    Parameters
    ----------
    par : np.ndarray
        Parameter vector
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

    Returns
    -------
    np.ndarray
        Moment function value (n_obs, n_features)
    """
    # Original moment - element-wise multiplication of X_orig and residuals
    # Ensure Y_orig is flattened for correct subtraction
    residuals_orig = (Y_orig.flatten() - X_orig @ par).reshape(-1, 1)
    # Broadcasting handles element-wise multiplication
    m_orig = X_orig * residuals_orig

    # Zero out unlabeled observations
    m_orig[labeled_ind == 0] = 0

    # Predicted moment - element-wise multiplication of X_pred and residuals
    # Ensure Y_pred is flattened for correct subtraction
    residuals_pred = (Y_pred.flatten() - X_pred @ par).reshape(-1, 1)
    # Broadcasting handles element-wise multiplication
    m_pred = X_pred * residuals_pred

    # Combined moment
    weights = (labeled_ind / sample_prob_use).reshape(-1, 1)
    m_dr = m_pred + (m_orig - m_pred) * weights

    # Return moments (n_obs, n_features)
    return m_dr


def lm_dsl_moment_orig(
    par: np.ndarray,
    labeled_ind: np.ndarray,
    sample_prob_use: np.ndarray,
    Y_orig: np.ndarray,
    X_orig: np.ndarray,
    Y_pred: np.ndarray,
    X_pred: np.ndarray,
) -> np.ndarray:
    """
    Original moment function for linear regression.
    """
    residuals_orig = (Y_orig.flatten() - X_orig @ par).reshape(-1, 1)
    m_orig = X_orig * residuals_orig
    m_orig[labeled_ind == 0] = 0
    return m_orig


def lm_dsl_moment_pred(
    par: np.ndarray,
    labeled_ind: np.ndarray,
    sample_prob_use: np.ndarray,
    Y_orig: np.ndarray,
    X_orig: np.ndarray,
    Y_pred: np.ndarray,
    X_pred: np.ndarray,
) -> np.ndarray:
    """
    Predicted moment function for linear regression.
    """
    residuals_pred = (Y_pred.flatten() - X_pred @ par).reshape(-1, 1)
    m_pred = X_pred * residuals_pred
    return m_pred


def lm_dsl_Jacobian(
    par: np.ndarray,
    labeled_ind: np.ndarray,
    sample_prob_use: np.ndarray,
    Y_orig: np.ndarray,
    X_orig: np.ndarray,
    Y_pred: np.ndarray,
    X_pred: np.ndarray,
    model: str,
) -> np.ndarray:
    """
    Jacobian for linear regression.

    Parameters
    ----------
    par : np.ndarray
        Parameter vector
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
    model : str
        Model type

    Returns
    -------
    np.ndarray
        Jacobian matrix
    """
    # Zero out unlabeled observations in X_orig
    X_orig = X_orig.copy()
    X_orig[labeled_ind == 0] = 0

    # Convert to sparse matrices for efficiency
    X_orig = csr_matrix(X_orig)
    X_pred = csr_matrix(X_pred)

    # Create diagonal matrices
    diag_1 = csr_matrix(
        (
            labeled_ind / sample_prob_use,
            (np.arange(len(labeled_ind)), np.arange(len(labeled_ind))),
        )
    )
    diag_2 = csr_matrix(
        (
            1 - labeled_ind / sample_prob_use,
            (np.arange(len(labeled_ind)), np.arange(len(labeled_ind))),
        )
    )

    # Compute Jacobian following R's implementation
    term1 = X_orig.T @ diag_1 @ X_orig
    term2 = X_pred.T @ diag_2 @ X_pred
    J = (term1 + term2) / X_orig.shape[0]

    return J.toarray()


def logit_dsl_moment_base(
    par: np.ndarray,
    labeled_ind: np.ndarray,
    sample_prob_use: np.ndarray,
    Y_orig: np.ndarray,
    X_orig: np.ndarray,
    Y_pred: np.ndarray,
    X_pred: np.ndarray,
) -> np.ndarray:
    """
    Base moment function for logistic regression.
    Returns moments with shape (n_obs, n_features).

    Parameters
    ----------
    par : np.ndarray
        Parameter vector
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

    Returns
    -------
    np.ndarray
        Moment function value (n_obs, n_features)
    """
    # Original moment - element-wise multiplication of X_orig and residuals
    p_orig = 1 / (1 + np.exp(-X_orig @ par))
    # Ensure Y_orig is flattened for correct subtraction
    residuals_orig = (Y_orig.flatten() - p_orig).reshape(-1, 1)
    # Broadcasting
    m_orig = X_orig * residuals_orig
    m_orig[labeled_ind == 0] = 0

    # Predicted moment - element-wise multiplication of X_pred and residuals
    p_pred = 1 / (1 + np.exp(-X_pred @ par))
    # Ensure Y_pred is flattened for correct subtraction
    residuals_pred = (Y_pred.flatten() - p_pred).reshape(-1, 1)
    # Broadcasting
    m_pred = X_pred * residuals_pred

    # Combined moment
    weights = (labeled_ind / sample_prob_use).reshape(-1, 1)
    m_dr = m_pred + (m_orig - m_pred) * weights

    # Return moments (n_obs, n_features)
    return m_dr


def logit_dsl_moment_orig(
    par: np.ndarray,
    labeled_ind: np.ndarray,
    sample_prob_use: np.ndarray,
    Y_orig: np.ndarray,
    X_orig: np.ndarray,
    Y_pred: np.ndarray,
    X_pred: np.ndarray,
) -> np.ndarray:
    """
    Original moment function for logistic regression.
    """
    p_orig = 1 / (1 + np.exp(-X_orig @ par))
    residuals_orig = (Y_orig.flatten() - p_orig).reshape(-1, 1)
    m_orig = X_orig * residuals_orig
    m_orig[labeled_ind == 0] = 0
    return m_orig


def logit_dsl_moment_pred(
    par: np.ndarray,
    labeled_ind: np.ndarray,
    sample_prob_use: np.ndarray,
    Y_orig: np.ndarray,
    X_orig: np.ndarray,
    Y_pred: np.ndarray,
    X_pred: np.ndarray,
) -> np.ndarray:
    """
    Predicted moment function for logistic regression.
    """
    p_pred = 1 / (1 + np.exp(-X_pred @ par))
    residuals_pred = (Y_pred.flatten() - p_pred).reshape(-1, 1)
    m_pred = X_pred * residuals_pred
    return m_pred


def logit_dsl_Jacobian(
    par: np.ndarray,
    labeled_ind: np.ndarray,
    sample_prob_use: np.ndarray,
    Y_orig: np.ndarray,
    X_orig: np.ndarray,
    Y_pred: np.ndarray,
    X_pred: np.ndarray,
    model: str,
) -> np.ndarray:
    """
    Average Jacobian (k x k) for logistic regression moments.
    J = E[d(m_i)/d(par)] = - E[w_i * x_i @ x_i.T]
    Estimated as J_hat = - (1/n) * sum(w_dr_i * x_pred_i @ x_pred_i.T)
                 = - (1/n) * X_pred.T @ W_dr @ X_pred

    Parameters
    ----------
    par : np.ndarray
        Parameter vector (k,)
    labeled_ind : np.ndarray
        Labeled indicator (n,)
    sample_prob_use : np.ndarray
        Sampling probability (n,)
    Y_orig : np.ndarray
        Original outcome (flattened) (n,)
    X_orig : np.ndarray
        Original features (n, k)
    Y_pred : np.ndarray
        Predicted outcome (flattened) (n,)
    X_pred : np.ndarray
        Predicted features (n, k)
    model : str
        Model type (unused)

    Returns
    -------
    np.ndarray
        Average Jacobian matrix (k, k)
    """
    n_obs = X_pred.shape[0]
    n_features = X_pred.shape[1]

    # Calculate probabilities and weights p*(1-p)
    p_orig = 1 / (1 + np.exp(-X_orig @ par))
    w_orig = p_orig * (1 - p_orig)
    w_orig[labeled_ind == 0] = 0  # Zero out weights for unlabeled original moments

    p_pred = 1 / (1 + np.exp(-X_pred @ par))
    w_pred = p_pred * (1 - p_pred)

    # Calculate doubly robust weights
    prob_ratio = np.nan_to_num(
        labeled_ind / sample_prob_use, nan=0.0, posinf=0.0, neginf=0.0
    )
    w_dr = w_pred + (w_orig - w_pred) * prob_ratio

    # Construct diagonal weight matrix W_dr
    W_dr_diag = np.diag(w_dr)

    # Compute average Jacobian: J = - (1/n) * X_pred.T @ W_dr @ X_pred
    # Using X_pred consistent with sandwich::vcovGMM which uses model matrix
    J_avg = -(X_pred.T @ W_dr_diag @ X_pred) / n_obs

    if J_avg.shape != (n_features, n_features):
        raise ValueError(
            f"Logit Jacobian shape error: expected ({n_features},{n_features}), "
            f"got {J_avg.shape}"
        )

    return J_avg


def felm_dsl_moment_base(
    par: np.ndarray,
    labeled_ind: np.ndarray,
    sample_prob_use: np.ndarray,
    Y_orig: np.ndarray,
    X_orig: np.ndarray,
    Y_pred: np.ndarray,
    X_pred: np.ndarray,
    fe_Y: np.ndarray,
    fe_X: np.ndarray,
) -> np.ndarray:
    """
    Base moment function for fixed effects regression.

    Parameters
    ----------
    par : np.ndarray
        Parameter vector
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
    fe_Y : np.ndarray
        Fixed effects outcome
    fe_X : np.ndarray
        Fixed effects features

    Returns
    -------
    np.ndarray
        Moment function value
    """
    # Split parameters into main effects and fixed effects
    n_main = X_orig.shape[1]
    par_main = par[:n_main]
    par_fe = par[n_main:]

    # Compute fixed effects
    fe_use = fe_Y - fe_X @ par_fe

    # Original moment with fixed effects
    m_orig = X_orig * (Y_orig - X_orig @ par_main - fe_use).reshape(-1, 1)
    m_orig = m_orig * labeled_ind.reshape(-1, 1)

    # Predicted moment with fixed effects
    m_pred = X_pred * (Y_pred - X_pred @ par_main - fe_use).reshape(-1, 1)

    # Combined moment
    m_dr = m_pred + (m_orig - m_pred) * (labeled_ind / sample_prob_use).reshape(-1, 1)

    # Zero out unlabeled observations
    m_dr = m_dr * labeled_ind.reshape(-1, 1)

    return m_dr


def demean_dsl(
    data_base: pd.DataFrame,
    adj_Y: np.ndarray,
    adj_X: np.ndarray,
    index: List[str],
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Demean data for fixed effects regression.

    Parameters
    ----------
    data_base : pd.DataFrame
        Base data frame
    adj_Y : np.ndarray
        Adjusted outcome
    adj_X : np.ndarray
        Adjusted features
    index : List[str]
        List of fixed effect variables

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, pd.DataFrame]
        Demeaned outcome, features, and data frame
    """
    # Create fixed effect matrix
    fixed_effect_use = pd.get_dummies(data_base[index])

    # Demean outcome and features
    fixed_effect_Y = (
        fixed_effect_use.values
        @ np.linalg.inv(fixed_effect_use.T @ fixed_effect_use)
        @ fixed_effect_use.T
        @ adj_Y
    )

    fixed_effect_X = (
        fixed_effect_use.values
        @ np.linalg.inv(fixed_effect_use.T @ fixed_effect_use)
        @ fixed_effect_use.T
        @ adj_X
    )

    # Create adjusted data frame
    adj_data = pd.DataFrame(
        np.column_stack([data_base[["id"] + index], adj_X]),
        columns=["id"] + index + [f"x{i+1}" for i in range(adj_X.shape[1])],
    )

    return fixed_effect_Y, fixed_effect_X, adj_data


def felm_dsl_Jacobian(
    par: np.ndarray,
    labeled_ind: np.ndarray,
    sample_prob_use: np.ndarray,
    Y_orig: np.ndarray,
    X_orig: np.ndarray,
    Y_pred: np.ndarray,
    X_pred: np.ndarray,
    model: str,
    fe_Y: np.ndarray,
    fe_X: np.ndarray,
) -> np.ndarray:
    """
    Jacobian for fixed effects regression.

    Parameters
    ----------
    par : np.ndarray
        Parameter vector
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
    model : str
        Model type
    fe_Y : np.ndarray
        Fixed effects outcome
    fe_X : np.ndarray
        Fixed effects features

    Returns
    -------
    np.ndarray
        Jacobian matrix
    """
    # Split parameters into main effects and fixed effects
    n_main = X_orig.shape[1]

    # Original moment with fixed effects
    X_o = X_orig.copy()
    X_o = X_o * labeled_ind.reshape(-1, 1)

    # For fixed effects, use sparse matrices
    X_o = csr_matrix(X_o)
    X_p = csr_matrix(X_pred)

    # Compute diagonal matrices
    d1 = np.diag(labeled_ind / sample_prob_use)
    d2 = np.diag(1 - labeled_ind / sample_prob_use)

    # Compute Jacobian for main effects
    J_main = (X_o.T @ d1 @ X_o + X_p.T @ d2 @ X_p) / len(X_orig)

    # Compute Jacobian for fixed effects
    J_fe = fe_X.T @ fe_X / len(X_orig)

    # Combine Jacobians
    J = np.block(
        [
            [J_main, np.zeros((n_main, fe_X.shape[1]))],
            [np.zeros((fe_X.shape[1], n_main)), J_fe],
        ]
    )

    return J
