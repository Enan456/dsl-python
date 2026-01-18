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
    Average Jacobian (k x k) for logistic regression moments using the R-equivalent
    decomposition: J = grad_pred + grad_orig - grad_pred_R, where weights are
    p*(1-p) on the appropriate design matrices.
    """
    n = X_pred.shape[0]

    # Predicted-side weights
    z_pred = X_pred @ par
    exp_neg_pred = np.exp(-z_pred)
    w_pred = exp_neg_pred / (1 + exp_neg_pred) ** 2  # == p_pred * (1 - p_pred)
    W_pred = np.diag(w_pred)
    W_pred_R = np.diag(w_pred * (labeled_ind / sample_prob_use))

    grad_pred = (X_pred.T @ W_pred @ X_pred) / n
    grad_pred_R = (X_pred.T @ W_pred_R @ X_pred) / n

    # Original-side weights (zero-out unlabeled in both X and weights)
    Xo = X_orig.copy()
    Xo[labeled_ind == 0] = 0
    z_orig = Xo @ par
    exp_neg_orig = np.exp(-z_orig)
    w_orig = exp_neg_orig / (1 + exp_neg_orig) ** 2
    w_orig[labeled_ind == 0] = 0
    W_orig_R = np.diag(w_orig * (labeled_ind / sample_prob_use))

    grad_orig = (Xo.T @ W_orig_R @ Xo) / n

    J = grad_pred + grad_orig - grad_pred_R
    return J


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
        Fixed effects outcome (should be 1D array of shape (n,))
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

    # Handle fixed effects - they should contribute as fe_X @ par_fe
    # fe_X has shape (n, n_fe) and par_fe has shape (n_fe,) if present
    if par_fe.size == 0:
        # No fixed effect parameters, use zeros
        fe_use = np.zeros(Y_orig.shape[0])
    else:
        # Compute fixed effects contribution: fe_X @ par_fe produces 1D array of shape (n,)
        fe_use = (fe_X @ par_fe).flatten()

    # Ensure all inputs are properly shaped for broadcasting
    Y_orig_flat = Y_orig.flatten()
    Y_pred_flat = Y_pred.flatten()

    # Original moment with fixed effects
    residuals_orig = (Y_orig_flat - X_orig @ par_main - fe_use).reshape(-1, 1)
    m_orig = X_orig * residuals_orig
    m_orig[labeled_ind == 0] = 0

    # Predicted moment with fixed effects
    residuals_pred = (Y_pred_flat - X_pred @ par_main - fe_use).reshape(-1, 1)
    m_pred = X_pred * residuals_pred

    # Combined moment (doubly robust)
    weights = (labeled_ind / sample_prob_use).reshape(-1, 1)
    m_dr = m_pred + (m_orig - m_pred) * weights

    return m_dr


def felm_dsl_Jacobian(
    par: np.ndarray,
    labeled_ind: np.ndarray,
    sample_prob_use: np.ndarray,
    Y_orig: np.ndarray,
    X_orig: np.ndarray,
    Y_pred: np.ndarray,
    X_pred: np.ndarray,
    model: str,
    fe_Y: np.ndarray = None,
    fe_X: np.ndarray = None,
) -> np.ndarray:
    """
    Jacobian for fixed effects linear regression.

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
    fe_Y : np.ndarray, optional
        Fixed effects outcome
    fe_X : np.ndarray, optional
        Fixed effects features

    Returns
    -------
    np.ndarray
        Jacobian matrix
    """
    # For fixed effects linear model, the Jacobian is the same as the linear model
    # because the moment conditions have the same structure with respect to par
    # Zero out unlabeled observations in X_orig
    X_orig_copy = X_orig.copy()
    X_orig_copy[labeled_ind == 0] = 0

    # Convert to sparse matrices for efficiency
    X_orig_sparse = csr_matrix(X_orig_copy)
    X_pred_sparse = csr_matrix(X_pred)

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
    term1 = X_orig_sparse.T @ diag_1 @ X_orig_sparse
    term2 = X_pred_sparse.T @ diag_2 @ X_pred_sparse
    J = (term1 + term2) / X_orig.shape[0]

    return J.toarray()


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
        Demeaned outcome, demeaned features, and data frame
    """
    n = len(data_base)
    n_features = adj_X.shape[1] if adj_X.ndim > 1 else 1

    # Initialize demeaned arrays
    demeaned_Y = np.zeros(n)
    demeaned_X = np.zeros((n, n_features)) if n_features > 1 else np.zeros(n)

    # Create group variable by combining all index columns
    if len(index) == 1:
        group_var = data_base[index[0]].values
    else:
        # For multiple fixed effects, create a combined group identifier
        group_var = data_base[index].apply(lambda x: tuple(x), axis=1).values

    # Demean within each group
    unique_groups = np.unique(group_var)
    for group in unique_groups:
        mask = group_var == group

        # Compute group means
        Y_group_mean = np.mean(adj_Y[mask])
        X_group_mean = np.mean(adj_X[mask], axis=0) if adj_X.ndim > 1 else np.mean(adj_X[mask])

        # Demean
        demeaned_Y[mask] = adj_Y[mask] - Y_group_mean
        if adj_X.ndim > 1:
            demeaned_X[mask] = adj_X[mask] - X_group_mean
        else:
            demeaned_X[mask] = adj_X[mask] - X_group_mean

    # Create adjusted data frame with demeaned features
    adj_data = pd.DataFrame(
        np.column_stack([data_base[["id"] + index], demeaned_X]),
        columns=["id"] + index + [f"x{i+1}" for i in range(n_features)],
    )

    return demeaned_Y, demeaned_X, adj_data
