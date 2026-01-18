"""
Unit tests for the moment estimation helper functions using real PanChen data
"""

import numpy as np
import pandas as pd
import pytest
import os
from dsl.helpers.moment import (
    demean_dsl,
    felm_dsl_moment_base,
    lm_dsl_Jacobian,
    lm_dsl_moment_base,
    lm_dsl_moment_orig,
    lm_dsl_moment_pred,
    logit_dsl_Jacobian,
    logit_dsl_moment_base,
    logit_dsl_moment_orig,
    logit_dsl_moment_pred,
)


def load_panchen_data():
    """Load real PanChen data for testing."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    panchen_dir = os.path.join(script_dir, "..", "..", "PanChen_test")
    file_path = os.path.join(panchen_dir, "PanChen.parquet")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PanChen data not found at {file_path}")
    
    data = pd.read_parquet(file_path)
    return data


def prepare_test_data(data, n_samples=100):
    """Prepare PanChen data for moment testing."""
    df = data.copy()
    
    # Use complete cases only
    complete_cases = df.dropna(subset=["countyWrong", "SendOrNot"])
    
    # Sample for testing
    if len(complete_cases) > n_samples:
        df = complete_cases.sample(n=n_samples, random_state=123)
    else:
        df = complete_cases
    
    # Create labeled indicator
    np.random.seed(123)
    n_labeled = min(80, len(df))
    labeled_indices = np.random.choice(df.index, size=n_labeled, replace=False)
    
    df["labeled"] = 0
    df.loc[labeled_indices, "labeled"] = 1
    
    # Create sample probability
    sample_prob = n_labeled / len(df)
    df["sample_prob"] = sample_prob
    
    # Handle missing values
    for col in ["countyWrong", "prefecWrong", "connect2b", "prevalence", "regionj", "groupIssue"]:
        mask = df["labeled"] == 0
        df.loc[mask, col] = df.loc[mask, col].fillna(0)
    
    df.loc[df["labeled"] == 0, "SendOrNot"] = df.loc[df["labeled"] == 0, "SendOrNot"].fillna(0)
    
    return df


def test_lm_dsl_moment_base():
    """Test the lm_dsl_moment_base function with real data"""
    # Load real data
    data = load_panchen_data()
    df = prepare_test_data(data, n_samples=100)
    
    # Extract features and outcome
    X_orig = df[["countyWrong", "prefecWrong", "connect2b", "prevalence", "regionj"]].values
    Y_orig = df["SendOrNot"].values
    X_pred = X_orig.copy()  # Use same features for prediction
    Y_pred = Y_orig.copy()
    
    # Generate parameters
    par = np.array([0.1, -0.2, 0.05, -0.1, 0.15])
    
    # Generate data
    labeled_ind = df["labeled"].values
    sample_prob_use = df["sample_prob"].values

    # Compute moment function
    m_dr = lm_dsl_moment_base(
        par, labeled_ind, sample_prob_use, Y_orig, X_orig, Y_pred, X_pred
    )

    # Check shape
    assert m_dr.shape == (len(df), 5)

    # For unlabeled rows, DR moment should equal m_pred
    residuals_pred = (Y_pred.flatten() - X_pred @ par).reshape(-1, 1)
    m_pred = X_pred * residuals_pred
    unlabeled_indices = np.where(labeled_ind == 0)[0]
    assert np.allclose(m_dr[unlabeled_indices], m_pred[unlabeled_indices])


def test_lm_dsl_moment_orig():
    """Test the lm_dsl_moment_orig function with real data"""
    # Load real data
    data = load_panchen_data()
    df = prepare_test_data(data, n_samples=100)
    
    # Extract features and outcome
    X_orig = df[["countyWrong", "prefecWrong", "connect2b", "prevalence", "regionj"]].values
    Y_orig = df["SendOrNot"].values
    X_pred = X_orig.copy()
    Y_pred = Y_orig.copy()
    
    # Generate parameters
    par = np.array([0.1, -0.2, 0.05, -0.1, 0.15])
    
    # Generate data
    labeled_ind = df["labeled"].values
    sample_prob_use = df["sample_prob"].values

    # Compute moment function
    m_orig = lm_dsl_moment_orig(
        par, labeled_ind, sample_prob_use, Y_orig, X_orig, Y_pred, X_pred
    )

    # Check shape
    assert m_orig.shape == (len(df), 5)

    # Check that unlabeled observations have zero moment
    unlabeled_indices = np.where(labeled_ind == 0)[0]
    assert np.allclose(m_orig[unlabeled_indices], 0)


def test_lm_dsl_moment_pred():
    """Test the lm_dsl_moment_pred function with real data"""
    # Load real data
    data = load_panchen_data()
    df = prepare_test_data(data, n_samples=100)
    
    # Extract features and outcome
    X_orig = df[["countyWrong", "prefecWrong", "connect2b", "prevalence", "regionj"]].values
    Y_orig = df["SendOrNot"].values
    X_pred = X_orig.copy()
    Y_pred = Y_orig.copy()
    
    # Generate parameters
    par = np.array([0.1, -0.2, 0.05, -0.1, 0.15])
    
    # Generate data
    labeled_ind = df["labeled"].values
    sample_prob_use = df["sample_prob"].values

    # Compute moment function
    m_pred = lm_dsl_moment_pred(
        par, labeled_ind, sample_prob_use, Y_orig, X_orig, Y_pred, X_pred
    )

    # Check shape
    assert m_pred.shape == (len(df), 5)


def test_lm_dsl_Jacobian():
    """Test the lm_dsl_Jacobian function with real data"""
    # Load real data
    data = load_panchen_data()
    df = prepare_test_data(data, n_samples=100)
    
    # Extract features and outcome
    X_orig = df[["countyWrong", "prefecWrong", "connect2b", "prevalence", "regionj"]].values
    Y_orig = df["SendOrNot"].values
    X_pred = X_orig.copy()
    Y_pred = Y_orig.copy()
    
    # Generate parameters
    par = np.array([0.1, -0.2, 0.05, -0.1, 0.15])
    
    # Generate data
    labeled_ind = df["labeled"].values
    sample_prob_use = df["sample_prob"].values

    # Compute Jacobian
    J = lm_dsl_Jacobian(
        par, labeled_ind, sample_prob_use, Y_orig, X_orig, Y_pred, X_pred, "lm"
    )

    # Check shape
    assert J.shape == (5, 5)

    # Check that Jacobian is symmetric
    assert np.allclose(J, J.T)


def test_logit_dsl_moment_base():
    """Test the logit_dsl_moment_base function with real data"""
    # Load real data
    data = load_panchen_data()
    df = prepare_test_data(data, n_samples=100)
    
    # Extract features and outcome
    X_orig = df[["countyWrong", "prefecWrong", "connect2b", "prevalence", "regionj"]].values
    Y_orig = df["SendOrNot"].values
    X_pred = X_orig.copy()
    Y_pred = Y_orig.copy()
    
    # Generate parameters
    par = np.array([0.1, -0.2, 0.05, -0.1, 0.15])
    
    # Generate data
    labeled_ind = df["labeled"].values
    sample_prob_use = df["sample_prob"].values

    # Compute moment function
    m_dr = logit_dsl_moment_base(
        par, labeled_ind, sample_prob_use, Y_orig, X_orig, Y_pred, X_pred
    )

    # Check shape
    assert m_dr.shape == (len(df), 5)

    # For unlabeled rows, DR moment should equal m_pred
    p_pred = 1 / (1 + np.exp(-(X_pred @ par)))
    residuals_pred = (Y_pred.flatten() - p_pred).reshape(-1, 1)
    m_pred = X_pred * residuals_pred
    unlabeled_indices = np.where(labeled_ind == 0)[0]
    assert np.allclose(m_dr[unlabeled_indices], m_pred[unlabeled_indices])


def test_logit_dsl_moment_orig():
    """Test the logit_dsl_moment_orig function with real data"""
    # Load real data
    data = load_panchen_data()
    df = prepare_test_data(data, n_samples=100)
    
    # Extract features and outcome
    X_orig = df[["countyWrong", "prefecWrong", "connect2b", "prevalence", "regionj"]].values
    Y_orig = df["SendOrNot"].values
    X_pred = X_orig.copy()
    Y_pred = Y_orig.copy()
    
    # Generate parameters
    par = np.array([0.1, -0.2, 0.05, -0.1, 0.15])
    
    # Generate data
    labeled_ind = df["labeled"].values
    sample_prob_use = df["sample_prob"].values

    # Compute moment function
    m_orig = logit_dsl_moment_orig(
        par, labeled_ind, sample_prob_use, Y_orig, X_orig, Y_pred, X_pred
    )

    # Check shape
    assert m_orig.shape == (len(df), 5)

    # Check that unlabeled observations have zero moment
    unlabeled_indices = np.where(labeled_ind == 0)[0]
    assert np.allclose(m_orig[unlabeled_indices], 0)


def test_logit_dsl_moment_pred():
    """Test the logit_dsl_moment_pred function with real data"""
    # Load real data
    data = load_panchen_data()
    df = prepare_test_data(data, n_samples=100)
    
    # Extract features and outcome
    X_orig = df[["countyWrong", "prefecWrong", "connect2b", "prevalence", "regionj"]].values
    Y_orig = df["SendOrNot"].values
    X_pred = X_orig.copy()
    Y_pred = Y_orig.copy()
    
    # Generate parameters
    par = np.array([0.1, -0.2, 0.05, -0.1, 0.15])
    
    # Generate data
    labeled_ind = df["labeled"].values
    sample_prob_use = df["sample_prob"].values

    # Compute moment function
    m_pred = logit_dsl_moment_pred(
        par, labeled_ind, sample_prob_use, Y_orig, X_orig, Y_pred, X_pred
    )

    # Check shape
    assert m_pred.shape == (len(df), 5)


def test_logit_dsl_Jacobian():
    """Test the logit_dsl_Jacobian function with real data"""
    # Load real data
    data = load_panchen_data()
    df = prepare_test_data(data, n_samples=100)
    
    # Extract features and outcome
    X_orig = df[["countyWrong", "prefecWrong", "connect2b", "prevalence", "regionj"]].values
    Y_orig = df["SendOrNot"].values
    X_pred = X_orig.copy()
    Y_pred = Y_orig.copy()
    
    # Generate parameters
    par = np.array([0.1, -0.2, 0.05, -0.1, 0.15])
    
    # Generate data
    labeled_ind = df["labeled"].values
    sample_prob_use = df["sample_prob"].values

    # Compute Jacobian
    J = logit_dsl_Jacobian(
        par, labeled_ind, sample_prob_use, Y_orig, X_orig, Y_pred, X_pred, "logit"
    )

    # Check shape
    assert J.shape == (5, 5)

    # Check that Jacobian is symmetric
    assert np.allclose(J, J.T)


def test_felm_dsl_moment_base():
    """Test the felm_dsl_moment_base function with real data"""
    # Load real data
    data = load_panchen_data()
    df = prepare_test_data(data, n_samples=100)

    # Extract features and outcome
    X_orig = df[["countyWrong", "prefecWrong", "connect2b", "prevalence", "regionj"]].values
    Y_orig = df["SendOrNot"].values
    X_pred = X_orig.copy()
    Y_pred = Y_orig.copy()

    # Generate parameters
    par = np.array([0.1, -0.2, 0.05, -0.1, 0.15])

    # Generate data
    labeled_ind = df["labeled"].values
    sample_prob_use = df["sample_prob"].values

    # Generate fixed effects using real data
    n_fixed_effects = 5
    fe_Y = df["groupIssue"].values if "groupIssue" in df.columns else np.random.randn(len(df))
    fe_X = np.random.randn(len(df), n_fixed_effects)

    # Compute moment function
    m_dr = felm_dsl_moment_base(
        par, labeled_ind, sample_prob_use, Y_orig, X_orig, Y_pred, X_pred, fe_Y, fe_X
    )

    # Check shape
    assert m_dr.shape == (len(df), 5)

    # For unlabeled rows, DR moment should equal the FE-adjusted m_pred
    # Match the implementation: when par_fe is empty, fe_use is zeros
    n_main = X_orig.shape[1]
    par_main = par[:n_main]
    par_fe = par[n_main:]
    if par_fe.size == 0:
        # When no fixed effect parameters, implementation uses zeros
        fe_use = np.zeros(len(df))
    else:
        # fe_use = fe_X @ par_fe (1D array)
        fe_use = (fe_X @ par_fe).flatten()
    residuals_pred = (Y_pred - X_pred @ par_main - fe_use).reshape(-1, 1)
    m_pred = X_pred * residuals_pred

    unlabeled_indices = np.where(labeled_ind == 0)[0]
    assert np.allclose(m_dr[unlabeled_indices], m_pred[unlabeled_indices])


def test_demean_dsl():
    """Test the demean_dsl function with real data"""
    # Load real data
    data = load_panchen_data()
    df = prepare_test_data(data, n_samples=100)

    # Reset index to avoid index alignment issues
    df = df.reset_index(drop=True)

    # Use real group data for fixed effects
    data_base = df[["groupIssue"]].copy()
    data_base["id"] = range(len(data_base))

    # Generate outcome and features using real data
    adj_Y = df["SendOrNot"].values
    adj_X = df[["countyWrong", "prefecWrong", "connect2b", "prevalence", "regionj"]].values

    # Compute demeaned data
    adj_Y_avg_exp, adj_X_avg_exp, adj_data = demean_dsl(
        data_base, adj_Y, adj_X, ["groupIssue"]
    )

    # Check shapes
    assert adj_Y_avg_exp.shape == (len(df),)
    assert adj_X_avg_exp.shape == (len(df), 5)
    assert adj_data.shape == (len(df), 7)  # id + groupIssue + 5 features

    # Check that demeaned data has mean zero within each group
    # Convert boolean series to numpy array for proper indexing
    for group in data_base["groupIssue"].unique():
        group_indices = (data_base["groupIssue"] == group).values
        assert np.allclose(np.mean(adj_Y_avg_exp[group_indices]), 0, atol=1e-10)
        assert np.allclose(np.mean(adj_X_avg_exp[group_indices], axis=0), 0, atol=1e-10)
