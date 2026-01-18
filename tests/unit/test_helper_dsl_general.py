"""
Unit tests for the DSL general helper functions using real PanChen data
"""

import numpy as np
import pandas as pd
import pytest
import os
from dsl.helpers.dsl_general import (
    dsl_general_moment_base_decomp,
    dsl_general_Jacobian,
    dsl_general_moment,
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
    """Prepare PanChen data for testing."""
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


def test_dsl_general_moment_base_decomp():
    """Test the dsl_general_moment_base_decomp function with real data"""
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

    # Compute moment decomposition
    main_1, main_23 = dsl_general_moment_base_decomp(
        par, labeled_ind, sample_prob_use, Y_orig, X_orig, Y_pred, X_pred, "lm"
    )

    # Check shapes
    assert main_1.shape == (5, 5)
    assert main_23.shape == (5, 5)

    # Check that both components are symmetric
    assert np.allclose(main_1, main_1.T)
    assert np.allclose(main_23, main_23.T)


def test_dsl_general_moment_base_decomp_logit():
    """Test the dsl_general_moment_base_decomp function with logistic regression and real data"""
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

    # Compute moment decomposition
    main_1, main_23 = dsl_general_moment_base_decomp(
        par, labeled_ind, sample_prob_use, Y_orig, X_orig, Y_pred, X_pred, "logit"
    )

    # Check shapes
    assert main_1.shape == (5, 5)
    assert main_23.shape == (5, 5)

    # Check that both components are symmetric
    assert np.allclose(main_1, main_1.T)
    assert np.allclose(main_23, main_23.T)


def test_dsl_general_moment_base_decomp_felm():
    """Test the dsl_general_moment_base_decomp function with fixed effects and real data"""
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

    # Compute moment decomposition
    main_1, main_23 = dsl_general_moment_base_decomp(
        par, labeled_ind, sample_prob_use, Y_orig, X_orig, Y_pred, X_pred, "felm"
    )

    # Check shapes
    assert main_1.shape == (5, 5)
    assert main_23.shape == (5, 5)

    # Check that both components are symmetric
    assert np.allclose(main_1, main_1.T)
    assert np.allclose(main_23, main_23.T)


def test_dsl_general_Jacobian():
    """Test the dsl_general_Jacobian function with real data"""
    # Load real data
    data = load_panchen_data()
    df = prepare_test_data(data, n_samples=100)

    # Extract features and outcome
    X_orig = df[["countyWrong", "prefecWrong", "connect2b", "prevalence", "regionj"]].values
    Y_orig = df["SendOrNot"].values
    X_pred = X_orig.copy()
    Y_pred = Y_orig.copy()

    # Generate parameters
    n_main = X_orig.shape[1]
    par = np.array([0.1, -0.2, 0.05, -0.1, 0.15])

    # Generate data
    labeled_ind = df["labeled"].values
    sample_prob_use = df["sample_prob"].values

    # Generate fixed effects for felm model
    n_fixed_effects = 3
    fe_Y = df["groupIssue"].values if "groupIssue" in df.columns else np.random.randn(len(df))
    fe_X = np.random.randn(len(df), n_fixed_effects)

    # Test with different models
    for model in ["lm", "logit", "felm"]:
        # Compute Jacobian (provide fe_Y and fe_X for felm)
        if model == "felm":
            J = dsl_general_Jacobian(
                par, labeled_ind, sample_prob_use, Y_orig, X_orig, Y_pred, X_pred,
                model, fe_Y=fe_Y, fe_X=fe_X
            )
        else:
            J = dsl_general_Jacobian(
                par, labeled_ind, sample_prob_use, Y_orig, X_orig, Y_pred, X_pred, model
            )

        # For all models, Jacobian should be (n_main, n_main) - the same shape as the moment dimensions
        expected_shape = (n_main, n_main)

        # Check shape
        assert J.shape == expected_shape, f"Model {model}: expected {expected_shape}, got {J.shape}"

        # Check that Jacobian is symmetric
        assert np.allclose(J, J.T), f"Model {model}: Jacobian is not symmetric"


def test_dsl_general_moment():
    """Test the dsl_general_moment function with real data"""
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

    # Test with different models
    for model in ["lm", "logit", "felm"]:
        # Compute moment function
        g_out = dsl_general_moment(
            par,
            labeled_ind,
            sample_prob_use,
            Y_orig,
            X_orig,
            Y_pred,
            X_pred,
            fe_Y,
            fe_X,
            model,
            0.00001,
        )

        # Check that g_out is a scalar
        assert isinstance(g_out, float)
        assert g_out >= 0
