"""
Unit tests for the estimate helper functions using real PanChen data
"""

import numpy as np
import pandas as pd
import pytest
import os
from dsl.helpers.estimate import (
    estimate_supervised,
    estimate_fixed_effects,
    available_method,
    fit_model,
    fit_test,
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


def test_estimate_supervised():
    """Test the estimate_supervised function with real data"""
    # Load real data
    data = load_panchen_data()
    df = prepare_test_data(data, n_samples=100)
    
    # Extract features and outcome
    X = df[["countyWrong", "prefecWrong", "connect2b", "prevalence", "regionj"]].values
    y = df["SendOrNot"].values

    # Test with linear method
    y_pred, se = estimate_supervised(y, X, method="linear")

    # Check shapes
    assert y_pred.shape == y.shape
    assert se.shape == (X.shape[1],)

    # Check that predictions are reasonable
    assert np.all(np.isfinite(y_pred))
    assert np.all(np.isfinite(se))

    # Check that standard errors are positive
    assert np.all(se > 0)


def test_estimate_supervised_logistic():
    """Test the estimate_supervised function with logistic regression and real data"""
    # Load real data
    data = load_panchen_data()
    df = prepare_test_data(data, n_samples=100)
    
    # Extract features and outcome
    X = df[["countyWrong", "prefecWrong", "connect2b", "prevalence", "regionj"]].values
    y = df["SendOrNot"].values

    # Test with logistic method
    y_pred, se = estimate_supervised(y, X, method="logistic")

    # Check shapes
    assert y_pred.shape == y.shape
    assert se.shape == (X.shape[1],)

    # Check that predictions are reasonable (between 0 and 1 for logistic)
    assert np.all(np.isfinite(y_pred))
    assert np.all(y_pred >= 0)
    assert np.all(y_pred <= 1)
    assert np.all(np.isfinite(se))

    # Check that standard errors are positive
    assert np.all(se > 0)


def test_estimate_fixed_effects():
    """Test the estimate_fixed_effects function with real data"""
    # Load real data
    data = load_panchen_data()
    df = prepare_test_data(data, n_samples=100)

    # Extract features and outcome
    X = df[["countyWrong", "prefecWrong", "connect2b", "prevalence", "regionj"]].values
    y = df["SendOrNot"].values

    # Create fixed effects using real data
    fe = df["groupIssue"].values if "groupIssue" in df.columns else np.random.randint(0, 5, len(df))

    # Test fixed effects estimation (returns y_pred, se, fe_pred)
    y_pred, se, fe_pred = estimate_fixed_effects(y, X, fe)

    # Check shapes
    assert y_pred.shape == y.shape
    assert se.shape == (X.shape[1],)
    assert fe_pred.shape == y.shape

    # Check that predictions are reasonable
    assert np.all(np.isfinite(y_pred))
    assert np.all(np.isfinite(se))
    assert np.all(np.isfinite(fe_pred))

    # Check that standard errors are positive
    assert np.all(se > 0)


def test_fit_model():
    """Test the fit_model function with real data"""
    # Load real data
    data = load_panchen_data()
    df = prepare_test_data(data, n_samples=100)
    
    # Create test data
    data = pd.DataFrame({
        "y": df["SendOrNot"].values,
        "labeled": df["labeled"].values,
        "sample_prob": df["sample_prob"].values,
    })

    # Add features from real data
    data["x1"] = df["countyWrong"].values
    data["x2"] = df["prefecWrong"].values
    data["x3"] = df["connect2b"].values
    data["x4"] = df["prevalence"].values
    data["x5"] = df["regionj"].values

    # Create outcome, labeled, and covariates
    outcome = "y"
    labeled = "labeled"
    covariates = ["x1", "x2", "x3", "x4", "x5"]

    # Test with linear method
    fit_out = fit_model(
        outcome=outcome,
        labeled=labeled,
        covariates=covariates,
        data=data,
        method="linear",
        sample_prob="sample_prob",
        family="gaussian",
    )

    # Check that fit_out is not None
    assert fit_out is not None


def test_fit_test():
    """Test the fit_test function with real data"""
    # Load real data
    data = load_panchen_data()
    df = prepare_test_data(data, n_samples=100)

    # Create test data
    test_data = pd.DataFrame({
        "y": df["SendOrNot"].values,
        "labeled": df["labeled"].values,
        "sample_prob": df["sample_prob"].values,
    })

    # Add features from real data
    test_data["x1"] = df["countyWrong"].values
    test_data["x2"] = df["prefecWrong"].values
    test_data["x3"] = df["connect2b"].values
    test_data["x4"] = df["prevalence"].values
    test_data["x5"] = df["regionj"].values

    # Create outcome, labeled, and covariates
    outcome = "y"
    labeled = "labeled"
    covariates = ["x1", "x2", "x3", "x4", "x5"]

    # First fit the model to get a proper fit_out
    fit_out = fit_model(
        outcome=outcome,
        labeled=labeled,
        covariates=covariates,
        data=test_data,
        method="linear",
        sample_prob="sample_prob",
        family="gaussian",
    )

    # Test with linear method
    Y_hat, RMSE = fit_test(
        fit_out=fit_out,
        outcome=outcome,
        labeled=labeled,
        covariates=covariates,
        data=test_data,
        method="linear",
        family="gaussian",
    )

    # Check shapes - should match the number of labeled observations
    n_labeled = int(df["labeled"].sum())
    assert Y_hat.shape == (n_labeled,)
    assert isinstance(RMSE, float)
    assert RMSE >= 0


def test_available_method():
    """Test the available_method function"""
    # Get available methods
    methods = available_method(print_out=False)

    # Check that methods is a list
    assert isinstance(methods, list)

    # Check that "linear" is in the list
    assert "linear" in methods
