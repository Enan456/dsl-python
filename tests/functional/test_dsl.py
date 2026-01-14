"""
Functional tests for the main DSL function
"""

import numpy as np
from patsy import dmatrices

from dsl.dsl import dsl


def test_dsl_linear_regression(sample_data):
    """Test DSL with linear regression"""
    # Use patsy to create design matrix from formula
    formula = "y ~ x1 + x2 + x3 + x4 + x5"
    y, X = dmatrices(formula, sample_data, return_type="dataframe")

    # Run DSL
    result = dsl(
        X=X.values,
        y=y.values.flatten(),
        labeled_ind=sample_data["labeled"].values,
        sample_prob=sample_data["sample_prob"].values,
        model="lm",
        method="linear",
    )

    # Check result
    assert result is not None
    assert hasattr(result, "coefficients")
    assert hasattr(result, "standard_errors")
    assert hasattr(result, "vcov")
    assert hasattr(result, "objective")
    assert hasattr(result, "success")
    assert hasattr(result, "message")
    assert hasattr(result, "niter")
    assert hasattr(result, "model")
    assert hasattr(result, "labeled_size")
    assert hasattr(result, "total_size")

    # Check shapes
    assert result.coefficients.shape == (6,)  # 5 features + intercept
    assert result.standard_errors.shape == (6,)
    assert result.vcov.shape == (6, 6)
    assert isinstance(result.objective, float)
    assert isinstance(result.success, bool)
    assert isinstance(result.message, str)
    assert isinstance(result.niter, int)
    assert isinstance(result.model, str)
    assert isinstance(result.labeled_size, int)
    assert isinstance(result.total_size, int)
    assert result.labeled_size <= result.total_size


def test_dsl_logistic_regression(sample_data_logit):
    """Test DSL with logistic regression"""
    # Use patsy to create design matrix from formula
    formula = "y ~ x1 + x2 + x3 + x4 + x5"
    y, X = dmatrices(formula, sample_data_logit, return_type="dataframe")

    # Run DSL
    result = dsl(
        X=X.values,
        y=y.values.flatten(),
        labeled_ind=sample_data_logit["labeled"].values,
        sample_prob=sample_data_logit["sample_prob"].values,
        model="logit",
        method="logistic",
    )

    # Check result
    assert result is not None
    assert hasattr(result, "coefficients")
    assert hasattr(result, "standard_errors")
    assert hasattr(result, "vcov")
    assert hasattr(result, "objective")
    assert hasattr(result, "success")
    assert hasattr(result, "message")
    assert hasattr(result, "niter")
    assert hasattr(result, "model")
    assert hasattr(result, "labeled_size")
    assert hasattr(result, "total_size")

    # Check shapes
    assert result.coefficients.shape == (6,)  # 5 features + intercept
    assert result.standard_errors.shape == (6,)
    assert result.vcov.shape == (6, 6)
    assert isinstance(result.objective, float)
    assert isinstance(result.success, bool)
    assert isinstance(result.message, str)
    assert isinstance(result.niter, int)
    assert isinstance(result.model, str)
    assert isinstance(result.labeled_size, int)
    assert isinstance(result.total_size, int)
    assert result.labeled_size <= result.total_size


def test_dsl_simple_linear():
    """Test DSL with simple linear regression using numpy arrays"""
    n_samples = 100
    n_features = 3

    # Generate simple data
    np.random.seed(1234)
    X = np.column_stack([np.ones(n_samples), np.random.randn(n_samples, n_features)])
    true_beta = np.array([1.0, 0.5, -0.3, 0.2])
    y = X @ true_beta + np.random.randn(n_samples) * 0.1

    # Create labeled indicator (80% labeled)
    labeled_ind = np.random.binomial(1, 0.8, n_samples)

    # Create sampling probability
    sample_prob = np.ones(n_samples) * 0.8

    # Run DSL
    result = dsl(
        X=X,
        y=y,
        labeled_ind=labeled_ind,
        sample_prob=sample_prob,
        model="lm",
        method="linear",
    )

    # Check result
    assert result is not None
    assert result.coefficients.shape == (4,)
    assert result.standard_errors.shape == (4,)
    assert result.vcov.shape == (4, 4)
    assert result.success
    assert abs(result.objective) < 1e-6  # Should converge to near 0


def test_dsl_simple_logistic():
    """Test DSL with simple logistic regression using numpy arrays"""
    n_samples = 100
    n_features = 3

    # Generate simple data
    np.random.seed(1234)
    X = np.column_stack([np.ones(n_samples), np.random.randn(n_samples, n_features)])
    true_beta = np.array([1.0, 0.5, -0.3, 0.2])
    logit = 1 / (1 + np.exp(-(X @ true_beta)))
    y = np.random.binomial(1, logit, n_samples)

    # Create labeled indicator (80% labeled)
    labeled_ind = np.random.binomial(1, 0.8, n_samples)

    # Create sampling probability
    sample_prob = np.ones(n_samples) * 0.8

    # Run DSL
    result = dsl(
        X=X,
        y=y,
        labeled_ind=labeled_ind,
        sample_prob=sample_prob,
        model="logit",
        method="logistic",
    )

    # Check result
    assert result is not None
    assert result.coefficients.shape == (4,)
    assert result.standard_errors.shape == (4,)
    assert result.vcov.shape == (4, 4)
    assert result.success
    assert abs(result.objective) < 1e-6  # Should converge to near 0


def test_dsl_all_labeled():
    """Test DSL when all observations are labeled"""
    n_samples = 100
    n_features = 3

    # Generate simple data
    np.random.seed(1234)
    X = np.column_stack([np.ones(n_samples), np.random.randn(n_samples, n_features)])
    true_beta = np.array([1.0, 0.5, -0.3, 0.2])
    y = X @ true_beta + np.random.randn(n_samples) * 0.1

    # All observations are labeled
    labeled_ind = np.ones(n_samples)
    sample_prob = np.ones(n_samples)

    # Run DSL
    result = dsl(
        X=X,
        y=y,
        labeled_ind=labeled_ind,
        sample_prob=sample_prob,
        model="lm",
        method="linear",
    )

    # Check result
    assert result is not None
    assert result.labeled_size == n_samples
    assert result.total_size == n_samples
    assert result.success
    assert abs(result.objective) < 1e-6
