"""
Functional tests for the main DSL function
"""

import numpy as np
from patsy import dmatrices

from dsl.dsl import dsl


def test_dsl_linear_regression(sample_data, sample_prediction):
    """Test DSL with linear regression"""
    # Extract labeled indicator
    labeled_ind = sample_data["labeled"].values

    # Create design matrices from formula
    y_mat, X_mat = dmatrices("y ~ x1 + x2 + x3 + x4 + x5", sample_data, return_type="dataframe")

    # Run DSL
    result = dsl(
        X=X_mat.values,
        y=y_mat.values.flatten(),
        labeled_ind=labeled_ind,
        sample_prob=sample_data["sample_prob"].values,
        model="lm",
        method="linear",
    )

    # Check result
    assert result is not None
    assert hasattr(result, "coefficients")
    assert hasattr(result, "standard_errors")
    assert hasattr(result, "predicted_values")
    assert hasattr(result, "residuals")
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


def test_dsl_logistic_regression(sample_data_logit, sample_prediction_logit):
    """Test DSL with logistic regression"""
    # Extract labeled indicator
    labeled_ind = sample_data_logit["labeled"].values

    # Create design matrices from formula
    y_mat, X_mat = dmatrices("y ~ x1 + x2 + x3 + x4 + x5", sample_data_logit, return_type="dataframe")

    # Run DSL
    result = dsl(
        X=X_mat.values,
        y=y_mat.values.flatten(),
        labeled_ind=labeled_ind,
        sample_prob=sample_data_logit["sample_prob"].values,
        model="logit",
        method="logistic",
    )

    # Check result
    assert result is not None
    assert hasattr(result, "coefficients")
    assert hasattr(result, "standard_errors")
    assert hasattr(result, "predicted_values")
    assert hasattr(result, "residuals")
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


def test_dsl_fixed_effects(sample_data, sample_prediction):
    """Test DSL with fixed effects"""
    # Extract labeled indicator
    labeled_ind = sample_data["labeled"].values

    # Create design matrices from formula (basic formula, fixed effects handling may vary)
    y_mat, X_mat = dmatrices("y ~ x1 + x2 + x3 + x4 + x5", sample_data, return_type="dataframe")

    # Run DSL with fixed_effects method
    result = dsl(
        X=X_mat.values,
        y=y_mat.values.flatten(),
        labeled_ind=labeled_ind,
        sample_prob=sample_data["sample_prob"].values,
        model="felm",
        method="fixed_effects",
    )

    # Check result
    assert result is not None
    assert hasattr(result, "coefficients")
    assert hasattr(result, "standard_errors")
    assert hasattr(result, "predicted_values")
    assert hasattr(result, "residuals")
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


def test_dsl_without_prediction(sample_data):
    """Test DSL without providing predictions"""
    # Extract labeled indicator
    labeled_ind = sample_data["labeled"].values

    # Create design matrices from formula
    y_mat, X_mat = dmatrices("y ~ x1 + x2 + x3 + x4 + x5", sample_data, return_type="dataframe")

    # Run DSL
    result = dsl(
        X=X_mat.values,
        y=y_mat.values.flatten(),
        labeled_ind=labeled_ind,
        sample_prob=sample_data["sample_prob"].values,
        model="lm",
        method="linear",
    )

    # Check result
    assert result is not None
    assert hasattr(result, "coefficients")
    assert hasattr(result, "standard_errors")
    assert hasattr(result, "predicted_values")
    assert hasattr(result, "residuals")
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


def test_dsl_without_labeled(sample_data, sample_prediction):
    """Test DSL without providing labeled indicator"""
    # Use all observations as labeled
    labeled_ind = np.ones(len(sample_data))

    # Create design matrices from formula
    y_mat, X_mat = dmatrices("y ~ x1 + x2 + x3 + x4 + x5", sample_data, return_type="dataframe")

    # Run DSL
    result = dsl(
        X=X_mat.values,
        y=y_mat.values.flatten(),
        labeled_ind=labeled_ind,
        sample_prob=sample_data["sample_prob"].values,
        model="lm",
        method="linear",
    )

    # Check result
    assert result is not None
    assert hasattr(result, "coefficients")
    assert hasattr(result, "standard_errors")
    assert hasattr(result, "predicted_values")
    assert hasattr(result, "residuals")
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


def test_dsl_without_sample_prob(sample_data, sample_prediction):
    """Test DSL without providing sample probabilities"""
    # Extract labeled indicator
    labeled_ind = sample_data["labeled"].values

    # Use uniform sampling probability
    sample_prob = np.ones(len(sample_data)) * (labeled_ind.sum() / len(sample_data))

    # Create design matrices from formula
    y_mat, X_mat = dmatrices("y ~ x1 + x2 + x3 + x4 + x5", sample_data, return_type="dataframe")

    # Run DSL
    result = dsl(
        X=X_mat.values,
        y=y_mat.values.flatten(),
        labeled_ind=labeled_ind,
        sample_prob=sample_prob,
        model="lm",
        method="linear",
    )

    # Check result
    assert result is not None
    assert hasattr(result, "coefficients")
    assert hasattr(result, "standard_errors")
    assert hasattr(result, "predicted_values")
    assert hasattr(result, "residuals")
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
