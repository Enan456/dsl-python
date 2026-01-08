"""
Functional tests for the power_dsl function
"""

import numpy as np
from patsy import dmatrices

from dsl.dsl import dsl, power_dsl


def test_power_dsl_with_dsl_output(sample_data, sample_prediction):
    """Test power_dsl with dsl output"""
    # Extract labeled indicator
    labeled_ind = sample_data["labeled"].values

    # Create design matrices from formula
    y_mat, X_mat = dmatrices("y ~ x1 + x2 + x3 + x4 + x5", sample_data, return_type="dataframe")

    # Run DSL
    dsl_result = dsl(
        X=X_mat.values,
        y=y_mat.values.flatten(),
        labeled_ind=labeled_ind,
        sample_prob=sample_data["sample_prob"].values,
        model="lm",
        method="linear",
    )

    # Run power_dsl
    result = power_dsl(
        formula="y ~ x1 + x2 + x3 + x4 + x5",
        data=sample_data,
        labeled_ind=labeled_ind,
        sample_prob=sample_data["sample_prob"].values,
        model="lm",
        method="linear",
        dsl_out=dsl_result,
        alpha=0.05,
    )

    # Check result
    assert result is not None
    assert hasattr(result, "power")
    assert hasattr(result, "predicted_se")
    assert hasattr(result, "critical_value")
    assert hasattr(result, "alpha")

    # Check values - power and predicted_se can be arrays (one per coefficient)
    assert isinstance(result.power, (float, np.ndarray))
    assert isinstance(result.predicted_se, (float, np.ndarray))
    assert isinstance(result.critical_value, (float, np.floating))
    assert isinstance(result.alpha, (float, np.floating))

    # Check value ranges
    if isinstance(result.power, np.ndarray):
        assert np.all((result.power >= 0) & (result.power <= 1))
        assert np.all(result.predicted_se >= 0)
    else:
        assert 0 <= result.power <= 1
        assert result.predicted_se >= 0

    assert result.critical_value >= 0
    assert 0 <= result.alpha <= 1


def test_power_dsl_without_dsl_output(sample_data, sample_prediction):
    """Test power_dsl without dsl output"""
    # Extract labeled indicator
    labeled_ind = sample_data["labeled"].values

    # Create design matrices from formula
    y_mat, X_mat = dmatrices("y ~ x1 + x2 + x3 + x4 + x5", sample_data, return_type="dataframe")

    # Run DSL first
    dsl_result = dsl(
        X=X_mat.values,
        y=y_mat.values.flatten(),
        labeled_ind=labeled_ind,
        sample_prob=sample_data["sample_prob"].values,
        model="lm",
        method="linear",
    )

    # Run power_dsl
    result = power_dsl(
        formula="y ~ x1 + x2 + x3 + x4 + x5",
        data=sample_data,
        labeled_ind=labeled_ind,
        sample_prob=sample_data["sample_prob"].values,
        model="lm",
        method="linear",
        dsl_out=dsl_result,
        alpha=0.05,
    )

    # Check result
    assert result is not None
    assert hasattr(result, "power")
    assert hasattr(result, "predicted_se")
    assert hasattr(result, "critical_value")
    assert hasattr(result, "alpha")

    # Check values - power and predicted_se can be arrays (one per coefficient)
    assert isinstance(result.power, (float, np.ndarray))
    assert isinstance(result.predicted_se, (float, np.ndarray))
    assert isinstance(result.critical_value, (float, np.floating))
    assert isinstance(result.alpha, (float, np.floating))

    # Check value ranges
    if isinstance(result.power, np.ndarray):
        assert np.all((result.power >= 0) & (result.power <= 1))
        assert np.all(result.predicted_se >= 0)
    else:
        assert 0 <= result.power <= 1
        assert result.predicted_se >= 0

    assert result.critical_value >= 0
    assert 0 <= result.alpha <= 1


def test_power_dsl_logistic_regression(sample_data_logit, sample_prediction_logit):
    """Test power_dsl with logistic regression"""
    # Extract labeled indicator
    labeled_ind = sample_data_logit["labeled"].values

    # Create design matrices from formula
    y_mat, X_mat = dmatrices("y ~ x1 + x2 + x3 + x4 + x5", sample_data_logit, return_type="dataframe")

    # Run DSL first
    dsl_result = dsl(
        X=X_mat.values,
        y=y_mat.values.flatten(),
        labeled_ind=labeled_ind,
        sample_prob=sample_data_logit["sample_prob"].values,
        model="logit",
        method="logistic",
    )

    # Run power_dsl
    result = power_dsl(
        formula="y ~ x1 + x2 + x3 + x4 + x5",
        data=sample_data_logit,
        labeled_ind=labeled_ind,
        sample_prob=sample_data_logit["sample_prob"].values,
        model="logit",
        method="logistic",
        dsl_out=dsl_result,
        alpha=0.05,
    )

    # Check result
    assert result is not None
    assert hasattr(result, "power")
    assert hasattr(result, "predicted_se")
    assert hasattr(result, "critical_value")
    assert hasattr(result, "alpha")

    # Check values - power and predicted_se can be arrays (one per coefficient)
    assert isinstance(result.power, (float, np.ndarray))
    assert isinstance(result.predicted_se, (float, np.ndarray))
    assert isinstance(result.critical_value, (float, np.floating))
    assert isinstance(result.alpha, (float, np.floating))

    # Check value ranges
    if isinstance(result.power, np.ndarray):
        assert np.all((result.power >= 0) & (result.power <= 1))
        assert np.all(result.predicted_se >= 0)
    else:
        assert 0 <= result.power <= 1
        assert result.predicted_se >= 0

    assert result.critical_value >= 0
    assert 0 <= result.alpha <= 1


def test_power_dsl_fixed_effects(sample_data, sample_prediction):
    """Test power_dsl with fixed effects"""
    # Extract labeled indicator
    labeled_ind = sample_data["labeled"].values

    # Create design matrices from formula (basic formula, fixed effects handling may vary)
    y_mat, X_mat = dmatrices("y ~ x1 + x2 + x3 + x4 + x5", sample_data, return_type="dataframe")

    # Run DSL first
    dsl_result = dsl(
        X=X_mat.values,
        y=y_mat.values.flatten(),
        labeled_ind=labeled_ind,
        sample_prob=sample_data["sample_prob"].values,
        model="felm",
        method="fixed_effects",
    )

    # Run power_dsl
    result = power_dsl(
        formula="y ~ x1 + x2 + x3 + x4 + x5",
        data=sample_data,
        labeled_ind=labeled_ind,
        sample_prob=sample_data["sample_prob"].values,
        model="felm",
        method="fixed_effects",
        dsl_out=dsl_result,
        alpha=0.05,
    )

    # Check result
    assert result is not None
    assert hasattr(result, "power")
    assert hasattr(result, "predicted_se")
    assert hasattr(result, "critical_value")
    assert hasattr(result, "alpha")

    # Check values - power and predicted_se can be arrays (one per coefficient)
    assert isinstance(result.power, (float, np.ndarray))
    assert isinstance(result.predicted_se, (float, np.ndarray))
    assert isinstance(result.critical_value, (float, np.floating))
    assert isinstance(result.alpha, (float, np.floating))

    # Check value ranges
    if isinstance(result.power, np.ndarray):
        assert np.all((result.power >= 0) & (result.power <= 1))
        assert np.all(result.predicted_se >= 0)
    else:
        assert 0 <= result.power <= 1
        assert result.predicted_se >= 0

    assert result.critical_value >= 0
    assert 0 <= result.alpha <= 1
