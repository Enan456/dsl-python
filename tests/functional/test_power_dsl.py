"""
Functional tests for the power_dsl function
"""

import numpy as np
import pytest
from dsl.dsl import dsl
from dsl.power_dsl import power_dsl


def test_power_dsl_with_dsl_output(sample_data, sample_prediction):
    """Test power_dsl with dsl output"""
    # Add prediction to data
    sample_data["prediction"] = sample_prediction

    # Extract labeled indicator
    labeled_ind = sample_data["labeled"].values

    # Run DSL
    dsl_result = dsl(
        model="lm",
        formula="y ~ x1 + x2 + x3 + x4 + x5",
        predicted_var=["y"],
        prediction="prediction",
        data=sample_data,
        labeled_ind=labeled_ind,
        sample_prob=sample_data["sample_prob"].values,
        sl_method="grf",
        feature=["x1", "x2", "x3", "x4", "x5"],
        family="gaussian",
        cross_fit=2,
        sample_split=2,
        seed=1234,
    )

    # Run power_dsl
    result = power_dsl(
        dsl_output=dsl_result,
        data=sample_data,
        labeled_ind=labeled_ind,
        sample_prob=sample_data["sample_prob"].values,
        alpha=0.05,
        power=0.8,
        seed=1234,
    )

    # Check result
    assert result is not None
    assert hasattr(result, "power")
    assert hasattr(result, "predicted_se")
    assert hasattr(result, "critical_value")
    assert hasattr(result, "alpha")

    # Check values - power is float, predicted_se is array
    assert isinstance(result.power, (int, float, np.floating))
    assert isinstance(result.predicted_se, np.ndarray)
    assert isinstance(result.critical_value, (int, float, np.floating))
    assert isinstance(result.alpha, (int, float, np.floating))
    assert 0 <= result.power <= 1
    assert np.all(result.predicted_se >= 0)
    assert result.critical_value >= 0
    assert 0 <= result.alpha <= 1


def test_power_dsl_without_dsl_output(sample_data, sample_prediction):
    """Test power_dsl without dsl output"""
    # Add prediction to data
    sample_data["prediction"] = sample_prediction

    # Extract labeled indicator
    labeled_ind = sample_data["labeled"].values

    # Run power_dsl
    result = power_dsl(
        model="lm",
        formula="y ~ x1 + x2 + x3 + x4 + x5",
        predicted_var=["y"],
        prediction="prediction",
        data=sample_data,
        labeled_ind=labeled_ind,
        sample_prob=sample_data["sample_prob"].values,
        sl_method="grf",
        feature=["x1", "x2", "x3", "x4", "x5"],
        family="gaussian",
        cross_fit=2,
        sample_split=2,
        alpha=0.05,
        power=0.8,
        seed=1234,
    )

    # Check result
    assert result is not None
    assert hasattr(result, "power")
    assert hasattr(result, "predicted_se")
    assert hasattr(result, "critical_value")
    assert hasattr(result, "alpha")

    # Check values - power is float, predicted_se is array
    assert isinstance(result.power, (int, float, np.floating))
    assert isinstance(result.predicted_se, np.ndarray)
    assert isinstance(result.critical_value, (int, float, np.floating))
    assert isinstance(result.alpha, (int, float, np.floating))
    assert 0 <= result.power <= 1
    assert np.all(result.predicted_se >= 0)
    assert result.critical_value >= 0
    assert 0 <= result.alpha <= 1


def test_power_dsl_logistic_regression(sample_data, sample_prediction):
    """Test power_dsl with logistic regression"""
    # Add prediction to data
    sample_data["prediction"] = sample_prediction

    # Extract labeled indicator
    labeled_ind = sample_data["labeled"].values

    # Run power_dsl
    result = power_dsl(
        model="logit",
        formula="y ~ x1 + x2 + x3 + x4 + x5",
        predicted_var=["y"],
        prediction="prediction",
        data=sample_data,
        labeled_ind=labeled_ind,
        sample_prob=sample_data["sample_prob"].values,
        sl_method="grf",
        feature=["x1", "x2", "x3", "x4", "x5"],
        family="binomial",
        cross_fit=2,
        sample_split=2,
        alpha=0.05,
        power=0.8,
        seed=1234,
    )

    # Check result
    assert result is not None
    assert hasattr(result, "power")
    assert hasattr(result, "predicted_se")
    assert hasattr(result, "critical_value")
    assert hasattr(result, "alpha")

    # Check values - power is float, predicted_se is array
    assert isinstance(result.power, (int, float, np.floating))
    assert isinstance(result.predicted_se, np.ndarray)
    assert isinstance(result.critical_value, (int, float, np.floating))
    assert isinstance(result.alpha, (int, float, np.floating))
    assert 0 <= result.power <= 1
    assert np.all(result.predicted_se >= 0)
    assert result.critical_value >= 0
    assert 0 <= result.alpha <= 1


def test_power_dsl_fixed_effects(sample_data, sample_prediction):
    """Test power_dsl with fixed effects"""
    # Add prediction to data
    sample_data["prediction"] = sample_prediction

    # Extract labeled indicator
    labeled_ind = sample_data["labeled"].values

    # Run power_dsl
    result = power_dsl(
        model="felm",
        formula="y ~ x1 + x2 + x3 + x4 + x5 | fe1 + fe2",
        predicted_var=["y"],
        prediction="prediction",
        data=sample_data,
        labeled_ind=labeled_ind,
        sample_prob=sample_data["sample_prob"].values,
        sl_method="grf",
        feature=["x1", "x2", "x3", "x4", "x5"],
        family="gaussian",
        cross_fit=2,
        sample_split=2,
        alpha=0.05,
        power=0.8,
        seed=1234,
    )

    # Check result
    assert result is not None
    assert hasattr(result, "power")
    assert hasattr(result, "predicted_se")
    assert hasattr(result, "critical_value")
    assert hasattr(result, "alpha")

    # Check values - power is float, predicted_se is array
    assert isinstance(result.power, (int, float, np.floating))
    assert isinstance(result.predicted_se, np.ndarray)
    assert isinstance(result.critical_value, (int, float, np.floating))
    assert isinstance(result.alpha, (int, float, np.floating))
    assert 0 <= result.power <= 1
    assert np.all(result.predicted_se >= 0)
    assert result.critical_value >= 0
    assert 0 <= result.alpha <= 1
