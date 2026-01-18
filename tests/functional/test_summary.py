import numpy as np
from dsl.power_dsl import power_dsl, summary_power_dsl

from dsl.dsl import dsl, summary


def test_summary_dsl(sample_data, sample_prediction):
    """Test summary_dsl function"""
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

    # Run summary
    result = summary(dsl_result)

    # Check result - summary returns a DataFrame
    assert result is not None
    import pandas as pd
    assert isinstance(result, pd.DataFrame)

    # Check columns exist
    assert "Estimate" in result.columns
    assert "Std. Error" in result.columns
    assert "t value" in result.columns
    assert "Pr(>|t|)" in result.columns

    # Check shapes - 5 features + intercept = 6 rows
    assert len(result) == 6

    # Check values
    assert np.all(np.isfinite(result["Estimate"].values))
    assert np.all(result["Std. Error"].values >= 0)
    assert np.all(np.isfinite(result["t value"].values))
    assert np.all((result["Pr(>|t|)"].values >= 0) & (result["Pr(>|t|)"].values <= 1))


def test_summary_power_dsl(sample_data, sample_prediction):
    """Test summary_power_dsl function"""
    # Add prediction to data
    sample_data["prediction"] = sample_prediction

    # Extract labeled indicator
    labeled_ind = sample_data["labeled"].values

    # Run power_dsl
    power_result = power_dsl(
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

    # Run summary_power_dsl
    result = summary_power_dsl(power_result)

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
