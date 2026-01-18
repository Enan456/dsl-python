"""
Functional tests for the summary and plot functions
"""

import numpy as np
import pandas as pd

from dsl.dsl import dsl, plot_power, summary
from dsl.power_dsl import power_dsl


def test_summary_dsl(sample_data, sample_prediction):
    """Test summary function for DSL results"""
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
    summary_table = summary(dsl_result)

    # Check result
    assert summary_table is not None
    assert isinstance(summary_table, pd.DataFrame)
    assert summary_table.shape[0] == 6  # 5 features + intercept
    assert summary_table.shape[1] >= 4  # At least coefficients, SE, t-value, p-value


def test_summary_power_dsl(sample_data, sample_prediction):
    """Test summary_power function for power_dsl results"""
    # Add prediction to data
    sample_data["prediction"] = sample_prediction

    # Extract labeled indicator
    labeled_ind = sample_data["labeled"].values

    # Run DSL first
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
    power_result = power_dsl(
        dsl_output=dsl_result,
        data=sample_data,
        labeled_ind=labeled_ind,
        sample_prob=sample_data["sample_prob"].values,
    )

    # Check result
    assert power_result is not None
    assert hasattr(power_result, "power")
    assert hasattr(power_result, "predicted_se")
    assert hasattr(power_result, "critical_value")
    assert hasattr(power_result, "alpha")


def test_plot_power(sample_data, sample_prediction):
    """Test plot_power function"""
    # Add prediction to data
    sample_data["prediction"] = sample_prediction

    # Extract labeled indicator
    labeled_ind = sample_data["labeled"].values

    # Run DSL first
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

    # Create a PowerDSLResult-like object for plot_power
    from dsl.dsl import PowerDSLResult

    power_result = PowerDSLResult(
        power=np.array([0.8, 0.85, 0.9, 0.75, 0.88, 0.92]),
        predicted_se=np.sqrt(np.diag(dsl_result.vcov)),
        critical_value=1.96,
        alpha=0.05,
    )

    # Run plot_power - just check it doesn't error
    # (it creates matplotlib plots which we don't want to display in tests)
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend

    # Run plot_power for all coefficients
    plot_power(power_result)

    # Run plot_power for a specific coefficient
    plot_power(power_result, coefficients="beta_0")

    # Run plot_power for multiple coefficients
    plot_power(power_result, coefficients=["beta_0", "beta_1"])
