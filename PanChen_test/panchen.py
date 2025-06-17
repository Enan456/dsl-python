#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DSL (Double-Supervised Learning) Framework Example

This script demonstrates how to use the DSL framework in Python,
replicating the R example from the documentation.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# DSL imports
from dsl import dsl, plot_power, power_dsl, summary, summary_power


def generate_sample_data(n_samples=1000, n_features=5, labeled_ratio=0.8, seed=1234):
    """
    Generate sample data for DSL estimation.

    Parameters
    ----------
    n_samples : int
        Number of samples
    n_features : int
        Number of features
    labeled_ratio : float
        Ratio of labeled samples
    seed : int
        Random seed

    Returns
    -------
    pd.DataFrame
        Sample data
    """
    np.random.seed(seed)

    # Generate features
    X = np.random.randn(n_samples, n_features)

    # Generate coefficients
    beta = np.array([1.0, 0.5, -0.3, 0.2, 0.1])

    # Generate outcome (linear model)
    y = X @ beta + np.random.randn(n_samples) * 0.1

    # Create labeled indicator
    labeled = np.random.binomial(1, labeled_ratio, n_samples)

    # Create sampling probability (equal probability)
    sample_prob = np.ones(n_samples) * labeled_ratio

    # Create cluster indicator (10 clusters)
    cluster = np.random.randint(0, 10, n_samples)

    # Create fixed effects (5 groups)
    fixed_effect = np.random.randint(0, 5, n_samples)

    # Create DataFrame
    data = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(n_features)])
    data["y"] = y
    data["labeled"] = labeled
    data["sample_prob"] = sample_prob
    data["cluster"] = cluster
    data["fixed_effect"] = fixed_effect

    return data


def generate_sample_prediction(data):
    """
    Generate sample predictions for DSL estimation.

    Parameters
    ----------
    data : pd.DataFrame
        Sample data

    Returns
    -------
    np.ndarray
        Sample predictions
    """
    # Simple prediction based on features
    X = data[[f"x{i+1}" for i in range(5)]].values
    beta = np.array([1.0, 0.5, -0.3, 0.2, 0.1])
    prediction = X @ beta

    return prediction


def main():
    """Main function to demonstrate DSL framework usage."""
    print("DSL Framework Example")
    print("===================")

    # Generate sample data
    print("\nGenerating sample data...")
    data = generate_sample_data()
    prediction = generate_sample_prediction(data)

    # Add prediction to data
    data["prediction"] = prediction

    # Extract labeled indicator
    labeled_ind = data["labeled"].values

    # Basic DSL estimation (linear regression)
    print("\nRunning DSL estimation (linear regression)...")
    result = dsl(
        formula="y ~ x1 + x2 + x3 + x4 + x5",
        data=data,
        labeled_ind=labeled_ind,
        sample_prob=data["sample_prob"].values,
        model="lm",
        method="linear",
    )

    # Print summary
    print("\nDSL Estimation Results (Linear Regression):")
    print("----------------------------------------")
    summary_df = summary(result)
    print(summary_df)

    # DSL estimation with logistic regression
    print("\nRunning DSL estimation (logistic regression)...")
    # Generate binary outcome for logistic regression
    data_logit = data.copy()
    features = data_logit[[f"x{i+1}" for i in range(5)]].values
    beta = np.array([1.0, 0.5, -0.3, 0.2, 0.1])
    linear_pred = features @ beta
    logit = 1 / (1 + np.exp(-linear_pred))
    data_logit["y"] = np.random.binomial(1, logit, len(data_logit))

    result_logit = dsl(
        formula="y ~ x1 + x2 + x3 + x4 + x5",
        data=data_logit,
        labeled_ind=labeled_ind,
        sample_prob=data_logit["sample_prob"].values,
        model="logit",
        method="linear",
    )

    # Print summary
    print("\nDSL Estimation Results (Logistic Regression):")
    print("-------------------------------------------")
    summary_df_logit = summary(result_logit)
    print(summary_df_logit)

    # DSL estimation with fixed effects
    print("\nRunning DSL estimation (fixed effects)...")
    result_fe = dsl(
        formula="y ~ x1 + x2 + x3 + x4 + x5 | fixed_effect",
        data=data,
        labeled_ind=labeled_ind,
        sample_prob=data["sample_prob"].values,
        model="felm",
        method="linear",
    )

    # Print summary
    print("\nDSL Estimation Results (Fixed Effects):")
    print("-------------------------------------")
    summary_df_fe = summary(result_fe)
    print(summary_df_fe)

    # Power analysis
    print("\nRunning power analysis...")
    power_result = power_dsl(
        model="lm",
        formula="y ~ x1 + x2 + x3 + x4 + x5",
        predicted_var=["y"],
        prediction="prediction",
        data=data,
        labeled_ind=labeled_ind,
        sample_prob=data["sample_prob"].values,
        sl_method="linear",
        feature=["x1", "x2", "x3", "x4", "x5"],
        family="gaussian",
        cross_fit=2,
        sample_split=2,
        alpha=0.05,
        power=0.8,
        seed=1234,
    )

    # Print summary
    print("\nPower Analysis Results:")
    print("----------------------")
    summary_power_df = summary_power(power_result)
    print(summary_power_df)

    # Plot power analysis results
    print("\nPlotting power analysis results...")
    plt.figure(figsize=(10, 6))
    plot_power(power_result)
    plt.savefig("power_analysis.png")
    print("Plot saved as 'power_analysis.png'")

    print("\nDSL Framework Example Completed!")


if __name__ == "__main__":
    main()
