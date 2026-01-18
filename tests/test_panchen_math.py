#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Mathematical comparison tests between R and Python DSL implementations
for the PanChen dataset.
"""

import logging
import sys

import numpy as np
from patsy import dmatrices, dmatrix

# Removed unused pandas and stats imports
from PanChen_test.compare_panchen import load_panchen_data, prepare_data_for_dsl

# DSL imports
from dsl import DSLResult, dsl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def test_data_loading():
    """Test data loading and initial statistics."""
    logger.info("\n=== Testing Data Loading ===")

    # Load data
    data = load_panchen_data()

    # Basic statistics that should match R
    logger.info("\nInitial Data Statistics:")
    logger.info(f"Total observations: {len(data)}")
    logger.info(f"Number of columns: {len(data.columns)}")

    # Column-wise statistics
    for col in data.columns:
        stats_dict = {
            "mean": data[col].mean(),
            "std": data[col].std(),
            "na_count": data[col].isna().sum(),
            "unique_values": len(data[col].unique()),
        }
        logger.info(f"\n{col} statistics:")
        for stat, value in stats_dict.items():
            logger.info(f"{stat}: {value}")


def test_data_preparation():
    """Test data preparation and transformation."""
    logger.info("\n=== Testing Data Preparation ===")

    # Load and prepare data
    data = load_panchen_data()
    df = prepare_data_for_dsl(data)

    # Test labeled/unlabeled split
    labeled_count = df["labeled"].sum()
    logger.info(f"\nLabeled observations: {labeled_count}")
    logger.info(f"Unlabeled observations: {len(df) - labeled_count}")

    # Test sample probabilities
    logger.info(f"\nSample probability value: {df['sample_prob'].iloc[0]}")
    unique_probs = df["sample_prob"].unique()
    logger.info(f"Sample probability unique values: {unique_probs}")

    # Test missing value handling
    for col in df.columns:
        na_count = df[col].isna().sum()
        logger.info(f"\n{col} NA count after preparation: {na_count}")


def test_model_inputs():
    """Test model input preparation."""
    logger.info("\n=== Testing Model Inputs ===")

    # Load and prepare data
    data = load_panchen_data()
    df = prepare_data_for_dsl(data)

    # Formula components
    formula = (
        "SendOrNot ~ countyWrong + prefecWrong + connect2b + "
        "prevalence + regionj + groupIssue"
    )

    # Extract X and y
    y, X = dmatrices(formula, df, return_type="dataframe")

    logger.info("\nDesign Matrix Statistics:")
    logger.info(f"X shape: {X.shape}")
    logger.info(f"y shape: {y.shape}")

    # Column means and standard deviations
    logger.info("\nX column statistics:")
    for col in X.columns:
        stats_dict = {"mean": X[col].mean(), "std": X[col].std()}
        logger.info(f"\n{col}:")
        for stat, value in stats_dict.items():
            logger.info(f"{stat}: {value}")


def test_dsl_estimation():
    """Test DSL estimation process."""
    logger.info("\n=== Testing DSL Estimation ===")

    # Load and prepare data
    data = load_panchen_data()
    df = prepare_data_for_dsl(data)

    # Formula
    formula = (
        "SendOrNot ~ countyWrong + prefecWrong + connect2b + "
        "prevalence + regionj + groupIssue"
    )

    # Run DSL
    y, X = dmatrices(formula, df, return_type="dataframe")
    result = dsl(
        X=X.values,
        y=y.values,
        labeled_ind=df["labeled"].values,
        sample_prob=df["sample_prob"].values,
        model="logit",
        method="logistic",
    )

    # Compare with R results
    r_coefs = {
        "(Intercept)": 2.0978,
        "countyWrong": -0.2617,
        "prefecWrong": -1.1162,
        "connect2b": -0.0788,
        "prevalence": -0.3271,
        "regionj": 0.1253,
        "groupIssue": -2.3222,
    }

    r_ses = {
        "(Intercept)": 0.3621,
        "countyWrong": 0.2230,
        "prefecWrong": 0.2970,
        "connect2b": 0.1197,
        "prevalence": 0.1520,
        "regionj": 0.4566,
        "groupIssue": 0.3597,
    }

    # Extract terms from formula
    formula_parts = formula.split("~")
    terms = ["(Intercept)"] + [t.strip() for t in formula_parts[1].split("+")]

    logger.info("\nCoefficient Comparison:")
    for i, term in enumerate(terms):
        py_coef = result.coefficients[i]
        py_se = result.standard_errors[i]
        r_coef = r_coefs[term]
        r_se = r_ses[term]

        coef_diff = abs(py_coef - r_coef)
        se_diff = abs(py_se - r_se)

        logger.info(f"\n{term}:")
        logger.info(
            f"Python coef: {py_coef:.4f}, R coef: {r_coef:.4f}, "
            f"Diff: {coef_diff:.4f}"
        )
        logger.info(f"Python SE: {py_se:.4f}, R SE: {r_se:.4f}, Diff: {se_diff:.4f}")


def test_prediction_accuracy():
    """Test prediction accuracy and model fit."""
    logger.info("\n=== Testing Prediction Accuracy ===")

    # Load and prepare data
    data = load_panchen_data()
    df = prepare_data_for_dsl(data)

    # Formula
    formula = (
        "SendOrNot ~ countyWrong + prefecWrong + connect2b + "
        "prevalence + regionj + groupIssue"
    )

    # Run DSL
    y, X = dmatrices(formula, df, return_type="dataframe")
    result = dsl(
        X=X.values,
        y=y.values,
        labeled_ind=df["labeled"].values,
        sample_prob=df["sample_prob"].values,
        model="logit",
        method="logistic",
    )

    # Get predictions for labeled data
    labeled_mask = df["labeled"] == 1
    labeled_data = df[labeled_mask]

    # Use dmatrix (singular) to get only the design matrix
    X_labeled = dmatrix(formula.split("~")[1], labeled_data, return_type="dataframe")

    # Calculate predicted probabilities
    # Ensure coefficients are treated as a column vector for matmul
    logits = X_labeled.values @ result.coefficients.reshape(-1, 1)
    pred_probs = 1 / (1 + np.exp(-logits))

    # Calculate metrics
    y_true = labeled_data["SendOrNot"].values
    y_pred = (pred_probs > 0.5).astype(int)

    accuracy = np.mean(y_pred == y_true)
    logger.info(f"\nAccuracy on labeled data: {accuracy:.4f}")

    # Calculate log-likelihood
    # Add small epsilon to avoid log(0)
    eps = 1e-15
    log_pred_probs = np.log(pred_probs + eps)
    log_one_minus_pred_probs = np.log(1 - pred_probs + eps)
    ll = np.sum(y_true * log_pred_probs + (1 - y_true) * log_one_minus_pred_probs)
    logger.info(f"Log-likelihood: {ll:.4f}")


def main():
    """Run all tests."""
    logger.info("Starting mathematical comparison tests")

    try:
        test_data_loading()
        test_data_preparation()
        test_model_inputs()
        test_dsl_estimation()
        test_prediction_accuracy()

    except Exception as e:
        logger.error(f"Error during testing: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
