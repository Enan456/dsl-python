#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Debug script to trace DSL optimization for PanChen dataset
"""

import logging
import numpy as np
from patsy import dmatrices
from PanChen_test.compare_panchen import load_panchen_data, prepare_data_for_dsl
from dsl import dsl

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    """Debug DSL estimation"""
    logger.info("Loading and preparing data...")
    data = load_panchen_data()
    df = prepare_data_for_dsl(data)

    formula = "SendOrNot ~ countyWrong + prefecWrong + connect2b + prevalence + regionj + groupIssue"
    y, X = dmatrices(formula, df, return_type="dataframe")

    logger.info(f"X shape: {X.shape}")
    logger.info(f"y shape: {y.shape}")
    logger.info(f"Labeled count: {df['labeled'].sum()}")

    # Check data statistics
    logger.info("\nX statistics (first 5 columns):")
    for col in X.columns[:5]:
        logger.info(f"{col}: mean={X[col].mean():.6f}, std={X[col].std():.6f}")

    logger.info("\ny statistics:")
    logger.info(f"mean={y.values.mean():.6f}, std={y.values.std():.6f}")

    logger.info("\nRunning DSL estimation...")
    result = dsl(
        X=X.values,
        y=y.values,
        labeled_ind=df["labeled"].values,
        sample_prob=df["sample_prob"].values,
        model="logit",
        method="logistic",
    )

    logger.info("\n=== RESULTS ===")
    logger.info(f"Coefficients: {result.coefficients}")
    logger.info(f"Standard errors: {result.standard_errors}")
    logger.info(f"Objective: {result.objective}")
    logger.info(f"Success: {result.success}")
    logger.info(f"Message: {result.message}")
    logger.info(f"Iterations: {result.niter}")

    # Compare with R
    r_coefs = np.array([2.0978, -0.2617, -1.1162, -0.0788, -0.3271, 0.1253, -2.3222])
    r_ses = np.array([0.3621, 0.2230, 0.2970, 0.1197, 0.1520, 0.4566, 0.3597])

    logger.info("\n=== COMPARISON WITH R ===")
    terms = ["(Intercept)", "countyWrong", "prefecWrong", "connect2b", "prevalence", "regionj", "groupIssue"]
    for i, term in enumerate(terms):
        py_coef = result.coefficients[i]
        py_se = result.standard_errors[i]
        r_coef = r_coefs[i]
        r_se = r_ses[i]
        coef_diff = abs(py_coef - r_coef)
        se_diff = abs(py_se - r_se)
        logger.info(f"\n{term}:")
        logger.info(f"  Python: coef={py_coef:.4f}, SE={py_se:.4f}")
        logger.info(f"  R:      coef={r_coef:.4f}, SE={r_se:.4f}")
        logger.info(f"  Diff:   coef={coef_diff:.4f}, SE={se_diff:.4f}")

if __name__ == "__main__":
    main()
