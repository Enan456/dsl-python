#!/usr/bin/env python
"""Test DSL without using predictions to isolate the issue"""

import logging
import numpy as np
from patsy import dmatrices
from PanChen_test.compare_panchen import load_panchen_data, prepare_data_for_dsl
from dsl.helpers.dsl_general import dsl_general
from dsl import DSLResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load data
data = load_panchen_data()
df = prepare_data_for_dsl(data)

formula = "SendOrNot ~ countyWrong + prefecWrong + connect2b + prevalence + regionj + groupIssue"
y, X = dmatrices(formula, df, return_type="dataframe")

print("Testing WITHOUT predictions (X_orig = X_pred)")
print("=" * 80)

# Run DSL with X_orig = X_pred (no predictions)
par, info = dsl_general(
    Y_orig=y.values.flatten(),
    X_orig=X.values,
    Y_pred=y.values.flatten(),
    X_pred=X.values,  # Same as X_orig, no predictions
    labeled_ind=df["labeled"].values,
    sample_prob_use=df["sample_prob"].values,
    model="logit",
)

print(f"\nCoefficients (no predictions):")
print(f"  prefecWrong: {par[2]:.4f}")
print(f"  SE: {info['standard_errors'][2]:.4f}")

print("\n" + "=" * 80)
print("Now testing WITH predictions (X_pred uses pred_countyWrong)")
print("=" * 80)

# Create X_pred with predictions
df_pred = df.copy()
df_pred["countyWrong"] = df_pred["pred_countyWrong"]
_, X_pred = dmatrices(formula, df_pred, return_type="dataframe")

par2, info2 = dsl_general(
    Y_orig=y.values.flatten(),
    X_orig=X.values,
    Y_pred=y.values.flatten(),
    X_pred=X_pred.values,  # Uses pred_countyWrong
    labeled_ind=df["labeled"].values,
    sample_prob_use=df["sample_prob"].values,
    model="logit",
)

print(f"\nCoefficients (with predictions):")
print(f"  prefecWrong: {par2[2]:.4f}")
print(f"  SE: {info2['standard_errors'][2]:.4f}")

print("\n" + "=" * 80)
print("R Target:")
print("  prefecWrong: -1.1162")
print("  SE: 0.2970")
