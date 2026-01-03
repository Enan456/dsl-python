#!/usr/bin/env python
"""Check optimization convergence details"""

import logging
import numpy as np
from patsy import dmatrices
from PanChen_test.compare_panchen import load_panchen_data, prepare_data_for_dsl
from dsl.helpers.dsl_general import dsl_general

logging.basicConfig(level=logging.WARNING)  # Suppress info logs

# Load data
data = load_panchen_data()
df = prepare_data_for_dsl(data)

formula = "SendOrNot ~ countyWrong + prefecWrong + connect2b + prevalence + regionj + groupIssue"
y, X = dmatrices(formula, df, return_type="dataframe")

print("=" * 80)
print("Convergence Analysis")
print("=" * 80)

# Test WITHOUT predictions
par1, info1 = dsl_general(
    Y_orig=y.values.flatten(),
    X_orig=X.values,
    Y_pred=y.values.flatten(),
    X_pred=X.values,
    labeled_ind=df["labeled"].values,
    sample_prob_use=df["sample_prob"].values,
    model="logit",
)

print("\nWithout predictions (X_orig = X_pred):")
print(f"  Converged: {info1['convergence']}")
print(f"  Iterations: {info1['iterations']}")
print(f"  Objective: {info1['objective']:.10f}")
print(f"  Message: {info1['message']}")

# Test WITH predictions
df_pred = df.copy()
df_pred["countyWrong"] = df_pred["pred_countyWrong"]
_, X_pred = dmatrices(formula, df_pred, return_type="dataframe")

par2, info2 = dsl_general(
    Y_orig=y.values.flatten(),
    X_orig=X.values,
    Y_pred=y.values.flatten(),
    X_pred=X_pred.values,
    labeled_ind=df["labeled"].values,
    sample_prob_use=df["sample_prob"].values,
    model="logit",
)

print("\nWith predictions (X_pred uses pred_countyWrong):")
print(f"  Converged: {info2['convergence']}")
print(f"  Iterations: {info2['iterations']}")
print(f"  Objective: {info2['objective']:.10f}")
print(f"  Message: {info2['message']}")

print("\n" + "=" * 80)
print("Note: Objective should be very close to 0 for GMM estimation")
print("=" * 80)
