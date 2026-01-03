#!/usr/bin/env python
"""Test to verify sample probability is being used correctly"""

import numpy as np
from patsy import dmatrices
from PanChen_test.compare_panchen import load_panchen_data, prepare_data_for_dsl
from dsl import dsl

# Load and prepare data
data = load_panchen_data()
df = prepare_data_for_dsl(data)

formula = "SendOrNot ~ countyWrong + prefecWrong + connect2b + prevalence + regionj + groupIssue"
y, X = dmatrices(formula, df, return_type="dataframe")

# Check sample probability
sample_probs = df["sample_prob"].unique()
print(f"Unique sample probabilities: {sample_probs}")
print(f"Sample prob for first labeled obs: {df.loc[df['labeled']==1, 'sample_prob'].iloc[0]}")

# Run DSL with correct sample probability
result1 = dsl(
    X=X.values,
    y=y.values,
    labeled_ind=df["labeled"].values,
    sample_prob=df["sample_prob"].values,
    model="logit",
    method="logistic",
)
print(f"\nResult with sample_prob={df['sample_prob'].iloc[0]:.4f}:")
print(f"prefecWrong coef: {result1.coefficients[2]:.4f}")

# Run DSL with sample probability = 1.0 (incorrect)
result2 = dsl(
    X=X.values,
    y=y.values,
    labeled_ind=df["labeled"].values,
    sample_prob=np.ones(len(df)),
    model="logit",
    method="logistic",
)
print(f"\nResult with sample_prob=1.0 (incorrect):")
print(f"prefecWrong coef: {result2.coefficients[2]:.4f}")

print(f"\nDifference: {abs(result1.coefficients[2] - result2.coefficients[2]):.4f}")
