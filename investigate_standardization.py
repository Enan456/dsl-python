#!/usr/bin/env python
"""Investigate standardization and X_orig vs X_pred differences"""

import numpy as np
from patsy import dmatrices
from PanChen_test.compare_panchen import load_panchen_data, prepare_data_for_dsl

# Load data
data = load_panchen_data()
df = prepare_data_for_dsl(data)

formula = "SendOrNot ~ countyWrong + prefecWrong + connect2b + prevalence + regionj + groupIssue"
y, X = dmatrices(formula, df, return_type="dataframe")

# Create X_pred with predictions
df_pred = df.copy()
df_pred["countyWrong"] = df_pred["pred_countyWrong"]
_, X_pred = dmatrices(formula, df_pred, return_type="dataframe")

print("=" * 80)
print("X_orig vs X_pred Statistics")
print("=" * 80)

for i, col in enumerate(X.columns):
    x_orig_col = X.iloc[:, i].values
    x_pred_col = X_pred.iloc[:, i].values

    print(f"\n{col}:")
    print(f"  X_orig: mean={x_orig_col.mean():.6f}, std={x_orig_col.std():.6f}")
    print(f"  X_pred: mean={x_pred_col.mean():.6f}, std={x_pred_col.std():.6f}")
    print(f"  Same?   {np.allclose(x_orig_col, x_pred_col)}")

    if not np.allclose(x_orig_col, x_pred_col):
        diff = np.abs(x_orig_col - x_pred_col)
        print(f"  Max diff: {diff.max():.6f}")
        print(f"  Mean diff: {diff.mean():.6f}")
        print(f"  Num different: {(diff > 1e-10).sum()}")

# Check labeled vs unlabeled statistics
print("\n" + "=" * 80)
print("Labeled vs Unlabeled Data")
print("=" * 80)

labeled_mask = df["labeled"] == 1
for col in ["countyWrong", "prefecWrong", "pred_countyWrong"]:
    print(f"\n{col}:")
    print(f"  Labeled:   mean={df.loc[labeled_mask, col].mean():.6f}, std={df.loc[labeled_mask, col].std():.6f}")
    print(f"  Unlabeled: mean={df.loc[~labeled_mask, col].mean():.6f}, std={df.loc[~labeled_mask, col].std():.6f}")
    print(f"  All:       mean={df[col].mean():.6f}, std={df[col].std():.6f}")

# Check if pred_countyWrong has the right structure
print("\n" + "=" * 80)
print("pred_countyWrong Analysis")
print("=" * 80)
print(f"Missing values: {df['pred_countyWrong'].isna().sum()}")
print(f"Unique values: {df['pred_countyWrong'].nunique()}")
print(f"Min: {df['pred_countyWrong'].min():.6f}")
print(f"Max: {df['pred_countyWrong'].max():.6f}")
print(f"Is binary? {set(df['pred_countyWrong'].unique()) == {0.0, 1.0}}")

# Sample of data
print("\n" + "=" * 80)
print("Sample Data (first 10 rows)")
print("=" * 80)
print(df[["countyWrong", "pred_countyWrong", "labeled"]].head(10))
