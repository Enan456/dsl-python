#!/usr/bin/env python
"""Check what pred_countyWrong represents"""

import numpy as np
from PanChen_test.compare_panchen import load_panchen_data

# Load data
data = load_panchen_data()

print("=" * 80)
print("Understanding pred_countyWrong")
print("=" * 80)

# Check correlation between countyWrong and pred_countyWrong
valid_mask = data["countyWrong"].notna()
corr = np.corrcoef(data.loc[valid_mask, "countyWrong"],
                    data.loc[valid_mask, "pred_countyWrong"])[0, 1]

print(f"\nCorrelation between countyWrong and pred_countyWrong: {corr:.4f}")

# Check if they're the same for complete cases
same_count = (data.loc[valid_mask, "countyWrong"] == data.loc[valid_mask, "pred_countyWrong"]).sum()
total_valid = valid_mask.sum()
print(f"Same values: {same_count}/{total_valid} ({100*same_count/total_valid:.1f}%)")

# Look at cases where they differ
diff_mask = (data["countyWrong"] != data["pred_countyWrong"]) & valid_mask
print(f"\nRows where countyWrong != pred_countyWrong: {diff_mask.sum()}")

if diff_mask.sum() > 0:
    print("\nSample of differing rows:")
    print(data.loc[diff_mask, ["countyWrong", "pred_countyWrong", "SendOrNot"]].head(10))

# Check if pred_countyWrong is a model-based prediction
print("\n" + "=" * 80)
print("Is pred_countyWrong a model prediction?")
print("=" * 80)

# If it's a prediction from other variables, it should have values even where countyWrong is NA
na_mask = data["countyWrong"].isna()
print(f"\nRows where countyWrong is NA: {na_mask.sum()}")
print(f"Of these, pred_countyWrong is not NA: {(na_mask & data['pred_countyWrong'].notna()).sum()}")

print("\nSample where countyWrong is NA:")
print(data.loc[na_mask, ["countyWrong", "pred_countyWrong", "SendOrNot",
                          "prefecWrong", "connect2b"]].head(10))
