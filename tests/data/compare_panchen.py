#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Shared data loading and preparation functions for PanChen comparison tests.
"""

import os
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def load_panchen_data():
    """Load the PanChen dataset from the test data directory."""
    # Find the PanChen_test directory relative to this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    file_path = os.path.join(project_root, "PanChen_test", "PanChen.parquet")

    logger.info(f"Loading data from {file_path}")
    try:
        data = pd.read_parquet(file_path)
        logger.info(f"Successfully loaded {len(data)} observations")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def prepare_data_for_dsl(data):
    """Prepare PanChen data for DSL estimation to match R output.

    The PanChen dataset has columns:
    - SendOrNot: outcome variable (y)
    - countyWrong, prefecWrong, connect2b, prevalence, regionj, groupIssue: predictors
    - pred_countyWrong: prediction
    """
    logger.info("Preparing data for DSL estimation")

    # Create a copy of the data
    df = data.copy()

    # Important: Only use complete cases for labeled data
    complete_cases = df.dropna(subset=["countyWrong", "SendOrNot"])

    # Set random seed for reproducibility - use same seed as R
    np.random.seed(123)

    # Randomly select 500 observations from complete cases as labeled
    n_labeled = 500
    available_indices = complete_cases.index.tolist()
    labeled_indices = np.random.choice(available_indices, size=n_labeled, replace=False)

    # Create labeled indicator
    df["labeled"] = 0
    df.loc[labeled_indices, "labeled"] = 1
    labeled_count = df["labeled"].sum()
    unlabeled_count = len(df) - labeled_count
    logger.info(f"Labeled: {labeled_count}, Unlabeled: {unlabeled_count}")

    # Create sample probability
    n_complete = len(complete_cases)
    sample_prob = n_labeled / n_complete
    df["sample_prob"] = sample_prob
    logger.info(f"Sample probability: {sample_prob}")

    # Handle missing values in predictors
    for col in ["countyWrong", "prefecWrong", "connect2b", "prevalence", "regionj", "groupIssue"]:
        mask = df["labeled"] == 0
        na_count = df.loc[mask, col].isna().sum()
        if na_count > 0:
            logger.info(f"Filling {na_count} NA values in '{col}' for unlabeled data with 0")
            df.loc[mask, col] = df.loc[mask, col].fillna(0)

    # For SendOrNot, fill NA values in unlabeled data
    na_count = df.loc[df["labeled"] == 0, "SendOrNot"].isna().sum()
    if na_count > 0:
        logger.info(f"Filling {na_count} NA values in SendOrNot for unlabeled data with 0")
        df.loc[df["labeled"] == 0, "SendOrNot"] = df.loc[df["labeled"] == 0, "SendOrNot"].fillna(0)

    return df
