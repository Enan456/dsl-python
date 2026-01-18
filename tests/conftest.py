"""
Shared test fixtures for DSL framework tests using real PanChen data
"""

import numpy as np
import pandas as pd
import pytest
import os


def load_panchen_data():
    """Load the real PanChen dataset for testing."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    panchen_dir = os.path.join(script_dir, "..", "PanChen_test")
    file_path = os.path.join(panchen_dir, "PanChen.parquet")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PanChen data not found at {file_path}")
    
    try:
        data = pd.read_parquet(file_path)
        return data
    except Exception as e:
        raise ImportError(f"Error reading PanChen data: {e}. Please install pyarrow: pip install pyarrow")


def prepare_panchen_for_testing(data, test_size=500):
    """Prepare PanChen data for testing with controlled labeled/unlabeled split."""
    df = data.copy()
    
    # Use complete cases only
    complete_cases = df.dropna(subset=["countyWrong", "SendOrNot"])
    
    # Create labeled indicator (controlled split for testing)
    np.random.seed(123)
    n_labeled = min(test_size, len(complete_cases))
    available_indices = complete_cases.index.tolist()
    labeled_indices = np.random.choice(available_indices, size=n_labeled, replace=False)
    
    df["labeled"] = 0
    df.loc[labeled_indices, "labeled"] = 1
    
    # Create sample probability
    n_complete = len(complete_cases)
    sample_prob = n_labeled / n_complete
    df["sample_prob"] = sample_prob
    
    # Handle missing values in predictors for unlabeled data
    for col in ["countyWrong", "prefecWrong", "connect2b", "prevalence", "regionj", "groupIssue"]:
        mask = df["labeled"] == 0
        df.loc[mask, col] = df.loc[mask, col].fillna(0)
    
    # Handle missing values in outcome for unlabeled data
    df.loc[df["labeled"] == 0, "SendOrNot"] = df.loc[df["labeled"] == 0, "SendOrNot"].fillna(0)
    
    return df


@pytest.fixture
def sample_data():
    """Load real PanChen data for testing"""
    data = load_panchen_data()
    df = prepare_panchen_for_testing(data, test_size=500)
    
    # Rename columns to match test expectations
    df = df.rename(columns={
        "SendOrNot": "y",
        "countyWrong": "x1", 
        "prefecWrong": "x2",
        "connect2b": "x3",
        "prevalence": "x4", 
        "regionj": "x5"
    })
    
    # Add cluster and fixed_effect columns if needed
    if "cluster" not in df.columns:
        df["cluster"] = np.random.randint(0, 10, len(df))
    if "fixed_effect" not in df.columns:
        df["fixed_effect"] = np.random.randint(0, 5, len(df))

    # Add fe1 and fe2 columns for fixed effects tests
    df["fe1"] = df["cluster"]
    df["fe2"] = df["fixed_effect"]

    return df


@pytest.fixture
def sample_data_logit():
    """Load real PanChen data for logistic regression testing"""
    data = load_panchen_data()
    df = prepare_panchen_for_testing(data, test_size=500)
    
    # Rename columns to match test expectations
    df = df.rename(columns={
        "SendOrNot": "y",
        "countyWrong": "x1", 
        "prefecWrong": "x2",
        "connect2b": "x3",
        "prevalence": "x4", 
        "regionj": "x5"
    })
    
    # Add cluster and fixed_effect columns if needed
    if "cluster" not in df.columns:
        df["cluster"] = np.random.randint(0, 10, len(df))
    if "fixed_effect" not in df.columns:
        df["fixed_effect"] = np.random.randint(0, 5, len(df))

    # Add fe1 and fe2 columns for fixed effects tests
    df["fe1"] = df["cluster"]
    df["fe2"] = df["fixed_effect"]

    return df


@pytest.fixture
def sample_prediction(sample_data):
    """Generate predictions using real data features"""
    # Use actual features from PanChen data
    X = sample_data[["x1", "x2", "x3", "x4", "x5"]].values
    
    # Simple linear prediction (for testing purposes)
    beta = np.array([0.1, -0.2, 0.05, -0.1, 0.15])
    prediction = X @ beta
    
    return prediction


@pytest.fixture
def sample_prediction_logit(sample_data_logit):
    """Generate logistic predictions using real data features"""
    # Use actual features from PanChen data
    X = sample_data_logit[["x1", "x2", "x3", "x4", "x5"]].values
    
    # Simple logistic prediction (for testing purposes)
    beta = np.array([0.1, -0.2, 0.05, -0.1, 0.15])
    logits = X @ beta
    prediction = 1 / (1 + np.exp(-logits))
    
    return prediction
