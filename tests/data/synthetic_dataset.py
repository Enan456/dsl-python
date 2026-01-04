#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Synthetic dataset generator for controlled Python-R validation.

Generates synthetic datasets with known properties for testing DSL implementation.
Unlike PanChen which has RNG incompatibility, synthetic datasets can use fixed
labeled indicators for exact Python-R matching.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def generate_synthetic_logistic_data(
    n_total: int = 1000,
    n_labeled: int = 500,
    n_features: int = 5,
    beta_true: Optional[np.ndarray] = None,
    labeled_indices: Optional[np.ndarray] = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic dataset for logistic regression DSL.

    Args:
        n_total: Total number of observations
        n_labeled: Number of labeled observations
        n_features: Number of features (excluding intercept)
        beta_true: True coefficient vector (including intercept). If None, randomly generated
        labeled_indices: Specific indices to use as labeled. If None, randomly selected
        random_seed: Random seed for reproducibility

    Returns:
        DataFrame with columns: y (outcome), x1...xn (features), labeled (indicator), sample_prob
    """
    np.random.seed(random_seed)

    # Generate features (standardized)
    X = np.random.randn(n_total, n_features)

    # Add intercept
    X_with_intercept = np.column_stack([np.ones(n_total), X])

    # Generate true coefficients if not provided
    if beta_true is None:
        beta_true = np.random.randn(n_features + 1) * 0.5
        beta_true[0] = 0.5  # Intercept

    # Generate outcomes using logistic model
    logits = X_with_intercept @ beta_true
    probs = 1 / (1 + np.exp(-logits))
    y = (np.random.rand(n_total) < probs).astype(int)

    # Create labeled indicator
    if labeled_indices is None:
        labeled_indices = np.random.choice(n_total, n_labeled, replace=False)

    labeled = np.zeros(n_total, dtype=int)
    labeled[labeled_indices] = 1

    # Sample probability (equal for all labeled)
    sample_prob = n_labeled / n_total

    # Create DataFrame
    df = pd.DataFrame(X, columns=[f'x{i+1}' for i in range(n_features)])
    df['y'] = y
    df['labeled'] = labeled
    df['sample_prob'] = sample_prob

    # Add metadata as attributes
    df.attrs['beta_true'] = beta_true
    df.attrs['n_labeled'] = n_labeled
    df.attrs['n_total'] = n_total
    df.attrs['random_seed'] = random_seed

    return df


def generate_synthetic_with_prediction(
    n_total: int = 1000,
    n_labeled: int = 500,
    n_features: int = 5,
    prediction_noise: float = 0.1,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic dataset with prediction variable.

    Creates a dataset where X_pred differs from X_orig by adding noise,
    simulating the scenario where predictions are used for unlabeled data.

    Args:
        n_total: Total number of observations
        n_labeled: Number of labeled observations
        n_features: Number of features
        prediction_noise: Standard deviation of noise added to predictions
        random_seed: Random seed for reproducibility

    Returns:
        DataFrame with original features, predictions, and labels
    """
    np.random.seed(random_seed)

    # Generate base dataset
    df = generate_synthetic_logistic_data(
        n_total=n_total,
        n_labeled=n_labeled,
        n_features=n_features,
        random_seed=random_seed,
    )

    # Create prediction variables (original + noise)
    for i in range(1, n_features + 1):
        col = f'x{i}'
        pred_col = f'x{i}_pred'

        # For labeled data, prediction = original
        # For unlabeled data, prediction = original + noise
        df[pred_col] = df[col].copy()

        unlabeled_mask = df['labeled'] == 0
        noise = np.random.randn(unlabeled_mask.sum()) * prediction_noise
        df.loc[unlabeled_mask, pred_col] = df.loc[unlabeled_mask, col] + noise

    # Add prediction noise level as metadata
    df.attrs['prediction_noise'] = prediction_noise

    return df


def generate_r_compatible_labeled_indicator(
    n_total: int,
    n_labeled: int,
    r_seed: int = 123
) -> np.ndarray:
    """Generate labeled indicator compatible with R's set.seed().

    Note: Python's RNG is fundamentally incompatible with R's RNG.
    This function provides a way to specify exact labeled indices
    that can be shared between Python and R implementations.

    Args:
        n_total: Total number of observations
        n_labeled: Number of labeled observations
        r_seed: R seed value (for documentation, not used for RNG)

    Returns:
        Array of labeled indices
    """
    # For exact Python-R matching, labeled indices should be
    # saved from R and loaded here, rather than generated
    print(
        f"Warning: Python RNG cannot match R's set.seed({r_seed}). "
        "For exact matching, save labeled indices from R and load them here."
    )

    # Return indices that can be explicitly set
    # In practice, these would come from R
    return np.arange(n_labeled)


def save_synthetic_dataset(df: pd.DataFrame, output_path: str, format: str = 'parquet'):
    """Save synthetic dataset to file.

    Args:
        df: DataFrame to save
        output_path: Output file path
        format: File format ('parquet', 'csv', or 'pickle')
    """
    if format == 'parquet':
        df.to_parquet(output_path)
    elif format == 'csv':
        df.to_csv(output_path, index=False)
    elif format == 'pickle':
        df.to_pickle(output_path)
    else:
        raise ValueError(f"Unsupported format: {format}")

    print(f"Synthetic dataset saved to: {output_path}")
    print(f"  Total observations: {len(df)}")
    print(f"  Labeled: {df['labeled'].sum()}")
    print(f"  Features: {len([c for c in df.columns if c.startswith('x')])}")


def load_synthetic_dataset(input_path: str) -> pd.DataFrame:
    """Load synthetic dataset from file.

    Args:
        input_path: Input file path

    Returns:
        Loaded DataFrame
    """
    if input_path.endswith('.parquet'):
        return pd.read_parquet(input_path)
    elif input_path.endswith('.csv'):
        return pd.read_csv(input_path)
    elif input_path.endswith('.pkl') or input_path.endswith('.pickle'):
        return pd.read_pickle(input_path)
    else:
        raise ValueError(f"Unsupported file format: {input_path}")


if __name__ == '__main__':
    # Example: Generate and save synthetic datasets
    import argparse

    parser = argparse.ArgumentParser(description='Generate synthetic datasets')

    parser.add_argument('--n-total', type=int, default=1000,
                       help='Total observations')
    parser.add_argument('--n-labeled', type=int, default=500,
                       help='Labeled observations')
    parser.add_argument('--n-features', type=int, default=5,
                       help='Number of features')
    parser.add_argument('--output', type=str, default='synthetic_data.parquet',
                       help='Output file path')
    parser.add_argument('--with-prediction', action='store_true',
                       help='Generate with prediction variables')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    if args.with_prediction:
        df = generate_synthetic_with_prediction(
            n_total=args.n_total,
            n_labeled=args.n_labeled,
            n_features=args.n_features,
            random_seed=args.seed,
        )
    else:
        df = generate_synthetic_logistic_data(
            n_total=args.n_total,
            n_labeled=args.n_labeled,
            n_features=args.n_features,
            random_seed=args.seed,
        )

    save_synthetic_dataset(df, args.output)

    print("\nDataset Summary:")
    print(df.describe())
    print(f"\nTrue coefficients: {df.attrs.get('beta_true')}")
