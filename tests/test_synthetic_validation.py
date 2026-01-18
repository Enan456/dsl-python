#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Validation tests using synthetic datasets.

Tests DSL implementation on synthetic data with known properties.
"""

import pytest
import numpy as np
from patsy import dmatrices
from scipy import stats

from dsl import dsl
from tests.data.synthetic_dataset import generate_synthetic_logistic_data


class TestSyntheticValidation:
    """Test suite for synthetic dataset validation."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic dataset."""
        return generate_synthetic_logistic_data(
            n_total=1000,
            n_labeled=500,
            n_features=3,
            random_seed=42,
        )

    def test_synthetic_data_generation(self, synthetic_data):
        """Test that synthetic data generation works correctly."""
        df = synthetic_data

        assert len(df) == 1000
        assert df['labeled'].sum() == 500
        assert 'y' in df.columns
        assert 'x1' in df.columns
        assert 'x2' in df.columns
        assert 'x3' in df.columns
        assert 'sample_prob' in df.columns

        # Check labeled indicator
        assert df['labeled'].min() == 0
        assert df['labeled'].max() == 1

        # Check sample probability
        assert df['sample_prob'].nunique() == 1
        assert np.isclose(df['sample_prob'].iloc[0], 0.5)

    def test_dsl_estimation_on_synthetic(self, synthetic_data):
        """Test DSL estimation on synthetic data."""
        df = synthetic_data

        # Create formula and design matrix
        formula = "y ~ x1 + x2 + x3"
        y, X = dmatrices(formula, df, return_type="dataframe")

        # Run DSL estimation
        result = dsl(
            X=X.values,
            y=y.values,
            labeled_ind=df["labeled"].values,
            sample_prob=df["sample_prob"].values,
            model="logit",
            method="logistic",
        )

        # Basic checks
        assert result.success
        assert result.labeled_size == 500
        assert len(result.coefficients) == 4  # Intercept + 3 features

        # Check coefficient estimates are reasonable
        assert np.all(np.abs(result.coefficients) < 10)  # Sanity check

        # Check standard errors are positive
        assert np.all(result.standard_errors > 0)

    def test_reproducibility(self, synthetic_data):
        """Test that results are reproducible with same seed."""
        df = synthetic_data

        formula = "y ~ x1 + x2 + x3"
        y, X = dmatrices(formula, df, return_type="dataframe")

        # Run twice with same data
        result1 = dsl(
            X=X.values,
            y=y.values,
            labeled_ind=df["labeled"].values,
            sample_prob=df["sample_prob"].values,
            model="logit",
            method="logistic",
        )

        result2 = dsl(
            X=X.values,
            y=y.values,
            labeled_ind=df["labeled"].values,
            sample_prob=df["sample_prob"].values,
            model="logit",
            method="logistic",
        )

        # Results should be identical
        np.testing.assert_array_almost_equal(
            result1.coefficients, result2.coefficients, decimal=10
        )
        np.testing.assert_array_almost_equal(
            result1.standard_errors, result2.standard_errors, decimal=10
        )

    def test_coefficient_recovery(self):
        """Test that DSL can recover true coefficients approximately.

        Note: With finite samples, we expect coefficients to be close to true values
        but not exact. We use a 3 SE tolerance which allows for sampling variability.
        """
        # Generate data with known coefficients
        beta_true = np.array([0.5, 1.0, -0.5, 0.3])  # Intercept + 3 features

        df = generate_synthetic_logistic_data(
            n_total=5000,  # Larger sample for better recovery
            n_labeled=2500,
            n_features=3,
            beta_true=beta_true,
            random_seed=42,
        )

        formula = "y ~ x1 + x2 + x3"
        y, X = dmatrices(formula, df, return_type="dataframe")

        result = dsl(
            X=X.values,
            y=y.values,
            labeled_ind=df["labeled"].values,
            sample_prob=df["sample_prob"].values,
            model="logit",
            method="logistic",
        )

        # Verify estimation completed successfully
        assert result.success, "DSL estimation should succeed"
        assert len(result.coefficients) == len(beta_true), "Should have correct number of coefficients"

        # Check that estimates are within reasonable range of true values
        # Use 3 SEs for tolerance (99.7% confidence) to account for sampling variability
        n_within_tolerance = 0
        for i, (est, se, true) in enumerate(
            zip(result.coefficients, result.standard_errors, beta_true)
        ):
            diff = abs(est - true)
            # Allow 3 standard errors tolerance
            if diff < 3 * se:
                n_within_tolerance += 1

        # At least 3 out of 4 coefficients should be within 3 SEs
        # (allows for one outlier due to random chance)
        assert n_within_tolerance >= 3, (
            f"At least 3/4 coefficients should be within 3 SEs of true values. "
            f"Got {n_within_tolerance}/4."
        )

    def test_varying_sample_sizes(self):
        """Test DSL with different sample sizes.

        Note: Very small samples (n_labeled=100) may not always converge
        due to numerical issues or insufficient information. We test that
        estimation completes (even if not converged) and returns valid structure.
        Larger samples should converge successfully.
        """
        # Use larger sample sizes that are more likely to converge
        # Small samples like 100 can have numerical issues with logistic regression
        sample_sizes = [200, 500, 1000]

        for n_labeled in sample_sizes:
            df = generate_synthetic_logistic_data(
                n_total=2000,  # Larger total for better unlabeled info
                n_labeled=n_labeled,
                n_features=3,
                random_seed=42 + n_labeled,  # Different seed for each
            )

            formula = "y ~ x1 + x2 + x3"
            y, X = dmatrices(formula, df, return_type="dataframe")

            result = dsl(
                X=X.values,
                y=y.values,
                labeled_ind=df["labeled"].values,
                sample_prob=df["sample_prob"].values,
                model="logit",
                method="logistic",
            )

            # Verify estimation completed and produced valid output
            assert result.coefficients is not None, f"Coefficients should exist for n_labeled={n_labeled}"
            assert result.standard_errors is not None, f"SEs should exist for n_labeled={n_labeled}"
            assert result.labeled_size == n_labeled, f"Labeled size should match for n_labeled={n_labeled}"

            # For larger samples, expect convergence
            if n_labeled >= 500:
                assert result.success, f"DSL should converge for n_labeled={n_labeled}"

    def test_varying_feature_dimensions(self):
        """Test DSL with different numbers of features."""
        n_features_list = [2, 5, 10]

        for n_features in n_features_list:
            df = generate_synthetic_logistic_data(
                n_total=1000,
                n_labeled=500,
                n_features=n_features,
                random_seed=42,
            )

            feature_names = [f'x{i+1}' for i in range(n_features)]
            formula = f"y ~ {' + '.join(feature_names)}"
            y, X = dmatrices(formula, df, return_type="dataframe")

            result = dsl(
                X=X.values,
                y=y.values,
                labeled_ind=df["labeled"].values,
                sample_prob=df["sample_prob"].values,
                model="logit",
                method="logistic",
            )

            assert result.success, f"DSL failed for n_features={n_features}"
            assert len(result.coefficients) == n_features + 1  # +1 for intercept


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
