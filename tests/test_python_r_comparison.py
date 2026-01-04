#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Automated Python-R comparison test suite.
Tests DSL Python implementation against R reference results with configurable tolerances.
"""

import logging
import sys
import pytest
import numpy as np
from patsy import dmatrices
from scipy import stats

from dsl import dsl
from tests.data.compare_panchen import load_panchen_data, prepare_data_for_dsl
from tests.comparison.r_reference import load_panchen_r_reference
from tests.comparison.comparator import (
    compare_implementations,
    assert_implementations_match,
    ComparisonConfig,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class TestPanChenComparison:
    """Test suite for PanChen dataset Python-R comparison."""

    @pytest.fixture(scope="class")
    def panchen_data(self):
        """Load and prepare PanChen data."""
        logger.info("Loading PanChen dataset")
        data = load_panchen_data()
        df = prepare_data_for_dsl(data)
        return df

    @pytest.fixture(scope="class")
    def python_result(self, panchen_data):
        """Run Python DSL estimation."""
        logger.info("Running Python DSL estimation")

        formula = (
            "SendOrNot ~ countyWrong + prefecWrong + connect2b + "
            "prevalence + regionj + groupIssue"
        )

        y, X = dmatrices(formula, panchen_data, return_type="dataframe")

        result = dsl(
            X=X.values,
            y=y.values,
            labeled_ind=panchen_data["labeled"].values,
            sample_prob=panchen_data["sample_prob"].values,
            model="logit",
            method="logistic",
        )

        logger.info(f"Python estimation complete: {result.labeled_size} labeled observations")
        return result, formula

    @pytest.fixture(scope="class")
    def r_reference(self):
        """Load R reference results."""
        logger.info("Loading R reference results")
        return load_panchen_r_reference()

    @pytest.fixture(scope="class")
    def comparison_result(self, python_result, r_reference):
        """Run full comparison between Python and R implementations."""
        result, formula = python_result
        r_ref = r_reference

        # Extract variable names from formula
        formula_parts = formula.split("~")
        terms = ["(Intercept)"] + [t.strip() for t in formula_parts[1].split("+")]

        # Create Python dictionaries
        python_coefs = {term: result.coefficients[i] for i, term in enumerate(terms)}
        python_ses = {term: result.standard_errors[i] for i, term in enumerate(terms)}

        # Calculate Python p-values
        t_stats = result.coefficients / result.standard_errors
        p_values = 2 * (1 - stats.norm.cdf(np.abs(t_stats)))
        python_pvalues = {term: p_values[i] for i, term in enumerate(terms)}

        # Configure tolerances
        config = ComparisonConfig(
            coef_abs_tol=0.01,  # ±0.01 for small coefficients
            coef_rel_tol=0.05,  # ±5% for larger coefficients
            se_abs_tol=0.01,  # ±0.01 for small SEs
            se_rel_tol=0.10,  # ±10% for larger SEs
            pvalue_abs_tol=0.05,  # ±0.05 for p-values
        )

        # Run comparison
        comparison = compare_implementations(
            python_coefs=python_coefs,
            python_ses=python_ses,
            python_pvalues=python_pvalues,
            r_coefs=r_ref.coefficients,
            r_ses=r_ref.standard_errors,
            r_pvalues=r_ref.p_values,
            config=config,
            name="PanChen Python-R Comparison",
        )

        logger.info(f"\nComparison Summary:")
        logger.info(f"Total: {comparison.n_comparisons}, "
                   f"Passed: {comparison.n_passed}, "
                   f"Failed: {comparison.n_failed}")

        return comparison

    def test_coefficients_match(self, comparison_result):
        """Test that all coefficients match within tolerance."""
        failed = [c for c in comparison_result.coefficient_comparisons if not c.passed]

        if failed:
            msg = "\nCoefficient mismatches:\n"
            for comp in failed:
                msg += f"  {comp.message}\n"
            pytest.fail(msg)

        assert len(failed) == 0, "All coefficients should match within tolerance"

    def test_standard_errors_match(self, comparison_result):
        """Test that all standard errors match within tolerance."""
        failed = [c for c in comparison_result.se_comparisons if not c.passed]

        if failed:
            msg = "\nStandard error mismatches:\n"
            for comp in failed:
                msg += f"  {comp.message}\n"
            pytest.fail(msg)

        assert len(failed) == 0, "All standard errors should match within tolerance"

    def test_pvalues_match(self, comparison_result):
        """Test that all p-values match within tolerance."""
        failed = [c for c in comparison_result.pvalue_comparisons if not c.passed]

        if failed:
            msg = "\nP-value mismatches:\n"
            for comp in failed:
                msg += f"  {comp.message}\n"
            pytest.fail(msg)

        assert len(failed) == 0, "All p-values should match within tolerance"

    def test_overall_comparison(self, comparison_result):
        """Test that overall comparison passes."""
        logger.info(comparison_result.summary())

        # This will raise AssertionError with detailed summary if any comparison fails
        assert_implementations_match(comparison_result)

    def test_specific_coefficients(self, python_result, r_reference):
        """Test specific coefficient values for critical variables."""
        result, formula = python_result
        r_ref = r_reference

        # Test critical coefficients
        critical_vars = ["(Intercept)", "prefecWrong", "groupIssue"]

        formula_parts = formula.split("~")
        terms = ["(Intercept)"] + [t.strip() for t in formula_parts[1].split("+")]

        for var in critical_vars:
            if var in r_ref.coefficients:
                idx = terms.index(var)
                py_val = result.coefficients[idx]
                r_val = r_ref.coefficients[var]

                # Use relative tolerance for larger values
                if abs(r_val) > 0.1:
                    rel_diff = abs(py_val - r_val) / abs(r_val)
                    assert rel_diff < 0.05, (
                        f"Coefficient {var}: Python={py_val:.6f}, R={r_val:.6f}, "
                        f"RelDiff={rel_diff:.4f} exceeds 5% tolerance"
                    )
                else:
                    abs_diff = abs(py_val - r_val)
                    assert abs_diff < 0.01, (
                        f"Coefficient {var}: Python={py_val:.6f}, R={r_val:.6f}, "
                        f"AbsDiff={abs_diff:.6f} exceeds 0.01 tolerance"
                    )

    def test_model_properties(self, python_result, r_reference):
        """Test that model properties match."""
        result, _ = python_result
        r_ref = r_reference

        # Test number of observations
        assert result.labeled_size == r_ref.n_obs, (
            f"Labeled observations mismatch: Python={result.labeled_size}, "
            f"R={r_ref.n_obs}"
        )

        # Test number of coefficients
        assert len(result.coefficients) == len(r_ref.coefficients), (
            f"Number of coefficients mismatch: Python={len(result.coefficients)}, "
            f"R={len(r_ref.coefficients)}"
        )

        # Test model type
        assert result.model == r_ref.model, (
            f"Model type mismatch: Python={result.model}, R={r_ref.model}"
        )


class TestComparisonFramework:
    """Test the comparison framework itself."""

    def test_comparison_config_defaults(self):
        """Test that default configuration is reasonable."""
        config = ComparisonConfig()

        assert config.coef_abs_tol == 0.01
        assert config.se_abs_tol == 0.01
        assert config.coef_rel_tol == 0.05
        assert config.se_rel_tol == 0.10

    def test_comparison_result_properties(self, comparison_result):
        """Test ComparisonResult properties."""
        assert comparison_result.n_comparisons > 0
        assert comparison_result.n_passed >= 0
        assert comparison_result.n_failed >= 0
        assert comparison_result.n_comparisons == (
            comparison_result.n_passed + comparison_result.n_failed
        )

    def test_summary_generation(self, comparison_result):
        """Test that summary is generated correctly."""
        summary = comparison_result.summary()

        assert "Comparison Summary" in summary
        assert "Total Comparisons" in summary
        assert "Passed" in summary
        assert "Configuration" in summary


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
