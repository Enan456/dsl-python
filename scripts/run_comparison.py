#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Convenience script to run Python-R comparison with report generation.

Usage:
    python scripts/run_comparison.py [--strict] [--report-dir reports]
"""

import argparse
import sys
import logging
import json
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy import stats
from patsy import dmatrices

from dsl import dsl
from tests.data.compare_panchen import load_panchen_data, prepare_data_for_dsl
from tests.comparison.r_reference import load_panchen_r_reference
from tests.comparison.comparator import (
    compare_implementations,
    ComparisonConfig,
    assert_implementations_match,
)
from tests.comparison.report_generator import (
    ComparisonReportGenerator,
    ReportConfig,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def run_comparison(strict: bool = False, report_dir: str = "reports", show_all: bool = False, save_metrics: str = None):
    """Run Python-R comparison with report generation.

    Args:
        strict: If True, use strict tolerances
        report_dir: Directory for output reports
        show_all: If True, show all comparisons in report (not just failures)
        save_metrics: Path to save metrics JSON file

    Returns:
        ComparisonResult object
    """
    logger.info("Starting Python-R comparison")

    # Load and prepare data
    logger.info("Loading PanChen dataset")
    data = load_panchen_data()
    df = prepare_data_for_dsl(data)

    # Run Python DSL estimation
    logger.info("Running Python DSL estimation")
    formula = (
        "SendOrNot ~ countyWrong + prefecWrong + connect2b + "
        "prevalence + regionj + groupIssue"
    )

    y, X = dmatrices(formula, df, return_type="dataframe")
    result = dsl(
        X=X.values,
        y=y.values,
        labeled_ind=df["labeled"].values,
        sample_prob=df["sample_prob"].values,
        model="logit",
        method="logistic",
    )

    logger.info(f"Python estimation complete: {result.labeled_size} labeled observations")

    # Extract Python results
    formula_parts = formula.split("~")
    terms = ["(Intercept)"] + [t.strip() for t in formula_parts[1].split("+")]

    py_coefs = {term: result.coefficients[i] for i, term in enumerate(terms)}
    py_ses = {term: result.standard_errors[i] for i, term in enumerate(terms)}

    # Calculate p-values
    t_stats = result.coefficients / result.standard_errors
    p_vals = 2 * (1 - stats.norm.cdf(np.abs(t_stats)))
    py_pvals = {term: p_vals[i] for i, term in enumerate(terms)}

    # Load R reference
    logger.info("Loading R reference results")
    r_ref = load_panchen_r_reference()

    # Configure comparison
    if strict:
        logger.info("Using strict tolerance configuration")
        config = ComparisonConfig(
            coef_abs_tol=0.001,
            coef_rel_tol=0.01,
            se_abs_tol=0.001,
            se_rel_tol=0.05,
            pvalue_abs_tol=0.01,
        )
    else:
        logger.info("Using default tolerance configuration")
        config = ComparisonConfig()

    # Run comparison
    logger.info("Comparing Python and R implementations")
    comparison = compare_implementations(
        python_coefs=py_coefs,
        python_ses=py_ses,
        python_pvalues=py_pvals,
        r_coefs=r_ref.coefficients,
        r_ses=r_ref.standard_errors,
        r_pvalues=r_ref.p_values,
        config=config,
        name="PanChen Python-R Comparison",
    )

    # Print summary
    print("\n" + "=" * 80)
    print(comparison.summary())
    print("=" * 80 + "\n")

    # Generate detailed report with visualizations
    logger.info("Generating comparison report")
    report_config = ReportConfig(
        output_dir=report_dir,
        include_plots=True,
        show_all_comparisons=show_all,
    )

    generator = ComparisonReportGenerator(report_config)
    report_path = generator.generate_full_report(
        comparison,
        "panchen_comparison_strict" if strict else "panchen_comparison",
    )

    print(f"\nDetailed report generated: {report_path}")
    print(f"Open in browser: file://{Path(report_path).absolute()}\n")

    # Save metrics if requested
    if save_metrics:
        logger.info(f"Saving metrics to {save_metrics}")
        metrics_data = {
            "timestamp": datetime.now().isoformat(),
            "dataset": "panchen",
            "strict_mode": strict,
            "n_comparisons": comparison.n_comparisons,
            "n_passed": comparison.n_passed,
            "n_failed": comparison.n_failed,
            "pass_rate": comparison.n_passed / comparison.n_comparisons if comparison.n_comparisons > 0 else 0,
            "config": {
                "coef_abs_tol": comparison.config.coef_abs_tol,
                "coef_rel_tol": comparison.config.coef_rel_tol,
                "se_abs_tol": comparison.config.se_abs_tol,
                "se_rel_tol": comparison.config.se_rel_tol,
                "pvalue_abs_tol": comparison.config.pvalue_abs_tol,
            },
            "by_metric": {
                "coefficients": {
                    "total": len(comparison.coefficient_comparisons),
                    "passed": sum(1 for c in comparison.coefficient_comparisons if c.passed),
                    "failed": sum(1 for c in comparison.coefficient_comparisons if not c.passed),
                },
                "standard_errors": {
                    "total": len(comparison.se_comparisons),
                    "passed": sum(1 for c in comparison.se_comparisons if c.passed),
                    "failed": sum(1 for c in comparison.se_comparisons if not c.passed),
                },
                "pvalues": {
                    "total": len(comparison.pvalue_comparisons),
                    "passed": sum(1 for c in comparison.pvalue_comparisons if c.passed),
                    "failed": sum(1 for c in comparison.pvalue_comparisons if not c.passed),
                },
            },
            "failures": [
                {
                    "metric": failure.name,
                    "variable": failure.variable,
                    "python_value": failure.python_value,
                    "r_value": failure.r_value,
                    "abs_diff": failure.abs_diff,
                    "rel_diff": failure.rel_diff,
                }
                for failure in comparison.get_failures()
            ],
        }

        with open(save_metrics, "w") as f:
            json.dump(metrics_data, f, indent=2)
        logger.info(f"Metrics saved to {save_metrics}")

    # Return comparison result for further analysis
    return comparison


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Python-R comparison with report generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Use strict tolerance configuration",
    )

    parser.add_argument(
        "--report-dir",
        type=str,
        default="reports",
        help="Directory for output reports",
    )

    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show all comparisons in report (not just failures)",
    )

    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Exit with error if any comparison fails",
    )

    parser.add_argument(
        "--save-metrics",
        type=str,
        default=None,
        help="Path to save metrics JSON file",
    )

    args = parser.parse_args()

    try:
        comparison = run_comparison(
            strict=args.strict,
            report_dir=args.report_dir,
            show_all=args.show_all,
            save_metrics=args.save_metrics,
        )

        # Exit with error if comparison failed and --fail-on-mismatch is set
        if args.fail_on_mismatch and not comparison.all_passed:
            logger.error("Comparison failed! Exiting with error.")
            sys.exit(1)

        logger.info("Comparison complete!")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Error during comparison: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
