#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Extract metrics from comparison reports for CI/CD tracking.

Usage:
    python scripts/extract_metrics.py --input reports/*/report.html --output metrics_summary.json
"""

import argparse
import json
import glob
import re
import sys
from pathlib import Path
from datetime import datetime


def extract_metrics_from_html(html_path: str) -> dict:
    """Extract metrics from HTML comparison report.

    Args:
        html_path: Path to HTML report

    Returns:
        Dictionary of extracted metrics
    """
    with open(html_path, 'r') as f:
        content = f.read()

    # Extract pass rate (looking for pattern like "21 / 21 Passed (100.0%)")
    pass_rate_match = re.search(r'(\d+)\s*/\s*(\d+)\s*Passed\s*\((\d+(?:\.\d+)?)\s*%\)', content)

    if pass_rate_match:
        n_passed = int(pass_rate_match.group(1))
        n_total = int(pass_rate_match.group(2))
        pass_rate = float(pass_rate_match.group(3))
    else:
        n_passed = None
        n_total = None
        pass_rate = None

    # Extract failed count
    failed_match = re.search(r'Failed:\s*(\d+)', content)
    n_failed = int(failed_match.group(1)) if failed_match else None

    # Extract configuration
    config = {}
    coef_tol_match = re.search(r'Coefficients:.*?Absolute=([\d.]+),\s*Relative=([\d.]+)', content)
    if coef_tol_match:
        config['coef_abs_tol'] = float(coef_tol_match.group(1))
        config['coef_rel_tol'] = float(coef_tol_match.group(2))

    se_tol_match = re.search(r'Standard Errors:.*?Absolute=([\d.]+),\s*Relative=([\d.]+)', content)
    if se_tol_match:
        config['se_abs_tol'] = float(se_tol_match.group(1))
        config['se_rel_tol'] = float(se_tol_match.group(2))

    return {
        'n_passed': n_passed,
        'n_total': n_total,
        'n_failed': n_failed,
        'pass_rate': pass_rate,
        'config': config,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Extract metrics from comparison reports',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Glob pattern for input HTML reports'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output JSON file path'
    )

    parser.add_argument(
        '--python-version',
        type=str,
        default='unknown',
        help='Python version for this run'
    )

    args = parser.parse_args()

    # Find report files
    report_files = glob.glob(args.input)

    if not report_files:
        print(f"Warning: No report files found matching '{args.input}'", file=sys.stderr)
        # Create empty metrics
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'python_version': args.python_version,
            'reports_found': 0,
            'metrics': {}
        }
    else:
        # Extract metrics from first report found
        report_path = report_files[0]
        print(f"Extracting metrics from: {report_path}")

        extracted = extract_metrics_from_html(report_path)

        metrics = {
            'timestamp': datetime.now().isoformat(),
            'python_version': args.python_version,
            'reports_found': len(report_files),
            'report_path': str(Path(report_path).name),
            **extracted
        }

    # Save metrics
    with open(args.output, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to: {args.output}")

    # Print summary
    if metrics.get('pass_rate') is not None:
        print(f"\nSummary:")
        print(f"  Pass Rate: {metrics['pass_rate']:.1f}%")
        print(f"  Passed: {metrics['n_passed']}/{metrics['n_total']}")
        print(f"  Failed: {metrics['n_failed']}")


if __name__ == '__main__':
    main()
