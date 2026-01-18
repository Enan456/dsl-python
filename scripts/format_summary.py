#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Format comparison summary for GitHub Actions summary.

Usage:
    python scripts/format_summary.py --metrics all-metrics/
"""

import argparse
import json
import glob
import sys


def format_summary(metrics_list: list) -> str:
    """Format metrics as GitHub-flavored markdown summary.

    Args:
        metrics_list: List of metrics dictionaries

    Returns:
        Formatted markdown string
    """
    if not metrics_list:
        return "⚠️ No metrics data available"

    # Group by Python version
    by_version = {}
    for metrics in metrics_list:
        py_version = metrics.get('python_version', 'unknown')
        if py_version not in by_version:
            by_version[py_version] = []
        by_version[py_version].append(metrics)

    # Calculate overall stats
    all_pass_rates = [m['pass_rate'] for m in metrics_list if m.get('pass_rate') is not None]
    overall_pass_rate = sum(all_pass_rates) / len(all_pass_rates) if all_pass_rates else 0.0

    # Determine status emoji
    if overall_pass_rate >= 95.0:
        status_emoji = "✅"
        status_text = "PASSING"
    elif overall_pass_rate >= 80.0:
        status_emoji = "⚠️"
        status_text = "DEGRADED"
    else:
        status_emoji = "❌"
        status_text = "FAILING"

    # Build markdown
    lines = [
        f"{status_emoji} **Status: {status_text}**",
        "",
        f"**Overall Pass Rate:** {overall_pass_rate:.1f}%",
        "",
        "| Python Version | Pass Rate | Passed | Failed | Total |",
        "|----------------|-----------|--------|--------|-------|",
    ]

    for version in sorted(by_version.keys()):
        version_metrics = by_version[version]
        # Use latest metrics for this version
        latest = version_metrics[-1] if version_metrics else {}

        pass_rate = latest.get('pass_rate', 0.0)
        n_passed = latest.get('n_passed', 0)
        n_failed = latest.get('n_failed', 0)
        n_total = latest.get('n_total', 0)

        # Status emoji for this version
        if pass_rate >= 95.0:
            version_emoji = "✅"
        elif pass_rate >= 80.0:
            version_emoji = "⚠️"
        else:
            version_emoji = "❌"

        lines.append(
            f"| {version_emoji} Python {version} | {pass_rate:.1f}% | {n_passed} | {n_failed} | {n_total} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "📊 Detailed reports available in workflow artifacts.",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Format comparison summary for GitHub Actions',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--metrics',
        type=str,
        required=True,
        help='Directory containing metric artifacts'
    )

    args = parser.parse_args()

    # Load all metrics files
    metrics_files = glob.glob(f"{args.metrics}/**/metrics*.json", recursive=True)

    metrics_list = []
    for mfile in metrics_files:
        try:
            with open(mfile, 'r') as f:
                metrics_list.append(json.load(f))
        except Exception as e:
            print(f"Warning: Could not load {mfile}: {e}", file=sys.stderr)

    # Format and print summary
    summary = format_summary(metrics_list)
    print(summary)


if __name__ == '__main__':
    main()
