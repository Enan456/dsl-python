#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Aggregate metrics from multiple test runs and generate dashboard.

Usage:
    python scripts/aggregate_metrics.py --input all-metrics/ --output dashboard/
"""

import argparse
import json
import glob
import sys
from pathlib import Path
from datetime import datetime
import os


def load_metrics_from_artifacts(input_dir: str) -> list:
    """Load all metrics JSON files from artifacts directory.

    Args:
        input_dir: Directory containing metric artifacts

    Returns:
        List of metrics dictionaries
    """
    metrics_files = glob.glob(f"{input_dir}/**/metrics*.json", recursive=True)

    all_metrics = []
    for mfile in metrics_files:
        try:
            with open(mfile, 'r') as f:
                data = json.load(f)
                data['source_file'] = str(Path(mfile).name)
                all_metrics.append(data)
        except Exception as e:
            print(f"Warning: Could not load {mfile}: {e}", file=sys.stderr)

    return all_metrics


def aggregate_metrics(metrics_list: list) -> dict:
    """Aggregate metrics across multiple runs.

    Args:
        metrics_list: List of metrics dictionaries

    Returns:
        Aggregated metrics dictionary
    """
    if not metrics_list:
        return {
            'timestamp': datetime.now().isoformat(),
            'n_runs': 0,
            'overall_pass_rate': 0.0,
            'by_python_version': {},
            'status': 'no_data',
        }

    # Group by Python version
    by_version = {}
    for metrics in metrics_list:
        py_version = metrics.get('python_version', 'unknown')
        if py_version not in by_version:
            by_version[py_version] = []
        by_version[py_version].append(metrics)

    # Calculate aggregates
    all_pass_rates = [m['pass_rate'] for m in metrics_list if m.get('pass_rate') is not None]
    overall_pass_rate = sum(all_pass_rates) / len(all_pass_rates) if all_pass_rates else 0.0

    version_summaries = {}
    for version, version_metrics in by_version.items():
        pass_rates = [m['pass_rate'] for m in version_metrics if m.get('pass_rate') is not None]
        version_summaries[version] = {
            'n_runs': len(version_metrics),
            'avg_pass_rate': sum(pass_rates) / len(pass_rates) if pass_rates else 0.0,
            'latest': version_metrics[-1] if version_metrics else {},
        }

    # Determine overall status
    if overall_pass_rate >= 95.0:
        status = 'passing'
    elif overall_pass_rate >= 80.0:
        status = 'degraded'
    else:
        status = 'failing'

    return {
        'timestamp': datetime.now().isoformat(),
        'n_runs': len(metrics_list),
        'overall_pass_rate': overall_pass_rate,
        'by_python_version': version_summaries,
        'status': status,
        'all_metrics': metrics_list,
    }


def load_historical_data(historical_dir: str) -> list:
    """Load historical metrics data.

    Args:
        historical_dir: Directory containing historical metrics

    Returns:
        List of historical metrics dictionaries
    """
    if not os.path.exists(historical_dir):
        return []

    history_files = glob.glob(f"{historical_dir}/metrics_*.json")
    history = []

    for hfile in sorted(history_files):
        try:
            with open(hfile, 'r') as f:
                history.append(json.load(f))
        except Exception as e:
            print(f"Warning: Could not load {hfile}: {e}", file=sys.stderr)

    return history


def generate_dashboard_html(aggregated: dict, output_path: str):
    """Generate dashboard HTML file.

    Args:
        aggregated: Aggregated metrics dictionary
        output_path: Path to save dashboard HTML
    """
    timestamp = aggregated['timestamp']
    status = aggregated['status']
    overall_pass_rate = aggregated['overall_pass_rate']
    by_version = aggregated['by_python_version']

    # Status color
    status_colors = {
        'passing': '#27ae60',
        'degraded': '#f39c12',
        'failing': '#e74c3c',
        'no_data': '#95a5a6',
    }
    status_color = status_colors.get(status, '#95a5a6')

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Python-R Comparison Dashboard</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .status {{
            text-align: center;
            padding: 30px;
            background-color: white;
            border-radius: 5px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .status-badge {{
            display: inline-block;
            padding: 10px 20px;
            border-radius: 20px;
            font-size: 18px;
            font-weight: bold;
            color: white;
            background-color: {status_color};
            text-transform: uppercase;
        }}
        .pass-rate {{
            font-size: 3em;
            font-weight: bold;
            color: {status_color};
            margin: 20px 0;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        .metric-card {{
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-title {{
            font-size: 14px;
            color: #7f8c8d;
            margin-bottom: 10px;
        }}
        .metric-value {{
            font-size: 28px;
            font-weight: bold;
            color: #2c3e50;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background-color: white;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #34495e;
            color: white;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Python-R Comparison Dashboard</h1>
        <p>Last Updated: {timestamp}</p>
    </div>

    <div class="status">
        <div class="status-badge">{status}</div>
        <div class="pass-rate">{overall_pass_rate:.1f}%</div>
        <p>Overall Pass Rate</p>
    </div>

    <h2>Metrics by Python Version</h2>
    <div class="metrics-grid">
"""

    for version, metrics in sorted(by_version.items()):
        html += f"""
        <div class="metric-card">
            <div class="metric-title">Python {version}</div>
            <div class="metric-value">{metrics['avg_pass_rate']:.1f}%</div>
            <p>{metrics['n_runs']} run(s)</p>
        </div>
"""

    html += """
    </div>

    <h2>Detailed Results</h2>
    <table>
        <thead>
            <tr>
                <th>Python Version</th>
                <th>Pass Rate</th>
                <th>Passed</th>
                <th>Failed</th>
                <th>Total</th>
            </tr>
        </thead>
        <tbody>
"""

    for version, metrics in sorted(by_version.items()):
        latest = metrics['latest']
        pass_rate = latest.get('pass_rate', 0.0)
        n_passed = latest.get('n_passed', 0)
        n_failed = latest.get('n_failed', 0)
        n_total = latest.get('n_total', 0)

        html += f"""
            <tr>
                <td>Python {version}</td>
                <td>{pass_rate:.1f}%</td>
                <td>{n_passed}</td>
                <td>{n_failed}</td>
                <td>{n_total}</td>
            </tr>
"""

    html += """
        </tbody>
    </table>
</body>
</html>
"""

    with open(output_path, 'w') as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate metrics and generate dashboard',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Directory containing metric artifacts'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for dashboard'
    )

    parser.add_argument(
        '--historical-data',
        type=str,
        default=None,
        help='Directory containing historical metrics'
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Load metrics from artifacts
    print(f"Loading metrics from: {args.input}")
    metrics_list = load_metrics_from_artifacts(args.input)
    print(f"Loaded {len(metrics_list)} metric file(s)")

    # Load historical data if available
    if args.historical_data:
        print(f"Loading historical data from: {args.historical_data}")
        history = load_historical_data(args.historical_data)
        print(f"Loaded {len(history)} historical record(s)")

    # Aggregate metrics
    aggregated = aggregate_metrics(metrics_list)

    # Save aggregated metrics
    metrics_file = os.path.join(args.output, f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(metrics_file, 'w') as f:
        json.dump(aggregated, f, indent=2)
    print(f"Aggregated metrics saved to: {metrics_file}")

    # Save summary (for badge generation)
    summary_file = os.path.join(args.output, "metrics_summary.json")
    summary = {
        'timestamp': aggregated['timestamp'],
        'status': aggregated['status'],
        'overall_pass_rate': aggregated['overall_pass_rate'],
        'n_runs': aggregated['n_runs'],
    }
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_file}")

    # Generate dashboard HTML
    dashboard_file = os.path.join(args.output, "index.html")
    generate_dashboard_html(aggregated, dashboard_file)
    print(f"Dashboard generated: {dashboard_file}")

    # Print summary
    print(f"\nDashboard Summary:")
    print(f"  Status: {aggregated['status'].upper()}")
    print(f"  Overall Pass Rate: {aggregated['overall_pass_rate']:.1f}%")
    print(f"  Total Runs: {aggregated['n_runs']}")


if __name__ == '__main__':
    main()
