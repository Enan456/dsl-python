#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comparison report generator with visualizations.
Generates HTML and PDF reports comparing Python and R implementations.
"""

import os
from typing import List, Optional
from dataclasses import dataclass
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import seaborn as sns
from datetime import datetime

from tests.comparison.comparator import ComparisonResult, ComparisonMetric


@dataclass
class ReportConfig:
    """Configuration for report generation."""

    output_dir: str = "reports"
    include_plots: bool = True
    plot_format: str = "png"
    plot_dpi: int = 300
    show_all_comparisons: bool = False  # If False, only show failures


class ComparisonReportGenerator:
    """Generate detailed comparison reports with visualizations."""

    def __init__(self, config: Optional[ReportConfig] = None):
        """Initialize report generator.

        Args:
            config: ReportConfig object (uses defaults if None)
        """
        self.config = config or ReportConfig()
        os.makedirs(self.config.output_dir, exist_ok=True)

    def generate_comparison_plot(
        self, comparisons: List[ComparisonMetric], metric_name: str, output_path: str
    ):
        """Generate comparison scatter plot.

        Args:
            comparisons: List of ComparisonMetric objects
            metric_name: Name of metric (e.g., "Coefficients", "Standard Errors")
            output_path: Path to save plot
        """
        if not comparisons:
            return

        # Extract data
        variables = [c.variable for c in comparisons]
        python_vals = [c.python_value for c in comparisons]
        r_vals = [c.r_value for c in comparisons]
        passed = [c.passed for c in comparisons]

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Python vs R scatter
        colors = ["green" if p else "red" for p in passed]
        ax1.scatter(r_vals, python_vals, c=colors, alpha=0.6, s=100)

        # Add diagonal line
        min_val = min(min(r_vals), min(python_vals))
        max_val = max(max(r_vals), max(python_vals))
        ax1.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.3, label="Perfect Agreement")

        ax1.set_xlabel("R Values", fontsize=12)
        ax1.set_ylabel("Python Values", fontsize=12)
        ax1.set_title(f"{metric_name}: Python vs R", fontsize=14, fontweight="bold")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Relative differences
        rel_diffs = [c.rel_diff * 100 for c in comparisons]  # Convert to percentage
        colors = ["green" if p else "red" for p in passed]

        bars = ax2.barh(range(len(variables)), rel_diffs, color=colors, alpha=0.6)
        ax2.set_yticks(range(len(variables)))
        ax2.set_yticklabels(variables, fontsize=10)
        ax2.set_xlabel("Relative Difference (%)", fontsize=12)
        ax2.set_title(f"{metric_name}: Relative Differences", fontsize=14, fontweight="bold")
        ax2.axvline(x=0, color="k", linestyle="--", alpha=0.3)
        ax2.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config.plot_dpi, bbox_inches="tight")
        plt.close()

    def generate_summary_plot(self, result: ComparisonResult, output_path: str):
        """Generate summary visualization.

        Args:
            result: ComparisonResult object
            output_path: Path to save plot
        """
        # Create summary data
        categories = ["Coefficients", "Standard Errors", "P-values"]
        n_total = [
            len(result.coefficient_comparisons),
            len(result.se_comparisons),
            len(result.pvalue_comparisons),
        ]
        n_passed = [
            sum(1 for c in result.coefficient_comparisons if c.passed),
            sum(1 for c in result.se_comparisons if c.passed),
            sum(1 for c in result.pvalue_comparisons if c.passed),
        ]
        n_failed = [t - p for t, p in zip(n_total, n_passed)]

        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=(10, 6))

        x = range(len(categories))
        width = 0.5

        bars1 = ax.bar(x, n_passed, width, label="Passed", color="green", alpha=0.7)
        bars2 = ax.bar(x, n_failed, width, bottom=n_passed, label="Failed", color="red", alpha=0.7)

        ax.set_ylabel("Number of Comparisons", fontsize=12)
        ax.set_title(
            f"Comparison Summary: {result.name}\n"
            f"Overall: {result.n_passed}/{result.n_comparisons} passed "
            f"({100 * result.n_passed / result.n_comparisons:.1f}%)",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for i, (p, f) in enumerate(zip(n_passed, n_failed)):
            if p > 0:
                ax.text(i, p / 2, str(p), ha="center", va="center", fontweight="bold")
            if f > 0:
                ax.text(i, p + f / 2, str(f), ha="center", va="center", fontweight="bold")

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config.plot_dpi, bbox_inches="tight")
        plt.close()

    def generate_html_report(self, result: ComparisonResult, output_path: str):
        """Generate HTML report.

        Args:
            result: ComparisonResult object
            output_path: Path to save HTML report
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Python-R Comparison Report: {result.name}</title>
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
        .summary {{
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .passed {{
            background-color: #d4edda;
            color: #155724;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }}
        .failed {{
            background-color: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
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
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .metric-section {{
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .plot {{
            text-align: center;
            margin: 20px 0;
        }}
        .plot img {{
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .pass-rate {{
            font-size: 2em;
            font-weight: bold;
            color: {"#27ae60" if result.all_passed else "#e74c3c"};
        }}
        .config {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Python-R Comparison Report</h1>
        <h2>{result.name}</h2>
        <p>Generated: {timestamp}</p>
    </div>

    <div class="summary">
        <h2>Summary</h2>
        <div class="pass-rate">
            {result.n_passed} / {result.n_comparisons} Passed
            ({100 * result.n_passed / result.n_comparisons:.1f}%)
        </div>
        <p><strong>Total Comparisons:</strong> {result.n_comparisons}</p>
        <p><strong>Passed:</strong> {result.n_passed}</p>
        <p><strong>Failed:</strong> {result.n_failed}</p>

        <div class="config">
            <h3>Tolerance Configuration</h3>
            <ul>
                <li><strong>Coefficients:</strong> Absolute={result.config.coef_abs_tol}, Relative={result.config.coef_rel_tol}</li>
                <li><strong>Standard Errors:</strong> Absolute={result.config.se_abs_tol}, Relative={result.config.se_rel_tol}</li>
                <li><strong>P-values:</strong> Absolute={result.config.pvalue_abs_tol}, Relative={result.config.pvalue_rel_tol}</li>
            </ul>
        </div>
    </div>
"""

        # Add plots if generated
        plot_dir = os.path.dirname(output_path)
        summary_plot = os.path.join(plot_dir, "summary.png")
        if os.path.exists(summary_plot):
            html += f"""
    <div class="plot">
        <h2>Summary Visualization</h2>
        <img src="summary.png" alt="Summary Plot">
    </div>
"""

        # Add coefficient comparisons
        if result.coefficient_comparisons:
            html += self._generate_metric_section(
                "Coefficients",
                result.coefficient_comparisons,
                plot_dir,
                "coefficients.png"
            )

        # Add SE comparisons
        if result.se_comparisons:
            html += self._generate_metric_section(
                "Standard Errors",
                result.se_comparisons,
                plot_dir,
                "standard_errors.png"
            )

        # Add p-value comparisons
        if result.pvalue_comparisons:
            html += self._generate_metric_section(
                "P-values",
                result.pvalue_comparisons,
                plot_dir,
                "pvalues.png"
            )

        html += """
</body>
</html>
"""

        with open(output_path, "w") as f:
            f.write(html)

    def _generate_metric_section(
        self, title: str, comparisons: List[ComparisonMetric], plot_dir: str, plot_filename: str
    ) -> str:
        """Generate HTML section for a metric type."""
        failed_comparisons = [c for c in comparisons if not c.passed]
        n_passed = len(comparisons) - len(failed_comparisons)

        html = f"""
    <div class="metric-section">
        <h2>{title}</h2>
        <p><strong>Passed:</strong> {n_passed} / {len(comparisons)}</p>
"""

        # Add plot if it exists
        plot_path = os.path.join(plot_dir, plot_filename)
        if os.path.exists(plot_path):
            html += f"""
        <div class="plot">
            <img src="{plot_filename}" alt="{title} Plot">
        </div>
"""

        # Add table of comparisons (show failures or all based on config)
        comparisons_to_show = failed_comparisons if not self.config.show_all_comparisons else comparisons

        if comparisons_to_show:
            html += """
        <table>
            <thead>
                <tr>
                    <th>Variable</th>
                    <th>Python</th>
                    <th>R</th>
                    <th>Abs Diff</th>
                    <th>Rel Diff</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
"""
            for comp in comparisons_to_show:
                status = "✓ PASS" if comp.passed else "✗ FAIL"
                row_class = "passed" if comp.passed else "failed"
                html += f"""
                <tr class="{row_class}">
                    <td>{comp.variable}</td>
                    <td>{comp.python_value:.6f}</td>
                    <td>{comp.r_value:.6f}</td>
                    <td>{comp.abs_diff:.6f}</td>
                    <td>{comp.rel_diff * 100:.2f}%</td>
                    <td>{status}</td>
                </tr>
"""
            html += """
            </tbody>
        </table>
"""

        html += """
    </div>
"""
        return html

    def generate_full_report(self, result: ComparisonResult, report_name: str = "comparison_report"):
        """Generate complete report with all visualizations.

        Args:
            result: ComparisonResult object
            report_name: Base name for report files

        Returns:
            Path to generated HTML report
        """
        # Create timestamped directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = os.path.join(self.config.output_dir, f"{report_name}_{timestamp}")
        os.makedirs(report_dir, exist_ok=True)

        # Generate plots
        if self.config.include_plots:
            # Summary plot
            summary_plot_path = os.path.join(report_dir, "summary.png")
            self.generate_summary_plot(result, summary_plot_path)

            # Coefficient plot
            if result.coefficient_comparisons:
                coef_plot_path = os.path.join(report_dir, "coefficients.png")
                self.generate_comparison_plot(
                    result.coefficient_comparisons, "Coefficients", coef_plot_path
                )

            # SE plot
            if result.se_comparisons:
                se_plot_path = os.path.join(report_dir, "standard_errors.png")
                self.generate_comparison_plot(
                    result.se_comparisons, "Standard Errors", se_plot_path
                )

            # P-value plot
            if result.pvalue_comparisons:
                pval_plot_path = os.path.join(report_dir, "pvalues.png")
                self.generate_comparison_plot(
                    result.pvalue_comparisons, "P-values", pval_plot_path
                )

        # Generate HTML report
        html_path = os.path.join(report_dir, "report.html")
        self.generate_html_report(result, html_path)

        print(f"Report generated: {html_path}")
        return html_path
