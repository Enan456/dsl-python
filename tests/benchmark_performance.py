#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Performance benchmarking suite for DSL implementation.

Measures execution time, memory usage, and scalability across different
configurations and compares Python implementation performance.
"""

import time
import psutil
import os
import json
import argparse
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from patsy import dmatrices

from dsl import dsl
from tests.data.synthetic_dataset import generate_synthetic_logistic_data


class PerformanceBenchmark:
    """Performance benchmarking for DSL implementation."""

    def __init__(self):
        """Initialize benchmark."""
        self.results = []
        self.process = psutil.Process(os.getpid())

    def measure_execution_time(self, func, *args, **kwargs) -> Tuple[float, any]:
        """Measure execution time of a function.

        Args:
            func: Function to benchmark
            *args, **kwargs: Arguments to pass to function

        Returns:
            Tuple of (execution_time_seconds, result)
        """
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        return end - start, result

    def measure_memory_usage(self) -> Dict[str, float]:
        """Measure current memory usage.

        Returns:
            Dictionary of memory metrics in MB
        """
        mem_info = self.process.memory_info()
        return {
            'rss_mb': mem_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': mem_info.vms / 1024 / 1024,  # Virtual Memory Size
        }

    def benchmark_dsl_estimation(
        self,
        n_total: int,
        n_labeled: int,
        n_features: int,
        random_seed: int = 42,
    ) -> Dict:
        """Benchmark DSL estimation on synthetic data.

        Args:
            n_total: Total observations
            n_labeled: Labeled observations
            n_features: Number of features
            random_seed: Random seed

        Returns:
            Dictionary of benchmark results
        """
        # Generate data
        print(f"  Generating data: n_total={n_total}, n_labeled={n_labeled}, n_features={n_features}")
        df = generate_synthetic_logistic_data(
            n_total=n_total,
            n_labeled=n_labeled,
            n_features=n_features,
            random_seed=random_seed,
        )

        # Prepare formula
        feature_names = [f'x{i+1}' for i in range(n_features)]
        formula = f"y ~ {' + '.join(feature_names)}"

        # Measure memory before
        mem_before = self.measure_memory_usage()

        # Prepare design matrix (include in timing)
        start_prep = time.time()
        y, X = dmatrices(formula, df, return_type="dataframe")
        prep_time = time.time() - start_prep

        # Measure DSL estimation
        print(f"  Running DSL estimation...")
        exec_time, result = self.measure_execution_time(
            dsl,
            X=X.values,
            y=y.values,
            labeled_ind=df["labeled"].values,
            sample_prob=df["sample_prob"].values,
            model="logit",
            method="logistic",
        )

        # Measure memory after
        mem_after = self.measure_memory_usage()

        benchmark_result = {
            'n_total': n_total,
            'n_labeled': n_labeled,
            'n_features': n_features,
            'prep_time_s': prep_time,
            'exec_time_s': exec_time,
            'total_time_s': prep_time + exec_time,
            'memory_delta_mb': mem_after['rss_mb'] - mem_before['rss_mb'],
            'convergence': result.success,
            'n_iterations': result.niter if hasattr(result, 'niter') else None,
            'objective_value': result.objective if hasattr(result, 'objective') else None,
        }

        self.results.append(benchmark_result)
        return benchmark_result

    def benchmark_scaling_by_sample_size(self, n_features: int = 5) -> List[Dict]:
        """Benchmark scaling with different sample sizes.

        Args:
            n_features: Number of features to use

        Returns:
            List of benchmark results
        """
        print(f"\n=== Benchmarking Sample Size Scaling (n_features={n_features}) ===")

        sample_sizes = [100, 500, 1000, 5000, 10000]
        results = []

        for n_total in sample_sizes:
            n_labeled = n_total // 2

            result = self.benchmark_dsl_estimation(
                n_total=n_total,
                n_labeled=n_labeled,
                n_features=n_features,
            )
            results.append(result)

            print(f"  n_total={n_total}: {result['total_time_s']:.3f}s")

        return results

    def benchmark_scaling_by_features(self, n_total: int = 1000) -> List[Dict]:
        """Benchmark scaling with different numbers of features.

        Args:
            n_total: Total observations to use

        Returns:
            List of benchmark results
        """
        print(f"\n=== Benchmarking Feature Scaling (n_total={n_total}) ===")

        feature_counts = [2, 5, 10, 20, 50]
        results = []

        for n_features in feature_counts:
            n_labeled = n_total // 2

            result = self.benchmark_dsl_estimation(
                n_total=n_total,
                n_labeled=n_labeled,
                n_features=n_features,
            )
            results.append(result)

            print(f"  n_features={n_features}: {result['total_time_s']:.3f}s")

        return results

    def benchmark_scaling_by_label_fraction(
        self, n_total: int = 1000, n_features: int = 5
    ) -> List[Dict]:
        """Benchmark scaling with different labeled fractions.

        Args:
            n_total: Total observations
            n_features: Number of features

        Returns:
            List of benchmark results
        """
        print(f"\n=== Benchmarking Label Fraction Scaling (n_total={n_total}, n_features={n_features}) ===")

        label_fractions = [0.1, 0.25, 0.5, 0.75, 0.9]
        results = []

        for frac in label_fractions:
            n_labeled = int(n_total * frac)

            result = self.benchmark_dsl_estimation(
                n_total=n_total,
                n_labeled=n_labeled,
                n_features=n_features,
            )
            results.append(result)

            print(f"  label_frac={frac:.2f}: {result['total_time_s']:.3f}s")

        return results

    def generate_report(self, output_path: str = "benchmark_report.json"):
        """Generate benchmark report.

        Args:
            output_path: Path to save report JSON
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'total_memory_gb': psutil.virtual_memory().total / 1024**3,
                'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            },
            'results': self.results,
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\nBenchmark report saved to: {output_path}")

        # Print summary
        print("\n=== Summary ===")
        print(f"Total benchmarks run: {len(self.results)}")

        if self.results:
            times = [r['total_time_s'] for r in self.results]
            print(f"Execution time range: {min(times):.3f}s - {max(times):.3f}s")
            print(f"Average execution time: {np.mean(times):.3f}s")

    def print_summary_table(self):
        """Print summary table of results."""
        if not self.results:
            print("No results to display")
            return

        df = pd.DataFrame(self.results)

        print("\n=== Benchmark Results ===")
        print(df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description='Run DSL performance benchmarks',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--benchmark-type',
        type=str,
        choices=['sample', 'features', 'labels', 'all'],
        default='all',
        help='Type of benchmark to run'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='benchmark_report.json',
        help='Output path for benchmark report'
    )

    args = parser.parse_args()

    benchmark = PerformanceBenchmark()

    print("=" * 60)
    print("DSL Performance Benchmarking Suite")
    print("=" * 60)

    if args.benchmark_type in ['sample', 'all']:
        benchmark.benchmark_scaling_by_sample_size()

    if args.benchmark_type in ['features', 'all']:
        benchmark.benchmark_scaling_by_features()

    if args.benchmark_type in ['labels', 'all']:
        benchmark.benchmark_scaling_by_label_fraction()

    # Generate report
    benchmark.print_summary_table()
    benchmark.generate_report(args.output)

    print("\n" + "=" * 60)
    print("Benchmarking complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
