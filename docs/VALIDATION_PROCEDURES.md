```yaml
# Validation Procedures for Contributors

Comprehensive guide for contributors on Python-R validation, testing procedures, and continuous monitoring.

## Overview

This document outlines validation procedures for ensuring Python DSL implementation maintains statistical equivalence with the R implementation. All contributors should follow these procedures when making changes to the DSL codebase.

## Key Concepts

### Statistical Equivalence vs. Exact Matching

**Important:** Python and R implementations will **not** produce identical results due to:
- Different random number generators (RNG incompatibility)
- Floating-point arithmetic variations
- Optimization algorithm differences (BFGS implementations)
- Different linear algebra libraries

**Goal:** Statistical equivalence within tolerance thresholds, not exact matching.

### Tolerance Framework

**Absolute Tolerance:** Used for small values where relative differences are unstable
- Coefficients: ±0.01 (for |coef| ≤ 0.1)
- Standard Errors: ±0.01 (for |SE| ≤ 0.1)
- P-values: ±0.05

**Relative Tolerance:** Used for larger values for scale-invariant comparison
- Coefficients: ±5% (for |coef| > 0.1)
- Standard Errors: ±10% (for |SE| > 0.1)
- P-values: ±20%

## Pre-Commit Validation

### Step 1: Run Local Comparison Tests

```bash
# Run all comparison tests
pytest tests/test_python_r_comparison.py -v

# Expected output: All tests passing or within documented tolerances
```

**If tests fail:**
1. Review failure details in test output
2. Check if failures are due to legitimate code changes
3. If changes affect coefficients, document in commit message
4. Consider updating tolerances if mathematically justified

### Step 2: Run Synthetic Validation

```bash
# Test on synthetic data with known properties
pytest tests/test_synthetic_validation.py -v
```

Synthetic tests should **always** pass as they use controlled data without RNG issues.

### Step 3: Run Performance Benchmarks (for optimization changes)

```bash
# Run performance benchmarks
python tests/benchmark_performance.py --benchmark-type all --output benchmark.json

# Compare with previous benchmarks
# Performance regressions >20% require justification
```

### Step 4: Generate Comparison Report

```bash
# Generate visual comparison report
python scripts/run_comparison.py --report-dir reports --save-metrics metrics.json

# Review HTML report in reports/ directory
# Check pass rates and failure patterns
```

## Pull Request Validation

### Automated CI/CD Checks

When you open a PR, GitHub Actions automatically runs:
1. Python-R comparison tests (Python 3.9, 3.10, 3.11)
2. Metrics extraction and dashboard generation
3. Automated status summary

**Review checklist:**
- ✅ All matrix builds passing
- ✅ Comparison dashboard shows acceptable pass rates
- ✅ No unexpected new failures
- ✅ Performance metrics within acceptable range

### Manual PR Review

1. **Check GitHub Actions summary**
   - View "Python-R Comparison Summary" in PR checks
   - Review pass rates for each Python version
   - Download comparison dashboard artifact if needed

2. **Review comparison dashboard**
   - Go to Actions → Your workflow run → Artifacts
   - Download `comparison-dashboard`
   - Open `index.html` in browser
   - Verify pass rates ≥95% or justify lower rates

3. **Document any intentional changes**
   - If changes affect comparison results, explain why
   - Reference mathematical justification if modifying algorithms
   - Update tolerance configuration if necessary

## Making Changes to DSL Core

### For Algorithm Changes

1. **Document mathematical justification**
   - Cite papers, equations, or theoretical basis
   - Explain why change improves correctness or efficiency

2. **Run extended validation**
   ```bash
   # Test on multiple datasets
   pytest tests/ -v

   # Run synthetic validation with coefficient recovery tests
   pytest tests/test_synthetic_validation.py::TestSyntheticValidation::test_coefficient_recovery -v

   # Benchmark performance impact
   python tests/benchmark_performance.py --benchmark-type all
   ```

3. **Update test expectations if needed**
   - If changes affect coefficients systematically, update R reference
   - Update tolerance configuration in `tests/comparison/comparator.py`
   - Document rationale in commit message and PR description

### For Optimization Changes

1. **Benchmark before and after**
   ```bash
   # Before changes
   git stash
   python tests/benchmark_performance.py --output before.json

   # After changes
   git stash pop
   python tests/benchmark_performance.py --output after.json

   # Compare results
   python scripts/compare_benchmarks.py before.json after.json
   ```

2. **Document performance impact**
   - Include benchmark comparison in PR
   - Justify any performance regressions
   - Highlight performance improvements

3. **Verify correctness maintained**
   ```bash
   # Ensure all comparison tests still pass
   pytest tests/test_python_r_comparison.py -v

   # Verify synthetic validation
   pytest tests/test_synthetic_validation.py -v
   ```

## Adding New Test Datasets

### Step 1: Prepare Dataset

```python
# For real datasets
from tests.data.compare_panchen import load_panchen_data

# Create new loader function in tests/data/
def load_new_dataset():
    # Load data
    data = pd.read_parquet("path/to/data.parquet")
    return data

# For synthetic datasets
from tests.data.synthetic_dataset import generate_synthetic_logistic_data

df = generate_synthetic_logistic_data(
    n_total=1000,
    n_labeled=500,
    n_features=5,
    random_seed=42
)
```

### Step 2: Generate R Reference

```R
# Run in R
library(DSL)

# Generate R results
result <- dsl(...)
summary(result)

# Save output to text file
cat(capture.output(summary(result)), file="r_reference_output.txt", sep="\n")
```

### Step 3: Create Parser

```python
# In tests/comparison/r_reference.py

def parse_r_output_new_dataset(file_path: str) -> ReferenceResult:
    """Parse R output for new dataset."""
    with open(file_path, 'r') as f:
        content = f.read()

    # Parse coefficients, SEs, p-values
    # ... (follow PanChen example)

    return ReferenceResult(
        name="new_dataset",
        coefficients=coefficients,
        standard_errors=standard_errors,
        p_values=p_values,
    )
```

### Step 4: Create Test Class

```python
# In tests/test_python_r_comparison.py or new file

class TestNewDatasetComparison:
    @pytest.fixture
    def new_data(self):
        return load_new_dataset()

    @pytest.fixture
    def python_result(self, new_data):
        # Run Python DSL
        ...
        return result

    @pytest.fixture
    def r_reference(self):
        return parse_r_output_new_dataset("r_reference_output.txt")

    def test_coefficients_match(self, python_result, r_reference):
        # Comparison tests
        ...
```

### Step 5: Update Documentation

- Add dataset description to testing documentation
- Document any dataset-specific tolerance considerations
- Update README with new test dataset information

## Monitoring Metrics Over Time

### Historical Metrics Tracking

Metrics are automatically saved to `.metrics-history/` on main branch:
```bash
# View historical metrics
ls -la .metrics-history/

# Load and analyze
python -c "
import json
import glob

files = sorted(glob.glob('.metrics-history/metrics_*.json'))
for f in files[-5:]:  # Last 5 runs
    with open(f) as fp:
        m = json.load(fp)
        print(f'{f}: {m[\"overall_pass_rate\"]:.1f}%')
"
```

### Detecting Regressions

1. **Pass Rate Decline**
   - Investigate if pass rate drops >5% from recent average
   - Check which metrics are newly failing
   - Identify commit range that introduced regression

2. **Performance Regression**
   - Compare benchmark results with historical data
   - Investigate if execution time increases >20%
   - Profile code to identify bottlenecks

3. **New Failure Patterns**
   - If specific variables consistently fail, investigate
   - Check if changes affected those variable calculations
   - Review mathematical correctness

## Troubleshooting Common Issues

### Test Failures Due to RNG

**Problem:** Coefficients don't match R results

**Solution:**
- This is **expected** due to RNG incompatibility
- Verify statistical correctness using synthetic data
- Check that optimization converges (objective ≈ 0)
- Ensure tolerances are appropriate for the problem

**Do NOT:**
- Try to match R's random number generator
- Reduce tolerances to force exact matching
- Ignore systematic failures across all coefficients

### Tolerance Configuration Issues

**Problem:** Legitimate implementation differences exceed tolerances

**Solution:**
1. Verify the difference is mathematically justified
2. Consider if tolerance is too strict for the problem
3. Update tolerance configuration with documented rationale:
   ```python
   # In ComparisonConfig
   coef_rel_tol=0.10  # Increased to 10% due to [reason]
   ```

### Performance Regression

**Problem:** Benchmarks show significant slowdown

**Steps:**
1. Profile code to identify bottleneck:
   ```bash
   python -m cProfile -o profile.stats tests/benchmark_performance.py
   python -m pstats profile.stats
   ```

2. Compare with baseline performance
3. Investigate optimization algorithm changes
4. Check for inefficient data structures or loops

### CI/CD Failures

**Problem:** Tests pass locally but fail in CI

**Common causes:**
- Different Python versions
- Missing dependencies in requirements.txt
- Environment-specific numerical differences
- Random seed not fixed

**Solution:**
1. Check CI logs for specific error messages
2. Test locally with same Python version as CI
3. Ensure all dependencies pinned in requirements.txt
4. Verify random seeds are set consistently

## Best Practices

### Code Changes

1. **Make atomic commits** - One logical change per commit
2. **Run tests before committing** - Catch issues early
3. **Document rationale** - Explain why changes were made
4. **Update tests** - Add tests for new functionality

### Testing

1. **Use synthetic data** - When possible for controlled validation
2. **Test edge cases** - Small samples, many features, etc.
3. **Verify convergence** - Check optimization succeeded
4. **Document failures** - If tests fail, explain why

### Documentation

1. **Update tolerance rationale** - When changing tolerances
2. **Document algorithm changes** - With mathematical justification
3. **Maintain test documentation** - Keep testing docs current
4. **Add examples** - For new features or datasets

### Performance

1. **Benchmark major changes** - Especially to core algorithms
2. **Profile before optimizing** - Identify actual bottlenecks
3. **Document tradeoffs** - Speed vs. accuracy, memory vs. speed
4. **Test scalability** - Verify performance on large datasets

## Approval Criteria

Pull requests must meet these criteria:

### Required (Blocking)
- ✅ All CI/CD checks passing
- ✅ Overall pass rate ≥80%
- ✅ No synthetic validation failures
- ✅ Performance regression <20% (or justified)
- ✅ Code review approved by maintainer

### Recommended (Non-Blocking)
- ✅ Overall pass rate ≥95%
- ✅ No new test failures introduced
- ✅ Performance improvements documented
- ✅ New tests added for new features
- ✅ Documentation updated

### Justification Required
- ⚠️ Pass rate 80-95%: Document why acceptable
- ⚠️ Performance regression >20%: Justify tradeoff
- ⚠️ New failures: Explain mathematical basis
- ⚠️ Tolerance changes: Provide detailed rationale

## Getting Help

### Documentation
- [Testing Framework Documentation](python_r_comparison_testing.md)
- [README Testing Section](../README.md#testing-and-validation)
- GitHub Actions workflow: `.github/workflows/python-r-comparison.yml`

### Reporting Issues
- Check existing issues on GitHub
- Provide reproducible example
- Include comparison report and metrics
- Tag with `validation` label

### Contact
- Open GitHub issue for validation questions
- Reference specific test failures
- Include comparison dashboard screenshots
- Provide benchmark results for performance issues

---

**Remember:** The goal is statistical equivalence, not exact matching. Focus on mathematical correctness and appropriate tolerance configuration.
```
