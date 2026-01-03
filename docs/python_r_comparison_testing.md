# Python-R Comparison Testing Framework

Automated testing infrastructure to validate Python DSL implementation against R reference results.

## Overview

This framework provides comprehensive automated testing to ensure the Python implementation of DSL produces results consistent with the original R implementation. It includes:

- **Automated coefficient comparison** with configurable tolerances
- **Standard error validation** across implementations
- **P-value verification** with statistical rigor
- **Visual comparison reports** with plots and charts
- **CI/CD integration** for continuous validation
- **Multi-dataset test fixtures** for comprehensive coverage

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install DSL package in development mode
pip install -e .
```

### Running Tests

```bash
# Run all comparison tests
pytest tests/test_python_r_comparison.py -v

# Run specific test class
pytest tests/test_python_r_comparison.py::TestPanChenComparison -v

# Run with HTML report
pytest tests/test_python_r_comparison.py --html=report.html --self-contained-html
```

### Generating Comparison Reports

```python
from tests.comparison.r_reference import load_panchen_r_reference
from tests.comparison.comparator import compare_implementations, ComparisonConfig
from tests.comparison.report_generator import ComparisonReportGenerator

# Load R reference results
r_ref = load_panchen_r_reference()

# Run Python implementation
# ... (run DSL estimation) ...

# Compare implementations
config = ComparisonConfig(
    coef_abs_tol=0.01,
    coef_rel_tol=0.05,
    se_abs_tol=0.01,
    se_rel_tol=0.10
)

comparison = compare_implementations(
    python_coefs, python_ses, python_pvalues,
    r_ref.coefficients, r_ref.standard_errors, r_ref.p_values,
    config=config
)

# Generate detailed report with visualizations
report_gen = ComparisonReportGenerator()
report_path = report_gen.generate_full_report(comparison, "panchen_comparison")
print(f"Report generated: {report_path}")
```

## Framework Components

### 1. R Reference Loader (`tests/comparison/r_reference.py`)

Parses R output files and loads reference results for comparison.

**Key Functions:**
- `parse_r_output_panchen(file_path)` - Parse R output text file
- `load_panchen_r_reference()` - Load PanChen R reference results

**Example:**
```python
from tests.comparison.r_reference import load_panchen_r_reference

r_ref = load_panchen_r_reference()
print(r_ref.coefficients)  # Dict of coefficients
print(r_ref.standard_errors)  # Dict of standard errors
print(r_ref.p_values)  # Dict of p-values
```

### 2. Comparison Framework (`tests/comparison/comparator.py`)

Automated comparison with configurable tolerances and detailed reporting.

**Key Classes:**
- `ComparisonConfig` - Configure comparison tolerances
- `ComparisonMetric` - Single metric comparison result
- `ComparisonResult` - Complete comparison with summary

**Tolerance Configuration:**

```python
from tests.comparison.comparator import ComparisonConfig

# Default configuration
config = ComparisonConfig()

# Custom configuration
config = ComparisonConfig(
    coef_abs_tol=0.01,      # Absolute tolerance for small coefficients
    coef_rel_tol=0.05,      # 5% relative tolerance for large coefficients
    se_abs_tol=0.01,        # Absolute tolerance for small SEs
    se_rel_tol=0.10,        # 10% relative tolerance for large SEs
    pvalue_abs_tol=0.05,    # Absolute tolerance for p-values
    coef_rel_threshold=0.1, # Use relative tolerance when |value| > 0.1
)
```

**Comparison Logic:**
- For small values (|value| ≤ threshold): Use absolute tolerance
- For large values (|value| > threshold): Use relative tolerance
- Both absolute and relative differences are computed and reported

### 3. Report Generator (`tests/comparison/report_generator.py`)

Generate comprehensive HTML reports with visualizations.

**Key Features:**
- Summary statistics and pass/fail rates
- Scatter plots comparing Python vs R values
- Bar charts showing relative differences by variable
- Detailed tables of all comparisons
- Color-coded pass/fail indicators

**Example:**
```python
from tests.comparison.report_generator import ComparisonReportGenerator, ReportConfig

# Configure report generation
config = ReportConfig(
    output_dir="reports",
    include_plots=True,
    plot_format="png",
    plot_dpi=300,
    show_all_comparisons=False  # Only show failures in tables
)

generator = ComparisonReportGenerator(config)
report_path = generator.generate_full_report(comparison_result, "my_comparison")
```

### 4. Automated Test Suite (`tests/test_python_r_comparison.py`)

Pytest-based automated tests with fixtures and assertions.

**Test Classes:**
- `TestPanChenComparison` - PanChen dataset comparison tests
- `TestComparisonFramework` - Framework validation tests

**Key Tests:**
- `test_coefficients_match` - All coefficients within tolerance
- `test_standard_errors_match` - All SEs within tolerance
- `test_pvalues_match` - All p-values within tolerance
- `test_overall_comparison` - Complete comparison validation
- `test_specific_coefficients` - Critical variable validation
- `test_model_properties` - Model metadata validation

## Test Fixtures

### PanChen Dataset

Located in `PanChen_test/`:
- `PanChen.parquet` - Dataset (parquet format)
- `target_r_output_panchen.txt` - R reference output
- `compare_panchen.py` - Original comparison script

**Variables:**
- Outcome: `SendOrNot`
- Predictors: `countyWrong`, `prefecWrong`, `connect2b`, `prevalence`, `regionj`, `groupIssue`
- Prediction: `pred_countyWrong`

### Adding New Test Fixtures

1. **Prepare R Reference Output:**
   ```R
   # Run DSL in R and save output
   result <- dsl(...)
   summary(result)
   # Copy output to new_dataset_r_output.txt
   ```

2. **Create Parser Function:**
   ```python
   # In tests/comparison/r_reference.py
   def load_new_dataset_r_reference() -> ReferenceResult:
       file_path = "path/to/new_dataset_r_output.txt"
       return parse_r_output(file_path)
   ```

3. **Create Test Class:**
   ```python
   # In tests/test_python_r_comparison.py
   class TestNewDatasetComparison:
       @pytest.fixture
       def new_data(self):
           # Load and prepare new dataset
           pass

       def test_comparison(self, new_data):
           # Run comparison tests
           pass
   ```

## Continuous Integration

### GitHub Actions Workflow

Located in `.github/workflows/python-r-comparison.yml`

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`
- Nightly scheduled runs (00:00 UTC)

**Matrix Testing:**
- Python versions: 3.9, 3.10, 3.11
- R version: 4.3

**Artifacts:**
- HTML test reports (pytest-html)
- Detailed comparison reports (with visualizations)

### Local CI Simulation

```bash
# Run tests with same configuration as CI
pytest tests/test_python_r_comparison.py -v \
    --html=comparison_report.html \
    --self-contained-html

# Check report
open comparison_report.html
```

## Tolerance Guidelines

### Coefficient Tolerances

**Absolute Tolerance (small values):**
- Recommended: `0.01` (±0.01)
- Use when: |coefficient| ≤ 0.1
- Rationale: Small coefficients may have larger relative differences due to numerical precision

**Relative Tolerance (large values):**
- Recommended: `0.05` (±5%)
- Use when: |coefficient| > 0.1
- Rationale: Large coefficients should match closely in relative terms

### Standard Error Tolerances

**Absolute Tolerance:**
- Recommended: `0.01` (±0.01)
- Use when: |SE| ≤ 0.1

**Relative Tolerance:**
- Recommended: `0.10` (±10%)
- Use when: |SE| > 0.1
- Rationale: SEs involve more complex calculations and may have higher variance

### P-value Tolerances

**Absolute Tolerance:**
- Recommended: `0.05` (±0.05)
- Rationale: P-values near significance thresholds need wider tolerance
- Note: Very small p-values (<0.001) may differ due to numerical precision

### Adjusting Tolerances

```python
# Strict comparison
strict_config = ComparisonConfig(
    coef_abs_tol=0.001,
    coef_rel_tol=0.01,
    se_abs_tol=0.001,
    se_rel_tol=0.05,
)

# Lenient comparison
lenient_config = ComparisonConfig(
    coef_abs_tol=0.05,
    coef_rel_tol=0.10,
    se_abs_tol=0.05,
    se_rel_tol=0.20,
)
```

## Troubleshooting

### Test Failures

**Symptom:** Coefficient comparison fails
- **Check:** Are seeds synchronized between Python and R?
- **Solution:** Ensure `np.random.seed()` matches R's `set.seed()`

**Symptom:** Large relative differences for small values
- **Check:** Are absolute tolerances appropriate?
- **Solution:** Increase `coef_abs_tol` or adjust `coef_rel_threshold`

**Symptom:** SE comparison fails systematically
- **Check:** Is variance-covariance calculation correct?
- **Solution:** Compare `result.vcov` between implementations

### Report Generation Issues

**Symptom:** Plots not showing in report
- **Check:** Is matplotlib using correct backend?
- **Solution:** Set `matplotlib.use('Agg')` for non-interactive backend

**Symptom:** Report directory not created
- **Check:** Do you have write permissions?
- **Solution:** Set `output_dir` to writable location

## Best Practices

### Writing Comparison Tests

1. **Use fixtures for data preparation** - Reuse data loading across tests
2. **Separate concerns** - Test coefficients, SEs, and p-values separately
3. **Document tolerance choices** - Explain why specific tolerances were chosen
4. **Check critical variables** - Always test key coefficients explicitly
5. **Validate metadata** - Verify n_obs, model type, etc.

### Maintaining Test Suite

1. **Update R references** - When R package updates, regenerate references
2. **Version control** - Commit test fixtures and reference outputs
3. **Document changes** - Note any tolerance adjustments in commit messages
4. **Monitor CI** - Check nightly runs for regressions
5. **Review failures** - Investigate any systematic failures

### Performance Optimization

1. **Cache fixtures** - Use `scope="class"` for expensive fixtures
2. **Parallel execution** - Run tests in parallel with `pytest -n auto`
3. **Skip slow tests** - Mark slow tests with `@pytest.mark.slow`
4. **Profile tests** - Use `pytest --durations=10` to find slow tests

## API Reference

### ComparisonConfig

```python
@dataclass
class ComparisonConfig:
    coef_abs_tol: float = 0.01
    se_abs_tol: float = 0.01
    pvalue_abs_tol: float = 0.05
    coef_rel_tol: float = 0.05
    se_rel_tol: float = 0.10
    pvalue_rel_tol: float = 0.20
    coef_rel_threshold: float = 0.1
    se_rel_threshold: float = 0.1
    pvalue_rel_threshold: float = 0.01
```

### ComparisonResult

```python
@dataclass
class ComparisonResult:
    name: str
    config: ComparisonConfig
    coefficient_comparisons: List[ComparisonMetric]
    se_comparisons: List[ComparisonMetric]
    pvalue_comparisons: List[ComparisonMetric]

    @property
    def all_passed(self) -> bool: ...
    @property
    def n_comparisons(self) -> int: ...
    @property
    def n_passed(self) -> int: ...
    @property
    def n_failed(self) -> int: ...

    def get_failures(self) -> List[ComparisonMetric]: ...
    def summary(self) -> str: ...
```

### Key Functions

```python
# Comparison
def compare_implementations(...) -> ComparisonResult: ...
def assert_implementations_match(comparison_result, raise_on_failure=True) -> bool: ...

# Reference loading
def load_panchen_r_reference() -> ReferenceResult: ...
def parse_r_output_panchen(file_path: str) -> ReferenceResult: ...

# Report generation
def generate_full_report(result: ComparisonResult, report_name: str) -> str: ...
```

## Examples

### Example 1: Basic Comparison

```python
import numpy as np
from scipy import stats
from patsy import dmatrices
from dsl import dsl
from tests.data.compare_panchen import load_panchen_data, prepare_data_for_dsl
from tests.comparison.r_reference import load_panchen_r_reference
from tests.comparison.comparator import compare_implementations

# Load data
data = load_panchen_data()
df = prepare_data_for_dsl(data)

# Run Python DSL
formula = "SendOrNot ~ countyWrong + prefecWrong + connect2b + prevalence + regionj + groupIssue"
y, X = dmatrices(formula, df, return_type="dataframe")
result = dsl(X=X.values, y=y.values, labeled_ind=df["labeled"].values,
             sample_prob=df["sample_prob"].values, model="logit", method="logistic")

# Extract Python results
terms = ["(Intercept)"] + formula.split("~")[1].split("+")
terms = [t.strip() for t in terms]
py_coefs = {term: result.coefficients[i] for i, term in enumerate(terms)}
py_ses = {term: result.standard_errors[i] for i, term in enumerate(terms)}
t_stats = result.coefficients / result.standard_errors
p_vals = 2 * (1 - stats.norm.cdf(np.abs(t_stats)))
py_pvals = {term: p_vals[i] for i, term in enumerate(terms)}

# Load R reference
r_ref = load_panchen_r_reference()

# Compare
comparison = compare_implementations(
    py_coefs, py_ses, py_pvals,
    r_ref.coefficients, r_ref.standard_errors, r_ref.p_values
)

print(comparison.summary())
```

### Example 2: Custom Tolerances

```python
from tests.comparison.comparator import ComparisonConfig, compare_implementations

# Strict configuration for production validation
config = ComparisonConfig(
    coef_abs_tol=0.001,
    coef_rel_tol=0.01,
    se_abs_tol=0.001,
    se_rel_tol=0.05,
    pvalue_abs_tol=0.01,
)

comparison = compare_implementations(
    py_coefs, py_ses, py_pvals,
    r_ref.coefficients, r_ref.standard_errors, r_ref.p_values,
    config=config,
    name="Strict Production Validation"
)

if not comparison.all_passed:
    print("VALIDATION FAILED!")
    for failure in comparison.get_failures():
        print(failure.message)
```

### Example 3: Report Generation

```python
from tests.comparison.report_generator import ComparisonReportGenerator, ReportConfig

# Configure report with all comparisons visible
config = ReportConfig(
    output_dir="validation_reports",
    include_plots=True,
    show_all_comparisons=True,  # Show all comparisons, not just failures
    plot_dpi=600  # High-resolution plots
)

generator = ComparisonReportGenerator(config)
report_path = generator.generate_full_report(comparison, "production_validation")

print(f"Validation report: {report_path}")
# Opens: validation_reports/production_validation_YYYYMMDD_HHMMSS/report.html
```

## Contributing

### Adding New Comparison Tests

1. Create test fixture in appropriate directory
2. Add R reference parser in `r_reference.py`
3. Write test class in `test_python_r_comparison.py`
4. Update this documentation with new test details

### Improving Framework

1. Fork repository
2. Create feature branch
3. Add tests for new features
4. Submit pull request with clear description

## References

- [DSL R Package Documentation](https://cran.r-project.org/package=DSL)
- [Pytest Documentation](https://docs.pytest.org/)
- [NumPy Testing Guidelines](https://numpy.org/doc/stable/reference/routines.testing.html)
- [GitHub Actions for Python](https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python)

## License

This testing framework is part of the DSL Python implementation and follows the same license.
