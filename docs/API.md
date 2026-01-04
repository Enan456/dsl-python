# DSL Python API Reference

Complete API documentation for the Python implementation of the DSL (Design-based Supervised Learning) framework.

## Table of Contents

- [Core Functions](#core-functions)
  - [dsl()](#dsl)
  - [dsl_general()](#dsl_general)
- [Result Classes](#result-classes)
  - [DSLResult](#dslresult)
  - [PowerDSLResult](#powerdslresult)
- [Power Analysis](#power-analysis)
  - [power_dsl()](#power_dsl)
- [Utility Functions](#utility-functions)
  - [summary()](#summary)
  - [summary_power()](#summary_power)
  - [plot_power()](#plot_power)
- [Usage Examples](#usage-examples)

---

## Core Functions

### dsl()

Simple wrapper function for DSL estimation.

**⚠️ Important Limitation**: The current `dsl()` wrapper does not support using separate prediction data (X_pred). For full prediction support, use [`dsl_general()`](#dsl_general) directly.

```python
def dsl(
    X: np.ndarray,
    y: np.ndarray,
    labeled_ind: np.ndarray,
    sample_prob: np.ndarray,
    model: str = "logit",
    method: str = "linear",
) -> DSLResult
```

#### Parameters

- **X** : `np.ndarray`
  - Design matrix (features). Shape: (n_samples, n_features)
  - Should include intercept column if desired (typically first column of 1s)

- **y** : `np.ndarray`
  - Response variable (outcome). Shape: (n_samples,) or (n_samples, 1)
  - For logistic regression: binary outcomes (0 or 1)
  - For linear regression: continuous outcomes

- **labeled_ind** : `np.ndarray`
  - Binary indicator for labeled observations. Shape: (n_samples,)
  - 1 = observation has true outcome label
  - 0 = observation is unlabeled (outcome predicted)

- **sample_prob** : `np.ndarray`
  - Sampling probability for each observation. Shape: (n_samples,)
  - Typically: `n_labeled / n_total` for simple random sampling
  - Used for doubly robust estimation weights

- **model** : `str`, optional (default: `"logit"`)
  - Model type for estimation
  - Options: `"logit"`, `"lm"`, `"felm"`

- **method** : `str`, optional (default: `"linear"`)
  - Method for supervised learning
  - Options: `"linear"`, `"logistic"`, `"fixed_effects"`
  - **Note**: Currently not used for prediction generation

#### Returns

- **DSLResult** : Estimation results object containing:
  - `coefficients` : Estimated parameters
  - `standard_errors` : Standard errors
  - `vcov` : Variance-covariance matrix
  - `objective` : Final objective function value (should be ≈ 0)
  - `success` : Whether optimization converged
  - `message` : Convergence message
  - `niter` : Number of optimization iterations
  - `model` : Model type used
  - `labeled_size` : Number of labeled observations
  - `total_size` : Total number of observations

#### Example

```python
import numpy as np
from patsy import dmatrices
from dsl import dsl

# Prepare data with patsy formula
y, X = dmatrices("outcome ~ feature1 + feature2", data, return_type="dataframe")

# Run DSL estimation
result = dsl(
    X=X.values,
    y=y.values.flatten(),
    labeled_ind=labeled_indicator,
    sample_prob=sampling_probabilities,
    model="logit",
    method="logistic"
)

print(f"Coefficients: {result.coefficients}")
print(f"Standard Errors: {result.standard_errors}")
print(f"Converged: {result.success}")
```

---

### dsl_general()

Core DSL estimation function with full support for doubly robust estimation using predictions.

**Use this function when:**
- You have separate predicted features (X_pred) different from original features (X_orig)
- You need full control over prediction data for doubly robust estimation
- You want to use pre-generated predictions

```python
def dsl_general(
    Y_orig: np.ndarray,
    X_orig: np.ndarray,
    Y_pred: np.ndarray,
    X_pred: np.ndarray,
    labeled_ind: np.ndarray,
    sample_prob_use: np.ndarray,
    model: str = "lm",
    fe_Y: Optional[np.ndarray] = None,
    fe_X: Optional[np.ndarray] = None,
    moment_fn: Optional[callable] = None,
    jac_fn: Optional[callable] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]
```

#### Parameters

- **Y_orig** : `np.ndarray`
  - Original outcome variable. Shape: (n_samples,)
  - True outcomes for labeled observations
  - Can be any value for unlabeled observations (not used)

- **X_orig** : `np.ndarray`
  - Original feature matrix. Shape: (n_samples, n_features)
  - True features for labeled observations
  - For unlabeled observations, typically filled with:
    - Predicted values (recommended)
    - Mean values
    - Zero (not recommended)

- **Y_pred** : `np.ndarray`
  - Predicted outcome variable. Shape: (n_samples,)
  - Currently typically same as Y_orig
  - Future: Could use model predictions for Y

- **X_pred** : `np.ndarray`
  - Predicted feature matrix. Shape: (n_samples, n_features)
  - **Key for doubly robust estimation**: Use predicted features here
  - Example: Replace missing variable with its prediction
  - This enables the doubly robust moment conditions to work

- **labeled_ind** : `np.ndarray`
  - Binary indicator for labeled observations. Shape: (n_samples,)
  - 1 = labeled, 0 = unlabeled

- **sample_prob_use** : `np.ndarray`
  - Sampling probability. Shape: (n_samples,)
  - Formula: `n_labeled / n_total`
  - Used in doubly robust weights: `labeled_ind / sample_prob_use`

- **model** : `str`, optional (default: `"lm"`)
  - Model type: `"lm"` (linear), `"logit"` (logistic), `"felm"` (fixed effects)

- **fe_Y** : `Optional[np.ndarray]`, optional (default: `None`)
  - Fixed effects for outcome (for FELM models)

- **fe_X** : `Optional[np.ndarray]`, optional (default: `None`)
  - Fixed effects for features (for FELM models)

- **moment_fn** : `Optional[callable]`, optional (default: `None`)
  - Custom moment function (advanced usage)
  - If None, uses built-in moment functions

- **jac_fn** : `Optional[callable]`, optional (default: `None`)
  - Custom Jacobian function (advanced usage)
  - If None, uses built-in Jacobian functions

#### Returns

- **Tuple[np.ndarray, Dict[str, Any]]** :
  1. **par** (`np.ndarray`) : Estimated parameters
  2. **info** (`Dict`) : Dictionary containing:
     - `"standard_errors"` : Standard errors
     - `"vcov"` : Variance-covariance matrix
     - `"objective"` : Final objective value
     - `"convergence"` : Boolean, whether converged
     - `"message"` : Convergence message
     - `"iterations"` : Number of iterations

#### Example with Predictions

```python
from dsl.helpers.dsl_general import dsl_general
from patsy import dmatrices

# Original data with some missing values
formula = "SendOrNot ~ countyWrong + prefecWrong + connect2b"
y, X = dmatrices(formula, df, return_type="dataframe")

# Create prediction data by filling missing countyWrong with predictions
df_pred = df.copy()
df_pred["countyWrong"] = df_pred["countyWrong"].fillna(df_pred["pred_countyWrong"])
_, X_pred = dmatrices(formula, df_pred, return_type="dataframe")

# Run DSL with predictions
par, info = dsl_general(
    Y_orig=y.values.flatten(),
    X_orig=X.values,
    Y_pred=y.values.flatten(),
    X_pred=X_pred.values,  # Uses predictions for missing values
    labeled_ind=df["labeled"].values,
    sample_prob_use=df["sample_prob"].values,
    model="logit"
)

print(f"Coefficients: {par}")
print(f"Standard Errors: {info['standard_errors']}")
print(f"Objective: {info['objective']}")  # Should be ≈ 0
print(f"Converged: {info['convergence']}")
```

---

## Result Classes

### DSLResult

Data class containing DSL estimation results.

```python
@dataclass
class DSLResult:
    coefficients: np.ndarray          # Estimated parameters
    standard_errors: np.ndarray       # Standard errors
    vcov: np.ndarray                  # Variance-covariance matrix
    objective: float                  # Final objective function value
    success: bool                     # Convergence status
    message: str                      # Convergence message
    niter: int                        # Number of iterations
    model: str                        # Model type ("logit", "lm", etc.)
    labeled_size: int                 # Number of labeled observations
    total_size: int                   # Total observations
    predicted_values: Optional[np.ndarray] = None  # Predicted values
    residuals: Optional[np.ndarray] = None         # Residuals
```

#### Accessing Results

```python
result = dsl(X, y, labeled_ind, sample_prob, model="logit")

# Access attributes
coefs = result.coefficients
ses = result.standard_errors
vcov = result.vcov

# Check convergence
if result.success:
    print(f"Converged in {result.niter} iterations")
    print(f"Objective: {result.objective}")  # Should be ≈ 0
else:
    print(f"Failed: {result.message}")

# Tuple-like access (legacy)
coefs = result[0]  # coefficients
ses = result[1]    # standard_errors
vcov = result[2]   # vcov matrix
```

---

### PowerDSLResult

Data class containing power analysis results.

```python
@dataclass
class PowerDSLResult:
    power: np.ndarray                     # Statistical power for each coefficient
    predicted_se: np.ndarray              # Predicted standard errors
    critical_value: float                 # Critical value for hypothesis test
    alpha: float                          # Significance level
    dsl_out: Optional[DSLResult] = None   # Original DSL estimation results
```

---

## Power Analysis

### power_dsl()

Perform statistical power analysis for DSL estimation.

```python
def power_dsl(
    formula: str,
    data: pd.DataFrame,
    labeled_ind: np.ndarray,
    sample_prob: Optional[np.ndarray] = None,
    model: str = "lm",
    fe: Optional[str] = None,
    method: str = "linear",
    n_samples: Optional[int] = None,
    alpha: float = 0.05,
    dsl_out: Optional[DSLResult] = None,
    **kwargs,
) -> PowerDSLResult
```

#### Parameters

- **formula** : `str`
  - Model formula in patsy format (e.g., `"y ~ x1 + x2"`)

- **data** : `pd.DataFrame`
  - Data frame containing variables

- **labeled_ind** : `np.ndarray`
  - Labeled indicator

- **sample_prob** : `Optional[np.ndarray]`, optional
  - Sampling probability

- **model** : `str`, optional (default: `"lm"`)
  - Model type

- **fe** : `Optional[str]`, optional (default: `None`)
  - Fixed effects variable name

- **method** : `str`, optional (default: `"linear"`)
  - Supervised learning method

- **n_samples** : `Optional[int]`, optional (default: `None`)
  - Number of samples for power calculation
  - If None, uses total dataset size

- **alpha** : `float`, optional (default: `0.05`)
  - Significance level for hypothesis tests

- **dsl_out** : `Optional[DSLResult]`, optional (default: `None`)
  - Pre-computed DSL results
  - If None, runs DSL estimation first

#### Returns

- **PowerDSLResult** : Power analysis results

---

## Utility Functions

### summary()

Generate summary table of DSL results.

```python
def summary(result: DSLResult) -> pd.DataFrame
```

Returns a DataFrame with:
- Estimate
- Std. Error
- t value
- Pr(>|t|)

---

### summary_power()

Generate summary table of power analysis results.

```python
def summary_power(result: PowerDSLResult) -> pd.DataFrame
```

Returns a DataFrame with:
- Power
- Predicted SE

---

### plot_power()

Plot power analysis results.

```python
def plot_power(
    result: PowerDSLResult,
    coefficients: Optional[Union[str, List[str]]] = None,
) -> None
```

Creates a plot showing power vs. predicted standard error for selected coefficients.

---

## Usage Examples

### Basic Logistic Regression

```python
import numpy as np
import pandas as pd
from patsy import dmatrices
from dsl import dsl

# Prepare data
np.random.seed(123)
n_total = 1000
n_labeled = 500

# Simulate data
df = pd.DataFrame({
    'outcome': np.random.binomial(1, 0.5, n_total),
    'x1': np.random.normal(0, 1, n_total),
    'x2': np.random.normal(0, 1, n_total),
})

# Create labeled indicator
labeled_indices = np.random.choice(n_total, n_labeled, replace=False)
df['labeled'] = 0
df.loc[labeled_indices, 'labeled'] = 1

# Calculate sampling probability
df['sample_prob'] = n_labeled / n_total

# Prepare design matrix
y, X = dmatrices("outcome ~ x1 + x2", df, return_type="dataframe")

# Run DSL
result = dsl(
    X=X.values,
    y=y.values.flatten(),
    labeled_ind=df['labeled'].values,
    sample_prob=df['sample_prob'].values,
    model="logit"
)

# Display results
print(f"Coefficients: {result.coefficients}")
print(f"Standard Errors: {result.standard_errors}")
print(f"Converged: {result.success}")
```

### With Predictions (Doubly Robust)

```python
from dsl.helpers.dsl_general import dsl_general
from dsl import DSLResult

# Scenario: countyWrong has missing values, but we have predictions
formula = "SendOrNot ~ countyWrong + prefecWrong + connect2b"

# Original data (with missing values for unlabeled)
y, X = dmatrices(formula, df, return_type="dataframe")

# Prediction data (fill missing with predictions)
df_pred = df.copy()
df_pred["countyWrong"] = df_pred["countyWrong"].fillna(
    df_pred["pred_countyWrong"]
)
_, X_pred = dmatrices(formula, df_pred, return_type="dataframe")

# Run DSL with predictions
par, info = dsl_general(
    Y_orig=y.values.flatten(),
    X_orig=X.values,
    Y_pred=y.values.flatten(),
    X_pred=X_pred.values,  # Key: uses predictions
    labeled_ind=df["labeled"].values,
    sample_prob_use=df["sample_prob"].values,
    model="logit"
)

# Create result object
result = DSLResult(
    coefficients=par,
    standard_errors=info["standard_errors"],
    vcov=info["vcov"],
    objective=info["objective"],
    success=info["convergence"],
    message=info["message"],
    niter=info["iterations"],
    model="logit",
    labeled_size=int(np.sum(df["labeled"].values)),
    total_size=len(df),
)

print(f"Objective: {result.objective:.10f}")  # Should be ≈ 0
```

---

## Important Notes

### Doubly Robust Estimation

The DSL framework uses doubly robust (DR) moment conditions:

```
m_dr = m_pred + (m_orig - m_pred) * (labeled_ind / sample_prob)
```

**Key points:**
- When `X_orig = X_pred`, the DR correction term becomes small
- Using separate prediction data (X_pred) enables the full doubly robust property
- This improves efficiency when predictions are good

### Convergence Verification

Always check convergence:

```python
if result.success and abs(result.objective) < 1e-6:
    print("✓ Optimization converged successfully")
else:
    print(f"⚠ Check convergence: objective={result.objective}")
```

### Sample Probability

For simple random sampling:
```python
sample_prob = n_labeled / n_total  # Not n_labeled / n_complete!
```

The denominator should be the total population size, not just complete cases.

### Random Number Generators

**Important**: Python's `np.random.seed()` produces different random numbers than R's `set.seed()` even with the same seed value. This means:
- Python and R will select different random samples
- Different samples → different coefficient estimates
- Both are statistically valid for their respective samples
- This is expected behavior, not a bug

---

## See Also

- [Main README](../README.md) - Installation and quick start
- [Documentation.md](Documentation.md) - R-to-Python function mapping
- [PanChen Example](../PanChen_test/compare_panchen.py) - Complete working example
