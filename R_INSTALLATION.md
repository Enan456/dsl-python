# R Environment Setup and DSL Package Installation Guide

## System Information

- **OS**: macOS (Darwin 24.4.0)
- **Architecture**: ARM64 (Apple Silicon)
- **Date**: 2026-01-03

## Installation Steps

### 1. R Installation

R was already installed via Homebrew:

```bash
# Verify R installation
R --version
# Output: R version 4.5.2 (2025-10-31)
```

If R is not installed, install it with:
```bash
brew install r
```

### 2. CMake Installation

CMake is required for compiling the `nloptr` dependency:

```bash
# Install CMake
brew install cmake

# Verify installation
cmake --version
# Output: cmake version 4.2.1
```

### 3. devtools Package

The `devtools` package is required to install packages from GitHub:

```R
# Check if devtools is installed, install if not
if (!requireNamespace("devtools", quietly = TRUE)) {
  install.packages("devtools", repos = "https://cloud.r-project.org")
}

library(devtools)
packageVersion("devtools")
# Output: 2.4.6
```

### 4. DSL Package Installation

Install the DSL package from the original GitHub repository:

```R
library(devtools)
devtools::install_github("naoki-egami/dsl", force = TRUE)

# The installation will automatically install dependencies:
# - nloptr (2.2.1)
# - lme4 (1.1-38)
# - arm (1.14-4)

# Verify installation
library(dsl)
packageVersion("dsl")
# Output: 0.1.0
```

### Installation Script

A complete installation script is available in `install_dsl.R`:

```bash
Rscript install_dsl.R
```

## Running DSL on PanChen Dataset

### Quick Start

```bash
Rscript run_panchen_dsl.R
```

This script will:
1. Load the PanChen dataset from `PanChen_test/PanChen.rdata`
2. Prepare the labeled indicator and sampling probability
3. Run DSL estimation with logistic regression
4. Save results to `r_panchen_output.txt`

### Manual Execution

```R
library(dsl)

# Load data
load("PanChen_test/PanChen.rdata")

# Create labeled indicator
PanChen$labeled <- ifelse(is.na(PanChen$countyWrong), 0, 1)

# Create sampling probability (equal probability)
PanChen$sample_prob <- sum(PanChen$labeled) / nrow(PanChen)

# Run DSL estimation
out <- dsl(
  model = "logit",
  formula = SendOrNot ~ countyWrong + prefecWrong + connect2b +
            prevalence + regionj + groupIssue,
  predicted_var = "countyWrong",
  prediction = "pred_countyWrong",
  data = PanChen,
  labeled = "labeled",
  sample_prob = "sample_prob",
  sl_method = "glm",
  family = "binomial",
  seed = 1234
)

# View results
summary(out)
```

## Key Parameters

- **model**: Regression model type (`"lm"`, `"logit"`, `"felm"`)
- **formula**: R formula specifying the regression model
- **predicted_var**: Name of variable to be predicted
- **prediction**: Name of column containing predictions
- **labeled**: Column name indicating labeled observations (1) vs unlabeled (0)
- **sample_prob**: Column name with sampling probabilities
- **sl_method**: Supervised learning method for internal prediction
  - Use `available_method()` to see all options
  - Common choices: `"glm"`, `"grf"`, `"lm"`, `"randomForest"`
- **family**: Variable type (`"gaussian"`, `"binomial"`)
- **seed**: Random seed for reproducibility

## Available Methods

To see all supported supervised learning methods:

```R
library(dsl)
available_method()
```

Common methods include:
- `glm` - Generalized Linear Models
- `grf` - Generalized Random Forest (default)
- `lm` - Linear Models
- `randomForest` - Random Forest
- `glmnet` - Elastic Net
- `xgboost` - XGBoost

## Troubleshooting

### nloptr compilation fails

**Issue**: CMake not found during nloptr installation

**Solution**:
```bash
brew install cmake
```

### DSL package installation fails

**Issue**: Dependencies not installing correctly

**Solution**:
```R
# Manually install dependencies first
install.packages(c("nloptr", "lme4", "arm"))

# Then retry DSL installation
devtools::install_github("naoki-egami/dsl", force = TRUE)
```

### "sl_method not supported" error

**Issue**: Using invalid `sl_method` parameter

**Solution**:
```R
# Check available methods
available_method()

# Use a valid method (e.g., "glm" instead of "linear")
```

## Validation

The installation has been validated against the reference output:
- Results match target output within numerical precision (< 0.5% difference)
- All statistical significance levels match exactly
- Standard errors are virtually identical

See `r_validation_comparison.md` for detailed validation results.

## References

- **Original R Package**: https://github.com/naoki-egami/dsl
- **Documentation**: http://naokiegami.com/dsl
- **Paper**: Egami et al. (2024) "Using Large Language Model Annotations for the Social Sciences"
