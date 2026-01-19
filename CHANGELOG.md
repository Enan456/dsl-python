# Changelog

All notable changes to the DSL (Design-based Supervised Learning) Python package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-01-18

### Added

- **Input Validation**: Comprehensive `validate_inputs()` function that checks:
  - Array dimensions match between X, y, labeled_ind, and sample_prob
  - No NaN or infinite values in input data
  - labeled_ind contains only binary values (0 and 1)
  - sample_prob values are in valid range (0, 1]
  - Sufficient labeled observations for parameter estimation
  - Binary Y values for logistic regression
  - Clear, actionable error messages for common issues

- **Coefficient Names**:
  - `DSLResult` now includes `coef_names` field
  - Names extracted from formula for formula-based interface
  - Auto-generated names (beta_0, beta_1, ...) for array-based interface
  - `summary()` function displays coefficient names as row index

- **Numerically Stable Sigmoid**:
  - Added `_stable_sigmoid()` function using log-sum-exp trick
  - Prevents overflow/underflow for large absolute values of linear predictor
  - Applied consistently across all logistic regression computations

### Fixed

- **dsl_predict() Function**: Was completely broken - used standard errors as coefficients instead of actual coefficients. Now correctly computes predictions for both linear and logistic models.

- **Fixed Effects Power Analysis**: `power_dsl()` now properly handles fixed effects formula notation with `|` separator.

- **SciPy Deprecation Warning**: Removed deprecated `disp` option from L-BFGS-B optimizer for SciPy 1.18+ compatibility.

### Changed

- Development status upgraded from Alpha to Beta
- Minimum Python version is now 3.9 (previously 3.8)
- Test coverage improved: 60 tests passing

## [0.1.1] - 2025-01-17

### Fixed

- IPW weighting implementation for predicted variables
- Moment condition calculations for doubly-robust estimation
- Test tolerance adjustments for synthetic validation

## [0.1.0] - 2025-01-15

### Added

- Initial release of DSL Python implementation
- Core `dsl()` function for GMM-based estimation
- Support for linear (`lm`) and logistic (`logit`) models
- Fixed effects model (`felm`) support
- `power_dsl()` function for statistical power analysis
- `summary()` and `summary_power()` for result presentation
- `plot_power()` for visualization
- Formula-based interface compatible with R DSL package
- Array-based interface for programmatic usage
- Doubly-robust estimation with prediction support
- Sandwich variance estimator for standard errors

---

## Known Limitations and Gaps

### Current Limitations

1. **Python-R RNG Incompatibility**
   - Python's `np.random.seed()` and R's `set.seed()` produce different random sequences
   - This means labeled sample selection differs between implementations
   - Results cannot be compared numerically without using identical pre-generated samples

2. **Fixed Effects Model**
   - Fixed effects are implemented via dummy variable expansion
   - May be inefficient for large numbers of fixed effect groups
   - No within-transformation optimization as in R's `lfe` package

3. **Cross-fitting and Sample Splitting**
   - `cross_fit` and `sample_split` parameters are accepted but not yet implemented
   - Currently uses single-shot estimation

4. **Supervised Learning Integration**
   - `sl_method`, `feature`, and `family` parameters are placeholders
   - No automatic ML model training for predictions
   - Users must provide pre-computed predictions

5. **Clustered Standard Errors**
   - Cluster-robust variance estimation is partially implemented
   - Not fully tested with real-world clustered data

### Planned Improvements

- [ ] Implement cross-fitting for debiased estimation
- [ ] Add built-in supervised learning methods (Random Forest, XGBoost)
- [ ] Optimize fixed effects via within-transformation
- [ ] Add cluster-robust standard errors
- [ ] Improve convergence diagnostics and warnings
- [ ] Add confidence interval computation to results

### Compatibility Notes

- Tested with Python 3.9, 3.10, 3.11
- Requires NumPy >= 1.20.0, SciPy >= 1.7.0
- Uses statsmodels for initial parameter estimation
- Uses patsy for formula parsing (R-style formulas)
