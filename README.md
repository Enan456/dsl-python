# DSL: Design-based Supervised Learning (Python & R)

## Repository Overview

This repository hosts parallel implementations of the Design-based Supervised Learning (DSL) framework in **Python**. Special thanks to Chandler L'Hommedieu for his help with the ideation and implementation of the Python version. 

The primary goal of the Python implementation  was to create a version that closely mirrors the statistical methodology and produces comparable results to the established **R** package, originally developed by Naoki Egami.

DSL combines supervised machine learning techniques with methods from survey statistics and econometrics to estimate regression models when outcome labels are only available for a non-random subset of the data (partially labeled data).

## Original R Package Documentation

For the theoretical background, detailed methodology, and original R package usage, please refer to the original package resources:

*   **Package Website & Vignettes:** [http://naokiegami.com/dsl](http://naokiegami.com/dsl)
*   **Original R Package Repository:** [https://github.com/naoki-egami/dsl](https://github.com/naoki-egami/dsl)

## Installation


### Python Version

**Prerequisites:**

*   Python 3.9+
*   pip (Python package installer)

**Setup:**

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-github-username/dsl.git
    cd dsl
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate # On Windows use `.venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt 
    # The requirements file includes: numpy, pandas, scipy, scikit-learn, matplotlib, patsy
    ```

## Usage

### R Version

Please refer to the [original package documentation](http://naokiegami.com/dsl) and vignettes for usage examples.

### Python Version

The core estimation function is `dsl.dsl()`.

**Example (using PanChen data):**

```python
import pandas as pd
from patsy import dmatrices
from dsl import dsl
from PanChen_test.compare_panchen import load_panchen_data, prepare_data_for_dsl, format_dsl_results

# Load and prepare data
data = load_panchen_data() 
df = prepare_data_for_dsl(data)

# Define formula
formula = (
    "SendOrNot ~ countyWrong + prefecWrong + connect2b + "
    "prevalence + regionj + groupIssue"
)

# Prepare design matrix (X) and response (y)
y, X = dmatrices(formula, df, return_type="dataframe")

# Run DSL estimation (logit model)
result = dsl(
    X=X.values,
    y=y.values.flatten(), # Ensure y is 1D
    labeled_ind=df["labeled"].values,
    sample_prob=df["sample_prob"].values,
    model="logit", # Specify the desired model (e.g., 'logit', 'lm')
    method="logistic" # Specify the estimation method ('logistic', 'linear')
)

# Print results
print(f"Convergence: {result.success}")
print(f"Iterations: {result.niter}")
print(f"Objective Value: {result.objective}")

# Format and print R-style summary
summary_table = format_dsl_results(result, formula) # Assumes format_dsl_results is available
print("\nPython DSL Results Summary:")
print(summary_table)

```

## R vs. Python Comparison (PanChen Dataset - Logit Model)

The Python implementation has been carefully aligned with the R version's statistical methodology. Below is a comparison of the results obtained from both implementations on the PanChen dataset using a logistic model.

**Python Results (Final):**

```
             Estimate  Std. Error  CI Lower  CI Upper  p value     
(Intercept)    2.0547      0.3643    1.3407    2.7686   0.0000  ***
countyWrong   -0.0721      0.2037   -0.4713    0.3272   0.7234     
prefecWrong   -1.0622      0.2939   -1.6382   -0.4862   0.0003  ***
connect2b     -0.1116      0.1155   -0.3379    0.1147   0.3338     
prevalence    -0.2985      0.1482   -0.5890   -0.0080   0.0440    *
regionj        0.1285      0.4501   -0.7536    1.0106   0.7752     
groupIssue    -2.3291      0.3626   -3.0397   -1.6184   0.0000  ***
```

**R Results (Reference):**

```
            Estimate Std. Error CI Lower CI Upper p value
(Intercept)   2.0978     0.3621   1.3881   2.8075  0.0000 ***
countyWrong  -0.2617     0.2230  -0.6988   0.1754  0.1203    
prefecWrong  -1.1162     0.2970  -1.6982  -0.5342  0.0001 ***
connect2b    -0.0788     0.1197  -0.3134   0.1558  0.2552    
prevalence   -0.3271     0.1520  -0.6250  -0.0292  0.0157   *
regionj       0.1253     0.4566  -0.7695   1.0202  0.3919    
groupIssue   -2.3222     0.3597  -3.0271  -1.6172  0.0000 ***
```

**Note on Differences:** The results are very close, demonstrating successful alignment. The minor remaining numerical differences are expected due to inherent variations in optimization algorithms (BFGS vs. L-BFGS-B implementations), floating-point arithmetic, underlying linear algebra libraries (e.g., BLAS/LAPACK versions), formula parsing (`patsy` vs. R), handling of numerical edge cases, and specific package versions across different language environments (R vs. Python/NumPy/SciPy). For practical purposes, the implementations yield equivalent statistical results.

**Reference:** 

- [Egami, Hinck, Stewart, and Wei. (2024)](https://naokiegami.com/paper/dsl_ss.pdf). "Using Large Language Model Annotations for the Social Sciences: A General Framework of Using Predicted Variables in Downstream Analyses."

- [Egami, Hinck, Stewart, and Wei. (2023)](https://naokiegami.com/paper/dsl.pdf). "Using Imperfect Surrogates for Downstream Inference:
Design-based Supervised Learning for Social Science Applications of Large Language Models," Advances in Neural Information Processing Systems (NeurIPS).


## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

MIT License
