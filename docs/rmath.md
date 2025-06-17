# Mathematical Implementation Summary: R DSL Package

This document summarizes the core mathematical steps involved in the R `dsl` package, primarily based on the functions within `helper_dsl_general.R` and `helper_moment.R`.

## I. Estimation (`dsl_general_moment_est`)

The main estimation workflow follows these steps:

1.  **Data Standardization:**
    *   Predictor variables (X) are standardized based on the means and standard deviations of the *prediction* dataset (`X_pred`).
    *   If an intercept is present, columns are centered and scaled: `X_use = (X - mean(X_pred)) / sd(X_pred)`.
    *   If no intercept, columns are only scaled: `X_use = X / sd(X_pred)`.
    *   The standardization means (`mean_X`) and standard deviations (`sd_X`) are stored for later rescaling.

2.  **Point Estimation (GMM):**
    *   The core parameters are estimated by minimizing a GMM objective function using `stats::optim` (typically with BFGS or a similar method).
    *   **Objective Function (`dsl_general_moment`):** The function minimizes the sum of squared *average* moment conditions:
        \[ Q(\beta) = \mathbf{\bar{m}}(\beta)^T \mathbf{\bar{m}}(\beta) = \sum_{j=1}^k \left( \frac{1}{n} \sum_{i=1}^n m_{ij}(\beta) \right)^2 \]
        where \( m_{ij}(\beta) \) is the j-th moment contribution for the i-th observation, calculated using the doubly robust moments described below. *Note: The R code comments mention minimizing \( \sum m^2 \), but the `optim` call uses `apply(m_dr, 2, mean)` which aligns with the GMM objective.*
    *   **Gradient:** `optim` typically uses numerical differentiation (or an analytical gradient if supplied, though not explicitly seen in the primary workflow for the GMM step itself).

3.  **Moment Calculation (`lm_dsl_moment_base`, `logit_dsl_moment_base`):**
    *   Calculates the \( n \times k \) matrix \( \mathbf{M} \) where each row \( \mathbf{m}_i(\beta) \) is the moment contribution for observation \( i \).
    *   Uses the doubly robust (DR) moment structure:
        \[ \mathbf{m}_{\text{dr}, i}(\beta) = \mathbf{m}_{\text{pred}, i}(\beta) + \frac{I_i}{\pi_i} (\mathbf{m}_{\text{orig}, i}(\beta) - \mathbf{m}_{\text{pred}, i}(\beta)) \]
        where \( I_i \) is the labeled indicator (1 if labeled, 0 otherwise) and \( \pi_i \) is the sampling probability.
    *   **Linear Model (LM):**
        *   \( \mathbf{m}_{\text{orig}, i}(\beta) = \mathbf{x}_i (y_i - \mathbf{x}_i^T \beta) \) (only non-zero if labeled)
        *   \( \mathbf{m}_{\text{pred}, i}(\beta) = \mathbf{x}_i (\hat{y}_i - \mathbf{x}_i^T \beta) \)
    *   **Logistic Model (Logit):**
        *   \( p_i(\beta) = \frac{1}{1 + e^{-\mathbf{x}_i^T \beta}} \)
        *   \( \mathbf{m}_{\text{orig}, i}(\beta) = \mathbf{x}_i (y_i - p_i(\beta)) \) (only non-zero if labeled)
        *   \( \mathbf{m}_{\text{pred}, i}(\beta) = \mathbf{x}_i (\hat{y}_i - p_i(\beta)) \)

## II. Variance Estimation (`dsl_general_moment_est` post-optimization)

Uses the standard GMM sandwich variance estimator formula:

\[ \text{Var}(\hat{\beta}) = \frac{1}{n} (\mathbf{J}^{-1}) \mathbf{\Omega} (\mathbf{J}^{-T}) \]

where \( \hat{\beta} \) are the estimated (scaled) parameters from `optim`.

1.  **Jacobian (`dsl_general_Jacobian`, `lm_dsl_Jacobian`, `logit_dsl_Jacobian`):**
    *   Calculates the \( k \times k \) average Jacobian matrix \( \mathbf{J} \) evaluated at the estimated parameters \( \hat{\beta} \).
    *   \( \mathbf{J} = E \left[ \frac{\partial \mathbf{m}_i(\beta)}{\partial \beta^T} \right] \approx \frac{1}{n} \sum_{i=1}^n \frac{\partial \mathbf{m}_i(\hat{\beta})}{\partial \beta^T} \)
    *   **LM:** \( \mathbf{J}_{lm} \approx -\frac{1}{n} \mathbf{X}_{\text{use}}^T \mathbf{X}_{\text{use}} \) (using the standardized design matrix).
    *   **Logit:** \( \mathbf{J}_{logit} \approx -\frac{1}{n} \mathbf{X}_{\text{use}}^T \mathbf{W}_{\text{dr}} \mathbf{X}_{\text{use}} \), where \( \mathbf{W}_{\text{dr}} \) is a diagonal matrix of doubly robust weights \( w_{\text{dr}, i} = w_{\text{pred}, i} + \frac{I_i}{\pi_i} (w_{\text{orig}, i} - w_{\text{pred}, i}) \) with \( w_i = p_i(1-p_i) \).

2.  **Meat Matrix (`dsl_general_moment_base_decomp`):**
    *   Estimates the \( k \times k \) variance of the moment conditions \( \mathbf{\Omega} = E[\mathbf{m}_i(\beta) \mathbf{m}_i(\beta)^T] \).
    *   Calculated as \( \hat{\mathbf{\Omega}} = \frac{1}{n} \sum_{i=1}^n \mathbf{m}_{\text{dr}, i}(\hat{\beta}) \mathbf{m}_{\text{dr}, i}(\hat{\beta})^T = \frac{1}{n} \mathbf{M}^T \mathbf{M} \).
    *   The function decomposes \( \hat{\mathbf{\Omega}} \) into components (`main_1`, `main_23`) related to the variance of the original moments and the prediction moments/covariance.

3.  **Bread:**
    *   The "bread" is the inverse Jacobian: \( \mathbf{J}^{-1} \). Computed using `solve(J)`.

4.  **Scaled Variance:**
    *   The variance-covariance matrix for the *scaled* parameters is computed:
        \[ \mathbf{V}_{\text{scaled}} = \frac{1}{n} (\mathbf{J}^{-1}) \hat{\mathbf{\Omega}} (\mathbf{J}^{-T}) \]

## III. Rescaling (`dsl_general_moment_est`)

The estimated coefficients and their variance-covariance matrix are rescaled back to the original scale of the variables.

1.  **Rescaling Matrix (Jacobian of Transformation):**
    *   Calculates the \( k \times k \) matrix \( \mathbf{D} = \frac{\partial \beta_{\text{orig}}}{\partial \beta_{\text{scaled}}} \).
    *   **With Intercept:**
        \[ \mathbf{D} = \begin{pmatrix} 1 & -\frac{\mu_1}{\sigma_1} & -\frac{\mu_2}{\sigma_2} & \dots \\ 0 & \frac{1}{\sigma_1} & 0 & \dots \\ 0 & 0 & \frac{1}{\sigma_2} & \dots \\ \vdots & \vdots & \vdots & \ddots \end{pmatrix} \]
        where \( \mu_j, \sigma_j \) are the mean and standard deviation used for standardization (from `X_pred`).
    *   **Without Intercept:** \( \mathbf{D} \) is a diagonal matrix with \( 1/\sigma_j \) on the diagonal.

2.  **Final Coefficients:**
    *   \( \hat{\beta}_{\text{orig}} = \mathbf{D} \hat{\beta}_{\text{scaled}} \)

3.  **Final Variance:**
    *   \( \mathbf{V}_{\text{orig}} = \mathbf{D} \mathbf{V}_{\text{scaled}} \mathbf{D}^T \)

## IV. Standard Errors and P-Values (Implicitly in `summary.dsl`)

*   Standard errors are the square root of the diagonal elements of \( \mathbf{V}_{\text{orig}} \).
*   P-values are typically calculated using a z-test (comparing `Estimate / Std.Error` to a standard normal distribution), consistent with asymptotic theory for GMM/Logit. 