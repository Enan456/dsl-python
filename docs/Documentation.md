DSL Framework Documentation
=========================

This document provides a mapping between R and Python functions in the DSL framework, along with pseudocode for key functions.

Core Functions
-------------

dsl()
~~~~~

R: dsl(model, formula, predicted_var, prediction, data, cluster, labeled, sample_prob, index, fixed_effect, sl_method, feature, family, cross_fit, sample_split, seed)

Python: dsl(formula, data, labeled_ind, sample_prob, model, fe, method, **kwargs)

Pseudocode:
```
function dsl(formula, data, labeled_ind, sample_prob, model, fe, method):
    # Parse formula
    y, X = parse_formula(formula, data)
    
    # Split data into labeled and unlabeled
    mask_labeled = labeled_ind == 1
    y_labeled = y[mask_labeled]
    X_labeled = X[mask_labeled]
    y_unlabeled = y[~mask_labeled]
    X_unlabeled = X[~mask_labeled]
    
    if fe is not None:
        # Fixed effects model
        fe_var = get_fixed_effects(data, fe)
        fe_labeled = fe_var[mask_labeled]
        fe_unlabeled = fe_var[~mask_labeled]
        
        # Estimate on labeled data
        pred_labeled, se_labeled, fe_pred = estimate_fixed_effects(
            y_labeled, X_labeled, fe_labeled, method
        )
        
        # Predict on unlabeled data
        pred_unlabeled = predict(X_unlabeled, se_labeled, model)
        
        # Combine predictions
        pred = combine_predictions(pred_labeled, pred_unlabeled, mask_labeled)
        
        # Estimate DSL model
        par, info = dsl_general(
            y, X, pred, X,
            labeled_ind, sample_prob,
            model=model,
            fe_Y=fe_pred,
            fe_X=X
        )
    else:
        # Standard model
        # Estimate on labeled data
        pred_labeled, se_labeled = estimate_supervised(
            y_labeled, X_labeled, method
        )
        
        # Predict on unlabeled data
        pred_unlabeled = predict(X_unlabeled, se_labeled, model)
        
        # Combine predictions
        pred = combine_predictions(pred_labeled, pred_unlabeled, mask_labeled)
        
        # Estimate DSL model
        par, info = dsl_general(
            y, X, pred, X,
            labeled_ind, sample_prob,
            model=model
        )
    
    # Compute residuals and variance-covariance matrix
    resid = compute_residuals(y, X, par, model)
    vcov = compute_vcov(X, par, info["se"], model)
    
    return DSLResult(
        coefficients=par,
        standard_errors=info["se"],
        predicted_values=pred,
        residuals=resid,
        vcov=vcov,
        objective=info["objective"],
        success=info["success"],
        message=info["message"],
        niter=info["niter"],
        model=model,
        labeled_size=sum(mask_labeled),
        total_size=len(y)
    )
```

power_dsl()
~~~~~~~~~~

R: power_dsl(labeled_size, dsl_out, ...)

Python: power_dsl(formula, data, labeled_ind, sample_prob, model, fe, method, n_samples, alpha, dsl_out, **kwargs)

Pseudocode:
```
function power_dsl(formula, data, labeled_ind, sample_prob, model, fe, method, n_samples, alpha, dsl_out):
    if dsl_out is None:
        dsl_out = dsl(formula, data, labeled_ind, sample_prob, model, fe, method)
    
    # Parse formula
    _, X = parse_formula(formula, data)
    
    # Set default number of samples
    if n_samples is None:
        n_samples = len(data)
    
    # Estimate power
    power_results = estimate_power(
        X, dsl_out.coefficients,
        dsl_out.standard_errors,
        n_samples, alpha
    )
    
    return PowerDSLResult(
        power=power_results["power"],
        predicted_se=power_results["predicted_se"],
        critical_value=power_results["critical_value"],
        alpha=power_results["alpha"],
        dsl_out=dsl_out
    )
```

Helper Functions
--------------

estimate_supervised()
~~~~~~~~~~~~~~~~~~

R: fit_model(Y, X, method, ...)

Python: estimate_supervised(Y, X, method, **kwargs)

Pseudocode:
```
function estimate_supervised(Y, X, method):
    if method == "linear":
        model = LinearRegression()
        model.fit(X, Y)
        pred = model.predict(X)
        # Compute standard errors
        residuals = Y - pred
        mse = mean(residuals ** 2)
        X_with_intercept = add_intercept(X)
        se = sqrt(mse * diag(inv(X_with_intercept.T @ X_with_intercept)))
        se = remove_intercept(se)
    elif method == "logistic":
        model = LogisticRegression()
        model.fit(X, Y)
        pred = model.predict_proba(X)[:, 1]
        # Compute standard errors
        p = pred
        w = p * (1 - p)
        X_w = X * w
        se = sqrt(diag(inv(X_w.T @ X_w)))
    elif method == "random_forest":
        if is_binary(Y):
            model = RandomForestClassifier()
            model.fit(X, Y)
            pred = model.predict_proba(X)[:, 1]
        else:
            model = RandomForestRegressor()
            model.fit(X, Y)
            pred = model.predict(X)
        # Compute standard errors using bootstrap
        se = bootstrap_standard_errors(X, Y, pred, method)
    else:
        raise ValueError("Unknown method")
    
    return pred, se
```

estimate_fixed_effects()
~~~~~~~~~~~~~~~~~~~~~

R: fit_fixed_effects(Y, X, fe, method, ...)

Python: estimate_fixed_effects(Y, X, fe, method, **kwargs)

Pseudocode:
```
function estimate_fixed_effects(Y, X, fe, method):
    # Demean data
    Y_demean = Y - mean(Y)
    X_demean = X - mean(X, axis=0)
    
    # Estimate model
    pred, se = estimate_supervised(Y_demean, X_demean, method)
    
    # Compute fixed effects
    fe_pred = zeros_like(Y)
    for i in unique(fe):
        mask = fe == i
        fe_pred[mask] = mean(Y[mask] - pred[mask])
    
    return pred, se, fe_pred
```

estimate_power()
~~~~~~~~~~~~~

R: power_analysis(X, par, se, n_samples, alpha)

Python: estimate_power(X, par, se, n_samples, alpha)

Pseudocode:
```
function estimate_power(X, par, se, n_samples, alpha):
    # Compute critical values
    z_crit = abs(random_normal(10000))
    z_crit = quantile(z_crit, 1 - alpha / 2)
    
    # Compute power
    power = zeros(len(par))
    for i in range(len(par)):
        z = abs(par[i] / se[i])
        power[i] = mean(z > z_crit)
    
    # Compute predicted standard errors
    X_w = X / se
    V = inv(X_w.T @ X_w)
    pred_se = sqrt(diag(V) / n_samples)
    
    return {
        "power": power,
        "predicted_se": pred_se,
        "critical_value": z_crit,
        "alpha": alpha
    }
```

Summary Functions
---------------

summary_dsl()
~~~~~~~~~~~

R: summary(obj, ci, digits)

Python: summary_dsl(result)

Pseudocode:
```
function summary_dsl(result):
    summary = DataFrame({
        "Estimate": result.coefficients,
        "Std. Error": result.standard_errors,
        "t value": result.coefficients / result.standard_errors,
        "Pr(>|t|)": 2 * (1 - t_cdf(
            abs(result.coefficients / result.standard_errors),
            len(result.residuals) - len(result.coefficients)
        ))
    })
    
    return summary
```

summary_power()
~~~~~~~~~~~~

R: summary_power(obj)

Python: summary_power(result)

Pseudocode:
```
function summary_power(result):
    summary = DataFrame({
        "Power": result.power,
        "Predicted SE": result.predicted_se
    })
    
    return summary
```

Plotting Functions
----------------

plot_power()
~~~~~~~~~~

R: plot_power(obj, coef_name)

Python: plot_power(result, coefficients)

Pseudocode:
```
function plot_power(result, coefficients):
    # Get coefficient names
    if result.dsl_out is not None:
        coef_names = get_coefficient_names(result.dsl_out)
    else:
        coef_names = ["beta_" + str(i) for i in range(len(result.power))]
    
    # Select coefficients to plot
    if coefficients is None:
        coefficients = coef_names
    elif is_string(coefficients):
        coefficients = [coefficients]
    
    # Create plot
    figure(figsize=(10, 6))
    for coef in coefficients:
        idx = index(coef_names, coef)
        plot(
            result.predicted_se[idx],
            result.power[idx],
            label=coef,
            marker="o"
        )
    
    xlabel("Predicted Standard Error")
    ylabel("Power")
    title("DSL Power Analysis")
    legend()
    grid(True)
    show()
```

Class Definitions
---------------

DSLResult
~~~~~~~~

R: DSLResult(coef, se, vcov, RMSE, internal)

Python: DSLResult(coefficients, standard_errors, predicted_values, residuals, vcov, objective, success, message, niter, model, labeled_size, total_size)

PowerDSLResult
~~~~~~~~~~~~

R: PowerDSLResult(predicted_se, labeled_size, dsl_out)

Python: PowerDSLResult(power, predicted_se, critical_value, alpha, dsl_out) 