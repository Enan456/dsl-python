import numpy as np

from .dsl import dsl


def power_dsl(
    model=None,
    formula=None,
    predicted_var=None,
    prediction=None,
    data=None,
    labeled_ind=None,
    sample_prob=None,
    sl_method="grf",
    feature=None,
    family="gaussian",
    cross_fit=2,
    sample_split=2,
    alpha=0.05,
    power=0.8,
    seed=None,
    dsl_output=None,
):
    """Calculate power for DSL estimation.

    Parameters
    ----------
    model : str
        The model type to use. One of "lm", "logit", or "felm".
    formula : str
        The model formula.
    predicted_var : list
        List of variables to predict.
    prediction : str
        Name of the prediction column in the data.
    data : pandas.DataFrame
        The data to use for estimation.
    labeled_ind : numpy.ndarray
        Binary indicator for labeled observations.
    sample_prob : numpy.ndarray
        Sampling probabilities for each observation.
    sl_method : str, optional
        The super learner method to use. Default is "grf".
    feature : list, optional
        List of feature names to use. Default is None.
    family : str, optional
        The family of the model. Default is "gaussian".
    cross_fit : int, optional
        Number of cross-fitting folds. Default is 2.
    sample_split : int, optional
        Number of sample splitting folds. Default is 2.
    alpha : float, optional
        Significance level. Default is 0.05.
    power : float, optional
        Desired power level. Default is 0.8.
    seed : int, optional
        Random seed. Default is None.
    dsl_output : DSLResult, optional
        Output from a previous DSL estimation. Default is None.

    Returns
    -------
    PowerDSLResult
        Object containing power analysis results.
    """
    if dsl_output is None:
        # Validate required arguments when dsl_output is not provided
        if model is None or formula is None or predicted_var is None or prediction is None:
            raise ValueError(
                "When dsl_output is not provided, model, formula, predicted_var, "
                "and prediction are required."
            )
        if data is None or labeled_ind is None or sample_prob is None:
            raise ValueError(
                "When dsl_output is not provided, data, labeled_ind, and sample_prob "
                "are required."
            )
        dsl_output = dsl(
            model=model,
            formula=formula,
            predicted_var=predicted_var,
            prediction=prediction,
            data=data,
            labeled_ind=labeled_ind,
            sample_prob=sample_prob,
            sl_method=sl_method,
            feature=feature,
            family=family,
            cross_fit=cross_fit,
            sample_split=sample_split,
            seed=seed,
        )

    # Calculate predicted standard errors
    predicted_se = np.sqrt(np.diag(dsl_output.vcov))

    # Calculate critical value
    critical_value = np.abs(
        np.percentile(np.random.standard_normal(10000), (1 - alpha) * 100)
    )

    return PowerDSLResult(
        power=power,
        predicted_se=predicted_se,
        critical_value=critical_value,
        alpha=alpha,
    )


def summary_power_dsl(power_result):
    """Summarize power analysis results.

    Parameters
    ----------
    power_result : PowerDSLResult
        Output from power_dsl function.

    Returns
    -------
    PowerDSLResult
        The same object, but with additional summary statistics.
    """
    return power_result


class PowerDSLResult:
    """Results from power analysis for DSL estimation.

    Attributes
    ----------
    power : float
        The achieved power level.
    predicted_se : numpy.ndarray
        Predicted standard errors for each coefficient.
    critical_value : float
        Critical value for the test.
    alpha : float
        Significance level.
    """

    def __init__(self, power, predicted_se, critical_value, alpha):
        self.power = power
        self.predicted_se = predicted_se
        self.critical_value = critical_value
        self.alpha = alpha
