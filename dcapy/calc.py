import pandas as pd
import numpy as np


def initialize_result_dataframes(event_rate, thresh_lo, thresh_hi, thresh_step):
    """Initializes the net_benefit and interventions_avoided dataFrames for the
    given threshold boundaries and event rate

    Parameters
    ----------
    event_rate : float
    thresh_lo : float
    thresh_hi : float
    thresh_step : float

    Returns
    -------
    tuple(pd.DataFrame, pd.DataFrame)
        properly initialized net_benefit, interventions_avoided dataframes
    """
    #initialize threshold series for each dataFrame
    net_benefit = pd.Series(frange(thresh_lo, thresh_hi+thresh_step, thresh_step),
                            name='threshold')
    interventions_avoided = pd.DataFrame(net_benefit).set_index("threshold")

    #construct 'all' and 'none' columns for net_benefit
    net_benefit_all = event_rate - (1-event_rate)*net_benefit/(1-net_benefit)
    net_benefit_all.name = 'all'
    net_benefit = pd.concat([net_benefit, net_benefit_all], axis=1)
    net_benefit['none'] = 0
    net_benefit = pd.DataFrame(net_benefit).set_index("threshold")

    return net_benefit, interventions_avoided


def calc_tf_positives(data, outcome, predictor, net_benefit_threshold, j):
    """Calculate the number of true/false positives for the given parameters

    Parameters
    ----------
    data : pd.DataFrame
        the data set to analyze
    outcome : str
        the column of the data frame to use as the outcome
    predictor : str
        the column to use as the predictor for this calculation
    net_benefit_threshold : pd.Series
        the threshold column of the net_benefit data frame
    j : int
        the index in the net_benefit data frame to use

    Returns
    -------
    tuple(float, float)
        the number of true positives, false positives
    """
    filter_mask = data[predictor] >= net_benefit_threshold[j]
    true_positives = data.loc[filter_mask, outcome].dropna().sum()
    false_positives = filter_mask.sum() - true_positives

    return true_positives, false_positives


def calculate_net_benefit(data, outcome, predictor, thresholds, harm=None):
    harm = harm or 0.0
    
    num_observations = data.shape[0]
    df = pd.concat([data[predictor] >= x for x in thresholds], axis=1)
    tp = df.apply(lambda x: data.loc[x.values, outcome].dropna().sum()).values
    fp = df.sum(axis=0).values - tp
    
    tp_norm = tp / num_observations
    fp_norm = fp / num_observations
    #calculate the multiplier for the false positives
    multiplier = thresholds / (1.0 - thresholds)

    return tp_norm - fp_norm*multiplier - harm

def calculate_interventions_avoided(predictor, net_benefit, intervention_per,
                                    interventions_avoided_threshold):
    """Calculate the interventions avoided for the given predictor

    Parameters
    ----------
    predictor : str
        the predictor to calculate for
    net_benefit : pd.DataFrame
        the net benefit dataframe for the analysis
    intervention_per : int
        TODO
    interventions_avoided_threshold : pd.Series
        the 'threshold' column of the interventions_avoided dataframe

    Returns
    -------
    pd.Series
        the number of interventions avoided for this predictor
    """
    net_benefit_factor = net_benefit[predictor] - net_benefit['all']
    interv_denom = (interventions_avoided_threshold/(1-interventions_avoided_threshold))

    return net_benefit_factor * intervention_per/interv_denom


def competing_risk(data, outcome, tt_outcome, use_kmf):
    """Gets the probability of the event for all subjects

    Notes
    -----
    This is used for the net benefit associated with treating all patients

    Parameters
    ----------
    data : pd.DataFrame
        the dataset to analyze
    outcome : str
        the column in `data` with outcome values
    tt_outcome : str
        the column in `data` with times to the outcome values 
    use_kmf : bool
        the algorithm to use for fitting the survival curve
        if `True`, use KaplanMeier; if `False`, use cumulative increase
    
    Returns
    -------

    """
    raise NotImplementedError()
    #construct a new dataframe of just the outcome and tt_outcome columns
    df = pd.DataFrame({outcome: data[outcome].values, 
                       tt_outcome: data[tt_outcome].values})
    if use_kmf:
        from statsmodels.sandbox.survival2 import KaplanMeier
        kmf = KaplanMeier(df.values, 1)
        kmf.fit()
    else:  # use cuminc
        pass


def lowess_smooth_results(predictor, net_benefit, interventions_avoided, 
                          lowess_frac):
    """Smooths the result data using local regression

    This function uses the index of the passed in dataframes as exogenous
    values for the smoothing

    Parameters
    ----------
    net_benefit : pd.DataFrame
        a dataframe of net benefit results
    interventions_avoided : pd.DataFrame
        a dataframe of interventions avoided results
    predictors : str
        the predictor to smooth for
        (must match a column in both net_benefit/interventions_avoided dataframes)
    lowess_frac : float
        the fraction of the data used when estimating each endogenous value

    Returns:
    --------
    tuple(pd.DataFrame, pd.DataFrame)
        smoothed net_benefit and interventions_avoided dataframes
    """
    
    #call smoothing function
    import statsmodels.api as sm
    lowess = sm.nonparametric.lowess
    smoothed_net_benefit = lowess(net_benefit[predictor], net_benefit.index.values,
                                  frac=lowess_frac, missing='drop')
    smoothed_interv = lowess(interventions_avoided[predictor],
                             interventions_avoided.index.values,
                             frac=lowess_frac, missing='drop')
    return pd.Series(smoothed_net_benefit[:, -1], index=net_benefit.index.values,
                     name='{}_sm'.format(predictor)), pd.Series(
                         smoothed_interv[:, -1], index=interventions_avoided.index.values,
                         name='{}_sm'.format(predictor))


def frange(start, stop, step):
    """Generator that can create ranges of floats

    Credit: http://stackoverflow.com/questions/7267226/range-for-floats

    Parameters
    ----------
    start : float
       the minimum value of the range
    stop : float
        the maximum value of the range
    step : float
        the step between values in the range

    Yields
    ------
    float
        the next number in the range `start` to `stop`-`step`
    """
    while start < stop:
        yield start
        start += step


def mean(iterable):
    """Calculates the mean of the given iterable

    Parameters
    ----------
    iterable: int, float
        an iterable of ints or floats

    Returns
    -------
    float
        the arithmetic mean of the iterable
    """
    return sum(iterable)/len(iterable)
