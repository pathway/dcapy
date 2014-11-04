"""
Decision Curve Analysis

Author: Matthew Black
"""

import pandas as pd
import numpy as np
from dcapy.validate import dca_input_validation


def stdca(data, outcome, tt_outcome, time_point, predictors,
          thresh_lb=0.01, thresh_ub=0.99, thresh_step=0.01,
          probability=None, harm=None, intervention_per=100,
          cmp_rsk=False):
    """Performs survival-time decision curve analysis on the input data set

    Parameters:
    -----------
    data :
    outcome :
    predictors :
    thresh_lb :
    thresh_ub :
    thresh_step :
    probability :
    harm :
    intervention_per :
    cmp_rsk :

    Returns:
    --------
    """
    raise NotImplementedError()


def dca(data, outcome, predictors,
        thresh_lb=0.01, thresh_ub=0.99, thresh_step=0.01,
        probability=None, harm=None, intervention_per=100):
    """Performs decision curve analysis on the input data set

    Parameters:
    -----------
    data : pd.DataFrame
        the data set to analyze
    outcome : str
        the column of the data frame to use as the outcome
    predictors : str OR list(str)
        the column(s) that will be used to predict the outcome
    thresh_lb : float
        lower bound for threshold probabilities (defaults to 0.01)
    thresh_ub : float
        upper bound for threshold probabilities (defaults to 0.99)
    thresh_step : float
        step size for the set of threshold probabilities [x_start:x_stop]
    probability : bool or list(bool)
    harm : float or list(float)
    intervention_per : int

    Returns:
    --------
    A tuple (net_benefit, interventions_avoided) of pandas DataFrames
    net_benefit :
    interventions_avoided :
    """
    #perform input validation
    data, predictors, probability, harm = dca_input_validation(
        data, outcome, predictors, thresh_lb, thresh_ub, thresh_step, probability,
        harm, intervention_per)

    if isinstance(predictors, str):  #single predictor (univariate analysis)
        #need to convert to a list
        predictors = [predictors]

    ##CALCULATE NET BENEFIT
    num_observations = len(data[outcome])
    event_rate = mean(data[outcome])

    #create DataFrames for holding results
    net_benefit, interventions_avoided = \
        initialize_result_dataframes(event_rate, thresh_lb, thresh_ub, thresh_step)
    for i in range(0, len(predictors)):  #for each predictor
        net_benefit[predictors[i]] = 0.00  #initialize new column of net_benefits
        for j in range(0, len(net_benefit['threshold'])):  #for each observation
            true_positives, false_positives = \
                calc_tf_positives(data, outcome, predictors[i],
                                  net_benefit['threshold'], j)
            #calculate net benefit
            net_benefit_value = \
                calculate_net_benefit(j, net_benefit['threshold'],
                                      true_positives, false_positives,
                                      num_observations)
            net_benefit.set_value(j, predictors[i], net_benefit_value)
        #calculate interventions_avoided
        interventions_avoided[predictors[i]] = calculate_interventions_avoided(
            predictors[i], net_benefit, intervention_per,
            interventions_avoided['threshold'])

    #TODO: implement smoothing with loess function

    return net_benefit, interventions_avoided


def initialize_result_dataframes(event_rate, thresh_lb, thresh_ub, thresh_step):
    """Initializes the net_benefit and interventions_avoided dataFrames for the
    given threshold boundaries and event rate

    Parameters:
    -----------
    event_rate : float
    thresh_lb : float
    thresh_ub : float
    thresh_step : float

    Returns:
    --------
    (pd.DataFrame, pd.DataFrame)
        properly initialized net_benefit, interventions_avoided dataframes
    """
    #initialize threshold series for each dataFrame
    net_benefit = pd.Series(frange(thresh_lb, thresh_ub+thresh_step, thresh_step),
                            name='threshold')
    interventions_avoided = pd.DataFrame(net_benefit)

    #construct 'all' and 'none' columns for net_benefit
    net_benefit_all = event_rate - (1-event_rate)*net_benefit/(1-net_benefit)
    net_benefit_all.name = 'all'
    net_benefit = pd.concat([net_benefit, net_benefit_all], axis=1)
    net_benefit['none'] = 0

    return net_benefit, interventions_avoided


def calc_tf_positives(data, outcome, predictor, net_benefit_threshold, j):
    """Calculate the number of true/false positives for the given parameters

    Parameters:
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

    Returns:
    --------
    (float, float)
        the number of true positives, false positives
    """
    true_positives = false_positives = 0
    #create a filter mask
    filter_mask = data[predictor] >= net_benefit_threshold[j]
    filter_mask_sum = filter_mask.sum()
    if filter_mask_sum == 0:
        pass
    else:
        #get all outcomes where the filter_mask is 'True'
        filtered_outcomes = map(lambda x,y: x if y == True else np.nan,
                                data[outcome],filter_mask)
        filtered_outcomes = [outcome for outcome in filtered_outcomes
                             if outcome is not np.nan]  #drop all NaN values
        true_positives = mean(filtered_outcomes)*filter_mask_sum
        false_positives = (1-mean(filtered_outcomes))*filter_mask_sum

    return true_positives, false_positives


def calculate_net_benefit(index, net_benefit_threshold, harm,
                          true_positives, false_positives, num_observations):
    """Calculates the net benefit for an index within the construction of net_benefit
    loop

    This function calculates the net_benefit for a particular predictor at the given index, however
    the predictor doesn't need to be supplied to this function and should already be determined
    from the true/false positive calculation

    NOTE: true/false positives should be generated by using the calc_tf_positives
    function for the predictor of interest

    Parameters:
    -----------
    net_benefit_threshold : pd.Series
        the 'threshold' column of the net_benefit dataframe for the analysis
    harm : list(float)
        the harm array for the analysis
    true_positives : float
        the number of true positives for the given predictor
    false_positives : float
        the number of false positives for the given predictor
    num_observations : int
        the number of observations in the data set
    index : int
        the index in the Series to compute for

    Returns:
    --------
    the value for the net benefit at 'index' for the predictor
    """
    #normalize the true/false positives by the number of observations
    tp_norm = true_positives/num_observations
    fp_norm = false_positives/num_observations
    #calculate the multiplier for the false positives
    multiplier = net_benefit_threshold[index]/(1-net_benefit_threshold[index])

    return tp_norm - fp_norm*multiplier - harm[index]


def calculate_interventions_avoided(predictor, net_benefit, intervention_per,
                                    interventions_avoided_threshold):
    """Calculate the interventions avoided for the given predictor

    Parameters:
    -----------
    predictor : str
        the predictor to calculate for
    net_benefit : pd.DataFrame
        the net benefit dataframe for the analysis
    intervention_per : int
        TODO
    interventions_avoided_threshold : pd.Series
        the 'threshold' column of the interventions_avoided dataframe

    Returns:
    --------
    pd.Series
        the number of interventions avoided for this predictor
    """
    net_benefit_factor = net_benefit[predictor] - net_benefit[all]
    interv_denom = (interventions_avoided_threshold/(1-interventions_avoided_threshold))

    return net_benefit_factor * intervention_per/interv_denom

def frange(start, stop, step):
    """Generator that can create ranges of floats

    See: http://stackoverflow.com/questions/7267226/range-for-floats

    Parameters:
    -----------
    start : float
       the minimum value of the range
    stop : float
        the maximum value of the range
    step : float
        the step between values in the range
    """
    while start < stop:
        yield start
        start += step


def mean(iterable):
    """Calculates the mean of the given iterable

    Parameters:
    -----------
    iterable: int, float
        an iterable of ints or floats

    Returns:
    --------
    float
        the arithmetic mean of the iterable
    """
    return sum(iterable)/len(iterable)

