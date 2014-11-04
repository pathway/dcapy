"""
Decision Curve Analysis

Algorithms for decision curve analysis

Author: Matthew Black
"""

from dcapy.calc import *
from dcapy.validate import dca_input_validation


def dca(data, outcome, predictors,
        thresh_lb=0.01, thresh_ub=0.99, thresh_step=0.01,
        probability=None, harm=None, intervention_per=100):
    """Performs decision curve analysis on the input data set

    Parameters
    ----------
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

    Returns
    -------
    pd.DataFrame, pd.DataFrame
        A tuple of length 2 with net_benefit, interventions_avoided
        net_benefit : TODO
        interventions_avoided : TODO
    """
    #perform input validation
    data, predictors, probability, harm = dca_input_validation(
        data, outcome, predictors, thresh_lb, thresh_ub, thresh_step, probability,
        harm, intervention_per)

    if isinstance(predictors, str):  # single predictor (univariate analysis)
        #need to convert to a list
        predictors = [predictors]

    ##CALCULATE NET BENEFIT
    num_observations = len(data[outcome])
    event_rate = mean(data[outcome])

    #create DataFrames for holding results
    net_benefit, interventions_avoided = \
        initialize_result_dataframes(event_rate, thresh_lb, thresh_ub, thresh_step)
    for i in range(0, len(predictors)):  # for each predictor
        net_benefit[predictors[i]] = 0.00  # initialize new column of net_benefits

        for j in range(0, len(net_benefit['threshold'])):  # for each observation
            #calculate true/false positives
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


def stdca(data, outcome, tt_outcome, time_point, predictors,
          thresh_lb=0.01, thresh_ub=0.99, thresh_step=0.01,
          probability=None, harm=None, intervention_per=100,
          cmp_rsk=False):
    """Performs survival-time decision curve analysis on the input data set

    Parameters
    ----------
    data : pd.DataFrame
        the data set to analyze
    outcome : str
        the column of the data frame to use as the outcome
    tt_outcome:
        TODO
    time_point:
        TODO
    predictors : str OR list(str)
        the column(s) that will be used to predict the outcome
    thresh_lb : float
        lower bound for threshold probabilities (defaults to 0.01)
    thresh_ub : float
        upper bound for threshold probabilities (defaults to 0.99)
    thresh_step : float
        step size for the set of threshold probabilities [x_start:x_stop]
    probability :
        TODO
    harm :
        TODO
    intervention_per :
        TODO
    cmp_rsk :
        TODO

    Returns
    -------
    """
    raise NotImplementedError()
