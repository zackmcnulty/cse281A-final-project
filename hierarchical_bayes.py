import numpy as np
from scipy.optimize import differential_evolution
import pandas as pd
from pathlib import Path
from scipy.stats import betabinom

"""
This code is used to run the Hierarchical Bayesian analysis of the TPR/FPR of individual mice in the behavior task
as described in the paper. It uses the output of the "extract_mouse_metrics.py" program, a csv of the performance
metrics for each individual mouse in each experiment, to perform the analysis

"""

save_results_to_csv = False  # whether to save the hierarchical bayes estimates for TPR/FPR to a csv
familiar_destination = "data\\results\\hbayes\\familiar_tpr_fpr.csv"  # destination to save familiar TPR, FPR results
novel_destination = "data\\results\\hbayes\\novel_tpr_fpr.csv"  # destination to save novel familiar TPR, FPR results


def aggregate_sessions(table):

    """
    Calculate true positive rates and false positive rates for each mouse, aggregating the trials across the several
    behavior sessions each mouse performed.

    :param table:  metric data table
    :return:       aggregated table
    """
    new_table = table.groupby(by=['mouse_id']).agg(['sum'])
    new_table['TPR'] = new_table.apply(lambda x: x['hit_trial_count'] / x['go_trial_count'], axis=1)
    new_table['FPR'] = new_table.apply(lambda x: x['false_alarm_trial_count'] / x['catch_trial_count'], axis=1)

    return new_table


def empirical_bayes(X, n, title=""):
    """
    given a vector X of observations, one for each individual, use empirical bayes to estimate alpha, beta in the
    Beta-Binomial heirarchical model discussed in the paper. Estimate alpha and beta using mle


    :param X:     vector; total number of successes per individual
    :param n:     vector same length as n; total number of observations per individual
    :return:
    """

    # -1 is because we want to maximize the sum, or equivalently minimize its negation
    #func = lambda x: -1 * np.sum(np.log([betabinom.pmf(X[i], n[i], x[0], x[1]) for i in range(len(X))]))
    func = lambda x: -1 * np.prod([betabinom.pmf(X[i], n[i], x[0], x[1]) for i in range(len(X))])

    result = differential_evolution(func, x0=[0.5, 0.5], bounds=[(0, 100), (0, 100)], popsize=100)
    mle_alpha, mle_beta = result.x

    print(title)
    print("The MLE Estimates for (alpha, beta) : ({}, {})".format(mle_alpha, mle_beta))

    empirical_theta = (mle_alpha + X) / (mle_alpha + mle_beta + n)
    return empirical_theta


# MAIN METHOD #########################################
if __name__ == "__main__":


    # metric table CSV
    filepath = Path("data\\results\\metrics_test.csv")

    # features of interest
    features = ['mouse_id', 'go_trial_count',
                'catch_trial_count', 'hit_trial_count', 'miss_trial_count',
                'false_alarm_trial_count', 'correct_reject_trial_count']

    # load performance metrics
    metrics_table = pd.read_csv(filepath, index_col="behavior_session_id")

    print(metrics_table.columns)

    # throw out passive session because the mice are not performing the task: they are simply viewing the
    # stimulus and neural recordings are taken; no licking involved...
    metrics_table = metrics_table.loc[~ metrics_table['session_type'].str.contains('passive')]

    # split into familiar trials and novel trials. OPHYS0, 1, 2,3 correspond to familiar, 4,5,6 to novel
    familiar_table = metrics_table.loc[metrics_table['session_type'].str.contains('[0-3]', regex=True)][features]
    novel_table = metrics_table.loc[metrics_table['session_type'].str.contains('[4-6]', regex=True)][features]

    # Choose which statistics you want to aggregate for each mouse
    familiar_table = aggregate_sessions(familiar_table)
    novel_table = aggregate_sessions(novel_table)

    print("\nFamiliar TPR")
    empirical_familiar_TPR = empirical_bayes(familiar_table['hit_trial_count'].values,
                                             familiar_table['go_trial_count'].values)
    print("\nFamiliar FPR")
    empirical_familiar_FPR = empirical_bayes(familiar_table['false_alarm_trial_count'].values,
                                             familiar_table['catch_trial_count'].values)
    print("\nNovel TPR")
    empirical_novel_TPR = empirical_bayes(novel_table['hit_trial_count'].values,
                                          novel_table['go_trial_count'].values)
    print("\n Novel FPR")
    empirical_novel_FPR = empirical_bayes(novel_table['false_alarm_trial_count'].values,
                                          novel_table['catch_trial_count'].values)

    familiar_table['bayes_TPR'] = empirical_familiar_TPR
    familiar_table['bayes_FPR'] = empirical_familiar_FPR
    novel_table['bayes_TPR'] = empirical_novel_TPR
    novel_table['bayes_FPR'] = empirical_novel_FPR

    # save results to CSV
    if save_results_to_csv:
        familiar_table.to_csv(Path(familiar_destination))
        novel_table.to_csv(Path(novel_destination))

    print("done!")


