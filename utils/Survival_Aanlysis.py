import numpy as np
import torch
import pandas as pd

class SurvivalAnalysis(object):
    """ This class contains methods used in survival analysis.
    """

    def c_index(self, risk, T, C):
        """Calculate concordance index to evaluate model prediction.

        C-index calulates the fraction of all pairs of subjects whose predicted
        survival times are correctly ordered among all subjects that can actually
		be ordered, i.e. both of them are uncensored or the uncensored time of
		one is smaller than the censored survival time of the other.

        Parameters
        ----------
        risk: numpy.ndarray
           m sized array of predicted risk (do not confuse with predicted survival time)
        T: numpy.ndarray
           m sized vector of time of death or last follow up
        C: numpy.ndarray
           m sized vector of censored status (do not confuse with observed status)

        Returns
        -------
        A value between 0 and 1 indicating concordance index.
        """
        n_orderable = 0.0
        score = 0.0
        for i in range(len(T)):
            for j in range(i + 1, len(T)):
                if (C[i] == 0 and C[j] == 0):
                    n_orderable = n_orderable + 1
                    if (T[i] > T[j]):
                        if (risk[j] > risk[i]):
                            score = score + 1
                    elif (T[j] > T[i]):
                        if (risk[i] > risk[j]):
                            score = score + 1
                    else:
                        if (risk[i] == risk[j]):
                            score = score + 1
                elif (C[i] == 1 and C[j] == 0):
                    if (T[i] >= T[j]):
                        n_orderable = n_orderable + 1
                        if (T[i] > T[j]):
                            if (risk[j] > risk[i]):
                                score = score + 1
                elif (C[j] == 1 and C[i] == 0):
                    if (T[j] >= T[i]):
                        n_orderable = n_orderable + 1
                        if (T[j] > T[i]):
                            if (risk[i] > risk[j]):
                                score = score + 1

        # print score to screen
        return score / n_orderable

    def calc_at_risk(self, X, T, O, img_names=[]):
        """Calculate the at risk group of all patients.

		For every patient i, this function returns the index of the first
		patient who died after i, after sorting the patients w.r.t. time of death.
        Refer to the definition of Cox proportional hazards log likelihood for
		details: https://goo.gl/k4TsEM

        Parameters
        ----------
        X: numpy.ndarray
           m*n matrix of input data
        T: numpy.ndarray
           m sized vector of time of death
        O: numpy.ndarray
           m sized vector of observed status (1 - censoring status)

        Returns
        -------
        X: numpy.ndarray
           m*n matrix of input data sorted w.r.t time of death
        T: numpy.ndarray
           m sized sorted vector of time of death
        O: numpy.ndarray
           m sized vector of observed status sorted w.r.t time of death
        at_risk: numpy.ndarray
           m sized vector of starting index of risk groups
        """

        df1 = pd.DataFrame({'T': T, 'O': O})
        df1.sort_values(['T', 'O'], ascending=[False, True], inplace=True)
        sort_idx = list(df1.index)
        X = X[sort_idx]
        O = O[sort_idx]
        T = T[sort_idx]
        if len(img_names) != 0:
            img_names = img_names[sort_idx]
        failures = {}
        at_risk = {}
        n, cnt = 0, 0

        for i in range(len(O)):
            if O[i]:
                if T[i] not in failures:
                    failures[T[i].item()] = [i]
                    n += 1
                else:
                    # ties occured
                    cnt += 1
                    failures[T[i].item()].append(i)

                if T[i] not in at_risk:
                    at_risk[T[i].item()] = []
                    for j in range(0, i + 1):
                        at_risk[T[i].item()].append(j)
                else:
                    at_risk[T[i].item()].append(i)
        # when ties occured frequently
        if cnt >= n / 2:
            ties = 'efron'
        elif cnt > 0:
            ties = 'breslow'
        else:
            ties = 'noties'


        return X, T, O, at_risk, failures, ties, img_names

