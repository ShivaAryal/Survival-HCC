import keras
import numpy as np
import pandas as pd
from scipy import stats
from scipy import spatial
from scipy import interp
from scipy import cluster
from collections import Counter
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import StandardScalar
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict

def select_clinical_factors(
    z,
    survival,
    duration_column="duration",
    observed_column="observed",
    alpha=0.05,
    cox_penalizer=0,
):
    #Select latent factors which are predictive of survival.
    cox_coefficients = _cph_coefs(
        z, survival, duration_column, observed_column, penalizer=cox_penalizer
    )
    signif_cox_coefs = cox_coefficients.T[cox_coefficients.T.p < alpha]
    return z.loc[:, signif_cox_coefs.index]

def _cph_coefs(z, survival, duration_column, observed_column, penalizer=0):
    #Compute one CPH model for each latent factor in z
    import lifelines
    return pd.concat(
        [
            lifelines.CoxPHFitter(penalizer=penalizer)
            .fit(
                survival.assign(LF=z.loc[:, i]).dropna(),
                duration_column,
                observed_column,
            )
            .summary.loc["LF"]
            .rename(i)
            for i in z.columns
        ],
        axis=1,
    )

def compute_harrells_c(
    z,
    survival,
    duration_column="duration",
    observed_column="observed",
    cox_penalties=None,
    cv_folds=5,
):
    #Compute's Harrell's c-Index 
    if cox_penalties is None:
        cox_penalties = [0.1, 1, 10, 100, 1000, 10000]
    cvcs = [
        _cv_coxph_c(z, survival, p, duration_column, observed_column, cv_folds)
        for p in cox_penalties
    ]
    return cvcs[np.argmax([np.median(e) for e in cvcs])]


def _cv_coxph_c(
    z,
    survival,
    penalty,
    duration_column="duration",
    observed_column="observed",
    cv_folds=5,
):
    import lifelines
    import lifelines.utils

    cph = lifelines.CoxPHFitter(penalizer=penalty)
    survdf = pd.concat([survival, z], axis=1, sort=False).dropna()

    kfold = KFold(cv_folds)
    scores = list()

    for train_index, test_index in kfold.split(survdf):
        x_train, x_test = survdf.iloc[train_index], survdf.iloc[test_index]

        cph.fit(x_train, duration_column, observed_column)
        cindex = lifelines.utils.concordance_index(
            x_test[duration_column],
            -cph.predict_partial_hazard(x_test),
            x_test[observed_column],
        )
        scores.append(cindex)

    return scores
