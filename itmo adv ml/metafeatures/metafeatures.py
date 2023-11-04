import pandas as pd
import numpy as np
from math import e
from scipy.stats.mstats import kurtosis


def basic_metafeatures(xs: pd.DataFrame, ys: pd.Series):
    res = {}
    res['row_count'] = xs.shape[0]
    res['col_count'] = xs.shape[1]
    res['class_count'] = len(ys.unique())
    res['categorial_count'] = xs.select_dtypes([object]).columns.shape[0]
    res['numerical_count'] = xs.select_dtypes([int, float]).columns.shape[0]
    return res


def statistics_metafeatures(xs: pd.DataFrame, ys: pd.Series):
    res = {}
# -----> some numerical statistics
    numerical = xs[xs.select_dtypes([int, float]).columns]
    res['min'] = numerical.apply(pd.Series.min).min()
    res['max'] = numerical.apply(pd.Series.max).max()
    res['mean'] = numerical.apply(pd.Series.mean).mean()
# -----> some categorial statistics
    categorial = xs[xs.select_dtypes([object]).columns]
    categorial_counts = categorial.apply(lambda s: len(s.unique()))
    res['categorial_counts_min'] = categorial_counts.min()
    res['categorial_counts_max'] = categorial_counts.max()
    res['categorial_counts_mean'] = categorial_counts.mean()

    def entropy(col):
        vc = col.value_counts(normalize=True, sort=False)
        return -(vc * np.log(vc)/np.log(e)).sum()
    categorial_entropy = categorial.apply(entropy)
    res['categorial_entropy_min'] = categorial_entropy.min()
    res['categorial_entropy_max'] = categorial_entropy.max()
    res['categorial_entropy_mean'] = categorial_entropy.mean()
    return {k: round(v, 3) for k, v in res.items()}


def structural_metafeatures(df, model):
    res = {}
    xs = df.iloc[:, :-1]
    ys = df.iloc[:, -1]
    model.fit(xs, ys)
    res['structural_min'] = model.coef_.min()
    res['structural_max'] = model.coef_.max()
    res['structural_mean'] = model.coef_.mean()
    # res['structural_kurtosis'] = kurtosis(model.coef_)
    return {k: round(v, 3) for k, v in res.items()}
