#!/usr/bin/env python

"""
    epi metrics regression model
    Created Feb 2023
    Copyright (C) Damien Farrell
"""

import sys,os,random,time,string
import subprocess
import math
import numpy as np
import pandas as pd
import pylab as plt
import geopandas as gpd
import networkx as nx
from . import models, utils

meta = pd.read_csv('../notebooks/meta.csv')
snpdist = pd.read_csv('../notebooks/snpdist.csv',index_col=0)
cent = gpd.read_file('../notebooks/cent.shp.zip')

def flatten_matrix(df):
    """Flatten a symmetrical matrix"""

    #user only upper triangle
    keep = np.triu(np.ones(df.shape)).astype('bool').reshape(df.size)
    S=df.unstack()[keep]
    #print (S)
    S.index = ['{}_{}'.format(i, j) for i, j in S.index]
    S = S[~S.index.duplicated(keep='first')]
    return S

def sampletimedist(meta, key='SeqID', col='Year'):

    def compare(x,y):
        return abs(x-y)

    std = meta.apply(lambda row: compare(row[col], meta[col]), axis=1)
    std.columns = meta[key]
    std.index = meta[key]
    return std

def samplegeodist(cent, key='SeqID'):

    sdist = cent.geometry.apply(lambda x: cent.distance(x))
    sdist.columns = cent[key]
    sdist.index = cent[key]
    sdist = sdist.fillna(-1)
    return sdist

def samegroup(meta, key='SeqID',herdcol='HERD_NO'):

    def compare(x,y):
        return (x==y)
    sg = meta.apply(lambda row: compare(row[herdcol], meta[herdcol]), axis=1)
    sg.columns = meta[key]
    sg.index = meta[key]
    return sg

def samespecies(meta, key='SeqID',col='Species'):
    return samegroup(meta, key, col)

def size_coeff(meta, key='SeqID',herd_col='HERD_SIZE'):

    def func(x,y):
        #print (y)
        if pd.isnull(x):
            return
        #y=y.fillna(0)
        return x*y.apply(math.sqrt)

    ssizes = meta.apply(lambda row: func(row[herd_col], meta[herd_col]), axis=1)
    ssizes.columns = meta[key]
    ssizes.index = meta[key]
    ssizes = ssizes.fillna(-1)
    return ssizes

def fit_gradientboostingregressor(X,y):

    from sklearn import datasets, ensemble
    params = {
        "n_estimators": 500,
        "max_depth": 6,
        "min_samples_split": 5,
        "learning_rate": 0.01,
        "loss": "squared_error",
    }
    reg = ensemble.GradientBoostingRegressor(**params)
    reg.fit(X, y)
    return reg

def model_report(reg, X, y):

    from sklearn.inspection import permutation_importance
    from sklearn.metrics import mean_squared_error
    feature_importance = reg.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + 0.5
    fig,axs = plt.subplots(1,2,figsize=(18, 6))
    ax=axs[0]
    ax.barh(pos, feature_importance[sorted_idx], align="center")
    ax.set_yticks(pos, np.array(X.columns)[sorted_idx])
    ax.set_title("Feature Importance (MDI)")

    result = permutation_importance(
        reg, X, y, n_repeats=10, random_state=42, n_jobs=2
    )
    sorted_idx = result.importances_mean.argsort()
    ax=axs[1]
    ax.boxplot(
        result.importances[sorted_idx].T,
        vert=False,
        labels=np.array(X.columns)[sorted_idx],
    )
    ax.set_title("Permutation Importance (test set)")
    fig.tight_layout()
    r=result
    for i in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            print(f"{X.columns[i]:<20}"
                  f"{r.importances_mean[i]:.3f}"
                  f" +/- {r.importances_std[i]:.3f}")
    return

def create_regr_data(M, colnames):
    """Make table of pairwise matrices for fitting"""

    flat=[] #flattened data
    for m in M:
        x = flatten_matrix(m)
        print (len(x))
        #print (x[:10])
        flat.append(x)

    #add random column
    dummy = pd.Series(np.random.normal(1,10,len(x)),index=x.index)
    flat.append(dummy)
    colnames.append('random')
    X=pd.concat(flat,axis=1,join='inner')
    X.columns=colnames
    return X

def filter_regr_data(X,y,cutoff=20):
    # drop distantly related pairs
    y=y[y<=cutoff]
    X = X.loc[y.index]
    return X,y

def get_test_data(meta, snpdist, cent, snp_cutoff=15, subsample=0):
    """get data into arrays for fitting"""

    col='id'

    if subsample > 5:
        meta = meta.sample(subsample)
        idx = list(meta.id)
        snpdist = snpdist.loc[idx,idx]
        cent = cent[cent.id.isin(idx)]

    idx = list(meta.id)
    print ('%s samples' %len(snpdist))
    #get metrics
    y = flatten_matrix(snpdist.loc[idx,idx])
    sg = samegroup(meta, 'id', 'herd')
    ss = samespecies(meta, key='id',col='species')
    std = sampletimedist(meta, 'id', 'inf_start')
    sdist = samplegeodist(cent, 'id')
    #ssizes =  size_coeff(meta, 'id', 'herd_size')

    X = create_regr_data([sg,ss,std,sdist],
                        ['SameGroup','SameSpecies','SampleTimeDist','SampleGeoDist'] )
    y = y.loc[X.index]

    X, y = filter_regr_data(X,y,snp_cutoff)
    print ('filtered samples')
    print(len(X), len(y))
    return X,y

def parameter_test():
    """Parameter space test """

    from sklearn import ensemble
    from sklearn.model_selection import GridSearchCV, KFold

    model = ensemble.GradientBoostingRegressor()
    n_estimators = [50, 200, 500]
    max_depth = [2, 4, 6, 8]

    param_grid = dict(max_depth=max_depth, n_estimators=n_estimators)
    kfold = KFold(n_splits=2, shuffle=True, random_state=0)
    grid_search = GridSearchCV(model, param_grid, scoring="neg_mean_absolute_error", n_jobs=-1, cv=kfold,verbose=1)

    X,y = get_test_data(meta, snpdist, cent, 15)
    grid_result = grid_search.fit(X, y)
    return grid_result

def test(snp_cutoff=15, model='gbr', subsample=0):
    """test with simulated data from ABM model"""

    X,y = get_test_data(meta, snpdist, cent, snp_cutoff, subsample)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=13
    )
    if model == 'gbr':
        reg = fit_gradientboostingregressor(X_train,y_train)
    elif model == 'forest':
        reg = fit_random_forest(X_train,y_train)

    model_report(reg, X_test, y_test)
    return X,y

if __name__ == '__main__':
    test()
