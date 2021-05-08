"""
This module contains functions helping in selecting the best threshold
Exemple:
cost, best_threshold, best_cost = optimize_thresholdcv(model, x_train, y_train, FP=20, FN=100, n_repeats=5)
plot_cost_curve(cost, best_threshold, best_cost)
print(f"Best threshold: {best_threshold}\nBest cost: {best_cost}")
"""
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
import numpy as np
import seaborn as sns
import pandas as pd

def calculate_cost(y_true, y_pred, threshold,  TP=0, FP=0, TN=0, FN=0):
    """
    Returns the cost given the truth, the predicted probabilities and the cost matrix
    """
    tp = ((y_pred > threshold) & (y_true > 0.5)).sum()
    tn = ((y_pred <= threshold) & (y_true <= 0.5)).sum()
    fp = ((y_pred > threshold) & (y_true <= 0.5)).sum()
    fn = ((y_pred <= threshold) & (y_true > 0.5)).sum()
    return tp*TP + tn*TN + fp*FP + fn*FN

def get_thresholds(y_true, y_pred):
    """
    Returns a list of in-between thresholds
    """
    df = pd.DataFrame({'y_pred':y_pred, 'y_true':y_true})
    df = df.sort_values(by='y_pred').reset_index()
    thres =  (df['y_pred'][:-1] + df['y_pred'][1:])/2
    thres.iloc[0] = -0.1
    thres.iloc[-1] = 1.1
    return thres.unique()

def optimize_threshold(model, X, y, TP=0, FP=0, TN=0, FN=0, minimize=True, pos_label=1):
    """
    Find the optimal threshold
    Returns a tuple containing:
        A pandas dataframe used for plotting
        The optimal threshold
        The cost associated to that threshold
    """
    y_pred = model.predict_proba(X)[:, pos_label]
    thresholds = get_thresholds(y, y_pred)
    cost = {}
    for thres in thresholds:
        cost[thres] = calculate_cost(y, y_pred, thres, TP, FP, TN, FN)
    cost_df = pd.DataFrame(data={'cost':cost.values(), 'threshold':cost.keys()}, index=cost.keys())
    if minimize:
        best_idx = cost_df['cost'].idxmin(axis=0)
    else:
        best_idx = cost_df['cost'].idxmax(axis=0)
    return cost_df, cost_df.loc[best_idx]['threshold'], cost_df.loc[best_idx]['cost']

def optimize_thresholdcv(model, X, y, TP=0, FP=0, TN=0, FN=0, minimize=True, pos_label=1, n_step=100, n_splits=5, n_repeats=1):
    """
    Find the optimal threshold using cross validation.
    Returns a tuple containing:
        A pandas dataframe used for plotting
        The optimal threshold
        The cost associated to that threshold
    """
    if n_repeats > 1:
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)
    else:
        cv = StratifiedKFold(n_splits=n_splits)
    
    threshold_linspace = np.linspace(0,1,n_step)
    
    cost_df = pd.DataFrame()
    for i, (train, test) in enumerate(cv.split(X, y)):
        x_train, x_test = X.iloc[train], X.iloc[test]
        y_train, y_test = y.iloc[train], y.iloc[test]
        
        model.fit(x_train, y_train)
        
        y_pred = model.predict_proba(x_test)[:, pos_label]
        thresholds = get_thresholds(y_test, y_pred)
        cost = {}
        for thres in thresholds:
            cost[thres] = calculate_cost(y_test, y_pred, thres, TP, FP, TN, FN)
        cost_values = np.interp(threshold_linspace, list(cost.keys()), list(cost.values()))
        df = pd.DataFrame(columns=['run', 'threshold', 'cost'])
        
        df['cost'] = cost_values
        df['threshold'] = threshold_linspace
        df['run'] = i
       
        cost_df = pd.concat([cost_df, df])
    
    mean_cost =  cost_df.set_index(['threshold', 'run']).mean(level='threshold')
    
    if minimize:
        best_idx = mean_cost['cost'].idxmin(axis=0)
    else:
        best_idx = mean_cost['cost'].idxmax(axis=0)
    
    return cost_df.reset_index(drop=True), best_idx, float(mean_cost.loc[best_idx])

def plot_cost_curve(cost_df, best_threshold, best_cost, title=None):
    fig = sns.lineplot(x='threshold', y='cost', data=cost_df);
    fig.axvline(x=best_threshold, c='k', lw=0.5)
    fig.axhline(y=best_cost, c='k', lw=0.5)
    if title is None:
        title = "Threshold optimization"
    fig.set_title(title)
    return fig

