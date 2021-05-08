"""
This module contains function to plot smooth ROC curves using KFold
Examples:
    result, aucs = roc_curve_cv(xgb.XGBClassifier(), X, y, n_splits=6)
    plot_roc_curve_cv(result)
    plt.show()
    plot_specificity_cv(result)
    plt.show()
    plot_specificity_cv(result, invert_x=True, invert_y=True)
    plt.show()
    print(f"AUC: {np.mean(aucs)} (std:{np.std(aucs)})")

    Comparing models:

    result_xgb, aucs = roc_curve_cv(xgb.XGBClassifier(), X, y, n_splits=6, n_repeats=4)
    result_rf, aucs = roc_curve_cv(RandomForestClassifier(), X, y, n_splits=6, n_repeats=4)

    plot_specificity_cv({'XGB': result_xgb, 'RF':result_rf})
    plt.show()

    Comparing hyperparameters

    results = []
    for max_depth in (3,10):
        for max_features in (0.5, 0.9):
            result, _ = roc_curve_cv(
                RandomForestClassifier(max_depth=max_depth, max_features=max_features),
                x_full, y_full, n_repeats=4,
                properties={'max features':max_features, 'max depth':max_depth})
            results.append(result)

    plot_specificity_cv(results, hue='max features', style='max depth', ci=False)
    plt.show()
    plot_roc_curve_cv(results, hue='max features', style='max depth', ci=False)
    plt.show()

"""
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from numpy import interp
import numpy as np
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import label_binarize


def roc_curve_simple(model, X, y):
    y_pred = model.predict_proba(X)[:,1]
    fpr, tpr, thres = roc_curve(y, y_pred)
    result_df = pd.DataFrame({'fpr':fpr, 'tpr':tpr, 'threshold':thres}, index=range(len(fpr)))

    return result_df, auc(fpr,tpr)

def roc_curve_cv(model, X, y, n_splits=5, n_repeats=1, properties=None):
    if n_repeats > 1:
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)
    else:
        cv = StratifiedKFold(n_splits=n_splits)
    auc_list = []
    
    result_df = pd.DataFrame()
    for i, (train, test) in enumerate(cv.split(X, y)):
        x_train, x_test = X.iloc[train], X.iloc[test]
        y_train, y_test = y.iloc[train], y.iloc[test]
        
        model.fit(x_train, y_train)
        y_test_pred = model.predict_proba(x_test)[:,1]
        
        fpr, tpr, thres = roc_curve(y_test, y_test_pred)
        # x_label = "False Positive Rate"
        # y_label = "True Positive Rate"
        df = pd.DataFrame({'run':i, 'fpr':fpr, 'tpr':tpr, 'threshold':thres}, index=range(len(fpr)))
        result_df = pd.concat([result_df, df])
       
        auc_list.append(auc(fpr,tpr))

    if properties is not None:
        for key, value, in properties.items():
            result_df[key] = value
        
    return result_df, auc_list

def plot_roc_curve_cv(result, n_step=100, title=None, **kwargs):
    """
    plot the ROC curve with a confidence interval
    """
    fpr_linspace = np.linspace(0,1,n_step)
    tpr_df = pd.DataFrame()

    x_label = "False Positive Rate"
    y_label = "True Positive Rate"

    if isinstance(result, dict):
        for key, value in result.items():
            value['model'] = key

        result = pd.concat(result.values())
        kwargs['hue'] = 'model'
    elif isinstance(result, list):
        result = pd.concat(result)

    result = result.rename(columns={'tpr':y_label, 'fpr':x_label})
    
    group_cols = list(set(result.columns)-{x_label, y_label,'threshold'})
    for name, group in result.groupby(group_cols):
        df = pd.DataFrame(columns=[y_label, x_label]+group_cols)
        df[y_label] = interp(fpr_linspace, group[x_label], group[y_label])
        df[x_label] = fpr_linspace
        df[group_cols] = name
        tpr_df = pd.concat([tpr_df,df])

    fig = plt.axes()
    sns.lineplot(x=x_label, y =y_label, data=tpr_df, **kwargs)
    if title is None:
        title = "Roc curve cv"
    fig.set_title(title)
    return fig
    
    
def plot_specificity_cv(result, n_step=100, invert_x=False, invert_y=False, title=None, **kwargs):
    """
    plot the curve of the specificity as a function of the sensibility
    """
    tpr_linspace = np.linspace(0,1,n_step)
    fpr_df = pd.DataFrame()

    if isinstance(result, dict):
        for key, value in result.items():
            value['model'] = key

        result = pd.concat(result.values())
        kwargs['hue'] = 'model'
    elif isinstance(result, list):
        result = pd.concat(result)

    group_cols = list(set(result.columns)-{'fpr','tpr','threshold'})
    for name, group in result.groupby(group_cols):
        df = pd.DataFrame(columns=['tpr', 'fpr']+group_cols)
        df['fpr'] = interp(tpr_linspace, group['tpr'], group['fpr'])[:-1]
        df['tpr'] = tpr_linspace[:-1]
        df[group_cols]=name
        fpr_df = pd.concat([fpr_df,df])

    if invert_x:
        x_label = 'False Negative Rate'
        fpr_df[x_label] = 1-fpr_df['tpr']
    else:
        x_label = 'Sensitivity'
        fpr_df[x_label] = fpr_df['tpr']
    if invert_y:
        y_label = 'False Positive Rate'
        fpr_df[y_label] = fpr_df['fpr']
    else:
        y_label = 'Specificity'
        fpr_df[y_label] = 1-fpr_df['fpr']

    fig = plt.axes()
    sns.lineplot(x=x_label, y =y_label, data=fpr_df)
    if title is None:
        title = "Specificity vs Sensitivity"
    fig.set_title(title)
    return fig

def plot_roc_threshold_cv(result, n_step=101, title=None, tpr=True, fpr=True, tnr=False, fnr=False, **kwargs):
    """
    plot the ROC curve with a confidence interval
    """
    fpr_linspace = np.linspace(0,1,n_step)
    tpr_df = pd.DataFrame()

    if isinstance(result, dict):
        for key, value in result.items():
            value['model'] = key

        result = pd.concat(result.values())
        kwargs['hue'] = 'model'
    elif isinstance(result, list):
        result = pd.concat(result)
    
    threshold_dfs = []
    
    group_cols = list(set(result.columns)-{'fpr','tpr','threshold'})
    for name, group in result.groupby(group_cols):
        group = group.sort_values(by='threshold')
        if fpr:
            df = pd.DataFrame(columns=['rate', 'metric','threshold']+group_cols)
            df['rate'] = interp(fpr_linspace, group['threshold'], group['fpr'])
            df['threshold'] = fpr_linspace
            df['metric'] = 'FPR'
            df[group_cols] = name
            threshold_dfs.append(df)
        if tpr:
            df = pd.DataFrame(columns=['rate', 'metric','threshold']+group_cols)
            df['rate'] = interp(fpr_linspace, group['threshold'], group['tpr'])
            df['threshold'] = fpr_linspace
            df['metric'] = 'TPR'
            df[group_cols] = name
            threshold_dfs.append(df)
        if tnr:
            df = pd.DataFrame(columns=['rate', 'metric','threshold']+group_cols)
            df['rate'] = 1- interp(fpr_linspace, group['threshold'], group['fpr'])
            df['threshold'] = fpr_linspace
            df['metric'] = 'TNR'
            df[group_cols] = name
            threshold_dfs.append(df)
        if fnr:
            df = pd.DataFrame(columns=['rate', 'metric','threshold']+group_cols)
            df['rate'] = 1- interp(fpr_linspace, group['threshold'], group['tpr'])
            df['threshold'] = fpr_linspace
            df['metric'] = 'FNR'
            df[group_cols] = name
            threshold_dfs.append(df)
            
    threshold_df = pd.concat(threshold_dfs)
            
    if 'hue' in kwargs.keys():
        kwargs['style'] = 'metric'
    else:
        kwargs['hue'] = 'metric'


    fig = plt.axes()
    sns.lineplot(x='threshold', y='rate', data=threshold_df, **kwargs)
    
    if title is None:
        title = "Threshold curve cv"
    fig.set_title(title)
    return fig

def plot_roc_threshold(result, n_step=101, title=None, tpr=True, fpr=True, tnr=False, fnr=False, **kwargs):


    """
    plot the ROC curve with a confidence interval
    """
    fpr_linspace = np.linspace(0,1,n_step)
    tpr_df = pd.DataFrame()

    if isinstance(result, dict):
        for key, value in result.items():
            value['model'] = key

        result = pd.concat(result.values())
        kwargs['hue'] = 'model'
    elif isinstance(result, list):
        result = pd.concat(result)
    
    threshold_dfs = []
    
    result = result.sort_values(by='threshold')
    if fpr:
        df = pd.DataFrame(columns=['rate', 'metric','threshold'])
        df['threshold'] = result['threshold']
        df['rate'] = result['fpr']
        df['metric'] = 'FPR'
        threshold_dfs.append(df)
    if tpr:
        df = pd.DataFrame(columns=['rate', 'metric','threshold'])
        df['threshold'] = result['threshold']
        df['rate'] = result['tpr']
        df['metric'] = 'TPR'
        threshold_dfs.append(df)
    if tnr:
        df = pd.DataFrame(columns=['rate', 'metric','threshold'])
        df['threshold'] = result['threshold']
        df['rate'] = 1-result['fpr']
        df['metric'] = 'TNR'
        threshold_dfs.append(df)
    if fnr:
        df = pd.DataFrame(columns=['rate', 'metric','threshold'])
        df['threshold'] = result['threshold']
        df['rate'] = 1-result['tpr']
        df['metric'] = 'FNR'
        threshold_dfs.append(df)
            
    threshold_df = pd.concat(threshold_dfs)
            
    if 'hue' in kwargs.keys():
        kwargs['style'] = 'metric'
    else:
        kwargs['hue'] = 'metric'


    fig = plt.axes()
    sns.lineplot(x='threshold', y='rate', data=threshold_df, **kwargs)
    fig.set(xlim=(0,1),ylim=(0,1))
    
    if title is None:
        title = "Error rate/Threshold"
    fig.set_title(title)
    return fig


def roc_master(y_test,y_score, classes=0):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y = label_binarize(y_score, classes=[0, 1])
    n_classes = y.shape[1]
    print(y)
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    lw = 2
    plt.plot(fpr[classes], tpr[classes], color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[classes])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()