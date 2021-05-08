import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

def plot_history(history, metrics=None, valid=False, figsize=(6,4), **kwargs):
    if metrics is None:
        metrics = []
    metrics += ['loss']
    df = pd.DataFrame()
    for metric in metrics:
        df[metric] = history.history[metric]

    df['epoch'] = df.index
    hue = None
    if valid:
        valid_df = pd.DataFrame()
        for metric in metrics:
            valid_df[metric] = history.history['val_' + metric]
        valid_df['epoch'] = valid_df.index
        valid_df['set'] = 'valid'
        df['set'] = 'train'
        
        df = df.append(valid_df)
        hue = 'set'
    
    plot_shape = (len(metrics), 1) 

    fig, axes = plt.subplots(*plot_shape, figsize=figsize, squeeze=False)
    for i, metric in enumerate(metrics):
        legend = (i==len(metrics)-1)
        subfig = sns.lineplot(x='epoch', y=metric, data=df, hue=hue, ax=axes[i][0], legend=legend, **kwargs)
    subfig.get_legend().set_title(None)
    return fig