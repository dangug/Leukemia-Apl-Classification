from sklearn.model_selection import ShuffleSplit, learning_curve
import numpy as np
import pandas as pd
import seaborn as sns

def plot_learning_curves(
    model, X, y,
    n_repeats=1,
    train_sizes=np.linspace(0.1, 1.0, 5),
    cv=None, scoring=None, title=None,
    **kwargs
    ):

    if cv is None and n_repeats > 1:
        cv = ShuffleSplit(n_splits=n_repeats)

    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, train_sizes=train_sizes, cv=cv, scoring=scoring
    )
    lc_df = pd.DataFrame(columns=['size', 'score', 'traintest'])
    for train_size, train_score, test_score in zip(train_sizes, train_scores, test_scores):
        for train, test in zip(train_score, test_score):
            lc_df = lc_df.append(
                { 'size': train_size, 'score': train, 'traintest':'train'},
                ignore_index=True
            )
            lc_df = lc_df.append(
                {'size': train_size, 'score': test, 'traintest':'test'},
                ignore_index=True
            )
    fig = sns.lineplot(x='size', y='score', data=lc_df, hue='traintest', **kwargs)
    fig.get_legend().set_title(None)
    if title is None:
        title = "Learning Curves"
    fig.set_title(title)
    return fig