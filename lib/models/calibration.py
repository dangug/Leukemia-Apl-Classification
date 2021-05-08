"""
calibration tools
Example:
calibrated = calibrate(model, x_train, y_train)
plot_reliability_diagram(model, x_valid, y_valid, title='uncalibrated')
plt.show()
plot_reliability_diagram(calibrated, x_valid, y_valid, title='calibrated')
plt.show()
print(f"score: {calibration_score(y_valid, model.predict_proba(x_valid)[:,1])}")
print(f"calibrated score: {calibration_score(y_valid, calibrated.predict_proba(x_valid)[:,1])}")
"""
from sklearn.calibration import calibration_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import brier_score_loss
import seaborn as sns
import pandas as pd

def plot_reliability_diagram(trained_model, x_valid, y_valid, title=None, pos_label=1, n_bins=10, strategy='quantile', **kwargs):
    y_proba = trained_model.predict_proba(x_valid)[:, pos_label]
    fop, mpv = calibration_curve(y_valid, y_proba, n_bins=n_bins, strategy=strategy)

    rd_df = pd.DataFrame({'Fraction of Positives':fop, 'Mean Predicted Probability':mpv})
    fig = sns.lineplot(x='Mean Predicted Probability', y='Fraction of Positives',  data=rd_df)
    fig.plot([0,1], [0,1], ':k')
    if title is None:
        fig.set_title('Reliability Diagram')
    else:
        fig.set_title('Reliability Diagram: ' + title)
    return rd_df
    
def calibration_score(y_true, y_proba, pos_label=None):
    """
    Return Brier Score.
    A lower score means a better calibration
    """
    return brier_score_loss(y_true, y_proba, pos_label)


def calibrate(model, x_train, y_train, method='sigmoid', cv=10):
    """
    Calibrate a model using cross validation.
    Use cv='prefit' to disable cross validation. In this case the data used for calibration
    must be different from train and validation data.
    """
    if isinstance(model, Pipeline):
        calibrated_model = clone(model)
        name, last_step = calibrated_model.steps.pop(-1)
        calibrated_step = CalibratedClassifierCV(last_step, method=method, cv=cv)
        calibrated_model.steps.append((name, calibrated_step))
    else:
        calibrated_model = CalibratedClassifierCV(model, method=method, cv=cv)
    calibrated_model.fit(x_train, y_train)
    return calibrated_model