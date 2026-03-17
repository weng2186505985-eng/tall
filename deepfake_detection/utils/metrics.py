from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

def calculate_eer(y_true, y_scores):
    """Calculate Equal Error Rate (EER)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer

def calculate_auc(y_true, y_scores):
    return roc_auc_score(y_true, y_scores)
