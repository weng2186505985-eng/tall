from sklearn.metrics import roc_auc_score

def calculate_eer(y_true, y_scores):
    """Calculate Equal Error Rate (EER)."""
    pass

def calculate_auc(y_true, y_scores):
    return roc_auc_score(y_true, y_scores)
