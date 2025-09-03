# go_odif/metrics.py
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

def pr_auc(y_true, scores):
    if y_true is None:
        return None
    try:
        return float(average_precision_score(y_true, scores))
    except Exception:
        return None

def roc_auc(y_true, scores):
    if y_true is None:
        return None
    try:
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return None
