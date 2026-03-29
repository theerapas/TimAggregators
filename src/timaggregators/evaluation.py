import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
)

def compute_metrics(y_true, y_pred, y_prob):
    result = {
        "MCC": matthews_corrcoef(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
    }

    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        result["AUROC"] = np.nan
        result["AUPRC"] = (
            np.nan if np.sum(y_true) == 0 else average_precision_score(y_true, y_prob)
        )
    else:
        result["AUROC"] = roc_auc_score(y_true, y_prob)
        result["AUPRC"] = average_precision_score(y_true, y_prob)

    return result

def get_probabilities(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        scores = np.asarray(scores, dtype=float)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
        return scores

    preds = model.predict(X)
    return np.asarray(preds, dtype=float)
