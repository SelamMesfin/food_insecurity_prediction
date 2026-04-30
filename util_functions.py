import numpy as np
import pandas as pd

from sklearn.metrics import ( accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score)

print("Libraries imported successfully.")

def evaluate_predictions(y_true, y_prob, threshold=0.50):
    """Return common classification metrics for a selected threshold."""
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Macro_F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, y_prob),
        "Average_Precision": average_precision_score(y_true, y_prob)
    }


def find_best_threshold(y_true, y_prob, metric="Macro_F1"):
    """Choose the threshold that maximizes the selected metric on the validation set."""
    thresholds = np.round(np.arange(0.10, 0.81, 0.05), 2)
    rows = []

    for threshold in thresholds:
        rows.append(evaluate_predictions(y_true, y_prob, threshold))

    threshold_df = pd.DataFrame(rows)
    best_row = threshold_df.sort_values(metric, ascending=False).iloc[0]

    return float(best_row["Threshold"]), threshold_df.sort_values(metric, ascending=False)


def get_positive_probability(model, X_data):
    """Return probability for class 1. Handles classifiers with or without predict_proba."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_data)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_data)
        return (scores - scores.min()) / (scores.max() - scores.min())
    else:
        return model.predict(X_data)


def assign_risk_group(prob):
    if prob >= 0.70:
        return "High Risk"
    elif prob >= 0.40:
        return "Medium Risk"
    else:
        return "Low Risk"


def recommend_action(risk_group):
    if risk_group == "High Risk":
        return "Prioritize for urgent screening and food-security support"
    elif risk_group == "Medium Risk":
        return "Monitor closely and consider livelihood/agricultural support"
    else:
        return "No immediate intervention; continue routine monitoring"

