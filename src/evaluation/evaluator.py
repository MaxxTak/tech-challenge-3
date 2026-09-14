"""
Evaluation Metrics and Diagnostic Reporting Module.
Computes comprehensive statistical metrics across fitted Scikit-Learn Pipelines.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Computes standard and probabilistic classification metrics.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'classification_report': classification_report(y_true, y_pred, zero_division=0)
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        except Exception:
            metrics['roc_auc'] = np.nan
    else:
        metrics['roc_auc'] = np.nan

    return metrics


def explain_confusion_matrix(cm: np.ndarray) -> Dict[str, int]:
    """Deconstructs binary confusion matrix into TN, FP, FN, TP."""
    if cm.shape != (2, 2):
        raise ValueError("Confusion matrix must be 2x2 binary.")
    tn, fp, fn, tp = cm.ravel()
    return {
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn),
        'TP': int(tp)
    }


def evaluate_fitted_pipelines(
    fitted_pipelines: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Evaluates a dictionary of fitted Scikit-Learn Pipelines on test data.
    Returns summary DataFrame and detailed metrics per model.
    """
    records = []
    detailed_metrics = {}

    y_test_arr = y_test.to_numpy()

    for name, pipe in fitted_pipelines.items():
        y_pred = pipe.predict(X_test)
        y_prob = None
        if hasattr(pipe, "predict_proba"):
            try:
                y_prob = pipe.predict_proba(X_test)[:, 1]
            except Exception:
                y_prob = None

        metrics = evaluate_predictions(y_test_arr, y_pred, y_prob=y_prob)
        detailed_metrics[name] = metrics

        records.append({
            'Modelo': name,
            'Acurácia': round(metrics['accuracy'], 4),
            'Precisão': round(metrics['precision'], 4),
            'Recall': round(metrics['recall'], 4),
            'F1-Score': round(metrics['f1_score'], 4),
            'ROC-AUC': round(metrics['roc_auc'], 4) if not np.isnan(metrics['roc_auc']) else 'N/A'
        })

    summary_df = pd.DataFrame(records).sort_values(by='F1-Score', ascending=False).reset_index(drop=True)
    return summary_df, detailed_metrics


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_regression_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_features: Optional[int] = None
) -> Dict[str, Any]:
    """
    Computes standard regression metrics: MSE, RMSE, MAE, R2, and Adjusted R2.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }
    
    if num_features is not None and len(y_true) > num_features + 1:
        n = len(y_true)
        p = num_features
        adj_r2 = 1.0 - ((1.0 - r2) * (n - 1) / (n - p - 1))
        metrics['R2_Ajustado'] = adj_r2
        
    return metrics


def evaluate_regression_pipelines(
    fitted_pipelines: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Evaluates a dictionary of fitted Scikit-Learn Regression Pipelines on test data.
    Returns summary DataFrame and detailed metrics per model.
    """
    records = []
    detailed_metrics = {}

    y_test_arr = y_test.to_numpy()
    num_features = X_test.shape[1]

    for name, pipe in fitted_pipelines.items():
        y_pred = pipe.predict(X_test)
        
        metrics = evaluate_regression_predictions(y_test_arr, y_pred, num_features=num_features)
        detailed_metrics[name] = metrics

        records.append({
            'Modelo': name,
            'R2': round(metrics['R2'], 4),
            'RMSE': round(metrics['RMSE'], 4),
            'MAE': round(metrics['MAE'], 4),
            'MSE': round(metrics['MSE'], 4)
        })

    summary_df = pd.DataFrame(records).sort_values(by='R2', ascending=False).reset_index(drop=True)
    return summary_df, detailed_metrics