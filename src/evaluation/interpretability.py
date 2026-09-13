"""
Model Interpretability & Explainability Module.
Computes Global Feature Importance and SHAP (SHapley Additive exPlanations) values.
"""

from typing import Tuple, List, Optional, Any
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import shap


def get_feature_names_from_pipeline(pipeline: Pipeline) -> List[str]:
    """
    Extracts the exact post-transformation feature names from the Pipeline's ColumnTransformer.
    """
    preprocessor = pipeline.named_steps['preprocessor']
    return list(preprocessor.get_feature_names_out())


def get_feature_importance_df(pipeline: Pipeline) -> pd.DataFrame:
    """
    Extracts intrinsic feature importance or model coefficients from a fitted Pipeline.
    Supports Tree-based models (feature_importances_) and Linear models (coef_).
    """
    feature_names = get_feature_names_from_pipeline(pipeline)
    classifier = pipeline.named_steps['classifier']

    if hasattr(classifier, 'feature_importances_'):
        importances = classifier.feature_importances_
        metric_name = 'Importância (MDI)'
    elif hasattr(classifier, 'coef_'):
        # Flatten coefficients if 2D
        importances = np.abs(classifier.coef_).ravel()
        metric_name = '|Coeficiente|'
    else:
        # Fallback uniform
        importances = np.zeros(len(feature_names))
        metric_name = 'N/A'

    df_importance = pd.DataFrame({
        'Feature': feature_names,
        'Valor': importances
    }).sort_values(by='Valor', ascending=False).reset_index(drop=True)

    df_importance['Métrica'] = metric_name
    return df_importance


def compute_shap_explanations(
    pipeline: Pipeline,
    X_sample: pd.DataFrame
) -> Tuple[Any, np.ndarray, List[str], np.ndarray]:
    """
    Computes SHAP values for the given fitted Pipeline on input data.

    Returns:
        - explainer: The fitted SHAP explainer instance
        - shap_values: Numpy array of calculated SHAP values
        - feature_names: Transformed column names
        - X_transformed: Transformed feature array
    """
    preprocessor = pipeline.named_steps['preprocessor']
    classifier = pipeline.named_steps['classifier']

    # Preprocess sample using pipeline's ColumnTransformer
    X_trans = preprocessor.transform(X_sample)
    feature_names = list(preprocessor.get_feature_names_out())

    # Choose appropriate explainer based on model family
    if hasattr(classifier, 'feature_importances_'):
        explainer = shap.TreeExplainer(classifier)
        shap_vals = explainer.shap_values(X_trans)
    elif hasattr(classifier, 'coef_'):
        # Masker for linear explainer
        explainer = shap.LinearExplainer(classifier, X_trans)
        shap_vals = explainer.shap_values(X_trans)
    else:
        # General model-agnostic kernel/explainer
        background = shap.sample(X_trans, min(10, len(X_trans)))
        explainer = shap.KernelExplainer(classifier.predict_proba, background)
        shap_vals = explainer.shap_values(X_trans)

    # For binary classification where shap_values is a list [class_0, class_1] or 3D
    if isinstance(shap_vals, list) and len(shap_vals) == 2:
        shap_vals_matrix = shap_vals[1]  # positive class
    elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
        shap_vals_matrix = shap_vals[:, :, 1]
    else:
        shap_vals_matrix = np.array(shap_vals)

    return explainer, shap_vals_matrix, feature_names, X_trans
