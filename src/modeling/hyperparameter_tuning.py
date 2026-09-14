"""
Hyperparameter Tuning & Statistical Validation Module.
Executes GridSearchCV with StratifiedKFold cross-validation across all pipelines.
"""

from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from src.modeling.models import get_base_classifiers, get_hyperparameter_grids
from src.modeling.pipeline import build_pipeline


def tune_all_models(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
    scoring: str = 'f1',
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any], str, Any]:
    """
    Runs systematic GridSearchCV across all registered classification pipelines.
    Uses StratifiedKFold to preserve class balance across validation splits.

    Returns:
        - results_df: Table comparing all models with their cross-validated performance.
        - best_estimators: Dictionary mapping model name to best fitted Pipeline.
        - best_overall_name: Name of the top-performing model.
        - best_overall_pipeline: The top-performing fitted Pipeline.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    classifiers = get_base_classifiers(random_state=random_state)
    param_grids = get_hyperparameter_grids()

    records = []
    best_estimators = {}

    for name, clf in classifiers.items():
        pipe = build_pipeline(preprocessor, clf)
        params = param_grids.get(name, {})

        grid = GridSearchCV(
            estimator=pipe,
            param_grid=params,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            refit=True
        )

        grid.fit(X_train, y_train)

        best_score = grid.best_score_
        best_params = grid.best_params_
        best_estimators[name] = grid.best_estimator_

        # Clean parameter names for display (strip 'classifier__')
        clean_params = {k.replace('classifier__', ''): v for k, v in best_params.items()}

        records.append({
            'Modelo': name,
            f'CV_{scoring.upper()}': round(best_score, 4),
            'Melhores_Parametros': str(clean_params)
        })

    results_df = pd.DataFrame(records).sort_values(by=f'CV_{scoring.upper()}', ascending=False).reset_index(drop=True)
    best_overall_name = results_df.iloc[0]['Modelo']
    best_overall_pipeline = best_estimators[best_overall_name]

    return results_df, best_estimators, best_overall_name, best_overall_pipeline


from sklearn.model_selection import KFold
from src.modeling.models import get_base_regressors, get_regression_hyperparameter_grids

def tune_all_regressors(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
    scoring: str = 'neg_mean_squared_error',
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any], str, Any]:
    """
    Runs systematic GridSearchCV across all registered regression pipelines.
    Uses KFold for continuous targets.
    """
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    regressors = get_base_regressors(random_state=random_state)
    param_grids = get_regression_hyperparameter_grids()

    records = []
    best_estimators = {}

    for name, reg in regressors.items():
        pipe = build_pipeline(preprocessor, reg, estimator_name='regressor')
        params = param_grids.get(name, {})

        grid = GridSearchCV(
            estimator=pipe,
            param_grid=params,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            refit=True
        )

        grid.fit(X_train, y_train)

        # For MSE, scikit-learn returns negative values, so we might want to display positive RMSE
        best_score = grid.best_score_
        display_score = np.sqrt(-best_score) if 'neg_mean_squared_error' in scoring else best_score
        
        best_params = grid.best_params_
        best_estimators[name] = grid.best_estimator_

        clean_params = {k.replace('regressor__', ''): v for k, v in best_params.items()}

        records.append({
            'Modelo': name,
            f'CV_RMSE' if 'neg_mean_squared_error' in scoring else f'CV_{scoring.upper()}': round(display_score, 4),
            'Melhores_Parametros': str(clean_params)
        })

    # Sort ascending if RMSE (lower is better), descending otherwise
    asc = True if 'neg_mean_squared_error' in scoring else False
    sort_col = 'CV_RMSE' if 'neg_mean_squared_error' in scoring else f'CV_{scoring.upper()}'
    
    results_df = pd.DataFrame(records).sort_values(by=sort_col, ascending=asc).reset_index(drop=True)
    best_overall_name = results_df.iloc[0]['Modelo']
    best_overall_pipeline = best_estimators[best_overall_name]

    return results_df, best_estimators, best_overall_name, best_overall_pipeline
