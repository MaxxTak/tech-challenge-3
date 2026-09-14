"""
Classification Models Factory & Hyperparameter Grids Definition.
Configures base estimators and search spaces for GridSearchCV.
"""

from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


def get_base_classifiers(random_state: int = 42) -> Dict[str, Any]:
    """
    Returns baseline un-tuned classifier instances.
    """
    return {
        'Logistic Regression': LogisticRegression(
            random_state=random_state,
            max_iter=1000
        ),
        'Decision Tree': DecisionTreeClassifier(
            random_state=random_state
        ),
        'SVM': SVC(
            random_state=random_state,
            probability=True
        ),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(
            random_state=random_state
        )
    }


def get_hyperparameter_grids() -> Dict[str, Dict[str, list]]:
    """
    Defines hyperparameter tuning grids prefixed with 'classifier__'
    for seamless integration with Scikit-learn Pipeline and GridSearchCV.
    """
    return {
        'Logistic Regression': {
            'classifier__C': [0.01, 0.1, 1.0, 10.0],
            'classifier__penalty': ['l2'],
            'classifier__solver': ['lbfgs']
        },
        'Decision Tree': {
            'classifier__max_depth': [2, 3, 4, 5, None],
            'classifier__min_samples_split': [2, 4],
            'classifier__min_samples_leaf': [1, 2],
            'classifier__criterion': ['gini', 'entropy']
        },
        'SVM': {
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__kernel': ['rbf', 'linear'],
            'classifier__gamma': ['scale', 'auto']
        },
        'Naive Bayes': {
            'classifier__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
        },
        'Random Forest': {
            'classifier__n_estimators': [50, 100],
            'classifier__max_depth': [3, 5, None],
            'classifier__min_samples_split': [2, 4]
        }
    }


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

def get_base_regressors(random_state: int = 42) -> Dict[str, Any]:
    """
    Returns baseline un-tuned regressor instances (blueprint algorithms).
    """
    return {
        'Decision Tree Regressor': DecisionTreeRegressor(
            random_state=random_state
        ),
        'Random Forest Regressor': RandomForestRegressor(
            random_state=random_state
        ),
        'KNN Regressor': KNeighborsRegressor(),
        'SVR': SVR()
    }


def get_regression_hyperparameter_grids() -> Dict[str, Dict[str, list]]:
    """
    Defines hyperparameter tuning grids prefixed with 'regressor__'
    for seamless integration with Scikit-learn Pipeline and GridSearchCV.
    """
    return {
        'Decision Tree Regressor': {
            'regressor__max_depth': [3, 5, 7, None],
            'regressor__min_samples_split': [2, 5, 10],
            'regressor__min_samples_leaf': [1, 2]
        },
        'Random Forest Regressor': {
            'regressor__n_estimators': [50, 100, 200],
            'regressor__max_depth': [3, 5, 7, None],
            'regressor__min_samples_split': [2, 5, 10]
        },
        'KNN Regressor': {
            'regressor__n_neighbors': [3, 5, 7, 9],
            'regressor__weights': ['uniform', 'distance'],
            'regressor__p': [1, 2]
        },
        'SVR': {
            'regressor__C': [0.1, 1.0, 10.0, 100.0],
            'regressor__kernel': ['rbf', 'linear'],
            'regressor__gamma': ['scale', 'auto'],
            'regressor__epsilon': [0.1, 0.2, 0.5]
        }
    }