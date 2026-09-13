"""
Classification Models Factory & Training Suite.
Implements the 4 foundational classifiers studied in IAS_Classificacao.ipynb:
1. Logistic Regression (Probabilistic Sigmoid estimator)
2. Decision Tree Classifier (Hierarchical rule partitioning)
3. Support Vector Machine (SVC with RBF kernel and maximum margin separation)
4. Gaussian Naive Bayes (Conditional probability with feature independence)
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


# Architectural lookup defining whether feature scaling is required
REQUIRES_SCALING: Dict[str, bool] = {
    'Logistic Regression': True,
    'Decision Tree': False,
    'SVM': True,
    'Naive Bayes': False
}


def get_classification_models(random_state: int = 42) -> Dict[str, ClassifierMixin]:
    """
    Returns an instantiated dictionary of the 4 classification algorithms
    with optimal default hyperparameter settings from IAS_Classificacao.ipynb.
    """
    return {
        'Logistic Regression': LogisticRegression(
            random_state=random_state,
            max_iter=1000
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=4,
            random_state=random_state
        ),
        'SVM': SVC(
            kernel='rbf',
            probability=True,
            random_state=random_state
        ),
        'Naive Bayes': GaussianNB()
    }


def train_all_models(
    models: Dict[str, ClassifierMixin],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    X_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    y_train: pd.Series
) -> Tuple[Dict[str, ClassifierMixin], Dict[str, np.ndarray]]:
    """
    Trains all models using the appropriate feature matrix (scaled vs. raw)
    and returns dictionaries of trained model instances and test set predictions.
    """
    trained_models = {}
    predictions = {}
    
    for name, model in models.items():
        needs_scaling = REQUIRES_SCALING.get(name, False)
        
        # Route scaled data to distance/gradient models, raw data to trees & Bayes
        train_data = X_train_scaled if needs_scaling else X_train
        test_data = X_test_scaled if needs_scaling else X_test
        
        # Fit model
        model.fit(train_data, y_train)
        
        # Generate predictions
        pred = model.predict(test_data)
        
        trained_models[name] = model
        predictions[name] = pred
        
    return trained_models, predictions