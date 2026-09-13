"""
Unified End-to-End Classification Pipeline.
Orchestrates data splitting, scaling, multi-model training, evaluation, and serialization.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import joblib

from src.preprocessing.preprocessor import split_data, scale_features
from src.modeling.models import get_classification_models, train_all_models, REQUIRES_SCALING
from src.evaluation.evaluator import (
    evaluate_predictions,
    explain_confusion_matrix,
    generate_comparison_table,
    plot_model_comparison,
    plot_decision_tree_structure
)


class ClassificationPipeline:
    """
    Automated classification pipeline encapsulating the entire supervised workflow.
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = None
        self.feature_names: List[str] = []
        self.models: Dict[str, Any] = {}
        self.predictions: Dict[str, np.ndarray] = {}
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self.comparison_table: Optional[pd.DataFrame] = None
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ClassificationPipeline":
        """Fits preprocessor, scaler, and all 4 classification models."""
        self.feature_names = list(X.columns)
        
        # 1. Train / Test Split
        self.X_train, self.X_test, self.y_train, self.y_test = split_data(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # 2. Standardization
        self.X_train_scaled, self.X_test_scaled, self.scaler = scale_features(
            self.X_train, self.X_test
        )
        
        # 3. Model Training
        initial_models = get_classification_models(random_state=self.random_state)
        self.models, self.predictions = train_all_models(
            models=initial_models,
            X_train=self.X_train,
            X_test=self.X_test,
            X_train_scaled=self.X_train_scaled,
            X_test_scaled=self.X_test_scaled,
            y_train=self.y_train
        )
        
        # 4. Metric Computation
        for name, pred in self.predictions.items():
            self.metrics[name] = evaluate_predictions(self.y_test.to_numpy(), pred)
            
        self.comparison_table = generate_comparison_table(
            self.y_test.to_numpy(), self.predictions
        )
        
        return self

    def evaluate(self) -> pd.DataFrame:
        """Returns the comparative performance summary table."""
        if self.comparison_table is None:
            raise RuntimeError("Pipeline must be fitted before calling evaluate().")
        return self.comparison_table

    def get_model_report(self, model_name: str) -> str:
        """Returns the full classification report text for a given model."""
        if model_name not in self.metrics:
            raise KeyError(f"Model '{model_name}' not found. Available: {list(self.metrics.keys())}")
        return self.metrics[model_name]['classification_report']

    def get_confusion_matrix(self, model_name: str) -> Dict[str, int]:
        """Returns TP, TN, FP, FN breakdown for a given model."""
        if model_name not in self.metrics:
            raise KeyError(f"Model '{model_name}' not found.")
        return explain_confusion_matrix(self.metrics[model_name]['confusion_matrix'])

    def predict(self, model_name: str, X_new: pd.DataFrame) -> np.ndarray:
        """Generates predictions for new unseen data using the specified model."""
        if model_name not in self.models:
            raise KeyError(f"Model '{model_name}' not fitted.")
            
        model = self.models[model_name]
        needs_scaling = REQUIRES_SCALING.get(model_name, False)
        
        if needs_scaling:
            X_trans = self.scaler.transform(X_new)
            return model.predict(X_trans)
        else:
            return model.predict(X_new)

    def export(self, filepath: str) -> None:
        """Serializes the entire fitted pipeline using joblib."""
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "ClassificationPipeline":
        """Loads a serialized pipeline."""
        return joblib.load(filepath)