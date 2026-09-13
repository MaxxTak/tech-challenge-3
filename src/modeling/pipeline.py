"""
Native Scikit-Learn Automated Pipeline Factory & Manager.
Encapsulates ColumnTransformer and Estimator into official Pipeline objects.
"""

from typing import Dict, Any, Optional
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def build_pipeline(preprocessor: ColumnTransformer, classifier: Any) -> Pipeline:
    """
    Creates an end-to-end native Scikit-Learn Pipeline combining
    preprocessing and classification.
    """
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])


class NativeClassificationPipeline:
    """
    Production-grade container managing native Scikit-Learn pipelines,
    fitted models, cross-validation results, and model serialization.
    """

    def __init__(self, preprocessor: ColumnTransformer):
        self.preprocessor = preprocessor
        self.pipelines: Dict[str, Pipeline] = {}
        self.fitted_pipelines: Dict[str, Pipeline] = {}
        self.cv_results: Dict[str, Dict[str, Any]] = {}
        self.best_model_name: Optional[str] = None
        self.best_pipeline: Optional[Pipeline] = None

    def register_classifier(self, name: str, classifier: Any) -> Pipeline:
        """Constructs and registers a native pipeline for the given classifier."""
        pipe = build_pipeline(self.preprocessor, classifier)
        self.pipelines[name] = pipe
        return pipe

    def fit_single(self, name: str, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        """Fits a single registered pipeline on training data."""
        if name not in self.pipelines:
            raise KeyError(f"Pipeline '{name}' not found. Registered: {list(self.pipelines.keys())}")
        pipe = self.pipelines[name]
        pipe.fit(X_train, y_train)
        self.fitted_pipelines[name] = pipe
        return pipe

    def predict(self, name: str, X: pd.DataFrame) -> np.ndarray:
        """Generates predictions using the specified fitted pipeline."""
        if name not in self.fitted_pipelines:
            raise KeyError(f"Pipeline '{name}' not yet fitted.")
        return self.fitted_pipelines[name].predict(X)

    def predict_proba(self, name: str, X: pd.DataFrame) -> np.ndarray:
        """Generates class probabilities using the specified fitted pipeline."""
        if name not in self.fitted_pipelines:
            raise KeyError(f"Pipeline '{name}' not yet fitted.")
        pipe = self.fitted_pipelines[name]
        if hasattr(pipe, "predict_proba"):
            return pipe.predict_proba(X)
        raise AttributeError(f"Estimator '{name}' does not implement predict_proba.")

    def export(self, filepath: str) -> None:
        """Serializes the entire pipeline container to disk via joblib."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "NativeClassificationPipeline":
        """Loads a serialized pipeline container from disk."""
        return joblib.load(filepath)