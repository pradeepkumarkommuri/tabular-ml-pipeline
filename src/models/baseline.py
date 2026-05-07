"""
models/baseline.py
------------------
XGBoost baseline for comparison against the TabNet model.
Exposes the same fit / predict_proba / predict interface.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    logger.warning("xgboost not installed — baseline model unavailable. "
                   "Install with: pip install xgboost")


class XGBoostBaseline:
    """
    Thin wrapper around XGBClassifier with sensible defaults for tabular data.

    Args:
        n_estimators: Number of boosting rounds.
        max_depth: Maximum tree depth.
        learning_rate: Step size shrinkage.
        subsample: Subsample ratio of training instances.
        colsample_bytree: Subsample ratio of columns per tree.
        seed: Random seed.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        seed: int = 42,
    ):
        if not _XGB_AVAILABLE:
            raise ImportError("xgboost is required for the baseline model.")

        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        )
        self._feature_names: list[str] = []

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> "XGBoostBaseline":
        if feature_names:
            self._feature_names = feature_names

        logger.info("Training XGBoost baseline …")
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )
        logger.info("XGBoost training complete ✓")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def feature_importance(self) -> dict[str, float]:
        if not self._feature_names:
            return {}
        importances = self.model.feature_importances_
        return dict(sorted(
            zip(self._feature_names, importances),
            key=lambda x: x[1],
            reverse=True,
        ))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Baseline model saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "XGBoostBaseline":
        with open(path, "rb") as f:
            return pickle.load(f)
