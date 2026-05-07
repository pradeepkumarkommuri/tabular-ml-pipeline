"""
pipeline/preprocess.py
----------------------
Feature engineering, encoding, scaling, and train/val/test splitting.
Designed to be fit on train data only — no leakage.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


class TabularPreprocessor:
    """
    Stateful preprocessor that fits on training data and transforms all splits.

    Handles:
    - Numerical imputation + StandardScaler
    - Categorical imputation + LabelEncoding
    - Train / val / test splitting (stratified)

    Args:
        target: Name of the target column.
        numerical_features: List of numerical column names.
        categorical_features: List of categorical column names.
        fill_strategy: How to fill numerical NaNs — 'median', 'mean', or 'constant'.
        test_size: Fraction held out for test.
        val_size: Fraction of remaining data held out for validation.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        target: str,
        numerical_features: list[str],
        categorical_features: list[str],
        fill_strategy: str = "median",
        test_size: float = 0.2,
        val_size: float = 0.1,
        seed: int = 42,
    ):
        self.target = target
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.fill_strategy = fill_strategy
        self.test_size = test_size
        self.val_size = val_size
        self.seed = seed

        # Fit artifacts (populated in fit_transform)
        self._num_fill_values: dict[str, float] = {}
        self._cat_fill_values: dict[str, str] = {}
        self._label_encoders: dict[str, LabelEncoder] = {}
        self._scaler = StandardScaler()
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(
        self, df: pd.DataFrame
    ) -> tuple[
        np.ndarray, np.ndarray,  # X_train, y_train
        np.ndarray, np.ndarray,  # X_val,   y_val
        np.ndarray, np.ndarray,  # X_test,  y_test
    ]:
        """Split, fit on train, and transform all splits. Returns numpy arrays."""
        logger.info("Splitting data into train / val / test …")
        train_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            stratify=df[self.target],
            random_state=self.seed,
        )
        adjusted_val = self.val_size / (1 - self.test_size)
        train_df, val_df = train_test_split(
            train_df,
            test_size=adjusted_val,
            stratify=train_df[self.target],
            random_state=self.seed,
        )
        logger.info(
            f"Split sizes — train: {len(train_df):,}  "
            f"val: {len(val_df):,}  test: {len(test_df):,}"
        )

        self._fit(train_df)

        X_train, y_train = self._transform(train_df)
        X_val, y_val = self._transform(val_df)
        X_test, y_test = self._transform(test_df)

        logger.info(f"Feature matrix shape: {X_train.shape}")
        return X_train, y_train, X_val, y_val, X_test, y_test

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Transform new data using fitted artifacts. Returns (X, y or None)."""
        if not self._fitted:
            raise RuntimeError("Call fit_transform before transform.")
        return self._transform(df)

    @property
    def feature_names(self) -> list[str]:
        return self.numerical_features + self.categorical_features

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Preprocessor saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "TabularPreprocessor":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"Preprocessor loaded from {path}")
        return obj

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fit(self, df: pd.DataFrame) -> None:
        """Compute fill values, fit encoders and scaler on training data."""
        logger.info("Fitting preprocessor on training data …")

        # Numerical fill values
        for col in self.numerical_features:
            if col not in df.columns:
                continue
            if self.fill_strategy == "median":
                self._num_fill_values[col] = df[col].median()
            elif self.fill_strategy == "mean":
                self._num_fill_values[col] = df[col].mean()
            else:
                self._num_fill_values[col] = 0.0

        # Categorical fill values + label encoders
        for col in self.categorical_features:
            if col not in df.columns:
                continue
            mode = df[col].mode()
            self._cat_fill_values[col] = mode[0] if not mode.empty else "UNKNOWN"
            le = LabelEncoder()
            filled = df[col].fillna(self._cat_fill_values[col]).astype(str)
            le.fit(filled)
            self._label_encoders[col] = le

        # Scaler on numerical columns
        num_cols = [c for c in self.numerical_features if c in df.columns]
        if num_cols:
            filled_num = df[num_cols].copy()
            for col in num_cols:
                filled_num[col] = filled_num[col].fillna(self._num_fill_values[col])
            self._scaler.fit(filled_num)

        self._fitted = True
        logger.info("Preprocessor fitting complete ✓")

    def _transform(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        parts = []

        # Numerical
        num_cols = [c for c in self.numerical_features if c in df.columns]
        if num_cols:
            num_df = df[num_cols].copy()
            for col in num_cols:
                num_df[col] = num_df[col].fillna(
                    self._num_fill_values.get(col, 0.0)
                )
            parts.append(self._scaler.transform(num_df))

        # Categorical
        cat_cols = [c for c in self.categorical_features if c in df.columns]
        for col in cat_cols:
            filled = df[col].fillna(self._cat_fill_values.get(col, "UNKNOWN")).astype(str)
            le = self._label_encoders[col]
            # Handle unseen labels gracefully
            known = set(le.classes_)
            safe = filled.apply(lambda x: x if x in known else "UNKNOWN")
            # Ensure UNKNOWN is in classes
            if "UNKNOWN" not in known:
                le.classes_ = np.append(le.classes_, "UNKNOWN")
            encoded = le.transform(safe).reshape(-1, 1).astype(float)
            parts.append(encoded)

        X = np.hstack(parts) if parts else np.empty((len(df), 0))

        y = df[self.target].values.astype(float) if self.target in df.columns else None
        return X, y
