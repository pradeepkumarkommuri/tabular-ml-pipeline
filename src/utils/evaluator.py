"""
utils/evaluator.py
------------------
Evaluation utilities: metrics computation, feature importance, report generation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluates a trained model on a DataLoader and produces a full metrics report.

    Supports both PyTorch TabNet and sklearn-compatible models (XGBoost).
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)

    def evaluate_torch(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        threshold: float = 0.5,
    ) -> dict[str, float]:
        """Run inference with a PyTorch model and compute classification metrics."""
        model.eval()
        all_probs, all_labels = [], []

        with torch.no_grad():
            for batch in loader:
                X_batch, y_batch = batch[0].to(self.device), batch[1]
                logits, _ = model(X_batch)
                probs = torch.sigmoid(logits.squeeze(1)).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(y_batch.numpy())

        return self._compute_metrics(
            np.array(all_labels),
            np.array(all_probs),
            threshold,
        )

    def evaluate_sklearn(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        threshold: float = 0.5,
    ) -> dict[str, float]:
        """Evaluate an sklearn-compatible model (e.g. XGBoost)."""
        probs = model.predict_proba(X)
        return self._compute_metrics(y, probs, threshold)

    def print_report(self, metrics: dict[str, float], model_name: str = "Model") -> None:
        """Pretty-print a metrics report to the logger."""
        sep = "─" * 40
        logger.info(sep)
        logger.info(f"  {model_name} — Test Results")
        logger.info(sep)
        for k, v in metrics.items():
            logger.info(f"  {k:<20} {v:.4f}")
        logger.info(sep)

    def save_report(
        self,
        metrics: dict[str, float],
        path: str | Path,
        model_name: str = "model",
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        report = {"model": model_name, "metrics": metrics}
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Evaluation report saved to {path}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_metrics(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float = 0.5,
    ) -> dict[str, float]:
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        y_pred = (y_prob >= threshold).astype(int)

        return {
            "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "f1_macro":  round(float(f1_score(y_true, y_pred, average="macro")), 4),
            "roc_auc":   round(float(roc_auc_score(y_true, y_prob)), 4),
            "avg_precision": round(float(average_precision_score(y_true, y_prob)), 4),
        }
