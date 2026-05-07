"""
utils/trainer.py
----------------
Production training loop for the TabNet model.
Features: early stopping, LR scheduling, mixed precision, checkpointing.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Stops training when the monitored metric stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.stop = False

    def __call__(self, score: float) -> bool:
        improved = (
            self.best_score is None
            or (self.mode == "max" and score > self.best_score + self.min_delta)
            or (self.mode == "min" and score < self.best_score - self.min_delta)
        )
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


class Trainer:
    """
    Full training loop for TabNet with:
    - Binary cross-entropy loss + sparsity regularisation
    - AdamW optimiser + cosine LR schedule
    - Mixed-precision (fp16) training on GPU
    - Early stopping on validation AUC
    - Checkpoint saving of the best model

    Args:
        model: TabNet (or compatible) nn.Module.
        device: 'cuda' | 'cpu' | 'mps'.
        lr: Initial learning rate.
        weight_decay: L2 regularisation.
        epochs: Maximum training epochs.
        patience: Early stopping patience.
        sparse_lambda: Weight on sparsity regularisation loss.
        mixed_precision: Enable AMP on CUDA.
        output_dir: Directory for saving checkpoints and logs.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "auto",
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        epochs: int = 100,
        patience: int = 10,
        sparse_lambda: float = 1e-3,
        mixed_precision: bool = True,
        output_dir: str = "outputs",
    ):
        self.device = self._resolve_device(device)
        self.model = model.to(self.device)
        self.epochs = epochs
        self.sparse_lambda = sparse_lambda
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.BCEWithLogitsLoss()
        self.scaler = GradScaler(enabled=mixed_precision and self.device.type == "cuda")
        self.early_stopping = EarlyStopping(patience=patience, mode="max")

        logger.info(f"Trainer initialised on device: {self.device}")
        logger.info(f"Model parameters: {model.num_parameters:,}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        scheduler_type: str = "cosine",
    ) -> dict[str, list[float]]:
        """
        Train the model. Returns history dict with per-epoch metrics.
        """
        scheduler = self._build_scheduler(scheduler_type)
        history: dict[str, list[float]] = {
            "train_loss": [], "val_loss": [], "val_auc": []
        }

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(train_loader)
            val_loss, val_auc = self._eval_epoch(val_loader)

            if scheduler:
                scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_auc"].append(val_auc)

            elapsed = time.time() - t0
            logger.info(
                f"Epoch {epoch:03d}/{self.epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_auc={val_auc:.4f} | "
                f"{elapsed:.1f}s"
            )

            # Checkpoint best model
            if self.early_stopping.best_score == val_auc:
                self._save_checkpoint(epoch, val_auc)

            if self.early_stopping(val_auc):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        logger.info(
            f"Training complete. Best val_auc={self.early_stopping.best_score:.4f}"
        )
        return history

    def load_best(self) -> None:
        """Load the best checkpoint back into the model."""
        ckpt_path = self.output_dir / "best_model.pt"
        state = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
        logger.info(
            f"Loaded best checkpoint (epoch {state['epoch']}, "
            f"val_auc={state['val_auc']:.4f})"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with autocast(enabled=self.scaler.is_enabled()):
                logits, entropy_loss = self.model(X_batch)
                bce = self.criterion(logits.squeeze(1), y_batch)
                loss = bce + self.sparse_lambda * entropy_loss

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * len(X_batch)

        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader) -> tuple[float, float]:
        from sklearn.metrics import roc_auc_score

        self.model.eval()
        total_loss = 0.0
        all_probs, all_labels = [], []

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)

            logits, entropy_loss = self.model(X_batch)
            loss = self.criterion(logits.squeeze(1), y_batch)
            total_loss += loss.item() * len(X_batch)

            probs = torch.sigmoid(logits.squeeze(1)).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y_batch.cpu().numpy())

        val_loss = total_loss / len(loader.dataset)
        val_auc = roc_auc_score(all_labels, all_probs)
        return val_loss, val_auc

    def _save_checkpoint(self, epoch: int, val_auc: float) -> None:
        path = self.output_dir / "best_model.pt"
        torch.save(
            {
                "epoch": epoch,
                "val_auc": val_auc,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def _build_scheduler(self, scheduler_type: str):
        if scheduler_type == "cosine":
            return CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-6)
        return None

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)
