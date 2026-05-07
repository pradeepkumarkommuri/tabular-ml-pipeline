"""
scripts/run_experiment.py
-------------------------
CLI entrypoint for the tabular ML pipeline.

Usage:
    python scripts/run_experiment.py --config configs/default.yaml
    python scripts/run_experiment.py --config configs/default.yaml --stage train
    python scripts/run_experiment.py --config configs/default.yaml --stage eval --checkpoint outputs/best_model.pt
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.dataloader import make_dataloaders
from src.pipeline.ingest import DataIngester, DataSchema, load_demo_data
from src.pipeline.preprocess import TabularPreprocessor
from src.models.tabnet import TabNet
from src.models.baseline import XGBoostBaseline
from src.utils.evaluator import Evaluator
from src.utils.logger import setup_logger
from src.utils.trainer import Trainer

logger = logging.getLogger("tabular_pipeline")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline stages
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(cfg: dict) -> None:
    """Full pipeline: ingest → preprocess → train → evaluate."""
    output_dir = Path(cfg["project"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg["project"]["seed"])

    # ── Ingest ────────────────────────────────────────────────────────
    data_cfg = cfg["data"]
    schema = DataSchema(
        target=data_cfg["target"],
        categorical_features=data_cfg.get("categorical_features", []),
        numerical_features=data_cfg.get("numerical_features", []),
    )

    data_path = data_cfg["path"]
    if not Path(data_path).exists():
        logger.warning(
            f"Data file not found at '{data_path}'. "
            "Falling back to demo dataset (UCI Adult Income)."
        )
        df = load_demo_data()
    else:
        ingester = DataIngester(schema=schema, drop_duplicates=data_cfg.get("drop_duplicates", True))
        df = ingester.load(data_path)

    # ── Preprocess ────────────────────────────────────────────────────
    preprocessor = TabularPreprocessor(
        target=schema.target,
        numerical_features=schema.numerical_features,
        categorical_features=schema.categorical_features,
        fill_strategy=data_cfg.get("fill_strategy", "median"),
        test_size=data_cfg.get("test_size", 0.2),
        val_size=data_cfg.get("val_size", 0.1),
        seed=cfg["project"]["seed"],
    )
    X_train, y_train, X_val, y_val, X_test, y_test = preprocessor.fit_transform(df)
    preprocessor.save(output_dir / "preprocessor.pkl")

    # ── DataLoaders ────────────────────────────────────────────────────
    train_cfg = cfg["training"]
    train_loader, val_loader, test_loader = make_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=train_cfg["batch_size"],
    )

    model_type = cfg["model"]["type"]

    # ── Train ─────────────────────────────────────────────────────────
    evaluator = Evaluator()

    if model_type == "tabnet":
        model_cfg = cfg["model"]
        model = TabNet(
            input_dim=preprocessor.n_features,
            output_dim=1,
            hidden_dim=model_cfg.get("hidden_dim", 128),
            num_steps=model_cfg.get("num_layers", 3),
            gamma=model_cfg.get("gamma", 1.3),
            dropout=model_cfg.get("dropout", 0.1),
        )

        trainer = Trainer(
            model=model,
            lr=train_cfg.get("lr", 1e-3),
            weight_decay=train_cfg.get("weight_decay", 1e-5),
            epochs=train_cfg.get("epochs", 100),
            patience=train_cfg.get("early_stopping_patience", 10),
            mixed_precision=train_cfg.get("mixed_precision", True),
            output_dir=str(output_dir),
        )

        trainer.fit(
            train_loader, val_loader,
            scheduler_type=train_cfg.get("scheduler", "cosine"),
        )
        trainer.load_best()

        metrics = evaluator.evaluate_torch(model, test_loader)
        evaluator.print_report(metrics, "TabNet")
        evaluator.save_report(metrics, output_dir / "tabnet_results.json", "TabNet")

    elif model_type == "xgboost":
        baseline = XGBoostBaseline(seed=cfg["project"]["seed"])
        baseline.fit(X_train, y_train, X_val, y_val, feature_names=preprocessor.feature_names)
        baseline.save(output_dir / "xgboost_model.pkl")

        metrics = evaluator.evaluate_sklearn(baseline, X_test, y_test)
        evaluator.print_report(metrics, "XGBoost")
        evaluator.save_report(metrics, output_dir / "xgboost_results.json", "XGBoost")

    else:
        raise ValueError(f"Unknown model type: '{model_type}'")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tabular ML Pipeline")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--stage",
        choices=["full", "train", "eval"],
        default="full",
        help="Pipeline stage to run (default: full)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint for eval-only mode",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    run_pipeline(cfg)
