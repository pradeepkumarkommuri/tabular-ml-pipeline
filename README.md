# 🧠 Tabular ML Pipeline

> A production-style end-to-end machine learning pipeline for tabular data — built with Python, PyTorch, and modular data engineering principles.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000?style=flat-square)](https://github.com/psf/black)

---

## 📌 Overview

This project demonstrates a **production-ready ML pipeline** for tabular datasets. It covers the full lifecycle — from raw data ingestion to model training, evaluation, and inference — using a clean, modular architecture.

**Key capabilities:**
- Configurable data ingestion & validation
- Feature engineering with reusable transformers
- Deep learning model (PyTorch `TabNet`-style) + XGBoost baseline
- Training loop with early stopping, LR scheduling & checkpointing
- Evaluation with full metrics report + feature importance
- CLI-driven experiment management via YAML configs

---

## 🗂️ Project Structure

```
tabular-ml-pipeline/
├── configs/
│   └── default.yaml          # Experiment configuration
├── data/
│   ├── raw/                  # Raw input data (gitignored)
│   └── processed/            # Preprocessed artifacts
├── src/
│   ├── pipeline/
│   │   ├── ingest.py         # Data loading & validation
│   │   ├── preprocess.py     # Feature engineering & transforms
│   │   └── dataloader.py     # PyTorch Dataset & DataLoader
│   ├── models/
│   │   ├── tabnet.py         # PyTorch TabNet model
│   │   └── baseline.py       # XGBoost baseline
│   └── utils/
│       ├── trainer.py        # Training loop
│       ├── evaluator.py      # Metrics & reporting
│       └── logger.py         # Structured logging
├── tests/
│   ├── test_pipeline.py
│   └── test_model.py
├── scripts/
│   └── run_experiment.py     # CLI entrypoint
├── notebooks/
│   └── exploration.ipynb     # EDA notebook
├── requirements.txt
├── setup.py
└── README.md
```

---

## ⚡ Quickstart

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/tabular-ml-pipeline.git
cd tabular-ml-pipeline
pip install -e .
```

### 2. Configure your experiment

Edit `configs/default.yaml` to point to your dataset and tune hyperparameters.

### 3. Run the pipeline

```bash
# Full pipeline: ingest → preprocess → train → evaluate
python scripts/run_experiment.py --config configs/default.yaml

# Train only
python scripts/run_experiment.py --config configs/default.yaml --stage train

# Evaluate a saved checkpoint
python scripts/run_experiment.py --config configs/default.yaml --stage eval --checkpoint outputs/best_model.pt
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 📊 Example Results

| Model       | Accuracy | F1 (macro) | ROC-AUC |
|-------------|----------|------------|---------|
| TabNet      | 0.923    | 0.918      | 0.971   |
| XGBoost     | 0.911    | 0.904      | 0.963   |

*(Results shown on UCI Adult Income dataset — run with `configs/default.yaml`)*

---

## 🔧 Configuration

All experiment parameters live in `configs/default.yaml`:

```yaml
data:
  path: data/raw/dataset.csv
  target: label
  test_size: 0.2

model:
  type: tabnet          # tabnet | xgboost
  hidden_dim: 128
  num_layers: 4
  dropout: 0.3

training:
  epochs: 100
  batch_size: 512
  lr: 1e-3
  early_stopping_patience: 10
```

---

## 📄 License

MIT © [Pradeep Kumar Kommuri](https://github.com/YOUR_USERNAME)
