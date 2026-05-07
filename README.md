# 🧠 Tabular ML Pipeline

> A production-style end-to-end machine learning pipeline for tabular data — built with Python, PyTorch, and modular data engineering principles.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CI](https://img.shields.io/github/actions/workflow/status/pradeepkumarkommuri/tabular-ml-pipeline/ci.yml?style=flat-square&label=CI)](https://github.com/pradeepkumarkommuri/tabular-ml-pipeline/actions)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000?style=flat-square)](https://github.com/psf/black)

---

## 📌 Overview

This project demonstrates a **production-ready ML pipeline** for tabular datasets. It covers the full lifecycle — from raw data ingestion to model training, evaluation, and inference — using a clean, modular architecture.

**Key capabilities:**
- Configurable data ingestion & schema validation with quality checks
- Feature engineering with reusable, fit-on-train transformers (no data leakage)
- Custom PyTorch **TabNet** model with ghost batch norm, sequential attention & sparsity regularisation
- XGBoost baseline for side-by-side comparison
- Training loop with early stopping, cosine LR scheduling, mixed precision & checkpointing
- Full evaluation report: Accuracy, Precision, Recall, F1, ROC-AUC, Avg Precision
- CLI-driven experiment management via YAML configs
- 16 unit tests with GitHub Actions CI

---

## 🗂️ Project Structure

```
tabular-ml-pipeline/
├── configs/
│   └── default.yaml              # Experiment configuration
├── data/
│   ├── raw/                      # Raw input data (gitignored)
│   └── processed/                # Preprocessed artifacts
├── src/
│   ├── pipeline/
│   │   ├── ingest.py             # Data loading, schema validation & QA
│   │   ├── preprocess.py         # Feature engineering, encoding, splitting
│   │   └── dataloader.py         # PyTorch Dataset & DataLoader factory
│   ├── models/
│   │   ├── tabnet.py             # PyTorch TabNet (147k params)
│   │   └── baseline.py           # XGBoost baseline
│   └── utils/
│       ├── trainer.py            # Training loop with AMP & checkpointing
│       ├── evaluator.py          # Metrics & JSON report generation
│       └── logger.py             # Structured logging
├── tests/
│   ├── test_pipeline.py          # 9 pipeline unit tests
│   └── test_model.py             # 7 model unit tests
├── scripts/
│   └── run_experiment.py         # CLI entrypoint
├── notebooks/
│   └── exploration.ipynb         # EDA notebook
├── .github/workflows/ci.yml      # GitHub Actions CI (Python 3.10 & 3.11)
├── requirements.txt
├── setup.py
└── README.md
```

---

## ⚡ Quickstart

### 1. Clone & install

```bash
git clone https://github.com/pradeepkumarkommuri/tabular-ml-pipeline.git
cd tabular-ml-pipeline
pip install -e .
```

### 2. Prepare your dataset

Place your CSV at `data/raw/dataset.csv` and update `configs/default.yaml` with your column names.  
No dataset? The pipeline auto-generates a synthetic demo dataset on first run.

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

16 tests covering ingestion validation, preprocessing correctness, no-NaN guarantees, save/load roundtrip, model forward pass, gradient flow, and batch size handling.

---

## 📊 Pipeline Run — Sample Output

```
INFO | src.pipeline.ingest      | Loaded 5,000 rows × 9 columns
INFO | src.pipeline.preprocess  | Split sizes — train: 3,500  val: 500  test: 1,000
INFO | src.pipeline.preprocess  | Feature matrix shape: (3500, 8)
INFO | src.utils.trainer        | Model parameters: 147,665
INFO | src.utils.trainer        | Epoch 001/100 | train_loss=0.8415 | val_loss=0.6924 | val_auc=0.5449
...
INFO | src.utils.trainer        | Early stopping triggered at epoch 16
INFO | src.utils.trainer        | Training complete. Best val_auc=0.5636
```

**Test set results (TabNet):**

| Metric        | Score  |
|---------------|--------|
| Accuracy      | 0.4980 |
| Precision     | 0.5113 |
| Recall        | 0.5292 |
| F1 Macro      | 0.4969 |
| ROC-AUC       | 0.4905 |
| Avg Precision | 0.5048 |

> Results shown on a synthetic random dataset — scores near 0.5 are expected (no real signal). On a real dataset, TabNet typically achieves ROC-AUC of 0.85+.

---

## 🔧 Configuration

All experiment parameters live in `configs/default.yaml`:

```yaml
data:
  path: data/raw/dataset.csv
  target: label
  categorical_features: ["education", "occupation", "marital_status"]
  numerical_features: ["age", "income", "hours_per_week", "capital_gain", "capital_loss"]
  test_size: 0.2
  val_size: 0.1
  fill_strategy: median       # median | mean | constant

model:
  type: tabnet                # tabnet | xgboost
  hidden_dim: 128
  num_layers: 4
  dropout: 0.3
  gamma: 1.3

training:
  epochs: 100
  batch_size: 512
  lr: 1.0e-3
  weight_decay: 1.0e-5
  scheduler: cosine           # cosine | step | none
  early_stopping_patience: 10
  mixed_precision: true
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189AB4?style=flat-square)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![pytest](https://img.shields.io/badge/pytest-0A9EDC?style=flat-square&logo=pytest&logoColor=white)

---

## 📄 License

MIT © [Pradeep Kumar Kommuri](https://github.com/pradeepkumarkommuri)
