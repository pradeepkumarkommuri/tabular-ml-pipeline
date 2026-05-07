"""
tests/test_pipeline.py
----------------------
Unit tests for the data ingestion and preprocessing modules.
"""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.ingest import DataIngester, DataSchema, DataIngestionError
from src.pipeline.preprocess import TabularPreprocessor


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 500
    return pd.DataFrame({
        "age":        np.random.randint(18, 70, n).astype(float),
        "income":     np.random.randn(n) * 20_000 + 60_000,
        "education":  np.random.choice(["high_school", "bachelor", "master", "phd"], n),
        "occupation": np.random.choice(["tech", "finance", "healthcare", "other"], n),
        "label":      np.random.randint(0, 2, n),
    })


@pytest.fixture
def schema():
    return DataSchema(
        target="label",
        numerical_features=["age", "income"],
        categorical_features=["education", "occupation"],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Ingestion tests
# ──────────────────────────────────────────────────────────────────────────────

class TestDataIngester:
    def test_validates_required_columns(self, schema):
        ingester = DataIngester(schema=schema)
        bad_df = pd.DataFrame({"x": [1, 2, 3]})  # missing 'label'
        with pytest.raises(DataIngestionError):
            ingester._validate(bad_df)

    def test_drops_high_missing_columns(self, schema, sample_df):
        ingester = DataIngester(schema=schema)
        sample_df["mostly_null"] = np.nan  # 100% missing
        cleaned = ingester._quality_check(sample_df)
        assert "mostly_null" not in cleaned.columns

    def test_file_not_found(self, schema):
        ingester = DataIngester(schema=schema)
        with pytest.raises(FileNotFoundError):
            ingester.load("/nonexistent/path/data.csv")

    def test_unsupported_format(self, schema, tmp_path):
        ingester = DataIngester(schema=schema)
        bad_file = tmp_path / "data.txt"
        bad_file.write_text("hello")
        with pytest.raises(DataIngestionError):
            ingester.load(bad_file)


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing tests
# ──────────────────────────────────────────────────────────────────────────────

class TestTabularPreprocessor:
    def test_output_shapes(self, sample_df):
        pp = TabularPreprocessor(
            target="label",
            numerical_features=["age", "income"],
            categorical_features=["education", "occupation"],
            test_size=0.2,
            val_size=0.1,
            seed=42,
        )
        X_train, y_train, X_val, y_val, X_test, y_test = pp.fit_transform(sample_df)

        total = len(sample_df)
        assert X_train.shape[0] + X_val.shape[0] + X_test.shape[0] == total
        assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1]
        assert y_train.shape[0] == X_train.shape[0]

    def test_no_nan_in_output(self, sample_df):
        # Introduce some NaNs
        sample_df.loc[:10, "age"] = np.nan
        sample_df.loc[5:15, "education"] = np.nan

        pp = TabularPreprocessor(
            target="label",
            numerical_features=["age", "income"],
            categorical_features=["education", "occupation"],
        )
        X_train, *_ = pp.fit_transform(sample_df)
        assert not np.isnan(X_train).any(), "Output contains NaNs"

    def test_transform_before_fit_raises(self, sample_df):
        pp = TabularPreprocessor(
            target="label",
            numerical_features=["age", "income"],
            categorical_features=["education", "occupation"],
        )
        with pytest.raises(RuntimeError):
            pp.transform(sample_df)

    def test_n_features(self, sample_df):
        pp = TabularPreprocessor(
            target="label",
            numerical_features=["age", "income"],
            categorical_features=["education", "occupation"],
        )
        pp.fit_transform(sample_df)
        assert pp.n_features == 4  # 2 numerical + 2 categorical

    def test_save_load_roundtrip(self, sample_df, tmp_path):
        pp = TabularPreprocessor(
            target="label",
            numerical_features=["age", "income"],
            categorical_features=["education", "occupation"],
        )
        pp.fit_transform(sample_df)

        save_path = tmp_path / "preprocessor.pkl"
        pp.save(save_path)
        loaded = TabularPreprocessor.load(save_path)

        X_orig, _ = pp.transform(sample_df)
        X_loaded, _ = loaded.transform(sample_df)
        np.testing.assert_array_almost_equal(X_orig, X_loaded)
