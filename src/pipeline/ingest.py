"""
pipeline/ingest.py
------------------
Data ingestion, schema validation, and initial quality checks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DataSchema:
    """Defines expected column types and constraints for validation."""

    target: str
    categorical_features: list[str] = field(default_factory=list)
    numerical_features: list[str] = field(default_factory=list)
    required_columns: list[str] = field(default_factory=list)
    max_missing_ratio: float = 0.5

    def __post_init__(self):
        if self.target not in self.required_columns:
            self.required_columns.append(self.target)


class DataIngestionError(Exception):
    pass


class DataIngester:
    """
    Loads raw tabular data, runs schema validation and quality checks.

    Args:
        schema: DataSchema describing expected structure.
        drop_duplicates: Whether to drop duplicate rows on load.
    """

    SUPPORTED_FORMATS = {".csv", ".parquet", ".feather", ".json"}

    def __init__(self, schema: DataSchema, drop_duplicates: bool = True):
        self.schema = schema
        self.drop_duplicates = drop_duplicates

    def load(self, path: str | Path) -> pd.DataFrame:
        """Load data from disk, validate, and return a clean DataFrame."""
        path = Path(path)
        logger.info(f"Loading data from: {path}")

        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        if path.suffix not in self.SUPPORTED_FORMATS:
            raise DataIngestionError(
                f"Unsupported format '{path.suffix}'. "
                f"Supported: {self.SUPPORTED_FORMATS}"
            )

        df = self._read(path)
        logger.info(f"Loaded {len(df):,} rows × {df.shape[1]} columns")

        df = self._validate(df)
        df = self._quality_check(df)

        if self.drop_duplicates:
            before = len(df)
            df = df.drop_duplicates()
            dropped = before - len(df)
            if dropped:
                logger.info(f"Dropped {dropped:,} duplicate rows")

        logger.info("Ingestion complete ✓")
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read(self, path: Path) -> pd.DataFrame:
        readers = {
            ".csv": pd.read_csv,
            ".parquet": pd.read_parquet,
            ".feather": pd.read_feather,
            ".json": pd.read_json,
        }
        return readers[path.suffix](path)

    def _validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check required columns are present."""
        missing = set(self.schema.required_columns) - set(df.columns)
        if missing:
            raise DataIngestionError(
                f"Missing required columns: {missing}"
            )

        # Auto-detect numerical features if not specified
        if not self.schema.numerical_features:
            self.schema.numerical_features = [
                c for c in df.select_dtypes(include="number").columns
                if c != self.schema.target
            ]
            logger.info(
                f"Auto-detected {len(self.schema.numerical_features)} "
                f"numerical features"
            )

        # Auto-detect categorical features if not specified
        if not self.schema.categorical_features:
            self.schema.categorical_features = [
                c for c in df.select_dtypes(include=["object", "category"]).columns
                if c != self.schema.target
            ]
            logger.info(
                f"Auto-detected {len(self.schema.categorical_features)} "
                f"categorical features"
            )

        return df

    def _quality_check(self, df: pd.DataFrame) -> pd.DataFrame:
        """Warn or drop columns exceeding missing-value threshold."""
        missing_ratio = df.isnull().mean()
        high_missing = missing_ratio[
            missing_ratio > self.schema.max_missing_ratio
        ]

        if not high_missing.empty:
            logger.warning(
                f"Dropping {len(high_missing)} columns with >"
                f"{self.schema.max_missing_ratio:.0%} missing values: "
                f"{list(high_missing.index)}"
            )
            df = df.drop(columns=high_missing.index)

        return df


def load_demo_data() -> pd.DataFrame:
    """
    Downloads the UCI Adult Income dataset for quick demos.
    Returns a cleaned DataFrame with a binary 'label' column.
    """
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases"
        "/adult/adult.data"
    )
    cols = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week",
        "native_country", "label",
    ]
    logger.info("Downloading UCI Adult Income dataset …")
    df = pd.read_csv(url, names=cols, na_values=" ?", skipinitialspace=True)
    df["label"] = (df["label"].str.strip() == ">50K").astype(int)
    return df
