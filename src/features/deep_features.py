"""
deep_features.py

Responsibilities:
  - Extract rolling OHLCV windows for deep models.
  - Normalize each window (z-score).
  - Log progress to logs/features.log.
"""

import logging
import numpy as np
import pandas as pd
from src.utils.io_helpers import setup_logging

# initialize logger
logger = logging.getLogger(__name__)
setup_logging()
fh = logging.FileHandler("logs/features.log")
fh.setLevel(logging.INFO)
logger.addHandler(fh)

def create_windows(
    df: pd.DataFrame,
    sequence_length: int = 60,
    feature_cols: list[str] = ["Open", "High", "Low", "Close", "Volume"]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a DataFrame into rolling windows and next-step targets.

    Returns:
      X: shape (n_windows, sequence_length, n_features)
      y: shape (n_windows,)
    """
    logger.info("Creating deep windows: seq_len=%d, features=%s",
                sequence_length, feature_cols)
    data = df[feature_cols].values
    X, y = [], []
    for i in range(sequence_length, len(data)):
        seq = data[i - sequence_length : i]
        mu = seq.mean(axis=0)
        sigma = seq.std(axis=0, ddof=0) + 1e-8
        seq_norm = (seq - mu) / sigma
        X.append(seq_norm)
        y.append(data[i, feature_cols.index("Close")])
    X_arr, y_arr = np.array(X), np.array(y)
    logger.info("Created windows: X=%s, y=%s", X_arr.shape, y_arr.shape)
    return X_arr, y_arr
