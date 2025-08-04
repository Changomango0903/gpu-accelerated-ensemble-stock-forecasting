# src/utils/io_helpers.py

"""
io_helpers.py

Responsibilities:
  - Common I/O helpers and logging setup.
"""

import os
import logging

def setup_logging(log_level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def read_csv(path: str):
    import pandas as pd
    return pd.read_csv(path, index_col=0, parse_dates=True)

def write_csv(df, path: str, index_label: str = None) -> None:
    """
    Writes a DataFrame to CSV with exactly one header row.

    Args:
      df: DataFrame to write.
      path: Target file path.
      index_label: Name to give the index column.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index_label=index_label)
