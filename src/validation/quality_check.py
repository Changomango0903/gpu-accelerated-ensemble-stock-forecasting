# src/validation/quality_check.py

"""
quality_check.py

Responsibility:
    - Load raw CSV files.
    - Perform integrity checks: missing dates, NaNs, extreme outliers.
    - Summarize results to DataFrame.
"""

import os
import glob
import logging
import pandas as pd
from src.utils.io_helpers import read_csv, setup_logging

# initialize logger
logger = logging.getLogger(__name__)
setup_logging()


def run_quality_checks(raw_dir: str = "data/raw/") -> pd.DataFrame:
    """
    Iterates CSVs in `raw_dir`, performs data integrity checks.

    Checks:
      1. Missing trading days (weekdays without data).
      2. NaN counts per column.
      3. Price spikes/drops beyond 5σ.

    Args:
        raw_dir: Path to raw CSV directory.

    Returns:
        DataFrame of summary metrics per file.
    """
    summaries = []
    pattern = os.path.join(raw_dir, "*.csv")
    logger.info("Running quality checks on files in %s", raw_dir)

    for path in glob.glob(pattern):
        logger.debug("Checking %s", path)
        df = read_csv(path)

        # 1. Missing dates
        all_days = pd.date_range(df.index.min(), df.index.max(), freq="B")
        missing = sorted(set(all_days) - set(df.index))

        # 2. NaN counts
        nan_counts = df.isna().sum().to_dict()

        # 3. Outliers
        pct = df["Close"].pct_change().dropna()
        sigma = pct.std()
        outliers = pct[(pct.abs() > 5 * sigma)].count()

        summaries.append({
            "file": os.path.basename(path),
            "start": df.index.min(),
            "end": df.index.max(),
            "missing_days": len(missing),
            "nan_counts": nan_counts,
            "price_outliers": int(outliers),
        })

    result = pd.DataFrame(summaries)
    logger.info("Quality check complete. %d files processed.", len(summaries))
    return result
