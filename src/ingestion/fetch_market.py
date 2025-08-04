# src/ingestion/fetch_market.py

import os
import logging
import pandas as pd
import yfinance as yf
from src.utils.io_helpers import write_csv, setup_logging

logger = logging.getLogger(__name__)
setup_logging()

def fetch_and_save(
    symbol: str,
    start_date: str,
    end_date: str,
    output_dir: str = "data/raw/"
) -> None:
    """
    Fetches historical OHLCV for `symbol` and writes a clean CSV.

    The CSV will have one header: Date,Open,High,Low,Close,Volume
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Fetching %s from %s to %s", symbol, start_date, end_date)

    df = yf.download(symbol, start=start_date, end=end_date, progress=False)
    if df.empty or df.isna().all().all():
        logger.error("No usable data for %s", symbol)
        raise ValueError(f"No usable data for {symbol}.")

    # Ensure index is named correctly
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"

    # Subset & flatten columns
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.columns = df.columns.get_level_values(0)  # <— flatten MultiIndex

    # Build output path
    filename = f"{symbol}_{start_date}_{end_date}.csv"
    out_path = os.path.join(output_dir, filename)

    # Write CSV with a single header row
    write_csv(df, out_path, index_label="Date")
    logger.info("Saved clean CSV to %s", out_path)
