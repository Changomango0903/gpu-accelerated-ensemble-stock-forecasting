"""
tree_features.py

Responsibilities:
  - Compute and assemble classic technical indicators.
  - Log progress to logs/features.log.
"""

import logging
import pandas as pd
from src.utils.io_helpers import setup_logging

# initialize logger
logger = logging.getLogger(__name__)
setup_logging()
fh = logging.FileHandler("logs/features.log")
fh.setLevel(logging.INFO)
logger.addHandler(fh)

def add_moving_averages(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Add Simple Moving Averages (SMA) for given window sizes."""
    for w in windows:
        df[f"SMA_{w}"] = df["Close"].rolling(window=w, min_periods=1).mean()
    return df

def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Compute Relative Strength Index (RSI)."""
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    df[f"RSI_{window}"] = 100 - (100 / (1 + rs))
    return df

def add_macd(df: pd.DataFrame,
             span_fast: int = 12,
             span_slow: int = 26,
             span_signal: int = 9) -> pd.DataFrame:
    """Compute MACD line and signal line."""
    ema_fast = df["Close"].ewm(span=span_fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=span_slow, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_signal"] = df["MACD"].ewm(span=span_signal, adjust=False).mean()
    return df

def add_bollinger_bands(df: pd.DataFrame,
                        window: int = 20,
                        num_std: float = 2.0) -> pd.DataFrame:
    """Compute Bollinger Bands (upper & lower)."""
    rolling_mean = df["Close"].rolling(window).mean()
    rolling_std = df["Close"].rolling(window).std()
    df["BB_upper"] = rolling_mean + (rolling_std * num_std)
    df["BB_lower"] = rolling_mean - (rolling_std * num_std)
    return df

def add_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Compute Average True Range (ATR)."""
    high_low = df["High"] - df["Low"]
    high_prev = (df["High"] - df["Close"].shift()).abs()
    low_prev  = (df["Low"]  - df["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)
    df[f"ATR_{window}"] = true_range.rolling(window, min_periods=1).mean()
    return df

def build_tree_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate all tree features from raw OHLCV.

    Args:
      raw_df: DataFrame indexed by Date with ['Open','High','Low','Close','Volume'].

    Returns:
      DataFrame including raw + indicators, NaN rows dropped.
    """
    logger.info("Starting tree feature build; input shape=%s", raw_df.shape)
    df = raw_df.copy()
    df = add_moving_averages(df, [20, 50, 200])
    df = add_rsi(df, window=14)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_atr(df, window=14)
    df = df.dropna()
    logger.info("Completed tree feature build; output shape=%s", df.shape)
    return df
