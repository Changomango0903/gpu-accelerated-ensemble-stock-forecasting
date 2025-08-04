"""
decompose.py

Responsibilities:
  - Signal decomposition utilities (ICEEMDAN stub, wavelet denoising).
  - A/B test harness comparing raw vs. denoised inputs.
  - Log progress to logs/decomposition.log.
"""

import logging
import numpy as np
import pandas as pd
from src.utils.io_helpers import setup_logging

# initialize logger
logger = logging.getLogger(__name__)
setup_logging()
fh = logging.FileHandler("logs/decomposition.log")
fh.setLevel(logging.INFO)
logger.addHandler(fh)

def decompose_iceemdan(signal: np.ndarray) -> np.ndarray:
    """
    Placeholder for ICEEMDAN decomposition.
    Returns the input unchanged until library integration.
    """
    logger.info("ICEEMDAN decomposition stub called; returning raw signal")
    return signal

def decompose_wavelet(
    signal: np.ndarray,
    wavelet: str = "db4",
    level: int = 3
) -> np.ndarray:
    """Perform wavelet denoising using PyWavelets."""
    import pywt
    logger.info("Starting wavelet denoising: wavelet=%s, level=%d", wavelet, level)
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    # zero detail coefficients except approximation
    coeffs[1:] = [np.zeros_like(c) for c in coeffs[1:]]
    rec = pywt.waverec(coeffs, wavelet)[: len(signal)]
    logger.info("Completed wavelet denoising")
    return rec

def ab_test_decomposition(
    raw: np.ndarray,
    method: str,
    model_train_fn
) -> dict[str, float]:
    """
    A/B test harness: compare raw vs. decomposed inputs.

    Args:
      raw: 1D price series
      method: 'iceemdan' or 'wavelet'
      model_train_fn: callable(X, y) -> model with .evaluate(X,y)

    Returns:
      {'raw_score':..., 'denoised_score':...}
    """
    from src.features.deep_features import create_windows

    if method == "iceemdan":
        proc = decompose_iceemdan(raw)
    else:
        proc = decompose_wavelet(raw)

    # build windows for raw & processed
    df_raw = pd.DataFrame({"Close": raw})
    X_raw, y_raw = create_windows(df_raw, sequence_length=60, feature_cols=["Close"])
    df_den = pd.DataFrame({"Close": proc})
    X_den, y_den = create_windows(df_den, sequence_length=60, feature_cols=["Close"])

    # train & evaluate
    m_raw = model_train_fn(X_raw, y_raw)
    m_den = model_train_fn(X_den, y_den)
    raw_score = m_raw.evaluate(X_raw, y_raw)
    den_score = m_den.evaluate(X_den, y_den)
    logger.info("A/B test complete: raw=%.4f, denoised=%.4f", raw_score, den_score)
    return {"raw_score": raw_score, "denoised_score": den_score}
