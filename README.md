# Phase 2: Feature Engineering & Conditional Decomposition

## Overview

Phase 2 transforms raw OHLCV data from Phase 1 into model-ready features and provides a framework to optionally denoise signals via decomposition methods.

---

## Goals

1. **Tree-based features**  
   - Simple Moving Averages (SMA) for 20, 50, 200 days  
   - Relative Strength Index (RSI, 14-day)  
   - MACD (12/26/9)  
   - Bollinger Bands (20-day, 2σ)  
   - Average True Range (ATR, 14-day)  
   - **Output:** `data/interim/tree_features.csv`

2. **Deep-model windows**  
   - Rolling OHLCV sequences (e.g. 60-day lookback)  
   - Per-window z-score normalization  
   - **Outputs:**  
     - `data/processed/X_windows.npy`  
     - `data/processed/y_targets.npy`

3. **Conditional decomposition**  
   - ICEEMDAN denoising  
   - Wavelet denoising  
   - A/B test harness to compare raw vs. denoised inputs  

4. **Documentation & sanity checks**  
   - Docstrings on every module and function  
   - Routines to assert no NaNs and correct array shapes  

---

## Directory Layout
project_root/
├── data/
│ ├── raw/ # Phase 1 outputs
│ ├── interim/
│ │ └── tree_features.csv # OHLCV + indicators
│ └── processed/
│ ├── X_windows.npy # Deep-model input arrays
│ └── y_targets.npy # Next-step Close targets
├── src/
│ ├── features/
│ │ ├── init.py
│ │ ├── tree_features.py # Indicator computations
│ │ └── deep_features.py # Window creation & normalization
│ └── decomposition/
│ ├── init.py
│ └── decompose.py # Denoising methods + A/B harness
└── notebooks/
└── 02_feature_engineering.ipynb

---

## Installation

```bash
# Phase 1 requirements
pip install -r requirements.txt

# Phase 2 requirements
pip install numpy pywt