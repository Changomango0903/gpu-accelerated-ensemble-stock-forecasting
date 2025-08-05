"""
cross_validation.py

Responsibilities:
  - Implement nested walk-forward cross-validation for time series.
  - Prevent lookahead bias critical for financial data.
  - Log cross-validation progress to logs/models.log.
  - Save fold-by-fold results for analysis.
"""

import logging
import os
from typing import Dict, List, Tuple, Type, Any
import numpy as np
import pandas as pd
from src.models.base_model import BaseModel
from src.utils.io_helpers import setup_logging

# Initialize logger
logger = logging.getLogger(__name__)
setup_logging()
fh = logging.FileHandler("logs/models.log")
fh.setLevel(logging.INFO)
logger.addHandler(fh)


def walk_forward_cv(
    X: np.ndarray,
    y: np.ndarray,
    model_class: Type[BaseModel],
    model_params: Dict[str, Any],
    min_train_size: int = 252,  # ~1 year of trading days
    test_size: int = 21,        # ~1 month ahead  
    step_size: int = 21,        # Monthly retraining
    max_folds: int = 20         # Limit computation time
) -> Dict[str, List[float]]:
    """
    Implement walk-forward cross-validation for time series data.
    
    This is critical for financial data to prevent lookahead bias. Each fold:
    1. Trains on historical data only
    2. Tests on the next period
    3. Steps forward in time
    
    Args:
        X: Feature matrix - shape depends on model type.
        y: Target vector of shape (n_samples,).
        model_class: Class to instantiate for each fold.
        model_params: Parameters to pass to model constructor.
        min_train_size: Minimum samples needed for training.
        test_size: Number of samples to test on each fold.
        step_size: How many samples to advance each fold.
        max_folds: Maximum number of folds to prevent excessive computation.
        
    Returns:
        Dictionary with lists of metrics from each fold.
    """
    logger.info("Starting walk-forward CV: train_size=%d, test_size=%d, step_size=%d",
                min_train_size, test_size, step_size)
    
    n_samples = len(X)
    
    # Validate parameters
    if min_train_size + test_size > n_samples:
        raise ValueError(f"Insufficient data: need {min_train_size + test_size}, got {n_samples}")
    
    # Storage for results from each fold
    fold_results = {
        'fold': [],
        'train_start': [],
        'train_end': [],
        'test_start': [],
        'test_end': [],
        'mae': [],
        'mse': [],
        'rmse': [],
        'r2': [],
        'directional_accuracy': []
    }
    
    fold_num = 0
    
    # Walk forward through time
    train_start = 0
    while train_start + min_train_size + test_size <= n_samples and fold_num < max_folds:
        
        # Define train and test periods
        train_end = train_start + min_train_size
        test_start = train_end
        test_end = min(test_start + test_size, n_samples)
        
        # Skip if test set is too small
        if test_end - test_start < test_size:
            logger.warning("Test set too small for fold %d, stopping CV", fold_num)
            break
            
        logger.info("Fold %d: Train[%d:%d], Test[%d:%d]", 
                   fold_num, train_start, train_end, test_start, test_end)
        
        try:
            # Extract train/test data
            X_train = X[train_start:train_end]
            y_train = y[train_start:train_end]
            X_test = X[test_start:test_end]
            y_test = y[test_start:test_end]
            
            # Validate data quality
            if np.isnan(X_train).any() or np.isnan(y_train).any():
                logger.error("NaN values found in fold %d training data", fold_num)
                raise ValueError(f"NaN values in fold {fold_num} training data")
                
            if np.isnan(X_test).any() or np.isnan(y_test).any():
                logger.error("NaN values found in fold %d test data", fold_num)
                raise ValueError(f"NaN values in fold {fold_num} test data")
            
            # Train model for this fold
            model = model_class(**model_params)
            model.fit(X_train, y_train)
            
            # Evaluate on test set
            metrics = model.evaluate(X_test, y_test)
            
            # Store results
            fold_results['fold'].append(fold_num)
            fold_results['train_start'].append(train_start)
            fold_results['train_end'].append(train_end)
            fold_results['test_start'].append(test_start)
            fold_results['test_end'].append(test_end)
            fold_results['mae'].append(metrics['mae'])
            fold_results['mse'].append(metrics['mse'])
            fold_results['rmse'].append(metrics['rmse'])
            fold_results['r2'].append(metrics['r2'])
            fold_results['directional_accuracy'].append(metrics['directional_accuracy'])
            
            logger.info("Fold %d complete - MAE: %.4f, Dir. Acc.: %.2f%%",
                       fold_num, metrics['mae'], metrics['directional_accuracy'] * 100)
            
        except Exception as e:
            logger.error("Error in fold %d: %s", fold_num, str(e))
            # Continue to next fold rather than failing entirely
            
        # Move to next fold
        fold_num += 1
        train_start += step_size
    
    logger.info("Walk-forward CV complete: %d folds processed", len(fold_results['fold']))
    
    return fold_results


def summarize_cv_results(fold_results: Dict[str, List[float]], model_name: str) -> Dict[str, float]:
    """
    Summarize cross-validation results across all folds.
    
    Args:
        fold_results: Results from walk_forward_cv.
        model_name: Name of the model for logging.
        
    Returns:
        Dictionary with mean and std of metrics across folds.
    """
    if len(fold_results['fold']) == 0:
        logger.error("No valid folds in CV results")
        return {}
    
    summary = {}
    metrics = ['mae', 'mse', 'rmse', 'r2', 'directional_accuracy']
    
    for metric in metrics:
        if metric in fold_results and len(fold_results[metric]) > 0:
            values = np.array(fold_results[metric])
            summary[f'{metric}_mean'] = np.mean(values)
            summary[f'{metric}_std'] = np.std(values)
    
    # Log summary
    logger.info("%s CV Summary:", model_name)
    logger.info("  MAE: %.4f ± %.4f", summary.get('mae_mean', 0), summary.get('mae_std', 0))
    logger.info("  RMSE: %.4f ± %.4f", summary.get('rmse_mean', 0), summary.get('rmse_std', 0))
    logger.info("  R²: %.4f ± %.4f", summary.get('r2_mean', 0), summary.get('r2_std', 0))
    logger.info("  Directional Accuracy: %.2f%% ± %.2f%%", 
               summary.get('directional_accuracy_mean', 0) * 100,
               summary.get('directional_accuracy_std', 0) * 100)
    
    return summary


def save_cv_results(fold_results: Dict[str, List[float]], summary: Dict[str, float], 
                   model_name: str, output_dir: str = "reports/summaries/") -> None:
    """
    Save cross-validation results to CSV files.
    
    Args:
        fold_results: Detailed fold-by-fold results.
        summary: Summary statistics across folds.
        model_name: Name of the model.
        output_dir: Directory to save results.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    df_folds = pd.DataFrame(fold_results)
    fold_path = os.path.join(output_dir, f"cv_folds_{model_name.lower()}.csv")
    df_folds.to_csv(fold_path, index=False)
    logger.info("Saved fold results to %s", fold_path)
    
    # Save summary
    df_summary = pd.DataFrame([summary])
    df_summary.insert(0, 'model', model_name)
    summary_path = os.path.join(output_dir, f"cv_summary_{model_name.lower()}.csv")
    df_summary.to_csv(summary_path, index=False)
    logger.info("Saved CV summary to %s", summary_path)


def expanding_window_cv(
    X: np.ndarray,
    y: np.ndarray,
    model_class: Type[BaseModel],
    model_params: Dict[str, Any],
    min_train_size: int = 252,
    test_size: int = 21,
    step_size: int = 21,
    max_folds: int = 20
) -> Dict[str, List[float]]:
    """
    Alternative CV strategy: expanding window instead of sliding window.
    
    Each fold uses all available historical data up to that point.
    This can be more realistic for financial models where more data helps.
    
    Args:
        Similar to walk_forward_cv but uses expanding training window.
        
    Returns:
        Dictionary with lists of metrics from each fold.
    """
    logger.info("Starting expanding window CV: min_train=%d, test_size=%d, step_size=%d",
                min_train_size, test_size, step_size)
    
    n_samples = len(X)
    fold_results = {
        'fold': [], 'train_start': [], 'train_end': [], 'test_start': [], 'test_end': [],
        'mae': [], 'mse': [], 'rmse': [], 'r2': [], 'directional_accuracy': []
    }
    
    fold_num = 0
    train_start = 0  # Always start from beginning
    
    # First test starts after minimum training period
    test_start = min_train_size
    
    while test_start + test_size <= n_samples and fold_num < max_folds:
        
        train_end = test_start
        test_end = test_start + test_size
        
        logger.info("Expanding fold %d: Train[%d:%d], Test[%d:%d]", 
                   fold_num, train_start, train_end, test_start, test_end)
        
        try:
            # Extract data (training window expands each fold)
            X_train = X[train_start:train_end]
            y_train = y[train_start:train_end]
            X_test = X[test_start:test_end]
            y_test = y[test_start:test_end]
            
            # Train and evaluate
            model = model_class(**model_params)
            model.fit(X_train, y_train)
            metrics = model.evaluate(X_test, y_test)
            
            # Store results (same structure as walk_forward_cv)
            fold_results['fold'].append(fold_num)
            fold_results['train_start'].append(train_start)
            fold_results['train_end'].append(train_end)
            fold_results['test_start'].append(test_start)
            fold_results['test_end'].append(test_end)
            fold_results['mae'].append(metrics['mae'])
            fold_results['mse'].append(metrics['mse'])
            fold_results['rmse'].append(metrics['rmse'])
            fold_results['r2'].append(metrics['r2'])
            fold_results['directional_accuracy'].append(metrics['directional_accuracy'])
            
        except Exception as e:
            logger.error("Error in expanding fold %d: %s", fold_num, str(e))
            
        fold_num += 1
        test_start += step_size
    
    logger.info("Expanding window CV complete: %d folds processed", len(fold_results['fold']))
    return fold_results