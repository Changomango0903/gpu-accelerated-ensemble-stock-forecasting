# src/models/train_baseline_models.py

"""
train_baseline_models.py

Responsibilities:
  - Train all three baseline models (LightGBM, XGBoost, LSTM).
  - Execute walk-forward cross-validation.
  - Generate evaluation plots and summaries.
  - Complete Phase 3 deliverables.
"""

import logging
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models import (
    LightGBMRegressor, XGBoostRegressor, LSTMRegressor,
    walk_forward_cv, summarize_cv_results, save_cv_results,
    plot_predictions_vs_actual, plot_residuals, plot_cv_performance,
    plot_model_comparison, plot_feature_importance, generate_performance_report
)
from src.utils.io_helpers import setup_logging, read_csv

# Initialize logging
logger = logging.getLogger(__name__)
setup_logging()
fh = logging.FileHandler("logs/models.log")
fh.setLevel(logging.INFO)
logger.addHandler(fh)


def load_data():
    """
    Load preprocessed data for model training.
    
    Returns:
        Tuple of (tree_features_df, X_windows, y_targets).
    """
    logger.info("Loading preprocessed data...")
    
    # Load tree features for gradient boosting models
    try:
        tree_features = read_csv("data/interim/tree_features.csv")
        logger.info("Loaded tree features: %s", tree_features.shape)
    except FileNotFoundError:
        logger.error("Tree features file not found. Run Phase 2 first.")
        raise
    
    # Load deep learning windows
    try:
        X_windows = np.load("data/processed/X_windows.npy")
        
        # Try both possible target file names
        if os.path.exists("data/processed/y_targets_continuous.npy"):
            y_targets = np.load("data/processed/y_targets_continuous.npy")
        elif os.path.exists("data/processed/y_targets.npy"):
            y_targets = np.load("data/processed/y_targets.npy")
        else:
            raise FileNotFoundError("No target file found")
            
        logger.info("Loaded windows: X=%s, y=%s", X_windows.shape, y_targets.shape)
    except FileNotFoundError:
        logger.error("Deep feature files not found. Run prepare_phase3_data.py first.")
        raise
    
    return tree_features, X_windows, y_targets


def prepare_tree_data(tree_features_df):
    """
    Prepare data for tree-based models (LightGBM, XGBoost).
    
    Args:
        tree_features_df: DataFrame with technical indicators.
        
    Returns:
        Tuple of (X_tree, y_tree, feature_names).
    """
    logger.info("Preparing tree-based model data...")
    
    # Remove raw OHLCV columns, keep only engineered features
    feature_cols = [col for col in tree_features_df.columns 
                   if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
    
    X_tree = tree_features_df[feature_cols].values
    
    # For tree models, we need to create our own targets (next-period Close)
    y_tree = tree_features_df['Close'].shift(-1).dropna().values
    
    # Align X and y (remove last row from X since y is shifted)
    X_tree = X_tree[:-1]
    
    logger.info("Tree data prepared: X=%s, y=%s, features=%d", 
               X_tree.shape, y_tree.shape, len(feature_cols))
    
    return X_tree, y_tree, feature_cols


def train_lightgbm(X, y, feature_names):
    """Train and evaluate LightGBM model."""
    logger.info("Starting LightGBM training and evaluation...")
    
    # Model parameters
    lgb_params = {
        'n_estimators': 300,
        'learning_rate': 0.05,
        'max_depth': 12,
        'num_leaves': 31,
        'random_state': 42
    }
    
    # Run cross-validation
    cv_results = walk_forward_cv(
        X=X, y=y,
        model_class=LightGBMRegressor,
        model_params=lgb_params,
        min_train_size=252,
        test_size=21,
        step_size=21,
        max_folds=15
    )
    
    # Summarize results
    summary = summarize_cv_results(cv_results, "LightGBM")
    
    # Save results
    save_cv_results(cv_results, summary, "LightGBM")
    
    # Generate plots
    plot_cv_performance(cv_results, "LightGBM")
    
    # Train final model for feature importance
    model = LightGBMRegressor(**lgb_params)
    model.fit(X, y)
    
    if model.get_feature_importance() is not None:
        plot_feature_importance(
            model.get_feature_importance(),
            feature_names,
            "LightGBM"
        )
    
    # Generate performance report
    generate_performance_report("LightGBM", cv_results, summary)
    
    return cv_results, summary, model


def train_xgboost(X, y, feature_names):
    """Train and evaluate XGBoost model."""
    logger.info("Starting XGBoost training and evaluation...")
    
    # Model parameters
    xgb_params = {
        'n_estimators': 300,
        'learning_rate': 0.05,
        'max_depth': 12,
        'random_state': 42
    }
    
    # Run cross-validation
    cv_results = walk_forward_cv(
        X=X, y=y,
        model_class=XGBoostRegressor,
        model_params=xgb_params,
        min_train_size=252,
        test_size=21,
        step_size=21,
        max_folds=15
    )
    
    # Summarize results
    summary = summarize_cv_results(cv_results, "XGBoost")
    
    # Save results
    save_cv_results(cv_results, summary, "XGBoost")
    
    # Generate plots
    plot_cv_performance(cv_results, "XGBoost")
    
    # Train final model for feature importance
    model = XGBoostRegressor(**xgb_params)
    model.fit(X, y)
    
    if model.get_feature_importance() is not None:
        plot_feature_importance(
            model.get_feature_importance(),
            feature_names,
            "XGBoost"
        )
    
    # Generate performance report
    generate_performance_report("XGBoost", cv_results, summary)
    
    return cv_results, summary, model


def train_lstm(X_windows, y_targets):
    """Train and evaluate LSTM model."""
    logger.info("Starting LSTM training and evaluation...")
    
    # Model parameters
    lstm_params = {
        'hidden_size': 64,  # Smaller to fit RTX 3080 Ti
        'num_layers': 2,
        'dropout': 0.2,
        'learning_rate': 0.001,
        'batch_size': 64,
        'epochs': 150,  # Reduced for faster training
        'patience': 10,
        'device': 'auto'
    }
    
    # Run cross-validation (fewer folds due to training time)
    cv_results = walk_forward_cv(
        X=X_windows, y=y_targets,
        model_class=LSTMRegressor,
        model_params=lstm_params,
        min_train_size=252,
        test_size=21,
        step_size=21,
        max_folds=10  # Reduced for computation time
    )
    
    # Summarize results
    summary = summarize_cv_results(cv_results, "LSTM")
    
    # Save results
    save_cv_results(cv_results, summary, "LSTM")
    
    # Generate plots
    plot_cv_performance(cv_results, "LSTM")
    
    # Train final model for demonstration
    model = LSTMRegressor(**lstm_params)
    train_size = min(1000, len(X_windows) - 100)  # Limit size for speed
    model.fit(X_windows[:train_size], y_targets[:train_size])
    
    # Generate performance report
    generate_performance_report("LSTM", cv_results, summary)
    
    return cv_results, summary, model


def main():
    """Main training pipeline for Phase 3."""
    logger.info("Starting Phase 3: Baseline Model Training")
    
    try:
        # Load data
        tree_features_df, X_windows, y_targets = load_data()
        
        # Prepare tree data
        X_tree, y_tree, feature_names = prepare_tree_data(tree_features_df)
        
        # Storage for all results
        all_cv_results = {}
        all_summaries = {}
        all_models = {}
        
        # Train LightGBM
        lgb_cv, lgb_summary, lgb_model = train_lightgbm(X_tree, y_tree, feature_names)
        all_cv_results['LightGBM'] = lgb_cv
        all_summaries['LightGBM'] = lgb_summary
        all_models['LightGBM'] = lgb_model
        
        # Train XGBoost
        xgb_cv, xgb_summary, xgb_model = train_xgboost(X_tree, y_tree, feature_names)
        all_cv_results['XGBoost'] = xgb_cv
        all_summaries['XGBoost'] = xgb_summary
        all_models['XGBoost'] = xgb_model
        
        # Train LSTM
        lstm_cv, lstm_summary, lstm_model = train_lstm(X_windows, y_targets)
        all_cv_results['LSTM'] = lstm_cv
        all_summaries['LSTM'] = lstm_summary
        all_models['LSTM'] = lstm_model
        
        # Generate comparison plots
        logger.info("Generating model comparison plots...")
        plot_model_comparison(
            model_summaries=[lgb_summary, xgb_summary, lstm_summary],
            model_names=['LightGBM', 'XGBoost', 'LSTM']
        )
        
        # Create overall summary
        create_phase3_summary(all_summaries)
        
        logger.info("Phase 3 training complete!")
        print_phase3_results(all_summaries)
        
    except Exception as e:
        logger.error("Error in Phase 3 training: %s", str(e))
        raise


def create_phase3_summary(all_summaries):
    """Create comprehensive Phase 3 summary."""
    summary_rows = []
    
    for model_name, summary in all_summaries.items():
        summary_rows.append({
            'Model': model_name,
            'MAE_Mean': summary.get('mae_mean', 0),
            'MAE_Std': summary.get('mae_std', 0),
            'RMSE_Mean': summary.get('rmse_mean', 0),
            'RMSE_Std': summary.get('rmse_std', 0),
            'R2_Mean': summary.get('r2_mean', 0),
            'R2_Std': summary.get('r2_std', 0),
            'Dir_Acc_Mean': summary.get('directional_accuracy_mean', 0) * 100,
            'Dir_Acc_Std': summary.get('directional_accuracy_std', 0) * 100,
            'Meets_60pct_Target': summary.get('directional_accuracy_mean', 0) >= 0.60
        })
    
    summary_df = pd.DataFrame(summary_rows)
    
    os.makedirs("reports/summaries", exist_ok=True)
    summary_df.to_csv("reports/summaries/phase3_model_comparison.csv", index=False)
    
    logger.info("Saved Phase 3 summary to reports/summaries/phase3_model_comparison.csv")


def print_phase3_results(all_summaries):
    """Print Phase 3 results to console."""
    print("\n" + "="*60)
    print("PHASE 3 BASELINE MODEL RESULTS")
    print("="*60)
    
    for model_name, summary in all_summaries.items():
        dir_acc = summary.get('directional_accuracy_mean', 0) * 100
        meets_target = "✓ PASS" if dir_acc >= 60 else "✗ FAIL"
        
        print(f"\n{model_name}:")
        print(f"  Directional Accuracy: {dir_acc:.2f}% ({meets_target})")
        print(f"  MAE: {summary.get('mae_mean', 0):.4f} ± {summary.get('mae_std', 0):.4f}")
        print(f"  R²: {summary.get('r2_mean', 0):.4f} ± {summary.get('r2_std', 0):.4f}")
    
    print(f"\nAll results saved to:")
    print(f"  - logs/models.log")
    print(f"  - reports/figures/")
    print(f"  - reports/summaries/")
    print("="*60)


if __name__ == "__main__":
    main()