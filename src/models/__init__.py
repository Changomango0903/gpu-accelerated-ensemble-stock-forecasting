# src/models/__init__.py

"""
models package

Provides all regression models and utilities for Phase 3 baseline evaluation.
"""

from src.models.base_model import BaseModel
from src.models.lightgbm_regressor import LightGBMRegressor
from src.models.xgboost_regressor import XGBoostRegressor
from src.models.lstm_regressor import LSTMRegressor
from src.models.cross_validation import (
    walk_forward_cv,
    expanding_window_cv,
    summarize_cv_results,
    save_cv_results
)
from src.models.evaluation import (
    plot_predictions_vs_actual,
    plot_residuals,
    plot_cv_performance,
    plot_model_comparison,
    plot_feature_importance,
    calculate_comprehensive_metrics,
    generate_performance_report
)

__all__ = [
    'BaseModel',
    'LightGBMRegressor',
    'XGBoostRegressor', 
    'LSTMRegressor',
    'walk_forward_cv',
    'expanding_window_cv',
    'summarize_cv_results',
    'save_cv_results',
    'plot_predictions_vs_actual',
    'plot_residuals',
    'plot_cv_performance',
    'plot_model_comparison',
    'plot_feature_importance',
    'calculate_comprehensive_metrics',
    'generate_performance_report'
]