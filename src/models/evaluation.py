"""
evaluation.py

Responsibilities:
  - Generate evaluation plots and visualizations.
  - Calculate comprehensive model performance metrics.
  - Save results to reports/figures/ following project structure.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.utils.io_helpers import setup_logging

# Initialize logger
logger = logging.getLogger(__name__)
setup_logging()
fh = logging.FileHandler("logs/models.log")
fh.setLevel(logging.INFO)
logger.addHandler(fh)

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_predictions_vs_actual(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    model_name: str,
    save_path: Optional[str] = None
) -> None:
    """
    Create scatter plot of predictions vs actual values.
    
    Args:
        y_true: Actual target values.
        y_pred: Model predictions.
        model_name: Name of the model for title.
        save_path: Path to save the plot. If None, uses default naming.
    """
    if save_path is None:
        os.makedirs("reports/figures", exist_ok=True)
        save_path = f"reports/figures/predictions_vs_actual_{model_name.lower().replace(' ', '_')}.png"
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create scatter plot
    ax.scatter(y_true, y_pred, alpha=0.6, s=20)
    
    # Add perfect prediction line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Calculate and display metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Add metrics to plot
    metrics_text = f'MAE: {mae:.4f}\nMSE: {mse:.4f}\nR²: {r2:.4f}'
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Actual Values', fontsize=12)
    ax.set_ylabel('Predicted Values', fontsize=12)
    ax.set_title(f'{model_name}: Predictions vs Actual Values', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Saved predictions vs actual plot to %s", save_path)


def plot_residuals(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    model_name: str,
    save_path: Optional[str] = None
) -> None:
    """
    Create residual analysis plots.
    
    Args:
        y_true: Actual target values.
        y_pred: Model predictions.
        model_name: Name of the model for title.
        save_path: Path to save the plot.
    """
    if save_path is None:
        os.makedirs("reports/figures", exist_ok=True)
        save_path = f"reports/figures/residuals_{model_name.lower().replace(' ', '_')}.png"
    
    residuals = y_true - y_pred
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Residuals vs Predictions
    ax1.scatter(y_pred, residuals, alpha=0.6, s=20)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.8)
    ax1.set_xlabel('Predicted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Predicted Values')
    ax1.grid(True, alpha=0.3)
    
    # 2. Residuals histogram
    ax2.hist(residuals, bins=50, alpha=0.7, density=True)
    ax2.axvline(x=0, color='r', linestyle='--', alpha=0.8)
    ax2.set_xlabel('Residuals')
    ax2.set_ylabel('Density')
    ax2.set_title('Distribution of Residuals')
    ax2.grid(True, alpha=0.3)
    
    # 3. Q-Q plot for normality check
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot (Normality Check)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Residuals over time (if we have temporal ordering)
    ax4.plot(residuals, alpha=0.7)
    ax4.axhline(y=0, color='r', linestyle='--', alpha=0.8)
    ax4.set_xlabel('Time Order')
    ax4.set_ylabel('Residuals')
    ax4.set_title('Residuals Over Time')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'{model_name}: Residual Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Saved residual analysis plot to %s", save_path)


def plot_cv_performance(
    cv_results: Dict[str, List[float]], 
    model_name: str,
    save_path: Optional[str] = None
) -> None:
    """
    Plot cross-validation performance across folds.
    
    Args:
        cv_results: Results from walk_forward_cv.
        model_name: Name of the model.
        save_path: Path to save the plot.
    """
    if save_path is None:
        os.makedirs("reports/figures", exist_ok=True)
        save_path = f"reports/figures/cv_performance_{model_name.lower().replace(' ', '_')}.png"
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    folds = cv_results['fold']
    
    # 1. MAE across folds
    ax1.plot(folds, cv_results['mae'], marker='o', linewidth=2, markersize=6)
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('MAE')
    ax1.set_title('Mean Absolute Error Across Folds')
    ax1.grid(True, alpha=0.3)
    
    # 2. Directional accuracy across folds
    dir_acc_pct = [acc * 100 for acc in cv_results['directional_accuracy']]
    ax2.plot(folds, dir_acc_pct, marker='o', linewidth=2, markersize=6, color='green')
    ax2.axhline(y=65, color='r', linestyle='--', alpha=0.8, label='Target (65%)')
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('Directional Accuracy (%)')
    ax2.set_title('Directional Accuracy Across Folds')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. R² across folds
    ax3.plot(folds, cv_results['r2'], marker='o', linewidth=2, markersize=6, color='purple')
    ax3.set_xlabel('Fold')
    ax3.set_ylabel('R² Score')
    ax3.set_title('R² Score Across Folds')
    ax3.grid(True, alpha=0.3)
    
    # 4. Training set size across folds
    train_sizes = [end - start for start, end in zip(cv_results['train_start'], cv_results['train_end'])]
    ax4.plot(folds, train_sizes, marker='o', linewidth=2, markersize=6, color='orange')
    ax4.set_xlabel('Fold')
    ax4.set_ylabel('Training Set Size')
    ax4.set_title('Training Set Size Across Folds')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'{model_name}: Cross-Validation Performance', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Saved CV performance plot to %s", save_path)


def plot_model_comparison(
    model_summaries: List[Dict[str, float]], 
    model_names: List[str],
    save_path: Optional[str] = None
) -> None:
    """
    Create comparison plots across multiple models.
    
    Args:
        model_summaries: List of CV summary dictionaries from each model.
        model_names: List of model names.
        save_path: Path to save the plot.
    """
    if save_path is None:
        os.makedirs("reports/figures", exist_ok=True)
        save_path = "reports/figures/model_comparison.png"
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Extract metrics for plotting
    mae_means = [summary.get('mae_mean', 0) for summary in model_summaries]
    mae_stds = [summary.get('mae_std', 0) for summary in model_summaries]
    
    dir_acc_means = [summary.get('directional_accuracy_mean', 0) * 100 for summary in model_summaries]
    dir_acc_stds = [summary.get('directional_accuracy_std', 0) * 100 for summary in model_summaries]
    
    r2_means = [summary.get('r2_mean', 0) for summary in model_summaries]
    r2_stds = [summary.get('r2_std', 0) for summary in model_summaries]
    
    rmse_means = [summary.get('rmse_mean', 0) for summary in model_summaries]
    rmse_stds = [summary.get('rmse_std', 0) for summary in model_summaries]
    
    x_pos = np.arange(len(model_names))
    
    # 1. MAE comparison
    ax1.bar(x_pos, mae_means, yerr=mae_stds, capsize=5, alpha=0.7)
    ax1.set_xlabel('Models')
    ax1.set_ylabel('MAE')
    ax1.set_title('Mean Absolute Error Comparison')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # 2. Directional accuracy comparison
    ax2.bar(x_pos, dir_acc_means, yerr=dir_acc_stds, capsize=5, alpha=0.7, color='green')
    ax2.axhline(y=65, color='r', linestyle='--', alpha=0.8, label='Target (65%)')
    ax2.set_xlabel('Models')
    ax2.set_ylabel('Directional Accuracy (%)')
    ax2.set_title('Directional Accuracy Comparison')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. R² comparison
    ax3.bar(x_pos, r2_means, yerr=r2_stds, capsize=5, alpha=0.7, color='purple')
    ax3.set_xlabel('Models')
    ax3.set_ylabel('R² Score')
    ax3.set_title('R² Score Comparison')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(model_names, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # 4. RMSE comparison
    ax4.bar(x_pos, rmse_means, yerr=rmse_stds, capsize=5, alpha=0.7, color='orange')
    ax4.set_xlabel('Models')
    ax4.set_ylabel('RMSE')
    ax4.set_title('Root Mean Square Error Comparison')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(model_names, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Saved model comparison plot to %s", save_path)


def plot_feature_importance(
    feature_importance: np.ndarray,
    feature_names: List[str],
    model_name: str,
    save_path: Optional[str] = None,
    top_n: int = 20
) -> None:
    """
    Plot feature importance for tree-based models.
    
    Args:
        feature_importance: Array of feature importance values.
        feature_names: List of feature names.
        model_name: Name of the model.
        save_path: Path to save the plot.
        top_n: Number of top features to display.
    """
    if save_path is None:
        os.makedirs("reports/figures", exist_ok=True)
        save_path = f"reports/figures/feature_importance_{model_name.lower().replace(' ', '_')}.png"
    
    # Get top N features
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=True).tail(top_n)
    
    fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.4)))
    
    ax.barh(range(len(importance_df)), importance_df['importance'], alpha=0.7)
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df['feature'])
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'{model_name}: Top {top_n} Feature Importance', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Saved feature importance plot to %s", save_path)


def calculate_comprehensive_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    prices_prev: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: Actual target values (next-period prices).
        y_pred: Model predictions (next-period prices).
        prices_prev: Previous period prices for directional accuracy calculation.
        
    Returns:
        Dictionary containing all evaluation metrics.
    """
    metrics = {}
    
    # Standard regression metrics
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['r2'] = r2_score(y_true, y_pred)
    
    # Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    metrics['mape'] = mape
    
    # Directional accuracy
    if prices_prev is not None:
        # Calculate actual direction (up/down from previous period)
        actual_direction = (y_true > prices_prev).astype(int)
        predicted_direction = (y_pred > prices_prev).astype(int)
        
        directional_accuracy = np.mean(actual_direction == predicted_direction)
        metrics['directional_accuracy'] = directional_accuracy
    else:
        # Fallback method using relative to mean
        y_true_mean = np.mean(y_true)
        y_pred_mean = np.mean(y_pred)
        actual_direction = (y_true > y_true_mean).astype(int)
        predicted_direction = (y_pred > y_pred_mean).astype(int)
        directional_accuracy = np.mean(actual_direction == predicted_direction)
        metrics['directional_accuracy'] = directional_accuracy
    
    # Maximum error
    metrics['max_error'] = np.max(np.abs(y_true - y_pred))
    
    # Explained variance
    metrics['explained_variance'] = 1 - np.var(y_true - y_pred) / np.var(y_true)
    
    return metrics


def generate_performance_report(
    model_name: str,
    cv_results: Dict[str, List[float]],
    cv_summary: Dict[str, float],
    save_path: Optional[str] = None
) -> str:
    """
    Generate a text-based performance report.
    
    Args:
        model_name: Name of the model.
        cv_results: Detailed CV results.
        cv_summary: Summary statistics.
        save_path: Path to save the report.
        
    Returns:
        Report text as string.
    """
    if save_path is None:
        os.makedirs("reports/summaries", exist_ok=True)
        save_path = f"reports/summaries/performance_report_{model_name.lower().replace(' ', '_')}.txt"
    
    report_lines = [
        f"Performance Report: {model_name}",
        "=" * 50,
        "",
        "Cross-Validation Summary:",
        f"  Number of folds: {len(cv_results['fold'])}",
        f"  Mean Absolute Error: {cv_summary.get('mae_mean', 0):.4f} ± {cv_summary.get('mae_std', 0):.4f}",
        f"  Root Mean Square Error: {cv_summary.get('rmse_mean', 0):.4f} ± {cv_summary.get('rmse_std', 0):.4f}",
        f"  R² Score: {cv_summary.get('r2_mean', 0):.4f} ± {cv_summary.get('r2_std', 0):.4f}",
        f"  Directional Accuracy: {cv_summary.get('directional_accuracy_mean', 0)*100:.2f}% ± {cv_summary.get('directional_accuracy_std', 0)*100:.2f}%",
        "",
        "Phase 3 Success Criteria:",
        f"  ≥60% Directional Accuracy: {'✓ PASS' if cv_summary.get('directional_accuracy_mean', 0) >= 0.60 else '✗ FAIL'}",
        f"  Current: {cv_summary.get('directional_accuracy_mean', 0)*100:.2f}%",
        "",
        "Fold-by-Fold Performance:",
    ]
    
    for i, fold in enumerate(cv_results['fold']):
        dir_acc = cv_results['directional_accuracy'][i] * 100
        mae = cv_results['mae'][i]
        report_lines.append(f"  Fold {fold}: MAE={mae:.4f}, Dir.Acc={dir_acc:.1f}%")
    
    report_text = "\n".join(report_lines)
    
    # Save to file
    with open(save_path, 'w') as f:
        f.write(report_text)
    
    logger.info("Saved performance report to %s", save_path)
    return report_text