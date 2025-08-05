"""
unified_pipeline.py

Complete unified pipeline for market forecasting project.
Executes Phases 1-3: Data Ingestion → Feature Engineering → Model Training
with configurable parameters and comprehensive wandb logging.

Usage:
    python unified_pipeline.py --config-env development
    python unified_pipeline.py --symbol TSLA --epochs 100
    python unified_pipeline.py --wandb-mode offline
"""

import sys
import argparse
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Core imports
import numpy as np
import pandas as pd
import wandb
from datetime import datetime

# Configuration
from config import ExperimentConfig, get_config_for_environment, validate_config

# Phase 1: Data Ingestion
from src.ingestion.fetch_market import fetch_and_save

# Phase 2: Feature Engineering  
from src.features.deep_features import create_windows

# Phase 3: Model Training
from src.models import (
    LightGBMRegressor, XGBoostRegressor, LSTMRegressor,
    walk_forward_cv, summarize_cv_results, save_cv_results,
    plot_cv_performance, plot_model_comparison, plot_feature_importance,
    generate_performance_report
)

# Utilities
from src.utils.io_helpers import setup_logging, read_csv, write_csv

# Setup robust plotting
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    # Robust plotting setup
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
    try:
        if 'seaborn-v0_8' in plt.style.available:
            plt.style.use('seaborn-v0_8')
        elif 'seaborn' in plt.style.available:
            plt.style.use('seaborn')
        else:
            plt.style.use('default')
        sns.set_palette("husl")
    except:
        plt.style.use('default')
except ImportError:
    pass  # Plotting will be handled by evaluation module


class UnifiedPipeline:
    """
    Unified pipeline orchestrating all phases of the market forecasting project.
    
    Handles data ingestion, feature engineering, model training, evaluation,
    and comprehensive logging to wandb with full reproducibility.
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: Complete experiment configuration
        """
        self.config = config
        self.logger = self._setup_logging()
        self.wandb_run: Optional[Any] = None  # wandb run object
        
        # Results storage
        self.results: Dict[str, Any] = {
            'phase1': {},
            'phase2': {},
            'phase3': {}
        }
        
        self.logger.info("="*80)
        self.logger.info("UNIFIED MARKET FORECASTING PIPELINE INITIALIZED")
        self.logger.info("="*80)
        self.logger.info(f"Symbol: {config.data.symbol}")
        self.logger.info(f"Date range: {config.data.start_date} to {config.data.end_date}")
        self.logger.info(f"Models: {config.models.models_to_train}")
        self.logger.info(f"Experiment: {config.wandb.experiment_name}")
    
    def _setup_logging(self) -> logging.Logger:
        """Configure logging for the pipeline."""
        setup_logging()
        logger = logging.getLogger(__name__)
        
        # File handler
        log_file = Path(self.config.system.logs_dir) / "unified_pipeline.log"
        fh = logging.FileHandler(log_file)
        fh.setLevel(getattr(logging, self.config.system.log_level))
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        return logger
    
    def initialize_wandb(self) -> Optional[Any]:
        """Initialize Weights & Biases tracking."""
        self.logger.info("Initializing wandb tracking...")
        
        try:
            self.wandb_run = wandb.init(
                project=self.config.wandb.project_name,
                entity=self.config.wandb.entity,
                name=self.config.wandb.experiment_name,
                tags=self.config.wandb.tags,
                notes=self.config.wandb.notes,
                config=self.config.to_dict(),
                save_code=self.config.wandb.save_code,
                dir=self.config.wandb.dir,
                mode=self.config.wandb.mode
            )
            
            self.logger.info(f"Wandb initialized: {self.wandb_run.url}")
            return self.wandb_run
            
        except Exception as e:
            self.logger.error(f"Failed to initialize wandb: {e}")
            # Continue without wandb if it fails
            self.wandb_run = None
            return None
    
    def run_phase1_data_ingestion(self) -> Dict[str, Any]:
        """
        Phase 1: Data Ingestion
        Download and validate market data.
        """
        self.logger.info("="*60)
        self.logger.info("PHASE 1: DATA INGESTION")
        self.logger.info("="*60)
        
        phase1_results = {}
        
        try:
            # Download data
            self.logger.info(f"Fetching {self.config.data.symbol} data...")
            fetch_and_save(
                symbol=self.config.data.symbol,
                start_date=self.config.data.start_date,
                end_date=self.config.data.end_date,
                output_dir=self.config.data.raw_data_dir
            )
            
            # Load and validate
            filename = f"{self.config.data.symbol}_{self.config.data.start_date}_{self.config.data.end_date}.csv"
            filepath = Path(self.config.data.raw_data_dir) / filename
            
            raw_data = read_csv(str(filepath))
            
            # Data quality metrics
            phase1_results = {
                'symbol': self.config.data.symbol,
                'rows': len(raw_data),
                'date_range': {
                    'start': raw_data.index.min().strftime('%Y-%m-%d'),
                    'end': raw_data.index.max().strftime('%Y-%m-%d')
                },
                'columns': list(raw_data.columns),
                'missing_values': raw_data.isnull().sum().to_dict(),
                'data_file': str(filepath)
            }
            
            self.logger.info(f"✅ Data loaded: {phase1_results['rows']} rows")
            self.logger.info(f"✅ Date range: {phase1_results['date_range']['start']} to {phase1_results['date_range']['end']}")
            
            # Log to wandb
            if self.wandb_run:
                wandb.log({
                    'phase1/data_rows': phase1_results['rows'],
                    'phase1/missing_values_total': sum(phase1_results['missing_values'].values())
                })
            
            self.results['phase1'] = phase1_results
            return phase1_results
            
        except Exception as e:
            self.logger.error(f"Phase 1 failed: {e}")
            raise
    
    def run_phase2_feature_engineering(self) -> Dict[str, Any]:
        """
        Phase 2: Feature Engineering
        Create tree features and deep learning windows.
        """
        self.logger.info("="*60)
        self.logger.info("PHASE 2: FEATURE ENGINEERING")
        self.logger.info("="*60)
        
        phase2_results = {}
        
        try:
            # Load raw data
            filename = f"{self.config.data.symbol}_{self.config.data.start_date}_{self.config.data.end_date}.csv"
            filepath = Path(self.config.data.raw_data_dir) / filename
            raw_data = read_csv(str(filepath))
            
            # Phase 2A: Tree Features
            self.logger.info("Creating tree features...")
            
            # Import the correct function
            from src.features.tree_features import build_tree_features
            
            tree_features = build_tree_features(raw_data)
            
            # Save tree features
            tree_features_path = "data/interim/tree_features.csv"
            Path("data/interim").mkdir(parents=True, exist_ok=True)
            write_csv(tree_features, tree_features_path, index_label="Date")
            
            # Phase 2B: Deep Features
            self.logger.info("Creating deep learning windows...")
            X_windows, y_targets = create_windows(
                raw_data,
                sequence_length=self.config.data.sequence_length,
                feature_cols=self.config.data.feature_cols
            )
            
            # Save deep features
            Path("data/processed").mkdir(parents=True, exist_ok=True)
            np.save("data/processed/X_windows.npy", X_windows)
            np.save("data/processed/y_targets_continuous.npy", y_targets)
            
            # Feature engineering metrics
            phase2_results = {
                'tree_features': {
                    'shape': tree_features.shape,
                    'columns': list(tree_features.columns),
                    'file': tree_features_path
                },
                'deep_features': {
                    'X_shape': X_windows.shape,
                    'y_shape': y_targets.shape,
                    'sequence_length': self.config.data.sequence_length,
                    'files': ["data/processed/X_windows.npy", "data/processed/y_targets_continuous.npy"]
                }
            }
            
            self.logger.info(f"✅ Tree features: {tree_features.shape}")
            self.logger.info(f"✅ Deep windows: {X_windows.shape}")
            self.logger.info(f"✅ Targets: {y_targets.shape}")
            
            # Log to wandb
            if self.wandb_run:
                wandb.log({
                    'phase2/tree_features_count': tree_features.shape[1],
                    'phase2/deep_windows_count': X_windows.shape[0],
                    'phase2/sequence_length': self.config.data.sequence_length
                })
            
            self.results['phase2'] = phase2_results
            return phase2_results
            
        except Exception as e:
            self.logger.error(f"Phase 2 failed: {e}")
            raise
    
    def run_phase3_model_training(self) -> Dict[str, Any]:
        """
        Phase 3: Model Training
        Train baseline models with walk-forward cross-validation.
        """
        self.logger.info("="*60)
        self.logger.info("PHASE 3: MODEL TRAINING")
        self.logger.info("="*60)
        
        phase3_results = {}
        
        try:
            # Load data
            tree_features = read_csv("data/interim/tree_features.csv")
            X_windows = np.load("data/processed/X_windows.npy")
            y_targets = np.load("data/processed/y_targets_continuous.npy")
            
            # Prepare tree data
            feature_cols = [col for col in tree_features.columns 
                           if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
            X_tree = tree_features[feature_cols].values
            y_tree = tree_features['Close'].shift(-1).dropna().values
            X_tree = X_tree[:-1]  # Align with shifted targets
            
            self.logger.info(f"Tree data: X={X_tree.shape}, y={y_tree.shape}")
            self.logger.info(f"Deep data: X={X_windows.shape}, y={y_targets.shape}")
            
            # Train models
            all_cv_results = {}
            all_summaries = {}
            all_models = {}
            
            # LightGBM
            if 'lightgbm' in self.config.models.models_to_train:
                self.logger.info("Training LightGBM...")
                lgb_cv, lgb_summary, lgb_model = self._train_lightgbm(X_tree, y_tree, feature_cols)
                all_cv_results['LightGBM'] = lgb_cv
                all_summaries['LightGBM'] = lgb_summary
                all_models['LightGBM'] = lgb_model
            
            # XGBoost
            if 'xgboost' in self.config.models.models_to_train:
                self.logger.info("Training XGBoost...")
                xgb_cv, xgb_summary, xgb_model = self._train_xgboost(X_tree, y_tree, feature_cols)
                all_cv_results['XGBoost'] = xgb_cv
                all_summaries['XGBoost'] = xgb_summary
                all_models['XGBoost'] = xgb_model
            
            # LSTM
            if 'lstm' in self.config.models.models_to_train:
                self.logger.info("Training LSTM...")
                lstm_cv, lstm_summary, lstm_model = self._train_lstm(X_windows, y_targets)
                all_cv_results['LSTM'] = lstm_cv
                all_summaries['LSTM'] = lstm_summary
                all_models['LSTM'] = lstm_model
            
            # Generate comparison plots
            if len(all_summaries) > 1:
                self.logger.info("Generating model comparison plots...")
                plot_model_comparison(
                    model_summaries=list(all_summaries.values()),
                    model_names=list(all_summaries.keys())
                )
            
            # Phase 3 results
            phase3_results = {
                'cv_results': all_cv_results,
                'summaries': all_summaries,
                'models_trained': list(all_summaries.keys())
            }
            
            # Log best model results
            best_model = max(all_summaries.keys(), 
                           key=lambda k: all_summaries[k].get('directional_accuracy_mean', 0))
            best_accuracy = all_summaries[best_model]['directional_accuracy_mean'] * 100
            
            self.logger.info(f"✅ Best model: {best_model} ({best_accuracy:.2f}% accuracy)")
            
            # Log to wandb
            if self.wandb_run:
                for model_name, summary in all_summaries.items():
                    wandb.log({
                        f'phase3/{model_name.lower()}/directional_accuracy': summary.get('directional_accuracy_mean', 0) * 100,
                        f'phase3/{model_name.lower()}/mae': summary.get('mae_mean', 0),
                        f'phase3/{model_name.lower()}/rmse': summary.get('rmse_mean', 0),
                        f'phase3/{model_name.lower()}/r2': summary.get('r2_mean', 0)
                    })
                
                wandb.log({
                    'phase3/best_model': best_model,
                    'phase3/best_accuracy': best_accuracy,
                    'phase3/target_met': best_accuracy >= 65.0
                })
            
            self.results['phase3'] = phase3_results
            return phase3_results
            
        except Exception as e:
            self.logger.error(f"Phase 3 failed: {e}")
            raise
    
    def _train_lightgbm(self, X: np.ndarray, y: np.ndarray, feature_names: list) -> Tuple[Dict, Dict, Any]:
        """Train LightGBM model with cross-validation."""
        cv_results = walk_forward_cv(
            X=X, y=y,
            model_class=LightGBMRegressor,
            model_params=self.config.models.lgb_params,
            min_train_size=self.config.models.min_train_size,
            test_size=self.config.models.test_size,
            step_size=self.config.models.step_size,
            max_folds=self.config.models.max_folds
        )
        
        summary = summarize_cv_results(cv_results, "LightGBM")
        save_cv_results(cv_results, summary, "LightGBM")
        plot_cv_performance(cv_results, "LightGBM")
        
        # Train final model
        model = LightGBMRegressor(**self.config.models.lgb_params)
        model.fit(X, y)
        
        if model.get_feature_importance() is not None:
            plot_feature_importance(model.get_feature_importance(), feature_names, "LightGBM")
        
        generate_performance_report("LightGBM", cv_results, summary)
        
        return cv_results, summary, model
    
    def _train_xgboost(self, X: np.ndarray, y: np.ndarray, feature_names: list) -> Tuple[Dict, Dict, Any]:
        """Train XGBoost model with cross-validation."""
        cv_results = walk_forward_cv(
            X=X, y=y,
            model_class=XGBoostRegressor,
            model_params=self.config.models.xgb_params,
            min_train_size=self.config.models.min_train_size,
            test_size=self.config.models.test_size,
            step_size=self.config.models.step_size,
            max_folds=self.config.models.max_folds
        )
        
        summary = summarize_cv_results(cv_results, "XGBoost")
        save_cv_results(cv_results, summary, "XGBoost")
        plot_cv_performance(cv_results, "XGBoost")
        
        # Train final model
        model = XGBoostRegressor(**self.config.models.xgb_params)
        model.fit(X, y)
        
        if model.get_feature_importance() is not None:
            plot_feature_importance(model.get_feature_importance(), feature_names, "XGBoost")
        
        generate_performance_report("XGBoost", cv_results, summary)
        
        return cv_results, summary, model
    
    def _train_lstm(self, X: np.ndarray, y: np.ndarray) -> Tuple[Dict, Dict, Any]:
        """Train LSTM model with cross-validation."""
        cv_results = walk_forward_cv(
            X=X, y=y,
            model_class=LSTMRegressor,
            model_params=self.config.models.lstm_params,
            min_train_size=self.config.models.min_train_size,
            test_size=self.config.models.test_size,
            step_size=self.config.models.step_size,
            max_folds=min(10, self.config.models.max_folds)  # Fewer folds for LSTM
        )
        
        summary = summarize_cv_results(cv_results, "LSTM")
        save_cv_results(cv_results, summary, "LSTM")
        plot_cv_performance(cv_results, "LSTM")
        
        # Train final model
        model = LSTMRegressor(**self.config.models.lstm_params)
        train_size = min(1000, len(X) - 100)  # Limit size for memory efficiency
        model.fit(X[:train_size], y[:train_size])
        
        generate_performance_report("LSTM", cv_results, summary)
        
        return cv_results, summary, model
    
    def generate_final_report(self, pipeline_start: datetime = None) -> Dict[str, Any]:
        """Generate comprehensive final report."""
        self.logger.info("="*60)
        self.logger.info("GENERATING FINAL REPORT")
        self.logger.info("="*60)
        
        report = {
            'experiment_config': self.config.to_dict(),
            'pipeline_results': self.results,
            'timestamp': datetime.now().isoformat(),
            'success': True
        }
        
        # Performance summary
        if 'phase3' in self.results and 'summaries' in self.results['phase3']:
            summaries = self.results['phase3']['summaries']
            
            # Best model identification
            best_model = max(summaries.keys(), 
                           key=lambda k: summaries[k].get('directional_accuracy_mean', 0))
            best_accuracy = summaries[best_model]['directional_accuracy_mean'] * 100
            
            report['performance_summary'] = {
                'best_model': best_model,
                'best_accuracy': best_accuracy,
                'target_achieved': best_accuracy >= 65.0,
                'all_models': {
                    name: {
                        'directional_accuracy': summary.get('directional_accuracy_mean', 0) * 100,
                        'mae': summary.get('mae_mean', 0),
                        'rmse': summary.get('rmse_mean', 0),
                        'r2': summary.get('r2_mean', 0)
                    }
                    for name, summary in summaries.items()
                }
            }
            
            self.logger.info("="*60)
            self.logger.info("FINAL RESULTS SUMMARY")
            self.logger.info("="*60)
            for model_name, summary in summaries.items():
                accuracy = summary.get('directional_accuracy_mean', 0) * 100
                mae = summary.get('mae_mean', 0)
                status = "✅ PASS" if accuracy >= 65.0 else "❌ FAIL"
                self.logger.info(f"{model_name:12s}: {accuracy:6.2f}% accuracy, MAE: {mae:6.3f} {status}")
            
            self.logger.info("="*60)
            self.logger.info(f"🏆 BEST MODEL: {best_model} ({best_accuracy:.2f}%)")
            self.logger.info(f"🎯 TARGET MET: {'YES' if best_accuracy >= 65.0 else 'NO'}")
            self.logger.info("="*60)
        
        # Log final results to wandb
        if self.wandb_run:
            # Calculate experiment duration safely
            try:
                if pipeline_start:
                    duration_minutes = (datetime.now() - pipeline_start).total_seconds() / 60
                elif hasattr(self.wandb_run, 'start_time'):
                    if isinstance(self.wandb_run.start_time, float):
                        # start_time is a timestamp, convert to datetime
                        start_time = datetime.fromtimestamp(self.wandb_run.start_time)
                        duration_minutes = (datetime.now() - start_time).total_seconds() / 60
                    else:
                        # start_time is already a datetime
                        duration_minutes = (datetime.now() - self.wandb_run.start_time).total_seconds() / 60
                else:
                    duration_minutes = 5.0  # Default fallback
            except Exception:
                # Fallback duration calculation
                duration_minutes = 5.0  # Default fallback
            
            wandb.log({
                'final/pipeline_success': True,
                'final/experiment_duration': duration_minutes
            })
            
            # Upload plots as artifacts
            self._upload_wandb_artifacts()
        
        return report
    
    def _upload_wandb_artifacts(self):
        """Upload generated plots and reports to wandb."""
        if not self.wandb_run:
            return
        
        try:
            # Upload plots from reports/figures/
            figures_dir = Path("reports/figures")
            if figures_dir.exists():
                for plot_file in figures_dir.glob("*.png"):
                    wandb.log({f"plots/{plot_file.stem}": wandb.Image(str(plot_file))})
            
            # Upload model summaries
            summaries_dir = Path("reports/summaries")
            if summaries_dir.exists():
                for summary_file in summaries_dir.glob("*.csv"):
                    artifact = wandb.Artifact(f"summary_{summary_file.stem}", type="results")
                    artifact.add_file(str(summary_file))
                    self.wandb_run.log_artifact(artifact)
            
            self.logger.info("✅ Artifacts uploaded to wandb")
            
        except Exception as e:
            self.logger.warning(f"Failed to upload wandb artifacts: {e}")
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """
        Execute the complete pipeline: Phases 1-3.
        
        Returns:
            Complete pipeline results
        """
        pipeline_start = datetime.now()
        
        try:
            # Initialize wandb
            self.initialize_wandb()
            
            # Execute phases
            self.run_phase1_data_ingestion()
            self.run_phase2_feature_engineering()  
            self.run_phase3_model_training()
            
            # Generate final report
            final_report = self.generate_final_report(pipeline_start)
            
            pipeline_duration = (datetime.now() - pipeline_start).total_seconds() / 60
            self.logger.info(f"🎉 PIPELINE COMPLETED SUCCESSFULLY in {pipeline_duration:.1f} minutes")
            
            return final_report
            
        except Exception as e:
            self.logger.error(f"💥 PIPELINE FAILED: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            if self.wandb_run:
                wandb.log({"final/pipeline_success": False, "final/error": str(e)})
            
            raise
        
        finally:
            # Clean up wandb
            if self.wandb_run:
                self.wandb_run.finish()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified Market Forecasting Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument('--config-env', type=str, default='development',
                       choices=['development', 'production', 'testing'],
                       help='Configuration environment')
    
    # Data parameters
    parser.add_argument('--symbol', type=str, help='Stock symbol to analyze')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--sequence-length', type=int, help='LSTM sequence length')
    
    # Model parameters
    parser.add_argument('--models', nargs='+', 
                       choices=['lightgbm', 'xgboost', 'lstm'],
                       help='Models to train')
    parser.add_argument('--max-folds', type=int, help='Maximum CV folds')
    parser.add_argument('--epochs', type=int, help='LSTM epochs')
    parser.add_argument('--batch-size', type=int, help='LSTM batch size')
    parser.add_argument('--learning-rate', type=float, help='LSTM learning rate')
    
    # Wandb parameters
    parser.add_argument('--wandb-project', type=str, help='Wandb project name')
    parser.add_argument('--wandb-mode', type=str, 
                       choices=['online', 'offline', 'disabled'],
                       help='Wandb mode')
    parser.add_argument('--experiment-name', type=str, help='Experiment name')
    parser.add_argument('--tags', nargs='+', help='Wandb tags')
    
    return parser.parse_args()


def create_config_from_args(args) -> ExperimentConfig:
    """Create configuration from command line arguments."""
    # Start with environment-specific config
    config = get_config_for_environment(args.config_env)
    
    # Override with command line arguments
    if args.symbol:
        config.data.symbol = args.symbol
    if args.start_date:
        config.data.start_date = args.start_date
    if args.end_date:
        config.data.end_date = args.end_date
    if args.sequence_length:
        config.data.sequence_length = args.sequence_length
    
    if args.models:
        config.models.models_to_train = args.models
    if args.max_folds:
        config.models.max_folds = args.max_folds
    if args.epochs:
        config.models.lstm_params['epochs'] = args.epochs
    if args.batch_size:
        config.models.lstm_params['batch_size'] = args.batch_size
    if args.learning_rate:
        config.models.lstm_params['learning_rate'] = args.learning_rate
    
    if args.wandb_project:
        config.wandb.project_name = args.wandb_project
    if args.wandb_mode:
        config.wandb.mode = args.wandb_mode
    if args.experiment_name:
        config.wandb.experiment_name = args.experiment_name
    if args.tags:
        config.wandb.tags = args.tags
    
    return config


def main():
    """Main entry point."""
    print("🚀 Starting Unified Market Forecasting Pipeline")
    print("="*80)
    
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Create configuration
        config = create_config_from_args(args)
        
        # Validate configuration
        if not validate_config(config):
            print("❌ Configuration validation failed!")
            sys.exit(1)
        
        # Run pipeline
        pipeline = UnifiedPipeline(config)
        results = pipeline.run_complete_pipeline()
        
        print("="*80)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        # Print key results
        if 'performance_summary' in results:
            perf = results['performance_summary']
            print(f"🏆 Best Model: {perf['best_model']}")
            print(f"📈 Best Accuracy: {perf['best_accuracy']:.2f}%")
            print(f"🎯 Target (≥65%) Met: {'YES' if perf['target_achieved'] else 'NO'}")
        
        print("="*80)
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️  Pipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())