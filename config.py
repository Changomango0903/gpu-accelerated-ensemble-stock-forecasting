"""
Centralized configuration for the unified market forecasting pipeline.
All parameters for data ingestion, feature engineering, model training, and logging.
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class DataConfig:
    """Data ingestion and processing configuration."""
    # Phase 1: Data Ingestion
    symbol: str = "AAPL"
    start_date: str = "2019-01-01"
    end_date: str = "2024-12-31"
    raw_data_dir: str = "data/raw"
    
    # Phase 2: Feature Engineering
    sequence_length: int = 60  # Deep learning window size
    feature_cols: List[str] = field(default_factory=lambda: ["Open", "High", "Low", "Close", "Volume"])
    
    # Note: Technical indicators use hardcoded parameters in build_tree_features():
    # - SMA: [20, 50, 200]
    # - RSI: 14-day
    # - MACD: (12, 26, 9)
    # - Bollinger Bands: 20-day, 2σ
    # - ATR: 14-day


@dataclass
class ModelConfig:
    """Model training configuration."""
    # Cross-validation settings
    min_train_size: int = 252  # ~1 trading year
    test_size: int = 21        # ~1 trading month
    step_size: int = 21        # Walk-forward step
    max_folds: int = 15        # Maximum CV folds
    
    # LightGBM parameters
    lgb_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 300,
        'learning_rate': 0.05,
        'max_depth': 6,
        'num_leaves': 31,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42
    })
    
    # XGBoost parameters
    xgb_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 300,
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42
    })
    
    # LSTM parameters
    lstm_params: Dict[str, Any] = field(default_factory=lambda: {
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.2,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 50,
        'patience': 10,
        'device': 'auto'  # Will auto-detect CUDA
    })
    
    # Model selection
    models_to_train: List[str] = field(default_factory=lambda: ['lightgbm', 'xgboost', 'lstm'])


@dataclass
class WandbConfig:
    """Weights & Biases configuration."""
    # Project settings
    project_name: str = "market-forecaster"
    entity: Optional[str] = None  # Your wandb username/org
    experiment_name: Optional[str] = None  # Will auto-generate if None
    tags: List[str] = field(default_factory=lambda: ["baseline", "phase3"])
    notes: str = "Unified pipeline run with configurable parameters"
    
    # Logging settings
    log_model: bool = True      # Save model artifacts
    log_code: bool = True       # Save source code
    log_gradients: bool = False # For deep models only
    log_parameters: bool = True # Log hyperparameters
    
    # Advanced settings
    save_code: bool = True
    dir: str = "./wandb"
    mode: str = "online"  # "online", "offline", "disabled"


@dataclass
class SystemConfig:
    """System and infrastructure configuration."""
    # Directories
    project_root: str = "."
    logs_dir: str = "logs"
    reports_dir: str = "reports"
    models_dir: str = "models"
    
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    log_to_console: bool = True
    
    # Performance
    n_jobs: int = -1  # Use all CPU cores
    random_state: int = 42
    
    # GPU settings
    use_gpu: bool = True
    gpu_memory_fraction: float = 0.8  # For PyTorch memory management


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Create directories
        for dir_path in [self.system.logs_dir, self.system.reports_dir, self.system.models_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Set experiment name if not provided
        if self.wandb.experiment_name is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.wandb.experiment_name = f"unified_pipeline_{self.data.symbol}_{timestamp}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for wandb logging."""
        return {
            'data': self.data.__dict__,
            'models': self.models.__dict__,
            'wandb': self.wandb.__dict__,
            'system': self.system.__dict__
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create config from dictionary."""
        return cls(
            data=DataConfig(**config_dict.get('data', {})),
            models=ModelConfig(**config_dict.get('models', {})),
            wandb=WandbConfig(**config_dict.get('wandb', {})),
            system=SystemConfig(**config_dict.get('system', {}))
        )


# Default configuration instance
DEFAULT_CONFIG = ExperimentConfig()

# Environment-specific overrides
def get_config_for_environment(env: str = "development") -> ExperimentConfig:
    """Get configuration for specific environment."""
    config = ExperimentConfig()
    
    if env == "development":
        # Faster training for development
        config.models.max_folds = 5
        config.models.lstm_params['epochs'] = 20
        config.wandb.mode = "offline"
    
    elif env == "production":
        # Full training for production
        config.models.max_folds = 15
        config.models.lstm_params['epochs'] = 100
        config.wandb.mode = "online"
    
    elif env == "testing":
        # Minimal config for testing
        config.models.max_folds = 2
        config.models.lstm_params['epochs'] = 5
        config.wandb.mode = "disabled"
    
    return config


# Configuration validation
def validate_config(config: ExperimentConfig) -> bool:
    """Validate configuration parameters."""
    try:
        # Data validation
        assert config.data.sequence_length > 0, "Sequence length must be positive"
        assert len(config.data.feature_cols) > 0, "Must have at least one feature column"
        
        # Model validation
        assert config.models.min_train_size > config.data.sequence_length, \
            "Training size must be larger than sequence length"
        assert config.models.test_size > 0, "Test size must be positive"
        assert config.models.max_folds > 0, "Must have at least one CV fold"
        
        # LSTM validation
        assert config.models.lstm_params['hidden_size'] > 0, "Hidden size must be positive"
        assert config.models.lstm_params['epochs'] > 0, "Epochs must be positive"
        assert 0 < config.models.lstm_params['dropout'] < 1, "Dropout must be between 0 and 1"
        
        return True
        
    except AssertionError as e:
        print(f"Configuration validation error: {e}")
        return False


if __name__ == "__main__":
    # Test configuration
    config = ExperimentConfig()
    print("Default configuration created successfully!")
    print(f"Experiment name: {config.wandb.experiment_name}")
    print(f"Models to train: {config.models.models_to_train}")
    print(f"Validation passed: {validate_config(config)}")