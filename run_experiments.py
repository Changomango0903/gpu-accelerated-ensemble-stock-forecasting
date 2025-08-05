"""
run_experiments.py

Easy-to-use experiment runner with predefined configurations.
Perfect for running different experiment setups without complex command lines.

Usage Examples:
    python run_experiments.py --preset quick-test
    python run_experiments.py --preset full-aapl
    python run_experiments.py --preset gpu-benchmark
    python run_experiments.py --custom
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import ExperimentConfig, DataConfig, ModelConfig, WandbConfig
from unified_pipeline import UnifiedPipeline


# Predefined experiment presets
EXPERIMENT_PRESETS = {
    'quick-test': {
        'description': 'Quick test run for development (5 min)',
        'config_overrides': {
            'data': {
                'symbol': 'AAPL',
                'start_date': '2020-01-01',
                'end_date': '2025-01-01',
                'sequence_length': 30
            },
            'models': {
                'models_to_train': ['lightgbm'],
                'max_folds': 5,
                'lgb_params': {'n_estimators': 100, 'learning_rate': 0.1},
                'lstm_params': {'epochs': 10, 'batch_size': 64}
            },
            'wandb': {
                'mode': 'offline',
                'tags': ['quick-test', 'development']
            }
        }
    },
    
    'full-aapl': {
        'description': 'Complete AAPL analysis with all models (30 min)',
        'config_overrides': {
            'data': {
                'symbol': 'AAPL',
                'start_date': '2019-01-01',
                'end_date': '2024-12-31',
                'sequence_length': 60
            },
            'models': {
                'models_to_train': ['lightgbm', 'xgboost', 'lstm'],
                'max_folds': 15,
                'lstm_params': {'epochs': 100, 'batch_size': 32, 'patience': 15}
            },
            'wandb': {
                'mode': 'online',
                'tags': ['full-analysis', 'aapl', 'production']
            }
        }
    },
    
    'gpu-benchmark': {
        'description': 'GPU performance benchmark with LSTM focus (20 min)',
        'config_overrides': {
            'data': {
                'symbol': 'TSLA',
                'start_date': '2020-01-01',
                'end_date': '2024-12-31',
                'sequence_length': 120
            },
            'models': {
                'models_to_train': ['lstm'],
                'max_folds': 10,
                'lstm_params': {
                    'hidden_size': 128,
                    'num_layers': 3,
                    'epochs': 200,
                    'batch_size': 64,
                    'learning_rate': 0.0005,
                    'patience': 20
                }
            },
            'wandb': {
                'mode': 'online',
                'tags': ['gpu-benchmark', 'lstm', 'performance']
            }
        }
    },
    
    'multi-asset': {
        'description': 'Multi-asset comparison (run multiple times)',
        'symbols': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA'],
        'config_overrides': {
            'data': {
                'start_date': '2020-01-01',
                'end_date': '2024-12-31',
                'sequence_length': 60
            },
            'models': {
                'models_to_train': ['lightgbm', 'xgboost'],
                'max_folds': 10
            },
            'wandb': {
                'mode': 'online',
                'tags': ['multi-asset', 'comparison']
            }
        }
    },
    
    'hyperparameter-sweep': {
        'description': 'Hyperparameter optimization setup',
        'config_overrides': {
            'data': {
                'symbol': 'SPY',
                'start_date': '2021-01-01',
                'end_date': '2024-12-31'
            },
            'models': {
                'models_to_train': ['lightgbm'],
                'max_folds': 8
            },
            'wandb': {
                'mode': 'online',
                'tags': ['hyperparameter-sweep', 'optimization']
            }
        }
    }
}


def create_config_from_preset(preset_name: str, symbol: str = None) -> ExperimentConfig:
    """Create configuration from preset."""
    if preset_name not in EXPERIMENT_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}")
    
    preset = EXPERIMENT_PRESETS[preset_name]
    config = ExperimentConfig()
    
    # Apply overrides
    overrides = preset['config_overrides']
    
    # Data config overrides
    if 'data' in overrides:
        for key, value in overrides['data'].items():
            setattr(config.data, key, value)
    
    # Models config overrides
    if 'models' in overrides:
        for key, value in overrides['models'].items():
            if key in ['lgb_params', 'xgb_params', 'lstm_params']:
                getattr(config.models, key).update(value)
            else:
                setattr(config.models, key, value)
    
    # Wandb config overrides
    if 'wandb' in overrides:
        for key, value in overrides['wandb'].items():
            setattr(config.wandb, key, value)
    
    # Override symbol if provided
    if symbol:
        config.data.symbol = symbol
    
    # Set experiment name
    symbol_suffix = f"_{symbol or config.data.symbol}"
    config.wandb.experiment_name = f"{preset_name}{symbol_suffix}"
    
    return config


def run_single_experiment(preset_name: str, symbol: str = None) -> Dict[str, Any]:
    """Run a single experiment with the given preset."""
    print(f"\n🚀 Starting experiment: {preset_name}")
    if symbol:
        print(f"📈 Symbol: {symbol}")
    
    preset_info = EXPERIMENT_PRESETS[preset_name]
    print(f"📝 Description: {preset_info['description']}")
    print("="*60)
    
    # Create config and run pipeline
    config = create_config_from_preset(preset_name, symbol)
    pipeline = UnifiedPipeline(config)
    results = pipeline.run_complete_pipeline()
    
    return results


def run_multi_asset_experiment(preset_name: str) -> Dict[str, Dict[str, Any]]:
    """Run experiment across multiple assets."""
    preset = EXPERIMENT_PRESETS[preset_name]
    symbols = preset.get('symbols', ['AAPL'])
    
    print(f"\n🎯 Running multi-asset experiment: {preset_name}")
    print(f"📊 Assets: {', '.join(symbols)}")
    print("="*60)
    
    all_results = {}
    
    for i, symbol in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] Processing {symbol}...")
        try:
            results = run_single_experiment(preset_name, symbol)
            all_results[symbol] = results
            
            # Print quick summary
            if 'performance_summary' in results:
                perf = results['performance_summary']
                print(f"✅ {symbol}: Best = {perf['best_model']} ({perf['best_accuracy']:.2f}%)")
            
        except Exception as e:
            print(f"❌ {symbol} failed: {e}")
            all_results[symbol] = {'error': str(e)}
    
    # Print final multi-asset summary
    print("\n" + "="*60)
    print("🏆 MULTI-ASSET RESULTS SUMMARY")
    print("="*60)
    
    for symbol, results in all_results.items():
        if 'error' in results:
            print(f"{symbol:8s}: ❌ FAILED - {results['error']}")
        elif 'performance_summary' in results:
            perf = results['performance_summary']
            status = "✅" if perf['target_achieved'] else "⚠️"
            print(f"{symbol:8s}: {status} {perf['best_model']:10s} {perf['best_accuracy']:6.2f}%")
    
    return all_results


def interactive_config() -> ExperimentConfig:
    """Create configuration interactively."""
    print("\n🔧 Interactive Configuration Setup")
    print("="*40)
    
    config = ExperimentConfig()
    
    # Data configuration
    symbol = input(f"Stock symbol [{config.data.symbol}]: ").strip() or config.data.symbol
    start_date = input(f"Start date [{config.data.start_date}]: ").strip() or config.data.start_date
    end_date = input(f"End date [{config.data.end_date}]: ").strip() or config.data.end_date
    
    config.data.symbol = symbol.upper()
    config.data.start_date = start_date
    config.data.end_date = end_date
    
    # Model selection
    print(f"\nAvailable models: {', '.join(['lightgbm', 'xgboost', 'lstm'])}")
    models_input = input("Models to train (space-separated) [lightgbm xgboost]: ").strip()
    if models_input:
        config.models.models_to_train = models_input.split()
    
    # LSTM parameters
    if 'lstm' in config.models.models_to_train:
        epochs = input(f"LSTM epochs [{config.models.lstm_params['epochs']}]: ").strip()
        if epochs.isdigit():
            config.models.lstm_params['epochs'] = int(epochs)
    
    # Wandb configuration
    wandb_mode = input("Wandb mode [online/offline/disabled] [online]: ").strip() or "online"
    config.wandb.mode = wandb_mode
    
    experiment_name = input("Experiment name (optional): ").strip()
    if experiment_name:
        config.wandb.experiment_name = experiment_name
    
    return config


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Easy Experiment Runner for Market Forecasting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--preset', type=str, choices=list(EXPERIMENT_PRESETS.keys()),
                       help='Run predefined experiment preset')
    parser.add_argument('--symbol', type=str, help='Override symbol for preset')
    parser.add_argument('--list-presets', action='store_true',
                       help='List available presets')
    parser.add_argument('--custom', action='store_true',
                       help='Interactive configuration setup')
    
    args = parser.parse_args()
    
    if args.list_presets:
        print("\n📋 Available Experiment Presets:")
        print("="*50)
        for name, preset in EXPERIMENT_PRESETS.items():
            print(f"{name:20s}: {preset['description']}")
        return 0
    
    try:
        if args.custom:
            # Interactive mode
            config = interactive_config()
            pipeline = UnifiedPipeline(config)
            pipeline.run_complete_pipeline()
            
        elif args.preset:
            # Preset mode
            preset = EXPERIMENT_PRESETS[args.preset]
            
            if 'symbols' in preset and not args.symbol:
                # Multi-asset experiment
                run_multi_asset_experiment(args.preset)
            else:
                # Single experiment
                run_single_experiment(args.preset, args.symbol)
        
        else:
            # Default: show help and run quick test
            parser.print_help()
            print(f"\n💡 Tip: Try 'python {sys.argv[0]} --preset quick-test' for a fast demo")
            return 1
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️  Experiment interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Experiment failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())