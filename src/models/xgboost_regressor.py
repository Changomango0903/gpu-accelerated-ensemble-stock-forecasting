"""
xgboost_regressor.py

Responsibilities:
  - XGBoost regression model implementation.
  - Financial-optimized hyperparameters for price forecasting.
  - Feature importance and model persistence capabilities.
"""

import logging
import pickle
from typing import Optional
import numpy as np
import xgboost as xgb
from src.models.base_model import BaseModel
from src.utils.io_helpers import setup_logging

# Initialize logger
logger = logging.getLogger(__name__)
setup_logging()
fh = logging.FileHandler("logs/models.log")
fh.setLevel(logging.INFO)
logger.addHandler(fh)


class XGBoostRegressor(BaseModel):
    """
    XGBoost regression model for price forecasting.
    
    Configured with parameters optimized for financial time series,
    providing strong baseline performance for price prediction tasks.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize XGBoost regressor with financial-optimized defaults.
        
        Args:
            **kwargs: XGBoost parameters. Common ones include:
                - n_estimators: Number of boosting rounds (default: 500)
                - learning_rate: Boosting learning rate (default: 0.05)
                - max_depth: Maximum tree depth (default: 6)
                - subsample: Fraction of samples to use (default: 0.8)
                - colsample_bytree: Fraction of features to use (default: 0.8)
                - min_child_weight: Minimum sum of weights in leaf (default: 3)
                - reg_alpha: L1 regularization (default: 0.1)
                - reg_lambda: L2 regularization (default: 0.1)
                - random_state: Random seed (default: 42)
        """
        # Financial-optimized defaults
        default_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'n_estimators': 500,
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'colsample_bylevel': 0.8,
            'min_child_weight': 3,
            'reg_alpha': 0.1,      # L1 regularization
            'reg_lambda': 0.1,     # L2 regularization
            'gamma': 0.0,          # Minimum split gain
            'random_state': 42,
            'n_jobs': -1,          # Use all cores
            'verbosity': 0         # Suppress output
        }
        
        # Update defaults with user parameters
        default_params.update(kwargs)
        
        super().__init__(model_name="XGBoost Regressor", **default_params)
        
        # Initialize model
        self._model = xgb.XGBRegressor(**self.model_params)
        self._feature_importance = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'XGBoostRegressor':
        """
        Train the XGBoost model.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target vector of shape (n_samples,).
            
        Returns:
            Self for method chaining.
        """
        logger.info("Training %s on data shape: X=%s, y=%s", 
                   self.model_name, X.shape, y.shape)
        
        # Validate input data
        if np.isnan(X).any() or np.isnan(y).any():
            raise ValueError("Training data contains NaN values")
            
        if len(X) != len(y):
            raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")
        
        # Train the model
        try:
            self._model.fit(X, y)
            self.is_fitted = True
            
            # Store feature importance
            self._feature_importance = self._model.feature_importances_
            
            logger.info("Successfully trained %s", self.model_name)
            
        except Exception as e:
            logger.error("Error training %s: %s", self.model_name, str(e))
            raise
            
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions using the trained model.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features).
            
        Returns:
            Predictions array of shape (n_samples,).
        """
        if not self.is_fitted:
            raise ValueError(f"{self.model_name} must be fitted before prediction")
            
        if np.isnan(X).any():
            raise ValueError("Prediction data contains NaN values")
        
        logger.debug("Generating predictions for %d samples", len(X))
        
        try:
            predictions = self._model.predict(X)
            return predictions
            
        except Exception as e:
            logger.error("Error during prediction: %s", str(e))
            raise
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance scores.
        
        Returns:
            Feature importance array or None if model not fitted.
        """
        if not self.is_fitted:
            logger.warning("Model not fitted, cannot return feature importance")
            return None
            
        return self._feature_importance.copy()
    
    def get_feature_importance_df(self, feature_names: list) -> Optional[object]:
        """
        Get feature importance as a sorted DataFrame.
        
        Args:
            feature_names: List of feature names.
            
        Returns:
            Pandas DataFrame with features and importance scores.
        """
        import pandas as pd
        
        importance = self.get_feature_importance()
        if importance is None:
            return None
            
        if len(feature_names) != len(importance):
            logger.warning("Feature names length (%d) doesn't match importance length (%d)",
                          len(feature_names), len(importance))
            return None
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df
    
    def get_booster(self):
        """
        Get the underlying XGBoost booster.
        
        Returns:
            XGBoost booster object for advanced analysis.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to access booster")
        return self._model.get_booster()
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model (should end with .pkl).
        """
        if not self.is_fitted:
            raise ValueError(f"{self.model_name} must be fitted before saving")
        
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save both the XGBoost model and our wrapper state
        save_dict = {
            'xgb_model': self._model,
            'model_name': self.model_name,
            'model_params': self.model_params,
            'feature_importance': self._feature_importance,
            'is_fitted': self.is_fitted
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(save_dict, f)
            logger.info("Saved %s to %s", self.model_name, filepath)
            
        except Exception as e:
            logger.error("Error saving model: %s", str(e))
            raise
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model file.
        """
        try:
            with open(filepath, 'rb') as f:
                save_dict = pickle.load(f)
            
            self._model = save_dict['xgb_model']
            self.model_name = save_dict['model_name']
            self.model_params = save_dict['model_params']
            self._feature_importance = save_dict['feature_importance']
            self.is_fitted = save_dict['is_fitted']
            
            logger.info("Loaded %s from %s", self.model_name, filepath)
            
        except Exception as e:
            logger.error("Error loading model: %s", str(e))
            raise
    
    def get_params(self) -> dict:
        """
        Get model parameters.
        
        Returns:
            Dictionary of model parameters.
        """
        return self.model_params.copy()
    
    def set_params(self, **params) -> 'XGBoostRegressor':
        """
        Set model parameters.
        
        Args:
            **params: Parameters to update.
            
        Returns:
            Self for method chaining.
        """
        self.model_params.update(params)
        self._model = xgb.XGBRegressor(**self.model_params)
        self.is_fitted = False  # Need to retrain after parameter change
        
        logger.info("Updated %s parameters: %s", self.model_name, params)
        return self