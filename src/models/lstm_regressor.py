"""
lstm_regressor.py

Responsibilities:
  - PyTorch LSTM regression model for sequential price forecasting.
  - GPU acceleration for RTX 3080 Ti compatibility.
  - Handles 3D windowed OHLCV data from deep_features.py.
"""

import logging
import os
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.models.base_model import BaseModel
from src.utils.io_helpers import setup_logging

# Initialize logger
logger = logging.getLogger(__name__)
setup_logging()
fh = logging.FileHandler("logs/models.log")
fh.setLevel(logging.INFO)
logger.addHandler(fh)


class SimpleLSTMNetwork(nn.Module):
    """
    Simple LSTM network for price regression.
    
    Architecture: LSTM → Dropout → Linear → Output
    """
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_layers: int = 2, dropout: float = 0.2):
        """
        Initialize LSTM network.
        
        Args:
            input_size: Number of input features (e.g., 5 for OHLCV).
            hidden_size: Size of LSTM hidden state.
            num_layers: Number of LSTM layers.
            dropout: Dropout probability.
        """
        super(SimpleLSTMNetwork, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.linear = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size).
            
        Returns:
            Output tensor of shape (batch_size, 1).
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take the last time step output
        last_output = lstm_out[:, -1, :]
        
        # Apply dropout
        dropped = self.dropout(last_output)
        
        # Final linear layer
        output = self.linear(dropped)
        
        return output


class LSTMRegressor(BaseModel):
    """
    LSTM regression model for price forecasting.
    
    Handles sequential OHLCV data and provides GPU acceleration
    for training on RTX 3080 Ti.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize LSTM regressor with optimized defaults.
        
        Args:
            **kwargs: Model parameters. Common ones include:
                - hidden_size: LSTM hidden state size (default: 128)
                - num_layers: Number of LSTM layers (default: 2)
                - dropout: Dropout probability (default: 0.2)
                - learning_rate: Learning rate (default: 0.001)
                - batch_size: Training batch size (default: 32)
                - epochs: Number of training epochs (default: 100)
                - patience: Early stopping patience (default: 10)
                - device: Training device ('auto', 'cuda', 'cpu') (default: 'auto')
        """
        # Default parameters optimized for financial data
        default_params = {
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'patience': 10,
            'device': 'auto',
            'random_state': 42
        }
        
        default_params.update(kwargs)
        super().__init__(model_name="LSTM Regressor", **default_params)
        
        # Set random seeds for reproducibility
        torch.manual_seed(self.model_params['random_state'])
        np.random.seed(self.model_params['random_state'])
        
        # Determine device
        if self.model_params['device'] == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.model_params['device'])
        
        logger.info("LSTM using device: %s", self.device)
        
        # Initialize model components
        self.network = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self._training_history = []
        
    def _create_network(self, input_size: int):
        """Create and initialize the LSTM network."""
        self.network = SimpleLSTMNetwork(
            input_size=input_size,
            hidden_size=self.model_params['hidden_size'],
            num_layers=self.model_params['num_layers'],
            dropout=self.model_params['dropout']
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.model_params['learning_rate']
        )
        
        logger.info("Created LSTM network with %d parameters", 
                   sum(p.numel() for p in self.network.parameters()))
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LSTMRegressor':
        """
        Train the LSTM model.
        
        Args:
            X: Input windows of shape (n_samples, sequence_length, n_features).
            y: Target values of shape (n_samples,).
            
        Returns:
            Self for method chaining.
        """
        logger.info("Training %s on data shape: X=%s, y=%s", 
                   self.model_name, X.shape, y.shape)
        
        # Validate input dimensions
        if len(X.shape) != 3:
            raise ValueError(f"X must be 3D array, got shape {X.shape}")
        
        if np.isnan(X).any() or np.isnan(y).any():
            raise ValueError("Training data contains NaN values")
        
        # Create network if not exists
        if self.network is None:
            input_size = X.shape[2]  # Number of features
            self._create_network(input_size)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).unsqueeze(1).to(self.device)  # Add dimension for consistency
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=self.model_params['batch_size'],
            shuffle=True
        )
        
        # Training loop
        self.network.train()
        self._training_history = []
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.model_params['epochs']):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_X, batch_y in dataloader:
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.network(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                
                # Update weights
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            self._training_history.append(avg_loss)
            
            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Log progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                logger.info("Epoch %d/%d - Loss: %.6f", 
                           epoch + 1, self.model_params['epochs'], avg_loss)
            
            # Early stopping
            if patience_counter >= self.model_params['patience']:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break
        
        self.is_fitted = True
        logger.info("Successfully trained %s", self.model_name)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions using the trained model.
        
        Args:
            X: Input windows of shape (n_samples, sequence_length, n_features).
            
        Returns:
            Predictions array of shape (n_samples,).
        """
        if not self.is_fitted:
            raise ValueError(f"{self.model_name} must be fitted before prediction")
        
        if len(X.shape) != 3:
            raise ValueError(f"X must be 3D array, got shape {X.shape}")
        
        if np.isnan(X).any():
            raise ValueError("Prediction data contains NaN values")
        
        self.network.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            
            # Handle batch prediction to avoid memory issues
            batch_size = self.model_params['batch_size']
            predictions = []
            
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i + batch_size]
                batch_pred = self.network(batch)
                predictions.append(batch_pred.cpu().numpy())
            
            # Concatenate all predictions and flatten
            all_predictions = np.concatenate(predictions, axis=0).flatten()
        
        logger.debug("Generated predictions for %d samples", len(all_predictions))
        return all_predictions
    
    def get_training_history(self) -> list:
        """
        Get training loss history.
        
        Returns:
            List of loss values from training.
        """
        return self._training_history.copy()
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model (should end with .pth).
        """
        if not self.is_fitted:
            raise ValueError(f"{self.model_name} must be fitted before saving")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save complete model state
        save_dict = {
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_params': self.model_params,
            'training_history': self._training_history,
            'is_fitted': self.is_fitted
        }
        
        try:
            torch.save(save_dict, filepath)
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
            save_dict = torch.load(filepath, map_location=self.device)
            
            self.model_params = save_dict['model_params']
            self._training_history = save_dict['training_history']
            self.is_fitted = save_dict['is_fitted']
            
            # Recreate network architecture (need input size from saved params)
            # This is a limitation - we need to know the input size
            # For now, we'll store it in model_params during first fit
            if 'input_size' in self.model_params:
                self._create_network(self.model_params['input_size'])
                self.network.load_state_dict(save_dict['network_state_dict'])
                self.optimizer.load_state_dict(save_dict['optimizer_state_dict'])
            else:
                raise ValueError("Cannot load model without input_size in saved parameters")
            
            logger.info("Loaded %s from %s", self.model_name, filepath)
            
        except Exception as e:
            logger.error("Error loading model: %s", str(e))
            raise
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        LSTM models don't have traditional feature importance.
        
        Returns:
            None (LSTM doesn't provide feature importance).
        """
        logger.info("LSTM models don't provide traditional feature importance")
        return None