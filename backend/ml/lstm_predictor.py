"""LSTM Time Series Forecasting for Price Prediction.

Implements sophisticated LSTM models for multi-timeframe price forecasting
with attention mechanisms and rolling window validation.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    
from loguru import logger


class LSTMPredictor:
    """
    LSTM-based price predictor with multi-timeframe support.
    
    Features:
    - Bidirectional LSTM layers
    - Attention mechanism
    - Multi-step ahead prediction
    - Rolling window validation
    """
    
    def __init__(
        self,
        lookback_steps: int = 60,
        forecast_horizon: int = 5,
        lstm_units: int = 128,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001
    ):
        """
        Initialize LSTM predictor.
        
        Args:
            lookback_steps: Number of historical steps to use
            forecast_horizon: Steps ahead to predict
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
        """
        if not TF_AVAILABLE:
            logger.error("TensorFlow not available. Install with: poetry add tensorflow")
            raise ImportError("TensorFlow required for LSTM predictor")
            
        self.lookback_steps = lookback_steps
        self.forecast_horizon = forecast_horizon
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.feature_names = None
        
        logger.info(
            f"LSTM Predictor initialized (lookback={lookback_steps}, "
            f"horizon={forecast_horizon}, units={lstm_units})"
        )
        
    def _build_model(self, n_features: int):
        """
        Build LSTM model with attention.
        
        Args:
            n_features: Number of input features
        """
        # Input layer
        inputs = keras.Input(shape=(self.lookback_steps, n_features))
        
        # Bidirectional LSTM layers
        x = layers.Bidirectional(
            layers.LSTM(self.lstm_units, return_sequences=True)
        )(inputs)
        x = layers.Dropout(self.dropout_rate)(x)
        
        x = layers.Bidirectional(
            layers.LSTM(self.lstm_units // 2, return_sequences=True)
        )(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Attention mechanism
        attention = layers.Attention()([x, x])
        x = layers.Add()([x, attention])
        
        # Final LSTM layer
        x = layers.LSTM(self.lstm_units // 4)(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Dense layers
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(32, activation='relu')(x)
        
        # Output layer (forecast_horizon predictions)
        outputs = layers.Dense(self.forecast_horizon)(x)
        
        # Build model
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        logger.info(f"LSTM model built with {model.count_params():,} parameters")
        
    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = 'close',
        feature_cols: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM training.
        
        Args:
            df: DataFrame with price data
            target_col: Column to predict
            feature_cols: Feature columns to use
            
        Returns:
            X, y arrays for training
        """
        if feature_cols is None:
            feature_cols = ['open', 'high', 'low', 'close', 'volume']
            
        self.feature_names = feature_cols
        
        # Extract features and target
        X_data = df[feature_cols].values
        y_data = df[target_col].values
        
        # Normalize data
        from sklearn.preprocessing import StandardScaler
        
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        X_scaled = self.scaler_X.fit_transform(X_data)
        y_scaled = self.scaler_y.fit_transform(y_data.reshape(-1, 1)).flatten()
        
        # Create sequences
        X, y = [], []
        
        for i in range(len(X_scaled) - self.lookback_steps - self.forecast_horizon + 1):
            X.append(X_scaled[i:i + self.lookback_steps])
            y.append(y_scaled[i + self.lookback_steps:i + self.lookback_steps + self.forecast_horizon])
            
        return np.array(X), np.array(y)
        
    def train(
        self,
        df: pd.DataFrame,
        target_col: str = 'close',
        feature_cols: Optional[List[str]] = None,
        validation_split: float = 0.2,
        epochs: int = 50,
        batch_size: int = 32
    ) -> Dict:
        """
        Train LSTM model.
        
        Args:
            df: Training data
            target_col: Target column
            feature_cols: Feature columns
            validation_split: Validation data fraction
            epochs: Training epochs
            batch_size: Batch size
            
        Returns:
            Training history
        """
        # Prepare data
        X, y = self.prepare_data(df, target_col, feature_cols)
        
        logger.info(f"Training data shape: X={X.shape}, y={y.shape}")
        
        # Build model if not exists
        if self.model is None:
            self._build_model(n_features=X.shape[2])
            
        # Train
        history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7
                )
            ]
        )
        
        logger.success(
            f"Training complete. Final loss: {history.history['loss'][-1]:.6f}, "
            f"Val loss: {history.history['val_loss'][-1]:.6f}"
        )
        
        return history.history
        
    def predict(
        self,
        df: pd.DataFrame,
        return_confidence: bool = True
    ) -> Dict:
        """
        Make predictions on new data.
        
        Args:
            df: Input data (must have feature columns)
            return_confidence: Return confidence intervals
            
        Returns:
            Dict with predictions and optionally confidence intervals
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        # Prepare input
        X_data = df[self.feature_names].values
        X_scaled = self.scaler_X.transform(X_data)
        
        # Need at least lookback_steps of data
        if len(X_scaled) < self.lookback_steps:
            raise ValueError(f"Need at least {self.lookback_steps} data points")
            
        # Take last lookback_steps
        X_input = X_scaled[-self.lookback_steps:].reshape(1, self.lookback_steps, -1)
        
        # Predict
        y_pred_scaled = self.model.predict(X_input, verbose=0)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled).flatten()
        
        result = {
            'predictions': y_pred,
            'horizon': self.forecast_horizon,
            'timestamps': [
                df.index[-1] + timedelta(minutes=i+1)
                for i in range(self.forecast_horizon)
            ] if isinstance(df.index, pd.DatetimeIndex) else list(range(self.forecast_horizon))
        }
        
        # Calculate confidence intervals (using prediction uncertainty)
        if return_confidence:
            # Monte Carlo dropout for uncertainty estimation
            n_iterations = 100
            predictions = []
            
            for _ in range(n_iterations):
                y_mc = self.model(X_input, training=True)  # Enable dropout
                y_mc_unscaled = self.scaler_y.inverse_transform(y_mc.numpy()).flatten()
                predictions.append(y_mc_unscaled)
                
            predictions = np.array(predictions)
            
            result['confidence_lower'] = np.percentile(predictions, 2.5, axis=0)
            result['confidence_upper'] = np.percentile(predictions, 97.5, axis=0)
            result['std'] = np.std(predictions, axis=0)
            
        return result
        
    def save_model(self, filepath: str):
        """Save model to file."""
        if self.model is not None:
            self.model.save(filepath)
            logger.info(f"Model saved to {filepath}")
            
    def load_model(self, filepath: str):
        """Load model from file."""
        self.model = keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")


class MultiTimeframeLSTM:
    """
    Ensemble of LSTM models for different timeframes.
    
    Combines predictions from 1min, 5min, 15min, 1h models.
    """
    
    def __init__(self):
        """Initialize multi-timeframe ensemble."""
        self.models = {
            '1min': LSTMPredictor(lookback_steps=60, forecast_horizon=5),
            '5min': LSTMPredictor(lookback_steps=60, forecast_horizon=5),
            '15min': LSTMPredictor(lookback_steps=48, forecast_horizon=4),
            '1h': LSTMPredictor(lookback_steps=24, forecast_horizon=3),
        }
        self.weights = {
            '1min': 0.4,
            '5min': 0.3,
            '15min': 0.2,
            '1h': 0.1,
        }
        
    def train_all(
        self,
        data_dict: Dict[str, pd.DataFrame],
        **kwargs
    ):
        """
        Train all timeframe models.
        
        Args:
            data_dict: Dict mapping timeframes to DataFrames
            **kwargs: Training arguments
        """
        for timeframe, model in self.models.items():
            if timeframe in data_dict:
                logger.info(f"Training {timeframe} model...")
                model.train(data_dict[timeframe], **kwargs)
                
    def predict_ensemble(
        self,
        data_dict: Dict[str, pd.DataFrame]
    ) -> np.ndarray:
        """
        Get weighted ensemble prediction.
        
        Args:
            data_dict: Dict mapping timeframes to DataFrames
            
        Returns:
            Weighted average prediction
        """
        predictions = []
        weights = []
        
        for timeframe, model in self.models.items():
            if timeframe in data_dict and model.model is not None:
                pred = model.predict(data_dict[timeframe], return_confidence=False)
                predictions.append(pred['predictions'][0])  # Next step
                weights.append(self.weights[timeframe])
                
        if not predictions:
            return None
            
        # Weighted average
        weights = np.array(weights) / sum(weights)
        ensemble_pred = np.average(predictions, weights=weights)
        
        return ensemble_pred


# Global instance
lstm_predictor = LSTMPredictor() if TF_AVAILABLE else None
multi_timeframe_lstm = MultiTimeframeLSTM() if TF_AVAILABLE else None
