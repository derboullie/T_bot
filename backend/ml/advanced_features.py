"""Advanced Feature Engineering for ML Models.

Creates sophisticated features from market data including:
- 50+ technical indicators
- Order flow features
- Market microstructure
- Volatility features
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from loguru import logger


class AdvancedFeatureEngineer:
    """
    Create advanced features for ML models.
    
    Features:
    - Technical indicators (RSI, MACD, Bollinger, etc.)
    - Momentum features
    - Volatility features
    - Volume features
    - Market microstructure
    """
    
    def __init__(self):
        """Initialize feature engineer."""
        self.feature_names = []
        logger.info("Advanced Feature Engineer initialized")
        
    def create_all_features(
        self,
        df: pd.DataFrame,
        price_col: str = 'close'
    ) -> pd.DataFrame:
        """
        Create all features.
        
        Args:
            df: DataFrame with OHLCV data
            price_col: Price column to use for calculations
            
        Returns:
            DataFrame with all features added
        """
        df = df.copy()
        
        # Technical indicators
        df = self.add_technical_indicators(df, price_col)
        
        # Momentum features
        df = self.add_momentum_features(df, price_col)
        
        # Volatility features
        df = self.add_volatility_features(df)
        
        # Volume features
        if 'volume' in df.columns:
            df = self.add_volume_features(df)
            
        # Price patterns
        df = self.add_price_patterns(df)
        
        return df
        
    def add_technical_indicators(
        self,
        df: pd.DataFrame,
        price_col: str = 'close'
    ) -> pd.DataFrame:
        """Add technical indicators."""
        # Moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = df[price_col].rolling(period).mean()
            df[f'ema_{period}'] = df[price_col].ewm(span=period).mean()
            
        # RSI
        for period in [7, 14, 21]:
            df[f'rsi_{period}'] = self._calculate_rsi(df[price_col], period)
            
        # MACD
        ema_12 = df[price_col].ewm(span=12).mean()
        ema_26 = df[price_col].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        for period in [20, 50]:
            sma = df[price_col].rolling(period).mean()
            std = df[price_col].rolling(period).std()
            df[f'bb_upper_{period}'] = sma + (2 * std)
            df[f'bb_lower_{period}'] = sma - (2 * std)
            df[f'bb_pct_{period}'] = (df[price_col] - df[f'bb_lower_{period}']) / \
                                     (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])
                                     
        return df
        
    def add_momentum_features(
        self,
        df: pd.DataFrame,
        price_col: str = 'close'
    ) -> pd.DataFrame:
        """Add momentum features."""
        # Rate of change
        for period in [1, 5, 10, 20]:
            df[f'roc_{period}'] = df[price_col].pct_change(period)
            
        # Momentum
        for period in [10, 20, 50]:
            df[f'momentum_{period}'] = df[price_col] - df[price_col].shift(period)
            
        # Stochastic Oscillator
        for period in [14, 21]:
            low_min = df['low'].rolling(period).min()
            high_max = df['high'].rolling(period).max()
            df[f'stoch_{period}'] = 100 * (df[price_col] - low_min) / (high_max - low_min)
            
        return df
        
    def add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features."""
        # ATR (Average True Range)
        for period in [7, 14, 21]:
            df[f'atr_{period}'] = self._calculate_atr(df, period)
            
        # Historical volatility
        returns = df['close'].pct_change()
        for period in [10, 20, 50]:
            df[f'volatility_{period}'] = returns.rolling(period).std() * np.sqrt(252)
            
        # Parkinson volatility (high-low)
        for period in [10, 20]:
            hl_ratio = np.log(df['high'] / df['low']) ** 2
            df[f'parkinson_vol_{period}'] = np.sqrt(
                hl_ratio.rolling(period).mean() / (4 * np.log(2))
            )
            
        return df
        
    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume features."""
        # Volume moving averages
        for period in [5, 10, 20]:
            df[f'volume_sma_{period}'] = df['volume'].rolling(period).mean()
            
        # Volume ratio
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # On-Balance Volume (OBV)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        
        # Volume-Price Trend
        df['vpt'] = (df['volume'] * df['close'].pct_change()).cumsum()
        
        # Money Flow Index
        df['mfi'] = self._calculate_mfi(df)
        
        return df
        
    def add_price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price pattern features."""
        # Support/Resistance levels
        df['support'] = df['low'].rolling(20).min()
        df['resistance'] = df['high'].rolling(20).max()
        
        # Price position in range
        df['price_position'] = (df['close'] - df['support']) / \
                               (df['resistance'] - df['support'])
                               
        # Pivot points
        df['pivot'] = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
        df['r1'] = 2 * df['pivot'] - df['low'].shift(1)
        df['s1'] = 2 * df['pivot'] - df['high'].shift(1)
        
        # Candlestick patterns
        df['body'] = abs(df['open'] - df['close'])
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        
        return df
        
    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    @staticmethod
    def _calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR."""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()
        
    @staticmethod
    def _calculate_mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0)
        
        positive_mf = positive_flow.rolling(period).sum()
        negative_mf = negative_flow.rolling(period).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        return mfi


# Global instance
feature_engineer = AdvancedFeatureEngineer()
