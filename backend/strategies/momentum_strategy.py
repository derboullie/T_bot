"""Momentum-based trading strategy using RSI and Moving Averages."""

from typing import Dict, Optional

import pandas as pd
import numpy as np

from backend.strategies.strategy_base import StrategyBase, Signal


class MomentumStrategy(StrategyBase):
    """Simple momentum strategy using RSI and moving averages."""

    def __init__(self, parameters: Optional[Dict] = None):
        """Initialize momentum strategy."""
        default_params = {
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "sma_short": 20,
            "sma_long": 50,
        }
        params = {**default_params, **(parameters or {})}
        super().__init__("Momentum Strategy", params)

    def get_required_data_length(self) -> int:
        """Need enough data for longest indicator."""
        return max(self.parameters["sma_long"], self.parameters["rsi_period"]) + 10

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate RSI indicator using optimized pandas operations.
        
        Uses Exponential Weighted Moving Average for smooth calculation.

        Args:
            prices: Close prices
            period: RSI period

        Returns:
            RSI values (0-100)
        """
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        # Use exponential weighted moving average for smoother RSI
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    async def analyze(self, symbol: str, data: pd.DataFrame) -> Signal:
        """
        Analyze data and generate signal.

        Args:
            symbol: Stock symbol
            data: Price data with OHLCV

        Returns:
            Trading signal
        """
        # Calculate indicators
        close = data["close"]

        # RSI
        rsi = self.calculate_rsi(close, self.parameters["rsi_period"])
        current_rsi = rsi.iloc[-1]

        # Moving averages
        sma_short = close.rolling(window=self.parameters["sma_short"]).mean()
        sma_long = close.rolling(window=self.parameters["sma_long"]).mean()

        current_sma_short = sma_short.iloc[-1]
        current_sma_long = sma_long.iloc[-1]

        current_price = close.iloc[-1]

        # Generate signal
        reasons = []

        # Buy conditions
        buy_signal = False
        if current_rsi < self.parameters["rsi_oversold"]:
            buy_signal = True
            reasons.append(f"RSI oversold ({current_rsi:.1f})")

        if current_sma_short > current_sma_long:
            if not buy_signal:
                buy_signal = True
            reasons.append(f"SMA bullish crossover")

        # Sell conditions
        sell_signal = False
        if current_rsi > self.parameters["rsi_overbought"]:
            sell_signal = True
            reasons.append(f"RSI overbought ({current_rsi:.1f})")

        if current_sma_short < current_sma_long:
            if not sell_signal:
                sell_signal = True
            reasons.append(f"SMA bearish crossover")

        # Determine action
        if buy_signal and not sell_signal:
            # Calculate signal strength based on RSI distance from oversold
            strength = min(1.0, (self.parameters["rsi_oversold"] - current_rsi) / 20 + 0.5)
            return Signal(
                symbol=symbol,
                action="buy",
                strength=max(0.5, strength),
                reason="; ".join(reasons),
                price=current_price,
            )

        elif sell_signal and not buy_signal:
            # Calculate signal strength based on RSI distance from overbought
            strength = min(1.0, (current_rsi - self.parameters["rsi_overbought"]) / 20 + 0.5)
            return Signal(
                symbol=symbol,
                action="sell",
                strength=max(0.5, strength),
                reason="; ".join(reasons),
                price=current_price,
            )

        else:
            return Signal(
                symbol=symbol,
                action="hold",
                strength=0.0,
                reason="No clear signal",
                price=current_price,
            )


# Default instance
momentum_strategy = MomentumStrategy()
