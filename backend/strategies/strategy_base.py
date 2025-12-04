"""Base strategy interface."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from datetime import datetime

import pandas as pd
from loguru import logger


class Signal:
    """Trading signal."""

    def __init__(
        self,
        symbol: str,
        action: str,  # "buy", "sell", "hold"
        strength: float = 1.0,  # 0.0 to 1.0
        reason: str = "",
        price: Optional[float] = None,
        quantity: Optional[float] = None,
    ):
        """Initialize signal."""
        self.symbol = symbol
        self.action = action
        self.strength = strength
        self.reason = reason
        self.price = price
        self.quantity = quantity
        self.timestamp = datetime.utcnow()

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "action": self.action,
            "strength": self.strength,
            "reason": self.reason,
            "price": self.price,
            "quantity": self.quantity,
            "timestamp": self.timestamp.isoformat(),
        }


class StrategyBase(ABC):
    """Base class for all trading strategies."""

    def __init__(self, name: str, parameters: Optional[Dict] = None):
        """
        Initialize strategy.

        Args:
            name: Strategy name
            parameters: Strategy parameters
        """
        self.name = name
        self.parameters = parameters or {}
        self.enabled = True
        self._signals_generated = 0
        self._last_update = None

    @abstractmethod
    async def analyze(self, symbol: str, data: pd.DataFrame) -> Signal:
        """
        Analyze market data and generate trading signal.

        Args:
            symbol: Stock symbol
            data: Historical price data

        Returns:
            Trading signal
        """
        pass

    @abstractmethod
    def get_required_data_length(self) -> int:
        """
        Get minimum number of data points required.

        Returns:
            Number of bars needed
        """
        pass

    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that data is sufficient for analysis.

        Args:
            data: Price data

        Returns:
            True if data is valid
        """
        required_length = self.get_required_data_length()
        if len(data) < required_length:
            logger.warning(
                f"{self.name}: Insufficient data - need {required_length}, "
                f"have {len(data)}"
            )
            return False

        required_columns = ["open", "high", "low", "close", "volume"]
        missing = [col for col in required_columns if col not in data.columns]
        if missing:
            logger.error(f"{self.name}: Missing columns: {missing}")
            return False

        return True

    async def generate_signal(self, symbol: str, data: pd.DataFrame) -> Signal:
        """
        Generate trading signal with validation.

        Args:
            symbol: Stock symbol
            data: Price data

        Returns:
            Trading signal
        """
        if not self.enabled:
            return Signal(symbol, "hold", reason="Strategy disabled")

        if not self.validate_data(data):
            return Signal(symbol, "hold", reason="Insufficient data")

        signal = await self.analyze(symbol, data)
        self._signals_generated += 1
        self._last_update = datetime.utcnow()

        logger.debug(
            f"{self.name} signal for {symbol}: {signal.action} "
            f"(strength: {signal.strength:.2f})"
        )

        return signal

    def get_stats(self) -> Dict:
        """Get strategy statistics."""
        return {
            "name": self.name,
            "enabled": self.enabled,
            "signals_generated": self._signals_generated,
            "last_update": self._last_update.isoformat() if self._last_update else None,
            "parameters": self.parameters,
        }

    def update_parameters(self, parameters: Dict):
        """Update strategy parameters."""
        self.parameters.update(parameters)
        logger.info(f"{self.name}: Parameters updated")

    def enable(self):
        """Enable strategy."""
        self.enabled = True
        logger.info(f"{self.name}: Enabled")

    def disable(self):
        """Disable strategy."""
        self.enabled = False
        logger.info(f"{self.name}: Disabled")
