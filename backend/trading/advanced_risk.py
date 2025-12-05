"""Advanced Risk Management Module.

Implements sophisticated risk management strategies including:
- Trailing stop-loss
- Time-based stops
- Volatility-adjusted stops
- Profit-lock mechanisms
"""

from typing import Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
from loguru import logger


@dataclass
class StopLossConfig:
    """Configuration for stop-loss mechanisms."""
    trailing_percent: float = 0.02  # 2% trailing stop
    fixed_percent: Optional[float] = None  # Fixed stop-loss
    time_based_hours: Optional[int] = None  # Exit after X hours
    volatility_multiplier: float = 2.0  # ATR multiplier for volatile stops
    profit_lock_percent: float = 0.5  # Lock in 50% of profits


class AdvancedStopLoss:
    """
    Advanced stop-loss manager with multiple mechanisms.
    
    Features:
    - Trailing stop-loss that follows price
    - Volatility-adjusted stops
    - Time-based exits
    - Profit-locking mechanism
    """
    
    def __init__(self, config: Optional[StopLossConfig] = None):
        """
        Initialize advanced stop-loss manager.
        
        Args:
            config: Stop-loss configuration
        """
        self.config = config or StopLossConfig()
        
        # Track positions and their stop levels
        self.position_stops: Dict[str, Dict] = {}
        
        logger.info("Advanced stop-loss manager initialized")
        
    def register_position(
        self,
        symbol: str,
        entry_price: float,
        quantity: float,
        entry_time: datetime,
        side: str = 'long'
    ):
        """
        Register a new position for stop-loss tracking.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            quantity: Position size
            entry_time: Time of entry
            side: 'long' or 'short'
        """
        # Calculate initial stop level
        if self.config.fixed_percent:
            if side == 'long':
                stop_price = entry_price * (1 - self.config.fixed_percent)
            else:
                stop_price = entry_price * (1 + self.config.fixed_percent)
        else:
            # Use trailing stop from entry
            if side == 'long':
                stop_price = entry_price * (1 - self.config.trailing_percent)
            else:
                stop_price = entry_price * (1 + self.config.trailing_percent)
                
        self.position_stops[symbol] = {
            'entry_price': entry_price,
            'entry_time': entry_time,
            'quantity': quantity,
            'side': side,
            'stop_price': stop_price,
            'highest_price': entry_price if side == 'long' else entry_price,
            'lowest_price': entry_price if side == 'short' else entry_price,
            'profit_locked': False,
        }
        
        logger.info(
            f"Position registered: {symbol} @ ${entry_price:.2f}, "
            f"Stop: ${stop_price:.2f}"
        )
        
    def update_stop(
        self,
        symbol: str,
        current_price: float,
        current_time: datetime,
        volatility: Optional[float] = None
    ) -> Optional[Dict]:
        """
        Update stop-loss levels and check if stop is hit.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            current_time: Current time
            volatility: Current volatility (ATR) for adaptive stops
            
        Returns:
            Stop signal dict if stop is hit, None otherwise
        """
        if symbol not in self.position_stops:
            return None
            
        position = self.position_stops[symbol]
        side = position['side']
        
        # Update highest/lowest prices
        if side == 'long':
            if current_price > position['highest_price']:
                position['highest_price'] = current_price
                
                # Update trailing stop
                new_stop = current_price * (1 - self.config.trailing_percent)
                
                # Adjust for volatility if provided
                if volatility and self.config.volatility_multiplier:
                    volatility_buffer = volatility * self.config.volatility_multiplier
                    new_stop = max(new_stop, current_price - volatility_buffer)
                    
                # Only move stop up, never down
                if new_stop > position['stop_price']:
                    old_stop = position['stop_price']
                    position['stop_price'] = new_stop
                    logger.debug(
                        f"{symbol}: Trailing stop updated ${old_stop:.2f} -> ${new_stop:.2f}"
                    )
                    
            # Check for profit lock
            if not position['profit_locked']:
                profit_pct = (current_price - position['entry_price']) / position['entry_price']
                if profit_pct >= self.config.profit_lock_percent:
                    # Lock in profits
                    lock_price = position['entry_price'] * (1 + profit_pct * 0.5)
                    position['stop_price'] = max(position['stop_price'], lock_price)
                    position['profit_locked'] = True
                    logger.info(
                        f"{symbol}: Profit locked at ${position['stop_price']:.2f} "
                        f"({profit_pct*100:.1f}% profit)"
                    )
                    
        else:  # short
            if current_price < position['lowest_price']:
                position['lowest_price'] = current_price
                
                # Update trailing stop
                new_stop = current_price * (1 + self.config.trailing_percent)
                
                if volatility and self.config.volatility_multiplier:
                    volatility_buffer = volatility * self.config.volatility_multiplier
                    new_stop = min(new_stop, current_price + volatility_buffer)
                    
                # Only move stop down, never up
                if new_stop < position['stop_price']:
                    position['stop_price'] = new_stop
                    
        # Check time-based stop
        if self.config.time_based_hours:
            time_in_position = current_time - position['entry_time']
            if time_in_position >= timedelta(hours=self.config.time_based_hours):
                return {
                    'type': 'time_based',
                    'symbol': symbol,
                    'reason': f'Time limit reached ({self.config.time_based_hours}h)',
                    'current_price': current_price,
                }
                
        # Check if stop is hit
        if side == 'long' and current_price <= position['stop_price']:
            return {
                'type': 'stop_loss',
                'symbol': symbol,
                'reason': 'Trailing stop-loss hit',
                'stop_price': position['stop_price'],
                'current_price': current_price,
                'loss': (current_price - position['entry_price']) / position['entry_price'] * 100,
            }
        elif side == 'short' and current_price >= position['stop_price']:
            return {
                'type': 'stop_loss',
                'symbol': symbol,
                'reason': 'Trailing stop-loss hit (short)',
                'stop_price': position['stop_price'],
                'current_price': current_price,
                'loss': (position['entry_price'] - current_price) / position['entry_price'] * 100,
            }
            
        return None
        
    def remove_position(self, symbol: str):
        """Remove position from tracking."""
        if symbol in self.position_stops:
            del self.position_stops[symbol]
            logger.info(f"Position removed from stop tracking: {symbol}")
            
    def get_stop_info(self, symbol: str) -> Optional[Dict]:
        """Get current stop information for a position."""
        return self.position_stops.get(symbol)
        
    def get_all_stops(self) -> Dict[str, Dict]:
        """Get all position stops."""
        return self.position_stops.copy()


class DynamicStopLoss:
    """
    Dynamic stop-loss that adjusts based on market conditions.
    
    Uses ATR (Average True Range) for volatility-based stops.
    """
    
    def __init__(self, atr_period: int = 14, atr_multiplier: float = 2.0):
        """
        Initialize dynamic stop-loss.
        
        Args:
            atr_period: Period for ATR calculation
            atr_multiplier: Multiplier for ATR-based stop distance
        """
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.price_history: Dict[str, List[Dict]] = {}
        
    def calculate_atr(self, symbol: str, high: float, low: float, close: float) -> float:
        """
        Calculate Average True Range.
        
        Args:
            symbol: Trading symbol
            high: High price
            low: Low price
            close: Close price
            
        Returns:
            ATR value
        """
        if symbol not in self.price_history:
            self.price_history[symbol] = []
            
        # Add current bar
        self.price_history[symbol].append({
            'high': high,
            'low': low,
            'close': close,
        })
        
        # Keep only recent history
        if len(self.price_history[symbol]) > self.atr_period + 1:
            self.price_history[symbol] = self.price_history[symbol][-(self.atr_period + 1):]
            
        # Need at least 2 bars
        if len(self.price_history[symbol]) < 2:
            return high - low
            
        # Calculate True Range for each bar
        true_ranges = []
        history = self.price_history[symbol]
        
        for i in range(1, len(history)):
            current = history[i]
            previous = history[i-1]
            
            tr = max(
                current['high'] - current['low'],
                abs(current['high'] - previous['close']),
                abs(current['low'] - previous['close'])
            )
            true_ranges.append(tr)
            
        # Return average
        return np.mean(true_ranges[-self.atr_period:]) if true_ranges else (high - low)
        
    def calculate_stop_distance(
        self,
        symbol: str,
        high: float,
        low: float,
        close: float
    ) -> float:
        """
        Calculate dynamic stop distance based on ATR.
        
        Args:
            symbol: Trading symbol
            high: High price
            low: Low price
            close: Close price
            
        Returns:
            Stop distance in price units
        """
        atr = self.calculate_atr(symbol, high, low, close)
        return atr * self.atr_multiplier


# Global instance
advanced_stop_loss = AdvancedStopLoss()
dynamic_stop_loss = DynamicStopLoss()
