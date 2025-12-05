"""Market-Making Strategy.

Provides liquidity by continuously quoting bid and ask prices,
profiting from the spread while managing inventory risk.
"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass

import pandas as pd
import numpy as np
from loguru import logger

from backend.strategies.strategy_base import StrategyBase, Signal


@dataclass
class Quote:
    """Market maker quote."""
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    spread: float
    mid_price: float


class MarketMakingStrategy(StrategyBase):
    """
    Market-Making Strategy.
    
    Continuously provides liquidity by placing limit orders on both
    sides of the order book, profiting from bid-ask spread.
    
    Features:
    - Dynamic spread adjustment based on volatility
    - Inventory management (avoid one-sided positions)
    - Adverse selection protection
    - Liquidity-based sizing
    """
    
    def __init__(
        self,
        base_spread_bps: float = 10.0,  # 10 basis points (0.1%)
        max_spread_bps: float = 100.0,  # Max 1% spread
        target_inventory: float = 0.0,  # Neutral position
        max_inventory: float = 100.0,  # Max position size
        volatility_multiplier: float = 2.0,  # Spread widens with volatility
        quote_size: float = 1000.0,  # Base quote size in USD
    ):
        """
        Initialize market-making strategy.
        
        Args:
            base_spread_bps: Base spread in basis points
            max_spread_bps: Maximum spread in basis points
            target_inventory: Target inventory level
            max_inventory: Maximum inventory
            volatility_multiplier: Volatility spread multiplier
            quote_size: Base quote size
        """
        super().__init__("Market Making")
        
        self.params = {
            'base_spread_bps': base_spread_bps,
            'max_spread_bps': max_spread_bps,
            'target_inventory': target_inventory,
            'max_inventory': max_inventory,
            'volatility_multiplier': volatility_multiplier,
            'quote_size': quote_size,
        }
        
        # Current state
        self.current_inventory = 0.0
        self.active_quotes = []
        self.fills_count = 0
        self.total_spread_captured = 0.0
        
        logger.info(
            f"Market Making Strategy initialized - "
            f"Base spread: {base_spread_bps}bps, "
            f"Quote size: ${quote_size}"
        )
        
    async def analyze(self, symbol: str, data: pd.DataFrame) -> Signal:
        """
        Generate market-making quotes.
        
        Args:
            symbol: Asset symbol
            data: Market data with OHLCV and order book depth
            
        Returns:
            Trading signal with quote information
        """
        if not self.validate_data(data):
            return Signal.hold(symbol, "Invalid data")
            
        # Calculate current market conditions
        mid_price = data['close'].iloc[-1]
        volatility = self._calculate_volatility(data)
        
        # Generate optimal quote
        quote = self._generate_quote(
            mid_price,
            volatility,
            self.current_inventory
        )
        
        # Check inventory limits
        if abs(self.current_inventory) >= self.params['max_inventory']:
            return Signal(
                action='reduce_inventory',
                symbol=symbol,
                confidence=0.9,
                reason="Inventory limit reached",
                metadata={'current_inventory': self.current_inventory}
            )
            
        return Signal(
            action='market_make',
            symbol=symbol,
            confidence=0.8,
            reason=f"Market making: spread {quote.spread:.4f}",
            metadata={
                'bid_price': quote.bid_price,
                'ask_price': quote.ask_price,
                'bid_size': quote.bid_size,
                'ask_size': quote.ask_size,
                'spread': quote.spread,
                'spread_bps': quote.spread / mid_price * 10000,
                'mid_price': quote.mid_price,
                'volatility': volatility,
                'inventory': self.current_inventory,
            }
        )
        
    def _generate_quote(
        self,
        mid_price: float,
        volatility: float,
        inventory: float
    ) -> Quote:
        """
        Generate bid/ask quote.
        
        Args:
            mid_price: Current mid-market price
            volatility: Market volatility
            inventory: Current inventory
            
        Returns:
            Quote object
        """
        # Base spread in price units
        base_spread = mid_price * (self.params['base_spread_bps'] / 10000)
        
        # Adjust spread for volatility
        volatility_adjustment = volatility * self.params['volatility_multiplier']
        adjusted_spread = base_spread * (1 + volatility_adjustment)
        
        # Cap at maximum spread
        max_spread = mid_price * (self.params['max_spread_bps'] / 10000)
        spread = min(adjusted_spread, max_spread)
        
        # Adjust for inventory skew
        inventory_skew = self._calculate_inventory_skew(inventory)
        
        # Calculate bid and ask
        half_spread = spread / 2
        bid_price = mid_price - half_spread + inventory_skew
        ask_price = mid_price + half_spread + inventory_skew
        
        # Calculate quote sizes (reduce size if inventory is skewed)
        inventory_ratio = abs(inventory) / self.params['max_inventory']
        size_multiplier = 1.0 - (inventory_ratio * 0.5)  # Reduce by up to 50%
        
        base_size = self.params['quote_size'] / mid_price
        
        # If long inventory, prefer to sell (larger ask size)
        # If short inventory, prefer to buy (larger bid size)
        if inventory > 0:
            bid_size = base_size * size_multiplier * 0.5
            ask_size = base_size * size_multiplier * 1.5
        elif inventory < 0:
            bid_size = base_size * size_multiplier * 1.5
            ask_size = base_size * size_multiplier * 0.5
        else:
            bid_size = ask_size = base_size * size_multiplier
            
        return Quote(
            bid_price=bid_price,
            ask_price=ask_price,
            bid_size=bid_size,
            ask_size=ask_size,
            spread=spread,
            mid_price=mid_price
        )
        
    def _calculate_inventory_skew(self, inventory: float) -> float:
        """
        Calculate price skew based on inventory.
        
        Skew quotes away from accumulating more inventory:
        - If long, push prices down (lower bid/ask)
        - If short, push prices up (higher bid/ask)
        
        Args:
            inventory: Current inventory
            
        Returns:
            Price skew adjustment
        """
        target = self.params['target_inventory']
        max_inv = self.params['max_inventory']
        
        # Normalized inventory (-1 to 1)
        normalized_inv = (inventory - target) / max_inv
        
        # Skew proportional to inventory deviation
        # Max skew is 50% of base spread
        base_spread_price = self.params['base_spread_bps'] / 10000
        max_skew = base_spread_price * 0.5
        
        skew = -normalized_inv * max_skew
        
        return skew
        
    def _calculate_volatility(self, data: pd.DataFrame, window: int = 20) -> float:
        """
        Calculate recent volatility.
        
        Args:
            data: Price data
            window: Lookback window
            
        Returns:
            Annualized volatility
        """
        if len(data) < window:
            return 0.0
            
        returns = data['close'].pct_change().tail(window)
        volatility = returns.std() * np.sqrt(252)  # Annualized
        
        return volatility
        
    def on_fill(self, side: str, price: float, size: float):
        """
        Update state when order fills.
        
        Args:
            side: 'buy' or 'sell'
            price: Fill price
            size: Fill size
        """
        self.fills_count += 1
        
        if side == 'buy':
            self.current_inventory += size
        else:
            self.current_inventory -= size
            
        logger.info(
            f"Order filled: {side} {size:.4f} @ ${price:.2f}, "
            f"Inventory: {self.current_inventory:.4f}"
        )
        
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate market data.
        
        Args:
            data: Market data
            
        Returns:
            True if valid
        """
        required_cols = ['close', 'high', 'low']
        
        if not all(col in data.columns for col in required_cols):
            return False
            
        if len(data) < 20:  # Need history for volatility
            return False
            
        return True
        
    def get_performance_stats(self) -> Dict:
        """Get strategy performance stats."""
        base_stats = super().get_performance_stats()
        
        avg_spread = (
            self.total_spread_captured / self.fills_count
            if self.fills_count > 0
            else 0
        )
        
        base_stats.update({
            'current_inventory': self.current_inventory,
            'fills_count': self.fills_count,
            'total_spread_captured': self.total_spread_captured,
            'avg_spread_captured': avg_spread,
            'active_quotes': len(self.active_quotes),
        })
        
        return base_stats
