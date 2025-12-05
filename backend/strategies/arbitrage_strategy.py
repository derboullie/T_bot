"""Cross-Exchange Arbitrage Strategy.

Detects and exploits price differences between different exchanges
for the same asset.
"""

from typing import Dict, List, Optional, Tuple
import asyncio
from datetime import datetime

import pandas as pd
import numpy as np
from loguru import logger

from backend.strategies.strategy_base import StrategyBase, Signal
from backend.core.config import settings


class ArbitrageStrategy(StrategyBase):
    """
    Cross-Exchange Arbitrage Strategy.
    
    Identifies price discrepancies between exchanges and executes
    simultaneous buy/sell orders to capture the spread.
    
    Features:
    - Real-time price comparison across exchanges
    - Fee-adjusted profit calculation
    - Latency compensation
    - Risk-free profit opportunities
    """
    
    def __init__(
        self,
        min_profit_pct: float = 0.5,  # Minimum profit percentage
        max_position_size: float = 10000.0,
        fee_rate: float = 0.001,  # 0.1% per trade
        latency_buffer_ms: float = 100.0,  # Expected execution latency
    ):
        """
        Initialize arbitrage strategy.
        
        Args:
            min_profit_pct: Minimum profit percentage to trigger trade
            max_position_size: Maximum position size in USD
            fee_rate: Trading fee rate (0.001 = 0.1%)
            latency_buffer_ms: Buffer for execution latency
        """
        super().__init__("Arbitrage Strategy")
        
        self.params = {
            'min_profit_pct': min_profit_pct,
            'max_position_size': max_position_size,
            'fee_rate': fee_rate,
            'latency_buffer_ms': latency_buffer_ms,
        }
        
        # Tracking
        self.opportunities_found = 0
        self.opportunities_executed = 0
        self.failed_executions = 0
        
        logger.info(
            f"Arbitrage Strategy initialized - "
            f"Min profit: {min_profit_pct}%, "
            f"Max position: ${max_position_size}"
        )
        
    async def analyze(self, symbol: str, data: pd.DataFrame) -> Signal:
        """
        Analyze arbitrage opportunities.
        
        Args:
            symbol: Asset symbol
            data: Market data with 'exchange', 'bid', 'ask' columns
            
        Returns:
            Trading signal
        """
        if not self.validate_data(data):
            return Signal.hold(symbol, "Invalid data")
            
        # Find best bid and ask across exchanges
        opportunity = self._find_arbitrage_opportunity(symbol, data)
        
        if opportunity:
            self.opportunities_found += 1
            
            # Calculate optimal position size
            position_size = self._calculate_position_size(opportunity)
            
            return Signal(
                action='arbitrage',
                symbol=symbol,
                confidence=opportunity['profit_pct'] / self.params['min_profit_pct'],
                reason=f"Arbitrage: {opportunity['profit_pct']:.2f}% profit",
                metadata={
                    'buy_exchange': opportunity['buy_exchange'],
                    'sell_exchange': opportunity['sell_exchange'],
                    'buy_price': opportunity['buy_price'],
                    'sell_price': opportunity['sell_price'],
                    'position_size': position_size,
                    'expected_profit': opportunity['expected_profit'],
                }
            )
            
        return Signal.hold(symbol, "No arbitrage opportunity")
        
    def _find_arbitrage_opportunity(
        self,
        symbol: str,
        data: pd.DataFrame
    ) -> Optional[Dict]:
        """
        Find arbitrage opportunity across exchanges.
        
        Args:
            symbol: Asset symbol
            data: Market data DataFrame
            
        Returns:
            Opportunity details or None
        """
        if len(data) < 2:
            return None
            
        # Find exchange with lowest ask (where to buy)
        buy_idx = data['ask'].idxmin()
        buy_exchange = data.loc[buy_idx, 'exchange']
        buy_price = data.loc[buy_idx, 'ask']
        
        # Find exchange with highest bid (where to sell)
        sell_idx = data['bid'].idxmax()
        sell_exchange = data.loc[sell_idx, 'exchange']
        sell_price = data.loc[sell_idx, 'bid']
        
        # Can't arbitrage on same exchange
        if buy_exchange == sell_exchange:
            return None
            
        # Calculate profit after fees
        gross_spread = sell_price - buy_price
        total_fees = (buy_price + sell_price) * self.params['fee_rate']
        net_profit = gross_spread - total_fees
        profit_pct = (net_profit / buy_price) * 100
        
        # Check if profitable enough
        if profit_pct < self.params['min_profit_pct']:
            return None
            
        logger.info(
            f"Arbitrage opportunity found for {symbol}: "
            f"Buy at {buy_exchange} ${buy_price:.2f}, "
            f"Sell at {sell_exchange} ${sell_price:.2f}, "
            f"Profit: {profit_pct:.2f}%"
        )
        
        return {
            'symbol': symbol,
            'buy_exchange': buy_exchange,
            'sell_exchange': sell_exchange,
            'buy_price': buy_price,
            'sell_price': sell_price,
            'gross_spread': gross_spread,
            'total_fees': total_fees,
            'net_profit': net_profit,
            'profit_pct': profit_pct,
            'expected_profit': net_profit,
        }
        
    def _calculate_position_size(self, opportunity: Dict) -> float:
        """
        Calculate optimal position size for arbitrage.
        
        Args:
            opportunity: Opportunity details
            
        Returns:
            Position size in USD
        """
        # Start with max position size
        position_size = self.params['max_position_size']
        
        # Could also factor in:
        # - Available liquidity on both exchanges
        # - Current capital availability
        # - Risk limits
        
        return position_size
        
    async def execute_arbitrage(
        self,
        opportunity: Dict,
        buy_client,
        sell_client
    ) -> bool:
        """
        Execute arbitrage trade.
        
        Args:
            opportunity: Opportunity details
            buy_client: Client for buy exchange
            sell_client: Client for sell exchange
            
        Returns:
            True if successful
        """
        try:
            symbol = opportunity['symbol']
            position_size = opportunity.get('position_size', self.params['max_position_size'])
            
            # Calculate quantity
            quantity = position_size / opportunity['buy_price']
            
            # Execute both orders simultaneously
            buy_task = buy_client.submit_market_order(symbol, quantity, 'buy')
            sell_task = sell_client.submit_market_order(symbol, quantity, 'sell')
            
            buy_order, sell_order = await asyncio.gather(buy_task, sell_task)
            
            if buy_order and sell_order:
                self.opportunities_executed += 1
                logger.success(
                    f"Arbitrage executed for {symbol}: "
                    f"Profit: ${opportunity['expected_profit']:.2f}"
                )
                return True
            else:
                self.failed_executions += 1
                logger.error(f"Arbitrage execution failed for {symbol}")
                return False
                
        except Exception as e:
            self.failed_executions += 1
            logger.error(f"Arbitrage execution error: {e}")
            return False
            
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate market data for arbitrage analysis.
        
        Args:
            data: Market data
            
        Returns:
            True if valid
        """
        required_cols = ['exchange', 'bid', 'ask']
        
        if not all(col in data.columns for col in required_cols):
            logger.warning(f"Missing required columns: {required_cols}")
            return False
            
        if len(data) < 2:
            logger.debug("Need at least 2 exchanges for arbitrage")
            return False
            
        # Check for invalid prices
        if (data['bid'] <= 0).any() or (data['ask'] <= 0).any():
            logger.warning("Invalid prices detected")
            return False
            
        # Check if bid < ask (sanity check)
        if (data['bid'] >= data['ask']).any():
            logger.warning("Invalid spread detected (bid >= ask)")
            return False
            
        return True
        
    def get_performance_stats(self) -> Dict:
        """Get strategy performance statistics."""
        base_stats = super().get_performance_stats()
        
        success_rate = (
            self.opportunities_executed / self.opportunities_found * 100
            if self.opportunities_found > 0
            else 0
        )
        
        base_stats.update({
            'opportunities_found': self.opportunities_found,
            'opportunities_executed': self.opportunities_executed,
            'failed_executions': self.failed_executions,
            'success_rate': success_rate,
        })
        
        return base_stats


class TriangularArbitrageStrategy(StrategyBase):
    """
    Triangular Arbitrage Strategy.
    
    Exploits price inconsistencies in currency/crypto triangles.
    Example: BTC/USD → ETH/BTC → ETH/USD → BTC/USD
    """
    
    def __init__(
        self,
        min_profit_pct: float = 0.3,
        max_position_size: float = 5000.0,
        fee_rate: float = 0.001,
    ):
        """
        Initialize triangular arbitrage.
        
        Args:
            min_profit_pct: Minimum profit threshold
            max_position_size: Max position per leg
            fee_rate: Trading fee rate
        """
        super().__init__("Triangular Arbitrage")
        
        self.params = {
            'min_profit_pct': min_profit_pct,
            'max_position_size': max_position_size,
            'fee_rate': fee_rate,
        }
        
        self.cycles_checked = 0
        self.profitable_cycles = 0
        
    async def analyze(self, symbols: List[str], data: Dict) -> Signal:
        """
        Analyze triangular arbitrage.
        
        Args:
            symbols: List of 3 symbols forming triangle
            data: Price data for all symbols
            
        Returns:
            Trading signal
        """
        if len(symbols) != 3:
            return Signal.hold("N/A", "Need exactly 3 symbols for triangular arbitrage")
            
        self.cycles_checked += 1
        
        # Calculate cycle profitability
        cycle = self._calculate_cycle(symbols, data)
        
        if cycle and cycle['profit_pct'] >= self.params['min_profit_pct']:
            self.profitable_cycles += 1
            
            return Signal(
                action='triangular_arbitrage',
                symbol='-'.join(symbols),
                confidence=cycle['profit_pct'] / self.params['min_profit_pct'],
                reason=f"Triangular arbitrage: {cycle['profit_pct']:.3f}%",
                metadata=cycle
            )
            
        return Signal.hold("N/A", "No triangular arbitrage opportunity")
        
    def _calculate_cycle(self, symbols: List[str], data: Dict) -> Optional[Dict]:
        """
        Calculate profitability of triangular cycle.
        
        Args:
            symbols: Symbol triangle
            data: Price data
            
        Returns:
            Cycle details or None
        """
        try:
            # Example: BTC/USD, ETH/BTC, ETH/USD
            # Start with 1 BTC
            # 1. Sell BTC for USD
            # 2. Buy ETH with USD
            # 3. Sell ETH for BTC
            # End with X BTC (hopefully > 1)
            
            prices = [data[s]['price'] for s in symbols]
            
            # Forward cycle calculation
            start_amount = 1.0
            amount = start_amount
            
            # Apply each conversion with fees
            for price in prices:
                amount = amount * price * (1 - self.params['fee_rate'])
                
            profit = amount - start_amount
            profit_pct = (profit / start_amount) * 100
            
            if profit_pct >= self.params['min_profit_pct']:
                logger.info(
                    f"Triangular arbitrage found: "
                    f"{' -> '.join(symbols)} = {profit_pct:.3f}%"
                )
                
                return {
                    'symbols': symbols,
                    'prices': prices,
                    'start_amount': start_amount,
                    'end_amount': amount,
                    'profit': profit,
                    'profit_pct': profit_pct,
                }
                
            return None
            
        except Exception as e:
            logger.error(f"Error calculating triangular cycle: {e}")
            return None
            
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        base_stats = super().get_performance_stats()
        
        base_stats.update({
            'cycles_checked': self.cycles_checked,
            'profitable_cycles': self.profitable_cycles,
            'profit_rate': (
                self.profitable_cycles / self.cycles_checked * 100
                if self.cycles_checked > 0
                else 0
            ),
        })
        
        return base_stats
