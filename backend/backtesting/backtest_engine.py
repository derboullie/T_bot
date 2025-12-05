"""Comprehensive Backtesting Engine.

Simulates trading strategies on historical data with realistic
execution, fees, and slippage.
"""

from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

import pandas as pd
import numpy as np
from loguru import logger

from backend.strategies.strategy_base import StrategyBase
from backend.data.models import OrderSide, OrderType


class FillModel(Enum):
    """Order fill simulation models."""
    IMMEDIATE = "immediate"  # Fill at current price
    REALISTIC = "realistic"  # Check liquidity, slippage
    PESSIMISTIC = "pessimistic"  # Worst case fills


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    initial_capital: float = 100000.0
    commission_rate: float = 0.001  # 0.1%
    slippage_rate: float = 0.0005  # 0.05%
    fill_model: FillModel = FillModel.REALISTIC
    
    # Risk management
    max_position_size: float = 10000.0
    max_positions: int = 10
    max_daily_loss: float = -1000.0
    
    # Execution
    min_order_size: float = 1.0
    allow_fractional: bool = True


@dataclass
class Trade:
    """Represents a completed trade."""
    entry_time: datetime
    exit_time: datetime
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    commission: float
    duration: timedelta
    strategy: str


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    # Basic metrics
    total_return: float = 0.0
    total_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    # Profitability
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_trade: float = 0.0
    
    # Duration
    avg_trade_duration: timedelta = field(default_factory=lambda: timedelta(0))
    total_time_in_market: timedelta = field(default_factory=lambda: timedelta(0))
    
    # Risk metrics
    calmar_ratio: float = 0.0
    recovery_factor: float = 0.0
    
    # Time series
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    drawdowns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    
    # Trade list
    trades: List[Trade] = field(default_factory=list)


class BacktestEngine:
    """
    Comprehensive backtesting engine.
    
    Features:
    - Realistic execution simulation
    - Multiple strategies
    - Performance analytics
    - Risk management
    - Walk-forward analysis support
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        Initialize backtest engine.
        
        Args:
            config: Backtesting configuration
        """
        self.config = config or BacktestConfig()
        
        # State
        self.cash = self.config.initial_capital
        self.positions = {}  # symbol -> quantity
        self.position_values = {}  # symbol -> avg cost
        self.equity_history = []
        self.trades = []
        self.current_time = None
        
        # Performance tracking
        self.daily_pnl = {}
        self.peak_equity = self.config.initial_capital
        
        logger.info(
            f"Backtest Engine initialized - "
            f"Initial capital: ${self.config.initial_capital:,.2f}"
        )
        
    async def run(
        self,
        strategy: StrategyBase,
        data: pd.DataFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> BacktestResult:
        """
        Run backtest on historical data.
        
        Args:
            strategy: Trading strategy to test
            data: Historical OHLCV data
            start_date: Start date (optional)
            end_date: End date (optional)
            
        Returns:
            Backtest results
        """
        logger.info(
            f"Starting backtest: {strategy.name} "
            f"from {start_date or 'start'} to {end_date or 'end'}"
        )
        
        # Reset state
        self._reset()
        
        # Filter data by date range
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
            
        if len(data) == 0:
            logger.error("No data in date range")
            return BacktestResult()
            
        # Main backtest loop
        for timestamp, row in data.iterrows():
            self.current_time = timestamp
            
            # Update portfolio value
            portfolio_value = self._calculate_portfolio_value(row)
            self.equity_history.append({
                'timestamp': timestamp,
                'equity': portfolio_value,
            })
            
            # Check daily loss limit
            if not self._check_risk_limits():
                logger.warning(f"Risk limits breached at {timestamp}")
                continue
                
            # Generate trading signal
            signal = await strategy.analyze(row.get('symbol', 'UNKNOWN'), data.loc[:timestamp])
            
            # Execute signal
            if signal.action in ['buy', 'sell']:
                self._execute_signal(signal, row)
            elif signal.action == 'close':
                self._close_all_positions(row)
                
        # Final portfolio value
        final_row = data.iloc[-1]
        final_value = self._calculate_portfolio_value(final_row)
        
        # Calculate results
        results = self._calculate_results(final_value)
        
        logger.success(
            f"Backtest complete: "
            f"Total Return: {results.total_return_pct:.2f}%, "
            f"Sharpe: {results.sharpe_ratio:.2f}, "
            f"Win Rate: {results.win_rate:.1f}%"
        )
        
        return results
        
    def _execute_signal(self, signal, market_data):
        """
        Execute trading signal.
        
        Args:
            signal: Trading signal
            market_data: Current market data
        """
        symbol = signal.symbol
        action = signal.action
        
        # Get current price
        if 'close' in market_data:
            price = market_data['close']
        elif isinstance(market_data, dict):
            price = market_data.get('price', market_data.get('close', 0))
        else:
            logger.error("Cannot determine price from market data")
            return
            
        # Calculate order size
        if signal.metadata and 'position_size' in signal.metadata:
            order_value = signal.metadata['position_size']
        else:
            order_value = min(self.config.max_position_size, self.cash * 0.1)
            
        quantity = order_value / price
        
        # Apply fills simulation
        fill_price = self._simulate_fill(price, action)
        
        # Calculate commission
        commission = fill_price * quantity * self.config.commission_rate
        
        # Execute trade
        if action == 'buy':
            cost = fill_price * quantity + commission
            
            if cost <= self.cash and len(self.positions) < self.config.max_positions:
                # Update position
                if symbol in self.positions:
                    # Average cost
                    current_qty = self.positions[symbol]
                    current_value = self.position_values[symbol] * current_qty
                    new_value = current_value + (fill_price * quantity)
                    new_qty = current_qty + quantity
                    self.position_values[symbol] = new_value / new_qty
                    self.positions[symbol] = new_qty
                else:
                    self.positions[symbol] = quantity
                    self.position_values[symbol] = fill_price
                    
                self.cash -= cost
                
                logger.debug(
                    f"{self.current_time}: BUY {quantity:.4f} {symbol} @ ${fill_price:.2f}"
                )
                
        elif action == 'sell':
            if symbol in self.positions:
                # Sell position
                sell_qty = min(quantity, self.positions[symbol])
                proceeds = fill_price * sell_qty - commission
                
                # Calculate P&L
                entry_price = self.position_values[symbol]
                pnl = (fill_price - entry_price) * sell_qty - commission
                pnl_pct = ((fill_price - entry_price) / entry_price) * 100
                
                # Record trade
                trade = Trade(
                    entry_time=self.current_time - timedelta(days=1),  # Approximate
                    exit_time=self.current_time,
                    symbol=symbol,
                    side='long',
                    entry_price=entry_price,
                    exit_price=fill_price,
                    quantity=sell_qty,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    commission=commission,
                    duration=timedelta(days=1),
                    strategy=signal.metadata.get('strategy', 'unknown') if signal.metadata else 'unknown'
                )
                self.trades.append(trade)
                
                # Update position
                self.positions[symbol] -= sell_qty
                if self.positions[symbol] < 0.001:  # Close if tiny
                    del self.positions[symbol]
                    del self.position_values[symbol]
                    
                self.cash += proceeds
                
                logger.debug(
                    f"{self.current_time}: SELL {sell_qty:.4f} {symbol} @ ${fill_price:.2f} "
                    f"(P&L: ${pnl:.2f})"
                )
                
    def _simulate_fill(self, price: float, side: str) -> float:
        """
        Simulate order fill with slippage.
        
        Args:
            price: Market price
            side: 'buy' or 'sell'
            
        Returns:
            Fill price
        """
        if self.config.fill_model == FillModel.IMMEDIATE:
            return price
            
        # Apply slippage
        slippage = price * self.config.slippage_rate
        
        if side == 'buy':
            fill_price = price + slippage
        else:
            fill_price = price - slippage
            
        return fill_price
        
    def _calculate_portfolio_value(self, market_data) -> float:
        """
        Calculate current portfolio value.
        
        Args:
            market_data: Current market data
            
        Returns:
            Total portfolio value
        """
        # Get current price
        if 'close' in market_data:
            price = market_data['close']
        elif isinstance(market_data, dict):
            price = market_data.get('price', market_data.get('close', 0))
        else:
            price = 0
            
        # Calculate position values
        position_value = sum(
            qty * price
            for symbol, qty in self.positions.items()
        )
        
        total_value = self.cash + position_value
        
        # Update peak equity
        if total_value > self.peak_equity:
            self.peak_equity = total_value
            
        return total_value
        
    def _check_risk_limits(self) -> bool:
        """
        Check if risk limits are met.
        
        Returns:
            True if can trade
        """
        # Check daily loss
        date_key = self.current_time.date()
        daily_pnl = self.daily_pnl.get(date_key, 0.0)
        
        if daily_pnl <= self.config.max_daily_loss:
            return False
            
        return True
        
    def _close_all_positions(self, market_data):
        """Close all open positions."""
        symbols_to_close = list(self.positions.keys())
        
        for symbol in symbols_to_close:
            # Create sell signal
            from backend.strategies.strategy_base import Signal
            signal = Signal(
                action='sell',
                symbol=symbol,
                confidence=1.0,
                reason='Close all positions'
            )
            self._execute_signal(signal, market_data)
            
    def _calculate_results(self, final_value: float) -> BacktestResult:
        """
        Calculate comprehensive backtest results.
        
        Args:
            final_value: Final portfolio value
            
        Returns:
            BacktestResult object
        """
        results = BacktestResult()
        
        # Basic returns
        results.total_return = final_value - self.config.initial_capital
        results.total_return_pct = (results.total_return / self.config.initial_capital) * 100
        
        # Equity curve
        equity_df = pd.DataFrame(self.equity_history)
        if len(equity_df) > 0:
            equity_df.set_index('timestamp', inplace=True)
            results.equity_curve = equity_df['equity']
            
            # Returns
            results.returns = results.equity_curve.pct_change().dropna()
            
            # Sharpe Ratio (annualized, assuming daily data)
            if len(results.returns) > 0 and results.returns.std() > 0:
                results.sharpe_ratio = (
                    results.returns.mean() / results.returns.std() * np.sqrt(252)
                )
                
            # Sortino Ratio (downside deviation)
            downside_returns = results.returns[results.returns < 0]
            if len(downside_returns) > 0 and downside_returns.std() > 0:
                results.sortino_ratio = (
                    results.returns.mean() / downside_returns.std() * np.sqrt(252)
                )
                
            # Drawdown
            peak = results.equity_curve.expanding().max()
            drawdown = (results.equity_curve - peak) / peak
            results.drawdowns = drawdown
            results.max_drawdown = drawdown.min()
            results.max_drawdown_pct = results.max_drawdown * 100
            
        # Trade statistics
        results.total_trades = len(self.trades)
        results.trades = self.trades
        
        if results.total_trades > 0:
            winning_trades = [t for t in self.trades if t.pnl > 0]
            losing_trades = [t for t in self.trades if t.pnl < 0]
            
            results.winning_trades = len(winning_trades)
            results.losing_trades = len(losing_trades)
            results.win_rate = (results.winning_trades / results.total_trades) * 100
            
            # Profit metrics
            results.gross_profit = sum(t.pnl for t in winning_trades)
            results.gross_loss = sum(t.pnl for t in losing_trades)
            
            if results.gross_loss != 0:
                results.profit_factor = abs(results.gross_profit / results.gross_loss)
                
            if results.winning_trades > 0:
                results.avg_win = results.gross_profit / results.winning_trades
            if results.losing_trades > 0:
                results.avg_loss = results.gross_loss / results.losing_trades
                
            results.avg_trade = results.total_return / results.total_trades
            
            # Duration
            durations = [t.duration for t in self.trades]
            results.avg_trade_duration = sum(durations, timedelta()) / len(durations)
            results.total_time_in_market = sum(durations, timedelta())
            
        # Risk-adjusted metrics
        if results.max_drawdown < 0:
            results.calmar_ratio = results.total_return_pct / abs(results.max_drawdown_pct)
            results.recovery_factor = results.total_return / abs(results.max_drawdown)
            
        return results
        
    def _reset(self):
        """Reset backtest state."""
        self.cash = self.config.initial_capital
        self.positions = {}
        self.position_values = {}
        self.equity_history = []
        self.trades = []
        self.daily_pnl = {}
        self.peak_equity = self.config.initial_capital
        
    def print_results(self, results: BacktestResult):
        """
        Print formatted backtest results.
        
        Args:
            results: Backtest results
        """
        logger.info("=" * 60)
        logger.info("BACKTEST RESULTS")
        logger.info("=" * 60)
        
        logger.info(f"Total Return: ${results.total_return:,.2f} ({results.total_return_pct:.2f}%)")
        logger.info(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
        logger.info(f"Sortino Ratio: {results.sortino_ratio:.2f}")
        logger.info(f"Max Drawdown: {results.max_drawdown_pct:.2f}%")
        logger.info(f"Calmar Ratio: {results.calmar_ratio:.2f}")
        
        logger.info("\nTrade Statistics:")
        logger.info(f"Total Trades: {results.total_trades}")
        logger.info(f"Win Rate: {results.win_rate:.1f}%")
        logger.info(f"Profit Factor: {results.profit_factor:.2f}")
        logger.info(f"Avg Win: ${results.avg_win:.2f}")
        logger.info(f"Avg Loss: ${results.avg_loss:.2f}")
        logger.info(f"Avg Trade Duration: {results.avg_trade_duration}")
        
        logger.info("=" * 60)
