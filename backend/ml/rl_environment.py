"""Reinforcement Learning environment for trading."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from loguru import logger

from backend.core.config import settings


class TradingEnvironment(gym.Env):
    """
    OpenAI Gym-compatible trading environment for RL agents.
    
    State space includes:
    - Order book depth (bid/ask levels)
    - Price movements (OHLCV)
    - Technical indicators (RSI, MACD, etc.)
    - Position status
    - Portfolio metrics
    
    Action space:
    - 0: Hold
    - 1: Buy small (10% of position limit)
    - 2: Buy medium (25%)
    - 3: Buy large (50%)
    - 4: Sell small (10%)
    - 5: Sell medium (25%)
    - 6: Sell large (50%)
    - 7: Close all positions
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 100000.0,
        transaction_cost_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        max_position_size: float = 10000.0,
        lookback_window: int = 50,
    ):
        """
        Initialize trading environment.
        
        Args:
            data: Historical OHLCV data with indicators
            initial_balance: Starting cash
            transaction_cost_pct: Transaction cost as percentage
            slippage_pct: Slippage as percentage
            max_position_size: Maximum position size in USD
            lookback_window: Number of past bars to include in state
        """
        super().__init__()
        
        self.data = data.copy()
        self.initial_balance = initial_balance
        self.transaction_cost_pct = transaction_cost_pct
        self.slippage_pct = slippage_pct
        self.max_position_size = max_position_size
        self.lookback_window = lookback_window
        
        # State dimensions
        # Price features: open, high, low, close, volume (5)
        # Technical indicators: RSI, MACD, signal, histogram, SMA20, SMA50 (6)
        # Order book: bid_price, ask_price, bid_volume, ask_volume, spread (5)
        # Portfolio: position_size, unrealized_pnl, cash, total_value (4)
        # Total: 20 features * lookback_window
        self.n_features = 20
        self.state_dim = self.n_features * lookback_window
        
        # Action space: 8 discrete actions
        self.action_space = spaces.Discrete(8)
        
        # Observation space: continuous values
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32
        )
        
        # Initialize state
        self.reset()
        
    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state.
        
        Returns:
            Initial observation
        """
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.position = 0.0  # Number of shares
        self.entry_price = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        
        # Performance tracking
        self.equity_curve = [self.initial_balance]
        self.trade_history = []
        
        return self._get_observation()
        
    def _get_observation(self) -> np.ndarray:
        """
        Get current state observation.
        
        Returns:
            State vector
        """
        # Get historical window
        start_idx = max(0, self.current_step - self.lookback_window)
        end_idx = self.current_step
        
        window_data = self.data.iloc[start_idx:end_idx]
        
        # Normalize price features
        price_features = window_data[['open', 'high', 'low', 'close', 'volume']].values
        price_features = self._normalize(price_features)
        
        # Get technical indicators (assuming they're precomputed in data)
        if 'rsi' in window_data.columns:
            indicators = window_data[['rsi']].values / 100.0  # Normalize RSI to 0-1
        else:
            indicators = np.zeros((len(window_data), 1))
            
        # Simulated order book features (in real trading, get from live data)
        current_price = window_data['close'].iloc[-1] if len(window_data) > 0 else 0
        order_book = np.array([
            current_price * 0.9995,  # bid_price
            current_price * 1.0005,  # ask_price
            1000.0,  # bid_volume (simulated)
            1000.0,  # ask_volume (simulated)
            current_price * 0.001,  # spread
        ]).reshape(1, -1).repeat(len(window_data), axis=0)
        
        # Portfolio features
        current_value = self.balance + (self.position * current_price)
        unrealized_pnl = self.position * (current_price - self.entry_price) if self.position != 0 else 0
        
        portfolio = np.array([
            self.position / 100.0,  # Normalized position
            unrealized_pnl / self.initial_balance,  # Normalized P&L
            self.balance / self.initial_balance,  # Normalized cash
            current_value / self.initial_balance,  # Normalized total value
        ]).reshape(1, -1).repeat(len(window_data), axis=0)
        
        # Combine all features
        features = np.concatenate([
            price_features,
            indicators,
            order_book,
            portfolio
        ], axis=1)
        
        # Flatten to 1D vector
        observation = features.flatten()
        
        # Pad if necessary
        if len(observation) < self.state_dim:
            observation = np.pad(observation, (0, self.state_dim - len(observation)))
        
        return observation.astype(np.float32)
        
    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize data using z-score."""
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0) + 1e-8
        return (data - mean) / std
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action and return result.
        
        Args:
            action: Action to take
            
        Returns:
            observation, reward, done, info
        """
        # Get current price
        current_price = self.data.iloc[self.current_step]['close']
        
        # Execute action
        reward = 0.0
        executed = False
        
        if action == 0:  # Hold
            pass
            
        elif action in [1, 2, 3]:  # Buy actions
            buy_pcts = {1: 0.10, 2: 0.25, 3: 0.50}
            buy_pct = buy_pcts[action]
            
            if self.position >= 0:  # Can only buy if not short
                position_value = buy_pct * self.max_position_size
                shares_to_buy = position_value / current_price
                
                # Apply slippage (buy at higher price)
                execution_price = current_price * (1 + self.slippage_pct)
                cost = shares_to_buy * execution_price
                transaction_cost = cost * self.transaction_cost_pct
                
                total_cost = cost + transaction_cost
                
                if total_cost <= self.balance:
                    # Update position (average price if adding to existing)
                    if self.position > 0:
                        total_shares = self.position + shares_to_buy
                        self.entry_price = (
                            (self.position * self.entry_price) + 
                            (shares_to_buy * execution_price)
                        ) / total_shares
                        self.position = total_shares
                    else:
                        self.position = shares_to_buy
                        self.entry_price = execution_price
                        
                    self.balance -= total_cost
                    executed = True
                    
        elif action in [4, 5, 6]:  # Sell actions
            sell_pcts = {4: 0.10, 5: 0.25, 6: 0.50}
            sell_pct = sell_pcts[action]
            
            if self.position > 0:  # Can only sell if long
                shares_to_sell = self.position * sell_pct
                
                # Apply slippage (sell at lower price)
                execution_price = current_price * (1 - self.slippage_pct)
                proceeds = shares_to_sell * execution_price
                transaction_cost = proceeds * self.transaction_cost_pct
                
                net_proceeds = proceeds - transaction_cost
                
                # Calculate P&L
                pnl = shares_to_sell * (execution_price - self.entry_price)
                
                self.position -= shares_to_sell
                self.balance += net_proceeds
                self.total_pnl += pnl
                
                # Track trade
                self.total_trades += 1
                if pnl > 0:
                    self.winning_trades += 1
                    
                self.trade_history.append({
                    'step': self.current_step,
                    'action': 'sell',
                    'shares': shares_to_sell,
                    'price': execution_price,
                    'pnl': pnl
                })
                
                executed = True
                
                if self.position < 0.01:  # Close position if very small
                    self.position = 0.0
                    self.entry_price = 0.0
                    
        elif action == 7:  # Close all
            if self.position > 0:
                execution_price = current_price * (1 - self.slippage_pct)
                proceeds = self.position * execution_price
                transaction_cost = proceeds * self.transaction_cost_pct
                net_proceeds = proceeds - transaction_cost
                
                pnl = self.position * (execution_price - self.entry_price)
                
                self.balance += net_proceeds
                self.total_pnl += pnl
                
                self.total_trades += 1
                if pnl > 0:
                    self.winning_trades += 1
                    
                self.trade_history.append({
                    'step': self.current_step,
                    'action': 'close',
                    'shares': self.position,
                    'price': execution_price,
                    'pnl': pnl
                })
                
                self.position = 0.0
                self.entry_price = 0.0
                executed = True
                
        # Calculate reward
        current_value = self.balance + (self.position * current_price)
        self.equity_curve.append(current_value)
        
        # Multi-component reward function
        reward = self._calculate_reward(current_value, executed)
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        # Get new observation
        observation = self._get_observation()
        
        # Info dict
        info = {
            'balance': self.balance,
            'position': self.position,
            'current_value': current_value,
            'total_pnl': self.total_pnl,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
        }
        
        return observation, reward, done, info
        
    def _calculate_reward(self, current_value: float, executed: bool) -> float:
        """
        Calculate multi-component reward.
        
        Components:
        1. Portfolio return
        2. Sharpe ratio (risk-adjusted return)
        3. Trade execution quality
        4. Drawdown penalty
        
        Args:
            current_value: Current portfolio value
            executed: Whether trade was executed
            
        Returns:
            Reward value
        """
        # 1. Portfolio return
        returns = (current_value - self.initial_balance) / self.initial_balance
        
        # 2. Sharpe ratio approximation (using recent equity curve)
        if len(self.equity_curve) > 10:
            recent_returns = np.diff(self.equity_curve[-10:]) / self.equity_curve[-11:-1]
            sharpe = np.mean(recent_returns) / (np.std(recent_returns) + 1e-8) * np.sqrt(252)
        else:
            sharpe = 0.0
            
        # 3. Trade quality (win rate)
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0.5
        
        # 4. Drawdown penalty
        peak = max(self.equity_curve)
        drawdown = (peak - current_value) / peak if peak > 0 else 0
        
        # Combine components
        reward = (
            returns * 1.0 +           # Portfolio return  weight
            sharpe * 0.5 +            # Risk-adjusted return weight
            (win_rate - 0.5) * 0.3 -  # Win rate bonus (above 50%)
            drawdown * 2.0            # Drawdown penalty
        )
        
        # Small penalty for inaction to encourage trading
        if not executed:
            reward -= 0.001
            
        return reward
        
    def render(self, mode='human'):
        """Render environment state."""
        current_value = self.balance + (
            self.position * self.data.iloc[self.current_step]['close']
        )
        
        logger.info(
            f"Step: {self.current_step} | "
            f"Value: ${current_value:.2f} | "
            f"P&L: ${self.total_pnl:.2f} | "
            f"Position: {self.position:.2f} | "
            f"Trades: {self.total_trades} | "
            f"Win Rate: {self.winning_trades/self.total_trades*100 if self.total_trades > 0 else 0:.1f}%"
        )
        
    def get_performance_metrics(self) -> Dict:
        """Calculate performance metrics."""
        final_value = self.equity_curve[-1]
        total_return = (final_value - self.initial_balance) / self.initial_balance
        
        # Sharpe ratio
        returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        
        # Max drawdown
        peak = np.maximum.accumulate(self.equity_curve)
        drawdown = (peak - self.equity_curve) / peak
        max_drawdown = np.max(drawdown)
        
        # Win rate
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        
        return {
            'final_value': final_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
        }
