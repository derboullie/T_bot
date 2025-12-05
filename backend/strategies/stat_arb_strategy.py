"""Statistical Arbitrage Strategies.

Mean reversion and pairs trading strategies based on statistical models.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import pandas as pd
import numpy as np
from scipy import stats
from loguru import logger

from backend.strategies.strategy_base import StrategyBase, Signal


@dataclass
class PairStats:
    """Statistics for a trading pair."""
    correlation: float
    cointegration_pvalue: float
    spread_mean: float
    spread_std: float
    z_score: float
    half_life: float


class StatisticalArbitrageStrategy(StrategyBase):
    """
    Statistical Arbitrage Strategy.
    
    Uses mean reversion and pairs trading based on:
    - Cointegration between asset pairs
    - Z-score thresholds for entry/exit
    - Half-life for mean reversion speed
    
    Theory: When two cointegrated assets diverge,
    they tend to revert to their historical relationship.
    """
    
    def __init__(
        self,
        entry_z_score: float = 2.0,  # Enter at ±2 std dev
        exit_z_score: float = 0.5,   # Exit at ±0.5 std dev
        stop_loss_z_score: float = 3.5,  # Stop at ±3.5 std dev
        lookback_period: int = 60,  # Days for cointegration
        min_correlation: float = 0.7,  # Minimum correlation
        max_pvalue: float = 0.05,  # Max p-value for cointegration
    ):
        """
        Initialize statistical arbitrage strategy.
        
        Args:
            entry_z_score: Z-score threshold for entry
            exit_z_score: Z-score threshold for exit
            stop_loss_z_score: Z-score for stop loss
            lookback_period: Historical lookback period
            min_correlation: Minimum correlation required
            max_pvalue: Maximum p-value for cointegration test
        """
        super().__init__("Statistical Arbitrage")
        
        self.params = {
            'entry_z_score': entry_z_score,
            'exit_z_score': exit_z_score,
            'stop_loss_z_score': stop_loss_z_score,
            'lookback_period': lookback_period,
            'min_correlation': min_correlation,
            'max_pvalue': max_pvalue,
        }
        
        # State tracking
        self.pairs_analyzed = 0
        self.cointegrated_pairs = 0
        self.trades_opened = 0
        
        logger.info(
            f"Statistical Arbitrage initialized - "
            f"Entry Z: ±{entry_z_score}, "
            f"Exit Z: ±{exit_z_score}"
        )
        
    async def analyze_pair(
        self,
        symbol_a: str,
        symbol_b: str,
        data_a: pd.Series,
        data_b: pd.Series
    ) -> Signal:
        """
        Analyze pair for statistical arbitrage.
        
        Args:
            symbol_a: First symbol
            symbol_b: Second symbol
            data_a: Price series for symbol A
            data_b: Price series for symbol B
            
        Returns:
            Trading signal
        """
        self.pairs_analyzed += 1
        
        # Calculate pair statistics
        pair_stats = self._calculate_pair_stats(data_a, data_b)
        
        if not pair_stats:
            return Signal.hold(
                f"{symbol_a}/{symbol_b}",
                "Insufficient data or not cointegrated"
            )
            
        # Check if pair is cointegrated
        if not self._is_cointegrated(pair_stats):
            return Signal.hold(
                f"{symbol_a}/{symbol_b}",
                f"Not cointegrated (p={pair_stats.cointegration_pvalue:.3f})"
            )
            
        self.cointegrated_pairs += 1
        
        # Generate trading signal based on z-score
        signal = self._generate_pair_signal(
            symbol_a,
            symbol_b,
            pair_stats
        )
        
        return signal
        
    def _calculate_pair_stats(
        self,
        series_a: pd.Series,
        series_b: pd.Series
    ) -> Optional[PairStats]:
        """
        Calculate statistical properties of pair.
        
        Args:
            series_a: Price series A
            series_b: Price series B
            
        Returns:
            Pair statistics or None
        """
        try:
            # Align series
            df = pd.DataFrame({'a': series_a, 'b': series_b}).dropna()
            
            if len(df) < self.params['lookback_period']:
                return None
                
            # Use recent data
            df = df.tail(self.params['lookback_period'])
            
            # Calculate correlation
            correlation = df['a'].corr(df['b'])
            
            # Cointegration test (Engle-Granger)
            spread = df['a'] - df['b']
            
            # ADF test on spread
            from statsmodels.tsa.stattools import adfuller
            adf_result = adfuller(spread)
            cointegration_pvalue = adf_result[1]
            
            # Spread statistics
            spread_mean = spread.mean()
            spread_std = spread.std()
            
            # Current z-score
            current_spread = df['a'].iloc[-1] - df['b'].iloc[-1]
            z_score = (current_spread - spread_mean) / spread_std if spread_std > 0 else 0
            
            # Half-life of mean reversion (Ornstein-Uhlenbeck)
            spread_lag = spread.shift(1).dropna()
            spread_diff = spread.diff().dropna()
            
            # Align for regression
            spread_lag = spread_lag[spread_diff.index]
            
            # Regression: Δspread = α + β*spread_lag
            from scipy.stats import linregress
            slope, intercept, _, _, _ = linregress(spread_lag, spread_diff)
            
            half_life = -np.log(2) / slope if slope < 0 else np.inf
            
            return PairStats(
                correlation=correlation,
                cointegration_pvalue=cointegration_pvalue,
                spread_mean=spread_mean,
                spread_std=spread_std,
                z_score=z_score,
                half_life=half_life
            )
            
        except Exception as e:
            logger.error(f"Error calculating pair stats: {e}")
            return None
            
    def _is_cointegrated(self, pair_stats: PairStats) -> bool:
        """
        Check if pair is cointegrated.
        
        Args:
            pair_stats: Pair statistics
            
        Returns:
            True if cointegrated
        """
        # Check correlation
        if abs(pair_stats.correlation) < self.params['min_correlation']:
            return False
            
        # Check cointegration p-value
        if pair_stats.cointegration_pvalue > self.params['max_pvalue']:
            return False
            
        # Check half-life (should be finite and reasonable)
        if pair_stats.half_life <= 0 or pair_stats.half_life > 100:
            return False
            
        return True
        
    def _generate_pair_signal(
        self,
        symbol_a: str,
        symbol_b: str,
        pair_stats: PairStats
    ) -> Signal:
        """
        Generate trading signal for pair.
        
        Args:
            symbol_a: First symbol
            symbol_b: Second symbol
            pair_stats: Pair statistics
            
        Returns:
            Trading signal
        """
        z = pair_stats.z_score
        entry_threshold = self.params['entry_z_score']
        exit_threshold = self.params['exit_z_score']
        stop_threshold = self.params['stop_loss_z_score']
        
        # Entry signals
        if z > entry_threshold:
            # Spread too high: short A, long B
            self.trades_opened += 1
            return Signal(
                action='pairs_trade',
                symbol=f"{symbol_a}/{symbol_b}",
                confidence=min(abs(z) / entry_threshold, 1.0),
                reason=f"Mean reversion: Z={z:.2f} (short {symbol_a}, long {symbol_b})",
                metadata={
                    'z_score': z,
                    'action_a': 'sell',
                    'action_b': 'buy',
                    'spread_mean': pair_stats.spread_mean,
                    'spread_std': pair_stats.spread_std,
                    'half_life': pair_stats.half_life,
                    'correlation': pair_stats.correlation,
                }
            )
            
        elif z < -entry_threshold:
            # Spread too low: long A, short B
            self.trades_opened += 1
            return Signal(
                action='pairs_trade',
                symbol=f"{symbol_a}/{symbol_b}",
                confidence=min(abs(z) / entry_threshold, 1.0),
                reason=f"Mean reversion: Z={z:.2f} (long {symbol_a}, short {symbol_b})",
                metadata={
                    'z_score': z,
                    'action_a': 'buy',
                    'action_b': 'sell',
                    'spread_mean': pair_stats.spread_mean,
                    'spread_std': pair_stats.spread_std,
                    'half_life': pair_stats.half_life,
                    'correlation': pair_stats.correlation,
                }
            )
            
        # Exit signals (positions mean reverting)
        elif abs(z) < exit_threshold:
            return Signal(
                action='exit_pairs_trade',
                symbol=f"{symbol_a}/{symbol_b}",
                confidence=0.8,
                reason=f"Mean reversion complete: Z={z:.2f}",
                metadata={'z_score': z}
            )
            
        # Stop loss
        elif abs(z) > stop_threshold:
            return Signal(
                action='stop_loss',
                symbol=f"{symbol_a}/{symbol_b}",
                confidence=1.0,
                reason=f"Stop loss triggered: Z={z:.2f}",
                metadata={'z_score': z}
            )
            
        # Hold
        return Signal.hold(
            f"{symbol_a}/{symbol_b}",
            f"Z-score {z:.2f} within normal range"
        )
        
    async def analyze(self, symbol: str, data: pd.DataFrame) -> Signal:
        """
        Standard analyze interface (for single symbol).
        
        For stat arb, use analyze_pair() instead.
        """
        return Signal.hold(symbol, "Use analyze_pair() for statistical arbitrage")
        
    def get_performance_stats(self) -> Dict:
        """Get strategy performance stats."""
        base_stats = super().get_performance_stats()
        
        cointegration_rate = (
            self.cointegrated_pairs / self.pairs_analyzed * 100
            if self.pairs_analyzed > 0
            else 0
        )
        
        base_stats.update({
            'pairs_analyzed': self.pairs_analyzed,
            'cointegrated_pairs': self.cointegrated_pairs,
            'cointegration_rate': cointegration_rate,
            'trades_opened': self.trades_opened,
        })
        
        return base_stats
