"""Multi-Asset Backtesting Engine.

Supports simultaneous backtesting across multiple asset classes:
- Stocks
- Cryptocurrencies
- Commodities
- Forex
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
from loguru import logger

from backend.backtesting.backtest_engine import BacktestEngine, BacktestConfig


@dataclass
class AssetAllocation:
    """Asset allocation configuration."""
    symbol: str
    asset_class: str  # 'stock', 'crypto', 'commodity', 'forex'
    allocation_pct: float  # Percentage of portfolio


class MultiAssetBacktester:
    """
    Backtest engine for multiple assets simultaneously.
    
    Features:
    - Cross-asset portfolio management
    - Correlation analysis
    - Sector rotation
    - Dynamic rebalancing
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        rebalance_frequency: str = 'monthly'  # 'daily', 'weekly', 'monthly', 'quarterly'
    ):
        """
        Initialize multi-asset backtester.
        
        Args:
            initial_capital: Starting capital
            rebalance_frequency: How often to rebalance
        """
        self.initial_capital = initial_capital
        self.rebalance_frequency = rebalance_frequency
        
        # Individual backtester for each asset
        self.asset_backtests: Dict[str, BacktestEngine] = {}
        
        logger.info(f"Multi-Asset Backtester initialized (capital=${initial_capital:,.0f})")
        
    def add_asset(
        self,
        symbol: str,
        asset_class: str,
        strategy: any,
        allocation_pct: float
    ):
        """
        Add asset to portfolio.
        
        Args:
            symbol: Asset symbol
            asset_class: Asset class type
            strategy: Trading strategy for this asset
            allocation_pct: % of portfolio allocated
        """
        config = BacktestConfig(
            initial_capital=self.initial_capital * allocation_pct,
            commission_pct=0.001,
            slippage_pct=0.0005
        )
        
        backtester = BacktestEngine(config)
        backtester.set_strategy(strategy)
        
        self.asset_backtests[symbol] = {
            'backtester': backtester,
            'asset_class': asset_class,
            'allocation_pct': allocation_pct
        }
        
        logger.info(f"Asset added: {symbol} ({asset_class}) - {allocation_pct*100:.1f}% allocation")
        
    def run(
        self,
        data_dict: Dict[str, pd.DataFrame],
        risk_free_rate: float = 0.02
    ) -> Dict:
        """
        Run backtest across all assets.
        
        Args:
            data_dict: Dict mapping symbols to their data
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Portfolio performance results
        """
        results = {}
        
        # Run individual backtests
        for symbol, asset_info in self.asset_backtests.items():
            if symbol not in data_dict:
                logger.warning(f"No data for {symbol}, skipping")
                continue
                
            logger.info(f"Running backtest for {symbol}...")
            
            backtester = asset_info['backtester']
            asset_results = backtester.run(data_dict[symbol])
            
            results[symbol] = {
                'performance': asset_results,
                'asset_class': asset_info['asset_class'],
                'allocation': asset_info['allocation_pct']
            }
            
        # Calculate portfolio metrics
        portfolio_metrics = self._calculate_portfolio_metrics(results, risk_free_rate)
        
        return {
            'individual_assets': results,
            'portfolio': portfolio_metrics
        }
        
    def _calculate_portfolio_metrics(
        self,
        asset_results: Dict,
        risk_free_rate: float
    ) -> Dict:
        """
        Calculate portfolio-level metrics.
        
        Args:
            asset_results: Results from individual assets
            risk_free_rate: Risk-free rate
            
        Returns:
            Portfolio metrics
        """
        # Aggregate equity curves
        equity_curves = []
        allocations = []
        
        for symbol, data in asset_results.items():
            perf = data['performance']
            if 'equity_curve' in perf:
                equity_curves.append(perf['equity_curve'])
                allocations.append(data['allocation'])
                
        if not equity_curves:
            return {}
            
        # Weighted portfolio equity curve
        allocations = np.array(allocations) / sum(allocations)  # Normalize
        
        # Align all curves to same length (take minimum)
        min_length = min(len(curve) for curve in equity_curves)
        aligned_curves = [curve[:min_length] for curve in equity_curves]
        
        # Calculate weighted portfolio value
        portfolio_equity = sum(
            np.array(curve) * weight 
            for curve, weight in zip(aligned_curves, allocations)
        )
        
        # Calculate returns
        portfolio_returns = np.diff(portfolio_equity) / portfolio_equity[:-1]
        
        # Performance metrics
        total_return = (portfolio_equity[-1] - portfolio_equity[0]) / portfolio_equity[0]
        
        # Sharpe Ratio
        excess_returns = portfolio_returns - (risk_free_rate / 252)
        sharpe = np.mean(excess_returns) / (np.std(portfolio_returns) + 1e-8) * np.sqrt(252)
        
        # Sortino Ratio
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_dev = np.std(downside_returns) if len(downside_returns) > 0 else 1e-8
        sortino = np.mean(excess_returns) / downside_dev * np.sqrt(252)
        
        # Max Drawdown
        running_max = np.maximum.accumulate(portfolio_equity)
        drawdown = (portfolio_equity - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Calmar Ratio
        annual_return = total_return * (252 / len(portfolio_returns))
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'volatility': np.std(portfolio_returns) * np.sqrt(252),
            'final_value': portfolio_equity[-1],
            'equity_curve': portfolio_equity.tolist()
        }
        
    def calculate_correlation_matrix(
        self,
        data_dict: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix between assets.
        
        Args:
            data_dict: Dict mapping symbols to data
            
        Returns:
            Correlation matrix
        """
        returns_dict = {}
        
        for symbol in self.asset_backtests.keys():
            if symbol in data_dict:
                df = data_dict[symbol]
                if 'close' in df.columns:
                    returns = df['close'].pct_change().dropna()
                    returns_dict[symbol] = returns
                    
        if not returns_dict:
            return pd.DataFrame()
            
        # Align all series
        returns_df = pd.DataFrame(returns_dict).dropna()
        
        return returns_df.corr()
        
    def sector_analysis(
        self,
        results: Dict
    ) -> Dict[str, Dict]:
        """
        Analyze performance by asset class.
        
        Args:
            results: Results from run()
            
        Returns:
            Asset class performance breakdown
        """
        sector_perf = {}
        
        for symbol, data in results['individual_assets'].items():
            asset_class = data['asset_class']
            
            if asset_class not in sector_perf:
                sector_perf[asset_class] = {
                    'symbols': [],
                    'total_allocation': 0.0,
                    'avg_return': 0.0,
                    'returns': []
                }
                
            sector_perf[asset_class]['symbols'].append(symbol)
            sector_perf[asset_class]['total_allocation'] += data['allocation']
            
            if 'total_return' in data['performance']:
                sector_perf[asset_class]['returns'].append(
                    data['performance']['total_return']
                )
                
        # Calculate averages
        for asset_class in sector_perf:
            returns = sector_perf[asset_class]['returns']
            if returns:
                sector_perf[asset_class]['avg_return'] = np.mean(returns)
                sector_perf[asset_class]['avg_return_pct'] = np.mean(returns) * 100
                
        return sector_perf


class WalkForwardAnalyzer:
    """
    Walk-Forward Analysis for strategy optimization validation.
    
    Helps detect overfitting by testing optimized parameters
    on out-of-sample data.
    """
    
    def __init__(
        self,
        train_period_days: int = 252,  # 1 year
        test_period_days: int = 63,    # 3 months
        step_days: int = 21            # Roll forward by 1 month
    ):
        """
        Initialize walk-forward analyzer.
        
        Args:
            train_period_days: Days for optimization
            test_period_days: Days for testing
            step_days: Days to roll forward
        """
        self.train_period_days = train_period_days
        self.test_period_days = test_period_days
        self.step_days = step_days
        
        logger.info("Walk-Forward Analyzer initialized")
        
    def analyze(
        self,
        data: pd.DataFrame,
        strategy_class: type,
        param_grid: Dict[str, List]
    ) -> Dict:
        """
        Perform walk-forward analysis.
        
        Args:
            data: Historical data
            strategy_class: Strategy class to test
            param_grid: Parameter grid for optimization
            
        Returns:
            Walk-forward results
        """
        results = []
        
        # Calculate number of windows
        total_days = len(data)
        current_start = 0
        
        window_num = 0
        
        while current_start + self.train_period_days + self.test_period_days <= total_days:
            window_num += 1
            
            train_end = current_start + self.train_period_days
            test_end = train_end + self.test_period_days
            
            train_data = data.iloc[current_start:train_end]
            test_data = data.iloc[train_end:test_end]
            
            logger.info(
                f"Window {window_num}: Train={len(train_data)} days, "
                f"Test={len(test_data)} days"
            )
            
            # Optimize on train data
            best_params = self._optimize_on_train(
                train_data, strategy_class, param_grid
            )
            
            # Test on out-of-sample data
            test_results = self._test_on_validation(
                test_data, strategy_class, best_params
            )
            
            results.append({
                'window': window_num,
                'train_start': current_start,
                'train_end': train_end,
                'test_end': test_end,
                'best_params': best_params,
                'test_performance': test_results
            })
            
            # Roll forward
            current_start += self.step_days
            
        # Aggregate results
        test_returns = [r['test_performance'].get('total_return', 0) for r in results]
        
        summary = {
            'n_windows': len(results),
            'avg_test_return': np.mean(test_returns),
            'std_test_return': np.std(test_returns),
            'min_test_return': np.min(test_returns),
            'max_test_return': np.max(test_returns),
            'windows': results
        }
        
        return summary
        
    def _optimize_on_train(
        self,
        train_data: pd.DataFrame,
        strategy_class: type,
        param_grid: Dict[str, List]
    ) -> Dict:
        """Optimize strategy parameters on training data."""
        # Simple grid search
        best_sharpe = -np.inf
        best_params = {}
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        from itertools import product
        
        for param_combo in product(*param_values):
            params = dict(zip(param_names, param_combo))
            
            # Create strategy with these params
            strategy = strategy_class(**params)
            
            # Backtest
            config = BacktestConfig(initial_capital=10000)
            backtester = BacktestEngine(config)
            backtester.set_strategy(strategy)
            
            try:
                results = backtester.run(train_data)
                sharpe = results.get('sharpe_ratio', -np.inf)
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = params
            except Exception as e:
                logger.debug(f"Backtest failed for params {params}: {e}")
                continue
                
        return best_params
        
    def _test_on_validation(
        self,
        test_data: pd.DataFrame,
        strategy_class: type,
        params: Dict
    ) -> Dict:
        """Test optimized parameters on validation data."""
        strategy = strategy_class(**params)
        
        config = BacktestConfig(initial_capital=10000)
        backtester = BacktestEngine(config)
        backtester.set_strategy(strategy)
        
        return backtester.run(test_data)


# Global instances
multi_asset_backtester = MultiAssetBacktester()
walk_forward_analyzer = WalkForwardAnalyzer()
