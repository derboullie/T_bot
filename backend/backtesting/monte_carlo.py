"""Monte Carlo Simulation Module for Portfolio Stress Testing.

Simulates thousands of market scenarios to test strategy robustness
against extreme events (Black Swan scenarios).
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from loguru import logger


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulations."""
    n_simulations: int = 10000
    time_horizon_days: int = 252  # 1 year
    confidence_levels: List[float] = None
    include_black_swan: bool = True
    black_swan_probability: float = 0.01  # 1% chance
    black_swan_magnitude: float = -0.20  # -20% shock
    
    def __post_init__(self):
        if self.confidence_levels is None:
            self.confidence_levels = [0.90, 0.95, 0.99]


class MonteCarloSimulator:
    """
    Monte Carlo simulator for portfolio analysis.
    
    Features:
    - Geometric Brownian Motion (GBM) simulation
    - Black Swan event simulation
    - Portfolio value projections
    - Risk metrics calculation
    - Stress testing scenarios
    """
    
    def __init__(self, config: Optional[MonteCarloConfig] = None):
        """
        Initialize Monte Carlo simulator.
        
        Args:
            config: Simulation configuration
        """
        self.config = config or MonteCarloConfig()
        logger.info(
            f"Monte Carlo Simulator initialized "
            f"({self.config.n_simulations:,} simulations)"
        )
        
    def simulate_returns(
        self,
        mean_return: float,
        volatility: float,
        time_horizon: Optional[int] = None,
        n_simulations: Optional[int] = None
    ) -> np.ndarray:
        """
        Simulate returns using Geometric Brownian Motion.
        
        Args:
            mean_return: Expected annual return
            volatility: Annual volatility (std dev)
            time_horizon: Days to simulate
            n_simulations: Number of simulations
            
        Returns:
            Array of shape (n_simulations, time_horizon) with returns
        """
        time_horizon = time_horizon or self.config.time_horizon_days
        n_simulations = n_simulations or self.config.n_simulations
        
        # Convert to daily parameters
        daily_return = mean_return / 252
        daily_vol = volatility / np.sqrt(252)
        
        # Generate random returns
        random_returns = np.random.normal(
            daily_return,
            daily_vol,
            (n_simulations, time_horizon)
        )
        
        # Add Black Swan events if enabled
        if self.config.include_black_swan:
            black_swan_mask = np.random.random(
                (n_simulations, time_horizon)
            ) < (self.config.black_swan_probability / 252)
            
            # Apply shock
            random_returns[black_swan_mask] = self.config.black_swan_magnitude
            
        return random_returns
        
    def simulate_portfolio_value(
        self,
        initial_value: float,
        mean_return: float,
        volatility: float,
        time_horizon: Optional[int] = None
    ) -> np.ndarray:
        """
        Simulate portfolio value paths.
        
        Args:
            initial_value: Starting portfolio value
            mean_return: Expected annual return
            volatility: Annual volatility
            time_horizon: Days to simulate
            
        Returns:
            Array of shape (n_simulations, time_horizon + 1) with portfolio values
        """
        returns = self.simulate_returns(mean_return, volatility, time_horizon)
        
        # Calculate cumulative values
        cumulative_returns = (1 + returns).cumprod(axis=1)
        
        # Add initial value as first column
        portfolio_values = np.column_stack([
            np.ones(self.config.n_simulations) * initial_value,
            initial_value * cumulative_returns
        ])
        
        return portfolio_values
        
    def calculate_risk_metrics(
        self,
        portfolio_values: np.ndarray
    ) -> Dict:
        """
        Calculate risk metrics from simulated paths.
        
        Args:
            portfolio_values: Simulated portfolio values
            
        Returns:
            Dict with risk metrics
        """
        final_values = portfolio_values[:, -1]
        
        metrics = {}
        
        # Expected value and percentiles
        metrics['expected_final_value'] = np.mean(final_values)
        metrics['median_final_value'] = np.median(final_values)
        
        # Confidence intervals
        for conf_level in self.config.confidence_levels:
            lower_pct = ((1 - conf_level) / 2) * 100
            upper_pct = (conf_level + (1 - conf_level) / 2) * 100
            
            metrics[f'ci_{int(conf_level*100)}_lower'] = np.percentile(
                final_values, lower_pct
            )
            metrics[f'ci_{int(conf_level*100)}_upper'] = np.percentile(
                final_values, upper_pct
            )
            
        # Value at Risk (VaR)
        for conf_level in self.config.confidence_levels:
            var_pct = (1 - conf_level) * 100
            metrics[f'var_{int(conf_level*100)}'] = np.percentile(
                final_values, var_pct
            )
            
        # Maximum drawdown across all paths
        max_drawdowns = []
        for path in portfolio_values:
            running_max = np.maximum.accumulate(path)
            drawdown = (path - running_max) / running_max
            max_drawdowns.append(drawdown.min())
            
        metrics['expected_max_drawdown'] = np.mean(max_drawdowns)
        metrics['worst_max_drawdown'] = np.min(max_drawdowns)
        
        # Probability of loss
        initial_value = portfolio_values[0, 0]
        metrics['prob_loss'] = np.sum(final_values < initial_value) / len(final_values)
        
        # Probability of severe loss (>20%)
        severe_loss_threshold = initial_value * 0.8
        metrics['prob_severe_loss'] = np.sum(
            final_values < severe_loss_threshold
        ) / len(final_values)
        
        return metrics
        
    def stress_test(
        self,
        initial_value: float,
        base_return: float,
        base_volatility: float,
        scenarios: Optional[Dict[str, Dict]] = None
    ) -> Dict[str, Dict]:
        """
        Run stress tests under different scenarios.
        
        Args:
            initial_value: Starting portfolio value
            base_return: Base case expected return
            base_volatility: Base case volatility
            scenarios: Dict of scenario definitions
            
        Returns:
            Dict with results for each scenario
        """
        if scenarios is None:
            scenarios = {
                'base_case': {
                    'return': base_return,
                    'volatility': base_volatility
                },
                'market_crash': {
                    'return': base_return - 0.15,
                    'volatility': base_volatility * 2.0
                },
                'high_volatility': {
                    'return': base_return,
                    'volatility': base_volatility * 1.5
                },
                'recession': {
                    'return': -0.05,
                    'volatility': base_volatility * 1.3
                },
                'bull_market': {
                    'return': base_return + 0.10,
                    'volatility': base_volatility * 0.8
                }
            }
            
        results = {}
        
        for scenario_name, params in scenarios.items():
            logger.info(f"Running stress test: {scenario_name}")
            
            portfolio_values = self.simulate_portfolio_value(
                initial_value,
                params['return'],
                params['volatility']
            )
            
            metrics = self.calculate_risk_metrics(portfolio_values)
            metrics['scenario_params'] = params
            
            results[scenario_name] = metrics
            
        return results
        
    def generate_report(
        self,
        stress_test_results: Dict[str, Dict],
        initial_value: float
    ) -> pd.DataFrame:
        """
        Generate summary report from stress test results.
        
        Args:
            stress_test_results: Results from stress_test()
            initial_value: Initial portfolio value
            
        Returns:
            DataFrame with summary statistics
        """
        rows = []
        
        for scenario, metrics in stress_test_results.items():
            params = metrics['scenario_params']
            
            row = {
                'Scenario': scenario,
                'Return': f"{params['return']*100:.1f}%",
                'Volatility': f"{params['volatility']*100:.1f}%",
                'Expected Value': f"${metrics['expected_final_value']:,.0f}",
                'Median Value': f"${metrics['median_final_value']:,.0f}",
                'VaR 95%': f"${metrics['var_95']:,.0f}",
                'Max Drawdown': f"{metrics['expected_max_drawdown']*100:.1f}%",
                'Prob Loss': f"{metrics['prob_loss']*100:.1f}%",
            }
            
            rows.append(row)
            
        return pd.DataFrame(rows)


class ScenarioGenerator:
    """Generate specific market scenarios for testing."""
    
    @staticmethod
    def black_swan_scenario(
        base_return: float,
        base_volatility: float
    ) -> Dict:
        """Generate Black Swan scenario parameters."""
        return {
            'return': -0.30,  # -30% annual return
            'volatility': base_volatility * 3.0,
            'description': '2008-style market crash'
        }
        
    @staticmethod
    def flash_crash_scenario() -> Dict:
        """Generate flash crash scenario."""
        return {
            'return': -0.15,
            'volatility': 0.40,  # 40% volatility
            'description': 'Sudden market drop with extreme volatility'
        }
        
    @staticmethod
    def perfect_storm_scenario() -> Dict:
        """Combined worst-case scenario."""
        return {
            'return': -0.40,
            'volatility': 0.60,
            'description': 'Multiple crises combined'
        }


# Global instance
monte_carlo_simulator = MonteCarloSimulator()
scenario_generator = ScenarioGenerator()
