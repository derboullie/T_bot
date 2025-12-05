"""Dynamic Portfolio Optimization Module.

Implements advanced portfolio optimization strategies:
- Kelly Criterion for optimal position sizing
- Mean-Variance Optimization (Markowitz)
- Risk Parity
- Dynamic rebalancing
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from loguru import logger


class KellyCriterion:
    """
    Kelly Criterion for optimal position sizing.
    
    Kelly formula: f* = (bp - q) / b
    where:
    - f* = fraction of capital to wager
    - b = odds received (payoff ratio)
    - p = probability of winning
    - q = probability of losing (1-p)
    """
    
    def __init__(self, fractional_kelly: float = 0.5):
        """
        Initialize Kelly Criterion calculator.
        
        Args:
            fractional_kelly: Fraction of full Kelly to use (0.5 = half Kelly)
        """
        self.fractional_kelly = fractional_kelly
        logger.info(f"Kelly Criterion initialized (fraction: {fractional_kelly})")
        
    def calculate_from_stats(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate Kelly fraction from trading statistics.
        
        Args:
            win_rate: Probability of winning (0-1)
            avg_win: Average win amount (positive)
            avg_loss: Average loss amount (positive)
            
        Returns:
            Optimal fraction of capital to risk
        """
        if avg_loss == 0:
            return 0.0
            
        # Win/Loss ratio
        win_loss_ratio = avg_win / avg_loss
        
        # Kelly formula
        kelly_fraction = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        
        # Apply fractional Kelly
        kelly_fraction *= self.fractional_kelly
        
        # Clamp to reasonable range
        kelly_fraction = max(0.0, min(kelly_fraction, 0.25))  # Max 25% per trade
        
        return kelly_fraction
        
    def calculate_from_returns(self, returns: np.ndarray) -> float:
        """
        Calculate Kelly fraction from return series.
        
        Args:
            returns: Array of historical returns
            
        Returns:
            Optimal fraction of capital to risk
        """
        if len(returns) == 0:
            return 0.0
            
        mean_return = np.mean(returns)
        variance = np.var(returns)
        
        if variance == 0:
            return 0.0
            
        # Kelly formula for continuous returns
        kelly_fraction = mean_return / variance
        
        # Apply fractional Kelly
        kelly_fraction *= self.fractional_kelly
        
        # Clamp
        kelly_fraction = max(0.0, min(kelly_fraction, 0.25))
        
        return kelly_fraction


class MeanVarianceOptimizer:
    """
    Mean-Variance Portfolio Optimization (Markowitz).
    
    Finds optimal portfolio weights to maximize return for given risk
    or minimize risk for given return.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize optimizer.
        
        Args:
            risk_free_rate: Annual risk-free rate
        """
        self.risk_free_rate = risk_free_rate
        
    def optimize(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        target_return: Optional[float] = None,
        max_weight: float = 0.3
    ) -> Dict[str, any]:
        """
        Find optimal portfolio weights.
        
        Args:
            expected_returns: Expected returns for each asset
            cov_matrix: Covariance matrix of returns
            target_return: Target return (optional)
            max_weight: Maximum weight per asset
            
        Returns:
            Dict with optimal weights and metrics
        """
        n_assets = len(expected_returns)
        
        # Objective function: minimize portfolio variance
        def portfolio_variance(weights):
            return weights @ cov_matrix @ weights
            
        # Constraints
        constraints = [
            # Weights sum to 1
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        # Add target return constraint if specified
        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda w: w @ expected_returns - target_return
            })
            
        # Bounds: 0 <= weight <= max_weight
        bounds = tuple((0, max_weight) for _ in range(n_assets))
        
        # Initial guess: equal weights
        initial_weights = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            portfolio_variance,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if not result.success:
            logger.warning("Optimization failed, using equal weights")
            weights = initial_weights
        else:
            weights = result.x
            
        # Calculate portfolio metrics
        portfolio_return = weights @ expected_returns
        portfolio_std = np.sqrt(weights @ cov_matrix @ weights)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std if portfolio_std > 0 else 0
        
        return {
            'weights': weights,
            'expected_return': portfolio_return,
            'volatility': portfolio_std,
            'sharpe_ratio': sharpe_ratio,
            'success': result.success
        }
        
    def efficient_frontier(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        n_points: int = 100
    ) -> pd.DataFrame:
        """
        Calculate efficient frontier.
        
        Args:
            expected_returns: Expected returns for each asset
            cov_matrix: Covariance matrix
            n_points: Number of points on frontier
            
        Returns:
            DataFrame with frontier points
        """
        min_return = np.min(expected_returns)
        max_return = np.max(expected_returns)
        target_returns = np.linspace(min_return, max_return, n_points)
        
        frontier_points = []
        
        for target_ret in target_returns:
            result = self.optimize(expected_returns, cov_matrix, target_ret)
            if result['success']:
                frontier_points.append({
                    'return': result['expected_return'],
                    'volatility': result['volatility'],
                    'sharpe': result['sharpe_ratio']
                })
                
        return pd.DataFrame(frontier_points)
        
    def max_sharpe_portfolio(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray
    ) -> Dict:
        """
        Find portfolio with maximum Sharpe ratio.
        
        Args:
            expected_returns: Expected returns for each asset
            cov_matrix: Covariance matrix
            
        Returns:
            Optimal portfolio dict
        """
        n_assets = len(expected_returns)
        
        # Objective: minimize negative Sharpe ratio
        def neg_sharpe(weights):
            portfolio_return = weights @ expected_returns
            portfolio_std = np.sqrt(weights @ cov_matrix @ weights)
            if portfolio_std == 0:
                return 0
            return -(portfolio_return - self.risk_free_rate) / portfolio_std
            
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_weights = np.array([1/n_assets] * n_assets)
        
        result = minimize(
            neg_sharpe,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        weights = result.x if result.success else initial_weights
        
        portfolio_return = weights @ expected_returns
        portfolio_std = np.sqrt(weights @ cov_matrix @ weights)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std if portfolio_std > 0 else 0
        
        return {
            'weights': weights,
            'expected_return': portfolio_return,
            'volatility': portfolio_std,
            'sharpe_ratio': sharpe_ratio,
            'success': result.success
        }


class RiskParityOptimizer:
    """
    Risk Parity Portfolio Optimization.
    
    Allocates capital such that each asset contributes equally to portfolio risk.
    """
    
    def __init__(self):
        """Initialize risk parity optimizer."""
        logger.info("Risk Parity Optimizer initialized")
        
    def optimize(self, cov_matrix: np.ndarray) -> np.ndarray:
        """
        Find risk parity weights.
        
        Args:
            cov_matrix: Covariance matrix
            
        Returns:
            Optimal weights
        """
        n_assets = len(cov_matrix)
        
        # Objective: minimize difference in risk contributions
        def risk_budget_objective(weights):
            portfolio_var = weights @ cov_matrix @ weights
            marginal_contrib = cov_matrix @ weights
            risk_contrib = weights * marginal_contrib
            
            # Target: equal risk contribution
            target_risk = portfolio_var / n_assets
            return np.sum((risk_contrib - target_risk) ** 2)
            
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_weights = np.array([1/n_assets] * n_assets)
        
        result = minimize(
            risk_budget_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x if result.success else initial_weights


class DynamicRebalancer:
    """
    Dynamic portfolio rebalancing based on thresholds and time.
    """
    
    def __init__(
        self,
        rebalance_threshold: float = 0.05,
        min_rebalance_days: int = 7
    ):
        """
        Initialize rebalancer.
        
        Args:
            rebalance_threshold: Trigger rebalance if drift > threshold
            min_rebalance_days: Minimum days between rebalances
        """
        self.rebalance_threshold = rebalance_threshold
        self.min_rebalance_days = min_rebalance_days
        self.last_rebalance = None
        
    def check_rebalance_needed(
        self,
        target_weights: np.ndarray,
        current_weights: np.ndarray,
        current_time: pd.Timestamp
    ) -> bool:
        """
        Check if rebalancing is needed.
        
        Args:
            target_weights: Target portfolio weights
            current_weights: Current portfolio weights
            current_time: Current timestamp
            
        Returns:
            True if rebalance needed
        """
        # Check time constraint
        if self.last_rebalance is not None:
            days_since = (current_time - self.last_rebalance).days
            if days_since < self.min_rebalance_days:
                return False
                
        # Check drift threshold
        max_drift = np.max(np.abs(current_weights - target_weights))
        
        if max_drift > self.rebalance_threshold:
            return True
            
        return False
        
    def calculate_rebalance_trades(
        self,
        target_weights: np.ndarray,
        current_weights: np.ndarray,
        portfolio_value: float
    ) -> Dict[int, float]:
        """
        Calculate trades needed to rebalance.
        
        Args:
            target_weights: Target weights
            current_weights: Current weights
            portfolio_value: Total portfolio value
            
        Returns:
            Dict mapping asset index to trade amount (positive = buy, negative = sell)
        """
        weight_diff = target_weights - current_weights
        trade_amounts = weight_diff * portfolio_value
        
        # Create trade dict (only non-zero trades)
        trades = {
            i: amount 
            for i, amount in enumerate(trade_amounts)
            if abs(amount) > 1.0  # Minimum $1 trade
        }
        
        return trades


# Global instances
kelly_criterion = KellyCriterion(fractional_kelly=0.5)
mv_optimizer = MeanVarianceOptimizer()
rp_optimizer = RiskParityOptimizer()
dynamic_rebalancer = DynamicRebalancer()
