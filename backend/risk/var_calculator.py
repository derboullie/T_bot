"""Value at Risk (VaR) Calculator.

Implements multiple VaR calculation methodologies:
- Historical VaR
- Parametric VaR (Variance-Covariance)
- Conditional VaR (CVaR / Expected Shortfall)
- Monte Carlo VaR
"""

from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger


class VaRCalculator:
    """
    Calculate Value at Risk using multiple methods.
    
    VaR represents the maximum expected loss over a given time period
    at a specified confidence level.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize VaR calculator.
        
        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95%)
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
        logger.info(f"VaR Calculator initialized (confidence: {confidence_level*100}%)")
        
    def historical_var(
        self,
        returns: np.ndarray,
        portfolio_value: float
    ) -> Dict[str, float]:
        """
        Calculate Historical VaR.
        
        Uses actual historical returns distribution.
        
        Args:
            returns: Array of historical returns
            portfolio_value: Current portfolio value
            
        Returns:
            Dict with VaR and CVaR values
        """
        if len(returns) == 0:
            return {'var': 0.0, 'cvar': 0.0}
            
        # Sort returns
        sorted_returns = np.sort(returns)
        
        # Find VaR at confidence level
        var_index = int(len(sorted_returns) * self.alpha)
        var_return = sorted_returns[var_index]
        var_value = abs(var_return * portfolio_value)
        
        # Calculate CVaR (average of losses beyond VaR)
        tail_returns = sorted_returns[:var_index]
        cvar_return = np.mean(tail_returns) if len(tail_returns) > 0 else var_return
        cvar_value = abs(cvar_return * portfolio_value)
        
        return {
            'var': var_value,
            'var_pct': var_return * 100,
            'cvar': cvar_value,
            'cvar_pct': cvar_return * 100,
            'method': 'historical'
        }
        
    def parametric_var(
        self,
        returns: np.ndarray,
        portfolio_value: float
    ) -> Dict[str, float]:
        """
        Calculate Parametric VaR (Variance-Covariance method).
        
        Assumes returns follow normal distribution.
        
        Args:
            returns: Array of historical returns
            portfolio_value: Current portfolio value
            
        Returns:
            Dict with VaR and CVaR values
        """
        if len(returns) == 0:
            return {'var': 0.0, 'cvar': 0.0}
            
        # Calculate mean and std of returns
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Get z-score for confidence level
        z_score = stats.norm.ppf(self.alpha)
        
        # Calculate VaR
        var_return = mean_return + (z_score * std_return)
        var_value = abs(var_return * portfolio_value)
        
        # Calculate CVaR for normal distribution
        # CVaR = μ - σ * φ(z) / α
        # where φ is the standard normal PDF
        phi_z = stats.norm.pdf(z_score)
        cvar_return = mean_return - (std_return * phi_z / self.alpha)
        cvar_value = abs(cvar_return * portfolio_value)
        
        return {
            'var': var_value,
            'var_pct': var_return * 100,
            'cvar': cvar_value,
            'cvar_pct': cvar_return * 100,
            'method': 'parametric',
            'mean': mean_return * 100,
            'std': std_return * 100
        }
        
    def monte_carlo_var(
        self,
        returns: np.ndarray,
        portfolio_value: float,
        num_simulations: int = 10000,
        time_horizon: int = 1
    ) -> Dict[str, float]:
        """
        Calculate Monte Carlo VaR.
        
        Simulates future returns based on historical distribution.
        
        Args:
            returns: Array of historical returns
            portfolio_value: Current portfolio value
            num_simulations: Number of Monte Carlo simulations
            time_horizon: Time horizon in days
            
        Returns:
            Dict with VaR and CVaR values
        """
        if len(returns) == 0:
            return {'var': 0.0, 'cvar': 0.0}
            
        # Calculate parameters from historical data
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Simulate future returns
        simulated_returns = np.random.normal(
            mean_return * time_horizon,
            std_return * np.sqrt(time_horizon),
            num_simulations
        )
        
        # Sort simulated returns
        sorted_returns = np.sort(simulated_returns)
        
        # Calculate VaR
        var_index = int(num_simulations * self.alpha)
        var_return = sorted_returns[var_index]
        var_value = abs(var_return * portfolio_value)
        
        # Calculate CVaR
        tail_returns = sorted_returns[:var_index]
        cvar_return = np.mean(tail_returns)
        cvar_value = abs(cvar_return * portfolio_value)
        
        return {
            'var': var_value,
            'var_pct': var_return * 100,
            'cvar': cvar_value,
            'cvar_pct': cvar_return * 100,
            'method': 'monte_carlo',
            'simulations': num_simulations,
            'time_horizon': time_horizon
        }
        
    def calculate_all(
        self,
        returns: np.ndarray,
        portfolio_value: float
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate VaR using all methods.
        
        Args:
            returns: Array of historical returns
            portfolio_value: Current portfolio value
            
        Returns:
            Dict with results from all methods
        """
        return {
            'historical': self.historical_var(returns, portfolio_value),
            'parametric': self.parametric_var(returns, portfolio_value),
            'monte_carlo': self.monte_carlo_var(returns, portfolio_value),
        }
        
    def get_risk_metrics(
        self,
        returns: np.ndarray,
        portfolio_value: float
    ) -> Dict[str, float]:
        """
        Get comprehensive risk metrics.
        
        Args:
            returns: Array of historical returns
            portfolio_value: Current portfolio value
            
        Returns:
            Dict with risk metrics
        """
        if len(returns) == 0:
            return {}
            
        # Use parametric VaR as default
        var_results = self.parametric_var(returns, portfolio_value)
        
        # Additional metrics
        metrics = {
            'var_1day': var_results['var'],
            'cvar_1day': var_results['cvar'],
            'var_pct': var_results['var_pct'],
            'cvar_pct': var_results['cvar_pct'],
            
            # Volatility
            'volatility_daily': np.std(returns) * 100,
            'volatility_annual': np.std(returns) * np.sqrt(252) * 100,
            
            # Downside risk
            'downside_deviation': self._downside_deviation(returns) * 100,
            
            # Maximum drawdown
            'max_drawdown': self._max_drawdown(returns) * 100,
        }
        
        return metrics
        
    def _downside_deviation(self, returns: np.ndarray) -> float:
        """Calculate downside deviation (semi-deviation)."""
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            return 0.0
        return np.std(negative_returns)
        
    def _max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown from returns."""
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())


class RealTimeVaR:
    """Real-time VaR monitoring."""
    
    def __init__(
        self,
        calculator: Optional[VaRCalculator] = None,
        lookback_period: int = 252
    ):
        """
        Initialize real-time VaR monitor.
        
        Args:
            calculator: VaR calculator instance
            lookback_period: Number of periods to use for calculations
        """
        self.calculator = calculator or VaRCalculator()
        self.lookback_period = lookback_period
        self.returns_history: List[float] = []
        
    def update(self, portfolio_value: float, previous_value: float) -> Dict[str, float]:
        """
        Update VaR with new portfolio value.
        
        Args:
            portfolio_value: Current portfolio value
            previous_value: Previous portfolio value
            
        Returns:
            Current VaR metrics
        """
        # Calculate return
        if previous_value > 0:
            ret = (portfolio_value - previous_value) / previous_value
            self.returns_history.append(ret)
            
        # Keep only recent history
        if len(self.returns_history) > self.lookback_period:
            self.returns_history = self.returns_history[-self.lookback_period:]
            
        # Calculate VaR if we have enough data
        if len(self.returns_history) >= 30:  # Minimum 30 observations
            returns_array = np.array(self.returns_history)
            return self.calculator.get_risk_metrics(returns_array, portfolio_value)
        else:
            return {'status': 'insufficient_data', 'observations': len(self.returns_history)}
            
    def get_breach_alert(self, current_loss: float, var_value: float) -> Optional[Dict]:
        """
        Check if current loss breaches VaR.
        
        Args:
            current_loss: Current unrealized loss (positive value)
            var_value: VaR threshold
            
        Returns:
            Alert dict if breach detected, None otherwise
        """
        if current_loss > var_value:
            return {
                'type': 'var_breach',
                'current_loss': current_loss,
                'var_limit': var_value,
                'breach_amount': current_loss - var_value,
                'breach_pct': ((current_loss - var_value) / var_value) * 100,
                'severity': 'high' if current_loss > var_value * 1.5 else 'medium'
            }
        return None


# Global instance
var_calculator = VaRCalculator(confidence_level=0.95)
realtime_var = RealTimeVaR()
