"""Simple test to verify risk manager functionality."""

import pytest
from backend.trading.risk_manager import RiskManager


def test_risk_manager_initialization():
    """Test risk manager initializes correctly."""
    rm = RiskManager()
    assert rm.max_position_size > 0
    assert rm.max_daily_loss > 0
    assert rm.max_positions > 0


def test_calculate_position_size():
    """Test position size calculation."""
    rm = RiskManager()
    
    # Test with $100,000 equity and $50 stock price
    size = rm.calculate_position_size("AAPL", 50.0, 100000.0, risk_percent=1.0)
    
    assert size > 0
    assert size * 50.0 <= rm.max_position_size  # Should not exceed max position size


def test_daily_pnl_update():
    """Test daily P&L tracking."""
    rm = RiskManager()
    rm.update_daily_pnl(100.0)
    stats = rm.get_stats()
    assert stats["daily_pnl"] == 100.0
