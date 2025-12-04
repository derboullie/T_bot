"""Risk management system."""

from typing import Dict, Optional
from datetime import datetime, timedelta

from loguru import logger
from sqlalchemy.orm import Session

from backend.core.config import settings
from backend.data.models import Order, Position, OrderStatus
from backend.trading.alpaca_client import alpaca_client


class RiskManager:
    """Manage trading risk and enforce limits."""

    def __init__(self):
        """Initialize risk manager."""
        self.max_position_size = settings.max_position_size
        self.max_daily_loss = settings.max_daily_loss
        self.max_positions = settings.max_positions
        self.risk_per_trade = settings.risk_per_trade

        self._daily_pnl = 0.0
        self._last_reset = datetime.utcnow().date()
        self._violations = []

    def check_can_trade(self, db: Session) -> tuple[bool, Optional[str]]:
        """
        Check if trading is allowed.

        Args:
            db: Database session

        Returns:
            tuple: (can_trade, reason_if_not)
        """
        # Reset daily PnL if new day
        self._reset_if_new_day()

        # Check daily loss limit
        if self._daily_pnl <= -self.max_daily_loss:
            reason = f"Daily loss limit reached: ${-self._daily_pnl:.2f}"
            logger.warning(reason)
            self._violations.append({"timestamp": datetime.utcnow(), "reason": reason})
            return False, reason

        # Check max positions
        open_positions = (
            db.query(Position).filter(Position.is_open == True).count()
        )
        if open_positions >= self.max_positions:
            reason = f"Max positions limit reached: {open_positions}/{self.max_positions}"
            logger.warning(reason)
            return False, reason

        return True, None

    def check_order_size(
        self,
        symbol: str,
        quantity: float,
        price: float,
        account_equity: float,
    ) -> tuple[bool, Optional[str]]:
        """
        Validate order size against risk limits.

        Args:
            symbol: Stock symbol
            quantity: Order quantity
            price: Order price
            account_equity: Account equity

        Returns:
            tuple: (is_valid, reason_if_not)
        """
        order_value = quantity * price

        # Check position size limit
        if order_value > self.max_position_size:
            reason = (
                f"Order size ${order_value:.2f} exceeds max position size "
                f"${self.max_position_size:.2f}"
            )
            logger.warning(f"{symbol}: {reason}")
            return False, reason

        # Check risk per trade
        max_trade_risk = account_equity * (self.risk_per_trade / 100)
        if order_value > max_trade_risk:
            reason = (
                f"Order value ${order_value:.2f} exceeds max risk per trade "
                f"${max_trade_risk:.2f} ({self.risk_per_trade}% of equity)"
            )
            logger.warning(f"{symbol}: {reason}")
            return False, reason

        return True, None

    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        account_equity: float,
        risk_percent: Optional[float] = None,
    ) -> float:
        """
        Calculate optimal position size.

        Args:
            symbol: Stock symbol
            price: Stock price
            account_equity: Account equity
            risk_percent: Risk percentage (default from settings)

        Returns:
            Quantity to trade
        """
        risk = risk_percent or self.risk_per_trade
        max_risk_amount = account_equity * (risk / 100)

        # Calculate quantity
        quantity = min(
            max_risk_amount / price,
            self.max_position_size / price,
        )

        return round(quantity, 2)

    def update_daily_pnl(self, pnl: float):
        """
        Update daily P&L.

        Args:
            pnl: Profit/loss to add
        """
        self._reset_if_new_day()
        self._daily_pnl += pnl
        logger.debug(f"Daily P&L updated: ${self._daily_pnl:.2f}")

    def _reset_if_new_day(self):
        """Reset daily counters if it's a new day."""
        today = datetime.utcnow().date()
        if today > self._last_reset:
            logger.info(f"Resetting daily P&L (previous: ${self._daily_pnl:.2f})")
            self._daily_pnl = 0.0
            self._last_reset = today

    def get_stats(self) -> Dict:
        """
        Get risk management statistics.

        Returns:
            Dict with risk stats
        """
        return {
            "daily_pnl": self._daily_pnl,
            "max_daily_loss": self.max_daily_loss,
            "max_position_size": self.max_position_size,
            "max_positions": self.max_positions,
            "risk_per_trade": self.risk_per_trade,
            "violations_today": len(self._violations),
            "last_reset": self._last_reset.isoformat(),
        }

    def get_violations(self) -> list:
        """Get risk violations."""
        return self._violations


# Global instance
risk_manager = RiskManager()
