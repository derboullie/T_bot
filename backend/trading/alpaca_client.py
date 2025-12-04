"""Alpaca API client for trading execution."""

from datetime import datetime
from typing import Dict, List, Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce, QueryOrderStatus
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    StopOrderRequest,
    StopLimitOrderRequest,
    GetOrdersRequest,
)
from alpaca.trading.models import Order as AlpacaOrder, Position as AlpacaPosition
from loguru import logger

from backend.core.config import settings


class AlpacaClient:
    """Client for Alpaca trading API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize Alpaca client.

        Args:
            api_key: Alpaca API key (default from settings)
            secret_key: Alpaca secret key (default from settings)
            base_url: Alpaca base URL (default from settings)
        """
        self.api_key = api_key or settings.alpaca_api_key
        self.secret_key = secret_key or settings.alpaca_secret_key
        self.base_url = base_url or settings.alpaca_base_url

        self.client = TradingClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
            paper=self.base_url.startswith("https://paper"),
        )

        self._account_info = None
        logger.info(
            f"Alpaca client initialized ({'paper' if self.is_paper() else 'live'} trading)"
        )

    def is_paper(self) -> bool:
        """Check if using paper trading."""
        return self.base_url.startswith("https://paper")

    async def get_account(self) -> Dict:
        """
        Get account information.

        Returns:
            Dict with account details
        """
        try:
            account = self.client.get_account()
            self._account_info = {
                "account_number": account.account_number,
                "status": account.status,
                "currency": account.currency,
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "buying_power": float(account.buying_power),
                "equity": float(account.equity),
                "last_equity": float(account.last_equity),
                "long_market_value": float(account.long_market_value),
                "short_market_value": float(account.short_market_value),
                "initial_margin": float(account.initial_margin),
                "maintenance_margin": float(account.maintenance_margin),
                "daytrade_count": account.daytrade_count,
                "pattern_day_trader": account.pattern_day_trader,
            }
            return self._account_info
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            raise

    async def get_positions(self) -> List[Dict]:
        """
        Get all open positions.

        Returns:
            List of position dictionaries
        """
        try:
            positions = self.client.get_all_positions()
            return [
                {
                    "symbol": pos.symbol,
                    "quantity": float(pos.qty),
                    "average_price": float(pos.avg_entry_price),
                    "current_price": float(pos.current_price),
                    "market_value": float(pos.market_value),
                    "cost_basis": float(pos.cost_basis),
                    "unrealized_pl": float(pos.unrealized_pl),
                    "unrealized_pl_percent": float(pos.unrealized_plpc),
                    "side": pos.side,
                }
                for pos in positions
            ]
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []

    async def get_position(self, symbol: str) -> Optional[Dict]:
        """
        Get position for a specific symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Position dict or None
        """
        try:
            pos = self.client.get_open_position(symbol)
            return {
                "symbol": pos.symbol,
                "quantity": float(pos.qty),
                "average_price": float(pos.avg_entry_price),
                "current_price": float(pos.current_price),
                "market_value": float(pos.market_value),
                "cost_basis": float(pos.cost_basis),
                "unrealized_pl": float(pos.unrealized_pl),
                "unrealized_pl_percent": float(pos.unrealized_plpc),
                "side": pos.side,
            }
        except Exception as e:
            logger.debug(f"No position for {symbol}: {e}")
            return None

    async def submit_market_order(
        self,
        symbol: str,
        quantity: float,
        side: str,
        time_in_force: str = "day",
    ) -> Optional[Dict]:
        """
        Submit a market order.

        Args:
            symbol: Stock symbol
            quantity: Number of shares (fractional allowed)
            side: "buy" or "sell"
            time_in_force: Order time in force

        Returns:
            Order dict or None
        """
        try:
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

            request = MarketOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=order_side,
                time_in_force=TimeInForce.DAY,
            )

            order = self.client.submit_order(request)
            logger.info(f"Market order submitted: {side} {quantity} {symbol}")

            return self._order_to_dict(order)

        except Exception as e:
            logger.error(f"Error submitting market order for {symbol}: {e}")
            return None

    async def submit_limit_order(
        self,
        symbol: str,
        quantity: float,
        side: str,
        limit_price: float,
        time_in_force: str = "day",
    ) -> Optional[Dict]:
        """
        Submit a limit order.

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            side: "buy" or "sell"
            limit_price: Limit price
            time_in_force: Order time in force

        Returns:
            Order dict or None
        """
        try:
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

            request = LimitOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=order_side,
                limit_price=limit_price,
                time_in_force=TimeInForce.DAY,
            )

            order = self.client.submit_order(request)
            logger.info(
                f"Limit order submitted: {side} {quantity} {symbol} @ ${limit_price}"
            )

            return self._order_to_dict(order)

        except Exception as e:
            logger.error(f"Error submitting limit order for {symbol}: {e}")
            return None

    async def submit_stop_order(
        self,
        symbol: str,
        quantity: float,
        side: str,
        stop_price: float,
        time_in_force: str = "day",
    ) -> Optional[Dict]:
        """
        Submit a stop order.

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            side: "buy" or "sell"
            stop_price: Stop price
            time_in_force: Order time in force

        Returns:
            Order dict or None
        """
        try:
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

            request = StopOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=order_side,
                stop_price=stop_price,
                time_in_force=TimeInForce.DAY,
            )

            order = self.client.submit_order(request)
            logger.info(f"Stop order submitted: {side} {quantity} {symbol} @ ${stop_price}")

            return self._order_to_dict(order)

        except Exception as e:
            logger.error(f"Error submitting stop order for {symbol}: {e}")
            return None

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.

        Args:
            order_id: Alpaca order ID

        Returns:
            True if successful
        """
        try:
            self.client.cancel_order_by_id(order_id)
            logger.info(f"Order {order_id} canceled")
            return True
        except Exception as e:
            logger.error(f"Error canceling order {order_id}: {e}")
            return False

    async def cancel_all_orders(self) -> bool:
        """
        Cancel all open orders.

        Returns:
            True if successful
        """
        try:
            self.client.cancel_orders()
            logger.info("All orders canceled")
            return True
        except Exception as e:
            logger.error(f"Error canceling all orders: {e}")
            return False

    async def get_order(self, order_id: str) -> Optional[Dict]:
        """
        Get order details.

        Args:
            order_id: Alpaca order ID

        Returns:
            Order dict or None
        """
        try:
            order = self.client.get_order_by_id(order_id)
            return self._order_to_dict(order)
        except Exception as e:
            logger.error(f"Error getting order {order_id}: {e}")
            return None

    async def get_orders(self, status: str = "all", limit: int = 100) -> List[Dict]:
        """
        Get orders.

        Args:
            status: Order status filter
            limit: Maximum number of orders

        Returns:
            List of order dicts
        """
        try:
            request = GetOrdersRequest(
                status=QueryOrderStatus.ALL if status == "all" else status,
                limit=limit,
            )
            orders = self.client.get_orders(request)
            return [self._order_to_dict(order) for order in orders]
        except Exception as e:
            logger.error(f"Error getting orders: {e}")
            return []

    async def close_position(self, symbol: str) -> bool:
        """
        Close a position.

        Args:
            symbol: Stock symbol

        Returns:
            True if successful
        """
        try:
            self.client.close_position(symbol)
            logger.info(f"Position closed for {symbol}")
            return True
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")
            return False

    async def close_all_positions(self) -> bool:
        """
        Close all positions.

        Returns:
            True if successful
        """
        try:
            self.client.close_all_positions()
            logger.info("All positions closed")
            return True
        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
            return False

    def _order_to_dict(self, order: AlpacaOrder) -> Dict:
        """Convert Alpaca order to dict."""
        return {
            "id": order.id,
            "symbol": order.symbol,
            "side": order.side.value,
            "type": order.type.value,
            "status": order.status.value,
            "quantity": float(order.qty),
            "filled_quantity": float(order.filled_qty) if order.filled_qty else 0.0,
            "limit_price": float(order.limit_price) if order.limit_price else None,
            "stop_price": float(order.stop_price) if order.stop_price else None,
            "filled_price": float(order.filled_avg_price) if order.filled_avg_price else None,
            "submitted_at": order.submitted_at,
            "filled_at": order.filled_at,
            "canceled_at": order.canceled_at,
            "created_at": order.created_at,
        }


# Global instance
alpaca_client = AlpacaClient()
