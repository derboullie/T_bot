"""Polygon.io API client for real-time and historical market data."""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import aiohttp
from loguru import logger
from polygon import RESTClient, WebSocketClient
from polygon.websocket.models import WebSocketMessage, EquityTrade, EquityQuote, EquityAgg

from backend.core.config import settings


class PolygonClient:
    """Client for Polygon.io market data API."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Polygon client.

        Args:
            api_key: Polygon.io API key (default from settings)
        """
        self.api_key = api_key or settings.polygon_api_key
        self.rest_client = RESTClient(self.api_key)
        self.ws_client = None
        self._subscriptions = set()
        self._handlers = {}
        self._connected = False

    async def get_stock_price(self, symbol: str) -> Optional[Dict]:
        """
        Get current stock price.

        Args:
            symbol: Stock symbol

        Returns:
            Dict with price information or None
        """
        try:
            # Get last trade
            trades = self.rest_client.get_last_trade(symbol)
            if trades:
                return {
                    "symbol": symbol,
                    "price": trades.price,
                    "size": trades.size,
                    "timestamp": datetime.fromtimestamp(trades.sip_timestamp / 1000000000),
                }
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
        return None

    async def get_stock_quote(self, symbol: str) -> Optional[Dict]:
        """
        Get current bid/ask quote.

        Args:
            symbol: Stock symbol

        Returns:
            Dict with quote information or None
        """
        try:
            quote = self.rest_client.get_last_quote(symbol)
            if quote:
                return {
                    "symbol": symbol,
                    "bid_price": quote.bid_price,
                    "bid_size": quote.bid_size,
                    "ask_price": quote.ask_price,
                    "ask_size": quote.ask_size,
                    "spread": quote.ask_price - quote.bid_price,
                    "timestamp": datetime.fromtimestamp(quote.sip_timestamp / 1000000000),
                }
        except Exception as e:
            logger.error(f"Error getting quote for {symbol}: {e}")
        return None

    async def get_historical_bars(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timespan: str = "day",
        limit: int = 5000,
    ) -> List[Dict]:
        """
        Get historical OHLCV bars.

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            timespan: Timespan (minute, hour, day, week, month)
            limit: Maximum number of bars

        Returns:
            List of price bars
        """
        try:
            logger.info(
                f"Fetching historical data for {symbol}: "
                f"{start_date.date()} to {end_date.date()}"
            )

            bars = []
            for bar in self.rest_client.list_aggs(
                symbol,
                1,
                timespan,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
                limit=limit,
            ):
                bars.append({
                    "timestamp": datetime.fromtimestamp(bar.timestamp / 1000),
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                    "vwap": bar.vwap if hasattr(bar, "vwap") else None,
                    "trades": bar.transactions if hasattr(bar, "transactions") else None,
                })

            logger.info(f"Retrieved {len(bars)} bars for {symbol}")
            return bars

        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return []

    async def get_stock_details(self, symbol: str) -> Optional[Dict]:
        """
        Get stock details and metadata.

        Args:
            symbol: Stock symbol

        Returns:
            Dict with stock details or None
        """
        try:
            details = self.rest_client.get_ticker_details(symbol)
            if details:
                return {
                    "symbol": symbol,
                    "name": details.name,
                    "market_cap": details.market_cap if hasattr(details, "market_cap") else None,
                    "sector": details.sic_description if hasattr(details, "sic_description") else None,
                    "exchange": details.primary_exchange if hasattr(details, "primary_exchange") else None,
                    "type": details.type if hasattr(details, "type") else None,
                }
        except Exception as e:
            logger.error(f"Error getting details for {symbol}: {e}")
        return None

    def _handle_ws_message(self, messages: List[WebSocketMessage]):
        """
        Handle WebSocket messages.

        Args:
            messages: List of WebSocket messages
        """
        for msg in messages:
            try:
                if isinstance(msg, EquityTrade):
                    self._handle_trade(msg)
                elif isinstance(msg, EquityQuote):
                    self._handle_quote(msg)
                elif isinstance(msg, EquityAgg):
                    self._handle_agg(msg)
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {e}")

    def _handle_trade(self, trade: EquityTrade):
        """Handle trade message."""
        if "trade" in self._handlers:
            data = {
                "symbol": trade.symbol,
                "price": trade.price,
                "size": trade.size,
                "timestamp": datetime.fromtimestamp(trade.timestamp / 1000000000),
            }
            self._handlers["trade"](data)

    def _handle_quote(self, quote: EquityQuote):
        """Handle quote message."""
        if "quote" in self._handlers:
            data = {
                "symbol": quote.symbol,
                "bid_price": quote.bid_price,
                "bid_size": quote.bid_size,
                "ask_price": quote.ask_price,
                "ask_size": quote.ask_size,
                "timestamp": datetime.fromtimestamp(quote.timestamp / 1000000000),
            }
            self._handlers["quote"](data)

    def _handle_agg(self, agg: EquityAgg):
        """Handle aggregate (bar) message."""
        if "agg" in self._handlers:
            data = {
                "symbol": agg.symbol,
                "open": agg.open,
                "high": agg.high,
                "low": agg.low,
                "close": agg.close,
                "volume": agg.volume,
                "timestamp": datetime.fromtimestamp(agg.start_timestamp / 1000),
            }
            self._handlers["agg"](data)

    async def subscribe_stocks(
        self,
        symbols: List[str],
        on_trade=None,
        on_quote=None,
        on_agg=None,
    ):
        """
        Subscribe to real-time stock data via WebSocket.

        Args:
            symbols: List of stock symbols
            on_trade: Callback for trade messages
            on_quote: Callback for quote messages
            on_agg: Callback for aggregate messages
        """
        if on_trade:
            self._handlers["trade"] = on_trade
        if on_quote:
            self._handlers["quote"] = on_quote
        if on_agg:
            self._handlers["agg"] = on_agg

        # Create WebSocket client
        if not self.ws_client:
            self.ws_client = WebSocketClient(
                api_key=self.api_key,
                feed="delayed.polygon.io",  # Use "delayed" for free tier, "stocks" for paid
                market="stocks",
                on_message=self._handle_ws_message,
            )

        # Subscribe to symbols
        for symbol in symbols:
            if on_trade:
                self.ws_client.subscribe(f"T.{symbol}")
            if on_quote:
                self.ws_client.subscribe(f"Q.{symbol}")
            if on_agg:
                self.ws_client.subscribe(f"AM.{symbol}")  # Minute aggregates

        self._subscriptions.update(symbols)
        logger.info(f"Subscribed to {len(symbols)} symbols via WebSocket")

    async def start_websocket(self):
        """Start WebSocket connection."""
        if self.ws_client and not self._connected:
            logger.info("Starting Polygon WebSocket connection...")
            self.ws_client.run()
            self._connected = True

    async def stop_websocket(self):
        """Stop WebSocket connection."""
        if self.ws_client and self._connected:
            logger.info("Stopping Polygon WebSocket connection...")
            self.ws_client.close()
            self._connected = False

    async def close(self):
        """Close all connections."""
        await self.stop_websocket()
        logger.info("Polygon client closed")


# Global instance
polygon_client = PolygonClient()
