"""Data manager for centralized market data handling."""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from collections import deque

import pandas as pd
from loguru import logger
from sqlalchemy.orm import Session

from backend.core.config import settings
from backend.core.throttler import throttler
from backend.data.database import database
from backend.data.models import Stock, PriceData
from backend.data.polygon_client import polygon_client


class DataManager:
    """
    Centralized manager for all market data operations.
    
    Features:
    - Real-time data streaming
    - Historical data caching
    - Memory-efficient storage
    - Automatic cleanup
    """

    def __init__(self, max_cache_size: int = 1000):
        """
        Initialize data manager.
        
        Args:
            max_cache_size: Maximum number of bars to keep in memory per symbol
        """
        self.max_cache_size = max_cache_size
        self._subscribed_symbols: Set[str] = set()
        self._price_cache: Dict[str, deque] = {}
        self._latest_prices: Dict[str, float] = {}
        self._running = False
        
    async def subscribe(self, symbols: List[str]):
        """
        Subscribe to real-time data for symbols.
        
        Args:
            symbols: List of stock symbols to subscribe
        """
        new_symbols = [s.upper() for s in symbols if s.upper() not in self._subscribed_symbols]
        
        if not new_symbols:
            return
            
        logger.info(f"Subscribing to {len(new_symbols)} symbols: {new_symbols}")
        
        # Initialize cache for new symbols
        for symbol in new_symbols:
            if symbol not in self._price_cache:
                self._price_cache[symbol] = deque(maxlen=self.max_cache_size)
                
        # Subscribe to Polygon.io WebSocket
        await polygon_client.subscribe_stocks(
            new_symbols,
            on_trade=self._handle_trade,
            on_quote=self._handle_quote,
        )
        
        self._subscribed_symbols.update(new_symbols)
        
    def _handle_trade(self, trade_data: Dict):
        """Handle incoming trade data."""
        symbol = trade_data["symbol"]
        price = trade_data["price"]
        
        self._latest_prices[symbol] = price
        logger.debug(f"{symbol}: ${price:.2f}")
        
    def _handle_quote(self, quote_data: Dict):
        """Handle incoming quote data."""
        symbol = quote_data["symbol"]
        mid_price = (quote_data["bid_price"] + quote_data["ask_price"]) / 2
        
        self._latest_prices[symbol] = mid_price
        
    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Get latest price for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Latest price or None
        """
        symbol = symbol.upper()
        
        # Try cache first
        if symbol in self._latest_prices:
            return self._latest_prices[symbol]
            
        # Fetch from API
        await throttler.throttle_if_needed()
        price_data = await polygon_client.get_stock_price(symbol)
        
        if price_data:
            price = price_data["price"]
            self._latest_prices[symbol] = price
            return price
            
        return None
        
    async def get_historical_data(
        self,
        symbol: str,
        days: int = 365,
        timespan: str = "day",
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data.
        
        Args:
            symbol: Stock symbol
            days: Number of days of history
            timespan: Bar timespan (minute, hour, day)
            
        Returns:
            DataFrame with OHLCV data
        """
        symbol = symbol.upper()
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        logger.info(f"Fetching {days} days of historical data for {symbol}")
        
        await throttler.throttle_if_needed()
        bars = await polygon_client.get_historical_bars(
            symbol, start_date, end_date, timespan
        )
        
        if not bars:
            logger.warning(f"No historical data found for {symbol}")
            return pd.DataFrame()
            
        # Convert to DataFrame
        df = pd.DataFrame(bars)
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        
        logger.info(f"Retrieved {len(df)} bars for {symbol}")
        return df
        
    async def store_historical_data(
        self,
        symbol: str,
        data: pd.DataFrame,
        db: Session
    ):
        """
        Store historical data in database.
        
        Args:
            symbol: Stock symbol
            data: Historical price data
            db: Database session
        """
        # Get or create stock entry
        stock = db.query(Stock).filter(Stock.symbol == symbol).first()
        
        if not stock:
            stock = Stock(symbol=symbol, is_active=True)
            db.add(stock)
            db.commit()
            db.refresh(stock)
            
        # Store price data
        for timestamp, row in data.iterrows():
            price_data = PriceData(
                stock_id=stock.id,
                timestamp=timestamp,
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
                vwap=row.get("vwap"),
                trades=row.get("trades"),
            )
            db.add(price_data)
            
        db.commit()
        logger.info(f"Stored {len(data)} bars for {symbol} in database")
        
    def get_subscribed_symbols(self) -> List[str]:
        """Get list of subscribed symbols."""
        return list(self._subscribed_symbols)
        
    def get_stats(self) -> Dict:
        """Get data manager statistics."""
        return {
            "subscribed_symbols": len(self._subscribed_symbols),
            "cached_symbols": len(self._price_cache),
            "latest_prices": len(self._latest_prices),
        }


# Global instance
data_manager = DataManager()
