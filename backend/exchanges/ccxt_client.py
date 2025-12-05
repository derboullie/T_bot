"""Multi-Exchange Manager using CCXT.

Supports multiple cryptocurrency exchanges for arbitrage and diversification.
"""

import asyncio
from typing import Dict, List, Optional
from datetime import datetime

import ccxt.async_support as ccxt
from loguru import logger

from backend.core.config import settings


class MultiExchangeManager:
    """
    Manage multiple cryptocurrency exchanges.
    
    Features:
    - Unified interface across exchanges
    - Concurrent price fetching
    - Order routing
    - Balance aggregation
    """
    
    def __init__(self):
        """Initialize multi-exchange manager."""
        self.exchanges = {}
        self._initialize_exchanges()
        
    def _initialize_exchanges(self):
        """Initialize configured exchanges."""
        # Binance
        if hasattr(settings, 'binance_api_key'):
            self.exchanges['binance'] = ccxt.binance({
                'apiKey': settings.binance_api_key,
                'secret': settings.binance_secret,
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'},
            })
            logger.info("Binance exchange initialized")
            
        # Kraken
        if hasattr(settings, 'kraken_api_key'):
            self.exchanges['kraken'] = ccxt.kraken({
                'apiKey': settings.kraken_api_key,
                'secret': settings.kraken_secret,
                'enableRateLimit': True,
            })
            logger.info("Kraken exchange initialized")
            
        # Coinbase
        if hasattr(settings, 'coinbase_api_key'):
            self.exchanges['coinbase'] = ccxt.coinbase({
                'apiKey': settings.coinbase_api_key,
                'secret': settings.coinbase_secret,
                'enableRateLimit': True,
            })
            logger.info("Coinbase exchange initialized")
            
        if not self.exchanges:
            logger.warning("No exchanges configured")
            
    async def get_ticker(self, symbol: str, exchange: Optional[str] = None) -> Dict:
        """
        Get ticker for symbol.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USD')
            exchange: Specific exchange or None for all
            
        Returns:
            Ticker data
        """
        if exchange:
            if exchange in self.exchanges:
                return await self.exchanges[exchange].fetch_ticker(symbol)
            else:
                raise ValueError(f"Exchange {exchange} not configured")
        else:
            # Get from all exchanges
            tasks = [
                ex.fetch_ticker(symbol)
                for ex in self.exchanges.values()
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            tickers = {}
            for ex_name, result in zip(self.exchanges.keys(), results):
                if isinstance(result, dict):
                    tickers[ex_name] = result
                else:
                    logger.error(f"Error fetching {symbol} from {ex_name}: {result}")
                    
            return tickers
            
    async def get_best_price(self, symbol: str, side: str) -> Dict:
        """
        Find best price across all exchanges.
        
        Args:
            symbol: Trading pair
            side: 'buy' or 'sell'
            
        Returns:
            Best price info
        """
        tickers = await self.get_ticker(symbol)
        
        if not tickers:
            return None
            
        if side == 'buy':
            # Find lowest ask
            best = min(tickers.items(), key=lambda x: x[1]['ask'])
            return {
                'exchange': best[0],
                'price': best[1]['ask'],
                'side': 'buy'
            }
        else:
            # Find highest bid
            best = max(tickers.items(), key=lambda x: x[1]['bid'])
            return {
                'exchange': best[0],
                'price': best[1]['bid'],
                'side': 'sell'
            }
            
    async def find_arbitrage_opportunities(self, symbols: List[str]) -> List[Dict]:
        """
        Find arbitrage opportunities across exchanges.
        
        Args:
            symbols: List of symbols to check
            
        Returns:
            List of opportunities
        """
        opportunities = []
        
        for symbol in symbols:
            try:
                tickers = await self.get_ticker(symbol)
                
                if len(tickers) < 2:
                    continue
                    
                # Find min ask and max bid
                min_ask = min(tickers.items(), key=lambda x: x[1]['ask'])
                max_bid = max(tickers.items(), key=lambda x: x[1]['bid'])
                
                if min_ask[0] != max_bid[0]:  # Different exchanges
                    spread = max_bid[1]['bid'] - min_ask[1]['ask']
                    spread_pct = (spread / min_ask[1]['ask']) * 100
                    
                    if spread_pct > 0.5:  # 0.5% minimum
                        opportunities.append({
                            'symbol': symbol,
                            'buy_exchange': min_ask[0],
                            'buy_price': min_ask[1]['ask'],
                            'sell_exchange': max_bid[0],
                            'sell_price': max_bid[1]['bid'],
                            'spread': spread,
                            'spread_pct': spread_pct,
                            'timestamp': datetime.now(),
                        })
                        
            except Exception as e:
                logger.error(f"Error checking arbitrage for {symbol}: {e}")
                
        return opportunities
        
    async def place_order(
        self,
        exchange: str,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: Optional[float] = None
    ) -> Dict:
        """
        Place order on specific exchange.
        
        Args:
            exchange: Exchange name
            symbol: Trading pair
            order_type: 'market' or 'limit'
            side: 'buy' or 'sell'
            amount: Order amount
            price: Limit price (for limit orders)
            
        Returns:
            Order info
        """
        if exchange not in self.exchanges:
            raise ValueError(f"Exchange {exchange} not configured")
            
        ex = self.exchanges[exchange]
        
        if order_type == 'market':
            if side == 'buy':
                order = await ex.create_market_buy_order(symbol, amount)
            else:
                order = await ex.create_market_sell_order(symbol, amount)
        else:  # limit
            if side == 'buy':
                order = await ex.create_limit_buy_order(symbol, amount, price)
            else:
                order = await ex.create_limit_sell_order(symbol, amount, price)
                
        logger.info(f"Order placed on {exchange}: {order['id']}")
        return order
        
    async def get_balance(self, exchange: Optional[str] = None) -> Dict:
        """
        Get balance from exchange(s).
        
        Args:
            exchange: Specific exchange or None for all
            
        Returns:
            Balance info
        """
        if exchange:
            return await self.exchanges[exchange].fetch_balance()
        else:
            balances = {}
            for ex_name, ex in self.exchanges.items():
                try:
                    balances[ex_name] = await ex.fetch_balance()
                except Exception as e:
                    logger.error(f"Error fetching balance from {ex_name}: {e}")
            return balances
            
    async def close(self):
        """Close all exchange connections."""
        for ex in self.exchanges.values():
            await ex.close()
        logger.info("All exchanges closed")


# Global instance
multi_exchange_manager = MultiExchangeManager()
