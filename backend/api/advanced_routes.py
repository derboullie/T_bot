"""Enhanced API endpoints for backtesting, ML, and multi-exchange support."""

from typing import List, Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel

from backend.data.database import get_db
from backend.backtesting.backtest_engine import BacktestEngine, BacktestConfig
from backend.strategies.momentum_strategy import MomentumStrategy
from backend.strategies.arbitrage_strategy import ArbitrageStrategy
from backend.strategies.market_making_strategy import MarketMakingStrategy
from backend.strategies.stat_arb_strategy import StatisticalArbitrageStrategy
from backend.ml.model_trainer import model_trainer
from backend.exchanges.ccxt_client import multi_exchange_manager
from backend.data.polygon_client import polygon_client

router = APIRouter(prefix="/api/advanced", tags=["advanced"])


# Pydantic models
class BacktestRequest(BaseModel):
    strategy: str
    symbol: str
    start_date: str
    end_date: str
    initial_capital: float = 100000.0


class TrainModelRequest(BaseModel):
    symbol: str
    episodes: int = 100


class ArbitrageCheckRequest(BaseModel):
    symbols: List[str]


# Backtesting endpoints
@router.post("/backtest")
async def run_backtest(request: BacktestRequest):
    """
    Run backtest for a strategy.
    
    Args:
        request: Backtest parameters
        
    Returns:
        Backtest results
    """
    try:
        # Select strategy
        strategy_map = {
            'momentum': MomentumStrategy(),
            'arbitrage': ArbitrageStrategy(),
            'market_making': MarketMakingStrategy(),
            'stat_arb': StatisticalArbitrageStrategy(),
        }
        
        if request.strategy not in strategy_map:
            raise HTTPException(400, f"Unknown strategy: {request.strategy}")
            
        strategy = strategy_map[request.strategy]
        
        # Fetch historical data
        start = datetime.fromisoformat(request.start_date)
        end = datetime.fromisoformat(request.end_date)
        
        bars = await polygon_client.get_historical_bars(
            request.symbol,
            start,
            end,
            'day'
        )
        
        if not bars:
            raise HTTPException(404, "No historical data found")
            
        import pandas as pd
        df = pd.DataFrame(bars)
        df.set_index('timestamp', inplace=True)
        df['symbol'] = request.symbol
        
        # Run backtest
        config = BacktestConfig(initial_capital=request.initial_capital)
        engine = BacktestEngine(config)
        
        results = await engine.run(strategy, df, start, end)
        
        # Return results
        return {
            'total_return': results.total_return,
            'total_return_pct': results.total_return_pct,
            'sharpe_ratio': results.sharpe_ratio,
            'sortino_ratio': results.sortino_ratio,
            'max_drawdown_pct': results.max_drawdown_pct,
            'total_trades': results.total_trades,
            'win_rate': results.win_rate,
            'profit_factor': results.profit_factor,
            'avg_win': results.avg_win,
            'avg_loss': results.avg_loss,
            'calmar_ratio': results.calmar_ratio,
        }
        
    except Exception as e:
        raise HTTPException(500, str(e))


# ML endpoints
@router.post("/ml/train")
async def train_model(request: TrainModelRequest):
    """
    Train ML model for a symbol.
    
    Args:
        request: Training parameters
        
    Returns:
        Training results
    """
    try:
        agent, metrics = await model_trainer.train_model(
            request.symbol,
            request.episodes
        )
        
        return {
            'status': 'completed',
            'symbol': request.symbol,
            'episodes': request.episodes,
            'metrics': metrics,
        }
        
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/ml/status")
async def get_ml_status():
    """Get ML training status."""
    stats = model_trainer.get_training_stats()
    return stats


# Multi-exchange endpoints
@router.get("/exchanges/tickers/{symbol}")
async def get_multi_exchange_tickers(symbol: str):
    """
    Get ticker from all exchanges.
    
    Args:
        symbol: Trading pair
        
    Returns:
        Tickers from all exchanges
    """
    try:
        tickers = await multi_exchange_manager.get_ticker(symbol)
        return tickers
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/exchanges/arbitrage")
async def check_arbitrage(request: ArbitrageCheckRequest):
    """
    Check for arbitrage opportunities.
    
    Args:
        request: Symbols to check
        
    Returns:
        Arbitrage opportunities
    """
    try:
        opportunities = await multi_exchange_manager.find_arbitrage_opportunities(
            request.symbols
        )
        return {'opportunities': opportunities}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/exchanges/balances")
async def get_all_balances():
    """Get balances from all exchanges."""
    try:
        balances = await multi_exchange_manager.get_balance()
        return balances
    except Exception as e:
        raise HTTPException(500, str(e))
