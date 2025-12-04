"""Database models for the trading bot."""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import enum

Base = declarative_base()


class OrderSide(str, enum.Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, enum.Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, enum.Enum):
    """Order status enumeration."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELED = "canceled"
    REJECTED = "rejected"


class Stock(Base):
    """Stock metadata."""

    __tablename__ = "stocks"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), unique=True, index=True, nullable=False)
    name = Column(String(200))
    exchange = Column(String(50))
    sector = Column(String(100))
    industry = Column(String(100))
    market_cap = Column(Float)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    price_data = relationship("PriceData", back_populates="stock", cascade="all, delete-orphan")
    positions = relationship("Position", back_populates="stock", cascade="all, delete-orphan")
    orders = relationship("Order", back_populates="stock", cascade="all, delete-orphan")


class PriceData(Base):
    """OHLCV price data."""

    __tablename__ = "price_data"
    __table_args__ = (UniqueConstraint("stock_id", "timestamp", name="uix_stock_timestamp"),)

    id = Column(Integer, primary_key=True, index=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    vwap = Column(Float)  # Volume-weighted average price
    trades = Column(Integer)  # Number of trades
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    stock = relationship("Stock", back_populates="price_data")


class Position(Base):
    """Current and historical positions."""

    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, index=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    quantity = Column(Float, nullable=False)
    average_price = Column(Float, nullable=False)
    current_price = Column(Float)
    unrealized_pl = Column(Float)
    realized_pl = Column(Float, default=0.0)
    is_open = Column(Boolean, default=True, index=True)
    opened_at = Column(DateTime, default=datetime.utcnow)
    closed_at = Column(DateTime)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    stock = relationship("Stock", back_populates="positions")


class Order(Base):
    """Order history."""

    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, index=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    alpaca_order_id = Column(String(100), unique=True, index=True)
    side = Column(Enum(OrderSide), nullable=False)
    type = Column(Enum(OrderType), nullable=False)
    status = Column(Enum(OrderStatus), nullable=False, index=True)
    quantity = Column(Float, nullable=False)
    filled_quantity = Column(Float, default=0.0)
    limit_price = Column(Float)
    stop_price = Column(Float)
    filled_price = Column(Float)
    requested_at = Column(DateTime, default=datetime.utcnow)
    submitted_at = Column(DateTime)
    filled_at = Column(DateTime)
    canceled_at = Column(DateTime)
    strategy_name = Column(String(100))
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    stock = relationship("Stock", back_populates="orders")


class Trade(Base):
    """Completed trade records (buy + sell pair)."""

    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True)
    stock_symbol = Column(String(10), nullable=False, index=True)
    entry_order_id = Column(Integer, ForeignKey("orders.id"))
    exit_order_id = Column(Integer, ForeignKey("orders.id"))
    quantity = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=False)
    profit_loss = Column(Float, nullable=False)
    profit_loss_percent = Column(Float, nullable=False)
    commission = Column(Float, default=0.0)
    net_profit_loss = Column(Float)
    hold_duration = Column(Integer)  # Duration in seconds
    strategy_name = Column(String(100))
    entered_at = Column(DateTime, nullable=False)
    exited_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class BacktestResult(Base):
    """Backtesting results."""

    __tablename__ = "backtest_results"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    strategy_name = Column(String(100), nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    initial_capital = Column(Float, nullable=False)
    final_capital = Column(Float, nullable=False)
    total_return = Column(Float)
    total_return_percent = Column(Float)
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    max_drawdown = Column(Float)
    max_drawdown_percent = Column(Float)
    win_rate = Column(Float)
    profit_factor = Column(Float)
    total_trades = Column(Integer)
    winning_trades = Column(Integer)
    losing_trades = Column(Integer)
    average_win = Column(Float)
    average_loss = Column(Float)
    parameters = Column(Text)  # JSON string of strategy parameters
    created_at = Column(DateTime, default=datetime.utcnow)


class MLModel(Base):
    """Machine learning model metadata."""

    __tablename__ = "ml_models"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    model_type = Column(String(50), nullable=False)  # e.g., "DQN", "Q-Learning"
    version = Column(String(50))
    file_path = Column(String(500), nullable=False)
    training_start = Column(DateTime)
    training_end = Column(DateTime)
    training_samples = Column(Integer)
    validation_accuracy = Column(Float)
    test_accuracy = Column(Float)
    hyperparameters = Column(Text)  # JSON string
    metrics = Column(Text)  # JSON string of performance metrics
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class SystemLog(Base):
    """System events and errors log."""

    __tablename__ = "system_logs"

    id = Column(Integer, primary_key=True, index=True)
    level = Column(String(20), nullable=False, index=True)  # INFO, WARNING, ERROR, CRITICAL
    module = Column(String(100))
    message = Column(Text, nullable=False)
    details = Column(Text)  # JSON string with additional details
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
