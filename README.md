# HFT Trading Bot

High-Frequency Trading (HFT) Bot for US stocks with real-time market data, backtesting, machine learning, and a modern web dashboard.

## 🚀 Features

- **Real-time Market Data**: Live stock prices, quotes, and trades via Polygon.io WebSocket
- **Trading Execution**: Automated trading through Alpaca API (paper & live trading)
- **Multiple Strategies**: Momentum, mean reversion, and ML-powered strategies
- **Backtesting Engine**: Test strategies on historical data with detailed metrics
- **Machine Learning**: Reinforcement learning (DQN) for strategy optimization
- **Risk Management**: Position size limits, daily loss limits, and automatic circuit breakers
- **CPU Management**: Intelligent throttling to stay under 85% CPU usage
- **Modern Web Dashboard**: Real-time monitoring with Material-UI
- **WebSocket Updates**: Live portfolio and performance updates

## 📋 Prerequisites

- **Python 3.11+**
- **Node.js 18+** and npm
- **Poetry** (Python package manager)
- **API Keys**:
  - [Polygon.io](https://polygon.io) API key
  - [Alpaca](https://alpaca.markets) API key and secret

## 🛠️ Installation

### 1. Clone the Repository

```bash
cd /home/der-boullie/work/robu/daham/Tradingbot
```

### 2. Backend Setup

```bash
# Install Poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Install Python dependencies
poetry install

# Copy environment file and add your API keys
cp .env.example .env
# Edit .env and add your API keys
```

### 3. Frontend Setup

```bash
cd frontend

# Install Node.js dependencies
npm install

cd ..
```

### 4. Configure Environment Variables

Edit `.env` and set your API credentials:

```bash
# Required API Keys
POLYGON_API_KEY=your_polygon_key_here
ALPACA_API_KEY=your_alpaca_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_here

# Use paper trading for testing (recommended)
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Or live trading (use with caution!)
# ALPACA_BASE_URL=https://api.alpaca.markets
```

## 🚦 Quick Start

### Option 1: Development Mode (Recommended)

**Terminal 1 - Backend:**
```bash
poetry run python -m backend.api.main
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

Then open [http://localhost:5173](http://localhost:5173) in your browser.

### Option 2: Production Build

```bash
# Build frontend
cd frontend
npm run build

# Run backend (serves frontend static files)
cd ..
poetry run uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
```

## 📊 Project Structure

```
Tradingbot/
├── backend/                 # Python backend
│   ├── api/                # FastAPI REST and WebSocket endpoints
│   ├── core/               # Configuration, CPU monitoring, throttling
│   ├── data/               # Database models, Polygon.io client
│   ├── trading/            # Alpaca client, order & risk management
│   ├── strategies/         # Trading strategies (momentum, ML, etc.)
│   ├── backtesting/        # Backtesting engine and metrics
│   ├── ml/                 # Machine learning models (DQN, RL)
│   └── tests/              # Unit and integration tests
├── frontend/               # React + TypeScript frontend
│   └── src/
│       ├── components/     # React components
│       ├── services/       # API and WebSocket services
│       └── store/          # Redux state management
├── .env                    # Environment variables (create from .env.example)
├── pyproject.toml          # Python dependencies
└── README.md               # This file
```

## 🎯 Usage

### Dashboard Overview

The web dashboard shows:
- **Portfolio Value**: Total account equity
- **Daily P&L**: Profit/Loss for the current day
- **Open Positions**: Number of active positions
- **CPU Usage**: Real-time CPU monitoring

### API Endpoints

- `GET /api/status` - Bot and system status
- `GET /api/account` - Account information
- `GET /api/positions` - All open positions
- `GET /api/orders` - Order history
- `POST /api/orders/market` - Place market order
- `GET /api/market/price/{symbol}` - Get stock price
- `WS /ws` - WebSocket for real-time updates

### Trading Strategies

Current strategies in `backend/strategies/`:

1. **Momentum Strategy**: Uses RSI and Moving Averages
   - Buy when RSI < 30 (oversold) or SMA crossover
   - Sell when RSI > 70 (overbought) or SMA crossunder

2. **Mean Reversion**: (To be implemented)
3. **ML Strategy**: Reinforcement Learning (DQN) - (To be implemented)

### Risk Management

Automatic risk controls:
- **Max Position Size**: $10,000 per position (configurable)
- **Max Daily Loss**: $1,000 (trading stops if reached)
- **Max Positions**: 10 concurrent positions
- **Risk Per Trade**: 1% of portfolio

### CPU Management

- Monitors CPU usage in real-time
- Automatically throttles when usage > 85%
- Pauses non-critical tasks to maintain system stability

## 🧪 Testing

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=backend --cov-report=html

# Run specific test file
poetry run pytest backend/tests/test_risk_manager.py

# Run integration tests
poetry run pytest backend/tests/integration/ -v
```

## 🔧 Configuration

Key settings in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `POLYGON_API_KEY` | Polygon.io API key | Required |
| `ALPACA_API_KEY` | Alpaca API key | Required |
| `ALPACA_SECRET_KEY` | Alpaca secret key | Required |
| `ALPACA_BASE_URL` | Paper or live trading | Paper trading |
| `MAX_POSITION_SIZE` | Max position size (USD) | 10000 |
| `MAX_DAILY_LOSS` | Max daily loss (USD) | 1000 |
| `CPU_LIMIT_PERCENT` | CPU usage limit | 85 |
| `WORKER_THREADS` | Number of worker threads | 4 |

## ⚠️ Important Notes

### Paper Trading First!
**Always start with paper trading** to test your strategies. Never use live trading without thorough testing.

### Financial Risk
This bot executes real trades. You can lose money. Use at your own risk.

### API Rate Limits
- Polygon.io: Check your plan's rate limits
- Alpaca: 200 requests/minute

### System Requirements
- Stable internet connection
- Sufficient CPU/RAM (bot monitors and throttles automatically)
- Min 4 CPU cores recommended

## 📈 Backtesting

```bash
# Run backtest via API
curl -X POST "http://localhost:8000/api/backtest" \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": "momentum",
    "symbols": ["AAPL", "GOOGL"],
    "start_date": "2023-01-01",
    "end_date": "2023-12-31"
  }'
```

## 🤖 Machine Learning

The ML module uses Deep Q-Networks (DQN) for reinforcement learning:

1. **Feature Engineering**: Technical indicators, price patterns
2. **Training**: Learn from historical data
3. **Validation**: Test on unseen data
4. **Deployment**: Use trained models for live trading

Enable ML in `.env`:
```bash
ML_TRAINING_ENABLED=true
```

## 🐛 Troubleshooting

**Backend won't start:**
- Check `.env` file has all required API keys
- Ensure database is accessible
- Check port 8000 is not in use

**Frontend won't start:**
- Run `npm install` in frontend directory
- Check port 5173 is available
- Ensure backend is running

**No market data:**
- Verify Polygon.io API key is correct
- Check market hours (US stocks: 9:30 AM - 4:00 PM ET)
- Free tier has 15-minute delays

**Orders not executing:**
- Verify Alpaca API keys
- Check account has sufficient buying power
- Ensure within risk limits

## 📝 License

This project is for educational purposes. Use at your own risk.

## 🤝 Contributing

This is a personal trading bot. Contributions and suggestions are welcome!

## 📧 Support

For issues and questions, please check:
- Polygon.io [documentation](https://polygon.io/docs)
- Alpaca [documentation](https://alpaca.markets/docs)
- Project issues on GitHub

---

**Disclaimer**: This software is provided for educational purposes only. Trading stocks involves risk. Past performance does not guarantee future results. The authors are not responsible for any financial losses incurred through the use of this software.
