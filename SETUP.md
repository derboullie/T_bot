# Quick Setup Guide

## 📦 Installation Steps

### 1. Install Poetry (if not installed)
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### 2. Install Backend Dependencies
```bash
poetry install
```

This will install all Python packages including:
- FastAPI, Uvicorn
- SQLAlchemy, psycopg2
- Polygon API client, Alpaca API
- TensorFlow, scikit-learn
- psutil, and more

### 3. Install Frontend Dependencies
```bash
cd frontend
npm install
cd ..
```

This will install:
- React, React Router
- Material-UI
- Recharts
- Axios, Socket.IO
- TypeScript, Vite

### 4. Configure Environment Variables
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your API keys
nano .env  # or use your preferred editor
```

**Required values:**
```
POLYGON_API_KEY=your_polygon_api_key
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
```

**Get API keys:**
- Polygon.io: https://polygon.io (Sign up → Dashboard → API Keys)
- Alpaca: https://alpaca.markets (Sign up → Paper Trading → API Keys)

### 5. Initialize Database
The database will be created automatically when you first run the backend.

SQLite database file: `tradingbot.db` (created in project root)

---

## 🚀 Starting the Application

### Option A: Use the Start Script (Easy)
```bash
./start.sh
```

This will start both backend and frontend automatically.

### Option B: Manual Start (for development)

**Terminal 1 - Backend:**
```bash
poetry run python -m backend.api.main
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

---

## 🌐 Access the Application

- **Frontend Dashboard**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs (Swagger UI)

---

## ✅ Verify Installation

### Test Backend
```bash
# Health check
curl http://localhost:8000/api/health

# Get status (requires API keys configured)
curl http://localhost:8000/api/status
```

### Test Frontend
Open your browser to http://localhost:5173 and you should see:
- Header with "HFT Trading Bot"
- 4 overview cards (Portfolio, P&L, Positions, CPU)
- Empty positions table (no trades yet)
- Account information (from Alpaca)
- Risk management panel

---

## 🔧 Troubleshooting

### Backend won't start

**Error: Missing .env file**
```bash
cp .env.example .env
# Edit and add API keys
```

**Error: Module not found**
```bash
poetry install
```

**Error: Port 8000 already in use**
```bash
# Find process using port 8000
lsof -i :8000
# Kill it or use a different port
```

### Frontend won't start

**Error: Dependencies missing**
```bash
cd frontend
npm install
```

**Error: Port 5173 in use**
```bash
# Vite will automatically try port 5174, 5175, etc.
```

### API Connection Issues

**401 Unauthorized / 403 Forbidden**
- Check your API keys in `.env`
- Verify keys are correct on Polygon.io and Alpaca dashboards

**No market data**
- Market must be open (9:30 AM - 4:00 PM ET, Mon-Fri)
- Free Polygon tier has 15-minute delay
- Check symbol is correct (e.g., "AAPL" not "Apple")

---

## 🧪 Testing

### Run Backend Tests
```bash
poetry run pytest
```

### Run with Coverage
```bash
poetry run pytest --cov=backend --cov-report=html
open htmlcov/index.html
```

---

## 📝 Next Steps After Installation

1. **Verify Dashboard Loads**
   - Check all stats display correctly
   - Verify WebSocket connection (check browser console)

2. **Test with Paper Trading**
   - Place a small test order via API
   - Verify it appears in Alpaca dashboard
   - Check position shows in bot dashboard

3. **Configure Trading Parameters**
   - Edit `.env` to adjust risk limits
   - Set `MAX_POSITION_SIZE`, `MAX_DAILY_LOSS`, etc.

4. **Develop Strategy**
   - Test momentum strategy with real data
   - Adjust parameters for better results
   - Create custom strategies

5. **Implement Backtesting**
   - Build backtesting engine
   - Test strategies on historical data
   - Optimize parameters

6. **Add Machine Learning**
   - Create training pipeline
   - Train DQN model
   - Evaluate and deploy

---

## ⚠️ Important Notes

### Paper Trading First!
The `.env.example` defaults to paper trading:
```
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

**Do NOT change to live trading until thoroughly tested!**

### CPU Monitoring
The bot will automatically throttle at 85% CPU usage. This is configurable:
```
CPU_LIMIT_PERCENT=85  # Adjust as needed
```

### Database
- Dev: SQLite (single file, easy)
- Prod: PostgreSQL (recommended)

To switch to PostgreSQL:
```
DATABASE_URL=postgresql://user:password@localhost:5432/tradingbot
```

---

## 📚 Documentation

- **README.md**: Full documentation
- **Walkthrough.md**: Implementation details
- **Implementation Plan**: Architecture and design
- **API Docs**: http://localhost:8000/docs (when running)

---

## 🆘 Support

If you encounter issues:

1. Check logs:
   - Backend: Terminal output
   - Frontend: Browser console (F12)
   - Files: `logs/tradingbot.log`

2. Verify configuration:
   - `.env` file exists and has all keys
   - API keys are valid

3. Check API status:
   - Polygon.io status page
   - Alpaca status page

4. Review documentation:
   - Polygon.io docs: https://polygon.io/docs
   - Alpaca docs: https://alpaca.markets/docs

---

Good luck with your trading bot! 🚀📈
