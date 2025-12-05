# 🚀 HFT Trading Bot - Quick Start Guide

## ⚡ One-Click Start

### Linux/Mac:
```bash
./start.sh
```

### Windows:
```powershell
.\start.bat
```

Das wars! Der Bot:
- ✅ Aktualisiert alle Dependencies automatisch
- ✅ Startet Backend (Port 8000)
- ✅ Startet Frontend (Port 5173)
- ✅ Öffnet Browser automatisch
- ✅ Zeigt alle URLs an

---

## 🛑 Bot Stoppen

### Linux/Mac:
```bash
./stop.sh
```

### Windows:
```powershell
.\stop.bat
```

---

## 📊 Zugriff

Nach dem Start ist der Bot erreichbar unter:

- **🎨 Dashboard:** http://localhost:5173
- **📚 API Dokumentation:** http://localhost:8000/docs
- **💚 Health Check:** http://localhost:8000/api/health

---

## 📝 Logs Anzeigen

### Backend Logs:
```bash
tail -f /tmp/hft_backend.log
```

### Frontend Logs:
```bash
tail -f /tmp/hft_frontend.log
```

---

## ⚙️ Manuelle Steuerung

Falls Sie lieber manuell starten möchten:

### Backend:
```bash
poetry run uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend:
```bash
cd frontend
npm run dev
```

---

## 🔧 Konfiguration

Bevor Sie starten, stellen Sie sicher dass `.env` konfiguriert ist:

```bash
cp .env.example .env
# Bearbeiten Sie .env und fügen Sie Ihre API-Keys ein
```

Wichtige Variablen:
- `POLYGON_API_KEY` - Für Marktdaten
- `ALPACA_API_KEY` - Für Trading (Paper Trading standardmäßig)
- `ALPACA_SECRET_KEY` - Alpaca Secret

---

## 🎯 Features

**Verfügbar nach dem Start:**

✅ **Modernes Dashboard** mit Animationen  
✅ **Manuelle Trading Controls** (Order Entry)  
✅ **4 Trading Strategien** (Momentum, Arbitrage, Market-Making, Stat Arb)  
✅ **Machine Learning** (Self-Rewarding Double DQN)  
✅ **Backtesting Engine** mit Performance-Metriken  
✅ **Multi-Exchange Support** (CCXT)  
✅ **Real-time Market Data** (WebSocket)  
✅ **Risk Management** System  

---

## 🆘 Probleme?

### Port bereits belegt?
```bash
# Beende alle laufenden Prozesse
./stop.sh

# Oder manuell:
pkill -f uvicorn
pkill -f vite
```

### Dependencies-Fehler?
```bash
# Python Dependencies neu installieren
poetry install

# Node Dependencies neu installieren
cd frontend && npm install
```

### Backend startet nicht?
```bash
# Prüfe Logs
tail -f /tmp/hft_backend.log

# Prüfe Imports
poetry run python -c "from backend.api.main import app; print('OK')"
```

---

## 📖 Dokumentation

Weitere Dokumentation finden Sie in:
- `README.md` - Projektübersicht
- `SETUP.md` - Detaillierte Installation
- `enhancement_plan.md` - Geplante Features
- `final_completion.md` - Vollständige Feature-Liste

---

## ✨ Viel Erfolg beim Trading!

**⚠️ WICHTIG:** Standardmäßig läuft der Bot im **Paper Trading** Modus. Kein echtes Geld wird verwendet.

Um auf Live Trading umzuschalten, ändern Sie in `.env`:
```
ALPACA_PAPER_TRADING=false
```

**Nur nach ausgiebigem Testing empfohlen!**
