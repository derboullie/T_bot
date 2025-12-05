#!/bin/bash

# HFT Trading Bot - One-Click Starter
# Automatisches Update und Start des gesamten Systems

set -e  # Exit on error

# Farben für Output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       🚀 HFT Trading Bot - Auto Starter 🚀          ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════╝${NC}"
echo ""

# Schritt 1: Dependencies aktualisieren
echo -e "${YELLOW}[1/5]${NC} ${GREEN}Aktualisiere Python Dependencies...${NC}"
poetry install --no-interaction 2>&1 | grep -E "(Installing|Updating|Already|Using)" || true
echo -e "${GREEN}✓ Python Dependencies aktualisiert${NC}"
echo ""

# Schritt 2: Frontend Dependencies aktualisieren
echo -e "${YELLOW}[2/5]${NC} ${GREEN}Aktualisiere Frontend Dependencies...${NC}"
cd frontend
if [ ! -d "node_modules" ]; then
    echo "  Installiere Node Modules (erste Installation)..."
    npm install --silent 2>&1 | tail -5
else
    echo "  Node Modules bereits installiert"
fi
cd ..
echo -e "${GREEN}✓ Frontend Dependencies aktualisiert${NC}"
echo ""

# Schritt 3: Alte Prozesse beenden
echo -e "${YELLOW}[3/5]${NC} ${GREEN}Beende alte Prozesse...${NC}"
pkill -f uvicorn 2>/dev/null && echo "  Gestoppt: Backend" || echo "  Kein Backend läuft"
pkill -f vite 2>/dev/null && echo "  Gestoppt: Frontend" || echo "  Kein Frontend läuft"
sleep 1
echo -e "${GREEN}✓ Alte Prozesse beendet${NC}"
echo ""

# Schritt 4: Backend starten
echo -e "${YELLOW}[4/5]${NC} ${GREEN}Starte Backend...${NC}"
poetry run uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload > /tmp/hft_backend.log 2>&1 &
BACKEND_PID=$!
echo "  Backend PID: $BACKEND_PID"

# Warte auf Backend Start
echo -n "  Warte auf Backend"
for i in {1..15}; do
    if curl -s http://localhost:8000/api/health > /dev/null 2>&1; then
        echo -e "\n${GREEN}✓ Backend läuft auf http://localhost:8000${NC}"
        break
    fi
    echo -n "."
    sleep 1
    if [ $i -eq 15 ]; then
        echo -e "\n${RED}⚠ Backend Start dauert länger als erwartet${NC}"
        echo -e "${YELLOW}  Logs: tail -f /tmp/hft_backend.log${NC}"
    fi
done
echo ""

# Schritt 5: Frontend starten
echo -e "${YELLOW}[5/5]${NC} ${GREEN}Starte Frontend...${NC}"
cd frontend
npm run dev > /tmp/hft_frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..
echo "  Frontend PID: $FRONTEND_PID"

# Warte auf Frontend Start
echo -n "  Warte auf Frontend"
for i in {1..15}; do
    if curl -s http://localhost:5173 > /dev/null 2>&1; then
        echo -e "\n${GREEN}✓ Frontend läuft auf http://localhost:5173${NC}"
        break
    fi
    echo -n "."
    sleep 1
    if [ $i -eq 15 ]; then
        echo -e "\n${YELLOW}⚠ Frontend Start dauert länger als erwartet${NC}"
        echo -e "${YELLOW}  Logs: tail -f /tmp/hft_frontend.log${NC}"
    fi
done
echo ""

# Status anzeigen
echo -e "${BLUE}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║              ✅ System erfolgreich gestartet!         ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}📊 Dashboard:${NC}       http://localhost:5173"
echo -e "${GREEN}📚 API Docs:${NC}        http://localhost:8000/docs"
echo -e "${GREEN}💚 Health Check:${NC}    http://localhost:8000/api/health"
echo ""
echo -e "${YELLOW}📝 Logs:${NC}"
echo -e "   Backend:  tail -f /tmp/hft_backend.log"
echo -e "   Frontend: tail -f /tmp/hft_frontend.log"
echo ""
echo -e "${YELLOW}🛑 Stoppen:${NC}"
echo -e "   Alle:     ./stop.sh"
echo -e "   Manuell:  kill $BACKEND_PID $FRONTEND_PID"
echo ""

# PIDs speichern für stop.sh
echo "$BACKEND_PID" > /tmp/hft_backend.pid
echo "$FRONTEND_PID" > /tmp/hft_frontend.pid

# Browser automatisch öffnen (nach 2 Sekunden)
sleep 2
if command -v xdg-open > /dev/null; then
    echo -e "${GREEN}🌐 Öffne Browser...${NC}"
    xdg-open http://localhost:5173 2>/dev/null &
elif command -v open > /dev/null; then
    echo -e "${GREEN}🌐 Öffne Browser...${NC}"
    open http://localhost:5173 2>/dev/null &
else
    echo -e "${YELLOW}ℹ  Bitte öffnen Sie manuell: http://localhost:5173${NC}"
fi

echo ""
echo -e "${GREEN}✨ HFT Trading Bot läuft! Viel Erfolg beim Trading! ✨${NC}"
echo ""
echo -e "${BLUE}Drücken Sie Ctrl+C um dieses Script zu beenden (Dienste laufen weiter)${NC}"
echo ""

# Warten (optional)
wait
