#!/bin/bash

# HFT Trading Bot - Stopper Script

# Farben
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Stoppe HFT Trading Bot...${NC}"
echo ""

# Prozesse aus PIDs stoppen
if [ -f /tmp/hft_backend.pid ]; then
    BACKEND_PID=$(cat /tmp/hft_backend.pid)
    if kill -0 $BACKEND_PID 2>/dev/null; then
        kill $BACKEND_PID
        echo -e "${GREEN}✓ Backend gestoppt (PID: $BACKEND_PID)${NC}"
    else
        echo -e "${YELLOW}⚠ Backend Prozess läuft nicht mehr${NC}"
    fi
    rm /tmp/hft_backend.pid
fi

if [ -f /tmp/hft_frontend.pid ]; then
    FRONTEND_PID=$(cat /tmp/hft_frontend.pid)
    if kill -0 $FRONTEND_PID 2>/dev/null; then
        kill $FRONTEND_PID
        echo -e "${GREEN}✓ Frontend gestoppt (PID: $FRONTEND_PID)${NC}"
    else
        echo -e "${YELLOW}⚠ Frontend Prozess läuft nicht mehr${NC}"
    fi
    rm /tmp/hft_frontend.pid
fi

# Fallback: Alle uvicorn und vite Prozesse beenden
pkill -f uvicorn 2>/dev/null && echo -e "${GREEN}✓ Alle uvicorn Prozesse gestoppt${NC}"
pkill -f vite 2>/dev/null && echo -e "${GREEN}✓ Alle vite Prozesse gestoppt${NC}"

echo ""
echo -e "${GREEN}✅ HFT Trading Bot vollständig gestoppt${NC}"
