#!/bin/bash

# HFT Trading Bot Startup Script

echo "========================================"
echo "  HFT Trading Bot - Startup"
echo "========================================"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found!"
    echo "Please copy .env.example to .env and configure your API keys:"
    echo "  cp .env.example .env"
    echo ""
    exit 1
fi

# Check if poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "❌ Error: Poetry is not installed!"
    echo "Install it with: curl -sSL https://install.python-poetry.org | python3 -"
    echo ""
    exit 1
fi

# Function to check if a command succeeded
check_success() {
    if [ $? -ne 0 ]; then
        echo "❌ Error occurred. Exiting."
        exit 1
    fi
}

# Install Python dependencies if needed
if [ ! -d ".venv" ]; then
    echo "📦 Installing Python dependencies..."
    poetry install
    check_success
    echo "✅ Python dependencies installed"
    echo ""
fi

# Install frontend dependencies if needed
if [ ! -d "frontend/node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    cd frontend
    npm install
    check_success
    cd ..
    echo "✅ Frontend dependencies installed"
    echo ""
fi

# Start the services
echo "🚀 Starting HFT Trading Bot..."
echo ""
echo "Starting backend on http://localhost:8000"
echo "Starting frontend on http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop both services"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start backend in background
poetry run python -m backend.api.main &
BACKEND_PID=$!

# Give backend time to start
sleep 3

# Start frontend in background
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

# Wait for both processes
wait
