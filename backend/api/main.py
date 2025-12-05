"""Main FastAPI application."""

import asyncio
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from sqlalchemy.orm import Session

# Initialize logging first
from backend.core.logging_config import setup_logging
setup_logging()

from backend.core.config import settings
from backend.core.cpu_monitor import cpu_monitor
from backend.data.database import database, get_db
from backend.data.polygon_client import polygon_client
from backend.trading.alpaca_client import alpaca_client
from backend.trading.risk_manager import risk_manager


# WebSocket connection manager
class ConnectionManager:
    """Manage WebSocket connections."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")


manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting HFT Trading Bot...")

    # Initialize database
    database.initialize()
    database.create_tables()

    # Start CPU monitoring
    asyncio.create_task(cpu_monitor.start_monitoring())

    logger.info("HFT Trading Bot started successfully")

    yield

    # Shutdown
    logger.info("Shutting down HFT Trading Bot...")
    cpu_monitor.stop_monitoring()
    await polygon_client.close()
    database.close()
    logger.info("HFT Trading Bot shut down")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="High-Frequency Trading Bot with ML and Backtesting",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url, "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include advanced routes
from backend.api.advanced_routes import router as advanced_router
app.include_router(advanced_router)


# Health check
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    from datetime import datetime
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# System status
@app.get("/api/status")
async def get_status():
    """Get bot status."""
    account = await alpaca_client.get_account()
    cpu_stats = cpu_monitor.get_stats()
    risk_stats = risk_manager.get_stats()

    return {
        "bot": {
            "version": settings.app_version,
            "mode": "paper" if alpaca_client.is_paper() else "live",
        },
        "account": account,
        "resources": cpu_stats,
        "risk": risk_stats,
    }


# Account endpoints
@app.get("/api/account")
async def get_account_info():
    """Get account information."""
    try:
        account = await alpaca_client.get_account()
        return account
    except Exception as e:
        logger.error(f"Error getting account: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/positions")
async def get_positions():
    """Get all positions."""
    try:
        positions = await alpaca_client.get_positions()
        return {"positions": positions}
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/positions/{symbol}")
async def get_position(symbol: str):
    """Get position for a symbol."""
    try:
        position = await alpaca_client.get_position(symbol)
        if not position:
            raise HTTPException(status_code=404, detail=f"No position for {symbol}")
        return position
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting position for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Order endpoints
@app.get("/api/orders")
async def get_orders(status: str = "all", limit: int = 100):
    """Get orders."""
    try:
        orders = await alpaca_client.get_orders(status=status, limit=limit)
        return {"orders": orders}
    except Exception as e:
        logger.error(f"Error getting orders: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/orders/{order_id}")
async def get_order(order_id: str):
    """Get specific order."""
    try:
        order = await alpaca_client.get_order(order_id)
        if not order:
            raise HTTPException(status_code=404, detail=f"Order {order_id} not found")
        return order
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting order {order_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/orders/market")
async def create_market_order(
    symbol: str,
    quantity: float,
    side: str,
    db: Session = Depends(get_db),
):
    """Create a market order."""
    try:
        # Check trading allowed
        can_trade, reason = risk_manager.check_can_trade(db)
        if not can_trade:
            raise HTTPException(status_code=403, detail=reason)

        # Get account info
        account = await alpaca_client.get_account()

        # Validate order size
        price_data = await polygon_client.get_stock_price(symbol)
        if not price_data:
            raise HTTPException(status_code=404, detail=f"Could not get price for {symbol}")

        price = price_data["price"]
        is_valid, reason = risk_manager.check_order_size(
            symbol, quantity, price, account["equity"]
        )
        if not is_valid:
            raise HTTPException(status_code=403, detail=reason)

        # Submit order
        order = await alpaca_client.submit_market_order(symbol, quantity, side)
        if not order:
            raise HTTPException(status_code=500, detail="Failed to submit order")

        logger.info(f"Market order created: {side} {quantity} {symbol}")
        return order

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating market order: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/orders/{order_id}")
async def cancel_order(order_id: str):
    """Cancel an order."""
    try:
        success = await alpaca_client.cancel_order(order_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to cancel order")
        return {"message": f"Order {order_id} canceled"}
    except Exception as e:
        logger.error(f"Error canceling order {order_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Market data endpoints
@app.get("/api/market/price/{symbol}")
async def get_stock_price(symbol: str):
    """Get current stock price."""
    try:
        price_data = await polygon_client.get_stock_price(symbol)
        if not price_data:
            raise HTTPException(status_code=404, detail=f"Could not get price for {symbol}")
        return price_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting price for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/market/quote/{symbol}")
async def get_stock_quote(symbol: str):
    """Get current stock quote."""
    try:
        quote_data = await polygon_client.get_stock_quote(symbol)
        if not quote_data:
            raise HTTPException(status_code=404, detail=f"Could not get quote for {symbol}")
        return quote_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting quote for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            logger.debug(f"Received WebSocket message: {data}")

            # Echo back (you can implement specific handlers)
            await websocket.send_json({"type": "ack", "message": "received"})

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
