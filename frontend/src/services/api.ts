import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

// System APIs
export const getStatus = () => api.get('/api/status');
export const getHealth = () => api.get('/api/health');

// Account APIs
export const getAccount = () => api.get('/api/account');
export const getPositions = () => api.get('/api/positions');
export const getPosition = (symbol: string) => api.get(`/api/positions/${symbol}`);

// Order APIs
export const getOrders = (status = 'all', limit = 100) =>
    api.get('/api/orders', { params: { status, limit } });

export const getOrder = (orderId: string) => api.get(`/api/orders/${orderId}`);

export const createMarketOrder = (symbol: string, quantity: number, side: 'buy' | 'sell') =>
    api.post('/api/orders/market', null, { params: { symbol, quantity, side } });

export const cancelOrder = (orderId: string) => api.delete(`/api/orders/${orderId}`);

// Market Data APIs
export const getStockPrice = (symbol: string) => api.get(`/api/market/price/${symbol}`);
export const getStockQuote = (symbol: string) => api.get(`/api/market/quote/${symbol}`);

export default api;
