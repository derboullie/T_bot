import { io, Socket } from 'socket.io-client';

const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';

class WebSocketService {
    private socket: Socket | null = null;
    private listeners: Map<string, Set<Function>> = new Map();

    connect() {
        if (this.socket?.connected) {
            return;
        }

        this.socket = io(WS_URL, {
            path: '/ws',
            transports: ['websocket'],
        });

        this.socket.on('connect', () => {
            console.log('WebSocket connected');
        });

        this.socket.on('disconnect', () => {
            console.log('WebSocket disconnected');
        });

        this.socket.on('message', (data) => {
            this.emit('message', data);
        });
    }

    disconnect() {
        if (this.socket) {
            this.socket.disconnect();
            this.socket = null;
        }
    }

    on(event: string, callback: Function) {
        if (!this.listeners.has(event)) {
            this.listeners.set(event, new Set());
        }
        this.listeners.get(event)!.add(callback);
    }

    off(event: string, callback: Function) {
        const eventListeners = this.listeners.get(event);
        if (eventListeners) {
            eventListeners.delete(callback);
        }
    }

    private emit(event: string, data: any) {
        const eventListeners = this.listeners.get(event);
        if (eventListeners) {
            eventListeners.forEach(callback => callback(data));
        }
    }

    send(data: any) {
        if (this.socket?.connected) {
            this.socket.emit('message', data);
        }
    }
}

export const wsService = new WebSocketService();
export default wsService;
