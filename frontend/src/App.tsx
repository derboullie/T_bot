import React, { useEffect, useState } from 'react';
import {
    AppBar,
    Box,
    Container,
    CssBaseline,
    Grid,
    Paper,
    ThemeProvider,
    Toolbar,
    Typography,
    createTheme,
    Card,
    CardContent,
    LinearProgress,
} from '@mui/material';
import {
    TrendingUp as TrendingUpIcon,
    AccountBalance as AccountBalanceIcon,
    ShowChart as ShowChartIcon,
    Speed as SpeedIcon,
} from '@mui/icons-material';
import { getStatus, getAccount, getPositions } from './services/api';
import wsService from './services/websocket';

// Create dark theme
const darkTheme = createTheme({
    palette: {
        mode: 'dark',
        primary: {
            main: '#00ff88',
        },
        secondary: {
            main: '#ff00ff',
        },
        background: {
            default: '#0a0e27',
            paper: '#14192d',
        },
    },
    typography: {
        fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    },
});

interface StatCardProps {
    title: string;
    value: string | number;
    icon: React.ReactNode;
    color?: string;
}

const StatCard: React.FC<StatCardProps> = ({ title, value, icon, color = '#00ff88' }) => {
    return (
        <Card sx={{ height: '100%', background: 'linear-gradient(135deg, #14192d 0%, #1a2038 100%)' }}>
            <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <Box sx={{ color, mr: 1 }}>{icon}</Box>
                    <Typography variant="h6" color="text.secondary">
                        {title}
                    </Typography>
                </Box>
                <Typography variant="h4" sx={{ fontWeight: 'bold', color }}>
                    {value}
                </Typography>
            </CardContent>
        </Card>
    );
};

function App() {
    const [status, setStatus] = useState<any>(null);
    const [account, setAccount] = useState<any>(null);
    const [positions, setPositions] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        // Connect to WebSocket
        wsService.connect();

        // Fetch initial data
        const fetchData = async () => {
            try {
                const [statusRes, accountRes, positionsRes] = await Promise.all([
                    getStatus(),
                    getAccount(),
                    getPositions(),
                ]);

                setStatus(statusRes.data);
                setAccount(accountRes.data);
                setPositions(positionsRes.data.positions || []);
            } catch (error) {
                console.error('Error fetching data:', error);
            } finally {
                setLoading(false);
            }
        };

        fetchData();

        // Refresh data periodically
        const interval = setInterval(fetchData, 5000);

        return () => {
            clearInterval(interval);
            wsService.disconnect();
        };
    }, []);

    if (loading) {
        return (
            <ThemeProvider theme={darkTheme}>
                <Box sx={{ width: '100%', mt: 20 }}>
                    <Typography variant="h4" align="center" gutterBottom>
                        Loading HFT Trading Bot...
                    </Typography>
                    <LinearProgress />
                </Box>
            </ThemeProvider>
        );
    }

    const cpuUsage = status?.resources?.current_usage || 0;
    const dailyPnL = status?.risk?.daily_pnl || 0;
    const portfolioValue = account?.portfolio_value || 0;

    return (
        <ThemeProvider theme={darkTheme}>
            <CssBaseline />
            <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
                {/* Header */}
                <AppBar position="static" sx={{ background: 'linear-gradient(90deg, #14192d 0%, #1a2038 100%)' }}>
                    <Toolbar>
                        <TrendingUpIcon sx={{ mr: 2, fontSize: 32, color: '#00ff88' }} />
                        <Typography variant="h5" component="div" sx={{ flexGrow: 1, fontWeight: 'bold' }}>
                            HFT Trading Bot
                        </Typography>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                            <Box
                                sx={{
                                    px: 2,
                                    py: 0.5,
                                    borderRadius: 1,
                                    background: status?.bot?.mode === 'paper' ? '#ff9800' : '#4caf50',
                                }}
                            >
                                <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                                    {status?.bot?.mode?.toUpperCase() || 'OFFLINE'}
                                </Typography>
                            </Box>
                        </Box>
                    </Toolbar>
                </AppBar>

                {/* Main Content */}
                <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
                    {/* Overview Stats */}
                    <Grid container spacing={3} sx={{ mb: 4 }}>
                        <Grid item xs={12} sm={6} md={3}>
                            <StatCard
                                title="Portfolio Value"
                                value={`$${portfolioValue.toLocaleString()}`}
                                icon={<AccountBalanceIcon />}
                                color="#00ff88"
                            />
                        </Grid>
                        <Grid item xs={12} sm={6} md={3}>
                            <StatCard
                                title="Daily P&L"
                                value={`$${dailyPnL.toFixed(2)}`}
                                icon={<ShowChartIcon />}
                                color={dailyPnL >= 0 ? '#00ff88' : '#ff4444'}
                            />
                        </Grid>
                        <Grid item xs={12} sm={6} md={3}>
                            <StatCard
                                title="Open Positions"
                                value={positions.length}
                                icon={<TrendingUpIcon />}
                                color="#00bfff"
                            />
                        </Grid>
                        <Grid item xs={12} sm={6} md={3}>
                            <StatCard
                                title="CPU Usage"
                                value={`${cpuUsage.toFixed(1)}%`}
                                icon={<SpeedIcon />}
                                color={cpuUsage > 85 ? '#ff4444' : '#00ff88'}
                            />
                        </Grid>
                    </Grid>

                    {/* Positions Table */}
                    <Paper sx={{ p: 3, mb: 4 }}>
                        <Typography variant="h6" gutterBottom>
                            Open Positions
                        </Typography>
                        {positions.length === 0 ? (
                            <Typography variant="body2" color="text.secondary" sx={{ py: 4, textAlign: 'center' }}>
                                No open positions
                            </Typography>
                        ) : (
                            <Box sx={{ overflowX: 'auto' }}>
                                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                                    <thead>
                                        <tr style={{ borderBottom: '1px solid #333' }}>
                                            <th style={{ padding: '12px', textAlign: 'left' }}>Symbol</th>
                                            <th style={{ padding: '12px', textAlign: 'right' }}>Quantity</th>
                                            <th style={{ padding: '12px', textAlign: 'right' }}>Avg Price</th>
                                            <th style={{ padding: '12px', textAlign: 'right' }}>Current Price</th>
                                            <th style={{ padding: '12px', textAlign: 'right' }}>P&L</th>
                                            <th style={{ padding: '12px', textAlign: 'right' }}>P&L %</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {positions.map((position, index) => (
                                            <tr key={index} style={{ borderBottom: '1px solid #222' }}>
                                                <td style={{ padding: '12px', fontWeight: 'bold' }}>{position.symbol}</td>
                                                <td style={{ padding: '12px', textAlign: 'right' }}>{position.quantity}</td>
                                                <td style={{ padding: '12px', textAlign: 'right' }}>
                                                    ${position.average_price?.toFixed(2)}
                                                </td>
                                                <td style={{ padding: '12px', textAlign: 'right' }}>
                                                    ${position.current_price?.toFixed(2)}
                                                </td>
                                                <td
                                                    style={{
                                                        padding: '12px',
                                                        textAlign: 'right',
                                                        color: position.unrealized_pl >= 0 ? '#00ff88' : '#ff4444',
                                                    }}
                                                >
                                                    ${position.unrealized_pl?.toFixed(2)}
                                                </td>
                                                <td
                                                    style={{
                                                        padding: '12px',
                                                        textAlign: 'right',
                                                        color: position.unrealized_pl_percent >= 0 ? '#00ff88' : '#ff4444',
                                                    }}
                                                >
                                                    {(position.unrealized_pl_percent * 100)?.toFixed(2)}%
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </Box>
                        )}
                    </Paper>

                    {/* Account Info */}
                    <Grid container spacing={3}>
                        <Grid item xs={12} md={6}>
                            <Paper sx={{ p: 3 }}>
                                <Typography variant="h6" gutterBottom>
                                    Account Information
                                </Typography>
                                <Box sx={{ mt: 2 }}>
                                    <Grid container spacing={2}>
                                        <Grid item xs={6}>
                                            <Typography variant="body2" color="text.secondary">
                                                Cash
                                            </Typography>
                                            <Typography variant="h6">${account?.cash?.toLocaleString()}</Typography>
                                        </Grid>
                                        <Grid item xs={6}>
                                            <Typography variant="body2" color="text.secondary">
                                                Equity
                                            </Typography>
                                            <Typography variant="h6">${account?.equity?.toLocaleString()}</Typography>
                                        </Grid>
                                        <Grid item xs={6}>
                                            <Typography variant="body2" color="text.secondary">
                                                Buying Power
                                            </Typography>
                                            <Typography variant="h6">
                                                ${account?.buying_power?.toLocaleString()}
                                            </Typography>
                                        </Grid>
                                        <Grid item xs={6}>
                                            <Typography variant="body2" color="text.secondary">
                                                Day Trades
                                            </Typography>
                                            <Typography variant="h6">{account?.daytrade_count || 0}</Typography>
                                        </Grid>
                                    </Grid>
                                </Box>
                            </Paper>
                        </Grid>

                        <Grid item xs={12} md={6}>
                            <Paper sx={{ p: 3 }}>
                                <Typography variant="h6" gutterBottom>
                                    Risk Management
                                </Typography>
                                <Box sx={{ mt: 2 }}>
                                    <Grid container spacing={2}>
                                        <Grid item xs={6}>
                                            <Typography variant="body2" color="text.secondary">
                                                Daily P&L
                                            </Typography>
                                            <Typography
                                                variant="h6"
                                                sx={{ color: dailyPnL >= 0 ? '#00ff88' : '#ff4444' }}
                                            >
                                                ${dailyPnL.toFixed(2)}
                                            </Typography>
                                        </Grid>
                                        <Grid item xs={6}>
                                            <Typography variant="body2" color="text.secondary">
                                                Max Daily Loss
                                            </Typography>
                                            <Typography variant="h6">
                                                ${status?.risk?.max_daily_loss || 0}
                                            </Typography>
                                        </Grid>
                                        <Grid item xs={6}>
                                            <Typography variant="body2" color="text.secondary">
                                                CPU Usage
                                            </Typography>
                                            <Typography
                                                variant="h6"
                                                sx={{ color: cpuUsage > 85 ? '#ff4444' : '#00ff88' }}
                                            >
                                                {cpuUsage.toFixed(1)}%
                                            </Typography>
                                        </Grid>
                                        <Grid item xs={6}>
                                            <Typography variant="body2" color="text.secondary">
                                                CPU Limit
                                            </Typography>
                                            <Typography variant="h6">
                                                {status?.resources?.limit_percent || 85}%
                                            </Typography>
                                        </Grid>
                                    </Grid>
                                </Box>
                            </Paper>
                        </Grid>
                    </Grid>
                </Container>
            </Box>
        </ThemeProvider>
    );
}

export default App;
