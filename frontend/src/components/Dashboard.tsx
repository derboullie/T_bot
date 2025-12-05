// import { Box, Container, Grid, Paper, Typography, Chip } from '@mui/material';
import { useEffect, useState } from 'react';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import AccountBalanceWalletIcon from '@mui/icons-material/AccountBalanceWallet';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import MemoryIcon from '@mui/icons-material/Memory';
import TradingControls from './TradingControls';
import PositionsTable from './PositionsTable';

interface Stats {
    portfolio_value: number;
    daily_pnl: number;
    open_positions: number;
    cpu_usage: number;
    account_status: string;
    trading_mode: string;
}

function Dashboard() {
    const [stats, setStats] = useState<Stats>({
        portfolio_value: 0,
        daily_pnl: 0,
        open_positions: 0,
        cpu_usage: 0,
        account_status: 'Loading...',
        trading_mode: 'PAPER',
    });

    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetchStats();
        const interval = setInterval(fetchStats, 5000);
        return () => clearInterval(interval);
    }, []);

    const fetchStats = async () => {
        try {
            const response = await fetch('http://localhost:8000/api/status');
            const data = await response.json();

            setStats({
                portfolio_value: data.account?.portfolio_value || 100000,
                daily_pnl: ((data.account?.portfolio_value || 100000) - 100000),
                open_positions: 0,
                cpu_usage: data.resources?.current_usage || 0,
                account_status: data.account?.status || 'ACTIVE',
                trading_mode: data.bot?.mode || 'paper',
            });
            setLoading(false);
        } catch (error) {
            console.error('Failed to fetch stats:', error);
        }
    };

    const StatCard = ({
        title,
        value,
        icon,
        color = 'success',
        delay = 0,
        suffix = ''
    }: any) => (
        <Box
            className="glass-card stat-card animate-fade-in hover-lift"
            sx={{
                p: 3,
                animationDelay: `${delay}s`
            }}
        >
            <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
                <Typography className="stat-label" sx={{ color: 'text.secondary' }}>
                    {title}
                </Typography>
                <Box sx={{
                    color: color === 'success' ? 'var(--neon-green)' : 'var(--danger)',
                    opacity: 0.7
                }}>
                    {icon}
                </Box>
            </Box>
            <Typography className="stat-number gradient-text" variant="h3">
                {loading ? (
                    <Box className="shimmer" sx={{ height: 48, borderRadius: 1 }} />
                ) : (
                    `${value}${suffix}`
                )}
            </Typography>
        </Box>
    );

    const PnlCard = () => {
        const isPositive = stats.daily_pnl >= 0;
        return (
            <Box
                className="glass-card stat-card animate-slide-in-right hover-lift"
                sx={{
                    p: 3,
                    borderColor: isPositive ? 'rgba(0, 255, 136, 0.3)' : 'rgba(255, 68, 68, 0.3)',
                }}
            >
                <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
                    <Typography className="stat-label">
                        Daily P&L
                    </Typography>
                    {isPositive ? (
                        <TrendingUpIcon sx={{ color: 'var(--neon-green)' }} />
                    ) : (
                        <TrendingDownIcon sx={{ color: 'var(--danger)' }} />
                    )}
                </Box>
                <Typography
                    className="stat-number animate-pulse"
                    sx={{
                        color: isPositive ? 'var(--neon-green)' : 'var(--danger)',
                        textShadow: isPositive
                            ? '0 0 10px rgba(0, 255, 136, 0.5)'
                            : '0 0 10px rgba(255, 68, 68, 0.5)'
                    }}
                    variant="h3"
                >
                    {isPositive ? '+' : ''}{stats.daily_pnl.toFixed(2)}
                </Typography>
                <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                    {isPositive ? '▲' : '▼'} {Math.abs((stats.daily_pnl / stats.portfolio_value) * 100).toFixed(2)}%
                </Typography>
            </Box>
        );
    };

    return (
        <Box className="animated-background" sx={{ minHeight: '100vh', py: 4 }}>
            <Container maxWidth="xl">
                {/* Header */}
                <Box className="animate-fade-in" mb={4}>
                    <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
                        <Typography
                            variant="h3"
                            className="gradient-text"
                            sx={{ fontWeight: 700 }}
                        >
                            HFT Trading Bot
                        </Typography>
                        <Box display="flex" gap={2}>
                            <Chip
                                label={stats.trading_mode.toUpperCase()}
                                className={`badge animate-glow ${stats.trading_mode === 'paper' ? 'badge-warning' : 'badge-success'}`}
                                sx={{
                                    fontSize: '12px',
                                    fontWeight: 600,
                                    animation: 'glow 2s ease-in-out infinite'
                                }}
                            />
                            <Chip
                                label={stats.account_status}
                                className="badge badge-success"
                                sx={{ fontSize: '12px', fontWeight: 600 }}
                            />
                        </Box>
                    </Box>
                    <Box className="progress-bar">
                        <Box
                            className="progress-fill"
                            sx={{ width: `${Math.min(stats.cpu_usage, 100)}%` }}
                        />
                    </Box>
                    <Typography variant="caption" sx={{ color: 'text.secondary', mt: 1, display: 'block' }}>
                        CPU Usage: {stats.cpu_usage.toFixed(1)}% / 85%
                    </Typography>
                </Box>

                {/* Stats Grid */}
                <Grid container spacing={3} mb={4}>
                    <Grid item xs={12} md={6} lg={3}>
                        <StatCard
                            title="Portfolio Value"
                            value={`$${stats.portfolio_value.toLocaleString()}`}
                            icon={<AccountBalanceWalletIcon />}
                            color="success"
                            delay={0}
                        />
                    </Grid>

                    <Grid item xs={12} md={6} lg={3}>
                        <PnlCard />
                    </Grid>

                    <Grid item xs={12} md={6} lg={3}>
                        <StatCard
                            title="Open Positions"
                            value={stats.open_positions}
                            icon={<ShowChartIcon />}
                            delay={0.2}
                        />
                    </Grid>

                    <Grid item xs={12} md={6} lg={3}>
                        <StatCard
                            title="CPU Usage"
                            value={stats.cpu_usage.toFixed(1)}
                            suffix="%"
                            icon={<MemoryIcon />}
                            color={stats.cpu_usage > 75 ? 'danger' : 'success'}
                            delay={0.3}
                        />
                    </Grid>
                </Grid>

                {/* Trading Controls */}
                <TradingControls />

                {/* Positions Table */}
                <PositionsTable />

                {/* Trading Info */}
                <Grid container spacing={3} mt={1}>
                    <Grid item xs={12} md={6}>
                        <Box className="glass-card animate-slide-in-left" sx={{ p: 3 }}>
                            <Typography variant="h6" className="gradient-text" mb={2}>
                                System Status
                            </Typography>
                            <Box display="flex" flexDirection="column" gap={2}>
                                <Box display="flex" justifyContent="space-between">
                                    <Typography sx={{ color: 'text.secondary' }}>Backend</Typography>
                                    <Chip label="RUNNING" size="small" className="badge badge-success" />
                                </Box>
                                <Box display="flex" justifyContent="space-between">
                                    <Typography sx={{ color: 'text.secondary' }}>Database</Typography>
                                    <Chip label="CONNECTED" size="small" className="badge badge-success" />
                                </Box>
                                <Box display="flex" justifyContent="space-between">
                                    <Typography sx={{ color: 'text.secondary' }}>Risk Manager</Typography>
                                    <Chip label="ACTIVE" size="small" className="badge badge-success" />
                                </Box>
                                <Box display="flex" justifyContent="space-between">
                                    <Typography sx={{ color: 'text.secondary' }}>ML Engine</Typography>
                                    <Chip label="READY" size="small" className="badge badge-success" />
                                </Box>
                            </Box>
                        </Box>
                    </Grid>

                    <Grid item xs={12} md={6}>
                        <Box className="glass-card animate-slide-in-right" sx={{ p: 3 }}>
                            <Typography variant="h6" className="gradient-text" mb={2}>
                                Quick Actions
                            </Typography>
                            <Box display="flex" flexDirection="column" gap={2}>
                                <button className="cyber-button">
                                    Start Trading
                                </button>
                                <button className="cyber-button" style={{
                                    background: 'linear-gradient(135deg, var(--neon-blue), var(--neon-purple))'
                                }}>
                                    Run Backtest
                                </button>
                                <button className="cyber-button" style={{
                                    background: 'linear-gradient(135deg, var(--neon-purple), var(--neon-pink))'
                                }}>
                                    Train ML Model
                                </button>
                            </Box>
                        </Box>
                    </Grid>
                </Grid>

                {/* Footer */}
                <Box mt={6} textAlign="center" className="animate-fade-in" sx={{ animationDelay: '0.6s' }}>
                    <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                        HFT Trading Bot v0.1.0 · Real-time Data · Advanced ML · Multi-Strategy
                    </Typography>
                    <Typography variant="caption" sx={{ color: 'text.muted', display: 'block', mt: 1 }}>
                        ⚠️ Paper Trading Mode - No Real Money
                    </Typography>
                </Box>
            </Container>
        </Box>
    );
}

export default Dashboard;
