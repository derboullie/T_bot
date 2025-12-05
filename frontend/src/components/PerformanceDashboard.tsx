import React, { useState, useEffect } from 'react';
import {
    Paper,
    Grid,
    Typography,
    Box,
    Card,
    CardContent,
} from '@mui/material';
import {
    LineChart,
    Line,
    AreaChart,
    Area,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    Legend,
} from 'recharts';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import ShowChartIcon from '@mui/icons-material/ShowChart';

interface PerformanceMetrics {
    totalReturn: number;
    sharpeRatio: number;
    maxDrawdown: number;
    winRate: number;
    totalTrades: number;
    avgWin: number;
    avgLoss: number;
}

const PerformanceDashboard: React.FC = () => {
    const [metrics, setMetrics] = useState<PerformanceMetrics>({
        totalReturn: 15.7,
        sharpeRatio: 2.3,
        maxDrawdown: -8.2,
        winRate: 62.5,
        totalTrades: 127,
        avgWin: 245.50,
        avgLoss: -123.75,
    });

    const [equityData, setEquityData] = useState<any[]>([]);
    const [drawdownData, setDrawdownData] = useState<any[]>([]);

    useEffect(() => {
        // Generate sample equity curve
        const equity = [];
        const drawdown = [];
        let value = 100000;
        let peak = value;

        for (let i = 0; i < 100; i++) {
            const change = (Math.random() - 0.45) * 1000;
            value += change;
            peak = Math.max(peak, value);
            const dd = ((value - peak) / peak) * 100;

            equity.push({
                date: `Day ${i + 1}`,
                value: value,
                benchmark: 100000 + i * 150,
            });

            drawdown.push({
                date: `Day ${i + 1}`,
                drawdown: dd,
            });
        }

        setEquityData(equity);
        setDrawdownData(drawdown);
    }, []);

    const MetricCard = ({ title, value, suffix = '', icon, positive = true }: any) => (
        <Card
            className="cyber-panel"
            sx={{
                background: 'linear-gradient(135deg, rgba(10, 14, 26, 0.9) 0%, rgba(20, 30, 50, 0.9) 100%)',
                backdropFilter: 'blur(10px)',
                border: `1px solid ${positive ? 'rgba(0, 255, 136, 0.3)' : 'rgba(255, 0, 85, 0.3)'}`,
                transition: 'all 0.3s ease',
                '&:hover': {
                    transform: 'translateY(-4px)',
                    boxShadow: `0 8px 24px ${positive ? 'rgba(0, 255, 136, 0.2)' : 'rgba(255, 0, 85, 0.2)'}`,
                },
            }}
        >
            <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Typography variant="subtitle2" sx={{ color: '#888', mb: 1 }}>
                        {title}
                    </Typography>
                    {icon}
                </Box>
                <Typography
                    variant="h4"
                    sx={{
                        color: positive ? '#00ff88' : '#ff0055',
                        fontWeight: 700,
                        textShadow: `0 0 20px ${positive ? 'rgba(0, 255, 136, 0.5)' : 'rgba(255, 0, 85, 0.5)'}`,
                    }}
                >
                    {value}
                    <Typography component="span" variant="h6" sx={{ ml: 0.5 }}>
                        {suffix}
                    </Typography>
                </Typography>
            </CardContent>
        </Card>
    );

    return (
        <Box>
            {/* Metrics Cards */}
            <Grid container spacing={2} sx={{ mb: 3 }}>
                <Grid item xs={12} sm={6} md={3}>
                    <MetricCard
                        title="Total Return"
                        value={metrics.totalReturn > 0 ? `+${metrics.totalReturn}` : metrics.totalReturn}
                        suffix="%"
                        icon={<TrendingUpIcon sx={{ color: '#00ff88' }} />}
                        positive={metrics.totalReturn > 0}
                    />
                </Grid>

                <Grid item xs={12} sm={6} md={3}>
                    <MetricCard
                        title="Sharpe Ratio"
                        value={metrics.sharpeRatio.toFixed(2)}
                        icon={<ShowChartIcon sx={{ color: '#00ff88' }} />}
                        positive={true}
                    />
                </Grid>

                <Grid item xs={12} sm={6} md={3}>
                    <MetricCard
                        title="Max Drawdown"
                        value={metrics.maxDrawdown}
                        suffix="%"
                        icon={<TrendingDownIcon sx={{ color: '#ff0055' }} />}
                        positive={false}
                    />
                </Grid>

                <Grid item xs={12} sm={6} md={3}>
                    <MetricCard
                        title="Win Rate"
                        value={metrics.winRate}
                        suffix="%"
                        icon={<TrendingUpIcon sx={{ color: '#00ff88' }} />}
                        positive={true}
                    />
                </Grid>
            </Grid>

            {/* Equity Curve */}
            <Paper
                className="cyber-panel"
                sx={{
                    p: 3,
                    mb: 3,
                    background: 'linear-gradient(135deg, rgba(10, 14, 26, 0.95) 0%, rgba(20, 30, 50, 0.95) 100%)',
                    backdropFilter: 'blur(10px)',
                    border: '1px solid rgba(0, 255, 136, 0.2)',
                }}
            >
                <Typography
                    variant="h6"
                    sx={{
                        color: '#00ff88',
                        mb: 2,
                        fontWeight: 600,
                        textShadow: '0 0 10px rgba(0, 255, 136, 0.5)',
                    }}
                >
                    Equity Curve
                </Typography>

                <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={equityData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                        <XAxis dataKey="date" stroke="#888" />
                        <YAxis stroke="#888" />
                        <Tooltip
                            contentStyle={{
                                background: 'rgba(10, 14, 26, 0.95)',
                                border: '1px solid rgba(0, 255, 136, 0.3)',
                                borderRadius: '8px',
                            }}
                        />
                        <Legend />
                        <Line
                            type="monotone"
                            dataKey="value"
                            stroke="#00ff88"
                            strokeWidth={2}
                            dot={false}
                            name="Portfolio Value"
                        />
                        <Line
                            type="monotone"
                            dataKey="benchmark"
                            stroke="#0088ff"
                            strokeWidth={2}
                            strokeDasharray="5 5"
                            dot={false}
                            name="Benchmark"
                        />
                    </LineChart>
                </ResponsiveContainer>
            </Paper>

            {/* Drawdown Chart */}
            <Paper
                className="cyber-panel"
                sx={{
                    p: 3,
                    background: 'linear-gradient(135deg, rgba(10, 14, 26, 0.95) 0%, rgba(20, 30, 50, 0.95) 100%)',
                    backdropFilter: 'blur(10px)',
                    border: '1px solid rgba(255, 0, 85, 0.2)',
                }}
            >
                <Typography
                    variant="h6"
                    sx={{
                        color: '#ff0055',
                        mb: 2,
                        fontWeight: 600,
                        textShadow: '0 0 10px rgba(255, 0, 85, 0.5)',
                    }}
                >
                    Drawdown
                </Typography>

                <ResponsiveContainer width="100%" height={200}>
                    <AreaChart data={drawdownData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                        <XAxis dataKey="date" stroke="#888" />
                        <YAxis stroke="#888" />
                        <Tooltip
                            contentStyle={{
                                background: 'rgba(10, 14, 26, 0.95)',
                                border: '1px solid rgba(255, 0, 85, 0.3)',
                                borderRadius: '8px',
                            }}
                        />
                        <Area
                            type="monotone"
                            dataKey="drawdown"
                            stroke="#ff0055"
                            fill="rgba(255, 0, 85, 0.2)"
                            name="Drawdown %"
                        />
                    </AreaChart>
                </ResponsiveContainer>
            </Paper>
        </Box>
    );
};

export default PerformanceDashboard;
