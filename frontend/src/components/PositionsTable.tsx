// import { Box, Paper, Typography, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, IconButton, Chip } from '@mui/material';
import { useState, useEffect } from 'react';
import CloseIcon from '@mui/icons-material/Close';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';

interface Position {
    symbol: string;
    quantity: number;
    avg_price: number;
    current_price: number;
    pnl: number;
    pnl_pct: number;
}

function PositionsTable() {
    const [positions, setPositions] = useState<Position[]>([]);

    useEffect(() => {
        fetchPositions();
        const interval = setInterval(fetchPositions, 5000);
        return () => clearInterval(interval);
    }, []);

    const fetchPositions = async () => {
        try {
            const response = await fetch('http://localhost:8000/api/positions');
            const data = await response.json();

            // Transform positions data
            const transformedPositions = data.positions?.map((p: any) => ({
                symbol: p.symbol,
                quantity: p.qty || 0,
                avg_price: p.avg_entry_price || 0,
                current_price: p.current_price || 0,
                pnl: p.unrealized_pl || 0,
                pnl_pct: p.unrealized_plpc ? p.unrealized_plpc * 100 : 0,
            })) || [];

            setPositions(transformedPositions);
        } catch (error) {
            console.error('Failed to fetch positions:', error);
        }
    };

    const handleClosePosition = async (symbol: string) => {
        console.log(`Closing position for ${symbol}`);
        // TODO: Implement close position API call
    };

    return (
        <Box className="glass-card animate-fade-in" sx={{ mt: 3 }}>
            <Box sx={{ p: 3, borderBottom: 1, borderColor: 'rgba(255,255,255,0.1)' }}>
                <Typography variant="h6" className="gradient-text">
                    Open Positions
                </Typography>
            </Box>

            <TableContainer>
                <Table>
                    <TableHead>
                        <TableRow>
                            <TableCell sx={{ color: 'text.secondary', fontWeight: 600 }}>Symbol</TableCell>
                            <TableCell align="right" sx={{ color: 'text.secondary', fontWeight: 600 }}>Quantity</TableCell>
                            <TableCell align="right" sx={{ color: 'text.secondary', fontWeight: 600 }}>Avg Price</TableCell>
                            <TableCell align="right" sx={{ color: 'text.secondary', fontWeight: 600 }}>Current</TableCell>
                            <TableCell align="right" sx={{ color: 'text.secondary', fontWeight: 600 }}>P&L</TableCell>
                            <TableCell align="right" sx={{ color: 'text.secondary', fontWeight: 600 }}>P&L %</TableCell>
                            <TableCell align="right" sx={{ color: 'text.secondary', fontWeight: 600 }}>Action</TableCell>
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        {positions.length === 0 ? (
                            <TableRow>
                                <TableCell colSpan={7} align="center" sx={{ color: 'text.secondary', py: 8 }}>
                                    <Typography variant="body2">
                                        No open positions
                                    </Typography>
                                </TableCell>
                            </TableRow>
                        ) : (
                            positions.map((position) => {
                                const isProfit = position.pnl >= 0;
                                return (
                                    <TableRow
                                        key={position.symbol}
                                        className="hover-lift"
                                        sx={{
                                            '&:hover': {
                                                backgroundColor: 'rgba(255,255,255,0.02)',
                                            }
                                        }}
                                    >
                                        <TableCell sx={{ color: 'white', fontWeight: 600 }}>
                                            {position.symbol}
                                        </TableCell>
                                        <TableCell align="right" sx={{ color: 'white' }}>
                                            {position.quantity.toFixed(2)}
                                        </TableCell>
                                        <TableCell align="right" sx={{ color: 'text.secondary' }}>
                                            ${position.avg_price.toFixed(2)}
                                        </TableCell>
                                        <TableCell align="right" sx={{ color: 'white' }}>
                                            ${position.current_price.toFixed(2)}
                                        </TableCell>
                                        <TableCell align="right">
                                            <Box display="flex" alignItems="center" justifyContent="flex-end" gap={0.5}>
                                                {isProfit ? (
                                                    <TrendingUpIcon sx={{ fontSize: 16, color: 'var(--neon-green)' }} />
                                                ) : (
                                                    <TrendingDownIcon sx={{ fontSize: 16, color: 'var(--danger)' }} />
                                                )}
                                                <Typography
                                                    sx={{
                                                        color: isProfit ? 'var(--neon-green)' : 'var(--danger)',
                                                        fontWeight: 600,
                                                    }}
                                                >
                                                    {isProfit ? '+' : ''}{position.pnl.toFixed(2)}
                                                </Typography>
                                            </Box>
                                        </TableCell>
                                        <TableCell align="right">
                                            <Chip
                                                label={`${isProfit ? '+' : ''}${position.pnl_pct.toFixed(2)}%`}
                                                size="small"
                                                className={isProfit ? 'badge-success' : 'badge-danger'}
                                                sx={{ fontWeight: 600 }}
                                            />
                                        </TableCell>
                                        <TableCell align="right">
                                            <IconButton
                                                size="small"
                                                onClick={() => handleClosePosition(position.symbol)}
                                                sx={{
                                                    color: 'var(--danger)',
                                                    '&:hover': {
                                                        backgroundColor: 'rgba(255, 68, 68, 0.1)',
                                                    }
                                                }}
                                            >
                                                <CloseIcon fontSize="small" />
                                            </IconButton>
                                        </TableCell>
                                    </TableRow>
                                );
                            })
                        )}
                    </TableBody>
                </Table>
            </TableContainer>
        </Box>
    );
}

export default PositionsTable;
