import {
    Box,
    Paper,
    Typography,
    TextField,
    Button,
    Select,
    MenuItem,
    FormControl,
    InputLabel,
    Grid,
    Chip,
    Alert,
    IconButton,
    Slider,
    Switch,
    FormControlLabel,
    Tabs,
    Tab
} from '@mui/material';
import { useState } from 'react';
import SendIcon from '@mui/icons-material/Send';
import CloseIcon from '@mui/icons-material/Close';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import StopIcon from '@mui/icons-material/Stop';
import SettingsIcon from '@mui/icons-material/Settings';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';

interface TabPanelProps {
    children?: React.ReactNode;
    index: number;
    value: number;
}

function TabPanel(props: TabPanelProps) {
    const { children, value, index, ...other } = props;
    return (
        <div hidden={value !== index} {...other}>
            {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
        </div>
    );
}

function TradingControls() {
    const [tabValue, setTabValue] = useState(0);
    const [botRunning, setBotRunning] = useState(false);
    const [orderForm, setOrderForm] = useState({
        symbol: 'AAPL',
        orderType: 'market',
        side: 'buy',
        quantity: 10,
        price: 0,
    });
    const [selectedStrategy, setSelectedStrategy] = useState('momentum');
    const [autoTrade, setAutoTrade] = useState(false);
    const [message, setMessage] = useState({ type: '', text: '' });

    const handlePlaceOrder = async () => {
        try {
            const response = await fetch('http://localhost:8000/api/orders/market', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    symbol: orderForm.symbol,
                    quantity: orderForm.quantity,
                    side: orderForm.side
                })
            });

            if (response.ok) {
                setMessage({ type: 'success', text: `Order placed: ${orderForm.side.toUpperCase()} ${orderForm.quantity} ${orderForm.symbol}` });
                setTimeout(() => setMessage({ type: '', text: '' }), 3000);
            } else {
                setMessage({ type: 'error', text: 'Failed to place order' });
            }
        } catch (error) {
            setMessage({ type: 'error', text: `Error: ${error}` });
        }
    };

    const handleStartBot = () => {
        setBotRunning(true);
        setMessage({ type: 'success', text: '🤖 Bot started with ' + selectedStrategy + ' strategy' });
        setTimeout(() => setMessage({ type: '', text: '' }), 3000);
    };

    const handleStopBot = () => {
        setBotRunning(false);
        setMessage({ type: 'warning', text: '⏸️ Bot stopped' });
        setTimeout(() => setMessage({ type: '', text: '' }), 3000);
    };

    return (
        <Box className="glass-card animate-fade-in" sx={{ mt: 3 }}>
            <Box sx={{ borderBottom: 1, borderColor: 'rgba(255,255,255,0.1)' }}>
                <Tabs
                    value={tabValue}
                    onChange={(_, newValue) => setTabValue(newValue)}
                    sx={{
                        '& .MuiTab-root': {
                            color: 'text.secondary',
                            '&.Mui-selected': {
                                color: 'var(--neon-green)',
                            }
                        },
                        '& .MuiTabs-indicator': {
                            backgroundColor: 'var(--neon-green)',
                            height: 3,
                            borderRadius: '3px 3px 0 0'
                        }
                    }}
                >
                    <Tab label="Manual Trading" />
                    <Tab label="Bot Control" />
                    <Tab label="Strategy Settings" />
                </Tabs>
            </Box>

            {message.text && (
                <Alert
                    severity={message.type as any}
                    className="animate-slide-in-right"
                    sx={{ m: 2 }}
                    onClose={() => setMessage({ type: '', text: '' })}
                >
                    {message.text}
                </Alert>
            )}

            {/* Manual Trading Tab */}
            <TabPanel value={tabValue} index={0}>
                <Grid container spacing={3}>
                    <Grid item xs={12} md={6}>
                        <Box className="glass-card" sx={{ p: 3 }}>
                            <Typography variant="h6" className="gradient-text" mb={3}>
                                Quick Order
                            </Typography>

                            <Grid container spacing={2}>
                                <Grid item xs={12}>
                                    <TextField
                                        fullWidth
                                        label="Symbol"
                                        value={orderForm.symbol}
                                        onChange={(e) => setOrderForm({ ...orderForm, symbol: e.target.value.toUpperCase() })}
                                        placeholder="AAPL"
                                        sx={{
                                            '& .MuiOutlinedInput-root': {
                                                color: 'white',
                                                '& fieldset': {
                                                    borderColor: 'rgba(255,255,255,0.2)',
                                                },
                                                '&:hover fieldset': {
                                                    borderColor: 'var(--neon-green)',
                                                }
                                            }
                                        }}
                                    />
                                </Grid>

                                <Grid item xs={6}>
                                    <FormControl fullWidth>
                                        <InputLabel sx={{ color: 'text.secondary' }}>Order Type</InputLabel>
                                        <Select
                                            value={orderForm.orderType}
                                            onChange={(e) => setOrderForm({ ...orderForm, orderType: e.target.value })}
                                            sx={{ color: 'white' }}
                                        >
                                            <MenuItem value="market">Market</MenuItem>
                                            <MenuItem value="limit">Limit</MenuItem>
                                            <MenuItem value="stop">Stop</MenuItem>
                                        </Select>
                                    </FormControl>
                                </Grid>

                                <Grid item xs={6}>
                                    <FormControl fullWidth>
                                        <InputLabel sx={{ color: 'text.secondary' }}>Side</InputLabel>
                                        <Select
                                            value={orderForm.side}
                                            onChange={(e) => setOrderForm({ ...orderForm, side: e.target.value })}
                                            sx={{ color: 'white' }}
                                        >
                                            <MenuItem value="buy">Buy</MenuItem>
                                            <MenuItem value="sell">Sell</MenuItem>
                                        </Select>
                                    </FormControl>
                                </Grid>

                                <Grid item xs={12}>
                                    <Typography gutterBottom sx={{ color: 'text.secondary' }}>
                                        Quantity: {orderForm.quantity} shares
                                    </Typography>
                                    <Slider
                                        value={orderForm.quantity}
                                        onChange={(_, value) => setOrderForm({ ...orderForm, quantity: value as number })}
                                        min={1}
                                        max={100}
                                        step={1}
                                        sx={{
                                            color: 'var(--neon-green)',
                                            '& .MuiSlider-thumb': {
                                                boxShadow: '0 0 10px rgba(0, 255, 136, 0.5)',
                                            }
                                        }}
                                    />
                                </Grid>

                                {orderForm.orderType === 'limit' && (
                                    <Grid item xs={12}>
                                        <TextField
                                            fullWidth
                                            label="Limit Price"
                                            type="number"
                                            value={orderForm.price}
                                            onChange={(e) => setOrderForm({ ...orderForm, price: parseFloat(e.target.value) })}
                                            sx={{
                                                '& .MuiOutlinedInput-root': {
                                                    color: 'white'
                                                }
                                            }}
                                        />
                                    </Grid>
                                )}

                                <Grid item xs={6}>
                                    <Button
                                        fullWidth
                                        variant="contained"
                                        className="cyber-button"
                                        onClick={handlePlaceOrder}
                                        disabled={orderForm.side !== 'buy'}
                                        startIcon={<TrendingUpIcon />}
                                        sx={{
                                            background: orderForm.side === 'buy'
                                                ? 'linear-gradient(135deg, var(--neon-green), var(--neon-blue))'
                                                : 'rgba(255,255,255,0.1)',
                                            '&:hover': {
                                                background: 'linear-gradient(135deg, var(--neon-blue), var(--neon-green))'
                                            }
                                        }}
                                    >
                                        Buy
                                    </Button>
                                </Grid>

                                <Grid item xs={6}>
                                    <Button
                                        fullWidth
                                        variant="contained"
                                        className="cyber-button"
                                        onClick={handlePlaceOrder}
                                        disabled={orderForm.side !== 'sell'}
                                        startIcon={<TrendingDownIcon />}
                                        sx={{
                                            background: orderForm.side === 'sell'
                                                ? 'linear-gradient(135deg, var(--danger), #ff6b6b)'
                                                : 'rgba(255,255,255,0.1)',
                                            '&:hover': {
                                                background: 'linear-gradient(135deg, #ff6b6b, var(--danger))'
                                            }
                                        }}
                                    >
                                        Sell
                                    </Button>
                                </Grid>
                            </Grid>
                        </Box>
                    </Grid>

                    <Grid item xs={12} md={6}>
                        <Box className="glass-card" sx={{ p: 3 }}>
                            <Typography variant="h6" className="gradient-text" mb={3}>
                                Quick Actions
                            </Typography>

                            <Grid container spacing={2}>
                                <Grid item xs={12}>
                                    <Button
                                        fullWidth
                                        className="cyber-button"
                                        startIcon={<CloseIcon />}
                                        sx={{
                                            background: 'linear-gradient(135deg, var(--danger), #ff6b6b)',
                                            '&:hover': {
                                                background: 'linear-gradient(135deg, #ff6b6b, var(--danger))',
                                                transform: 'scale(1.02)'
                                            }
                                        }}
                                    >
                                        Close All Positions
                                    </Button>
                                </Grid>

                                <Grid item xs={6}>
                                    <Button
                                        fullWidth
                                        variant="outlined"
                                        sx={{
                                            borderColor: 'rgba(0, 255, 136, 0.3)',
                                            color: 'var(--neon-green)',
                                            '&:hover': {
                                                borderColor: 'var(--neon-green)',
                                                backgroundColor: 'rgba(0, 255, 136, 0.1)'
                                            }
                                        }}
                                    >
                                        Cancel Orders
                                    </Button>
                                </Grid>

                                <Grid item xs={6}>
                                    <Button
                                        fullWidth
                                        variant="outlined"
                                        sx={{
                                            borderColor: 'rgba(0, 212, 255, 0.3)',
                                            color: 'var(--neon-blue)',
                                            '&:hover': {
                                                borderColor: 'var(--neon-blue)',
                                                backgroundColor: 'rgba(0, 212, 255, 0.1)'
                                            }
                                        }}
                                    >
                                        Refresh
                                    </Button>
                                </Grid>
                            </Grid>

                            <Box mt={3}>
                                <Typography variant="subtitle2" sx={{ color: 'text.secondary', mb: 2 }}>
                                    Risk Limits
                                </Typography>
                                <Box display="flex" justifyContent="space-between" mb={1}>
                                    <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                                        Max Position Size
                                    </Typography>
                                    <Typography variant="body2" sx={{ color: 'var(--neon-green)' }}>
                                        $10,000
                                    </Typography>
                                </Box>
                                <Box display="flex" justifyContent="space-between" mb={1}>
                                    <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                                        Daily Loss Limit
                                    </Typography>
                                    <Typography variant="body2" sx={{ color: 'var(--danger)' }}>
                                        -$1,000
                                    </Typography>
                                </Box>
                                <Box display="flex" justifyContent="space-between">
                                    <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                                        Used Today
                                    </Typography>
                                    <Typography variant="body2" sx={{ color: 'var(--neon-green)' }}>
                                        $0 / $1,000
                                    </Typography>
                                </Box>
                            </Box>
                        </Box>
                    </Grid>
                </Grid>
            </TabPanel>

            {/* Bot Control Tab */}
            <TabPanel value={tabValue} index={1}>
                <Grid container spacing={3}>
                    <Grid item xs={12}>
                        <Box className="glass-card" sx={{ p: 4, textAlign: 'center' }}>
                            <Typography variant="h5" className="gradient-text" mb={3}>
                                Automated Trading Bot
                            </Typography>

                            <Box display="flex" justifyContent="center" alignItems="center" gap={2} mb={4}>
                                <Chip
                                    label={botRunning ? "RUNNING" : "STOPPED"}
                                    className={`badge animate-pulse ${botRunning ? 'badge-success' : 'badge-danger'}`}
                                    sx={{ fontSize: '14px', px: 3, py: 2 }}
                                />
                            </Box>

                            <Box display="flex" justifyContent="center" gap={3} mb={4}>
                                <Button
                                    variant="contained"
                                    size="large"
                                    className="cyber-button"
                                    startIcon={<PlayArrowIcon />}
                                    onClick={handleStartBot}
                                    disabled={botRunning}
                                    sx={{
                                        px: 5,
                                        py: 2,
                                        fontSize: '16px',
                                        background: botRunning
                                            ? 'rgba(255,255,255,0.1)'
                                            : 'linear-gradient(135deg, var(--neon-green), var(--neon-blue))'
                                    }}
                                >
                                    Start Bot
                                </Button>

                                <Button
                                    variant="contained"
                                    size="large"
                                    className="cyber-button"
                                    startIcon={<StopIcon />}
                                    onClick={handleStopBot}
                                    disabled={!botRunning}
                                    sx={{
                                        px: 5,
                                        py: 2,
                                        fontSize: '16px',
                                        background: !botRunning
                                            ? 'rgba(255,255,255,0.1)'
                                            : 'linear-gradient(135deg, var(--danger), #ff6b6b)'
                                    }}
                                >
                                    Stop Bot
                                </Button>
                            </Box>

                            <FormControlLabel
                                control={
                                    <Switch
                                        checked={autoTrade}
                                        onChange={(e) => setAutoTrade(e.target.checked)}
                                        sx={{
                                            '& .MuiSwitch-switchBase.Mui-checked': {
                                                color: 'var(--neon-green)',
                                            },
                                            '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': {
                                                backgroundColor: 'var(--neon-green)',
                                            }
                                        }}
                                    />
                                }
                                label="Enable Auto-Trading"
                                sx={{ color: 'text.secondary' }}
                            />
                        </Box>
                    </Grid>

                    <Grid item xs={12} md={6}>
                        <Box className="glass-card" sx={{ p: 3 }}>
                            <Typography variant="h6" className="gradient-text" mb={2}>
                                Active Strategy
                            </Typography>
                            <FormControl fullWidth>
                                <Select
                                    value={selectedStrategy}
                                    onChange={(e) => setSelectedStrategy(e.target.value)}
                                    sx={{
                                        color: 'white',
                                        '& .MuiOutlinedInput-notchedOutline': {
                                            borderColor: 'rgba(0, 255, 136, 0.3)'
                                        }
                                    }}
                                >
                                    <MenuItem value="momentum">Momentum Strategy</MenuItem>
                                    <MenuItem value="arbitrage">Arbitrage</MenuItem>
                                    <MenuItem value="market_making">Market Making</MenuItem>
                                    <MenuItem value="stat_arb">Statistical Arbitrage</MenuItem>
                                    <MenuItem value="ml">ML (Self-Rewarding DQN)</MenuItem>
                                </Select>
                            </FormControl>
                        </Box>
                    </Grid>

                    <Grid item xs={12} md={6}>
                        <Box className="glass-card" sx={{ p: 3 }}>
                            <Typography variant="h6" className="gradient-text" mb={2}>
                                Performance
                            </Typography>
                            <Box>
                                <Box display="flex" justifyContent="space-between" mb={1}>
                                    <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                                        Trades Today
                                    </Typography>
                                    <Typography variant="body2" sx={{ color: 'var(--neon-green)' }}>
                                        0
                                    </Typography>
                                </Box>
                                <Box display="flex" justifyContent="space-between" mb={1}>
                                    <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                                        Win Rate
                                    </Typography>
                                    <Typography variant="body2" sx={{ color: 'var(--neon-green)' }}>
                                        N/A
                                    </Typography>
                                </Box>
                                <Box display="flex" justifyContent="space-between">
                                    <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                                        Signals Generated
                                    </Typography>
                                    <Typography variant="body2" sx={{ color: 'var(--neon-blue)' }}>
                                        0
                                    </Typography>
                                </Box>
                            </Box>
                        </Box>
                    </Grid>
                </Grid>
            </TabPanel>

            {/* Strategy Settings Tab */}
            <TabPanel value={tabValue} index={2}>
                <Box className="glass-card" sx={{ p: 3 }}>
                    <Typography variant="h6" className="gradient-text" mb={3}>
                        Strategy Configuration
                    </Typography>

                    <Grid container spacing={3}>
                        <Grid item xs={12} md={6}>
                            <Typography gutterBottom sx={{ color: 'text.secondary' }}>
                                RSI Period (Momentum)
                            </Typography>
                            <Slider
                                defaultValue={14}
                                min={5}
                                max={30}
                                marks
                                valueLabelDisplay="auto"
                                sx={{ color: 'var(--neon-green)' }}
                            />
                        </Grid>

                        <Grid item xs={12} md={6}>
                            <Typography gutterBottom sx={{ color: 'text.secondary' }}>
                                Moving Average Period
                            </Typography>
                            <Slider
                                defaultValue={20}
                                min={10}
                                max={50}
                                marks
                                valueLabelDisplay="auto"
                                sx={{ color: 'var(--neon-blue)' }}
                            />
                        </Grid>

                        <Grid item xs={12} md={6}>
                            <Typography gutterBottom sx={{ color: 'text.secondary' }}>
                                Min Profit % (Arbitrage)
                            </Typography>
                            <Slider
                                defaultValue={0.5}
                                min={0.1}
                                max={2.0}
                                step={0.1}
                                marks
                                valueLabelDisplay="auto"
                                sx={{ color: 'var(--neon-purple)' }}
                            />
                        </Grid>

                        <Grid item xs={12} md={6}>
                            <Typography gutterBottom sx={{ color: 'text.secondary' }}>
                                Spread BPS (Market Making)
                            </Typography>
                            <Slider
                                defaultValue={10}
                                min={5}
                                max={50}
                                marks
                                valueLabelDisplay="auto"
                                sx={{ color: 'var(--neon-pink)' }}
                            />
                        </Grid>

                        <Grid item xs={12}>
                            <Button
                                fullWidth
                                variant="contained"
                                className="cyber-button"
                                startIcon={<SettingsIcon />}
                                sx={{
                                    background: 'linear-gradient(135deg, var(--neon-purple), var(--neon-pink))',
                                    py: 1.5
                                }}
                            >
                                Save Configuration
                            </Button>
                        </Grid>
                    </Grid>
                </Box>
            </TabPanel>
        </Box>
    );
}

export default TradingControls;
