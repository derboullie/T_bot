import React, { useEffect, useRef, useState } from 'react';
import { createChart, IChartApi, ISeriesApi, CandlestickData } from 'lightweight-charts';
// import { Box, Paper, Typography, ToggleButtonGroup, ToggleButton } from '@mui/material';

interface TradingChartProps {
    symbol: string;
    data?: CandlestickData[];
    height?: number;
}

const TradingChart: React.FC<TradingChartProps> = ({
    symbol,
    data = [],
    height = 400
}) => {
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const candlestickSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
    const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);

    const [timeframe, setTimeframe] = useState('1H');

    useEffect(() => {
        if (!chartContainerRef.current) return;

        // Create chart
        const chart = createChart(chartContainerRef.current, {
            width: chartContainerRef.current.clientWidth,
            height: height,
            layout: {
                background: { color: '#0a0e1a' },
                textColor: '#00ff88',
            },
            grid: {
                vertLines: { color: 'rgba(255, 255, 255,0.1)' },
                horzLines: { color: 'rgba(255, 255, 255, 0.1)' },
            },
            crosshair: {
                mode: 1,
            },
            rightPriceScale: {
                borderColor: 'rgba(255, 255, 255, 0.2)',
            },
            timeScale: {
                borderColor: 'rgba(255, 255, 255, 0.2)',
                timeVisible: true,
                secondsVisible: false,
            },
        });

        chartRef.current = chart;

        // Add candlestick series
        const candlestickSeries = chart.addCandlestickSeries({
            upColor: '#00ff88',
            downColor: '#ff0055',
            borderUpColor: '#00ff88',
            borderDownColor: '#ff0055',
            wickUpColor: '#00ff88',
            wickDownColor: '#ff0055',
        });

        candlestickSeriesRef.current = candlestickSeries;

        // Add volume series
        const volumeSeries = chart.addHistogramSeries({
            color: '#26a69a',
            priceFormat: {
                type: 'volume',
            },
            priceScaleId: '',
        });

        volumeSeriesRef.current = volumeSeries;

        // Handle resize
        const handleResize = () => {
            if (chartContainerRef.current) {
                chart.applyOptions({
                    width: chartContainerRef.current.clientWidth
                });
            }
        };

        window.addEventListener('resize', handleResize);

        return () => {
            window.removeEventListener('resize', handleResize);
            chart.remove();
        };
    }, [height]);

    useEffect(() => {
        if (candlestickSeriesRef.current && data.length > 0) {
            candlestickSeriesRef.current.setData(data);

            // Generate volume data from candlestick data
            const volumeData = data.map(d => ({
                time: d.time,
                value: (d.high - d.low) * 1000000, // Simulated volume
                color: d.close > d.open ? '#00ff8844' : '#ff005544',
            }));

            if (volumeSeriesRef.current) {
                volumeSeriesRef.current.setData(volumeData);
            }

            // Fit content
            chartRef.current?.timeScale().fitContent();
        }
    }, [data]);

    const handleTimeframeChange = (_event: React.MouseEvent<HTMLElement>, newTimeframe: string | null) => {
        if (newTimeframe !== null) {
            setTimeframe(newTimeframe);
            // TODO: Fetch data for new timeframe
        }
    };

    return (
        <Paper
            className="cyber-panel"
            sx={{
                p: 2,
                background: 'linear-gradient(135deg, rgba(10, 14, 26, 0.95) 0%, rgba(20, 30, 50, 0.95) 100%)',
                backdropFilter: 'blur(10px)',
                border: '1px solid rgba(0, 255, 136, 0.2)',
            }}
        >
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography
                    variant="h6"
                    sx={{
                        color: '#00ff88',
                        fontWeight: 600,
                        textShadow: '0 0 10px rgba(0, 255, 136, 0.5)'
                    }}
                >
                    {symbol} Chart
                </Typography>

                <ToggleButtonGroup
                    value={timeframe}
                    exclusive
                    onChange={handleTimeframeChange}
                    size="small"
                    sx={{
                        '& .MuiToggleButton-root': {
                            color: '#00ff88',
                            borderColor: 'rgba(0, 255, 136, 0.3)',
                            '&.Mui-selected': {
                                background: 'rgba(0, 255, 136, 0.2)',
                                color: '#00ff88',
                            },
                        },
                    }}
                >
                    <ToggleButton value="1M">1M</ToggleButton>
                    <ToggleButton value="5M">5M</ToggleButton>
                    <ToggleButton value="15M">15M</ToggleButton>
                    <ToggleButton value="1H">1H</ToggleButton>
                    <ToggleButton value="1D">1D</ToggleButton>
                </ToggleButtonGroup>
            </Box>

            <div ref={chartContainerRef} style={{ position: 'relative' }} />
        </Paper>
    );
};

export default TradingChart;
