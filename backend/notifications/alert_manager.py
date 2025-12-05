"""Alert Manager for Trading Bot.

Sends notifications via multiple channels:
- Email
- Telegram (optional)
- Discord (optional)
- Webhooks
"""

from typing import Dict, List, Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import aiohttp
from loguru import logger


class AlertLevel:
    """Alert severity levels."""
    INFO = 'info'
    WARNING = 'warning'
    ERROR = 'error'
    CRITICAL = 'critical'


class AlertManager:
    """
    Manage alerts and notifications.
    
    Supports multiple notification channels with priority-based routing.
    """
    
    def __init__(
        self,
        email_config: Optional[Dict] = None,
        telegram_config: Optional[Dict] = None,
        webhook_urls: Optional[List[str]] = None
    ):
        """
        Initialize alert manager.
        
        Args:
            email_config: SMTP configuration
            telegram_config: Telegram bot configuration
            webhook_urls: List of webhook URLs
        """
        self.email_config = email_config
        self.telegram_config = telegram_config
        self.webhook_urls = webhook_urls or []
        
        self.alert_history = []
        self.alert_counts = {
            AlertLevel.INFO: 0,
            AlertLevel.WARNING: 0,
            AlertLevel.ERROR: 0,
            AlertLevel.CRITICAL: 0
        }
        
        logger.info("Alert Manager initialized")
        
    async def send_alert(
        self,
        title: str,
        message: str,
        level: str = AlertLevel.INFO,
        data: Optional[Dict] = None
    ):
        """
        Send alert through configured channels.
        
        Args:
            title: Alert title
            message: Alert message
            level: Alert level (info/warning/error/critical)
            data: Additional data
        """
        alert = {
            'timestamp': datetime.now().isoformat(),
            'title': title,
            'message': message,
            'level': level,
            'data': data or {}
        }
        
        # Track alert
        self.alert_history.append(alert)
        self.alert_counts[level] += 1
        
        # Keep only recent history
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]
            
        # Send based on level
        if level in [AlertLevel.ERROR, AlertLevel.CRITICAL]:
            # Send via all channels for critical alerts
            await self._send_email(alert)
            await self._send_telegram(alert)
            await self._send_webhooks(alert)
        elif level == AlertLevel.WARNING:
            # Send via email and webhooks
            await self._send_email(alert)
            await self._send_webhooks(alert)
        else:
            # Info: only log and webhooks
            await self._send_webhooks(alert)
            
        logger.log(
            level.upper() if level != 'info' else 'INFO',
            f"Alert: {title} - {message}"
        )
        
    async def _send_email(self, alert: Dict):
        """Send email notification."""
        if not self.email_config:
            return
            
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config['from']
            msg['To'] = self.email_config['to']
            msg['Subject'] = f"[{alert['level'].upper()}] {alert['title']}"
            
            body = f"""
Trading Bot Alert

Level: {alert['level'].upper()}
Time: {alert['timestamp']}
Title: {alert['title']}

Message:
{alert['message']}

{'-' * 50}
Additional Data:
{alert['data']}
"""
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port']) as server:
                if self.email_config.get('use_tls'):
                    server.starttls()
                if 'username' in self.email_config:
                    server.login(self.email_config['username'], self.email_config['password'])
                server.send_message(msg)
                
            logger.debug(f"Email sent: {alert['title']}")
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            
    async def _send_telegram(self, alert: Dict):
        """Send Telegram notification."""
        if not self.telegram_config:
            return
            
        try:
            message = f"""
🤖 *Trading Bot Alert*

⚠️ Level: {alert['level'].upper()}
🕒 Time: {alert['timestamp']}
📌 {alert['title']}

{alert['message']}
"""
            
            url = f"https://api.telegram.org/bot{self.telegram_config['bot_token']}/sendMessage"
            
            async with aiohttp.ClientSession() as session:
                await session.post(url, json={
                    'chat_id': self.telegram_config['chat_id'],
                    'text': message,
                    'parse_mode': 'Markdown'
                })
                
            logger.debug(f"Telegram sent: {alert['title']}")
            
        except Exception as e:
            logger.error(f"Failed to send Telegram: {e}")
            
    async def _send_webhooks(self, alert: Dict):
        """Send to webhook URLs."""
        if not self.webhook_urls:
            return
            
        async with aiohttp.ClientSession() as session:
            for url in self.webhook_urls:
                try:
                    await session.post(url, json=alert)
                except Exception as e:
                    logger.error(f"Failed to send webhook to {url}: {e}")
                    
    def get_recent_alerts(self, limit: int = 50) -> List[Dict]:
        """Get recent alerts."""
        return self.alert_history[-limit:]
        
    def get_alert_summary(self) -> Dict:
        """Get alert summary statistics."""
        return {
            'total_alerts': len(self.alert_history),
            'counts_by_level': self.alert_counts.copy(),
            'last_alert': self.alert_history[-1] if self.alert_history else None
        }


class TradingAlerts:
    """Pre-configured trading-specific alerts."""
    
    def __init__(self, alert_manager: AlertManager):
        """
        Initialize trading alerts.
        
        Args:
            alert_manager: AlertManager instance
        """
        self.alert_manager = alert_manager
        
    async def position_opened(self, symbol: str, size: float, price: float):
        """Alert when position is opened."""
        await self.alert_manager.send_alert(
            title="Position Opened",
            message=f"Opened position: {symbol} @ ${price:.2f}, Size: {size}",
            level=AlertLevel.INFO,
            data={'symbol': symbol, 'size': size, 'price': price}
        )
        
    async def position_closed(self, symbol: str, pnl: float, pnl_pct: float):
        """Alert when position is closed."""
        level = AlertLevel.INFO if pnl >= 0 else AlertLevel.WARNING
        
        await self.alert_manager.send_alert(
            title="Position Closed",
            message=f"Closed {symbol}: P&L ${pnl:.2f} ({pnl_pct:.2f}%)",
            level=level,
            data={'symbol': symbol, 'pnl': pnl, 'pnl_pct': pnl_pct}
        )
        
    async def stop_loss_hit(self, symbol: str, loss: float):
        """Alert when stop-loss is hit."""
        await self.alert_manager.send_alert(
            title="Stop-Loss Hit",
            message=f"Stop-loss triggered for {symbol}: Loss ${loss:.2f}",
            level=AlertLevel.WARNING,
            data={'symbol': symbol, 'loss': loss}
        )
        
    async def daily_limit_reached(self, limit_type: str, current: float, limit: float):
        """Alert when daily limit is reached."""
        await self.alert_manager.send_alert(
            title="Daily Limit Reached",
            message=f"{limit_type} limit reached: ${current:.2f} / ${limit:.2f}",
            level=AlertLevel.ERROR,
            data={'type': limit_type, 'current': current, 'limit': limit}
        )
        
    async def strategy_performance(self, strategy: str, sharpe: float, return_pct: float):
        """Alert on strategy performance."""
        level = AlertLevel.INFO if return_pct >= 0 else AlertLevel.WARNING
        
        await self.alert_manager.send_alert(
            title="Strategy Performance",
            message=f"{strategy}: Return {return_pct:.2f}%, Sharpe {sharpe:.2f}",
            level=level,
            data={'strategy': strategy, 'sharpe': sharpe, 'return': return_pct}
        )
        
    async def var_breach(self, var_limit: float, current_loss: float):
        """Alert when VaR limit is breached."""
        await self.alert_manager.send_alert(
            title="VaR Limit Breached",
            message=f"Current loss ${current_loss:.2f} exceeds VaR ${var_limit:.2f}",
            level=AlertLevel.CRITICAL,
            data={'var_limit': var_limit, 'current_loss': current_loss}
        )


# Global instance
alert_manager = AlertManager()
trading_alerts = TradingAlerts(alert_manager)
