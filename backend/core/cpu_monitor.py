"""CPU monitoring and resource management."""

import asyncio
import time
from typing import Optional

import psutil
from loguru import logger

from backend.core.config import settings


class CPUMonitor:
    """Monitor and track CPU usage to enforce limits."""

    def __init__(self, limit_percent: Optional[int] = None):
        """
        Initialize CPU monitor.

        Args:
            limit_percent: CPU usage limit (default from settings)
        """
        self.limit_percent = limit_percent or settings.cpu_limit_percent
        self.process = psutil.Process()
        self._monitoring = False
        self._current_usage = 0.0
        self._peak_usage = 0.0
        self._violation_count = 0

    def get_current_usage(self) -> float:
        """
        Get current CPU usage percentage.

        Returns:
            float: CPU usage percentage (0-100)
        """
        # Get system-wide CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        self._current_usage = cpu_percent

        # Track peak usage
        if cpu_percent > self._peak_usage:
            self._peak_usage = cpu_percent

        return cpu_percent

    def is_over_limit(self) -> bool:
        """
        Check if CPU usage is over the limit.

        Returns:
            bool: True if over limit
        """
        usage = self.get_current_usage()
        over_limit = usage > self.limit_percent

        if over_limit:
            self._violation_count += 1
            logger.warning(
                f"CPU usage over limit: {usage:.1f}% > {self.limit_percent}% "
                f"(violations: {self._violation_count})"
            )

        return over_limit

    async def wait_until_below_limit(self, check_interval: float = 0.5):
        """
        Wait until CPU usage drops below the limit.

        Args:
            check_interval: How often to check (seconds)
        """
        while self.is_over_limit():
            logger.debug(f"Waiting for CPU to drop below {self.limit_percent}%...")
            await asyncio.sleep(check_interval)

    def get_stats(self) -> dict:
        """
        Get CPU monitoring statistics.

        Returns:
            dict: Statistics including current, peak usage, and violations
        """
        return {
            "current_usage": self._current_usage,
            "peak_usage": self._peak_usage,
            "limit_percent": self.limit_percent,
            "violation_count": self._violation_count,
            "over_limit": self._current_usage > self.limit_percent,
        }

    async def start_monitoring(self, interval: float = 1.0):
        """
        Start continuous monitoring task.

        Args:
            interval: Monitoring interval in seconds
        """
        self._monitoring = True
        logger.info(f"Starting CPU monitoring (limit: {self.limit_percent}%)")

        while self._monitoring:
            usage = self.get_current_usage()
            logger.debug(f"CPU usage: {usage:.1f}%")
            await asyncio.sleep(interval)

    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self._monitoring = False
        logger.info("Stopped CPU monitoring")

    def reset_stats(self):
        """Reset monitoring statistics."""
        self._peak_usage = 0.0
        self._violation_count = 0
        logger.debug("Reset CPU monitoring stats")


# Global instance
cpu_monitor = CPUMonitor()
