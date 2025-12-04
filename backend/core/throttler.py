"""Task throttling to manage CPU usage."""

import asyncio
from typing import Callable, Optional

from loguru import logger

from backend.core.cpu_monitor import cpu_monitor


class Throttler:
    """Throttle task execution based on CPU usage."""

    def __init__(self, check_interval: float = 0.5):
        """
        Initialize throttler.

        Args:
            check_interval: How often to check CPU when throttling (seconds)
        """
        self.check_interval = check_interval
        self._throttled_tasks = 0
        self._total_wait_time = 0.0

    async def throttle_if_needed(self):
        """Check CPU and throttle if over limit."""
        if cpu_monitor.is_over_limit():
            start_time = asyncio.get_event_loop().time()
            logger.debug("Throttling due to high CPU usage...")
            
            await cpu_monitor.wait_until_below_limit(self.check_interval)
            
            wait_time = asyncio.get_event_loop().time() - start_time
            self._throttled_tasks += 1
            self._total_wait_time += wait_time
            
            logger.debug(f"Resumed after {wait_time:.2f}s throttle")

    async def execute_with_throttle(
        self, 
        func: Callable, 
        *args, 
        **kwargs
    ):
        """
        Execute a function with CPU throttling.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result of the function
        """
        await self.throttle_if_needed()
        
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    def get_stats(self) -> dict:
        """
        Get throttling statistics.

        Returns:
            dict: Statistics about throttling
        """
        return {
            "throttled_tasks": self._throttled_tasks,
            "total_wait_time": self._total_wait_time,
            "average_wait_time": (
                self._total_wait_time / self._throttled_tasks 
                if self._throttled_tasks > 0 
                else 0.0
            ),
        }

    def reset_stats(self):
        """Reset throttling statistics."""
        self._throttled_tasks = 0
        self._total_wait_time = 0.0


# Global instance
throttler = Throttler()
