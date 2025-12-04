"""Worker pool for parallel task execution with CPU management."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional

from loguru import logger

from backend.core.config import settings
from backend.core.throttler import throttler


class WorkerPool:
    """Manage a pool of worker threads for parallel execution."""

    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize worker pool.

        Args:
            max_workers: Maximum number of worker threads (default from settings)
        """
        self.max_workers = max_workers or settings.worker_threads
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self._tasks_executed = 0
        logger.info(f"Initialized worker pool with {self.max_workers} threads")

    async def submit_task(
        self, 
        func: Callable, 
        *args, 
        throttle: bool = True,
        **kwargs
    ):
        """
        Submit a task to the worker pool.

        Args:
            func: Function to execute
            *args: Positional arguments
            throttle: Whether to apply CPU throttling
            **kwargs: Keyword arguments

        Returns:
            Result of the function
        """
        if throttle:
            await throttler.throttle_if_needed()

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self.executor, func, *args)
        self._tasks_executed += 1
        
        return result

    async def map_tasks(
        self, 
        func: Callable, 
        items: List, 
        throttle: bool = True
    ) -> List:
        """
        Map a function over a list of items in parallel.

        Args:
            func: Function to apply to each item
            items: List of items to process
            throttle: Whether to apply CPU throttling

        Returns:
            List of results
        """
        tasks = []
        for item in items:
            task = self.submit_task(func, item, throttle=throttle)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results

    def get_stats(self) -> dict:
        """
        Get worker pool statistics.

        Returns:
            dict: Statistics about task execution
        """
        return {
            "max_workers": self.max_workers,
            "tasks_executed": self._tasks_executed,
        }

    def shutdown(self, wait: bool = True):
        """
        Shutdown the worker pool.

        Args:
            wait: Whether to wait for pending tasks to complete
        """
        logger.info("Shutting down worker pool...")
        self.executor.shutdown(wait=wait)
        logger.info("Worker pool shut down")


# Global instance
worker_pool = WorkerPool()
