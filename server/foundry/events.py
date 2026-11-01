"""In-process pub/sub for SSE progress streams.

Jobs run in worker threads, the SSE endpoints run on the asyncio loop. Publishing
therefore hops back to the loop with ``call_soon_threadsafe`` when a loop is bound.
"""

from __future__ import annotations

import asyncio
import itertools
import threading
from typing import Any


class EventBus:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: dict[int, asyncio.Queue] = {}
        self._ids = itertools.count(1)
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self) -> tuple[int, asyncio.Queue]:
        sub_id = next(self._ids)
        queue: asyncio.Queue = asyncio.Queue(maxsize=512)
        with self._lock:
            self._subscribers[sub_id] = queue
        return sub_id, queue

    def unsubscribe(self, sub_id: int) -> None:
        with self._lock:
            self._subscribers.pop(sub_id, None)

    def publish(self, event: dict[str, Any]) -> None:
        with self._lock:
            queues = list(self._subscribers.values())
        for queue in queues:
            self._offer(queue, event)

    def _offer(self, queue: asyncio.Queue, event: dict[str, Any]) -> None:
        def put() -> None:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # Drop the oldest event rather than blocking a worker thread.
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    pass

        loop = self._loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(put)
        else:
            put()
