from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Callable, Set
import contextlib

class IndexQueue:
    """Fila assíncrona que de-duplia caminhos e aplica quiet-time antes de processar."""

    def __init__(self, process_fn: Callable[[str], None], quiet_time: float = 1.0):
        self.process_fn = process_fn
        self.quiet_time = quiet_time
        self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=1000)
        self._seen: Set[str] = set()
        self._task: asyncio.Task | None = None

    async def _worker(self):
        while True:
            path = await self.queue.get()
            # Aguarda quiet_time para acumular eventos duplicados
            await asyncio.sleep(self.quiet_time)
            paths_to_process = list(self._seen)
            self._seen.clear()
            for p in paths_to_process:
                try:
                    self.process_fn(p)
                except Exception as exc:  # pragma: no cover
                    print(f"[IndexQueue] erro ao processar {p}: {exc}")
            self.queue.task_done()

    def start(self):
        if self._task is None:
            self._task = asyncio.create_task(self._worker())

    def put(self, path: str):
        if len(self._seen) > 5000:
            # backlog grande: remove item arbitrário mais antigo
            self._seen.pop(next(iter(self._seen)))
        if path not in self._seen and Path(path).is_file():
            self._seen.add(path)
            if not self.queue.full():
                self.queue.put_nowait(path)

    async def stop(self):
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task 