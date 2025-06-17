from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, List
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

class FileWatcher:
    """Observa alterações e aciona callback de indexação incremental."""

    def __init__(self, paths: List[str], on_change: Callable[[str], None], patterns: List[str] | None = None):
        self.paths = paths
        self.on_change = on_change
        self.patterns = patterns or ["*.py", "*.js", "*.ts", "*.tsx", "*.java", "*.go", "*.rb", "*.cs"]
        self.observer = Observer()

    def _build_handler(self):
        handler = PatternMatchingEventHandler(patterns=self.patterns, ignore_directories=True)

        def _on_any(event):
            if event.event_type in {"created", "modified"}:
                self.on_change(event.src_path)
            elif event.event_type == "deleted":
                self.on_change(event.src_path + "::deleted")
        handler.on_modified = _on_any  # type: ignore[assignment]
        handler.on_created = _on_any  # type: ignore[assignment]
        return handler

    def start(self):
        handler = self._build_handler()
        for p in self.paths:
            self.observer.schedule(handler, str(Path(p)), recursive=True)
        self.observer.start()

    def stop(self):
        self.observer.stop()
        self.observer.join() 