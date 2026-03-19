"""
Persistent checkpoint state — survives process restarts.

Stored as a JSON file at data/.state.json.
Tracks `last_fetched_utc` per (job_name, optional key) so each incremental
run only fetches data that arrived after the last successful fetch.

Thread-safe: uses a file lock so concurrent processes don't corrupt it.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

STATE_FILE = Path("./data/.state.json")
_EPOCH = "2009-12-31T00:00:00+00:00"   # default start if no checkpoint exists (matches START_EQUITY)


class StateStore:

    def __init__(self, path: Path = STATE_FILE) -> None:
        self._path = path
        self._lock = threading.Lock()   # serialise concurrent read-modify-write
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict = self._load()

    def _load(self) -> dict:
        if self._path.exists():
            try:
                with open(self._path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("State file corrupt (%s) — starting fresh", exc)
        return {}

    def _save(self) -> None:
        # Caller must hold self._lock.
        tmp = self._path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self._data, f, indent=2, default=str)
        os.replace(tmp, self._path)   # atomic on POSIX

    def get_last_fetched(self, job: str, key: str = "") -> datetime:
        """
        Return the last successfully fetched UTC datetime for this job+key.
        Returns epoch (2009-12-31) if no checkpoint exists.
        """
        with self._lock:
            raw = self._data.get(job, {}).get(key or "_", _EPOCH)
        return datetime.fromisoformat(raw)

    def set_last_fetched(self, job: str, key: str = "",
                         dt: datetime | None = None) -> None:
        """Record a successful fetch. Defaults to utcnow."""
        ts = (dt or datetime.now(timezone.utc)).isoformat()
        with self._lock:
            if job not in self._data:
                self._data[job] = {}
            self._data[job][key or "_"] = ts
            self._save()
        log.debug("Checkpoint saved: %s/%s → %s", job, key or "_", ts)

    def get_all(self) -> dict:
        with self._lock:
            return dict(self._data)

    def reset(self, job: str | None = None) -> None:
        """Reset one job or all checkpoints (forces full re-fetch)."""
        with self._lock:
            if job:
                self._data.pop(job, None)
            else:
                self._data.clear()
            self._save()
        log.info("Checkpoint reset: %s", job or "ALL")
