"""Request/prediction log store.

Each prediction is appended as one JSON line. The log is later read back by the
monitoring module to build Evidently drift reports (reference vs. live traffic).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class RequestLogStore(Protocol):
    """Append-only store of prediction records."""

    def append(self, record: Mapping[str, object]) -> None: ...  # pragma: no cover - typing only

    def read_all(self) -> list[dict[str, object]]: ...  # pragma: no cover - typing only


@dataclass(frozen=True)
class JsonlFileLogStore:
    """A :class:`RequestLogStore` backed by a JSON-lines file."""

    path: Path

    def append(self, record: Mapping[str, object]) -> None:
        """Append one record as a JSON line, creating the parent directory if needed."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(record)) + "\n")

    def read_all(self) -> list[dict[str, object]]:
        """Return all stored records (empty list if the file does not exist yet)."""
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
