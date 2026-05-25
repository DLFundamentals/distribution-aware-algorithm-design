from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


@dataclass
class MetricsLogger:
    """Small in-memory logger with optional JSONL persistence."""

    path: Path | None = None
    rows: list[dict[str, Any]] = field(default_factory=list)

    def log(self, **metrics: Any) -> dict[str, Any]:
        row = {key: _json_safe(value) for key, value in metrics.items()}
        self.rows.append(row)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        return row

    def latest(self) -> dict[str, Any] | None:
        return self.rows[-1] if self.rows else None

    def to_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return destination


def mean_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                buckets.setdefault(key, []).append(float(value))
    return {
        key: sum(values) / len(values)
        for key, values in buckets.items()
        if values
    }
