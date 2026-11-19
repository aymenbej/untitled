"""Journal d'audit de toutes les décisions de conformité (fichier JSONL, en ajout seul)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any


def _default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "value"):  # Enum
        return value.value
    return str(value)


@dataclass
class AuditLog:
    path: Path | None = None
    events: list[dict[str, Any]] = field(default_factory=list)

    def record(self, event: str, allowed: bool, reasons: list[str] | tuple[str, ...] = (), **details: Any) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "decision": "AUTORISÉ" if allowed else "BLOQUÉ",
            "reasons": list(reasons),
            **details,
        }
        self.events.append(entry)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False, default=_default) + "\n")
