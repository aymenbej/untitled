"""Registre de purification : tout montant d'origine non conforme reçu par erreur
(intérêt, récompense, part illicite d'un dividende) est mis de côté pour être donné."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path


@dataclass(frozen=True)
class PurificationEntry:
    transaction_id: str
    timestamp: datetime
    amount: Decimal
    currency: str
    reason: str
    status: str = "à_donner"  # puis "donné"


@dataclass
class PurificationRegister:
    path: Path | None = None
    entries: list[PurificationEntry] = field(default_factory=list)

    def add(self, entry: PurificationEntry) -> None:
        if any(e.transaction_id == entry.transaction_id for e in self.entries):
            return  # une transaction n'est purifiée qu'une fois
        self.entries.append(entry)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                row = asdict(entry)
                row["amount"] = str(entry.amount)
                row["timestamp"] = entry.timestamp.isoformat()
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def total_pending(self) -> dict[str, Decimal]:
        totals: dict[str, Decimal] = {}
        for e in self.entries:
            if e.status == "à_donner":
                totals[e.currency] = totals.get(e.currency, Decimal("0")) + e.amount
        return totals


def dividend_purification_amount(dividend: Decimal, non_permissible_income_ratio: Decimal) -> Decimal:
    """Part d'un dividende à purifier. En tolérance zéro, une société conforme a un
    ratio de 0 ; un ratio > 0 signifie qu'elle a perdu sa conformité depuis l'achat."""
    if non_permissible_income_ratio < 0 or non_permissible_income_ratio > 1:
        raise ValueError("ratio de revenus non conformes hors de [0, 1]")
    return abs(dividend) * non_permissible_income_ratio
