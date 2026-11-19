"""Univers d'investissement validé : seuls les actifs ayant passé le filtre y figurent."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from .screening import ScreeningResult


@dataclass
class ApprovedUniverse:
    max_age_days: int
    _results: dict[str, ScreeningResult] = field(default_factory=dict)

    def add(self, result: ScreeningResult) -> None:
        """Enregistre le dernier résultat de filtrage. Un actif exclu est retiré de l'univers."""
        self._results[result.symbol.upper()] = result

    def check(self, symbol: str, now: datetime | None = None) -> list[str]:
        """Renvoie la liste des raisons de refus (vide = actif autorisé)."""
        now = now or datetime.now(timezone.utc)
        result = self._results.get(symbol.upper())
        if result is None:
            return [f"{symbol} n'a jamais été filtré"]
        if not result.compliant:
            return [f"{symbol} exclu au dernier filtrage : " + " ; ".join(result.reasons)]
        if now - result.screened_at > timedelta(days=self.max_age_days):
            return [f"filtrage de {symbol} périmé (> {self.max_age_days} jours), re-filtrage obligatoire"]
        return []

    def approved_symbols(self, now: datetime | None = None) -> list[str]:
        return sorted(s for s in self._results if not self.check(s, now))
