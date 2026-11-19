from __future__ import annotations


class ComplianceError(Exception):
    """Erreur de configuration de la politique de conformité."""


class ComplianceViolation(Exception):
    """Une action a été bloquée parce qu'elle enfreint la politique halal.

    ``reasons`` liste toutes les règles enfreintes, pas seulement la première,
    pour que l'audit soit complet.
    """

    def __init__(self, action: str, reasons: list[str]):
        self.action = action
        self.reasons = list(reasons)
        super().__init__(f"{action} bloqué : " + " ; ".join(self.reasons))
