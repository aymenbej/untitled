"""Vérification du compte broker au démarrage : le bot refuse de tourner si un
réglage du compte peut faire entrer de l'intérêt."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from halal_trading.domain import AccountSettings, AccountType

from .audit import AuditLog
from .errors import ComplianceViolation
from .policy import CompliancePolicy


def account_violations(
    settings: AccountSettings, policy: CompliancePolicy, now: datetime | None = None
) -> list[str]:
    now = now or datetime.now(timezone.utc)
    reasons: list[str] = []

    if settings.account_type != AccountType.CASH:
        reasons.append(f"le compte doit être de type « cash », trouvé « {settings.account_type.value} »")

    flags = {
        "marge (emprunt)": settings.margin_enabled,
        "prêt de titres": settings.securities_lending_enabled,
        "rémunération des liquidités": settings.cash_interest_enabled,
        "staking": settings.staking_enabled,
        "récompenses de rendement (ex. USDC rewards)": settings.yield_rewards_enabled,
    }
    for label, value in flags.items():
        if value is None:
            reasons.append(f"{label} : état inconnu, à confirmer")
        elif value:
            reasons.append(f"{label} activé : doit être désactivé")

    age = now - settings.confirmed_at
    if age > timedelta(days=policy.max_account_confirmation_age_days):
        reasons.append(
            f"vérification du compte trop ancienne ({age.days} jours > "
            f"{policy.max_account_confirmation_age_days}), à refaire"
        )
    if settings.source == "manual" and not (settings.confirmed_by or "").strip():
        reasons.append("confirmation manuelle sans nom de la personne qui a vérifié")

    return reasons


def check_account(
    settings: AccountSettings, policy: CompliancePolicy, audit: AuditLog, now: datetime | None = None
) -> None:
    reasons = account_violations(settings, policy, now)
    audit.record("account_check", allowed=not reasons, reasons=reasons, broker=settings.broker)
    if reasons:
        raise ComplianceViolation(f"Compte {settings.broker}", reasons)
