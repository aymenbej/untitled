"""Filtrage halal des actifs (actions et crypto), profil tolérance zéro.

Principe « fail-closed » : une donnée manquante, inconnue ou trop ancienne
entraîne un REFUS. Un actif n'est jamais accepté par défaut.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal

from halal_trading.domain import InstrumentType

from .policy import STABLECOIN_CODES, CompliancePolicy


@dataclass(frozen=True)
class CompanyFinancials:
    """Données financières nécessaires au filtre. ``None`` = inconnu = refus.

    Tous les montants sont dans la même devise.
    """

    period_end: date
    total_revenue: Decimal | None
    market_cap: Decimal | None
    interest_bearing_debt: Decimal | None  # emprunts bancaires, obligations émises, découverts
    finance_lease_liabilities: Decimal | None  # crédit-bail de financement
    operating_lease_liabilities: Decimal | None  # locations simples (loyers)
    interest_bearing_investments: Decimal | None  # dépôts rémunérés, obligations, monétaire
    interest_income: Decimal | None
    interest_expense: Decimal | None  # charges d'intérêts sur dette financière
    non_permissible_income: Decimal | None  # autres revenus illicites (hors intérêts)


@dataclass(frozen=True)
class CompanyProfile:
    symbol: str
    name: str
    instrument_type: InstrumentType
    sector: str | None
    industry: str | None
    business_activities: tuple[str, ...] = ()
    revenue_segments: dict[str, Decimal] = field(default_factory=dict)  # segment -> chiffre d'affaires
    financials: CompanyFinancials | None = None


@dataclass(frozen=True)
class CryptoAsset:
    symbol: str
    name: str
    categories: frozenset[str] = frozenset()
    interest_bearing: bool | None = None  # le jeton verse-t-il un rendement / intérêt ?


@dataclass(frozen=True)
class ScreeningResult:
    symbol: str
    compliant: bool
    reasons: tuple[str, ...]
    screened_at: datetime
    instrument_type: InstrumentType

    @property
    def status(self) -> str:
        return "CONFORME" if self.compliant else "EXCLU"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _keyword_patterns(policy: CompliancePolicy) -> list[tuple[str, str, re.Pattern[str]]]:
    patterns = []
    for category, keywords in policy.prohibited_activities.items():
        for kw in keywords:
            # mot entier, pluriel simple toléré ; insensible à la casse
            pattern = re.compile(r"(?<!\w)" + re.escape(kw.lower()) + r"(?:s|es)?(?!\w)", re.IGNORECASE)
            patterns.append((category, kw, pattern))
    return patterns


def find_prohibited_activities(texts: list[str], policy: CompliancePolicy) -> list[str]:
    """Renvoie la liste des correspondances « catégorie (mot-clé) dans 'texte' »."""
    hits = []
    patterns = _keyword_patterns(policy)
    for text in texts:
        if not text:
            continue
        for category, kw, pattern in patterns:
            if pattern.search(text):
                hits.append(f"{category} (« {kw} » dans « {text} »)")
    return hits


def _ratio(numerator: Decimal, denominator: Decimal) -> Decimal:
    return abs(numerator) / denominator


def screen_company(
    profile: CompanyProfile,
    policy: CompliancePolicy,
    today: date | None = None,
) -> ScreeningResult:
    today = today or _now().date()
    reasons: list[str] = []

    # 1. Type d'instrument
    if profile.instrument_type != InstrumentType.COMMON_STOCK:
        reasons.append(f"type d'instrument non autorisé : {profile.instrument_type.value}")

    # 2. Activité
    if not profile.sector and not profile.industry and not profile.business_activities:
        reasons.append("activité inconnue (secteur, industrie et activités absents)")
    texts = [profile.sector or "", profile.industry or "", *profile.business_activities]
    texts += [segment for segment, revenue in profile.revenue_segments.items() if revenue != 0]
    for hit in find_prohibited_activities(texts, policy):
        reasons.append(f"activité interdite : {hit}")

    # 3. Finances
    fin = profile.financials
    if fin is None:
        reasons.append("données financières absentes")
    else:
        reasons.extend(_screen_financials(fin, policy, today))

    return ScreeningResult(
        symbol=profile.symbol,
        compliant=not reasons,
        reasons=tuple(reasons),
        screened_at=_now(),
        instrument_type=profile.instrument_type,
    )


def _screen_financials(fin: CompanyFinancials, policy: CompliancePolicy, today: date) -> list[str]:
    reasons: list[str] = []

    age = (today - fin.period_end).days
    if age > policy.max_financials_age_days:
        reasons.append(f"données financières trop anciennes ({age} jours > {policy.max_financials_age_days})")

    required = {
        "total_revenue": fin.total_revenue,
        "market_cap": fin.market_cap,
        "interest_bearing_debt": fin.interest_bearing_debt,
        "interest_bearing_investments": fin.interest_bearing_investments,
        "interest_income": fin.interest_income,
        "interest_expense": fin.interest_expense,
        "non_permissible_income": fin.non_permissible_income,
    }
    if policy.count_finance_leases_as_debt:
        required["finance_lease_liabilities"] = fin.finance_lease_liabilities
    if policy.count_operating_leases_as_debt:
        required["operating_lease_liabilities"] = fin.operating_lease_liabilities

    missing = [name for name, value in required.items() if value is None]
    if missing:
        reasons.append("données manquantes : " + ", ".join(missing))
        return reasons

    if fin.market_cap <= 0 or fin.total_revenue <= 0:
        reasons.append("capitalisation ou chiffre d'affaires nul ou négatif : ratios incalculables")
        return reasons

    debt = fin.interest_bearing_debt
    if policy.count_finance_leases_as_debt:
        debt += fin.finance_lease_liabilities
    if policy.count_operating_leases_as_debt:
        debt += fin.operating_lease_liabilities

    checks = [
        ("dette à intérêt / capitalisation", debt, fin.market_cap, policy.max_interest_bearing_debt_ratio),
        (
            "placements à intérêt / capitalisation",
            fin.interest_bearing_investments,
            fin.market_cap,
            policy.max_interest_bearing_investments_ratio,
        ),
        ("revenus d'intérêts / CA", fin.interest_income, fin.total_revenue, policy.max_interest_income_ratio),
        ("charges d'intérêts / CA", fin.interest_expense, fin.total_revenue, policy.max_interest_expense_ratio),
        (
            "revenus non conformes / CA",
            fin.non_permissible_income,
            fin.total_revenue,
            policy.max_non_permissible_income_ratio,
        ),
    ]
    for label, numerator, denominator, threshold in checks:
        ratio = _ratio(numerator, denominator)
        if ratio > threshold:
            reasons.append(f"{label} = {ratio:.4%} > seuil {threshold:.4%} (montant : {numerator})")
    return reasons


def screen_crypto(asset: CryptoAsset, policy: CompliancePolicy) -> ScreeningResult:
    reasons: list[str] = []
    symbol = asset.symbol.upper()

    if symbol in STABLECOIN_CODES or "stablecoin" in {c.lower() for c in asset.categories}:
        reasons.append("stablecoin : réserves placées à intérêt")

    approval = policy.crypto_approval(symbol)
    if approval is None:
        reasons.append("absent de la liste blanche crypto validée")

    forbidden = {c.lower() for c in asset.categories} & policy.forbidden_crypto_categories
    forbidden.discard("stablecoin")  # déjà signalé
    if forbidden:
        reasons.append("catégories interdites : " + ", ".join(sorted(forbidden)))

    if not asset.categories:
        reasons.append("catégories inconnues")

    if asset.interest_bearing is None:
        reasons.append("nature du rendement inconnue (interest_bearing non renseigné)")
    elif asset.interest_bearing:
        reasons.append("le jeton verse un rendement / intérêt")

    return ScreeningResult(
        symbol=symbol,
        compliant=not reasons,
        reasons=tuple(reasons),
        screened_at=_now(),
        instrument_type=InstrumentType.CRYPTO_SPOT,
    )
