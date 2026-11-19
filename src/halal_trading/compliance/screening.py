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

from .policy import ACTIVITY_SCREEN, STABLECOIN_CODES, CompliancePolicy


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
    notes: tuple[str, ...] = ()  # estimations prudentes utilisées (bornes hautes...)


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
    warnings: tuple[str, ...] = ()
    # Part du CA d'origine non conforme (intérêts + autres) : sert à purifier les dividendes.
    impure_income_ratio: Decimal | None = None

    @property
    def status(self) -> str:
        return "CONFORME" if self.compliant else "EXCLU"


def _now() -> datetime:
    return datetime.now(timezone.utc)


# Négations : « non-alcoholic », « sans alcool », « alcohol-free »... ne sont pas des activités interdites.
_NEGATION_BEFORE = r"(?<!non-)(?<!non )(?<!sans )(?<!no )"
_NEGATION_AFTER = r"(?![- ]free)"


def _keyword_patterns(policy: CompliancePolicy) -> list[tuple[str, str, re.Pattern[str]]]:
    patterns = []
    for category, keywords in policy.prohibited_activities.items():
        for kw in keywords:
            # mot entier, pluriel simple toléré ; insensible à la casse
            pattern = re.compile(
                _NEGATION_BEFORE + r"(?<!\w)" + re.escape(kw.lower()) + r"(?:s|es)?(?!\w)" + _NEGATION_AFTER,
                re.IGNORECASE,
            )
            patterns.append((category, kw, pattern))
    return patterns


def _snippet(text: str, match: re.Match[str], width: int = 40) -> str:
    if len(text) <= 2 * width + 20:
        return text
    start, end = max(0, match.start() - width), min(len(text), match.end() + width)
    return ("…" if start else "") + text[start:end] + ("…" if end < len(text) else "")


def find_prohibited_activities(texts: list[str], policy: CompliancePolicy) -> list[str]:
    """Renvoie la liste des correspondances « catégorie (mot-clé) dans 'extrait' »,
    une seule par catégorie et par texte."""
    hits = []
    patterns = _keyword_patterns(policy)
    for text in texts:
        if not text:
            continue
        seen: set[str] = set()
        for category, kw, pattern in patterns:
            if category in seen:
                continue
            match = pattern.search(text)
            if match:
                seen.add(category)
                hits.append(f"{category} (« {kw} » dans « {_snippet(text, match)} »)")
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

    # 2. Exclusion manuelle (décision humaine, toujours prioritaire)
    manual = policy.manual_exclusions.get(profile.symbol.upper())
    if manual:
        reasons.append(f"exclusion manuelle : {manual}")

    # 3. Activité
    if not profile.sector and not profile.industry and not profile.business_activities:
        reasons.append("activité inconnue (secteur, industrie et activités absents)")
    texts = [profile.sector or "", profile.industry or "", *profile.business_activities]
    texts += [segment for segment, revenue in profile.revenue_segments.items() if revenue != 0]
    for hit in find_prohibited_activities(texts, policy):
        reasons.append(f"activité interdite : {hit}")

    # 4. Finances
    warnings: list[str] = []
    impure_ratio = None
    fin = profile.financials
    if fin is None:
        reasons.append("données financières absentes")
    else:
        fin_reasons, warnings, impure_ratio = _screen_financials(fin, policy, today)
        reasons.extend(fin_reasons)

    return ScreeningResult(
        symbol=profile.symbol,
        compliant=not reasons,
        reasons=tuple(reasons),
        screened_at=_now(),
        instrument_type=profile.instrument_type,
        warnings=tuple(warnings),
        impure_income_ratio=impure_ratio,
    )


def _screen_financials(
    fin: CompanyFinancials, policy: CompliancePolicy, today: date
) -> tuple[list[str], list[str], Decimal | None]:
    """Renvoie (raisons d'exclusion, avertissements, ratio de revenus non conformes)."""
    reasons: list[str] = []
    warnings: list[str] = list(fin.notes)

    age = (today - fin.period_end).days
    if age > policy.max_financials_age_days:
        reasons.append(f"données financières trop anciennes ({age} jours > {policy.max_financials_age_days})")

    non_permissible = fin.non_permissible_income
    if (
        non_permissible is None
        and policy.max_non_permissible_income_ratio is not None
        and policy.missing_non_permissible_income == ACTIVITY_SCREEN
    ):
        non_permissible = Decimal("0")
        warnings.append(
            "autres revenus non conformes non fournis par la source : contrôle limité au filtre d'activité"
        )

    # Seules les données utiles aux critères actifs sont exigées.
    required: dict[str, Decimal | None] = {"total_revenue": fin.total_revenue, "market_cap": fin.market_cap}
    if policy.max_interest_bearing_debt_ratio is not None:
        required["interest_bearing_debt"] = fin.interest_bearing_debt
        if policy.count_finance_leases_as_debt:
            required["finance_lease_liabilities"] = fin.finance_lease_liabilities
        if policy.count_operating_leases_as_debt:
            required["operating_lease_liabilities"] = fin.operating_lease_liabilities
    if policy.max_interest_bearing_investments_ratio is not None:
        required["interest_bearing_investments"] = fin.interest_bearing_investments
    if policy.max_interest_income_ratio is not None or policy.max_non_permissible_income_ratio is not None:
        required["interest_income"] = fin.interest_income
    if policy.max_interest_expense_ratio is not None:
        required["interest_expense"] = fin.interest_expense
    if policy.max_non_permissible_income_ratio is not None:
        required["non_permissible_income"] = non_permissible

    missing = [name for name, value in required.items() if value is None]
    if missing:
        reasons.append("données manquantes : " + ", ".join(missing))

    # Les ratios calculables sont tout de même vérifiés, pour que le rapport
    # montre toutes les causes d'exclusion, pas seulement l'absence de données.
    if fin.market_cap is None or fin.total_revenue is None:
        return reasons, warnings, None
    if fin.market_cap <= 0 or fin.total_revenue <= 0:
        reasons.append("capitalisation ou chiffre d'affaires nul ou négatif : ratios incalculables")
        return reasons, warnings, None

    debt_parts = [fin.interest_bearing_debt]
    if policy.count_finance_leases_as_debt:
        debt_parts.append(fin.finance_lease_liabilities)
    if policy.count_operating_leases_as_debt:
        debt_parts.append(fin.operating_lease_liabilities)
    known_debt = [d for d in debt_parts if d is not None]
    debt = sum(known_debt, Decimal("0")) if known_debt else None

    impure = None
    if fin.interest_income is not None and non_permissible is not None:
        impure = abs(fin.interest_income) + abs(non_permissible)
    impure_ratio = None if impure is None else impure / fin.total_revenue

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
            "revenus non conformes totaux (intérêts compris) / CA",
            impure,
            fin.total_revenue,
            policy.max_non_permissible_income_ratio,
        ),
    ]
    for label, numerator, denominator, threshold in checks:
        if numerator is None or threshold is None:
            continue
        ratio = _ratio(numerator, denominator)
        if ratio > threshold:
            reasons.append(f"{label} = {ratio:.3%} > seuil {threshold:.2%}")
    return reasons, warnings, impure_ratio


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
