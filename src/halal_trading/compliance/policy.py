"""Chargement et validation de la politique de conformité halal."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

import yaml

from halal_trading.domain import InstrumentType

from .errors import ComplianceError

ZERO_TOLERANCE = "zero_tolerance"

# Ces types ne peuvent jamais être autorisés, quelle que soit la configuration.
NEVER_ALLOWED_TYPES = frozenset(
    {
        InstrumentType.PREFERRED_STOCK,
        InstrumentType.BOND,
        InstrumentType.MONEY_MARKET_FUND,
        InstrumentType.CFD,
        InstrumentType.FUTURE,
        InstrumentType.OPTION,
        InstrumentType.PERPETUAL,
        InstrumentType.LEVERAGED_PRODUCT,
        InstrumentType.STABLECOIN,
    }
)

STABLECOIN_CODES = frozenset({"USDC", "USDT", "DAI", "EURC", "PYUSD", "TUSD", "BUSD", "FDUSD", "USDE", "EURT"})


@dataclass(frozen=True)
class CryptoApproval:
    symbol: str
    approved_by: str
    note: str = ""


@dataclass(frozen=True)
class CompliancePolicy:
    profile: str = ZERO_TOLERANCE
    allowed_instrument_types: frozenset[InstrumentType] = frozenset(
        {InstrumentType.COMMON_STOCK, InstrumentType.CRYPTO_SPOT}
    )
    allowed_quote_currencies: frozenset[str] = frozenset({"EUR", "USD"})
    max_interest_bearing_debt_ratio: Decimal = Decimal("0")
    max_interest_bearing_investments_ratio: Decimal = Decimal("0")
    max_interest_income_ratio: Decimal = Decimal("0")
    max_interest_expense_ratio: Decimal = Decimal("0")
    max_non_permissible_income_ratio: Decimal = Decimal("0")
    count_finance_leases_as_debt: bool = True
    count_operating_leases_as_debt: bool = False
    max_financials_age_days: int = 400
    max_screening_age_days: int = 100
    max_account_confirmation_age_days: int = 90
    prohibited_activities: dict[str, tuple[str, ...]] = field(default_factory=dict)
    crypto_whitelist: tuple[CryptoApproval, ...] = ()
    forbidden_crypto_categories: frozenset[str] = frozenset(
        {"lending", "yield", "staking", "stablecoin", "gambling", "derivatives", "leverage"}
    )

    def __post_init__(self) -> None:
        self.validate()

    @property
    def thresholds(self) -> dict[str, Decimal]:
        return {
            "max_interest_bearing_debt_ratio": self.max_interest_bearing_debt_ratio,
            "max_interest_bearing_investments_ratio": self.max_interest_bearing_investments_ratio,
            "max_interest_income_ratio": self.max_interest_income_ratio,
            "max_interest_expense_ratio": self.max_interest_expense_ratio,
            "max_non_permissible_income_ratio": self.max_non_permissible_income_ratio,
        }

    def crypto_approval(self, symbol: str) -> CryptoApproval | None:
        for approval in self.crypto_whitelist:
            if approval.symbol.upper() == symbol.upper():
                return approval
        return None

    def validate(self) -> None:
        forbidden = self.allowed_instrument_types & NEVER_ALLOWED_TYPES
        if forbidden:
            names = ", ".join(sorted(t.value for t in forbidden))
            raise ComplianceError(f"Types d'instruments jamais autorisés présents dans la configuration : {names}")

        stablecoins = {c.upper() for c in self.allowed_quote_currencies} & STABLECOIN_CODES
        if stablecoins:
            raise ComplianceError(f"Stablecoins interdits comme devise de cotation : {', '.join(sorted(stablecoins))}")

        for name, value in self.thresholds.items():
            if value < 0:
                raise ComplianceError(f"Seuil négatif invalide : {name}={value}")

        for approval in self.crypto_whitelist:
            if not approval.approved_by.strip():
                raise ComplianceError(f"Crypto {approval.symbol} sans validateur (approved_by)")
            if approval.symbol.upper() in STABLECOIN_CODES:
                raise ComplianceError(f"Stablecoin interdit dans la liste blanche : {approval.symbol}")

        if self.profile == ZERO_TOLERANCE:
            non_zero = {k: v for k, v in self.thresholds.items() if v != 0}
            if non_zero:
                raise ComplianceError(
                    "Profil tolérance zéro : tous les seuils doivent valoir 0, trouvé "
                    + ", ".join(f"{k}={v}" for k, v in non_zero.items())
                )
            if not self.count_finance_leases_as_debt:
                raise ComplianceError("Profil tolérance zéro : le crédit-bail financier doit compter comme dette.")


def load_policy(path: str | Path) -> CompliancePolicy:
    """Charge la politique depuis un fichier YAML (voir ``config/compliance.yaml``)."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    thresholds = raw.get("thresholds", {}) or {}

    try:
        types = frozenset(InstrumentType(t) for t in raw.get("allowed_instrument_types", []))
    except ValueError as exc:
        raise ComplianceError(f"Type d'instrument inconnu : {exc}") from exc

    whitelist = tuple(
        CryptoApproval(
            symbol=str(entry["symbol"]).upper(),
            approved_by=str(entry.get("approved_by", "")),
            note=str(entry.get("note", "")),
        )
        for entry in raw.get("crypto_whitelist", []) or []
    )

    return CompliancePolicy(
        profile=raw.get("profile", ZERO_TOLERANCE),
        allowed_instrument_types=types,
        allowed_quote_currencies=frozenset(c.upper() for c in raw.get("allowed_quote_currencies", [])),
        max_interest_bearing_debt_ratio=Decimal(str(thresholds.get("max_interest_bearing_debt_ratio", 0))),
        max_interest_bearing_investments_ratio=Decimal(
            str(thresholds.get("max_interest_bearing_investments_ratio", 0))
        ),
        max_interest_income_ratio=Decimal(str(thresholds.get("max_interest_income_ratio", 0))),
        max_interest_expense_ratio=Decimal(str(thresholds.get("max_interest_expense_ratio", 0))),
        max_non_permissible_income_ratio=Decimal(str(thresholds.get("max_non_permissible_income_ratio", 0))),
        count_finance_leases_as_debt=bool(raw.get("count_finance_leases_as_debt", True)),
        count_operating_leases_as_debt=bool(raw.get("count_operating_leases_as_debt", False)),
        max_financials_age_days=int(raw.get("max_financials_age_days", 400)),
        max_screening_age_days=int(raw.get("max_screening_age_days", 100)),
        max_account_confirmation_age_days=int(raw.get("max_account_confirmation_age_days", 90)),
        prohibited_activities={
            category: tuple(str(k).lower() for k in keywords)
            for category, keywords in (raw.get("prohibited_activities", {}) or {}).items()
        },
        crypto_whitelist=whitelist,
        forbidden_crypto_categories=frozenset(
            c.lower() for c in raw.get("forbidden_crypto_categories", []) or []
        ),
    )
