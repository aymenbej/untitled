from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from halal_trading.compliance import (
    AuditLog,
    CompanyFinancials,
    CompanyProfile,
    CompliancePolicy,
    CryptoApproval,
    load_policy,
)
from halal_trading.domain import InstrumentType

CONFIG = Path(__file__).resolve().parents[1] / "config" / "compliance.yaml"
TODAY = date(2026, 11, 19)


@pytest.fixture
def policy() -> CompliancePolicy:
    base = load_policy(CONFIG)
    # Liste blanche de test uniquement : la config réelle est vide par défaut.
    return CompliancePolicy(
        **{
            **{f: getattr(base, f) for f in base.__dataclass_fields__},
            "crypto_whitelist": (CryptoApproval("BTC", approved_by="test"),),
        }
    )


@pytest.fixture
def audit() -> AuditLog:
    return AuditLog()


def clean_financials(**overrides) -> CompanyFinancials:
    values = dict(
        period_end=date(2026, 6, 30),
        total_revenue=Decimal("1000000"),
        market_cap=Decimal("50000000"),
        interest_bearing_debt=Decimal("0"),
        finance_lease_liabilities=Decimal("0"),
        operating_lease_liabilities=Decimal("200000"),
        interest_bearing_investments=Decimal("0"),
        interest_income=Decimal("0"),
        interest_expense=Decimal("0"),
        non_permissible_income=Decimal("0"),
    )
    values.update(overrides)
    return CompanyFinancials(**values)


def clean_company(**overrides) -> CompanyProfile:
    values = dict(
        symbol="HLL",
        name="Halal Software SA",
        instrument_type=InstrumentType.COMMON_STOCK,
        sector="Technology",
        industry="Software - Application",
        business_activities=("logiciels de gestion",),
        revenue_segments={"Licences": Decimal("600000"), "Services": Decimal("400000")},
        financials=clean_financials(),
    )
    values.update(overrides)
    return CompanyProfile(**values)


def utc(*args) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)
