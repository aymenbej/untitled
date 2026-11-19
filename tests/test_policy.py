from decimal import Decimal

import pytest

from halal_trading.compliance import ComplianceError, CompliancePolicy, CryptoApproval, load_policy
from halal_trading.domain import InstrumentType

from .conftest import CONFIG


def test_config_file_loads_as_aaoifi():
    policy = load_policy(CONFIG)
    assert policy.profile == "aaoifi"
    assert policy.max_interest_bearing_debt_ratio == Decimal("0.30")
    assert policy.max_non_permissible_income_ratio == Decimal("0.05")
    assert policy.max_interest_expense_ratio is None
    assert policy.allowed_instrument_types == {InstrumentType.COMMON_STOCK, InstrumentType.CRYPTO_SPOT}
    assert policy.crypto_whitelist == ()  # rien d'autorisé sans validation explicite
    assert {"alcool", "finance_conventionnelle"} <= set(policy.prohibited_activities)


@pytest.mark.parametrize(
    "change",
    [
        {"max_interest_bearing_debt_ratio": Decimal("0.33")},
        {"max_interest_bearing_investments_ratio": None},
        {"max_non_permissible_income_ratio": Decimal("0.10")},
        {"count_finance_leases_as_debt": False},
    ],
)
def test_aaoifi_profile_cannot_be_more_permissive_than_standard(change):
    base = dict(
        profile="aaoifi",
        max_interest_bearing_debt_ratio=Decimal("0.30"),
        max_interest_bearing_investments_ratio=Decimal("0.30"),
        max_interest_income_ratio=None,
        max_interest_expense_ratio=None,
        max_non_permissible_income_ratio=Decimal("0.05"),
    )
    CompliancePolicy(**base)  # la norme elle-même est acceptée
    with pytest.raises(ComplianceError):
        CompliancePolicy(**{**base, **change})


def test_zero_tolerance_requires_exclusion_of_missing_data():
    with pytest.raises(ComplianceError):
        CompliancePolicy(missing_non_permissible_income="activity_screen")


def test_unknown_profile_rejected():
    with pytest.raises(ComplianceError):
        CompliancePolicy(profile="laxiste")


def test_zero_tolerance_rejects_any_positive_threshold():
    with pytest.raises(ComplianceError, match="tolérance zéro"):
        CompliancePolicy(max_interest_bearing_debt_ratio=Decimal("0.01"))


def test_zero_tolerance_requires_finance_leases_counted_as_debt():
    with pytest.raises(ComplianceError):
        CompliancePolicy(count_finance_leases_as_debt=False)


@pytest.mark.parametrize(
    "bad_type", [InstrumentType.BOND, InstrumentType.CFD, InstrumentType.PREFERRED_STOCK, InstrumentType.STABLECOIN]
)
def test_never_allowed_instrument_types_cannot_be_configured(bad_type):
    with pytest.raises(ComplianceError):
        CompliancePolicy(allowed_instrument_types=frozenset({InstrumentType.COMMON_STOCK, bad_type}))


def test_stablecoin_quote_currency_rejected():
    with pytest.raises(ComplianceError, match="Stablecoin"):
        CompliancePolicy(allowed_quote_currencies=frozenset({"EUR", "USDC"}))


def test_crypto_whitelist_requires_approver_and_no_stablecoin():
    with pytest.raises(ComplianceError):
        CompliancePolicy(crypto_whitelist=(CryptoApproval("BTC", approved_by=" "),))
    with pytest.raises(ComplianceError):
        CompliancePolicy(crypto_whitelist=(CryptoApproval("USDT", approved_by="x"),))
