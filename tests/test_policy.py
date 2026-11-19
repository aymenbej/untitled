from decimal import Decimal

import pytest

from halal_trading.compliance import ComplianceError, CompliancePolicy, CryptoApproval, load_policy
from halal_trading.domain import InstrumentType

from .conftest import CONFIG


def test_config_file_loads_as_zero_tolerance():
    policy = load_policy(CONFIG)
    assert policy.profile == "zero_tolerance"
    assert all(v == 0 for v in policy.thresholds.values())
    assert policy.allowed_instrument_types == {InstrumentType.COMMON_STOCK, InstrumentType.CRYPTO_SPOT}
    assert policy.crypto_whitelist == ()  # rien d'autorisé sans validation explicite
    assert "alcool" in policy.prohibited_activities


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
