from dataclasses import replace
from datetime import timedelta
from decimal import Decimal

import pytest

from halal_trading.compliance import (
    ComplianceViolation,
    LedgerMonitor,
    PurificationRegister,
    account_violations,
    check_account,
    dividend_purification_amount,
)
from halal_trading.domain import AccountSettings, AccountType, Transaction, TransactionType

from .conftest import utc

NOW = utc(2026, 11, 19)


@pytest.fixture
def settings():
    return AccountSettings(
        broker="ibkr",
        account_type=AccountType.CASH,
        margin_enabled=False,
        securities_lending_enabled=False,
        cash_interest_enabled=False,
        staking_enabled=False,
        yield_rewards_enabled=False,
        source="manual",
        confirmed_at=NOW - timedelta(days=1),
        confirmed_by="vérificateur",
    )


def test_clean_cash_account_passes(policy, audit, settings):
    check_account(settings, policy, audit, now=NOW)
    assert audit.events[-1]["decision"] == "AUTORISÉ"


@pytest.mark.parametrize(
    "change",
    [
        {"account_type": AccountType.MARGIN},
        {"account_type": AccountType.UNKNOWN},
        {"margin_enabled": True},
        {"securities_lending_enabled": True},
        {"cash_interest_enabled": True},
        {"cash_interest_enabled": None},  # inconnu = refus
        {"staking_enabled": True},
        {"yield_rewards_enabled": True},
        {"confirmed_by": None},
        {"confirmed_at": NOW - timedelta(days=91)},
    ],
)
def test_any_interest_related_setting_blocks_startup(policy, audit, settings, change):
    bad = replace(settings, **change)
    assert account_violations(bad, policy, now=NOW)
    with pytest.raises(ComplianceViolation):
        check_account(bad, policy, audit, now=NOW)


def tx(id_, type_, amount, description="", symbol=None, currency="EUR"):
    return Transaction(id_, NOW, type_, Decimal(amount), currency, description, symbol)


@pytest.fixture
def monitor(audit):
    return LedgerMonitor(PurificationRegister(), audit)


def test_normal_activity_does_not_halt(monitor):
    report = monitor.scan(
        [
            tx("1", TransactionType.DEPOSIT, "1000"),
            tx("2", TransactionType.TRADE, "-500", "Buy 5 HLL"),
            tx("3", TransactionType.FEE, "-1", "Commission"),
            tx("4", TransactionType.DIVIDEND, "3", symbol="HLL"),
        ],
        dividend_impure_ratios={"HLL": Decimal("0")},
    )
    assert not report.halt_trading
    assert monitor.purification.entries == []


@pytest.mark.parametrize(
    "type_",
    [
        TransactionType.INTEREST_CREDIT,
        TransactionType.STAKING_REWARD,
        TransactionType.LENDING_INCOME,
        TransactionType.YIELD_REWARD,
    ],
)
def test_interest_income_is_fully_purified_and_halts(monitor, type_):
    report = monitor.scan([tx("9", type_, "2.50")])
    assert report.halt_trading
    assert monitor.purification.total_pending() == {"EUR": Decimal("2.50")}


def test_interest_paid_halts(monitor):
    report = monitor.scan([tx("9", TransactionType.INTEREST_DEBIT, "-0.80")])
    assert report.halt_trading
    assert report.alerts[0].severity == "critique"


@pytest.mark.parametrize("desc", ["Credit Interest for Oct", "USDC Rewards", "SYEP income", "Intérêts créditeurs"])
def test_suspicious_labels_require_review(monitor, desc):
    report = monitor.scan([tx("9", TransactionType.OTHER, "1", desc)])
    assert report.halt_trading
    assert report.alerts[0].severity == "à_revoir"


def test_dividend_without_known_ratio_requires_review(monitor):
    assert monitor.scan([tx("9", TransactionType.DIVIDEND, "3", symbol="HLL")]).halt_trading


def test_dividend_of_company_that_lost_compliance_is_purified(monitor):
    report = monitor.scan(
        [tx("9", TransactionType.DIVIDEND, "10", symbol="HLL")],
        dividend_impure_ratios={"HLL": Decimal("0.03")},
    )
    assert report.halt_trading
    assert monitor.purification.total_pending() == {"EUR": Decimal("0.30")}


def test_same_transaction_purified_only_once(monitor):
    t = tx("9", TransactionType.INTEREST_CREDIT, "1")
    monitor.scan([t])
    monitor.scan([t])
    assert len(monitor.purification.entries) == 1


def test_dividend_purification_ratio_bounds():
    with pytest.raises(ValueError):
        dividend_purification_amount(Decimal("10"), Decimal("1.5"))
