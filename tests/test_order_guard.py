from dataclasses import replace
from datetime import timedelta
from decimal import Decimal

import pytest

from halal_trading.compliance import ApprovedUniverse, ComplianceViolation, OrderGuard, screen_company
from halal_trading.compliance.screening import ScreeningResult
from halal_trading.domain import AccountState, Instrument, InstrumentType, OrderRequest, OrderType, Side

from .conftest import TODAY, clean_company

STOCK = Instrument("HLL", InstrumentType.COMMON_STOCK, "EUR")


@pytest.fixture
def universe(policy):
    u = ApprovedUniverse(max_age_days=policy.max_screening_age_days)
    u.add(screen_company(clean_company(), policy, today=TODAY))
    return u


@pytest.fixture
def guard(policy, universe, audit):
    return OrderGuard(policy, universe, audit)


@pytest.fixture
def account():
    return AccountState(cash={"EUR": Decimal("1000")}, positions={"HLL": Decimal("10")})


def buy(qty="5", price="100", fee="1", **kw):
    return OrderRequest(
        instrument=kw.pop("instrument", STOCK),
        side=Side.BUY,
        quantity=Decimal(qty),
        estimated_price=Decimal(price),
        estimated_fee=Decimal(fee),
        order_type=kw.pop("order_type", OrderType.LIMIT),
        limit_price=kw.pop("limit_price", Decimal(price)),
        **kw,
    )


def sell(qty="5", price="100", fee="1"):
    return OrderRequest(STOCK, Side.SELL, Decimal(qty), Decimal(price), Decimal(fee), limit_price=Decimal(price))


def test_valid_buy_and_sell_pass_and_are_audited(guard, account, audit):
    guard.check(buy(), account)
    guard.check(sell(), account)
    assert [e["decision"] for e in audit.events] == ["AUTORISÉ", "AUTORISÉ"]


def test_buy_exceeding_cash_including_fees_blocked(guard, account, audit):
    with pytest.raises(ComplianceViolation, match="liquidités insuffisantes"):
        guard.check(buy(qty="10", price="100", fee="0.01"), account)  # 1000.01 > 1000
    assert audit.events[-1]["decision"] == "BLOQUÉ"


def test_market_buy_keeps_price_buffer(guard, account):
    # 9 x 100 = 900 + 2 % de marge = 918 + 1 frais -> OK ; 9.8 x 100 x 1.02 > 1000 -> bloqué
    guard.check(buy(qty="9", order_type=OrderType.MARKET, limit_price=None), account)
    with pytest.raises(ComplianceViolation):
        guard.check(buy(qty="9.8", order_type=OrderType.MARKET, limit_price=None), account)


def test_reserved_cash_is_not_available(guard, account):
    account.reserved_cash["EUR"] = Decimal("600")
    with pytest.raises(ComplianceViolation):
        guard.check(buy(qty="5"), account)


def test_short_selling_blocked(guard, account):
    with pytest.raises(ComplianceViolation, match="découvert"):
        guard.check(sell(qty="11"), account)


def test_selling_quantity_already_in_open_order_blocked(guard, account):
    account.reserved_positions["HLL"] = Decimal("8")
    with pytest.raises(ComplianceViolation):
        guard.check(sell(qty="5"), account)


def test_margin_and_leverage_blocked(guard, account):
    with pytest.raises(ComplianceViolation, match="marge") as exc:
        guard.check(buy(margin=True, leverage=Decimal("2")), account)
    assert any("levier" in r for r in exc.value.reasons)


@pytest.mark.parametrize("itype", [InstrumentType.CFD, InstrumentType.OPTION, InstrumentType.BOND])
def test_forbidden_instrument_types_blocked(guard, account, itype):
    inst = Instrument("HLL", itype, "EUR")
    with pytest.raises(ComplianceViolation, match="type d'instrument interdit"):
        guard.check(buy(instrument=inst), account)


def test_stablecoin_quote_blocked(guard, account):
    inst = Instrument("HLL", InstrumentType.COMMON_STOCK, "USDC")
    with pytest.raises(ComplianceViolation, match="stablecoin"):
        guard.check(buy(instrument=inst), account)


def test_unscreened_symbol_blocked(guard, account):
    inst = Instrument("XYZ", InstrumentType.COMMON_STOCK, "EUR")
    with pytest.raises(ComplianceViolation, match="jamais été filtré"):
        guard.check(buy(instrument=inst), account)


def test_excluded_symbol_blocked(policy, guard, universe, account):
    universe.add(screen_company(clean_company(industry="Banks - Regional"), policy, today=TODAY))
    with pytest.raises(ComplianceViolation, match="exclu"):
        guard.check(buy(), account)


def test_stale_screening_blocked(guard, universe, account):
    result: ScreeningResult = universe._results["HLL"]
    universe.add(replace(result, screened_at=result.screened_at - timedelta(days=101)))
    with pytest.raises(ComplianceViolation, match="périmé"):
        guard.check(buy(), account)


@pytest.mark.parametrize("qty,price,fee", [("0", "100", "1"), ("1", "0", "1"), ("1", "100", "-1")])
def test_invalid_numbers_blocked(guard, account, qty, price, fee):
    with pytest.raises(ComplianceViolation):
        guard.check(buy(qty=qty, price=price, fee=fee), account)
