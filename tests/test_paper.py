"""Tests du portefeuille virtuel : exécution, sorties, frais, conversion, conformité."""

from decimal import Decimal

import pytest

from halal_trading.compliance import ApprovedUniverse, AuditLog, OrderGuard, screen_company
from halal_trading.paper.engine import FeeModel, PaperConfig, PaperEngine, PaperState

from .conftest import TODAY, clean_company

FX = {"USD": 1.25}  # 1 € = 1,25 $


@pytest.fixture
def engine(policy):
    universe = ApprovedUniverse(max_age_days=policy.max_screening_age_days)
    for s in ("AAA", "BBB"):
        universe.add(screen_company(clean_company(symbol=s), policy, today=TODAY))
    guard = OrderGuard(policy, universe, AuditLog())
    cfg = PaperConfig(capital_eur=500, max_positions=2, horizon_days=3, tp_mult=1.0, sl_mult=1.0,
                      fees=FeeModel(min_fee=1.0, rate=0.0005, slippage=0.0, fx_rate=0.0))
    return PaperEngine(cfg, guard)


def bar(o, h, l, c, cur="USD"):
    return {"open": o, "high": h, "low": l, "close": c, "currency": cur}


def ranking(*symbols, vol=0.05):
    return [(s, "USD", vol / 3 ** 0.5, 1.0) for s in symbols]  # objectif/stop = ±5 %


def test_decision_then_fill_at_next_open(engine):
    st = PaperState.new("2026-01-05", 500)
    engine.process_day(st, "2026-01-05", {"AAA": bar(10, 10, 10, 10)}, FX, ranking("AAA"))
    assert len(st.pending) == 1 and not st.positions  # décidé à la clôture, pas encore exécuté
    engine.process_day(st, "2026-01-06", {"AAA": bar(10, 10.1, 9.9, 10)}, FX, None)
    pos = st.positions[0]
    assert pos.entry_price == 10
    assert pos.cost_eur == pytest.approx(250)  # 500 € / 2 positions
    # 250 € − 1 € de frais minimum = 249 € convertis en 311,25 $, soit 31,125 actions à 10 $
    assert pos.quantity == pytest.approx(31.125)
    assert st.cash_eur == pytest.approx(250)


def test_take_profit_and_fees(engine):
    st = PaperState.new("2026-01-05", 500)
    engine.process_day(st, "2026-01-05", {"AAA": bar(10, 10, 10, 10)}, FX, ranking("AAA"))
    engine.process_day(st, "2026-01-06", {"AAA": bar(10, 10.1, 9.9, 10)}, FX, None)
    engine.process_day(st, "2026-01-07", {"AAA": bar(10.2, 10.6, 10.1, 10.5)}, FX, None)
    trade = st.trades[0]
    assert trade["motif"] == "objectif" and trade["prix_sortie"] == pytest.approx(10.5)
    gross = 31.125 * 10.5 / 1.25
    assert trade["recupere_eur"] == pytest.approx(gross - 1.0, abs=0.01)  # frais minimum 1 €
    assert not st.positions


def test_stop_has_priority_and_timeout(engine):
    st = PaperState.new("2026-01-05", 500)
    engine.process_day(st, "2026-01-05", {"AAA": bar(10, 10, 10, 10), "BBB": bar(20, 20, 20, 20)}, FX,
                       ranking("AAA", "BBB"))
    engine.process_day(st, "2026-01-06", {"AAA": bar(10, 11, 9, 10), "BBB": bar(20, 20.1, 19.9, 20)}, FX, None)
    assert [t["motif"] for t in st.trades] == ["stop"]  # AAA : objectif et stop le même jour -> stop
    for d in ("2026-01-07", "2026-01-08"):
        engine.process_day(st, d, {"BBB": bar(20, 20.1, 19.9, 20)}, FX, None)
    assert st.trades[-1]["motif"] == "échéance"  # BBB : 3 séances sans toucher objectif ni stop


def test_non_compliant_symbol_is_blocked(engine):
    st = PaperState.new("2026-01-05", 500)
    engine.process_day(st, "2026-01-05", {"ZZZ": bar(10, 10, 10, 10)}, FX, [("ZZZ", "USD", 0.03, 1.0)])
    engine.process_day(st, "2026-01-06", {"ZZZ": bar(10, 10, 10, 10)}, FX, None)
    assert not st.positions
    assert st.events[0]["type"] == "ordre bloqué (conformité)"
    assert st.cash_eur == pytest.approx(500)


def test_state_roundtrip(engine):
    st = PaperState.new("2026-01-05", 500)
    engine.process_day(st, "2026-01-05", {"AAA": bar(10, 10, 10, 10)}, FX, ranking("AAA"))
    again = PaperState.from_dict(st.to_dict())
    assert again.to_dict() == st.to_dict()
    assert again.journal[0]["valeur_eur"] == pytest.approx(500)
