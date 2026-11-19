"""Tests de la reconstitution point-in-time à partir de faits EDGAR simulés."""

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from halal_trading.compliance import screen_company
from halal_trading.data.edgar import facts_to_frame, user_agent
from halal_trading.data.pit import PitCompany, fundamental_features, market_cap, split_factor_after, to_financials

from .conftest import clean_company


def fact(start, end, val, filed, form="10-K"):
    return {"start": start, "end": end, "val": val, "filed": filed, "form": form, "fy": 2020, "fp": "FY"}


def company_facts():
    """Société fictive : CA 2019 et 2020, retraitement du CA 2020 publié en 2022,
    changement de nom du poste « chiffre d'affaires » en 2021."""
    usd = lambda *rows: {"units": {"USD": list(rows)}}  # noqa: E731
    return {"facts": {
        "us-gaap": {
            "Revenues": usd(
                fact("2019-01-01", "2019-12-31", 1000, "2020-02-20"),
                fact("2020-01-01", "2020-12-31", 1200, "2021-02-20"),
                fact("2020-01-01", "2020-12-31", 1150, "2022-02-20"),  # retraitement publié en 2022
            ),
            "RevenueFromContractWithCustomerExcludingAssessedTax": usd(
                fact("2021-01-01", "2021-12-31", 1500, "2022-02-20"),
            ),
            "NetIncomeLoss": usd(fact("2020-01-01", "2020-12-31", 120, "2021-02-20")),
            "InvestmentIncomeInterest": usd(fact("2020-01-01", "2020-12-31", 30, "2021-02-20")),
            # trimestriel : ne doit pas être pris pour une valeur annuelle
            "InterestExpense": usd(fact("2020-10-01", "2020-12-31", 99, "2021-02-20", "10-Q")),
            "LongTermDebt": usd(fact(None, "2020-12-31", 400, "2021-02-20")),
            "StockholdersEquity": usd(fact(None, "2020-12-31", 800, "2021-02-20")),
            "Assets": usd(fact(None, "2020-12-31", 2000, "2021-02-20")),
            "Liabilities": usd(fact(None, "2020-12-31", 1200, "2021-02-20")),
            "CashAndCashEquivalentsAtCarryingValue": usd(fact(None, "2020-12-31", 300, "2021-02-20")),
        },
        "dei": {
            "EntityCommonStockSharesOutstanding": {"units": {"shares": [
                {"end": "2021-01-31", "val": 100, "filed": "2021-02-20", "form": "10-K"},
            ]}},
        },
    }}


@pytest.fixture
def pit():
    return PitCompany(facts_to_frame(company_facts()))


def test_nothing_known_before_first_filing(pit):
    assert pit.snapshot("2020-02-19").get("revenue") is None
    assert pit.snapshot("2020-02-20").get("revenue") == 1000


def test_uses_original_figure_until_restatement_is_filed(pit):
    assert pit.snapshot("2021-06-30").get("revenue") == 1200  # chiffre d'origine
    snap = pit.snapshot("2022-01-31")
    assert snap.get("revenue") == 1200  # le retraitement n'est pas encore publié


def test_renamed_concept_is_followed(pit):
    snap = pit.snapshot("2022-03-01")
    assert snap.get("revenue") == 1500  # nouvelle période, nouveau nom de poste
    assert snap.values["revenue"].tag.startswith("RevenueFromContract")
    assert snap.prior_revenue.value == 1150  # année précédente, version retraitée connue en 2022


def test_quarterly_values_are_not_taken_as_annual(pit):
    assert pit.snapshot("2021-06-30").get("interest_expense") is None


def test_stale_figures_are_dropped(pit):
    assert pit.snapshot("2024-06-30").get("net_income") is None  # dernier chiffre de 2020 : trop ancien


def test_market_cap_aligns_shares_with_split_adjusted_prices(pit):
    snap = pit.snapshot("2021-06-30")
    splits = pd.Series([0.0, 4.0], index=pd.to_datetime(["2021-01-01", "2022-07-01"]))
    assert split_factor_after(splits, pd.Timestamp("2021-01-31")) == 4.0
    # 100 actions réelles ; cours ajusté de la division 4:1 = 25 → capitalisation réelle 100 × 100
    assert market_cap(snap, 25.0, splits) == pytest.approx(10_000)


def test_financials_feed_the_halal_screen(pit, aaoifi_policy):
    snap = pit.snapshot("2021-06-30")
    fin = to_financials(snap, mcap=10_000)
    assert fin.total_revenue == Decimal("1200")
    assert fin.interest_bearing_debt == Decimal("400")
    assert fin.interest_bearing_investments == Decimal("300")  # majoré par toute la trésorerie
    assert any("trésorerie" in n for n in fin.notes)
    result = screen_company(clean_company(financials=fin), aaoifi_policy, today=snap.as_of.date())
    # revenus d'intérêts 30 / CA 1200 = 2,5 % < 5 % ; dette 4 % ; placements 3 % : conforme
    assert result.compliant, result.reasons
    assert result.impure_income_ratio == Decimal("0.025")


def test_debt_falls_back_to_total_liabilities():
    raw = company_facts()
    del raw["facts"]["us-gaap"]["LongTermDebt"]
    fin = to_financials(PitCompany(facts_to_frame(raw)).snapshot("2021-06-30"), mcap=10_000)
    assert fin.interest_bearing_debt == Decimal("1200")
    assert any("passifs" in n for n in fin.notes)


def test_fundamental_features(pit):
    feats = fundamental_features(pit.snapshot("2022-03-01"), mcap=10_000)
    assert feats["sales_yield"] == pytest.approx(0.15)
    assert feats["revenue_growth"] == pytest.approx(1500 / 1150 - 1)
    # le résultat net 2021 n'est pas publié : on utilise le dernier connu (2020), 120 / 10 000
    assert feats["earnings_yield"] == pytest.approx(0.012)
    assert feats["roe"] == pytest.approx(120 / 800)


def test_user_agent_requires_contact(monkeypatch):
    monkeypatch.delenv("HALAL_EDGAR_CONTACT", raising=False)
    with pytest.raises(RuntimeError):
        user_agent()
    monkeypatch.setenv("HALAL_EDGAR_CONTACT", "contact@example.org")
    assert user_agent().endswith("contact@example.org")
