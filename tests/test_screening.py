from datetime import date
from decimal import Decimal

import pytest

from halal_trading.compliance import CryptoAsset, screen_company, screen_crypto
from halal_trading.domain import InstrumentType

from .conftest import TODAY, clean_company, clean_financials


def test_clean_company_is_compliant(policy):
    result = screen_company(clean_company(), policy, today=TODAY)
    assert result.compliant, result.reasons
    assert result.status == "CONFORME"


@pytest.mark.parametrize(
    "field",
    [
        "interest_bearing_debt",
        "finance_lease_liabilities",
        "interest_bearing_investments",
        "interest_income",
        "interest_expense",
        "non_permissible_income",
    ],
)
def test_any_interest_amount_excludes_company(policy, field):
    company = clean_company(financials=clean_financials(**{field: Decimal("1")}))
    result = screen_company(company, policy, today=TODAY)
    assert not result.compliant
    assert len(result.reasons) == 1


def test_operating_leases_not_counted_by_default(policy):
    company = clean_company(financials=clean_financials(operating_lease_liabilities=Decimal("5000000")))
    assert screen_company(company, policy, today=TODAY).compliant


@pytest.mark.parametrize("field", ["interest_income", "interest_bearing_debt", "market_cap", "total_revenue"])
def test_missing_data_excludes_company(policy, field):
    company = clean_company(financials=clean_financials(**{field: None}))
    result = screen_company(company, policy, today=TODAY)
    assert not result.compliant
    assert any("manquantes" in r for r in result.reasons)


def test_no_financials_excludes_company(policy):
    assert not screen_company(clean_company(financials=None), policy, today=TODAY).compliant


def test_stale_financials_exclude_company(policy):
    company = clean_company(financials=clean_financials(period_end=date(2025, 1, 1)))
    result = screen_company(company, policy, today=TODAY)
    assert any("trop anciennes" in r for r in result.reasons)


@pytest.mark.parametrize(
    "sector,industry",
    [
        ("Consumer Defensive", "Beverages - Wineries & Distilleries"),
        ("Consumer Defensive", "Beverages - Brewers"),
        ("Financial Services", "Banks - Diversified"),
        ("Financial Services", "Insurance - Life"),
        ("Consumer Cyclical", "Gambling"),
        ("Consumer Cyclical", "Resorts & Casinos"),
        ("Consumer Defensive", "Tobacco"),
        ("Industrials", "Aerospace & Defense"),
        ("Financial Services", "Credit Services"),
    ],
)
def test_prohibited_sectors_excluded(policy, sector, industry):
    result = screen_company(clean_company(sector=sector, industry=industry), policy, today=TODAY)
    assert not result.compliant
    assert any("activité interdite" in r for r in result.reasons)


def test_prohibited_revenue_segment_excluded(policy):
    company = clean_company(
        sector="Consumer Defensive",
        industry="Packaged Foods",
        revenue_segments={"Volaille": Decimal("900"), "Pork products": Decimal("100")},
    )
    result = screen_company(company, policy, today=TODAY)
    assert any("porc" in r for r in result.reasons)


def test_no_false_positive_on_partial_words(policy):
    # « pig » ne doit pas exclure « pigment », ni « ham » « Hamburg », ni « vin » « Vinci »
    company = clean_company(industry="Specialty Chemicals", business_activities=("pigments", "Hamburg plant", "Vinci"))
    assert screen_company(company, policy, today=TODAY).compliant


def test_unknown_activity_excluded(policy):
    company = clean_company(sector=None, industry=None, business_activities=())
    assert not screen_company(company, policy, today=TODAY).compliant


@pytest.mark.parametrize("itype", [InstrumentType.PREFERRED_STOCK, InstrumentType.ETF, InstrumentType.BOND])
def test_non_common_stock_excluded(policy, itype):
    assert not screen_company(clean_company(instrument_type=itype), policy, today=TODAY).compliant


def test_whitelisted_crypto_compliant(policy):
    btc = CryptoAsset("BTC", "Bitcoin", categories=frozenset({"currency"}), interest_bearing=False)
    assert screen_crypto(btc, policy).compliant


def test_crypto_not_whitelisted_excluded(policy):
    eth = CryptoAsset("ETH", "Ether", categories=frozenset({"platform"}), interest_bearing=False)
    result = screen_crypto(eth, policy)
    assert any("liste blanche" in r for r in result.reasons)


@pytest.mark.parametrize(
    "asset",
    [
        CryptoAsset("BTC", "x", categories=frozenset({"currency", "lending"}), interest_bearing=False),
        CryptoAsset("BTC", "x", categories=frozenset({"currency"}), interest_bearing=True),
        CryptoAsset("BTC", "x", categories=frozenset({"currency"}), interest_bearing=None),
        CryptoAsset("BTC", "x", categories=frozenset(), interest_bearing=False),
        CryptoAsset("USDC", "x", categories=frozenset({"currency"}), interest_bearing=False),
    ],
)
def test_crypto_fail_closed(policy, asset):
    assert not screen_crypto(asset, policy).compliant


@pytest.mark.parametrize("text", ["Beverages - Non-Alcoholic", "boissons sans alcool", "alcohol-free drinks"])
def test_negated_keywords_are_not_prohibited(policy, text):
    assert screen_company(clean_company(industry=text), policy, today=TODAY).compliant


def test_business_summary_catches_activity_hidden_by_sector(policy):
    summary = "The company operates through Wines and Spirits, Fashion and Leather Goods, Perfumes segments."
    company = clean_company(industry="Luxury Goods", business_activities=(summary,))
    result = screen_company(company, policy, today=TODAY)
    assert any("alcool" in r for r in result.reasons)


def test_known_ratios_still_reported_when_other_data_missing(policy):
    company = clean_company(
        financials=clean_financials(non_permissible_income=None, interest_bearing_debt=Decimal("1000"))
    )
    reasons = screen_company(company, policy, today=TODAY).reasons
    assert any("manquantes" in r for r in reasons)
    assert any("dette à intérêt" in r for r in reasons)
