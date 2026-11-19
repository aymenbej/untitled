"""Tests du module de collecte, sans réseau (sources simulées)."""

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from halal_trading.compliance.screening import CompanyProfile
from halal_trading.data.coinbase import CoinbaseClient
from halal_trading.data.pipeline import DataConfig, screen_stocks, update_crypto, update_stocks
from halal_trading.data.quality import check_prices
from halal_trading.data.reviews import ManualReview, apply_review, load_reviews
from halal_trading.data.storage import ParquetStore, normalize_prices
from halal_trading.data.yahoo import fetch_company, fetch_stock_prices, financials_from_statements
from halal_trading.domain import InstrumentType

from .conftest import CONFIG, TODAY, clean_company, clean_financials


# ------------------------------------------------------------------ helpers
def price_frame(days: int, start: str = "2024-01-01", step: int = 1) -> pd.DataFrame:
    dates = pd.date_range(start, periods=days, freq=f"{step}D", tz="UTC")
    close = np.linspace(100, 120, days)
    return normalize_prices(
        pd.DataFrame(
            {"date": dates, "open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1000.0}
        ),
        "test",
    )


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload, self.status_code = payload, status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeCoinbaseSession:
    """Simule l'API : renvoie une bougie par jour de la fenêtre demandée, du plus récent au plus ancien."""

    def __init__(self):
        self.calls = []

    def get(self, url, params, timeout):
        self.calls.append(params)
        start = datetime.fromisoformat(params["start"])
        end = datetime.fromisoformat(params["end"])
        rows, day = [], end
        while day >= start:
            ts = int(day.timestamp())
            rows.append([ts, 99.0, 102.0, 100.0, 101.0, 5.0])
            day -= timedelta(days=1)
        assert len(rows) <= 300, "l'API refuse plus de 300 bougies"
        return FakeResponse(rows)


# ------------------------------------------------------------------ coinbase
def test_coinbase_paginates_without_gaps_or_duplicates():
    session = FakeCoinbaseSession()
    client = CoinbaseClient(session=session, pause=0)
    df = client.daily_candles("BTC-EUR", date(2023, 1, 1), date(2024, 12, 31))
    assert len(df) == (date(2024, 12, 31) - date(2023, 1, 1)).days + 1
    assert df["date"].is_monotonic_increasing and not df["date"].duplicated().any()
    assert len(session.calls) == 3
    assert list(df.columns)[:6] == ["date", "open", "high", "low", "close", "adj_close"]


def test_coinbase_retries_on_rate_limit():
    class Flaky(FakeCoinbaseSession):
        def get(self, url, params, timeout):
            if not self.calls:
                self.calls.append("429")
                return FakeResponse([], status=429)
            return super().get(url, params, timeout)

    df = CoinbaseClient(session=Flaky(), pause=0).daily_candles("BTC-EUR", date(2024, 1, 1), date(2024, 1, 10))
    assert len(df) == 10


# ------------------------------------------------------------------ stockage
def test_store_merges_incrementally(tmp_path):
    store = ParquetStore(tmp_path)
    store.write_prices("crypto", "BTC-EUR", price_frame(10))
    update = price_frame(5, start="2024-01-08")  # 3 jours de chevauchement
    update.loc[:, "close"] = 999.0
    merged = store.write_prices("crypto", "BTC-EUR", update)
    assert len(merged) == 12
    assert merged.loc[merged["date"] == pd.Timestamp("2024-01-08", tz="UTC"), "close"].item() == 999.0
    assert store.last_date("crypto", "BTC-EUR") == date(2024, 1, 12)
    assert store.list_symbols("crypto") == ["BTC-EUR"]


def test_update_crypto_resumes_from_last_date(tmp_path):
    store = ParquetStore(tmp_path)
    today = datetime.now(timezone.utc).date()
    config = DataConfig(tmp_path, ["BTC-EUR"], today - timedelta(days=20), [], date(2020, 1, 1))
    session = FakeCoinbaseSession()
    update_crypto(config, store, CoinbaseClient(session=session, pause=0))
    session.calls.clear()
    reports = update_crypto(config, store, CoinbaseClient(session=session, pause=0))
    assert reports[0].rows == 21 and reports[0].error is None
    first_requested = datetime.fromisoformat(session.calls[-1]["start"]).date()
    assert first_requested == today - timedelta(days=5)  # reprise incrémentale


def test_update_stocks_refetches_full_history_on_new_split(tmp_path):
    store = ParquetStore(tmp_path)
    config = DataConfig(tmp_path, [], date(2024, 1, 1), ["HLL"], date(2024, 1, 1))
    store.write_prices("stocks", "HLL", price_frame(10))
    calls = []

    def fetch(symbol, start):
        calls.append(start)
        df = price_frame(15)
        df.loc[12, "splits"] = 2.0  # division après la dernière date connue
        return df

    update_stocks(config, store, fetch=fetch)
    assert calls == [date(2024, 1, 5), date(2024, 1, 1)]


def test_update_isolates_failures(tmp_path):
    store = ParquetStore(tmp_path)
    config = DataConfig(tmp_path, [], date(2024, 1, 1), ["OK", "KO"], date(2024, 1, 1))

    def fetch(symbol, start):
        if symbol == "KO":
            raise RuntimeError("source indisponible")
        return price_frame(5)

    reports = update_stocks(config, store, fetch=fetch)
    assert reports[0].error is None and "indisponible" in reports[1].error


# ------------------------------------------------------------------ qualité
def test_clean_series_has_no_issue():
    assert check_prices(price_frame(30), "crypto") == []


def test_quality_detects_problems():
    df = price_frame(30)
    df.loc[3, "high"] = 1.0  # bougie incohérente
    df.loc[5, "close"] = -1.0  # prix négatif
    df.loc[10, "adj_close"] = df.loc[9, "adj_close"] * 3  # saut extrême
    df = df.drop(index=[20, 21, 22])  # trou de cotation
    messages = " | ".join(i.message for i in check_prices(df, "crypto"))
    for expected in ("incohérentes", "négatifs", "extrêmes", "trous"):
        assert expected in messages


def test_quality_tolerates_weekends_for_stocks():
    business_days = price_frame(60)
    business_days = business_days[business_days["date"].dt.dayofweek < 5]
    assert check_prices(business_days, "stocks") == []
    assert any("trous" in i.message for i in check_prices(business_days, "crypto"))


def test_empty_series_is_an_error():
    assert check_prices(None, "stocks")[0].severity == "erreur"


# ------------------------------------------------------------------ yahoo
def statements(**balance_rows):
    col = pd.Timestamp("2026-06-30")
    income = pd.DataFrame(
        {col: {"Total Revenue": 1_000_000.0, "Interest Income": 0.0, "Interest Expense": 0.0}}
    )
    balance_defaults = {
        "Total Debt": 0.0,
        "Cash Equivalents": 0.0,
        "Other Short Term Investments": 0.0,
    }
    balance_defaults.update(balance_rows)
    balance = pd.DataFrame({col: balance_defaults})
    return {"marketCap": 5e7}, income, balance


def test_yahoo_mapping_debt_free_company():
    fin = financials_from_statements(*statements())
    assert fin.interest_bearing_debt == 0 and fin.finance_lease_liabilities == 0
    assert fin.interest_bearing_investments == 0
    assert fin.non_permissible_income is None  # jamais fourni par Yahoo
    assert fin.period_end == date(2026, 6, 30)


def test_yahoo_mapping_treats_all_leases_as_financing():
    fin = financials_from_statements(*statements(**{"Total Debt": 500.0, "Capital Lease Obligations": 500.0}))
    assert fin.interest_bearing_debt == 0
    assert fin.finance_lease_liabilities == 500


def test_yahoo_missing_rows_are_unknown_not_zero():
    info, income, balance = statements()
    balance = balance.drop(index=["Total Debt", "Cash Equivalents"])
    income.loc["Interest Income"] = np.nan
    fin = financials_from_statements(info, income, balance)
    assert fin.interest_bearing_debt is None
    assert fin.interest_bearing_investments is None
    assert fin.interest_income is None


def test_fetch_prices_and_company_with_fake_ticker():
    info, income, balance = statements()

    class FakeTicker:
        def __init__(self, symbol):
            self.info = {**info, "quoteType": "EQUITY", "longName": "Halal Soft", "sector": "Technology",
                         "industry": "Software - Application"}
            self.income_stmt = income
            self.quarterly_balance_sheet = balance
            self.balance_sheet = balance

        def history(self, **kwargs):
            idx = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=3, tz="America/New_York"), name="Date")
            return pd.DataFrame(
                {"Open": [1.0, 2, 3], "High": [1.0, 2, 3], "Low": [1.0, 2, 3], "Close": [1.0, 2, 3],
                 "Adj Close": [0.9, 1.9, 2.9], "Volume": [10, 10, 10], "Dividends": [0.0, 0, 0],
                 "Stock Splits": [0.0, 0, 0]},
                index=idx,
            )

    prices = fetch_stock_prices("HLL", date(2024, 1, 1), ticker_factory=FakeTicker)
    assert len(prices) == 3 and prices["adj_close"].iloc[0] == 0.9

    profile, raw = fetch_company("HLL", ticker_factory=FakeTicker)
    assert profile.instrument_type == InstrumentType.COMMON_STOCK
    assert raw["income_statement"]["2026-06-30"]["Total Revenue"] == 1_000_000.0


# ------------------------------------------------------------------ vérifications manuelles
def review(**fields):
    return ManualReview("HLL", "vérificateur", TODAY, "Rapport annuel 2025", {k: Decimal(v) for k, v in fields.items()})


def test_review_fills_only_missing_fields():
    fin = clean_financials(non_permissible_income=None, interest_income=Decimal("5"))
    filled, names = apply_review(fin, review(non_permissible_income="0", interest_income="0"), TODAY, 400)
    assert names == ["non_permissible_income"]
    assert filled.non_permissible_income == 0
    assert filled.interest_income == 5  # la donnée de la source n'est jamais écrasée


def test_stale_review_is_ignored():
    fin = clean_financials(non_permissible_income=None)
    old = ManualReview("HLL", "x", TODAY - timedelta(days=500), "src", {"non_permissible_income": Decimal("0")})
    assert apply_review(fin, old, TODAY, 400)[0].non_permissible_income is None


def test_reviews_file_requires_author_and_source(tmp_path):
    path = tmp_path / "r.yaml"
    path.write_text("reviews:\n  - {symbol: HLL, reviewed_by: '', reviewed_at: 2026-11-19, source: x, fields: {}}\n")
    with pytest.raises(ValueError):
        load_reviews(path)
    path.write_text("reviews:\n  - {symbol: HLL, reviewed_by: a, reviewed_at: 2026-11-19, source: x, fields: {foo: 1}}\n")
    with pytest.raises(ValueError):
        load_reviews(path)


def test_shipped_config_files_load():
    from halal_trading.data.pipeline import load_data_config

    root = CONFIG.parent
    assert load_reviews(root / "manual_reviews.yaml") == {}
    cfg = load_data_config(root / "data_sources.yaml")
    assert "BTC-EUR" in cfg.crypto_products and "MSFT" in cfg.stock_symbols


# ------------------------------------------------------------------ filtrage de bout en bout
def test_screen_stocks_end_to_end(tmp_path, policy):
    companies = {
        "HLL": clean_company(financials=clean_financials(non_permissible_income=None)),
        "BNK": clean_company(symbol="BNK", industry="Banks - Regional"),
    }

    def fetch(symbol):
        if symbol == "ERR":
            raise RuntimeError("timeout")
        return companies[symbol], {"info": {}}

    store = ParquetStore(tmp_path)
    reviews = {"HLL": review(non_permissible_income="0")}
    rows = screen_stocks(["HLL", "BNK", "ERR"], policy, store, reviews, fetch=fetch, today=TODAY)
    status = {r.result.symbol: r.result.compliant for r in rows}
    assert status == {"HLL": True, "BNK": False, "ERR": False}
    assert rows[0].reviewed_fields == ["non_permissible_income"]
    assert list((tmp_path / "screening").glob("*.json"))
    assert list((tmp_path / "fundamentals" / "HLL").glob("*.json"))


def test_without_review_missing_data_excludes(tmp_path, policy):
    company = clean_company(financials=clean_financials(non_permissible_income=None))
    rows = screen_stocks(["HLL"], policy, ParquetStore(tmp_path), {}, fetch=lambda s: (company, {}), today=TODAY)
    assert not rows[0].result.compliant


def test_local_exchange_date_is_kept():
    idx = pd.date_range("2024-01-02", periods=2, tz="Europe/Paris")
    df = normalize_prices(pd.DataFrame({"date": idx, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0}), "t")
    assert [d.date() for d in df["date"]] == [date(2024, 1, 2), date(2024, 1, 3)]
