"""Tests de la reconstitution de la composition historique du S&P 500."""

import pandas as pd

from halal_trading.data.sp500_history import Change, members_on, membership_intervals, membership_mask, to_yahoo


def current():
    return pd.DataFrame({"symbol": ["AAA", "NEW", "META"], "name": ["A", "New", "Meta"],
                         "sector": ["x", "y", "z"],
                         "date_added": pd.to_datetime(["1990-01-01", "2015-06-01", "2013-12-23"]),
                         "cik": [1, 2, 3]})


CHANGES = [
    Change(pd.Timestamp("2015-06-01"), "NEW", "OLD"),   # NEW remplace OLD
    Change(pd.Timestamp("2013-12-23"), "FB", "GONE"),   # FB (devenu META, renommage absent de la liste)
]


def test_backward_reconstruction():
    iv = membership_intervals(current(), CHANGES, pd.Timestamp("2026-01-01"), earliest=pd.Timestamp("2000-01-01"))
    assert members_on(iv, "2016-01-01") == {"AAA", "NEW", "META"}
    assert members_on(iv, "2014-06-01") == {"AAA", "OLD", "META"}  # avant le remplacement : OLD était membre
    assert members_on(iv, "2010-01-01") == {"AAA", "OLD", "GONE"}  # META exclu avant sa date d'entrée publiée


def test_ticker_format():
    assert to_yahoo("BRK.B") == "BRK-B"


def test_membership_mask_on_panel():
    iv = membership_intervals(current(), CHANGES, pd.Timestamp("2026-01-01"), earliest=pd.Timestamp("2000-01-01"))
    panel = pd.DataFrame({"symbol": ["OLD", "OLD", "NEW"],
                          "date": pd.to_datetime(["2015-05-29", "2015-06-02", "2015-06-02"]).tz_localize("UTC")})
    assert membership_mask(panel, iv).tolist() == [True, False, True]
