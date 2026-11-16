"""Rule D (tracker OF1): 60 consecutive sessions below the MC1 5th percentile steps a book down to paper."""

import numpy as np
import pandas as pd

from nse_engine import capital_ladder as cl

PCTS = [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98, 99]


def _doc():
    """Flat bands: 5th percentile return -5%, drawdown -10%, at every horizon from 5 to 756."""
    ret = list(np.linspace(-0.10, 0.30, len(PCTS)))
    ret[PCTS.index(5)] = -0.05
    dd = list(np.linspace(-0.20, 0.0, len(PCTS)))
    dd[PCTS.index(5)] = -0.10
    ret, dd = sorted(ret), sorted(dd)
    hs = [5, 756]
    return {"percentiles": PCTS, "books": {"deployed": {"return": {str(h): ret for h in hs},
                                                        "drawdown": {str(h): dd for h in hs}}}}


def _equity(path):
    return pd.Series(path, index=pd.bdate_range("2027-01-01", periods=len(path)), dtype="float64")


def test_healthy_book_never_trips():
    e = _equity(np.linspace(100, 110, 120))
    r = cl.mc_breach_streak(e, _doc(), "deployed")
    assert r["streak"] == 0 and not r["tripped"]


def test_sixty_sessions_below_trips():
    path = [100.0] * 5 + [80.0] * 70                   # -20% return and drawdown, far below the 5th
    r = cl.mc_breach_streak(_equity(path), _doc(), "deployed")
    assert r["streak"] >= cl.MC_STREAK_SESSIONS and r["tripped"]


def test_fifty_nine_does_not_trip_and_recovery_resets():
    path = [100.0] * 5 + [80.0] * 59
    r = cl.mc_breach_streak(_equity(path), _doc(), "deployed")
    assert not r["tripped"]
    # a recovery above the floor on both measures resets the streak (drawdown floor is since the start, so
    # use bands with a loose drawdown floor for this case)
    doc = _doc()
    doc["books"]["deployed"]["drawdown"] = {k: sorted(np.linspace(-0.9, -0.5, len(PCTS))) for k in ("5", "756")}
    path = [100.0] * 5 + [80.0] * 40 + [120.0] + [80.0] * 40
    r = cl.mc_breach_streak(_equity(path), doc, "deployed")
    assert r["streak"] == 40 and not r["tripped"]


def test_no_bands_reads_as_not_tripped():
    r = cl.mc_breach_streak(_equity([100.0, 50.0] * 50), None, "deployed")
    assert not r["readable"] and not r["tripped"]


def test_readiness_refuses_go_live_when_tripped():
    tripped = {"readable": True, "streak": 60, "tripped": True, "last": {}}
    checks = cl.readiness({"verdict": "PASS", "sessions": 60}, [{"clean": True}] * 5, sell_path=(True, "ok"),
                          capital=cl.RUNGS[0], mc=tripped)
    row = [c for c in checks if c[0] == "MC floor (rule D)"][0]
    assert row[1] is False


def test_the_paper_alarm_counts_return_alone_and_dates_its_run():
    """MCA1 reads Rule D's counter on return only: a deep early fall the book has recovered from keeps Rule D's
    drawdown leg below its floor, never the alarm's."""
    recovered = _equity([100.0] * 5 + [80.0] + [100.0] * 30)        # -20% once, then back to the start
    assert cl.mc_breach_streak(recovered, _doc(), "deployed")["streak"] == 31          # Rule D: drawdown counts
    assert cl.mc_breach_streak(recovered, _doc(), "deployed", kinds=("return",))["streak"] == 0
    low = _equity([100.0] * 5 + [90.0] * 20)                          # -10% return, below the -5% line
    r = cl.mc_breach_streak(low, _doc(), "deployed", kinds=("return",))
    assert r["streak"] == 20 and r["since"] == low.index[5].date() and not r["tripped"]
    assert cl.mc_breach_streak(_equity(np.linspace(100, 110, 50)), _doc(), "deployed")["since"] is None
