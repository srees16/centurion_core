"""B1 regression: universe eligibility must not depend on later corporate actions."""
from __future__ import annotations

import pandas as pd

from conftest import truncate
from nse_engine.config import UniverseConfig
from nse_engine.universe import compute_universe_panel, select_universe

CUT = "2021-06-30"          # the synthetic split happens after this date
BASE = dict(top_n_liquid=25, min_history_days=60, liquidity_lookback_days=40,
            refresh_every_n_days=21, min_median_value_inr=0.0)


def test_price_filter_reads_the_printed_price(panel):
    """With the fix on, a name whose printed price is below the floor is excluded."""
    cfg = UniverseConfig(**BASE, min_price_inr=50.0, price_filter_unadjusted=True)
    mask = compute_universe_panel(panel, cfg).mask
    printed = panel.close_unadj.reindex_like(mask).ffill()
    selected_below_floor = ((printed < 50.0) & mask).to_numpy().sum()
    assert selected_below_floor == 0


def test_membership_does_not_change_when_later_data_is_removed(panel):
    """The fix makes membership identical whatever date the load ends (B1)."""
    cfg = UniverseConfig(**BASE, min_price_inr=50.0, price_filter_unadjusted=True)
    full = compute_universe_panel(panel, cfg).mask.loc[:CUT]
    short = compute_universe_panel(truncate(panel, CUT), cfg).mask
    assert full.columns.equals(short.columns)
    assert (full.to_numpy() == short.to_numpy()).all(), "universe changed when the future was removed"


def test_legacy_behaviour_still_available_and_differs(panel):
    """The legacy filter is kept for the deployed config's hash - and it does differ."""
    legacy = UniverseConfig(**BASE, min_price_inr=50.0, price_filter_unadjusted=False)
    fixed = UniverseConfig(**BASE, min_price_inr=50.0, price_filter_unadjusted=True)
    a = compute_universe_panel(panel, legacy).mask
    b = compute_universe_panel(panel, fixed).mask
    assert (a.to_numpy() != b.to_numpy()).any(), "the split symbol should be treated differently"


def test_single_date_selection_matches_the_panel(panel):
    """The live path (select_universe) must agree with the backtest panel."""
    cfg = UniverseConfig(**BASE, min_price_inr=50.0, price_filter_unadjusted=True)
    mask = compute_universe_panel(panel, cfg).mask
    as_of = panel.dates[-1]
    live = set(select_universe(panel, cfg, as_of))
    panel_names = set(mask.columns[mask.loc[as_of].to_numpy()])
    assert live == panel_names


def test_universe_respects_the_size_cap(panel):
    cfg = UniverseConfig(**{**BASE, "top_n_liquid": 10}, min_price_inr=0.0, price_filter_unadjusted=True)
    mask = compute_universe_panel(panel, cfg).mask
    assert int(mask.sum(axis=1).max()) <= 10
