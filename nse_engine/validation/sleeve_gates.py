"""
The research protocol's pass rules (section 5) for a sleeve funded from the core (GT4, GT5).

* Gate 1: deflated Sharpe >= 0.95 at the family's N, on excess returns over the book's 6.5%.
* Gate 2: 0.75 x the live book + 0.25 x the sleeve, daily, raises CAGR, excess Sharpe and alpha against
  NIFTY 50 TRI, with MaxDD at most 2 points worse.
* Phase 1 of the golden-target review: excess Sharpe >= 0.5 alone; the combined Sharpe >= the book's + 0.10.
* Significance: the Newey-West t of the sleeve's alpha against NIFTY 50 TRI >= 2 (a published effect).

Reported beside them: the monthly correlation with the book and each calendar year's return.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

BOOK_RF = 0.065
COMBINED_WEIGHT = 0.25


def carry_to(returns: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Daily returns compounded onto another calendar: a day missing from ``index`` rolls into the next one."""
    level = (1 + returns).cumprod()
    start = level.index[0]
    on = level.reindex(level.index.union(index)).ffill().reindex(index[index >= start])
    base = level.iloc[0] / (1 + returns.iloc[0])
    return on.pct_change().fillna(on.iloc[0] / base - 1).dropna()


def sleeve_gates(sleeve: pd.Series, trials_matrix: Optional[pd.DataFrame], book: pd.Series, tri: pd.Series,
                 rf_annual: float = BOOK_RF, weight: float = COMBINED_WEIGHT) -> Dict[str, Any]:
    """Every rule for one sleeve's daily returns, aligned to the book's sessions by the caller."""
    from nse_engine.validation.diagnostics import alpha_beta
    from nse_engine.validation.dsr import deflated_sharpe, performance_summary

    b = book.reindex(sleeve.index).dropna()
    s = sleeve.reindex(b.index).fillna(0.0)
    many = trials_matrix is not None and trials_matrix.shape[1] > 1
    dsr = deflated_sharpe(s, trials_matrix=trials_matrix if many else None, n_trials=None if many else 1,
                          rf_annual=rf_annual)["dsr"]
    base, ss = performance_summary(b, rf_annual), performance_summary(s, rf_annual)
    comb = (1 - weight) * b + weight * s
    cs = performance_summary(comb, rf_annual)
    base_alpha = alpha_beta(b, tri, rf_annual=rf_annual)["alpha_annual"]
    sa, ca = alpha_beta(s, tri, rf_annual=rf_annual), alpha_beta(comb, tri, rf_annual=rf_annual)["alpha_annual"]
    gate2 = {"cagr": cs["cagr"] > base["cagr"], "excess_sharpe": cs["excess_sharpe"] > base["excess_sharpe"],
             "alpha": ca > base_alpha, "max_drawdown": cs["max_drawdown"] >= base["max_drawdown"] - 0.02}
    rules = {"gate1": bool(ss["excess_sharpe"] > 0 and dsr >= 0.95), "gate2": all(gate2.values()),
             "sharpe_alone_0.5": bool(ss["excess_sharpe"] >= 0.5),
             "combined_sharpe_plus_0.10": bool(cs["excess_sharpe"] >= base["excess_sharpe"] + 0.10),
             "alpha_t_2": bool(sa["alpha_t_hac"] >= 2)}

    def monthly(r: pd.Series) -> pd.Series:
        return (1 + r).resample("ME").prod() - 1

    def yearly(r: pd.Series) -> pd.Series:
        return (1 + r).groupby(r.index.year).prod() - 1

    by_year = yearly(b)
    return {
        "dsr": dsr,
        "book": {**{k: base[k] for k in ("cagr", "excess_sharpe", "max_drawdown")}, "alpha_annual": base_alpha},
        "sleeve": {**{k: ss[k] for k in ("cagr", "excess_sharpe", "max_drawdown")},
                   "alpha_annual": sa["alpha_annual"], "alpha_t_hac": sa["alpha_t_hac"], "beta": sa["beta"]},
        "combined": {**{k: cs[k] for k in ("cagr", "excess_sharpe", "max_drawdown")}, "alpha_annual": ca,
                     "gate2_checks": gate2},
        "rules": rules, "pass": all(rules.values()),
        "monthly_correlation_with_book": float(monthly(s).corr(monthly(b))),
        "years": {int(y): {"sleeve": round(float(v), 4), "book": round(float(by_year.get(y)), 4)}
                  for y, v in yearly(s).items()},
    }
