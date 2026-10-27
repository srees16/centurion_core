"""
Is a return series' edge real, or could luck alone have produced it? (tracker HT1)

A hypothesis test of the mean return, for a backtest and for paper or live
returns alike:

* **The null hypothesis:** the true mean excess return (over the risk-free
  rate the books' Sharpe uses) is zero: no edge.
* **t-statistic:** the mean over its standard error.  The standard error is
  Newey-West (HAC): daily returns are not independent (volatility clusters,
  positions persist), and an i.i.d. standard error would overstate t.  The
  i.i.d. t is reported beside it.
* **p-value:** the probability of a mean at least this far from zero if the
  null were true; two-sided, and one-sided (a positive edge).  A p-value is
  not the probability that the strategy works.
* **Confidence intervals** of the mean at 90, 95 and 99%: the true means the
  data fits.  Significant at a level when the interval excludes zero.
* **t ~ Sharpe x sqrt(years):** a Sharpe of 1 needs about 4 years to reach
  t = 2; ``years_for_t2`` is (2 / Sharpe)^2 at the observed Sharpe.
* **Many rules tested:** of N strategies with no edge, about N x alpha pass at
  alpha by luck.  ``multiple_testing`` gives that count and the Bonferroni
  p; the deflated Sharpe (``dsr``) and PBO remain the registry's measures.
* **Assumptions, checked:** the first and second halves' means compared (a
  market that changed), and the edge before costs against after them (an
  edge too small to survive costs).

For paper and live returns, ``vs_expected`` asks the question that matters
while the record is short: is the realised mean consistent with the edge
the backtest claims, or significantly below it?

    python -m nse_engine.validation.significance --run <run_id>          # a recorded backtest
    python -m nse_engine.validation.significance --returns returns.csv   # any date,return CSV
    python -m nse_engine.validation.significance --book deployed         # a paper book (Neon)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from nse_engine.validation.diagnostics import newey_west_cov

PERIODS = 252
LEVELS = (0.90, 0.95, 0.99)
ALPHA = 0.05


def _hac_lags(n: int) -> int:
    return int(math.floor(4 * (n / 100.0) ** (2.0 / 9.0)))       # Newey-West's rule, as ``alpha_beta``


def mean_test(returns: pd.Series, rf_annual: float = 0.0, null_daily: float = 0.0,
              levels: Sequence[float] = LEVELS, hac_lags: Optional[int] = None, periods: int = PERIODS) -> Dict[str, Any]:
    """The hypothesis test of the mean daily excess return against ``null_daily`` (0: no edge)."""
    r = pd.Series(returns, dtype="float64").dropna()
    x = r.to_numpy() - rf_annual / periods
    n = len(x)
    if n < 3:
        return {"n": n, "error": "fewer than 3 returns"}
    mean, sd = float(x.mean()), float(x.std(ddof=1))
    lags = _hac_lags(n) if hac_lags is None else int(hac_lags)
    ones = np.ones((n, 1))
    se_hac = float(math.sqrt(max(newey_west_cov(ones, x - mean, lags)[0, 0], 0.0)))
    se_iid = sd / math.sqrt(n)
    dist = sp_stats.t(df=n - 1)
    t = (mean - null_daily) / se_hac if se_hac > 0 else float("nan")
    t_iid = (mean - null_daily) / se_iid if se_iid > 0 else float("nan")
    years = n / periods
    sharpe = mean / sd * math.sqrt(periods) if sd > 0 else float("nan")
    ci = {f"{lv:.0%}": [mean - dist.ppf(0.5 + lv / 2) * se_hac, mean + dist.ppf(0.5 + lv / 2) * se_hac] for lv in levels}
    out = {
        "n": n, "years": years, "rf_annual": rf_annual, "null_daily": null_daily,
        "mean_daily": mean, "mean_annual": mean * periods, "sd_daily": sd,
        "se_hac": se_hac, "se_iid": se_iid, "hac_lags": lags, "autocorr_lag1": float(pd.Series(x).autocorr(1)),
        "t": t, "t_iid": t_iid,
        "p_two_sided": float(2 * dist.sf(abs(t))) if np.isfinite(t) else float("nan"),
        "p_one_sided": float(dist.sf(t)) if np.isfinite(t) else float("nan"),
        "ci_daily": ci, "ci_annual": {k: [v[0] * periods, v[1] * periods] for k, v in ci.items()},
        "significant": {k: bool(v[0] > null_daily or v[1] < null_daily) for k, v in ci.items()},
        "sharpe_annual": sharpe, "t_rule": sharpe * math.sqrt(years) if np.isfinite(sharpe) else float("nan"),
        "years_for_t2": (2.0 / sharpe) ** 2 if np.isfinite(sharpe) and sharpe > 0 else float("inf"),
    }
    return out


def multiple_testing(p_value: float, n_trials: int, alpha: float = ALPHA) -> Dict[str, Any]:
    """With ``n_trials`` rules tested, how many pass at ``alpha`` by luck alone, and the Bonferroni p."""
    n = max(int(n_trials), 1)
    bonf = min(1.0, float(p_value) * n) if np.isfinite(p_value) else float("nan")
    return {"n_trials": n, "alpha": alpha, "expected_false_passes": n * alpha,
            "p_bonferroni": bonf, "passes_after_correction": bool(bonf <= alpha)}


def stability(returns: pd.Series, rf_annual: float = 0.0, periods: int = PERIODS) -> Dict[str, Any]:
    """Did the market stay the same?  The first and second halves' mean excess returns, and Welch's t of their
    difference (a p below 0.05 says the edge changed)."""
    x = pd.Series(returns, dtype="float64").dropna() - rf_annual / periods
    a, b = x.iloc[: len(x) // 2], x.iloc[len(x) // 2:]
    if len(a) < 3 or len(b) < 3:
        return {"error": "too few returns"}
    t, p = sp_stats.ttest_ind(b, a, equal_var=False)
    return {"first_half": [str(a.index[0].date()), str(a.index[-1].date()), float(a.mean() * periods)],
            "second_half": [str(b.index[0].date()), str(b.index[-1].date()), float(b.mean() * periods)],
            "t_difference": float(t), "p_difference": float(p)}


def after_costs(test: Dict[str, Any], cost_drag_annual: Optional[float], periods: int = PERIODS) -> Dict[str, Any]:
    """The edge before costs against after them: ``cost_drag_annual`` (a run's recorded drag, a positive share
    of capital a year) added back gives the gross mean and its t on the same standard error."""
    if cost_drag_annual is None or not np.isfinite(cost_drag_annual) or not test.get("se_hac"):
        return {}
    gross = test["mean_daily"] + cost_drag_annual / periods
    return {"cost_drag_annual": cost_drag_annual, "gross_mean_annual": gross * periods,
            "t_gross": (gross - test["null_daily"]) / test["se_hac"], "t_net": test["t"],
            "costs_share_of_edge": cost_drag_annual / (gross * periods) if gross > 0 else float("nan")}


def vs_expected(returns: pd.Series, expected_sharpe: float, rf_annual: float = 0.0,
                periods: int = PERIODS) -> Dict[str, Any]:
    """For a short paper or live record: is its mean consistent with the edge its backtest claims?  The expected
    daily excess mean is the backtest's Sharpe at the record's own volatility; the one-sided p is the chance of a
    mean this far below it if the backtest's edge were real (a low p: the edge may be gone)."""
    base = mean_test(returns, rf_annual, periods=periods)
    if "error" in base or not np.isfinite(expected_sharpe):
        return {"error": base.get("error", "no backtest Sharpe")}
    expected = expected_sharpe * base["sd_daily"] / math.sqrt(periods)
    t = (base["mean_daily"] - expected) / base["se_hac"] if base["se_hac"] > 0 else float("nan")
    return {"expected_sharpe": expected_sharpe, "expected_mean_annual": expected * periods,
            "realised_mean_annual": base["mean_annual"], "t": t,
            "p_below": float(sp_stats.t(df=base["n"] - 1).cdf(t)) if np.isfinite(t) else float("nan"),
            "years_for_t2_at_backtest": (2.0 / expected_sharpe) ** 2 if expected_sharpe > 0 else float("inf")}


def report(returns: pd.Series, rf_annual: float, n_trials: int = 1, cost_drag_annual: Optional[float] = None,
           expected_sharpe: Optional[float] = None) -> Dict[str, Any]:
    """Everything above for one return series."""
    test = mean_test(returns, rf_annual)
    if "error" in test:
        return {"test": test}
    out: Dict[str, Any] = {"test": test, "multiple_testing": multiple_testing(test["p_one_sided"], n_trials),
                           "stability": stability(returns, rf_annual)}
    costs = after_costs(test, cost_drag_annual)
    if costs:
        out["costs"] = costs
    if expected_sharpe is not None:
        out["vs_backtest"] = vs_expected(returns, expected_sharpe, rf_annual)
    return out


def one_line(rep: Dict[str, Any]) -> str:
    """The test in a sentence, for scorecards and emails."""
    t = rep["test"]
    if "error" in t:
        return f"hypothesis test: {t['error']}"
    lo, hi = t["ci_annual"]["95%"]
    s = (f"mean excess return {t['mean_annual']:+.1%} a year, t {t['t']:+.2f} (Newey-West; i.i.d. {t['t_iid']:+.2f}), "
         f"p {t['p_two_sided']:.3f} two-sided; 95% interval {lo:+.1%} to {hi:+.1%} a year "
         f"({'excludes' if t['significant']['95%'] else 'includes'} zero); Sharpe {t['sharpe_annual']:.2f} over "
         f"{t['years']:.1f} years (t ~ Sharpe x sqrt(years) = {t['t_rule']:+.2f}; t = 2 needs "
         f"{t['years_for_t2']:.1f} years at this Sharpe)")
    m = rep.get("multiple_testing") or {}
    if m.get("n_trials", 1) > 1:
        s += (f"; {m['n_trials']} configurations tested: about {m['expected_false_passes']:.0f} pass at 5% by luck, "
              f"Bonferroni p {m['p_bonferroni']:.3f}")
    c = rep.get("costs") or {}
    if c:
        s += f"; before costs t {c['t_gross']:+.2f} (costs {c['cost_drag_annual']:.1%} a year)"
    st = rep.get("stability") or {}
    if "p_difference" in st:
        s += (f"; halves {st['first_half'][2]:+.1%} vs {st['second_half'][2]:+.1%} a year "
              f"(p {st['p_difference']:.2f} that they differ by chance)")
    v = rep.get("vs_backtest") or {}
    if "t" in v:
        s += (f"; against the backtest's edge (Sharpe {v['expected_sharpe']:.2f}): t {v['t']:+.2f}, "
              f"p {v['p_below']:.2f} of falling this short by chance")
    return s


# ── CLI ──────────────────────────────────────────────────────────

def _series(df: pd.DataFrame) -> pd.Series:
    s = pd.Series(df.iloc[:, -1].to_numpy(dtype="float64"), index=pd.DatetimeIndex(pd.to_datetime(df.iloc[:, 0])))
    return s.sort_index()


def main(argv: Optional[Sequence[str]] = None) -> int:
    from nse_engine.config import EngineConfig

    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", help="a recorded run id (its returns.csv and manifest)")
    g.add_argument("--returns", help="a CSV of date,return")
    g.add_argument("--book", help="a paper book's returns since its start, from Neon (needs CENTURION_DATABASE_URL)")
    p.add_argument("--runs-dir", default=EngineConfig().runs_dir)
    p.add_argument("--rf", type=float, default=EngineConfig().risk_free_annual)
    p.add_argument("--trials", type=int, default=1, help="configurations tested (multiple testing)")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    cost, expected = None, None
    if args.run:
        d = Path(args.runs_dir) / args.run
        r = _series(pd.read_csv(d / "returns.csv"))
        cost = (json.loads((d / "manifest.json").read_text()).get("metrics") or {}).get("cost_drag")
    elif args.returns:
        r = _series(pd.read_csv(args.returns))
    else:
        from nse_engine import books as bk
        from tools.books_report import load_day_books

        book = next((b for b in bk.discover_books() if b.name == args.book), None)
        if book is None:
            raise SystemExit(f"no book {args.book!r}")
        eq = load_day_books([book])[book.name].paper.equity
        r = eq.pct_change().dropna()
        reg = bk.read_register().set_index("book")
        expected = float(reg.loc[book.name, "bt_sharpe"]) if book.name in reg.index else None
    rep = report(r, args.rf, n_trials=args.trials, cost_drag_annual=cost, expected_sharpe=expected)
    print(json.dumps(rep, indent=1, default=str) if args.json else one_line(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
