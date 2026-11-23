"""The maximal Sharpe ratio estimator (nse_engine/validation/max_sharpe.py, tracker HSR1): Chernov, Kelly, Malamud and
Schwab (2026), "The Hidden Sharpe Ratio", NBER WP 35988, checked against its own theorems on economies whose answer is
known, and against the direct matrix definitions."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import scipy.linalg

from nse_engine.validation import max_sharpe as ms


def _economy(p: int, t: int, seed: int, r2_total: float = 2.0, identity: bool = False):
    """Normalised returns (Assumption 1) with a known answer: eigenvalues uniform on [0.2, 1.8] (or all 1) and the
    same Sharpe ratio on every principal component, so R*^2 = r2_total per period.  The covariance is diagonal (the
    eigenvectors are the assets): every statistic here is invariant to rotating the assets, and Gaussian noise is too,
    so no random rotation is drawn.  (A 600 x 600 QR on this Mac's Accelerate, numpy 2.4.6 or scipy 1.17.1, corrupts
    the heap and crashes the process at a later garbage collection.)"""
    rng = np.random.default_rng(seed)
    lam = np.ones(p) if identity else np.sort(rng.uniform(0.2, 1.8, p))[::-1]
    v = np.eye(p)
    mu = np.sqrt(np.full(p, r2_total / p) * lam) * rng.choice([-1.0, 1.0], p)
    r = mu + rng.standard_normal((t, p)) * np.sqrt(lam)
    return lam, v, mu, r


def _frame(r: np.ndarray) -> pd.DataFrame:
    """The module rescales by P^-1/2 itself, so it is handed P^1/2 times the normalised returns."""
    return pd.DataFrame(r * math.sqrt(r.shape[1]), index=pd.date_range("2000-01-31", periods=r.shape[0], freq="ME"))


def _population(lam, v, mu, z_star):
    """B_0 and B_1 of eqs. 11 and 20 at the effective shrinkage."""
    a = (v.T @ mu) ** 2
    return float((a / (z_star + lam)).sum()), float((z_star * a / (z_star + lam) ** 2).sum())


@pytest.mark.parametrize("p,t", [(8, 40), (60, 30)])
def test_the_spectral_computation_matches_the_matrix_definitions(p, t):
    """The SVD route (T x T work, zero eigenvalues summed exactly) equals eqs. 10-16 written with P x P matrices, on
    both sides of P = T; derivatives (eqs. C.7-C.9) against central finite differences."""
    r = np.random.default_rng(5).standard_normal((t, p)) * 0.05 + 0.01
    cfg = ms.MaxSharpeConfig(z_grid=(0.05, 0.5), min_periods=10, z_scale="absolute")
    rows = ms.max_sharpe(pd.DataFrame(r, index=pd.date_range("2000-01-31", periods=t, freq="ME")), cfg)["grid"]
    x = r / math.sqrt(p)
    mean, centred = x.mean(axis=0), x - x.mean(axis=0)
    sigma, n = centred.T @ centred / (t - 1), t - 1

    def level(z):
        inv = scipy.linalg.inv(z * np.eye(p) + sigma)
        d = 1 - p / n + p / n * z * np.trace(inv) / p
        return z / d, d * (mean @ inv @ mean + 1) - 1             # Z*(z), Theta_d(z)

    for row in rows:
        z, h = row["z"], row["z"] * 1e-4
        z_star, theta_d = level(z)
        z_star_prime = (level(z + h)[0] - level(z - h)[0]) / (2 * h)
        theta_1 = z_star * -((level(z + h)[1] - level(z - h)[1]) / (2 * h)) / z_star_prime
        assert row["z_star"] == pytest.approx(z_star, rel=1e-9) and row["theta_0"] == pytest.approx(theta_d, rel=1e-9)
        assert row["z_star_prime"] == pytest.approx(z_star_prime, rel=1e-5)
        assert row["theta_1"] == pytest.approx(theta_1, rel=1e-4)


@pytest.mark.parametrize("p,t,expected", [(600, 201, 1.5), (2640, 401, 1.1786)])
def test_lemma_2s_equality_case_reproduces_the_papers_floor(p, t, expected):
    """Lemma 2 (eq. 37): Z*'(0) >= c / (c - 1), with equality when all eigenvalues are equal.  Section 5.2 states the
    floor as 1.18 at the paper's full-sample c = 6.6.  Tolerance 2% (finite P and T)."""
    _, _, _, r = _economy(p, t, 3, identity=True)
    row = ms.max_sharpe(_frame(r), ms.MaxSharpeConfig(z_grid=(1e-4,)))["grid"][0]
    assert row["z_star_prime"] == pytest.approx(expected, rel=0.02)


@pytest.mark.parametrize("p,t", [(200, 401), (900, 301)])
def test_the_debiased_statistics_hit_their_targets_and_ridge_portfolios_stay_under_the_ceiling(p, t):
    """Lemma 1 and Theorem 2: Theta_0 and Theta_1 estimate B_0 and B_1 at the effective shrinkage (median error
    within 10% and 35% over 40 economies, about 3 standard errors; the second is the noisier).  Proposition 4 (eq. 34): the exact out-of-sample
    squared Sharpe ratio of the ridge portfolio (zI + cov)^-1 mean never exceeds H_L1(B) / Z*', B the population
    values; against the estimated H_L1 it may by finite-sample noise, at most 5% here."""
    errors0, errors1 = [], []
    for seed in range(40):
        lam, v, mu, r = _economy(p, t, seed)
        out = ms.max_sharpe(_frame(r), ms.MaxSharpeConfig(z_grid=(1e-2, 1e-1)))
        centred = r - r.mean(axis=0)
        sigma, cov = centred.T @ centred / (t - 1), (v * lam) @ v.T
        for row in out["grid"]:
            b0, b1 = _population(lam, v, mu, row["z_star"])
            errors0.append(row["theta_0"] / b0 - 1)
            errors1.append(row["theta_1"] / b1 - 1)
            pi = scipy.linalg.solve(row["z"] * np.eye(p) + sigma, r.mean(axis=0))
            out_of_sample = (pi @ mu) ** 2 / (pi @ cov @ pi)
            assert out_of_sample <= b0 ** 2 / (b0 - b1) / row["z_star_prime"]
            if row["status"] == "ok":
                assert out_of_sample <= 1.05 * row["ridge_ceiling_sq"]
    assert abs(np.median(errors0)) < 0.10 and abs(np.median(errors1)) < 0.35


def test_the_estimate_ignores_asset_order_and_units_and_keeps_lemma_2s_shape():
    """Permuting the test assets changes nothing.  With the default relative grid, returns in other units (x 3) give
    the same estimate; with an absolute grid the penalty must scale by k^2 for that.  Z* rises and Z*' falls with the
    penalty, and Z*' >= 1 (Lemma 2, eq. 35)."""
    lam, v, mu, r = _economy(120, 90, 9)
    frame, grid = _frame(r), (1e-3, 1e-2, 1e-1, 1.0)
    base = ms.max_sharpe(frame, ms.MaxSharpeConfig(z_grid=grid))["grid"]
    shuffled = ms.max_sharpe(frame[np.random.default_rng(1).permutation(frame.columns)],
                             ms.MaxSharpeConfig(z_grid=grid))["grid"]
    scaled = ms.max_sharpe(frame * 3.0, ms.MaxSharpeConfig(z_grid=grid))["grid"]
    absolute = ms.max_sharpe(frame, ms.MaxSharpeConfig(z_grid=grid, z_scale="absolute"))["grid"]
    absolute_scaled = ms.max_sharpe(frame * 3.0, ms.MaxSharpeConfig(z_grid=tuple(9.0 * z for z in grid),
                                                                    z_scale="absolute"))["grid"]
    for a, b, c in zip(base, shuffled, scaled, strict=True):
        for key in ("theta_0", "theta_1", "z_star_prime"):
            assert b[key] == pytest.approx(a[key], rel=1e-9) and c[key] == pytest.approx(a[key], rel=1e-9)
    for a, b in zip(absolute, absolute_scaled, strict=True):
        for key in ("theta_0", "theta_1", "z_star_prime"):
            assert b[key] == pytest.approx(a[key], rel=1e-9)
    z_star = [row["z_star"] for row in base]
    z_star_prime = [row["z_star_prime"] for row in base]
    assert z_star == sorted(z_star) and z_star_prime == sorted(z_star_prime, reverse=True) and min(z_star_prime) >= 1


def test_bad_inputs_fail_loudly():
    idx = pd.date_range("2000-01-31", periods=80, freq="ME")
    good = pd.DataFrame(np.random.default_rng(0).standard_normal((80, 5)) * 0.05, index=idx)
    with pytest.raises(ValueError, match="NaN"):
        ms.max_sharpe(good.mask(good > 0.1))
    with pytest.raises(ValueError, match="min_periods"):
        ms.max_sharpe(good.iloc[:30])
    with pytest.raises(ValueError, match="positive"):
        ms.max_sharpe(good, ms.MaxSharpeConfig(z_grid=(0.0,)))
    with pytest.raises(ValueError, match="J = 1"):
        ms.max_sharpe(good, ms.MaxSharpeConfig(order=2))
    with pytest.raises(ValueError, match="increasing"):
        ms.max_sharpe(good.iloc[::-1])
    with pytest.raises(ValueError, match="19 draws"):
        ms.detectability(good, draws=10)


def test_the_randomisation_test_keeps_its_size_and_finds_a_strong_signal():
    """On pure noise the sign-flip p-value is at most 0.05 in about one economy in twenty; with an annual maximal
    Sharpe ratio of 3.5 (1.0 a month, squared) it is found in most.  10 economies each, 49 draws."""
    def rejections(r2):
        hits = 0
        for seed in range(10):
            _, _, _, r = _economy(400, 160, 100 + seed, r2_total=r2)
            hits += ms.detectability(_frame(r), draws=49, seed=seed)["p_value"] <= 0.05
        return hits

    assert rejections(1e-12) <= 2 and rejections(1.0) >= 7


def test_managed_portfolios_follow_eq_48_and_the_random_features_are_fixed_by_the_seed():
    day = pd.Timestamp("2020-01-31")
    returns = pd.Series({"A": 0.10, "B": -0.05, "C": 0.02, "D": float("nan")})
    table = pd.DataFrame({"mom": [3.0, 1.0, 2.0, 5.0], "size": [1.0, 1.0, float("nan"), 2.0]}, index=list("ABCD"))
    linear, skipped = ms.managed_portfolios({day: (returns, table)}, ["mom", "size"], min_names=3)
    # ranks across A, B, C (D has no return): mom -> +0.5, -0.5, 0; size -> 0, 0 (tied), missing -> 0
    assert skipped == 0 and linear.loc[day, "mom"] == pytest.approx((0.5 * 0.10 - 0.5 * -0.05 + 0) / 3)
    assert linear.loc[day, "size"] == pytest.approx(0.0)
    first, _ = ms.managed_portfolios({day: (returns, table)}, ["mom", "size"], random_features=50, seed=7, min_names=3)
    again, _ = ms.managed_portfolios({day: (returns, table)}, ["mom", "size"], random_features=50, seed=7, min_names=3)
    assert first.shape == (1, 50) and first.equals(again)
    full = pd.Series({"A": 0.01, "B": 0.02, "C": 0.03, "D": 0.04})
    later, skipped = ms.managed_portfolios({day: (returns, table), day + pd.offsets.MonthEnd(1): (full, table)},
                                           ["mom"], min_names=4)
    assert skipped == 1 and list(later.index) == [day + pd.offsets.MonthEnd(1)]   # 3 names with a return: skipped
    with pytest.raises(ValueError, match="no date had enough"):
        ms.managed_portfolios({day: (returns, table)}, ["mom"], min_names=4)
    with pytest.raises(ValueError, match="missing"):
        ms.managed_portfolios({day: (returns, table)}, ["value"], min_names=3)


def _signal_panel(months: int = 80, stocks: int = 150, seed: int = 0, strength: float = 0.04):
    """Monthly cross-sections where next month's return loads on ``b1`` (a base characteristic, weakly) and on a
    hidden ``h`` (strongly); ``noise`` carries nothing.  Returns are decimal."""
    rng = np.random.default_rng(seed)
    periods = {}
    names = [f"S{i}" for i in range(stocks)]
    for day in pd.date_range("2015-01-31", periods=months, freq="ME"):
        x = pd.DataFrame(rng.standard_normal((stocks, 4)), index=names, columns=["b1", "b2", "h", "noise"])
        rank = x.rank(pct=True) - 0.5
        returns = 0.01 * rank["b1"] + strength * rank["h"] + rng.standard_normal(stocks) * 0.08
        periods[day] = (returns, x)
    return periods


def test_the_spanning_screen_goes_for_an_informative_signal_and_stops_for_noise():
    """Phase 0's screen (HSR2): a characteristic that drives returns widens the frontier the base spans (GO, small
    p); one that is pure noise does not (STOP).  Small sizes for speed: 200 random features, 2 seeds, 19 shuffles."""
    periods = _signal_panel()
    kwargs = dict(random_features=200, seeds=(1, 2), permutations=19, min_names=100)
    informative = ms.spanning_increment(periods, ["b1", "b2"], "h", **kwargs)
    noise = ms.spanning_increment(periods, ["b1", "b2"], "noise", **kwargs)
    assert informative["verdict"] == "GO" and informative["p_value"] <= 0.05 and min(informative["rise_sq_by_seed"]) > 0
    assert informative["annual_with_candidate"] > informative["annual_base"]
    assert informative["linear_p_value"] <= 0.05
    assert noise["verdict"] == "STOP" and noise["p_value"] > 0.10
    with pytest.raises(ValueError, match="already in the base"):
        ms.spanning_increment(periods, ["b1", "h"], "h", **kwargs)


def test_a_candidate_must_cover_every_month_with_finite_values():
    periods = {d: v for d, v in list(_signal_panel(months=3, stocks=5).items())}
    days = sorted(periods)
    rows = [{"date": d, "symbol": f"S{i}", "value": float(i)} for d in days for i in range(4)]   # S4 missing
    out = ms.attach_characteristic(periods, pd.DataFrame(rows), "cand")
    assert list(out[days[0]][1]["cand"].iloc[:4]) == [0.0, 1.0, 2.0, 3.0] and np.isnan(out[days[0]][1]["cand"].iloc[4])
    with pytest.raises(ValueError, match="does not cover 1"):
        ms.attach_characteristic(periods, pd.DataFrame([r for r in rows if r["date"] != days[1]]), "cand")
    with pytest.raises(ValueError, match="infinite"):
        ms.attach_characteristic(periods, pd.DataFrame(rows + [{"date": days[0], "symbol": "S4", "value": np.inf}]),
                                 "cand")
    with pytest.raises(ValueError, match="more than one"):
        ms.attach_characteristic(periods, pd.DataFrame(rows + rows[:1]), "cand")
    with pytest.raises(ValueError, match="already"):
        ms.attach_characteristic(periods, pd.DataFrame(rows), "h")


def test_the_spanning_command_refuses_the_sealed_holdout(capsys):
    from runners import run_spanning

    assert run_spanning.main(["--candidate", "unused.parquet", "--name", "x", "--end", "2026-03-31"]) == 2
    assert "sealed 2026 holdout" in capsys.readouterr().err
