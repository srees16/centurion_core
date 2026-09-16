# Running backtests and walk-forwards on Kaggle

Paper and live trading stay on GitHub Actions. Kaggle is for the research jobs
that do not fit an Actions job: the walk-forward grid and Stage-A sweeps.

## Why

The 2013–2025 walk-forward is 775 backtests at about 108 s each — roughly
**23 hours** on one core. An Actions job is capped at 6 hours. A free Kaggle
CPU session gives **4 cores, 30 GB RAM and 12 hours**, so the grid runs four
wide (≈ 6 hours) and, because each fold is written as it finishes, a second
session picks up where the first stopped.

## One-time setup

```bash
pip install kaggle
# Kaggle → Account → Create New API Token → saves kaggle.json
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
python -m cloud.kaggle_local check          # says what is still missing
```

Two private datasets keep code and data apart, so a code change does not
re-upload 200 MB of prices:

| Dataset | Holds | Push when |
|---|---|---|
| `<user>/centurion-nse-store` | `data/nse_engine/store` (204 MB parquet) | the store is rebuilt |
| `<user>/centurion-nse-code` | `nse_engine/`, `runners/`, `cloud/`, `config/`, `job.json` | every job |

```bash
python -m cloud.kaggle_local push-store      # first run, then after build-store
```

## Running a walk-forward

```bash
GRID='{"signals.use_low_vol":[false],"portfolio.stop_atr_mult":[6.0],
       "portfolio.rebalance_days":[5,21],"regime.neutral_scale":[0.6,1.0],
       "portfolio.target_positions":[20,30]}'

python -m cloud.kaggle_local run --task walk-forward --args \
  "--grid '$GRID' --start 2013-01-01 --end 2025-12-31 --data-start 2011-01-01 \
   --folds 0-3 --workers 4 --max-hours 11"

python -m cloud.kaggle_local watch           # polls until the kernel stops
python -m cloud.kaggle_local pull            # → data/nse_engine/kaggle_out/<ts>/
```

Then the next session takes the folds that are left (`--folds 4-`), and when
every fold is in:

```bash
python -m cloud.wf_stitch import-runs --src data/nse_engine/kaggle_out/latest/runs
python -m cloud.wf_stitch stitch --dirs data/nse_engine/kaggle_out/*/wf
python -m runners.run_nse_engine validate --run-id <full-period run of the last fold's choice>
```

`import-runs` matters: PBO and the deflated Sharpe are only honest if every
configuration ever evaluated is in the registry, including the ones Kaggle ran.

### Every session of one walk-forward must use the same window

`load_market_data` applies its liquidity filter over the window it is asked
for, so a session with a different `--end` or `--data-start` selects a
different universe. On a 2-fold toy run, `--end 2018-12-31` versus
`2019-12-31` moved the stitched OOS Sharpe by 0.003. Each fold file records
its job signature; the runner refuses to add a fold to a directory built with
a different window, and the stitcher refuses to stitch across them. Use a
fresh `--out-dir` for a new window.

Verified parity: the same toy job through `cloud.kaggle_runner` and through
`nse_engine.validation.walk_forward.run_walk_forward` gives the same stitched
OOS Sharpe to 13 decimal places, and picks the same parameters per fold.

## Stage-A sweep

```bash
python -m cloud.kaggle_local run --task grid --args \
  "--grid '$GRID' --start 2013-01-01 --end 2025-12-31 --data-start 2011-01-01 \
   --workers 4 --tag stage-a"
```

Grid points whose config hash is already in the registry are skipped, so a
re-run after a timeout only does what is left.

## Monitoring

Every job writes `heartbeat.json` next to its fold files: status, message,
elapsed time, folds done and folds left. Point it at an uptime service to be
told when a session dies:

```bash
export CENTURION_HEARTBEAT_URL=https://hc-ping.com/<uuid>     # healthchecks.io
```

The convention is healthchecks.io's — `/start` when the job begins, the bare
URL for progress (throttled to one ping per 30 s), `/fail` on an exception —
and an UptimeRobot heartbeat monitor, Better Stack or a self-hosted endpoint
sees the same pings. Set the expected period to about 90 minutes: a fold
pings when it finishes, and folds take 30–60 minutes at four workers.

`cloud.kaggle_local watch` pings the same URL from here while it polls, so a
missed heartbeat means either the kernel or the watcher stopped.

Pings are best-effort: a monitoring failure is logged, never raised, and never
kills the job.

## What not to run on Kaggle

- **Paper and live trading** — they need the deployment file, broker
  credentials and a fixed daily schedule; they stay in Actions.
- **The holdout** — one evaluation only, enforced by
  `data/nse_engine/holdout.lock`. Run it locally so the lock file is the
  authority.
- **Anything writing `config/nse_engine_deployed.json`** — promotion happens
  locally, after `validate` has seen the imported Kaggle runs.

## Limits worth knowing

- 12 hours per CPU session; `--max-hours 11` stops cleanly between folds and
  writes `state.json` naming the next fold.
- Internet is off by default in a kernel and is not needed: the store arrives
  as a dataset. `run --internet` turns it on if a job ever needs to sync.
- Datasets are mounted read-only, so `kaggle_entry.py` copies the code into
  `/kaggle/working` before running.
- Check Kaggle's current quotas before planning a long series of sessions;
  they change, and the weekly accelerator quota does not apply to CPU-only
  sessions the same way.
