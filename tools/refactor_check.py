#!/usr/bin/env python3
"""After every refactor step, did anything Centurion runs change?  (Tracker RF0h; the owner's answers, 26 Nov 2026.)

The refactor's rule is that structure changes and behaviour does not (docs/AUDIT.md).  This checks it from outside,
against a baseline recorded on the last commit before the refactor's first change (c90b906, 17 Nov):

- entry points: every repository module, script and inline import the GitHub workflows and the Space's start script
  run, every module the code starts with ``-m``, and every workflow it dispatches by file name, still resolve;
- the API's route table, and the frontend's calls (``../centurion_core-fe``) matched against it;
- the order path: the game day's fault replay (``tools.game_day``, the live session's own code against a simulated
  Kite) over the five faults' weeks, session for session;
- the engine: the store guards, which include every book's 2013-25 backtest against config/canary_expected.json;
- the fast suite on the full install and on the core-only one CI uses (``../corevenv``).

The replay, both suites and the store guards run at the same time; the replay, the longest, sets the time.

    python -m tools.refactor_check baseline     # once: the surface and the replay at BASELINE_COMMIT (about 9 min)
    python -m tools.refactor_check surface      # the static part against the baseline (seconds)
    python -m tools.refactor_check run          # after every refactor step: everything (11 min on 26 Nov)

``run`` records its outcome in RECORD and exits 1 on any difference.  A change of behaviour that was approved is
accepted by recording a new baseline on the commit that made it (``baseline --commit <sha>``), never by editing it.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shared.clock import IST  # noqa: E402 - after the path bootstrap

#: The last commit before the refactor's first change (19f2167, 20 Nov 2026).
BASELINE_COMMIT = "c90b906"
OUT = Path("data/nse_engine/refactor_check")             # git-ignored: this machine's record
BASELINE, RECORD = OUT / "baseline.json", OUT / "last_run.json"
FRONTEND = _ROOT.parent / "centurion_core-fe"
CORE_PYTHON = _ROOT.parent / "corevenv" / "bin" / "python"
#: The game day's fault scenario up to the session after its last fault (all five faults, about 27 sessions).
REPLAY_WINDOW = ("2024-05-02", "2024-06-11")
#: Files that start the code: the workflows, and the container's entry point on the Space.
LAUNCHERS = (".github/workflows/*.yml", "deployment/start.sh")
_SKIP_DIRS = {".git", "node_modules", "__pycache__", "data", "myenv", "venv", ".venv", ".next", "tests"}

_PY = r"python3?(?:\.\d+)?"
_DASH_M = re.compile(_PY + r"\s+-m\s+([A-Za-z_][\w.]*)")
_SCRIPT = re.compile(_PY + r"\s+([\w./-]+\.py)\b")
_HEREDOC = re.compile(_PY + r"\s+-\s[^\n]*?<<-?\s*['\"]?(\w+)['\"]?[^\n]*\n(.*?)\n\s*\1\s*$", re.S | re.M)
_DASH_C = re.compile(_PY + r"\s+-c\s+([\"'])(.*?)(?<!\\)\1", re.S)
_FROM_PAREN = re.compile(r"from\s+([A-Za-z_][\w.]*)\s+import\s*\(([^)]*)\)", re.S)
_FROM = re.compile(r"^\s*from\s+([A-Za-z_][\w.]*)\s+import\s+(.+)$")
_IMPORT = re.compile(r"^\s*import\s+([A-Za-z_][\w.]*(?:\s+as\s+\w+)?(?:\s*,\s*[A-Za-z_][\w.]*(?:\s+as\s+\w+)?)*)\s*$")
_CODE_DASH_M = re.compile(r"[\"']-m[\"']\s*,\s*[\"']([A-Za-z_][\w.]*)[\"']")
_WORKFLOW_NAME = re.compile(r"[\"']([\w.-]+\.yml)[\"']")


# ------------------------------------------------------------------------------------------------ entry points

def _repo_tops(root: Path) -> set:
    """The repository's own top-level packages and modules: what an entry point may resolve into."""
    return {p.name for p in root.iterdir() if p.is_dir() and (p / "__init__.py").exists()} | \
        {p.stem for p in root.glob("*.py")}


def _module_file(root: Path, module: str) -> Optional[Path]:
    base = root.joinpath(*module.split("."))
    for p in (base.with_suffix(".py"), base / "__init__.py"):
        if p.exists():
            return p
    return None


def _bound_names(path: Path) -> Tuple[set, bool]:
    """Names a module binds at its top level (inside if / try / with blocks too), and whether it has a module
    ``__getattr__`` (lazy names, which only an import could prove)."""
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    names: set = set()

    def visit(body: Sequence[ast.stmt]) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    names.update(n.id for n in ast.walk(t) if isinstance(n, ast.Name))
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names.update((a.asname or a.name).split(".")[0] for a in node.names)
            for field in ("body", "orelse", "finalbody", "handlers"):
                inner = getattr(node, field, None)
                if inner and not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    visit([h for h in inner if isinstance(h, ast.stmt)] if field != "handlers" else
                          [s for h in inner for s in h.body])

    visit(tree.body)
    return names, "__getattr__" in names


def _resolve(root: Path, module: str, names: Sequence[str] = ()) -> str:
    """"ok", or what is missing: the module, or one of the names imported from it."""
    path = _module_file(root, module)
    if path is None:
        return "module missing"
    if not names:
        return "ok"
    bound, lazy = _bound_names(path)
    missing = [n for n in names if n != "*" and n not in bound and not lazy
               and _module_file(root, f"{module}.{n}") is None]
    return f"missing: {', '.join(missing)}" if missing else "ok"


def _imports(code: str) -> List[Tuple[str, List[str]]]:
    """(module, names) for each import statement in a piece of inline Python."""
    out = [(m, [n.split(" as ")[0].strip() for n in names.split(",") if n.strip()])
           for m, names in _FROM_PAREN.findall(code)]
    code = _FROM_PAREN.sub("", code)
    for stmt in (s for line in code.splitlines() for s in line.split(";")):
        m = _FROM.match(stmt)
        if m:
            out.append((m.group(1), [n.split(" as ")[0].strip() for n in m.group(2).split(",") if n.strip()]))
            continue
        m = _IMPORT.match(stmt)
        if m:
            out += [(n.split(" as ")[0].strip(), []) for n in m.group(1).split(",")]
    return out


def _python_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]
        for f in filenames:
            if f.endswith(".py"):
                yield Path(dirpath) / f


def entry_points(root: Path) -> Dict[str, str]:
    """``"<file>: <what it starts>"`` -> "ok" or what is missing, for everything that starts repository code."""
    tops, out = _repo_tops(root), {}
    workflows = {p.name for p in (root / ".github" / "workflows").glob("*.yml")}
    for pattern in LAUNCHERS:
        for f in sorted(root.glob(pattern)):
            text, where = f.read_text(encoding="utf-8"), f.relative_to(root).as_posix()
            for mod in _DASH_M.findall(text):
                if mod.split(".")[0] in tops or _module_file(root, mod):
                    out[f"{where}: -m {mod}"] = _resolve(root, mod)
            for script in _SCRIPT.findall(text):
                out[f"{where}: {script}"] = "ok" if (root / script).exists() else "file missing"
            inline = [body for _, body in _HEREDOC.findall(text)] + [body for _, body in _DASH_C.findall(text)]
            for mod, names in (i for body in inline for i in _imports(body)):
                if mod.split(".")[0] in tops:
                    out[f"{where}: from {mod} import {', '.join(names) or '(module)'}"] = _resolve(root, mod, names)
    for f in _python_files(root):
        text, where = f.read_text(encoding="utf-8", errors="replace"), f.relative_to(root).as_posix()
        for mod in _CODE_DASH_M.findall(text):
            if mod.split(".")[0] in tops:
                out[f"{where}: -m {mod}"] = _resolve(root, mod)
        for line in text.splitlines():
            if re.search(r"workflow|dispatch", line, re.I):
                for name in _WORKFLOW_NAME.findall(line):
                    out[f"{where}: workflow {name}"] = "ok" if name in workflows else "workflow missing"
    return dict(sorted(out.items()))


# ------------------------------------------------------------------------------------------- routes and frontend

_ROUTES_CODE = """
import json, sys
sys.path.insert(0, ".")
from api.main import app
out = set()
for r in app.routes:
    methods = getattr(r, "methods", None) or ["WS" if "WebSocket" in type(r).__name__ else type(r).__name__]
    out.update(f"{m} {r.path}" for m in methods)
print(json.dumps(sorted(out)))
"""


def routes(root: Path, python: str = sys.executable) -> List[str]:
    """``"METHOD /path"`` for every route the API serves, imported in a clean process (no secrets, no database)."""
    env = {"HOME": os.environ.get("HOME", ""), "PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1"}
    done = subprocess.run([python, "-c", _ROUTES_CODE], cwd=root, env=env, capture_output=True, text=True)
    if done.returncode:
        raise RuntimeError(f"the API did not import in {root}: {done.stderr.strip().splitlines()[-1:]}")
    return json.loads(done.stdout.strip().splitlines()[-1])


_API_CALL = re.compile(r"\bapi\.\w+\s*(?:<(?:[^<>]|<[^<>]*>)*>)?\(\s*(`[^`]*`|\"[^\"]*\"|'[^']*')")
_LITERAL = re.compile(r"(`[^`]*`|\"[^\"]*\"|'[^']*')")


def _normalise(literal: str) -> Optional[str]:
    """A call's path with each interpolated segment as ``{}``, the base URL and the query string dropped."""
    s = re.sub(r"^\$\{[^}]*\}(?=/)", "", literal[1:-1])
    s = re.split(r"\?", s, maxsplit=1)[0]
    s = re.sub(r"(?<=[^/])\$\{[^}]*\}$", "", s)          # an appended query or suffix builder, not a segment
    s = re.sub(r"\$\{[^}]*\}", "{}", s)
    return s.rstrip("/") or None if s.startswith("/") else None


def frontend_calls(fe_root: Path) -> List[str]:
    """The backend paths the frontend calls: every ``api.<method>(path)`` (the client sends them all to the
    backend) and every literal under ``/api/v1/`` (the direct fetches).  The frontend's own routes, fetched by
    relative ``/api/<name>`` paths, are not the backend's and are not read."""
    found = set()
    for sub in ("app", "hooks", "components", "lib"):
        for f in sorted((fe_root / sub).rglob("*.ts*")):
            if "node_modules" in f.parts:
                continue
            text = f.read_text(encoding="utf-8", errors="replace")
            lits = _API_CALL.findall(text) + [x for x in _LITERAL.findall(text) if "/api/v1/" in x]
            for lit in lits:
                p = _normalise(lit)
                if p:
                    found.add(p)
    return sorted(found)


def _route_pattern(path: str) -> "re.Pattern[str]":
    parts = re.split(r"(\{[^}]+\})", path)
    rx = "".join((".+" if p.endswith(":path}") else "[^/]+") if p.startswith("{") else re.escape(p) for p in parts)
    return re.compile(f"^{rx}/?$")


def match(calls: Sequence[str], route_list: Sequence[str]) -> Dict[str, str]:
    """Each call -> the route path it reaches, or "" when none does (an interpolated segment counts as a value)."""
    paths = sorted({r.split(" ", 1)[1] for r in route_list})
    pats = [(p, _route_pattern(p)) for p in paths]
    out = {}
    for c in calls:
        probe = c.replace("{}", "x")
        out[c] = next((p for p, rx in pats if rx.match(probe)), "")
    return out


def surface(root: Path, python: str = sys.executable) -> Dict[str, Any]:
    return {"entry_points": entry_points(root), "routes": routes(root, python)}


# ------------------------------------------------------------------------------------------------------ replay

_REPLAY_CODE = """
import json, logging, math, sys
sys.path.insert(0, ".")
logging.disable(logging.CRITICAL)
from tools import game_day as gd
start, end, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
faults = {d: f for d, f in gd.SCENARIOS["faults"]["faults"].items() if start <= d <= end}
res = gd.run("faults", start, end, faults)
rows = res.pop("_sessions")
def clean(v):
    if isinstance(v, float):
        return None if math.isnan(v) else round(v, 2)
    return v
keep = ("orders_sent", "rejected", "events", "drawdown_states", "max_drawdown_pct", "session_errors", "checks",
        "verdict")
json.dump({"sessions": [{k: clean(v) for k, v in r.items()} for r in rows.to_dict("records")],
           "summary": {k: res.get(k) for k in keep}}, open(out_path, "w"), default=str, sort_keys=True)
"""


def _start_replay(root: Path, out_path: Path, python: str = sys.executable) -> subprocess.Popen:
    return subprocess.Popen([python, "-c", _REPLAY_CODE, *REPLAY_WINDOW, str(out_path)], cwd=root,
                            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)


def _finish_replay(proc: subprocess.Popen, out_path: Path) -> Dict[str, Any]:
    _, err = proc.communicate()
    if proc.returncode or not out_path.exists():
        raise RuntimeError(f"the replay failed: {(err or '').strip().splitlines()[-3:]}")
    return json.loads(out_path.read_text())


# --------------------------------------------------------------------------------------------------- comparing

def differences(base: Dict[str, Any], now: Dict[str, Any], calls: Sequence[str] = ()) -> Tuple[List[str], List[str]]:
    """(problems, notes) between the baseline and now.  Problems: an entry point that resolved and no longer does
    (or is gone), a route gone, a frontend call that reached a route and no longer does, a replayed session or the
    replay's summary that differs.  Notes: what was added, and frontend calls that reached nothing in either."""
    problems: List[str] = []
    notes: List[str] = []
    b_ep, n_ep = base["surface"]["entry_points"], now["surface"]["entry_points"] if "surface" in now else {}
    if "surface" in now:
        for k, v in b_ep.items():
            if v == "ok" and n_ep.get(k) != "ok":
                problems.append(f"entry point {k}: {n_ep.get(k, 'gone')}")
        problems += [f"entry point {k}: {v} (new)" for k, v in n_ep.items() if k not in b_ep and v != "ok"]
        b_rt, n_rt = set(base["surface"]["routes"]), set(now["surface"]["routes"])
        problems += [f"route gone: {r}" for r in sorted(b_rt - n_rt)]
        if n_rt - b_rt:
            notes.append(f"routes added: {', '.join(sorted(n_rt - b_rt))}")
        before, after = match(calls, sorted(b_rt)), match(calls, sorted(n_rt))
        problems += [f"frontend call {c} reaches nothing now (it reached {before[c]})" for c in calls
                     if before[c] and not after[c]]
        never = [c for c in calls if not before[c] and not after[c]]
        if never:
            notes.append(f"frontend calls that reach no route, before or now: {', '.join(never)}")
    if "replay" in now:
        b_s, n_s = base["replay"]["sessions"], now["replay"]["sessions"]
        if len(b_s) != len(n_s):
            problems.append(f"replay: {len(n_s)} sessions, the baseline had {len(b_s)}")
        diff = [(b.get("day"), k, b.get(k), n.get(k)) for b, n in zip(b_s, n_s, strict=False)
                for k in sorted(set(b) | set(n)) if b.get(k) != n.get(k)]          # unequal lengths: above
        problems += [f"replay {day} {k}: {old} -> {new}" for day, k, old, new in diff[:10]]
        if len(diff) > 10:
            problems.append(f"replay: and {len(diff) - 10} more differences")
        b_sum, n_sum = base["replay"]["summary"], now["replay"]["summary"]
        problems += [f"replay summary {k}: {b_sum.get(k)} -> {n_sum.get(k)}" for k in sorted(set(b_sum) | set(n_sum))
                     if b_sum.get(k) != n_sum.get(k)]
    return problems, notes


def _pytest(python: Path, root: Path) -> subprocess.Popen:
    return subprocess.Popen([str(python), "-m", "pytest", "tests", "-m", "not slow", "-p", "no:cacheprovider"],
                            cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})


def _suite(name: str, proc: subprocess.Popen) -> Tuple[Dict[str, int], List[str]]:
    from tools.guards import parse_summary

    out, _ = proc.communicate()
    counts = parse_summary(out or "")
    bad = [] if proc.returncode == 0 and counts["passed"] and not counts["failed"] else \
        [f"{name}: {counts['failed']} failed, exit {proc.returncode}: "
         + " | ".join(x for x in (out or "").splitlines() if x.startswith(("FAILED", "ERROR")))[:600]]
    return counts, bad


# ---------------------------------------------------------------------------------------------------- commands

def _tree(commit: str, into: Path) -> str:
    """The commit's files in ``into``, with this checkout's data and .env linked in (both are git-ignored)."""
    sha = subprocess.run(["git", "rev-parse", commit], cwd=_ROOT, capture_output=True, text=True, check=True)
    archive = into / "tree.tar"
    with open(archive, "wb") as fh:
        subprocess.run(["git", "archive", "--format=tar", sha.stdout.strip()], cwd=_ROOT, stdout=fh, check=True)
    tree = into / "tree"
    with tarfile.open(archive) as tar:
        tar.extractall(tree, filter="data")
    archive.unlink()
    for sub in ("data", "data/nse_engine"):
        (tree / sub).mkdir(parents=True, exist_ok=True)
        for p in (_ROOT / sub).iterdir():
            if not (tree / sub / p.name).exists() and not (sub == "data" and p.name == "nse_engine"):
                (tree / sub / p.name).symlink_to(p)
    if (_ROOT / ".env").exists():
        (tree / ".env").symlink_to(_ROOT / ".env")
    return sha.stdout.strip()


def record_baseline(commit: str = BASELINE_COMMIT) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="refactor_check_") as tmp:
        sha = _tree(commit, Path(tmp))
        tree = Path(tmp) / "tree"
        proc = _start_replay(tree, Path(tmp) / "replay.json")
        surf = surface(tree)
        replay = _finish_replay(proc, Path(tmp) / "replay.json")
    base = {"commit": sha, "recorded": datetime.now(IST).isoformat(timespec="minutes"), "window": list(REPLAY_WINDOW),
            "surface": surf, "replay": replay}
    (_ROOT / OUT).mkdir(parents=True, exist_ok=True)
    (_ROOT / BASELINE).write_text(json.dumps(base, indent=1, sort_keys=True))
    return base


def _load_baseline() -> Dict[str, Any]:
    path = _ROOT / BASELINE
    if not path.exists():
        raise SystemExit(f"no baseline at {BASELINE}: python -m tools.refactor_check baseline")
    base = json.loads(path.read_text())
    if base.get("window") != list(REPLAY_WINDOW):
        raise SystemExit(f"the baseline replayed {base.get('window')}, not {list(REPLAY_WINDOW)}: record it again")
    return base


def check(full: bool = True) -> Dict[str, Any]:
    """The surface (and, when ``full``, the replay, both suites and the store guards) against the baseline."""
    base = _load_baseline()
    began = time.monotonic()
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=_ROOT, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=_ROOT, capture_output=True, text=True)
    problems: List[str] = []
    now: Dict[str, Any] = {}
    counts: Dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="refactor_check_") as tmp, ThreadPoolExecutor(max_workers=1) as pool:
        replay_out = Path(tmp) / "replay.json"
        procs = {}
        guard_run: Optional["Future[Dict[str, Any]]"] = None
        if full:
            from tools import guards

            # All four at once: none reads what another writes (the fast tests that read the guards' record or the
            # data catalogue use temporary folders, bar the preflight's, which only needs a line to exist).
            guard_run = pool.submit(guards.run)
            procs["replay"] = _start_replay(_ROOT, replay_out)
            procs["fast suite"] = _pytest(Path(sys.executable), _ROOT)
            if CORE_PYTHON.exists():
                procs["core-only suite"] = _pytest(CORE_PYTHON, _ROOT)
            else:
                problems.append(f"core-only suite not run: no {CORE_PYTHON} (python3.13 -m venv ../corevenv && "
                                "../corevenv/bin/pip install -r requirements-core.txt pytest)")
        now["surface"] = surface(_ROOT)
        calls = frontend_calls(FRONTEND) if FRONTEND.exists() else []
        if not calls:
            problems.append(f"no frontend calls read from {FRONTEND}")
        if full and guard_run is not None:
            for name in ("fast suite", "core-only suite"):
                if name in procs:
                    counts[name], bad = _suite(name, procs[name])
                    problems += bad
            g = guard_run.result()
            counts["store guards"] = {k: g[k] for k in ("passed", "failed", "skipped")}
            if g["failed"] or g["skipped"] or g["exit"]:
                problems.append(f"store guards: {g['failed']} failed, {g['skipped']} skipped (python -m tools.guards)")
            now["replay"] = _finish_replay(procs["replay"], replay_out)
    found, notes = differences(base, now, calls)
    problems += found
    result = {"when": datetime.now(IST).isoformat(timespec="minutes"), "head": head.stdout.strip(),
              "uncommitted": len([x for x in dirty.stdout.splitlines() if x.strip()]), "full": full,
              "baseline": base["commit"][:7], "counts": counts, "entry_points": len(now["surface"]["entry_points"]),
              "routes": len(now["surface"]["routes"]), "frontend_calls": len(calls),
              "replayed_sessions": len(now.get("replay", {}).get("sessions", [])), "notes": notes,
              "problems": problems, "verdict": "PASS" if not problems else "FAIL",
              "seconds": round(time.monotonic() - began)}
    if full:
        (_ROOT / RECORD).write_text(json.dumps(result, indent=1))
    return result


def report(r: Dict[str, Any]) -> str:
    head = (f"Refactor check {r['verdict']} ({'everything' if r['full'] else 'surface only'}), HEAD {r['head']} "
            f"with {r['uncommitted']} uncommitted file(s), against {r['baseline']}; {r['seconds']} s")
    sizes = (f"  entry points {r['entry_points']}, routes {r['routes']}, frontend calls {r['frontend_calls']}"
             + (f", replayed sessions {r['replayed_sessions']}" if r["full"] else ""))
    lines = [head, sizes] + [f"  {k}: {v['passed']} passed, {v['failed']} failed, {v['skipped']} skipped"
                             for k, v in r["counts"].items()]
    lines += [f"  note: {n}" for n in r["notes"]] + [f"  ! {p}" for p in r["problems"]]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("cmd", choices=("run", "surface", "baseline"))
    p.add_argument("--commit", default=BASELINE_COMMIT, help="baseline: the commit to record (default %(default)s)")
    args = p.parse_args(argv)
    os.chdir(_ROOT)
    if args.cmd == "baseline":
        b = record_baseline(args.commit)
        print(f"Baseline {b['commit'][:7]} recorded in {BASELINE}: {len(b['surface']['entry_points'])} entry points "
              f"({sum(v != 'ok' for v in b['surface']['entry_points'].values())} not resolving), "
              f"{len(b['surface']['routes'])} routes, {len(b['replay']['sessions'])} replayed sessions, "
              f"replay {b['replay']['summary']['verdict']}")
        return 0
    r = check(full=args.cmd == "run")
    print(report(r))
    return 0 if r["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
