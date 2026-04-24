"""
SN32 Miner Dashboard — live TUI.

Combines two signals into one view on the VPS:

  1. Latest W&B metrics for your UID from the subnet32 validator runs
     (polled every ``--wandb-interval`` seconds). The headline number is
     ``ema_ood`` (weighted_out_of_domain_f1_score) — the EMA the validator
     gates reward on at 0.9.

  2. A tail of ``neurons/validator_logs/summary.log`` showing each incoming
     validator request with its prediction average (updates in real time).

Usage::

    pip install rich wandb
    python scripts/dashboard.py --uid 75

Press Ctrl-C to exit.
"""
import argparse
import datetime as dt
import json
import re
import sys
import threading
import time
from pathlib import Path

try:
    from rich.console import Console  # noqa: F401
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ImportError:
    sys.exit("Install dependencies first:  pip install rich wandb")

try:
    import wandb
except ImportError:
    sys.exit("Install dependencies first:  pip install rich wandb")


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_LOG = REPO_ROOT / "neurons" / "validator_logs" / "summary.log"

# Parses a line written by neurons/miner.py:log_validator_request:
#   [2026-04-23T19:47:56+00:00] hk=5DWt... ver=1.0.0 ok=1 n_texts=120
#   words(min/avg/max)=65/205/350 dups=3 latency_ms=1243 avg_pred=0.487
SUMMARY_RE = re.compile(
    r"^\[(?P<ts>[^\]]+)\]\s+hk=(?P<hk>\S+)\s+ver=(?P<ver>\S+)\s+"
    r"ok=(?P<ok>\d+)\s+n_texts=(?P<n>\d+)\s+"
    r"words\(min/avg/max\)=(?P<wmin>\d+)/(?P<wavg>\d+)/(?P<wmax>\d+)\s+"
    r"dups=(?P<dups>\d+)\s+latency_ms=(?P<lat>\d+)\s+avg_pred=(?P<pred>\S+)"
    r"(?:\s+err=(?P<err>\S+))?"
)


def parse_summary_line(line):
    m = SUMMARY_RE.match(line.strip())
    if not m:
        return None
    d = m.groupdict()
    try:
        d["pred_num"] = float(d["pred"])
    except ValueError:
        d["pred_num"] = None
    # err is optional and "none" means no inference error occurred.
    raw_err = d.get("err")
    d["err"] = None if raw_err in (None, "none") else raw_err
    return d


def read_recent_requests(path, limit):
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            tail = f.readlines()[-(limit * 3):]
    except Exception:
        return []
    parsed = [p for p in (parse_summary_line(l) for l in tail) if p]
    return parsed[-limit:]


# ---------------------------------------------------------------------------
# W&B polling — runs in a background thread so the UI stays responsive even
# while a poll is in flight. Cached result is read each UI refresh.
# ---------------------------------------------------------------------------

def _validator_uid_from_run(run_name):
    # runs are named "validator-<UID>-<timestamp>" (see detection/base/validator.py:418)
    parts = run_name.split("-")
    if len(parts) >= 2 and parts[0] == "validator":
        return parts[1]
    return None


def latest_entries_per_validator(api, project, uid, max_runs):
    """Scan the most-recent ``max_runs`` validator-* runs and return a dict
    keyed by validator UID (extracted from run name) → newest entry for the
    target miner uid. Multiple runs for the same validator (after restarts)
    are collapsed to the newest one.
    """
    by_validator = {}
    scanned = 0
    for run in api.runs(project, order="-created_at"):
        if not run.name.startswith("validator-"):
            continue
        vuid = _validator_uid_from_run(run.name)
        if vuid is None:
            continue
        scanned += 1

        # Find the newest row in this run that has the target uid.
        best_in_run = None
        for row in run.scan_history(keys=["original_format_json", "_timestamp"]):
            payload = row.get("original_format_json")
            if not payload:
                continue
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                continue
            ts = float(parsed.get("timestamp") or row.get("_timestamp") or 0)
            m = parsed.get("uid_metrics", {}).get(str(uid))
            if m is None:
                continue
            best_in_run = {"run": run.name, "vuid": vuid, "ts": ts, "metrics": m}
            break  # scan_history yields newest-first

        if best_in_run is not None:
            existing = by_validator.get(vuid)
            if existing is None or best_in_run["ts"] > existing["ts"]:
                by_validator[vuid] = best_in_run

        if scanned >= max_runs:
            break
    return by_validator


class WandbPoller(threading.Thread):
    def __init__(self, uid, project, interval, max_runs):
        super().__init__(daemon=True)
        self.uid = uid
        self.project = project
        self.interval = interval
        self.max_runs = max_runs

        self.latest = None
        self.error = None
        self.last_poll = None
        self.next_poll_at = time.time()
        self.stop_flag = False
        self._api = None

    def run(self):
        while not self.stop_flag:
            if time.time() >= self.next_poll_at:
                try:
                    if self._api is None:
                        self._api = wandb.Api()
                    self.latest = latest_entries_per_validator(
                        self._api, self.project, self.uid, self.max_runs,
                    )
                    self.error = None
                except Exception as e:
                    self.error = str(e)[:180]
                self.last_poll = time.time()
                self.next_poll_at = time.time() + self.interval
            time.sleep(0.5)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _format_age(seconds):
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    if s < 86400:
        return f"{s // 3600}h{(s % 3600) // 60:02d}m"
    return f"{s // 86400}d{(s % 86400) // 3600:02d}h"


def render_wandb(poller):
    # Status panel for the pre-data / error cases
    if poller.last_poll is None and poller.error is None:
        body = Text("warming up (first poll in progress)...", style="yellow")
        return Panel(body, title=f"[bold]Per-validator scores · UID {poller.uid}[/bold]",
                     border_style="cyan", padding=(1, 2))
    if poller.error:
        body = Text(f"error: {poller.error}", style="red")
        return Panel(body, title=f"[bold]Per-validator scores · UID {poller.uid}[/bold]",
                     border_style="red", padding=(1, 2))
    if not poller.latest:
        body = Text(f"no validator data found for uid {poller.uid} in last {poller.max_runs} runs",
                    style="yellow")
        return Panel(body, title=f"[bold]Per-validator scores · UID {poller.uid}[/bold]",
                     border_style="yellow", padding=(1, 2))

    # Build a table row per validator, sorted by age (freshest at top)
    entries = sorted(poller.latest.values(), key=lambda e: -e["ts"])
    now = time.time()

    table = Table(expand=True, show_lines=False, header_style="bold")
    table.add_column("validator", style="cyan", no_wrap=True)
    table.add_column("age", justify="right", style="dim", no_wrap=True)
    table.add_column("reward", justify="right")
    table.add_column("pen", justify="center")
    table.add_column("f1", justify="right")
    table.add_column("ood_f1", justify="right")
    table.add_column("ema_ood", justify="right")
    table.add_column("gate", justify="center")
    table.add_column("weight", justify="right")

    # Summary counters for the subtitle
    pass_count = 0
    fail_count = 0
    unknown_count = 0
    rewards = []

    for e in entries:
        m = e["metrics"]
        age = _format_age(now - e["ts"])
        reward = float(m.get("reward") or 0)
        rewards.append(reward)
        penalty = m.get("penalty")
        f1 = float(m.get("f1_score") or 0)
        ood = float(m.get("out_of_domain_f1_score") or 0)
        ema = m.get("weighted_out_of_domain_f1_score")
        weight = float(m.get("weight") or 0)

        reward_style = "bold green" if reward > 0 else "red"
        penalty_style = "green" if penalty == 1 else "red"
        penalty_text = Text(str(penalty) if penalty is not None else "-", style=penalty_style)

        if isinstance(ema, (int, float)):
            ema_val = float(ema)
            if ema_val >= 0.9:
                ema_text = Text(f"{ema_val:.4f}", style="bold green")
                gate_text = Text("✓", style="bold green")
                pass_count += 1
            else:
                ema_text = Text(f"{ema_val:.4f}", style="bold red")
                gate_text = Text("✗", style="bold red")
                fail_count += 1
        else:
            ema_text = Text("-", style="dim")
            gate_text = Text("?", style="yellow")
            unknown_count += 1

        table.add_row(
            e["vuid"],
            age,
            Text(f"{reward:.4f}", style=reward_style),
            penalty_text,
            f"{f1:.4f}",
            f"{ood:.4f}",
            ema_text,
            gate_text,
            f"{weight:.6f}",
        )

    # Panel subtitle with polling status + pass/fail summary
    if poller.last_poll:
        poll_age = _format_age(now - poller.last_poll)
        next_in = max(0, int(poller.next_poll_at - now))
        poll_s = f"polled {poll_age} ago · next in {next_in}s"
    else:
        poll_s = f"first poll in {max(0, int(poller.next_poll_at - now))}s"

    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    summary = (f"{len(entries)} validators · gate PASS={pass_count} FAIL={fail_count} ?={unknown_count} "
               f"· mean_reward={mean_reward:.4f} · {poll_s}")

    return Panel(
        table,
        title=f"[bold]Per-validator scores · UID {poller.uid}[/bold]  {poller.project}",
        subtitle=Text(summary, style="dim"),
        border_style="cyan",
    )


def render_requests(requests):
    table = Table(expand=True, show_lines=False, header_style="bold")
    table.add_column("time (UTC)", style="white", no_wrap=True)
    table.add_column("hotkey", style="cyan", no_wrap=True)
    table.add_column("ok", justify="center", no_wrap=True)
    table.add_column("texts", justify="right")
    table.add_column("words min/avg/max", justify="right")
    table.add_column("dups", justify="right")
    table.add_column("latency", justify="right")
    table.add_column("avg_pred", justify="right")
    table.add_column("err", style="red", no_wrap=True)

    if not requests:
        table.add_row("—", "waiting...", "", "", "", "", "", "", "")
    # Render newest at the TOP so the first row in the panel is the most
    # recent request. Rich truncates tables that are taller than the panel
    # at the bottom, so putting newest-first keeps the useful rows visible
    # when you ask for many (--requests 120 etc.).
    for r in reversed(requests):
        pred = r.get("pred_num")
        if pred is None:
            pred_text = Text("-", style="dim")
        elif pred <= 0.3 or pred >= 0.7:
            # confident (the model has an opinion) — good sign
            pred_text = Text(f"{pred:.3f}", style="green")
        elif 0.45 <= pred <= 0.55:
            # right in the middle — model is uncertain
            pred_text = Text(f"{pred:.3f}", style="yellow")
        else:
            pred_text = Text(f"{pred:.3f}", style="white")

        ok_text = Text("✓", style="green") if r["ok"] == "1" else Text("✗", style="red")
        # "MM-DD HH:MM:SS" from an ISO timestamp like 2026-04-23T19:47:56+00:00
        # so requests spanning multiple days are visually distinguishable.
        ts = r["ts"]
        if "T" in ts:
            date_part, time_part = ts.split("T", 1)
            md = date_part[5:] if len(date_part) >= 10 else date_part
            hms = time_part[:8]
            ts_short = f"{md} {hms}"
        else:
            ts_short = ts[-14:]

        err = r.get("err")
        err_text = Text(err, style="red bold") if err else Text("", style="dim")

        table.add_row(
            ts_short,
            r["hk"][:10],
            ok_text,
            r["n"],
            f'{r["wmin"]}/{r["wavg"]}/{r["wmax"]}',
            r["dups"],
            f'{r["lat"]}ms',
            pred_text,
            err_text,
        )

    return Panel(
        table,
        title=f"[bold]Recent validator requests · last {len(requests)} (newest first)[/bold]",
        subtitle=Text(
            "newest at top · green avg_pred = confident · yellow = uncertain (~0.5)",
            style="dim",
        ),
        border_style="magenta",
    )


def build_layout(poller, summary_path, n_requests):
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="wandb", ratio=1),
        Layout(name="requests", ratio=1),
        Layout(name="footer", size=1),
    )

    now = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
    header = Text(
        f" SN32 Miner Dashboard  ·  UID {poller.uid}  ·  now {now}",
        style="bold white on blue",
    )
    layout["header"].update(Panel(header, border_style="blue", padding=(0, 1)))

    layout["wandb"].update(render_wandb(poller))
    layout["requests"].update(render_requests(read_recent_requests(summary_path, n_requests)))

    layout["footer"].update(Text("  Ctrl-C to exit", style="dim"))
    return layout


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--uid", required=True, type=int, help="Your miner UID.")
    ap.add_argument("--project", default="itsai-dev/subnet32", help="W&B project path.")
    ap.add_argument("--wandb-interval", type=int, default=120,
                    help="Seconds between W&B polls. Default 120.")
    ap.add_argument("--max-runs", type=int, default=25,
                    help="How many recent validator-* runs to scan each poll. "
                         "Each validator usually has 1-2 recent runs, so 25 catches ~10-15 validators.")
    ap.add_argument("--summary-log", default=str(DEFAULT_SUMMARY_LOG),
                    help="Path to the miner's summary.log.")
    ap.add_argument("--requests", type=int, default=12,
                    help="How many recent requests to display.")
    ap.add_argument("--ui-refresh", type=float, default=2.0,
                    help="UI refresh interval in seconds.")
    args = ap.parse_args()

    poller = WandbPoller(
        uid=args.uid,
        project=args.project,
        interval=args.wandb_interval,
        max_runs=args.max_runs,
    )
    poller.start()

    summary_path = Path(args.summary_log)

    try:
        with Live(
            build_layout(poller, summary_path, args.requests),
            refresh_per_second=4,
            screen=True,
        ) as live:
            while True:
                live.update(build_layout(poller, summary_path, args.requests))
                time.sleep(args.ui_refresh)
    except KeyboardInterrupt:
        pass
    finally:
        poller.stop_flag = True


if __name__ == "__main__":
    main()
