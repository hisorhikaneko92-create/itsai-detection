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

def latest_entry_for_uid(api, project, uid, max_runs):
    best = None
    best_ts = -1.0
    scanned = 0
    for run in api.runs(project, order="-created_at"):
        if not run.name.startswith("validator-"):
            continue
        scanned += 1
        for row in run.scan_history(keys=["original_format_json", "_timestamp"]):
            payload = row.get("original_format_json")
            if not payload:
                continue
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                continue
            ts = parsed.get("timestamp") or row.get("_timestamp") or 0
            if ts <= best_ts:
                continue
            m = parsed.get("uid_metrics", {}).get(str(uid))
            if m is None:
                continue
            best = {"run": run.name, "ts": float(ts), "metrics": m}
            best_ts = float(ts)
            break  # first row returned by scan_history is already the newest
        if scanned >= max_runs:
            break
    return best


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
                    entry = latest_entry_for_uid(
                        self._api, self.project, self.uid, self.max_runs,
                    )
                    self.latest = entry
                    self.error = None
                except Exception as e:
                    self.error = str(e)[:180]
                self.last_poll = time.time()
                self.next_poll_at = time.time() + self.interval
            time.sleep(0.5)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_wandb(poller):
    table = Table.grid(padding=(0, 2))
    table.add_column(style="cyan", justify="right", min_width=16)
    table.add_column(style="white")

    if poller.last_poll is None and poller.error is None:
        table.add_row("status", Text("warming up (first poll in progress)...", style="yellow"))
    elif poller.error:
        table.add_row("status", Text(f"error: {poller.error}", style="red"))
    elif poller.latest is None:
        table.add_row("status", Text(f"no data yet for uid {poller.uid}", style="yellow"))
    else:
        m = poller.latest["metrics"]
        ts = dt.datetime.utcfromtimestamp(poller.latest["ts"]).strftime("%Y-%m-%d %H:%M:%SZ")

        reward = float(m.get("reward") or 0)
        weight = float(m.get("weight") or 0)
        f1 = float(m.get("f1_score") or 0)
        fp = float(m.get("fp_score") or 0)
        ap = float(m.get("ap_score") or 0)
        ood = float(m.get("out_of_domain_f1_score") or 0)
        ema = m.get("weighted_out_of_domain_f1_score")
        penalty = m.get("penalty")
        stake = m.get("enough_stake")

        reward_style = "bold green" if reward > 0 else "bold red"
        penalty_style = "bold green" if penalty == 1 else "bold red"

        if isinstance(ema, (int, float)):
            ema_val = float(ema)
            if ema_val >= 0.9:
                ema_text = Text(f"{ema_val:.4f}", style="bold green")
                gate_text = Text("✓ PASS", style="bold green")
            else:
                ema_text = Text(f"{ema_val:.4f}  (need +{0.9 - ema_val:.4f})", style="bold red")
                gate_text = Text("✗ FAIL", style="bold red")
        else:
            ema_text = Text("-", style="dim")
            gate_text = Text("?", style="yellow")

        table.add_row("last data", ts)
        table.add_row("run", poller.latest["run"])
        table.add_row("reward", Text(f"{reward:.4f}", style=reward_style))
        table.add_row("penalty", Text(str(penalty), style=penalty_style))
        table.add_row("enough stake", str(stake))
        table.add_row("f1 (in-domain)", f"{f1:.4f}")
        table.add_row("fp / ap", f"{fp:.4f}  /  {ap:.4f}")
        table.add_row("ood_f1 (this round)", f"{ood:.4f}")
        table.add_row("ema_ood (gated)", ema_text)
        table.add_row("gate ≥ 0.9", gate_text)
        table.add_row("weight", f"{weight:.6f}")

    # Footer inside the panel: polling status
    if poller.last_poll:
        age = int(time.time() - poller.last_poll)
        nxt = max(0, int(poller.next_poll_at - time.time()))
        sub = f"polled {age}s ago · next in {nxt}s · project {poller.project}"
    else:
        nxt = max(0, int(poller.next_poll_at - time.time()))
        sub = f"first poll in {nxt}s · project {poller.project}"

    return Panel(
        table,
        title=f"[bold]W&B metrics · UID {poller.uid}[/bold]",
        subtitle=Text(sub, style="dim"),
        border_style="cyan",
    )


def render_requests(requests):
    table = Table(expand=True, show_lines=False, header_style="bold")
    table.add_column("time", style="white", no_wrap=True)
    table.add_column("hotkey", style="cyan", no_wrap=True)
    table.add_column("ok", justify="center", no_wrap=True)
    table.add_column("texts", justify="right")
    table.add_column("words min/avg/max", justify="right")
    table.add_column("dups", justify="right")
    table.add_column("latency", justify="right")
    table.add_column("avg_pred", justify="right")

    if not requests:
        table.add_row("—", "waiting...", "", "", "", "", "", "")
    for r in requests:
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
        # HH:MM:SS from an ISO timestamp like 2026-04-23T19:47:56+00:00
        ts = r["ts"]
        ts_short = ts.split("T")[1][:8] if "T" in ts else ts[-8:]

        table.add_row(
            ts_short,
            r["hk"][:10],
            ok_text,
            r["n"],
            f'{r["wmin"]}/{r["wavg"]}/{r["wmax"]}',
            r["dups"],
            f'{r["lat"]}ms',
            pred_text,
        )

    return Panel(
        table,
        title=f"[bold]Recent validator requests · last {len(requests)}[/bold]",
        subtitle=Text("green avg_pred = confident · yellow = uncertain (~0.5)", style="dim"),
        border_style="magenta",
    )


def build_layout(poller, summary_path, n_requests):
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="wandb", size=18),
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
    ap.add_argument("--max-runs", type=int, default=10,
                    help="How many recent validator-* runs to scan each poll.")
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
