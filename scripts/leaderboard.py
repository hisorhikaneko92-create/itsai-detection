"""
SN32 miner leaderboard.

Scans recent ``validator-*`` runs on W&B, aggregates per-miner metrics across
validators, and prints a sorted table so you can compare miners at a glance.

Each validator maintains its own independent EMA (see
``detection/validator/reward.py:130-132``), so a single miner's "true" score
is really a set of per-validator scores. This script averages the latest
score each validator reported for each miner.

Typical usage::

    # One-shot: top 30 miners by mean EMA OOD F1 (the gate metric)
    python scripts/leaderboard.py --top 30 --highlight 75

    # Sorted by mean reward, only scanning validator 222's runs
    python scripts/leaderboard.py --top 20 --sort reward_mean --validator-uid 222

    # Live mode, refresh every 60 seconds
    python scripts/leaderboard.py --top 25 --highlight 75 --watch 60
"""
import argparse
import datetime as dt
import json
import sys
import time
from collections import defaultdict

try:
    import wandb
except ImportError:
    sys.exit("Install dependencies first:  pip install rich wandb")

try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ImportError:
    sys.exit("Install dependencies first:  pip install rich wandb")


def _validator_uid_from_run(run_name):
    parts = run_name.split("-")
    if len(parts) >= 2 and parts[0] == "validator":
        return parts[1]
    return None


def fetch_per_validator_snapshots(api, project, validator_uid_filter, max_runs):
    """Scan up to ``max_runs`` recent validator-* runs. For each run, grab its
    newest ``uid_metrics`` payload. Return a dict:
        { miner_uid: { validator_uid: (ts, metrics) } }
    Collapses multiple runs from the same validator to the newest.

    ``validator_uid_filter`` may be ``None`` (no filter), a single string/int
    UID (single-validator filter), or a set of UID strings (multi-validator).
    """
    if validator_uid_filter is None:
        allowed = None
    elif isinstance(validator_uid_filter, (set, frozenset, list, tuple)):
        allowed = {str(v) for v in validator_uid_filter}
    else:
        allowed = {str(validator_uid_filter)}

    by_miner = defaultdict(dict)
    scanned = 0
    for run in api.runs(project, order="-created_at"):
        if not run.name.startswith("validator-"):
            continue
        vuid = _validator_uid_from_run(run.name)
        if vuid is None:
            continue
        if allowed is not None and vuid not in allowed:
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
            ts = float(parsed.get("timestamp") or row.get("_timestamp") or 0)
            uid_metrics = parsed.get("uid_metrics", {}) or {}
            for uid_str, m in uid_metrics.items():
                try:
                    miner_uid = int(uid_str)
                except (TypeError, ValueError):
                    continue
                existing = by_miner[miner_uid].get(vuid)
                if existing is None or ts > existing[0]:
                    by_miner[miner_uid][vuid] = (ts, m)
            break  # scan_history yields newest-first; first match is enough per run

        if scanned >= max_runs:
            break
    return by_miner


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def summarize(by_miner):
    summary = []
    for uid, v_data in by_miner.items():
        metrics_list = [vd[1] for vd in v_data.values()]
        timestamps = [vd[0] for vd in v_data.values()]
        if not metrics_list:
            continue

        def _float(key):
            out = []
            for m in metrics_list:
                v = m.get(key)
                if isinstance(v, (int, float)):
                    out.append(float(v))
            return out

        rewards = _float("reward")
        emas = _float("weighted_out_of_domain_f1_score")
        f1s = _float("f1_score")
        ood_single = _float("out_of_domain_f1_score")
        weights = _float("weight")
        penalties = [m.get("penalty") for m in metrics_list]
        pass_rate = (sum(1 for p in penalties if p == 1) / len(penalties)) if penalties else 0.0

        summary.append({
            "uid": uid,
            "n_validators": len(metrics_list),
            "newest_ts": max(timestamps),
            "reward_mean": _mean(rewards) or 0.0,
            "reward_max": max(rewards) if rewards else 0.0,
            "ema_ood_mean": _mean(emas),
            "ema_ood_min": min(emas) if emas else None,
            "ema_ood_max": max(emas) if emas else None,
            "f1_mean": _mean(f1s) or 0.0,
            "ood_f1_mean": _mean(ood_single) or 0.0,
            "weight_mean": _mean(weights) or 0.0,
            "weight_max": max(weights) if weights else 0.0,
            "pass_rate": pass_rate,
        })
    return summary


def _format_age(seconds):
    s = int(max(0, seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m"
    if s < 86400:
        return f"{s // 3600}h{(s % 3600) // 60:02d}m"
    return f"{s // 86400}d{(s % 86400) // 3600:02d}h"


def build_table(summary, sort_key, top, highlight_uid, filter_uids=None):
    # Miners without the sort key get pushed to the bottom
    def _sort_val(s):
        v = s.get(sort_key)
        return float("-inf") if v is None else v

    # Rank over the full field first so filtered rows still show global rank
    full_ranked = sorted(summary, key=_sort_val, reverse=True)
    for i, s in enumerate(full_ranked, start=1):
        s["_rank"] = i

    if filter_uids:
        ranked = [s for s in full_ranked if s["uid"] in filter_uids]
    else:
        ranked = full_ranked[:top]

    now = time.time()
    table = Table(expand=True, show_lines=False, header_style="bold")
    table.add_column("rank", justify="right", no_wrap=True)
    table.add_column("uid", justify="right", no_wrap=True)
    table.add_column("reward(mean)", justify="right")
    table.add_column("ema_ood(mean)", justify="right")
    table.add_column("ema range", justify="right", style="dim")
    table.add_column("f1(mean)", justify="right")
    table.add_column("ood_f1(mean)", justify="right")
    table.add_column("weight(mean)", justify="right")
    table.add_column("pass%", justify="right")
    table.add_column("#val", justify="right", style="dim")
    table.add_column("latest", justify="right", style="dim", no_wrap=True)

    for i, s in enumerate(ranked, start=1):
        highlight = highlight_uid is not None and s["uid"] == int(highlight_uid)
        row_style = "bold yellow" if highlight else ""

        ema = s["ema_ood_mean"]
        if ema is None:
            ema_text = Text("-", style="dim")
        elif ema >= 0.9:
            ema_text = Text(f"{ema:.4f}", style="bold green")
        else:
            ema_text = Text(f"{ema:.4f}", style="red")

        if s["ema_ood_min"] is not None and s["ema_ood_max"] is not None:
            ema_range = f"{s['ema_ood_min']:.3f}…{s['ema_ood_max']:.3f}"
        else:
            ema_range = "-"

        reward = s["reward_mean"]
        reward_text = Text(f"{reward:.4f}", style="green" if reward > 0 else "red")

        pass_pct = int(s["pass_rate"] * 100)
        pass_style = "green" if pass_pct >= 80 else ("yellow" if pass_pct >= 50 else "red")

        table.add_row(
            str(s["_rank"]),
            str(s["uid"]),
            reward_text,
            ema_text,
            ema_range,
            f"{s['f1_mean']:.4f}",
            f"{s['ood_f1_mean']:.4f}",
            f"{s['weight_mean']:.6f}",
            Text(f"{pass_pct}%", style=pass_style),
            str(s["n_validators"]),
            _format_age(now - s["newest_ts"]),
            style=row_style,
        )

    return table, ranked


def find_my_rank(summary, sort_key, uid):
    if uid is None:
        return None
    ranked = sorted(
        summary,
        key=lambda s: (float("-inf") if s.get(sort_key) is None else s[sort_key]),
        reverse=True,
    )
    for i, s in enumerate(ranked, start=1):
        if s["uid"] == int(uid):
            return i, s, len(ranked)
    return None


def build_panel(summary, args, filter_uids=None):
    table, ranked = build_table(
        summary, args.sort, args.top, args.highlight, filter_uids=filter_uids,
    )

    my_rank_info = find_my_rank(summary, args.sort, args.highlight)
    now_iso = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")

    if filter_uids:
        missing = sorted(filter_uids - {s["uid"] for s in ranked})
    else:
        missing = []

    if my_rank_info is not None:
        rank, my_row, total = my_rank_info
        mark = "✓" if my_row.get("ema_ood_mean") and my_row["ema_ood_mean"] >= 0.9 else "✗"
        my_line = (f"your uid {args.highlight}: rank {rank}/{total} · "
                   f"reward={my_row['reward_mean']:.4f} · "
                   f"ema_ood={my_row['ema_ood_mean']:.4f} " if my_row['ema_ood_mean'] is not None
                   else f"your uid {args.highlight}: rank {rank}/{total} · reward={my_row['reward_mean']:.4f} · ema_ood=- ")
        my_line += f"· gate {mark}"
    elif args.highlight is not None:
        my_line = f"your uid {args.highlight}: not found in any scanned run"
    else:
        my_line = ""

    if filter_uids:
        mode_s = f"uids={','.join(str(u) for u in sorted(filter_uids))}"
    else:
        mode_s = f"top {args.top}"
    subtitle_parts = [f"sort={args.sort}", mode_s,
                      f"project {args.project}",
                      f"{len(summary)} miners across {args.max_runs} recent runs",
                      now_iso]
    if missing:
        subtitle_parts.insert(2, f"not found: {','.join(str(u) for u in missing)}")
    subtitle = Text(" · ".join(subtitle_parts), style="dim")

    title = "[bold]SN32 miner leaderboard[/bold]"
    if args.validator_uid:
        title += f"  (validator-{args.validator_uid} only)"
    if my_line:
        title += f"  —  {my_line}"

    return Panel(table, title=title, subtitle=subtitle, border_style="cyan")


METRIC_FIELDS = {
    # cli name -> json field in uid_metrics, and "higher is better" flag
    "ema_ood": ("weighted_out_of_domain_f1_score", True),
    "reward":  ("reward", True),
    "f1":      ("f1_score", True),
    "ood_f1":  ("out_of_domain_f1_score", True),
    "weight":  ("weight", True),
}

GATE_METRICS = {"ema_ood", "f1", "ood_f1"}  # metrics with a 0.9 threshold semantics


def _metric_cell(value, metric_key):
    if value is None:
        return Text("-", style="dim")
    if metric_key in GATE_METRICS:
        style = "bold green" if value >= 0.9 else "red"
    elif metric_key == "reward":
        style = "green" if value > 0 else "red"
    else:
        style = ""
    return Text(f"{value:.4f}", style=style)


def build_matrix_panel(by_miner, args, filter_uids, filter_validator_uids):
    metric_key = args.metric
    metric_field = METRIC_FIELDS[metric_key][0]

    # Which miners to show as rows
    if filter_uids:
        miner_uids = [u for u in sorted(filter_uids) if u in by_miner]
        missing_miners = sorted(filter_uids - set(miner_uids))
    else:
        miner_uids = sorted(by_miner.keys())
        missing_miners = []

    # Which validators to show as columns — union of validators that scored any row
    validators_present = set()
    for mu in miner_uids:
        validators_present.update(by_miner[mu].keys())

    if filter_validator_uids:
        wanted = {str(v) for v in filter_validator_uids}
        validator_uids = sorted(
            [v for v in validators_present if v in wanted],
            key=lambda x: int(x) if x.isdigit() else 10**9,
        )
        missing_validators = sorted(wanted - validators_present)
    else:
        validator_uids = sorted(
            validators_present,
            key=lambda x: int(x) if x.isdigit() else 10**9,
        )
        missing_validators = []

    # Build rows
    rows = []
    for mu in miner_uids:
        values = []
        for vu in validator_uids:
            cell = by_miner[mu].get(vu)
            if cell is None:
                values.append(None)
            else:
                v = cell[1].get(metric_field)
                values.append(float(v) if isinstance(v, (int, float)) else None)
        present = [v for v in values if v is not None]
        mean_v = sum(present) / len(present) if present else None
        rows.append({"uid": mu, "values": values, "mean": mean_v})

    rows.sort(
        key=lambda r: (r["mean"] if r["mean"] is not None else float("-inf")),
        reverse=True,
    )

    # Render the table
    table = Table(expand=True, show_lines=False, header_style="bold")
    table.add_column("uid", justify="right", no_wrap=True)
    for vu in validator_uids:
        table.add_column(f"val-{vu}", justify="right", no_wrap=True)
    table.add_column("mean", justify="right", style="bold")
    table.add_column("cov", justify="right", style="dim",
                     no_wrap=True)  # coverage: how many validators scored this miner

    for row in rows:
        highlight = args.highlight is not None and row["uid"] == int(args.highlight)
        row_style = "bold yellow" if highlight else ""

        cells = [str(row["uid"])]
        for v in row["values"]:
            cells.append(_metric_cell(v, metric_key))
        cells.append(_metric_cell(row["mean"], metric_key))
        n_covered = sum(1 for v in row["values"] if v is not None)
        cells.append(f"{n_covered}/{len(validator_uids)}")
        table.add_row(*cells, style=row_style)

    now_iso = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
    parts = [f"metric={metric_key}",
             f"rows={len(rows)} · cols={len(validator_uids)}",
             f"project {args.project}", now_iso]
    if missing_miners:
        parts.insert(1, f"miners not found: {','.join(str(u) for u in missing_miners)}")
    if missing_validators:
        parts.insert(1, f"validators not found: {','.join(missing_validators)}")
    subtitle = Text(" · ".join(parts), style="dim")

    title = "[bold]SN32 miner × validator matrix[/bold]"
    if args.highlight is not None:
        title += f"  —  your uid {args.highlight}"

    return Panel(table, title=title, subtitle=subtitle, border_style="magenta")


def build_per_validator_view(by_miner, args, filter_uids, filter_validator_uids):
    """One Panel per validator. Each panel is a table with rows = selected
    miner UIDs and columns = the full metric set (reward / penalty / f1 / fp /
    ap / ood_f1 / ema_ood / weight / age) plus an average row at the bottom.
    """
    if filter_uids:
        miner_uids = sorted(filter_uids)
    else:
        miner_uids = sorted(by_miner.keys())

    validators_present = set()
    for mu in miner_uids:
        validators_present.update(by_miner.get(mu, {}).keys())

    if filter_validator_uids:
        wanted = {str(v) for v in filter_validator_uids}
        validator_uids = sorted(
            [v for v in validators_present if v in wanted],
            key=lambda x: int(x) if x.isdigit() else 10**9,
        )
        missing_validators = sorted(wanted - validators_present)
    else:
        validator_uids = sorted(
            validators_present, key=lambda x: int(x) if x.isdigit() else 10**9,
        )
        missing_validators = []

    now_iso = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
    header = Panel(
        Text(
            f" SN32 per-validator breakdown · "
            f"{len(miner_uids)} miners × {len(validator_uids)} validators · {now_iso}",
            style="bold white on blue",
        ),
        border_style="blue",
        padding=(0, 1),
    )

    renderables = [header]
    if missing_validators:
        renderables.append(
            Panel(Text(f"Validators not found in recent runs: {', '.join(missing_validators)}",
                       style="yellow"),
                  border_style="yellow")
        )

    now = time.time()
    for vu in validator_uids:
        table = Table(expand=True, show_lines=False, header_style="bold")
        table.add_column("uid", justify="right", no_wrap=True)
        table.add_column("reward", justify="right")
        table.add_column("pen", justify="center")
        table.add_column("f1", justify="right")
        table.add_column("fp", justify="right")
        table.add_column("ap", justify="right")
        table.add_column("ood_f1", justify="right")
        table.add_column("ema_ood", justify="right")
        table.add_column("weight", justify="right")
        table.add_column("age", justify="right", style="dim", no_wrap=True)

        rewards, f1s, fps, aps, oods, emas, weights = [], [], [], [], [], [], []
        newest_ts = None

        for mu in miner_uids:
            cell = by_miner.get(mu, {}).get(vu)
            row_style = (
                "bold yellow" if args.highlight is not None and mu == int(args.highlight)
                else ""
            )

            if cell is None:
                table.add_row(
                    str(mu),
                    *([Text("-", style="dim")] * 8),
                    "-",
                    style=row_style,
                )
                continue

            ts, m = cell
            reward = float(m.get("reward") or 0)
            penalty = m.get("penalty")
            f1 = float(m.get("f1_score") or 0)
            fp = float(m.get("fp_score") or 0)
            ap = float(m.get("ap_score") or 0)
            ood = float(m.get("out_of_domain_f1_score") or 0)
            ema = m.get("weighted_out_of_domain_f1_score")
            weight = float(m.get("weight") or 0)

            rewards.append(reward)
            f1s.append(f1); fps.append(fp); aps.append(ap); oods.append(ood)
            weights.append(weight)
            if isinstance(ema, (int, float)):
                emas.append(float(ema))

            newest_ts = ts if newest_ts is None else max(newest_ts, ts)

            reward_text = Text(f"{reward:.4f}", style="green" if reward > 0 else "red")
            pen_text = Text(
                str(penalty) if penalty is not None else "-",
                style="green" if penalty == 1 else "red",
            )
            if isinstance(ema, (int, float)):
                ema_val = float(ema)
                ema_text = (
                    Text(f"{ema_val:.4f}", style="bold green") if ema_val >= 0.9
                    else Text(f"{ema_val:.4f}", style="red")
                )
            else:
                ema_text = Text("-", style="dim")

            table.add_row(
                str(mu),
                reward_text,
                pen_text,
                f"{f1:.4f}",
                f"{fp:.4f}",
                f"{ap:.4f}",
                f"{ood:.4f}",
                ema_text,
                f"{weight:.6f}",
                _format_age(now - ts),
                style=row_style,
            )

        # Bottom "avg" row across the miners that appeared
        if rewards:
            mean_ema = (sum(emas) / len(emas)) if emas else None
            if mean_ema is None:
                mean_ema_text = Text("-", style="dim")
            elif mean_ema >= 0.9:
                mean_ema_text = Text(f"{mean_ema:.4f}", style="bold green")
            else:
                mean_ema_text = Text(f"{mean_ema:.4f}", style="red")

            table.add_row(
                Text("avg", style="bold cyan"),
                Text(f"{sum(rewards)/len(rewards):.4f}",
                     style="bold " + ("green" if sum(rewards) > 0 else "red")),
                "",
                Text(f"{sum(f1s)/len(f1s):.4f}", style="bold"),
                Text(f"{sum(fps)/len(fps):.4f}", style="bold"),
                Text(f"{sum(aps)/len(aps):.4f}", style="bold"),
                Text(f"{sum(oods)/len(oods):.4f}", style="bold"),
                mean_ema_text,
                Text(f"{sum(weights)/len(weights):.6f}", style="bold"),
                "",
            )

        age_s = _format_age(now - newest_ts) if newest_ts else "—"
        panel_title = f"[bold]Validator {vu}[/bold]  ·  latest data {age_s} ago"
        renderables.append(Panel(table, title=panel_title, border_style="cyan"))

    renderables.append(
        Text(f"  Ctrl-C to exit  ·  project {args.project}  ·  polled just now", style="dim")
    )
    return Group(*renderables)


def run_once(api, args, filter_uids=None, filter_validator_uids=None):
    v_filter = filter_validator_uids if filter_validator_uids else args.validator_uid
    by_miner = fetch_per_validator_snapshots(
        api, args.project, v_filter, args.max_runs,
    )
    if args.per_validator:
        return build_per_validator_view(by_miner, args, filter_uids, filter_validator_uids)
    if args.matrix:
        return build_matrix_panel(by_miner, args, filter_uids, filter_validator_uids)
    summary = summarize(by_miner)
    return build_panel(summary, args, filter_uids=filter_uids)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--project", default="itsai-dev/subnet32")
    ap.add_argument("--validator-uid", default=None,
                    help="Optional: restrict to runs from a single validator UID, e.g. 222. "
                         "Use --validator-uids for multiple.")
    ap.add_argument("--validator-uids", default=None,
                    help="Comma-separated validator UIDs to include, e.g. '222,3,48'. "
                         "Combine with --matrix for a miners × validators grid.")
    ap.add_argument("--matrix", action="store_true",
                    help="Render a miner × validator matrix instead of the aggregate "
                         "leaderboard. Rows = miner UIDs, columns = validator UIDs, "
                         "cells = --metric value.")
    ap.add_argument("--per-validator", action="store_true",
                    help="Render one table per validator, each showing the full metric "
                         "set (reward / penalty / f1 / fp / ap / ood_f1 / ema_ood / "
                         "weight) for every selected miner, with an 'avg' row.")
    ap.add_argument("--metric", default="ema_ood",
                    choices=list(METRIC_FIELDS.keys()),
                    help="Which per-cell metric to display in --matrix mode. "
                         "Default: ema_ood (the gate metric).")
    ap.add_argument("--max-runs", type=int, default=25,
                    help="How many recent validator-* runs to scan. Default 25.")
    ap.add_argument("--top", type=int, default=25,
                    help="Show top N miners. Ignored if --uids is set. Default 25.")
    ap.add_argument("--uids", default=None,
                    help="Comma-separated list of UIDs to display, e.g. '75,12,47,123'. "
                         "When set, only these UIDs appear in the table (sorted by --sort, "
                         "showing each one's global rank in the full field).")
    ap.add_argument("--highlight", type=int, default=None,
                    help="Your miner UID — highlighted in the table and summarized in the title.")
    ap.add_argument(
        "--sort",
        default="ema_ood_mean",
        choices=["ema_ood_mean", "reward_mean", "f1_mean", "weight_mean", "pass_rate", "ood_f1_mean"],
        help="Column to sort by. Default ema_ood_mean (the gate metric).",
    )
    ap.add_argument("--watch", type=int, default=0,
                    help="If > 0, refresh every N seconds in a live view. Default: one-shot.")
    args = ap.parse_args()

    filter_uids = None
    if args.uids:
        try:
            filter_uids = {int(u.strip()) for u in args.uids.split(",") if u.strip()}
        except ValueError as e:
            sys.exit(f"--uids must be comma-separated integers: {e}")

    # --validator-uids (plural) overrides --validator-uid for filtering.
    filter_validator_uids = None
    if args.validator_uids:
        filter_validator_uids = {u.strip() for u in args.validator_uids.split(",") if u.strip()}
    elif args.validator_uid:
        filter_validator_uids = {str(args.validator_uid)}

    if args.matrix and not filter_uids and not filter_validator_uids:
        print("Note: --matrix without --uids / --validator-uids will show every "
              "miner and validator it finds; consider adding filters for a compact view.",
              file=sys.stderr)

    api = wandb.Api()
    console = Console()

    def render():
        return run_once(
            api, args,
            filter_uids=filter_uids,
            filter_validator_uids=filter_validator_uids,
        )

    if args.watch <= 0:
        console.print(render())
        return

    try:
        with Live(render(), refresh_per_second=1, screen=False, console=console) as live:
            while True:
                time.sleep(args.watch)
                try:
                    live.update(render())
                except Exception as e:
                    console.print(Text(f"refresh error: {e}", style="red"))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
