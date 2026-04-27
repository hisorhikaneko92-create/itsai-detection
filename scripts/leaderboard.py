"""
SN32 miner leaderboard.

Scans recent ``validator-*`` runs on W&B, aggregates per-miner metrics across
validators, and prints a sorted table so you can compare miners at a glance.

Each validator maintains its own independent EMA (see
``detection/validator/reward.py:130-132``), so a single miner's "true" score
is really a set of per-validator scores.

The default filtering and aggregation now mirrors the public
``ai-detection-leaderboard`` (gradio app at
``/root/ai-detection-leaderboard/app.py``):

* server-side W&B filters drop runs older than 24h, runs not in state
  ``finished``, and runs from the explicitly-excluded validator UIDs
  (86, 50, 106).
* the **Yuma main validator** (UID 222) contributes the **average of its
  five most-recent runs** for each miner; all other validators contribute
  their newest run only.

These match what the public leaderboard counts. Override any of them via
flags below to inspect raw / unfiltered data.

Typical usage::

    # Top 30 miners by mean EMA OOD F1 (the gate metric), official-style filters
    python scripts/leaderboard.py --top 30 --highlight 75

    # Match the public site exactly (sort by reward, no EMA emphasis)
    python scripts/leaderboard.py --official --top 20 --highlight 75

    # Inspect raw / unfiltered W&B data (useful when debugging)
    python scripts/leaderboard.py --max-age-hours 0 --state any \\
        --exclude-validators "" --main-validator 0

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


def _build_wandb_filters(max_age_hours, state, excluded_validator_uids):
    """Build the MongoDB-style ``filters=`` dict for ``wandb.Api.runs``.

    Mirrors ``ai-detection-leaderboard/app.py:360-378``: server-side
    drop-old-runs, drop-not-finished, and drop-blacklisted-validators.
    Returns ``None`` when nothing is configured (no filter passed to wandb).
    """
    clauses = []
    if max_age_hours and max_age_hours > 0:
        cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=max_age_hours)
        clauses.append({"created_at": {"$gte": cutoff.isoformat()}})
    if state and state.lower() != "any":
        states = [s.strip() for s in state.split(",") if s.strip()]
        if states:
            clauses.append({"state": {"$in": states}})
    if excluded_validator_uids:
        # config.uid is what the validator wrote into its W&B run config —
        # same field the public leaderboard filters on.
        excluded_ints = []
        for v in excluded_validator_uids:
            try:
                excluded_ints.append(int(v))
            except (TypeError, ValueError):
                pass
        if excluded_ints:
            clauses.append({"config.uid": {"$nin": excluded_ints}})
    if not clauses:
        return None
    return {"$and": clauses}


def _average_metrics(metrics_list):
    """Mean over a list of ``uid_metrics`` dicts.

    Numeric fields (reward, f1_score, fp_score, ap_score, penalty,
    weighted_out_of_domain_f1_score, weight, ...) are averaged. Non-numeric
    fields fall back to the value from the newest run (first list element).
    Mirrors ``ai-detection-leaderboard/app.py:260-281`` but generalised so
    we don't lose less-common keys like ``out_of_domain_f1_score``.
    """
    if not metrics_list:
        return {}
    if len(metrics_list) == 1:
        return dict(metrics_list[0])

    out = dict(metrics_list[0])  # newest-run wins for non-numeric fields
    sums = defaultdict(float)
    counts = defaultdict(int)
    for m in metrics_list:
        for k, v in m.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                sums[k] += float(v)
                counts[k] += 1
    for k, total in sums.items():
        out[k] = total / counts[k]
    return out


def fetch_per_validator_snapshots(
    api,
    project,
    validator_uid_filter,
    max_runs,
    *,
    wandb_filters=None,
    excluded_validator_uids=None,
    main_validator_uid=None,
    main_validator_keep_n=5,
):
    """Scan recent ``validator-*`` runs and return:

        { miner_uid: { validator_uid: (ts, metrics) } }

    For ``main_validator_uid`` (typically 222 — the Yuma main validator),
    up to ``main_validator_keep_n`` newest runs are collected and **averaged
    together**, matching the public ``ai-detection-leaderboard`` behaviour.
    Other validators contribute their newest run only.

    Server-side filtering ``wandb_filters`` (built by ``_build_wandb_filters``)
    drops runs by age / state / excluded-config-uid before any client work.

    ``validator_uid_filter`` is the user's per-view filter (``None`` / single
    UID / set of UIDs) and is applied client-side after scanning.
    """
    if validator_uid_filter is None:
        allowed = None
    elif isinstance(validator_uid_filter, (set, frozenset, list, tuple)):
        allowed = {str(v) for v in validator_uid_filter}
    else:
        allowed = {str(validator_uid_filter)}

    excluded = {str(v) for v in (excluded_validator_uids or [])}
    main_uid_str = str(main_validator_uid) if main_validator_uid else None

    by_miner = defaultdict(dict)
    # main-validator buckets — collected raw, averaged at the end
    main_runs_per_miner = defaultdict(list)
    main_runs_processed = 0

    runs_iter = (
        api.runs(project, order="-created_at", filters=wandb_filters)
        if wandb_filters
        else api.runs(project, order="-created_at")
    )

    scanned = 0
    for run in runs_iter:
        if not run.name.startswith("validator-"):
            continue
        vuid = _validator_uid_from_run(run.name)
        if vuid is None:
            continue
        # Client-side guard for the excluded list, in case a validator's
        # config.uid disagreed with the run name (rare but possible).
        if vuid in excluded:
            continue
        if allowed is not None and vuid not in allowed:
            continue

        is_main = main_uid_str is not None and vuid == main_uid_str
        # Stop pulling main-validator history once we have enough — but keep
        # iterating so other validators after it still get scanned.
        if is_main and main_runs_processed >= main_validator_keep_n:
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
                if is_main:
                    main_runs_per_miner[miner_uid].append((ts, m))
                else:
                    existing = by_miner[miner_uid].get(vuid)
                    if existing is None or ts > existing[0]:
                        by_miner[miner_uid][vuid] = (ts, m)
            if is_main:
                main_runs_processed += 1
            break  # scan_history yields newest-first; first match is enough per run

        if scanned >= max_runs:
            break

    # Average the main validator's accumulated runs and merge them in.
    if main_uid_str is not None and main_runs_per_miner:
        for miner_uid, runs in main_runs_per_miner.items():
            runs.sort(reverse=True, key=lambda x: x[0])
            top = runs[:main_validator_keep_n]
            if not top:
                continue
            avg_metrics = _average_metrics([m for _, m in top])
            newest_ts = top[0][0]
            by_miner[miner_uid][main_uid_str] = (newest_ts, avg_metrics)

    return by_miner


def fetch_uid_history_per_validator(
    api,
    project,
    uid,
    n_per_validator,
    max_runs,
    *,
    wandb_filters=None,
    excluded_validator_uids=None,
):
    """For one specific miner ``uid``, return the most recent ``n_per_validator``
    raw scoring rounds reported by each validator. Useful for inspecting how a
    single miner's score is moving round-by-round, *without* the per-validator
    averaging applied by the main leaderboard.

    Returns ``{ validator_uid: [(ts, metrics), ...] }`` — newest first per
    validator, capped at ``n_per_validator`` rows each.
    """
    excluded = {str(v) for v in (excluded_validator_uids or [])}
    history = defaultdict(list)

    runs_iter = (
        api.runs(project, order="-created_at", filters=wandb_filters)
        if wandb_filters
        else api.runs(project, order="-created_at")
    )

    scanned = 0
    for run in runs_iter:
        if not run.name.startswith("validator-"):
            continue
        vuid = _validator_uid_from_run(run.name)
        if vuid is None or vuid in excluded:
            continue
        # Already have enough rounds from this validator -> skip its older runs
        if len(history.get(vuid, [])) >= n_per_validator:
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
            uid_metrics = parsed.get("uid_metrics", {}) or {}
            m = uid_metrics.get(str(uid))
            if m is None:
                continue
            ts = float(parsed.get("timestamp") or row.get("_timestamp") or 0)
            history[vuid].append((ts, m))
            if len(history[vuid]) >= n_per_validator:
                break

        if scanned >= max_runs:
            break

    # Each list is already newest-first, but enforce + cap to be safe
    for vuid in list(history.keys()):
        history[vuid].sort(reverse=True, key=lambda x: x[0])
        history[vuid] = history[vuid][:n_per_validator]

    return dict(history)


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


def _format_active_filters(args):
    """Compact one-liner describing which official-mode filters are active.
    Goes into the panel subtitle so users can tell at a glance whether they
    are looking at the public-site-equivalent view or raw data.
    """
    parts = []
    if args.max_age_hours and args.max_age_hours > 0:
        if args.max_age_hours == int(args.max_age_hours):
            parts.append(f"age≤{int(args.max_age_hours)}h")
        else:
            parts.append(f"age≤{args.max_age_hours}h")
    else:
        parts.append("age=any")
    if args.state and args.state.lower() != "any":
        parts.append(f"state={args.state}")
    excluded = getattr(args, "_excluded_validator_uids", None) or set()
    if excluded:
        parts.append(f"exclude={','.join(sorted(excluded, key=lambda x: int(x)))}")
    if args.main_validator:
        parts.append(f"val{args.main_validator}-avg={args.main_validator_runs}")
    return " ".join(parts) if parts else "no filters"


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
                      _format_active_filters(args),
                      f"project {args.project}",
                      f"{len(summary)} miners across {args.max_runs} recent runs",
                      now_iso]
    if missing:
        subtitle_parts.insert(3, f"not found: {','.join(str(u) for u in missing)}")
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


def build_uid_history_panel(history, uid, n_per_validator):
    """Per-UID round-by-round history. One row per (validator, round), so the
    user can see each validator's individual recent scores instead of the
    averaged number that lands in the leaderboard.
    """
    table = Table(expand=True, show_lines=False, header_style="bold")
    table.add_column("validator", justify="right", no_wrap=True)
    table.add_column("round time (UTC)", style="white", no_wrap=True)
    table.add_column("reward", justify="right")
    table.add_column("f1", justify="right")
    table.add_column("ood_f1", justify="right")
    table.add_column("ema_ood", justify="right")
    table.add_column("weight", justify="right")
    table.add_column("pen", justify="center")

    if not history:
        table.add_row("—", f"no scoring rounds found for UID {uid}",
                      "", "", "", "", "", "")
    else:
        for vuid in sorted(history.keys(),
                           key=lambda x: int(x) if x.isdigit() else 10**9):
            for ts, m in history[vuid]:
                if ts:
                    round_time = dt.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
                else:
                    round_time = "—"
                reward = float(m.get("reward") or 0)
                f1 = float(m.get("f1_score") or 0)
                ood = float(m.get("out_of_domain_f1_score") or 0)
                ema = m.get("weighted_out_of_domain_f1_score")
                weight = float(m.get("weight") or 0)
                penalty = m.get("penalty")

                reward_text = Text(f"{reward:.4f}",
                                   style="green" if reward > 0 else "red")
                if isinstance(ema, (int, float)):
                    ema_val = float(ema)
                    ema_text = (Text(f"{ema_val:.4f}", style="bold green")
                                if ema_val >= 0.9
                                else Text(f"{ema_val:.4f}", style="red"))
                else:
                    ema_text = Text("-", style="dim")
                pen_text = Text(
                    str(penalty) if penalty is not None else "-",
                    style="green" if penalty == 1 else "red",
                )
                weight_text = Text(f"{weight:.6f}",
                                   style="bold green" if weight > 0 else "dim")
                table.add_row(
                    str(vuid),
                    round_time,
                    reward_text,
                    f"{f1:.4f}",
                    f"{ood:.4f}",
                    ema_text,
                    weight_text,
                    pen_text,
                )

    title = (f"[bold]UID {uid} · last {n_per_validator} round(s) per validator "
             "(raw, not averaged)[/bold]")
    return Panel(table, title=title, border_style="magenta")


def run_once(api, args, filter_uids=None, filter_validator_uids=None):
    v_filter = filter_validator_uids if filter_validator_uids else args.validator_uid
    wandb_filters = _build_wandb_filters(
        args.max_age_hours, args.state, args._excluded_validator_uids,
    )
    by_miner = fetch_per_validator_snapshots(
        api,
        args.project,
        v_filter,
        args.max_runs,
        wandb_filters=wandb_filters,
        excluded_validator_uids=args._excluded_validator_uids,
        main_validator_uid=(args.main_validator if args.main_validator else None),
        main_validator_keep_n=args.main_validator_runs,
    )
    if args.per_validator:
        main_view = build_per_validator_view(by_miner, args, filter_uids, filter_validator_uids)
    elif args.matrix:
        main_view = build_matrix_panel(by_miner, args, filter_uids, filter_validator_uids)
    else:
        summary = summarize(by_miner)
        main_view = build_panel(summary, args, filter_uids=filter_uids)

    # Per-UID round-by-round history below the main view, when requested
    if args.highlight is not None and args.history_rounds > 0:
        history = fetch_uid_history_per_validator(
            api,
            args.project,
            args.highlight,
            n_per_validator=args.history_rounds,
            max_runs=args.max_runs,
            wandb_filters=wandb_filters,
            excluded_validator_uids=args._excluded_validator_uids,
        )
        history_panel = build_uid_history_panel(
            history, args.highlight, args.history_rounds,
        )
        return Group(main_view, history_panel)

    return main_view


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
        default="weight_mean",
        choices=["ema_ood_mean", "reward_mean", "f1_mean", "weight_mean", "pass_rate", "ood_f1_mean"],
        help="Column to sort by. Default weight_mean — the on-chain weight is "
             "what actually decides emissions, so this is the truest ranking. "
             "Use ema_ood_mean to inspect the gate metric instead.",
    )
    ap.add_argument("--history-rounds", type=int, default=2,
                    help="When --highlight is set, show this many recent rounds "
                         "per validator for that UID in a separate panel below "
                         "the leaderboard. Default 2. Set to 0 to hide.")
    ap.add_argument("--watch", type=int, default=0,
                    help="If > 0, refresh every N seconds in a live view. Default: one-shot.")

    # ---- Public-leaderboard parity controls ----------------------------------
    ap.add_argument("--max-age-hours", type=float, default=24,
                    help="Drop validator runs older than this. Default 24 hours, "
                         "matching the public ai-detection-leaderboard. Set to 0 "
                         "to disable the time filter.")
    ap.add_argument("--state", default="finished",
                    help="Comma-separated W&B run states to include. Default 'finished' "
                         "(matches the public site). Use 'any' to include 'running' / "
                         "'crashed' / 'killed' runs too.")
    ap.add_argument("--exclude-validators", default="86,50,106",
                    help="Comma-separated validator UIDs to drop. Default '86,50,106' "
                         "matches the public leaderboard's blacklist. Empty string "
                         "to include every validator.")
    ap.add_argument("--main-validator", type=int, default=222,
                    help="Validator UID treated as the 'main' (Yuma) validator. Up to "
                         "--main-validator-runs of its newest runs are averaged together "
                         "before scoring, matching the public leaderboard. Set to 0 to "
                         "disable averaging (every validator counted as one latest run).")
    ap.add_argument("--main-validator-runs", type=int, default=5,
                    help="How many of --main-validator's newest runs to average. "
                         "Default 5 (matches the public site).")
    ap.add_argument("--official", action="store_true",
                    help="Public-leaderboard parity preset: forces --exclude-validators "
                         "'86,50,106', --main-validator 222, --main-validator-runs 5, "
                         "and switches --sort to reward_mean unless explicitly set.")

    args = ap.parse_args()

    if args.official:
        # Lock in the public site's filter and aggregation defaults
        args.exclude_validators = "86,50,106"
        args.main_validator = 222
        args.main_validator_runs = 5
        # Public sorts by reward — only override if the user kept our default
        if args.sort in ("weight_mean", "ema_ood_mean"):
            args.sort = "reward_mean"

    args._excluded_validator_uids = set()
    if args.exclude_validators:
        for piece in args.exclude_validators.split(","):
            piece = piece.strip()
            if not piece:
                continue
            try:
                int(piece)  # validate
                args._excluded_validator_uids.add(piece)
            except ValueError:
                sys.exit(f"--exclude-validators must be comma-separated integers: got {piece!r}")

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
