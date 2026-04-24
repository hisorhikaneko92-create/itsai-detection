"""
Poll the subnet32 validator's W&B runs every N seconds and print your UID's
latest metrics as a one-liner. The headline metric is ``weighted_ood_f1`` —
that's the EMA the validator gates reward on (threshold 0.9).

Usage (run on the VPS, or anywhere with W&B API access):

    python scripts/watch_wandb_uid.py --uid 75 --interval 120

If you haven't run ``wandb login`` before, the API still works anonymously for
public projects like ``itsai-dev/subnet32``, but setting a ``WANDB_API_KEY``
(or running ``wandb login``) gives you higher rate limits.
"""
import argparse
import datetime as dt
import json
import sys
import time

try:
    import wandb
except ImportError:
    sys.exit("This script needs `pip install wandb` in the same env.")


def latest_entry_for_uid(api, project, uid, max_runs):
    """Scan the most-recent ``max_runs`` validator runs and return the newest
    uid_metrics dict for the given uid across all of them.
    """
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
            break  # first-found is the newest row in this run
        if scanned >= max_runs:
            break
    return best


def format_line(entry):
    m = entry["metrics"]
    ts = dt.datetime.utcfromtimestamp(entry["ts"]).strftime("%Y-%m-%d %H:%M:%S")
    weighted = m.get("weighted_out_of_domain_f1_score")
    weighted_s = f"{weighted:.4f}" if isinstance(weighted, (int, float)) else "-"
    gate = (
        "PASS" if isinstance(weighted, (int, float)) and weighted >= 0.9
        else "FAIL" if isinstance(weighted, (int, float))
        else "?"
    )
    return (
        f"[{ts}Z] uid={m.get('uid')} "
        f"reward={float(m.get('reward') or 0):.4f} "
        f"penalty={m.get('penalty')} "
        f"f1={float(m.get('f1_score') or 0):.3f} "
        f"ood_f1={float(m.get('out_of_domain_f1_score') or 0):.3f} "
        f"ema_ood={weighted_s} gate0.9={gate} "
        f"weight={float(m.get('weight') or 0):.4f} "
        f"run={entry['run']}"
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--uid", required=True, type=int)
    ap.add_argument("--project", default="itsai-dev/subnet32")
    ap.add_argument("--interval", type=int, default=120,
                    help="Seconds between polls. Default: 120.")
    ap.add_argument("--max-runs", type=int, default=10,
                    help="How many recent validator-* runs to scan each poll. Default: 10.")
    args = ap.parse_args()

    api = wandb.Api()
    print(f"# polling {args.project} uid={args.uid} every {args.interval}s", flush=True)
    last_ts = 0.0
    while True:
        now = dt.datetime.utcnow().strftime("%H:%M:%S")
        try:
            entry = latest_entry_for_uid(api, args.project, args.uid, max_runs=args.max_runs)
            if entry is None:
                print(f"[{now}Z] no matching row yet for uid {args.uid}", flush=True)
            elif entry["ts"] <= last_ts:
                pass
            else:
                print(format_line(entry), flush=True)
                last_ts = entry["ts"]
        except KeyboardInterrupt:
            print("\n# stopped", flush=True)
            return
        except Exception as e:
            print(f"[{now}Z] poll error: {e}", flush=True)

        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n# stopped", flush=True)
            return


if __name__ == "__main__":
    main()
