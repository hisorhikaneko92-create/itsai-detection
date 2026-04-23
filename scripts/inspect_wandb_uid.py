import argparse
import json
from collections import Counter

import wandb


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect miner metrics for a UID from the subnet32 W&B validator logs."
    )
    parser.add_argument("--uid", required=True, help="Miner UID to inspect, for example 75.")
    parser.add_argument(
        "--project",
        default="itsai-dev/subnet32",
        help="W&B entity/project path. Default: itsai-dev/subnet32",
    )
    parser.add_argument(
        "--validator-prefix",
        default="",
        help='Optional run-name prefix, for example "validator-222-" or "validator-3-".',
    )
    parser.add_argument(
        "--limit-runs",
        type=int,
        default=20,
        help="How many recent validator runs to scan. Default: 20",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Print every matching history row instead of only the newest match per run.",
    )
    return parser.parse_args()


def iter_uid_metrics(api, project, uid, validator_prefix, limit_runs, show_all):
    seen_runs = 0
    for run in api.runs(project, order="-created_at"):
        if validator_prefix and not run.name.startswith(validator_prefix):
            continue

        seen_runs += 1
        matched_this_run = False
        for row in run.scan_history(keys=["original_format_json"]):
            payload = row.get("original_format_json")
            if not payload:
                continue

            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                continue

            metrics = parsed.get("uid_metrics", {}).get(uid)
            if metrics is None:
                continue

            yield {
                "run_name": run.name,
                "run_url": run.url,
                "timestamp": parsed.get("timestamp"),
                "metrics": metrics,
            }
            matched_this_run = True
            if not show_all:
                break

        if seen_runs >= limit_runs:
            break

        if matched_this_run and not show_all:
            continue


def print_entry(entry):
    metrics = entry["metrics"]
    print("=" * 80)
    print(f'run:        {entry["run_name"]}')
    print(f'url:        {entry["run_url"]}')
    print(f'uid:        {metrics.get("uid")}')
    print(f'reward:     {metrics.get("reward")}')
    print(f'penalty:    {metrics.get("penalty")}')
    print(f'f1_score:   {metrics.get("f1_score")}')
    print(f'fp_score:   {metrics.get("fp_score")}')
    print(f'ap_score:   {metrics.get("ap_score")}')
    print(f'ood_f1:     {metrics.get("out_of_domain_f1_score")}')
    print(f'ood_moving: {metrics.get("weighted_out_of_domain_f1_score")}')
    print(f'enough_stake: {metrics.get("enough_stake")}')
    print(f'weight:     {metrics.get("weight")}')


def print_summary(entries):
    if not entries:
        print("No matching UID metrics found.")
        return

    penalties = Counter(entry["metrics"].get("penalty") for entry in entries)
    zero_penalty = sum(1 for entry in entries if entry["metrics"].get("penalty") == 0)
    zero_reward = sum(1 for entry in entries if float(entry["metrics"].get("reward", 0) or 0) == 0)
    ood_below_threshold = sum(
        1
        for entry in entries
        if entry["metrics"].get("out_of_domain_f1_score") is not None
        and float(entry["metrics"]["out_of_domain_f1_score"]) < 0.9
    )

    print("\nSummary")
    print("-" * 80)
    print(f"matches:              {len(entries)}")
    print(f"reward == 0:          {zero_reward}")
    print(f"penalty == 0:         {zero_penalty}")
    print(f"out_of_domain < 0.9:  {ood_below_threshold}")
    print(f"penalty histogram:    {dict(penalties)}")


def main():
    args = parse_args()
    api = wandb.Api()
    entries = list(
        iter_uid_metrics(
            api=api,
            project=args.project,
            uid=str(args.uid),
            validator_prefix=args.validator_prefix,
            limit_runs=args.limit_runs,
            show_all=args.show_all,
        )
    )

    for entry in entries:
        print_entry(entry)

    print_summary(entries)


if __name__ == "__main__":
    main()
