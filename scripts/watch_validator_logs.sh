#!/usr/bin/env bash
# Live dashboard for validator requests.
# Usage:
#   scripts/watch_validator_logs.sh              # refresh every 10s
#   scripts/watch_validator_logs.sh 5            # refresh every 5s
#   INTERVAL=5 TAIL=30 scripts/watch_validator_logs.sh

set -u

INTERVAL="${1:-${INTERVAL:-10}}"
TAIL="${TAIL:-15}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/sn32/bin/python}"
VIEWER="${REPO_ROOT}/scripts/view_validator_logs.py"

if [ ! -x "$PYTHON" ]; then
  PYTHON="$(command -v python3)"
fi

while true; do
  clear
  printf '╔══ Validator Dashboard ══ refresh=%ss  tail=%s  time=%s\n' \
    "$INTERVAL" "$TAIL" "$(date -u +%H:%M:%SZ)"
  printf '╚═════════════════════════════════════════════════════════════════\n\n'

  "$PYTHON" "$VIEWER" 2>/dev/null || echo "(viewer script failed)"

  printf '\n── Recent requests (last %s) ──────────────────────────────────\n' "$TAIL"
  "$PYTHON" "$VIEWER" --tail "$TAIL" 2>/dev/null || echo "(no summary.log yet)"

  printf '\n(Ctrl-C to exit)\n'
  sleep "$INTERVAL"
done
