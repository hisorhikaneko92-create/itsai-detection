#!/usr/bin/env bash
# Watch upstream main detection/__init__.py for a version bump.
#
# Compares __version__ and __least_acceptable_version__ on
# https://raw.githubusercontent.com/it-s-ai/llm-detection/main/detection/__init__.py
# against the local checkout's detection/__init__.py.
#
# Behavior:
#   * Match  -> append OK line to log, exit 0.
#   * Mismatch -> append WARN line to log AND write a sticky alert file
#                 (cleared automatically once local catches up). Exit 2 so
#                 cron's MAILTO surfaces the run if mail is configured.
#   * Network/parse failure -> append ERR line, exit 1.
#
# Usage:
#   scripts/check_upstream_version.sh              # one-shot, intended for cron
#   scripts/check_upstream_version.sh --quiet      # suppress stdout, log only
#
# Suggested cron (every 30 min on the VPS that runs the miner):
#   */30 * * * * /root/llm-detection/scripts/check_upstream_version.sh --quiet
#
# Files written:
#   neurons/validator_logs/upstream_version.log    one line per run
#   neurons/validator_logs/upstream_version.alert  exists iff currently behind

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_INIT="${REPO_ROOT}/detection/__init__.py"
LOG_DIR="${REPO_ROOT}/neurons/validator_logs"
LOG_FILE="${LOG_DIR}/upstream_version.log"
ALERT_FILE="${LOG_DIR}/upstream_version.alert"
URL="${UPSTREAM_VERSION_URL:-https://raw.githubusercontent.com/it-s-ai/llm-detection/main/detection/__init__.py}"

QUIET=0
if [ "${1:-}" = "--quiet" ]; then QUIET=1; fi

mkdir -p "$LOG_DIR"

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

emit() {
  # emit <level> <msg>
  local line="[$(ts)] $1 $2"
  printf '%s\n' "$line" >> "$LOG_FILE"
  if [ "$QUIET" -eq 0 ]; then printf '%s\n' "$line"; fi
}

extract() {
  # extract <key> <file>  ->  prints version string or empty
  awk -v k="$1" '
    $0 ~ "^__"k"__[[:space:]]*=" {
      n = split($0, a, "\"")
      if (n >= 2) { print a[2]; exit }
      n = split($0, a, "\x27")
      if (n >= 2) { print a[2]; exit }
    }
  ' "$2"
}

if [ ! -f "$LOCAL_INIT" ]; then
  emit "ERR" "local file missing: $LOCAL_INIT"
  exit 1
fi

LOCAL_VER="$(extract version "$LOCAL_INIT")"
LOCAL_MIN="$(extract least_acceptable_version "$LOCAL_INIT")"

TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

if ! curl -fsSL --max-time 20 "$URL" -o "$TMP"; then
  emit "ERR" "fetch failed: $URL"
  exit 1
fi

UP_VER="$(extract version "$TMP")"
UP_MIN="$(extract least_acceptable_version "$TMP")"

if [ -z "$UP_VER" ] || [ -z "$LOCAL_VER" ]; then
  emit "ERR" "could not parse version (local='$LOCAL_VER' upstream='$UP_VER')"
  exit 1
fi

if [ "$UP_VER" = "$LOCAL_VER" ] && [ "$UP_MIN" = "$LOCAL_MIN" ]; then
  emit "OK" "version=$LOCAL_VER min=$LOCAL_MIN (upstream matches)"
  rm -f "$ALERT_FILE"
  exit 0
fi

MSG="upstream moved: __version__ ${LOCAL_VER}->${UP_VER}  __least_acceptable_version__ ${LOCAL_MIN}->${UP_MIN}"
emit "WARN" "$MSG"
{
  echo "[$(ts)] $MSG"
  echo "  local : $LOCAL_INIT"
  echo "  url   : $URL"
  echo "  action: pull/merge upstream main, restart pm2 net32-miner"
} > "$ALERT_FILE"
exit 2
