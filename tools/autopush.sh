#!/usr/bin/env bash
# Periodically snapshot catalog progress to both remotes while a run is in flight.
#
# The generation run writes files for hours. Without this, a crash or a quota
# wall leaves everything since the last manual commit only on this disk.
#
#   tools/autopush.sh [interval_seconds]
#
# Stops when the generation run is gone AND there is nothing left to commit.
set -uo pipefail
cd "$(dirname "$0")/.."

INTERVAL=${1:-1800}
REMOTES=${REMOTES:-"origin github"}

while true; do
  ./gen_index.sh >/dev/null 2>&1
  # Record the audit verdict in the message rather than gating on it: a mid-run
  # snapshot is worth having even when a page is malformed, and the next pass
  # usually rewrites it anyway.
  verdict=$(python3 tools/audit.py 2>&1 | tail -1 | sed 's/^-- //; s/ --$//')
  n=$(find topics -name '*.md' ! -name 'README.md' | wc -l | tr -d ' ')

  if [ -n "$(git status --porcelain)" ]; then
    git add -A
    git commit -q -m "Catalog progress: ${n} problems

Automated snapshot during the generation run. Audit: ${verdict}."
    for r in $REMOTES; do
      git push -q "$r" HEAD 2>&1 | tail -2
      echo "$(date +%H:%M) pushed ${n} problems to ${r}"
    done
  else
    echo "$(date +%H:%M) no change at ${n} problems"
  fi

  if ! pgrep -f 'run_catalog.sh' >/dev/null && [ -z "$(git status --porcelain)" ]; then
    echo "$(date +%H:%M) run finished and tree clean; autopush exiting at ${n}"
    break
  fi
  sleep "$INTERVAL"
done
