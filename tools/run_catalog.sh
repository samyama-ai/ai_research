#!/usr/bin/env bash
# Drive the full catalog generation to a target file count.
#
# Backend is the claude CLI by default; CATALOG_BACKEND=agy selects the original.
# The two are interchangeable print-mode CLIs, so only argv differs.
#
# The run is resumable in every stage. Problem files that exist are skipped, the
# candidate pool is reused rather than re-discovered, and a quota wall aborts the
# pass loop (exit 2) instead of spinning against it. Re-running after a reset
# picks up exactly where it stopped.
#
#   tools/run_catalog.sh [target] [passes]
#
# Env: CATALOG_BACKEND (claude|agy)  AGY_CONCURRENCY  PER_TOPIC  TOPUP  POOL
set -uo pipefail
cd "$(dirname "$0")/.."

TARGET=${1:-1000}
PASSES=${2:-8}
PER_TOPIC=${PER_TOPIC:-140}
TOPUP=${TOPUP:-30}
POOL=${POOL:-$((TARGET + 250))}     # candidates to hold, ahead of the file target
LOG=${LOG:-catalog-run.log}
export CATALOG_BACKEND=${CATALOG_BACKEND:-claude}

count() { find topics -name '*.md' ! -name 'README.md' | wc -l | tr -d ' '; }
pool()  { python3 - <<'PY'
import glob, json
print(sum(len(json.load(open(f))) for f in glob.glob("topics/*/candidates.json")))
PY
}
# Topics that have no candidate pool yet. The guard below used to test the *total*
# pool against 100, which silently skipped discovery whenever an earlier pass had
# been interrupted part-way: 640 candidates across 16 of 35 topics reads as "pool
# is healthy" while 19 topics have nothing and would never get any.
uncovered() { python3 - <<'PY'
import glob, os
have = {os.path.basename(os.path.dirname(f)) for f in glob.glob("topics/*/candidates.json")}
all_ = {os.path.basename(d.rstrip("/")) for d in glob.glob("topics/*/")}
print(len(all_ - have))
PY
}

started_at=$(date +%s)
say() { echo "$*" | tee -a "$LOG"; }

say "=== run start $(date -u +%FT%TZ) — $(count) files, target ${TARGET}, backend ${CATALOG_BACKEND} ==="

# Stage 1: discovery, skipped when a pool already exists so a restart does not re-ask.
if [ "$(uncovered)" -gt 0 ]; then
  say "=== discovery: $(uncovered) topics with no candidate pool ==="
  python3 -u tools/generate_catalog_cli.py --stage discovery \
    --limit-candidates "$PER_TOPIC" 2>&1 | tee -a "$LOG"
fi
python3 tools/dedupe_candidates.py --target "$TARGET" 2>&1 | tail -5 | tee -a "$LOG"

# Stage 2: top up until the candidate pool clears the target with headroom. Calls
# fail and duplicates get dropped, so the pool has to run ahead of the file target.
for round in 1 2 3 4; do
  p=$(pool)
  [ "$p" -ge "$POOL" ] && break
  say "=== topup round ${round} — pool ${p}/${POOL} ==="
  python3 -u tools/generate_catalog_cli.py --stage topup \
    --limit-candidates "$TOPUP" 2>&1 | tail -20 | tee -a "$LOG"
  python3 tools/dedupe_candidates.py --target "$TARGET" 2>&1 | tail -3 | tee -a "$LOG"
done
say "=== candidate pool: $(pool) ==="

# Stage 3: generation, repeated until the target is met or the passes run out.
# Repeats matter: a call that times out or returns a short page writes nothing,
# and the next pass simply retries that file.
for pass in $(seq 1 "$PASSES"); do
  n=$(count)
  if [ "$n" -ge "$TARGET" ]; then
    say "=== target met: ${n} files ==="
    break
  fi
  say "=== pass ${pass}/${PASSES} — ${n} files, $(( $(date +%s) - started_at ))s elapsed ==="
  python3 -u tools/generate_catalog_cli.py --stage generation \
    --limit-generate "$PER_TOPIC" 2>&1 | tee -a "$LOG"
  # exit 2 = quota wall. Retrying is pure waste until it resets.
  if [ "${PIPESTATUS[0]}" -eq 2 ]; then
    say "=== aborted on quota at $(count) files; re-run after reset ==="
    break
  fi
  if [ "$(count)" -eq "$n" ]; then
    say "=== pass ${pass} wrote nothing; stopping rather than looping ==="
    break
  fi
done

./gen_index.sh | tee -a "$LOG"
python3 tools/audit.py 2>&1 | tail -30 | tee -a "$LOG"
say "=== run end $(date -u +%FT%TZ) — $(count) files, $(( ($(date +%s) - started_at) / 60 )) min ==="
