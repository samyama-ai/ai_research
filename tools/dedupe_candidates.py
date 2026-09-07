#!/usr/bin/env python3
"""Make candidate slugs unique across the whole catalog, then report the shortfall.

Discovery runs per topic and the topics overlap: the first sweep produced 1,100
candidates carrying only 1,013 distinct slugs. `riemann-hypothesis` appeared under
both Number Theory and Analysis, `navier-stokes-smoothness` under both PDEs and
Analysis. Nothing on disk collides -- the files sit in different directories -- so
this is invisible until the catalog ships the same problem twice under two ids.

A duplicate is kept in the lowest-numbered topic that proposed it, which is stable
across reruns; every later copy is dropped. Existing problem files always win, so
a page already generated is never orphaned by a reassignment.

  python3 tools/dedupe_candidates.py           # rewrite candidates.json in place
  python3 tools/dedupe_candidates.py --dry-run # report only
"""
import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--target", type=int, default=1000)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(ROOT, "topics", "*", "candidates.json")))

    # A slug already written as a problem file is owned by that topic, whatever
    # discovery later proposes.
    owner = {}
    for md in glob.glob(os.path.join(ROOT, "topics", "*", "*.md")):
        if os.path.basename(md) == "README.md":
            continue
        owner[os.path.basename(md)[:-3]] = os.path.basename(os.path.dirname(md))

    seen, kept_total, dropped = dict(owner), 0, []
    for f in files:
        slug_topic = os.path.basename(os.path.dirname(f))
        cands = json.load(open(f, encoding="utf-8"))
        kept = []
        for c in cands:
            s = c.get("slug", "").strip()
            if not s:
                continue
            if seen.get(s, slug_topic) != slug_topic:
                dropped.append((s, slug_topic, seen[s]))
                continue
            if s in seen and any(k["slug"] == s for k in kept):
                continue                      # duplicate within this topic's own list
            seen[s] = slug_topic
            kept.append(c)
        kept_total += len(kept)
        print(f"{slug_topic}: {len(cands)} -> {len(kept)}")
        if not a.dry_run and len(kept) != len(cands):
            json.dump(kept, open(f, "w", encoding="utf-8"), indent=2)

    for s, loser, winner in dropped:
        print(f"  dropped {s} from {loser} (kept in {winner})")
    print(f"-- {kept_total} unique candidates across {len(files)} topics "
          f"({len(dropped)} duplicates removed) --")
    if kept_total < a.target:
        need = a.target - kept_total
        print(f"-- SHORT by {need}: run the topup stage before generating --")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
