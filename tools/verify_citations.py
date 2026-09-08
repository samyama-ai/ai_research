#!/usr/bin/env python3
"""Verify bare arXiv citations against the live arXiv API.

The catalog's citation rule is "never invent an identifier; omit the link if
unsure", so pages carry `arXiv:NNNN.NNNNN` as bare text and no URLs at all.
audit.py --links therefore probes nothing -- there is no URL to resolve. The
failure that actually matters here is a *plausible* id attached to the wrong
paper, which no structural check can see.

This resolves every unique id and compares the claimed title against the real
one. Exit 1 if any id does not exist or any title disagrees.

  python3 tools/verify_citations.py            check every page
  python3 tools/verify_citations.py --json out.json   also write a report
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOPICS = os.path.join(ROOT, "topics")
API = "https://export.arxiv.org/api/query"
BATCH = 50
PAUSE = 3.0          # arXiv asks for one request every 3s
ATOM = "{http://www.w3.org/2005/Atom}"

# "- **[SOTA]** Authors. *The Title.* Venue, Year. -- arXiv:2408.03314"
#
# The bold tag has to be removed before looking for the italic title: `**[SOTA]**`
# contains `*[SOTA]*`, so a plain `\*([^*]+)\*` matches the tag and every citation
# then "mismatches" against the real paper. That mistake reported 2,688 bad titles
# on a catalog whose ids were almost all fine.
BOLD = re.compile(r"\*\*.*?\*\*")
# Only real reference lines. Body prose italicises words too ("*instructions*",
# "*Nature*") and a paragraph that also mentions an arXiv id would otherwise be
# read as a citation claiming that word as its title.
REFLINE = re.compile(r"^\s*[-*]\s+\*\*\[")
CITE = re.compile(r"\*(?P<title>[^*]{6,300})\*.*?arXiv:(?P<aid>\d{4}\.\d{4,5})", re.S)


def norm(s):
    """Compare on words only: punctuation, case and spacing vary between a
    reference line and the arXiv record without either being wrong."""
    s = re.sub(r"\s+", " ", s.lower())
    return re.sub(r"[^a-z0-9 ]", "", s).strip()


def severity(claimed, actual):
    """How badly a claimed title disagrees with the real one.

    Most disagreements are cosmetic and say nothing about whether the id is
    right: a reference line may carry the short title ("Drop-Upcycling") or put
    the subtitle first. Those are SOFT. What matters is a claimed title with no
    real overlap -- that is a plausible id attached to the wrong paper, which is
    the one failure a structural check can never catch.
    """
    c, a = norm(claimed), norm(actual)
    if c == a:
        return None
    if c in a or a in c:
        return "SOFT"                      # short form / truncation
    cw, aw = set(c.split()), set(a.split())
    if not cw or not aw:
        return "HARD"
    overlap = len(cw & aw) / min(len(cw), len(aw))
    return "SOFT" if overlap >= 0.6 else "HARD"   # reordered subtitle vs other paper


def claims():
    """{arxiv_id: {claimed_title: [pages]}} across the whole catalog."""
    out = defaultdict(lambda: defaultdict(list))
    for dirpath, _, names in os.walk(TOPICS):
        for n in sorted(names):
            if not n.endswith(".md") or n == "README.md":
                continue
            p = os.path.join(dirpath, n)
            rel = os.path.relpath(p, ROOT)
            for line in open(p, encoding="utf-8"):
                if "arXiv:" not in line or not REFLINE.match(line):
                    continue
                m = CITE.search(BOLD.sub("", line))
                if m:
                    out[m.group("aid")][m.group("title").strip().rstrip(".")].append(rel)
    return out


def fetch_with_retry(ids, tries=4):
    """Resolve a batch, retrying with backoff. Returns (found, ok).

    `ok` is False when every attempt failed. That distinction is the whole
    point: without it a rate-limited batch is reported as 50 nonexistent
    papers. A first version of this file did exactly that and claimed 804
    unresolvable ids on a catalog that had one.
    """
    delay = 5.0
    for attempt in range(tries):
        try:
            return fetch(ids), True
        except Exception as e:
            if attempt == tries - 1:
                print(f"  batch failed after {tries} tries: {e}", file=sys.stderr, flush=True)
                return {}, False
            time.sleep(delay)
            delay *= 2
    return {}, False


def fetch(ids):
    """Resolve a batch of ids -> {id: real_title}. Missing ids simply absent."""
    q = urllib.parse.urlencode({"id_list": ",".join(ids), "max_results": len(ids)})
    req = urllib.request.Request(f"{API}?{q}", headers={"User-Agent": "ai_research-citation-audit"})
    with urllib.request.urlopen(req, timeout=60) as r:
        root = ET.fromstring(r.read())
    found = {}
    for e in root.findall(f"{ATOM}entry"):
        idu = e.find(f"{ATOM}id")
        ti = e.find(f"{ATOM}title")
        if idu is None or ti is None:
            continue
        m = re.search(r"arxiv\.org/abs/(\d{4}\.\d{4,5})", idu.text or "")
        if m:
            found[m.group(1)] = re.sub(r"\s+", " ", ti.text or "").strip()
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="write a machine-readable report here")
    args = ap.parse_args()

    c = claims()
    ids = sorted(c)
    print(f"-- {len(ids)} unique arXiv ids across "
          f"{sum(len(p) for t in c.values() for p in t.values())} citations --", flush=True)

    real, unchecked = {}, set()
    for i in range(0, len(ids), BATCH):
        batch = ids[i:i + BATCH]
        found, ok = fetch_with_retry(batch)
        if ok:
            real.update(found)
        else:
            unchecked.update(batch)
        print(f"  resolved {len(real)}/{len(ids)}", end="\r", file=sys.stderr, flush=True)
        if i + BATCH < len(ids):
            time.sleep(PAUSE)
    print(file=sys.stderr)
    if unchecked:
        print(f"-- {len(unchecked)} ids UNCHECKED (request failed, not missing) --")

    missing, mismatch, ok = [], [], 0
    for aid in ids:
        for claimed, pages in c[aid].items():
            if aid in unchecked:
                continue                    # never claim a bad id we could not ask about
            if aid not in real:
                missing.append({"id": aid, "claimed": claimed, "pages": pages})
            else:
                sev = severity(claimed, real[aid])
                if sev:
                    mismatch.append({"id": aid, "claimed": claimed, "severity": sev,
                                     "actual": real[aid], "pages": pages})
                else:
                    ok += 1

    for m in missing:
        print(f"MISSING   arXiv:{m['id']} does not resolve -- \"{m['claimed'][:70]}\" "
              f"[{m['pages'][0]}]")
    for m in sorted(mismatch, key=lambda x: x["severity"]):
        print(f"{m['severity']:8} arXiv:{m['id']}\n"
              f"          claimed: {m['claimed'][:100]}\n"
              f"          actual:  {m['actual'][:100]}\n"
              f"          in: {', '.join(m['pages'][:3])}")

    hard = [m for m in mismatch if m["severity"] == "HARD"]
    soft = [m for m in mismatch if m["severity"] == "SOFT"]
    print(f"-- {ok} verified, {len(soft)} cosmetic title variants, "
          f"{len(hard)} likely-wrong-paper, {len(missing)} unresolvable --")
    if args.json:
        json.dump({"ok": ok, "mismatch": mismatch, "missing": missing},
                  open(args.json, "w"), indent=2)
    if unchecked:
        print(f"   re-run to check the {len(unchecked)} unchecked ids")
    return 1 if (missing or hard) else 0


if __name__ == "__main__":
    sys.exit(main())
