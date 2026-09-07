#!/usr/bin/env python3
"""Catalog audit — run before every push, and after every generation batch.

  python3 tools/audit.py            structure, frontmatter, index/taxonomy sync, safety
  python3 tools/audit.py --links    also resolve every external URL (slow, network)

Ported from dbms_research/tools/audit.py, with the section list and status
vocabulary of this catalog. Exit 0 = clean. Exit 1 = at least one FAIL.
"""
import argparse
import concurrent.futures as cf
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOPICS = os.path.join(ROOT, "topics")

REQUIRED_FM = ["id", "title", "topic", "status", "first_added",
               "last_reviewed", "last_substantive_update", "stale_since", "provenance"]
VALID_STATUS = {"open", "partially-solved", "empirically-open",
                "methodologically-blocked", "solved-but-impractical", "stale"}
VALID_PROV = {"synthesized", "verified"}
SECTIONS = ["1. Problem Statement", "2. Formal Setting", "3. State of the Art",
            "4. What Is Known", "5. What Is Not Known", "6. Why It Is Hard",
            "7. Current Research", "8. Concrete Next Experiment",
            "9. Key References", "10. Worked Example"]

# This repo is intended to be public. Nothing internal may appear in it.
LEAK = re.compile(
    r"(?<![\w-])(?:sk-[A-Za-z0-9_-]{16,}|sk-ant-[A-Za-z0-9_-]{16,}|AKIA[0-9A-Z]{16}"
    r"|gh[pousr]_[A-Za-z0-9]{20,}|glpat-[A-Za-z0-9_-]{16,}|xox[baprs]-[A-Za-z0-9-]{10,}"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY|git\.samyama\.ai|samyama-research"
    r"|/home/[a-z0-9_-]+/|/Users/[A-Za-z0-9_-]+/|qorro|trufluence|inpharmd|brightcone)")

FAILS, WARNS = [], []


def fail(m):
    FAILS.append(m)


def warn(m):
    WARNS.append(m)


def problem_files():
    out = []
    for d, _, fs in os.walk(TOPICS):
        for f in fs:
            if f.endswith(".md") and f != "README.md":
                out.append(os.path.join(d, f))
    return sorted(out)


def frontmatter(text):
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---\n", 4)
    if end < 0:
        return None
    fm = {}
    for line in text[4:end].split("\n"):
        m = re.match(r"^([a-z_]+):\s*(.*)$", line)
        if m:
            fm[m.group(1)] = m.group(2).strip().strip('"')
    return fm


def check_structure(files):
    for p in files:
        rel = os.path.relpath(p, ROOT)
        text = open(p, encoding="utf-8").read()
        fm = frontmatter(text)
        if fm is None:
            fail(f"{rel}: no YAML frontmatter")
            continue
        for k in REQUIRED_FM:
            if k not in fm:
                fail(f"{rel}: frontmatter missing `{k}`")
        if fm.get("status") and fm["status"] not in VALID_STATUS:
            fail(f"{rel}: status '{fm['status']}' not in {sorted(VALID_STATUS)}")
        if fm.get("provenance") and fm["provenance"] not in VALID_PROV:
            fail(f"{rel}: provenance '{fm['provenance']}' not in {sorted(VALID_PROV)}")
        # id must equal its path -- it is the identity used to diff iterations
        want = os.path.relpath(p, TOPICS)[:-3]
        if fm.get("id") and fm["id"] != want:
            fail(f"{rel}: id '{fm['id']}' does not match path (expected '{want}')")
        if fm.get("status") == "stale" and not fm.get("stale_since"):
            fail(f"{rel}: status stale with empty stale_since")
        # all ten sections, in order
        heads = re.findall(r"^## (.+)$", text, re.M)
        pos = 0
        for want_s in SECTIONS:
            key = want_s.split(".")[0] + "."
            found = next((i for i, h in enumerate(heads[pos:], pos)
                          if h.startswith(key)), None)
            if found is None:
                fail(f"{rel}: missing section '{want_s}'")
            else:
                pos = found + 1
        # KaTeX hazard: a bare # inside math breaks rendering (tools/fix_math_hash.py).
        # Split on unescaped $ -- odd segments are inside math.
        for lineno, line in enumerate(text.split("\n"), 1):
            parts = re.split(r"(?<!\\)\$", line)
            if len(parts) % 2 == 0:
                continue                       # unbalanced on this line; not our check
            for seg in parts[1::2]:
                if re.search(r"(?<!\\)#", seg):
                    fail(f"{rel}:{lineno}: unescaped '#' inside math: {seg[:56]}")
        for m in LEAK.finditer(text):
            fail(f"{rel}: internal/secret token in a PUBLIC repo: '{m.group(0)[:40]}'")
        for m in re.finditer(r"\b(TODO|FIXME|XXX|Lorem ipsum)\b|\bTBD\b", text):
            warn(f"{rel}: placeholder '{m.group(0)}'")


def check_index(files):
    """Compare INDEX.md against a fresh generation WITHOUT touching the real file."""
    real = os.path.join(ROOT, "INDEX.md")
    if not os.path.exists(real):
        fail("INDEX.md does not exist -- run ./gen_index.sh")
        return
    with tempfile.TemporaryDirectory() as td:
        backup = os.path.join(td, "INDEX.md.orig")
        shutil.copy2(real, backup)
        try:
            gen = subprocess.run(["bash", os.path.join(ROOT, "gen_index.sh")],
                                 cwd=ROOT, capture_output=True, text=True)
            fresh = open(real, encoding="utf-8").read() if gen.returncode == 0 else None
        finally:
            shutil.copy2(backup, real)          # always restore, even on exception
    if gen.returncode != 0:
        fail(f"gen_index.sh failed: {gen.stderr.strip()[:200]}")
        return
    idx = open(real, encoding="utf-8").read()
    if fresh != idx:
        fail("INDEX.md is stale -- run ./gen_index.sh and commit the result.")
    m = re.search(r"\*\*Total: (\d+) problems\.\*\*", idx)
    if m and int(m.group(1)) != len(files):
        fail(f"INDEX.md says {m.group(1)} problems; {len(files)} files on disk")
    linked = set(re.findall(r"\]\(\./topics/([^)]+\.md)\)", idx))
    on_disk = {os.path.relpath(p, TOPICS) for p in files}
    for miss in sorted(on_disk - linked):
        fail(f"INDEX.md does not link {miss}")
    for extra in sorted(linked - on_disk):
        fail(f"INDEX.md links a file that does not exist: {extra}")


def check_taxonomy(files):
    """Every topic on disk must be one of the 10 declared in TAXONOMY.md."""
    tax_path = os.path.join(ROOT, "TAXONOMY.md")
    if not os.path.exists(tax_path):
        return
    tax = open(tax_path, encoding="utf-8").read()
    declared = set(re.findall(r"`(\d{2}-[a-z-]+)`", tax))
    on_disk = {os.path.basename(os.path.dirname(p)) for p in files}
    for slug in sorted(on_disk - declared):
        fail(f"topics/{slug} is not declared in TAXONOMY.md")
    for slug in sorted(declared - on_disk):
        warn(f"TAXONOMY.md declares {slug}, which has no problem files yet")


def urls_in(files):
    seen = {}
    for p in files:
        for u in re.findall(r"\((https?://[^)\s]+)\)", open(p, encoding="utf-8").read()):
            seen.setdefault(u.rstrip(".,);"), []).append(os.path.relpath(p, ROOT))
    return seen


def probe(u):
    for method in ("HEAD", "GET"):
        try:
            r = urllib.request.Request(u, method=method,
                                       headers={"User-Agent": "maths_research-audit/1.0"})
            with urllib.request.urlopen(r, timeout=12) as resp:
                return u, resp.status
        except Exception as e:
            code = getattr(e, "code", None)
            if code in (403, 405) and method == "HEAD":
                continue                       # some hosts refuse HEAD; retry as GET
            if code:
                return u, code
    return u, 0


LINK_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "maths_research-linkcheck.tsv")

# Tiered like the dbms_research sweep, for the same measured reasons: doi.org and the
# big publishers answer 403 to a non-browser client, so probing them is noise; arXiv
# rate-limits individual probes but answers a batched API query; zbMATH and MathSciNet
# gate behind a subscription. Each tier is reported, so "0 failures" can never quietly
# mean "0 failures among the half we bothered to check".
DEFERRED_HOSTS = {"zbmath.org", "mathscinet.ams.org", "dblp.org", "dblp.uni-trier.de"}
DOI_BEARING_HOSTS = {"doi.org", "dx.doi.org", "link.springer.com", "www.sciencedirect.com",
                     "onlinelibrary.wiley.com", "www.ams.org", "www.jstor.org",
                     "www.tandfonline.com", "academic.oup.com"}
ALIVE = {200, 201, 202, 203, 204, 301, 302, 303, 307, 308}


def classify(u):
    host = u.split("/")[2].lower() if "://" in u else ""
    if host in DEFERRED_HOSTS:
        return "deferred"
    if host in DOI_BEARING_HOSTS:
        return "doi"
    if host == "arxiv.org":
        return "arxiv"
    return "probe"


def check_arxiv(urls, seen):
    """Resolve arXiv ids in batched API calls instead of one probe per URL."""
    ids = []
    for u in urls:
        m = re.search(r"arxiv\.org/abs/([a-z-]+/\d{7}|\d{4}\.\d{4,5})", u)
        if m:
            ids.append((m.group(1), u))
        else:
            warn(f"{seen[u][0]}: unparseable arXiv URL: {u}")
    for i in range(0, len(ids), 100):
        chunk = ids[i:i + 100]
        q = "id_list=" + ",".join(a for a, _ in chunk) + f"&max_results={len(chunk)}"
        try:
            req = urllib.request.Request("https://export.arxiv.org/api/query?" + q,
                                         headers={"User-Agent": "maths_research-audit/1.0"})
            with urllib.request.urlopen(req, timeout=90) as r:
                body = r.read().decode()
        except Exception as e:
            fail(f"--links: arXiv API unreachable ({e}); {len(chunk)} ids UNVERIFIED")
            continue
        found = set(re.findall(r"arxiv\.org/abs/([a-z-]*/?\d+\.?\d*)v\d+", body))
        tails = {f.split("/")[-1] for f in found}
        for aid, u in chunk:
            if aid not in found and aid.split("/")[-1] not in tails:
                fail(f"{seen[u][0]}: arXiv does not return {aid}: {u}")
        time.sleep(3)                      # arXiv API etiquette


def check_dois(urls, seen):
    """Validate the DOI these URLs carry. They cannot be fetched (403 by design)."""
    for u in urls:
        m = re.search(r"(10\.\d{4,9}/\S+)$", u)
        if not m:
            warn(f"{seen[u][0]}: publisher URL with no extractable DOI: {u}")
            continue
        if not re.match(r"^10\.\d{4,9}/\S+$", m.group(1)):
            fail(f"{seen[u][0]}: malformed DOI: {u}")


def check_links(files):
    """Resolve external URLs, tiered by host, checkpointing after every probe."""
    seen = urls_in(files)
    tiers = {}
    for u in seen:
        tiers.setdefault(classify(u), []).append(u)
    print("  tiers: " + ", ".join(f"{k}={len(v)}" for k, v in sorted(tiers.items())),
          file=sys.stderr, flush=True)

    check_dois(tiers.get("doi", []), seen)
    check_arxiv(tiers.get("arxiv", []), seen)

    probe_set = tiers.get("probe", [])
    os.makedirs(os.path.dirname(LINK_CACHE), exist_ok=True)
    done = {}
    if os.path.exists(LINK_CACHE):
        for line in open(LINK_CACHE, encoding="utf-8"):
            p = line.rstrip("\n").split("\t")
            if len(p) == 2 and p[1].lstrip("-").isdigit():
                done[p[0]] = int(p[1])
    todo = [u for u in probe_set if u not in done]
    print(f"  probing {len(probe_set)} ({len(todo)} new); "
          f"{len(tiers.get('doi', []))} DOIs syntax-only; "
          f"{len(tiers.get('arxiv', []))} arXiv via API; "
          f"{len(tiers.get('deferred', []))} deferred (rate limits / paywall)",
          file=sys.stderr, flush=True)

    if todo:
        # as_completed, NOT ex.map: map yields in submission order, so one slow URL
        # stalls every write behind it and defeats the checkpointing.
        with open(LINK_CACHE, "a", encoding="utf-8") as cache, \
                cf.ThreadPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(probe, u): u for u in todo}
            for n, fut in enumerate(cf.as_completed(futs), 1):
                u, code = fut.result()
                cache.write(f"{u}\t{code}\n")
                cache.flush()
                done[u] = code
                if n % 100 == 0:
                    print(f"    {n}/{len(todo)}", file=sys.stderr, flush=True)

    missing = [u for u in probe_set if u not in done]
    if missing:
        fail(f"--links did not finish: {len(missing)} URLs never probed. Re-run to resume; "
             f"do NOT read this run as clean.")
    for u in sorted(probe_set):
        code = done.get(u)
        if code is None or code in ALIVE:
            continue
        where = seen[u][0] + (f" (+{len(seen[u]) - 1})" if len(seen[u]) > 1 else "")
        (fail if code in (404, 410) else warn)(f"{where}: HTTP {code} {u}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--links", action="store_true")
    a = ap.parse_args()

    files = problem_files()
    print(f"-- audit: {len(files)} problem files --")
    check_structure(files)
    check_index(files)
    check_taxonomy(files)
    if a.links:
        check_links(files)

    for w in WARNS:
        print(f"WARN  {w}")
    for f in FAILS:
        print(f"FAIL  {f}")
    print(f"-- {len(FAILS)} failures, {len(WARNS)} warnings --")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
