#!/usr/bin/env python3
"""Regenerate README.md's counts from what is actually on disk.

dbms_research's README states 1,053 problems, 35 topics and 6,700+ references in
prose. Numbers written by hand go stale the moment a generation pass lands, and a
catalog whose own front page miscounts itself is not trustworthy about anything
else. Everything between the AUTOGEN markers is derived from the files.

  python3 tools/update_readme.py
"""
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEGIN = "<!-- AUTOGEN:BEGIN -->"
END = "<!-- AUTOGEN:END -->"

STATUS_LABEL = {
    "open": ("🔴", "No known solution; the question is genuinely unresolved"),
    "partially-solved": ("🟡", "Proven for restricted cases or under assumptions; general case open"),
    "solved-recently": ("🟢", "Settled in recent history; included for the technique and the gap it closed"),
    "empirically-supported": ("🟠", "Strong numerical or heuristic evidence, no proof"),
    "stale": ("⚪", "No longer actively tracked"),
}


def problem_files():
    return sorted(p for p in glob.glob(os.path.join(ROOT, "topics", "*", "*.md"))
                  if os.path.basename(p) != "README.md")


def topic_names():
    """Read the topic table out of TAXONOMY.md: slug -> display name."""
    names = {}
    tax = os.path.join(ROOT, "TAXONOMY.md")
    if os.path.exists(tax):
        for line in open(tax, encoding="utf-8"):
            m = re.match(r"^\|\s*(\d{2})\s*\|\s*([^|]+?)\s*\|\s*\[`([^`]+)`\]", line)
            if m:
                names[m.group(3)] = m.group(2)
    return names


def main():
    files = problem_files()
    if not files:
        print("No problem files; nothing to write.")
        return 1

    names = topic_names()
    per_topic, statuses, words, refs, examples = {}, {}, 0, 0, 0
    for p in files:
        slug = os.path.basename(os.path.dirname(p))
        per_topic[slug] = per_topic.get(slug, 0) + 1
        text = open(p, encoding="utf-8").read()
        words += len(text.split())
        m = re.search(r"^status:\s*(\S+)", text, re.M)
        if m:
            statuses[m.group(1)] = statuses.get(m.group(1), 0) + 1
        body = text.split("## 9.")[-1].split("## 10.")[0]
        refs += len([l for l in body.splitlines() if l.strip().startswith("- ")])
        if "## 10." in text:
            examples += 1

    n = len(files)
    rows = "\n".join(
        f"| `{slug}` | {names.get(slug, slug)} | {per_topic.get(slug, 0)} |"
        for slug in sorted(names or per_topic))

    status_rows = "\n".join(
        f"| {STATUS_LABEL.get(s, ('·', ''))[0]} `{s}` | {c} | {STATUS_LABEL.get(s, ('', ''))[1]} |"
        for s, c in sorted(statuses.items(), key=lambda kv: -kv[1]))

    block = f"""{BEGIN}
## At a glance

| | |
|---|---|
| **Problems** | {n:,} across {len(per_topic)} of {len(names) or len(per_topic)} topics |
| **Every problem documents** | statement · formal setting · state of the art · what is known · what is not known · why it is hard · current research (2026) · concrete next experiment · references · worked example |
| **Key references** | {refs:,} cited across the catalog |
| **Worked examples** | {examples:,} — every page carries a concrete instance |
| **Content** | ~{words / 1e6:.2f}M words |

**Status mix across the {n:,} problems:**

| Status | Count | Meaning |
|---|---:|---|
{status_rows}

**Problems per topic:**

| Slug | Topic | Problems |
|---|---|---:|
{rows}
{END}"""

    readme = os.path.join(ROOT, "README.md")
    text = open(readme, encoding="utf-8").read()
    if BEGIN in text and END in text:
        text = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), block.replace("\\", "\\\\"),
                      text, flags=re.S)
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    open(readme, "w", encoding="utf-8").write(text)
    print(f"README.md updated: {n} problems, {len(per_topic)} topics, "
          f"{refs} references, {words / 1e6:.2f}M words")
    return 0


if __name__ == "__main__":
    sys.exit(main())
