#!/usr/bin/env python3
import os
import sys
import json
import time
import argparse
import asyncio
from pathlib import Path
import subprocess
import re

ROOT = Path(__file__).resolve().parent.parent

# sys.path is fixed up here, not in main(). fix_math_hash was imported inside
# generate_problem_file and only resolved because main() appended tools/ first,
# so every page failed with ModuleNotFoundError the moment the module was driven
# from anywhere but its own CLI.
sys.path.insert(0, str(ROOT / "tools"))
from fix_math_hash import transform as fix_math_hash_transform  # noqa: E402

# Backend: "claude" or "agy". Both are print-mode CLIs that take a prompt and
# return text, so only the argv differs. agy is kept because it was the original
# mandate; claude is the default after agy's subscription quota stopped a run at
# 124 of 1,000 files.
BACKEND = os.getenv("CATALOG_BACKEND", "claude").lower()

# Each call takes tens of seconds, so the run is latency-bound, not CPU-bound.
# Bound total in-flight calls across all topics.
AGY_CONCURRENCY = int(os.getenv("AGY_CONCURRENCY", "8"))
AGY_TIMEOUT = os.getenv("AGY_PRINT_TIMEOUT", "10m")
AGY_SEMAPHORE = asyncio.Semaphore(AGY_CONCURRENCY)

# A quota wall is not a transient error. The first long run burned 20 minutes and
# ~1,100 queued calls retrying against "Individual quota reached", producing nothing:
# every call returned in under 10 seconds, so the retry loop just spun. Once this is
# seen, the whole run aborts and says when the quota resets.
QUOTA_RE = re.compile(
    r"quota reached|quota exceeded|rate limit|resets? (in|at|\d)|"
    r"(session|usage|weekly|daily) limit|hit your limit|"
    r"upgrade your subscription|too many requests|429|"
    # An admin disabling Code access is not a rate limit, but retrying is just
    # as pointless: it wedged a run for 75 minutes with the service still
    # reporting active and its log frozen.
    r"organization has disabled|subscription access for claude code|"
    r"use an anthropic api key", re.I)


class QuotaExhausted(RuntimeError):
    pass


# Progress accounting, so the log shows completions rather than only the queue
# being drained. Every "+ Generating" line is printed at enqueue time, which made
# 2,081 of them appear in the first seconds of a run that had written no files.
PROGRESS = {"done": 0, "failed": 0, "total": 0}


def note(msg):
    print(msg, flush=True)

# System instruction to enforce math detail and schema adherence
SYSTEM_INSTRUCTION = """You are a world-class AI/ML researcher and the editor of a catalog of open problems in artificial intelligence.
You write research-grade catalog pages: precise problem statements, the formal setting, what is genuinely established versus merely claimed, and the smallest experiment that would settle the question.
Mathematical notation must use LaTeX: $...$ for inline and $$...$$ for display blocks.
All citations must be real, verifiable publications with accurate titles, authors, venues, and years. Never fabricate an arXiv ID, DOI, or URL. If an identifier is uncertain, give authors/title/venue/year and omit the link.
Distinguish carefully between problems that are theoretically open (no proof either way), empirically open (the experiment is runnable but unrun at the right scale), and methodologically blocked (the measurement is not yet well defined).
You must adhere strictly to the 10-section TEMPLATE.md schema.
Do not skip any sections. Do not use placeholders."""

# The 10 section headings TEMPLATE.md requires, matched by prefix.
REQUIRED_SECTIONS = [f"{i}." for i in range(1, 11)]

# Let's define the topics from TAXONOMY.md
TOPICS = {
    "01-tokenization": "Tokenization & Vocabulary",
    "02-attention": "Attention Mechanisms",
    "03-training-dynamics": "Training Dynamics & Optimization",
    "04-alignment": "Alignment & Preference Learning",
    "05-retrieval-and-agents": "Retrieval & Agentic Systems",
    "06-data-pipeline": "Data Pipelines & Curation",
    "07-embeddings": "Embeddings & Representations",
    "08-loss-and-heads": "Loss Functions & Output Heads",
    "09-model-design": "Architecture & Model Design",
    "10-scaling-laws": "Scaling Laws & Compute Allocation",
    "11-inference-and-serving": "Inference & Serving",
    "12-quantization-compression": "Quantization & Compression",
    "13-parameter-efficient-adaptation": "Parameter-Efficient Adaptation",
    "14-long-context": "Long Context",
    "15-mixture-of-experts": "Mixture of Experts",
    "16-state-space-models": "State-Space & Recurrent Models",
    "17-reasoning": "Reasoning & Inference-Time Compute",
    "18-rl-for-llms": "Reinforcement Learning for LLMs",
    "19-evaluation": "Evaluation & Benchmarking",
    "20-interpretability": "Interpretability",
    "21-factuality": "Hallucination & Factuality",
    "22-safety-robustness": "Safety & Robustness",
    "23-privacy-memorization": "Privacy & Memorization",
    "24-multimodal": "Multimodal Models",
    "25-speech-and-audio": "Speech & Audio",
    "26-code-generation": "Code Generation & Program Synthesis",
    "27-multilingual": "Multilingual & Low-Resource",
    "28-knowledge-editing": "Knowledge Editing & Model Updating",
    "29-distillation": "Distillation & Transfer",
    "30-synthetic-data": "Synthetic Data",
    "31-distributed-training": "Distributed Training Systems",
    "32-hardware-and-kernels": "Hardware & Kernels",
    "33-uncertainty-calibration": "Uncertainty & Calibration",
    "34-diffusion-generative": "Diffusion & Generative Modeling",
    "35-world-models": "World Models & Planning",
}


FM_ORDER = ["id", "title", "topic", "status", "first_added", "last_reviewed",
            "last_substantive_update", "stale_since", "provenance"]


def normalize_markdown(text, topic_slug, slug, title, status, month=None):
    """Strip code fences the model wraps output in, then force the frontmatter.

    The model reliably fences the YAML block (```yaml ... ```), which makes the file
    fail the frontmatter check and renders the block as a code sample. Frontmatter
    fields are also identity, not prose, so they are rewritten from the candidate
    rather than trusted: `id` must equal the file's path or the audit rejects it.
    """
    text = text.strip()

    # A whole-response ```markdown fence.
    if text.startswith("```"):
        first, _, rest = text.partition("\n")
        if first.strip("` ").lower() in ("markdown", "md", "yaml", "", "text"):
            text = rest
            if text.rstrip().endswith("```"):
                text = text.rstrip()[:-3]
    text = text.strip()

    # A fence closing right after the frontmatter block.
    text = re.sub(r"\A(---\n.*?\n---)\n```\s*\n", r"\1\n", text, flags=re.S)

    month = month or time.strftime("%Y-%m")
    body = text
    if body.startswith("---"):
        end = body.find("\n---", 3)
        if end >= 0:
            body = body[end + 4:].lstrip("\n")

    fm = {
        "id": f"{topic_slug}/{slug}",
        "title": f'"{title}"',
        "topic": topic_slug,
        "status": status,
        "first_added": month,
        "last_reviewed": month,
        "last_substantive_update": month,
        "stale_since": '""',
        "provenance": "synthesized",
    }
    header = "---\n" + "\n".join(f"{k}: {fm[k]}" for k in FM_ORDER) + "\n---\n\n"
    return header + body


async def call_agy_cli(prompt, is_json=False):
    """Run one prompt through the configured CLI backend and return its text."""
    
    # The claude backend takes the role instruction as a real system prompt, which
    # also replaces Claude Code's own harness prompt. Left as a prefix, the default
    # agent boots MCP servers and project context for every call: one page took
    # 283s that way and 30s with --strict-mcp-config and --system-prompt.
    full_prompt = prompt if BACKEND == "claude" else SYSTEM_INSTRUCTION + "\n\n" + prompt
    if is_json:
        full_prompt += "\n\nCRITICAL: Return ONLY valid JSON. Do not wrap it in markdown blocks (e.g. ```json ... ```). Just raw JSON."
        
    loop = asyncio.get_event_loop()
    
    def _run():
        if BACKEND == "agy":
            cmd = ["agy", "-p", full_prompt, "--print-timeout", AGY_TIMEOUT]
        else:
            cmd = ["claude", "-p", full_prompt,
                   "--output-format", "text",
                   "--strict-mcp-config",
                   "--system-prompt", SYSTEM_INSTRUCTION]
        model = os.getenv("CATALOG_MODEL") or os.getenv("AGY_MODEL")
        if model:
            cmd += ["--model", model]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            err = (result.stderr + result.stdout).strip()
            if QUOTA_RE.search(err):
                raise QuotaExhausted(err[:200])
            raise RuntimeError(f"{BACKEND} -p failed: {err[:500]}")
        return result.stdout.strip()

    async with AGY_SEMAPHORE:
        response_text = ""
        for attempt in range(3):
            try:
                response_text = await loop.run_in_executor(None, _run)
                if response_text:
                    break
                # rc 0 with nothing on stdout. This is how the CLI reports being
                # unable to start a session, and it was the run's worst failure
                # mode: silent. 16 workers cycled every 60s writing nothing, and
                # the swallowed exception meant neither the log nor the file count
                # showed a reason. Always say it happened.
                note(f"    {BACKEND} returned an empty response "
                     f"(attempt {attempt + 1}/3)")
            except QuotaExhausted:
                raise
            except Exception as e:
                note(f"    {BACKEND} call failed (attempt {attempt + 1}/3): {e}")
            await asyncio.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"{BACKEND} returned nothing after 3 attempts")
    
    if is_json:
        # Strip potential markdown formatting if model didn't listen
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
    return response_text

async def discover_candidates(topic_slug, topic_name, limit=100):
    """Generate a list of candidate AI research problems for a topic."""
    print(f"=== Discovering {limit} candidates for {topic_name} ===")
    
    prompt = f"""Identify exactly {limit} distinct open or hard research problems in the field of {topic_name}, within artificial intelligence and machine learning.\nPrefer problems that are specific and decidable by an experiment or a proof over broad research agendas. Avoid restating well-known benchmarks as problems.
For each problem, you must provide:
1. A precise Title naming the problem (e.g. "Compute-Optimal Vocabulary Size", "Length Generalization in Arithmetic").
2. A kebab-case filename slug (e.g. "compute-optimal-vocabulary-size", "length-generalization-arithmetic").
3. The Status: one of "open" (no accepted answer), "partially-solved" (settled under restrictive assumptions or at small scale only), "empirically-open" (the deciding experiment is runnable but has not been run at the right scale), "solved-but-impractical" (a correct method exists but its cost rules it out in practice), or "methodologically-blocked" (the quantity in question is not yet well enough defined to measure).
4. A very brief 1-sentence Description (maximum 12 words) summarizing the core question. Do NOT use LaTeX, math blocks ($ or $$), or backslashes. Keep it strictly plain text.

Return this list as a JSON array of objects matching the following schema:
[
  {{
    "title": "Problem Title",
    "slug": "problem-slug",
    "status": "open|partially-solved|empirically-open|solved-but-impractical|methodologically-blocked",
    "description": "1-sentence summary of the problem."
  }}
]
Ensure the response contains only the valid JSON array."""

    result_text = await call_agy_cli(prompt, is_json=True)
    try:
        # Escape any backslashes that are not part of valid JSON escape sequences
        sanitized_text = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', result_text)
        candidates = json.loads(sanitized_text)
        # Save to candidates.json
        topic_dir = ROOT / "topics" / topic_slug
        topic_dir.mkdir(parents=True, exist_ok=True)
        candidates_file = topic_dir / "candidates.json"
        with open(candidates_file, "w", encoding="utf-8") as f:
            json.dump(candidates, f, indent=2)
        print(f"Discovered {len(candidates)} candidates for {topic_slug} -> saved to {candidates_file.relative_to(ROOT)}")
        return candidates
    except Exception as e:
        print(f"Error parsing candidates JSON for {topic_slug}: {e}")
        print("Model returned:")
        print(result_text)
        return []


async def topup_candidates(topic_slug, topic_name, extra):
    """Ask for `extra` more candidates for a topic, excluding the ones already held.

    Deduplication across topics removes candidates, and some generation calls fail,
    so the pool has to run ahead of the target rather than exactly meet it. The
    exclusion list is sent verbatim: without it the model re-proposes the same
    headline conjectures and the pool does not grow.
    """
    topic_dir = ROOT / "topics" / topic_slug
    topic_dir.mkdir(parents=True, exist_ok=True)
    candidates_file = topic_dir / "candidates.json"
    existing = json.loads(candidates_file.read_text(encoding="utf-8")) if candidates_file.exists() else []
    have = {c["slug"] for c in existing}

    print(f"=== Topping up {topic_name} by {extra} (have {len(have)}) ===")
    prompt = f"""Identify exactly {extra} distinct open or hard research problems in the field of {topic_name}, within artificial intelligence and machine learning.

You must NOT return any of the following, which are already catalogued:
{", ".join(sorted(have))}

Go deeper into the specialised sub-areas of the field to find problems that are genuinely distinct from that list. Favour concrete, under-explored questions over headline agendas.
For each problem, provide:
1. The official/common Title.
2. A kebab-case filename slug.
3. The Status: one of "open", "partially-solved", "empirically-open", "solved-but-impractical", or "methodologically-blocked".
4. A very brief 1-sentence Description (maximum 12 words). Do NOT use LaTeX, math blocks ($ or $$), or backslashes. Keep it strictly plain text.

Return a JSON array of objects with keys: title, slug, status, description.
Ensure the response contains only the valid JSON array."""

    result_text = await call_agy_cli(prompt, is_json=True)
    try:
        sanitized_text = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', result_text)
        new = json.loads(sanitized_text)
    except Exception as e:
        print(f"Error parsing topup JSON for {topic_slug}: {e}")
        return existing

    added = [c for c in new if c.get("slug") and c["slug"] not in have]
    merged = existing + added
    candidates_file.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"Topped up {topic_slug}: +{len(added)} -> {len(merged)}")
    return merged


async def generate_problem_file(topic_slug, topic_name, candidate):
    """Generate a single problem markdown file following TEMPLATE.md."""
    title = candidate["title"]
    slug = candidate["slug"]
    status = candidate["status"]
    
    topic_dir = ROOT / "topics" / topic_slug
    file_path = topic_dir / f"{slug}.md"
    
    if file_path.exists():
        return True
        
    note(f"  + queued: {title} ({status})")
    
    # Read TEMPLATE.md to feed into prompt
    template_path = ROOT / "TEMPLATE.md"
    template_content = template_path.read_text(encoding="utf-8") if template_path.exists() else ""
    
    prompt = f"""Write a comprehensive, research-grade catalog page for the AI/ML research problem: "{title}" under the topic "{topic_name}".

You must strictly adhere to the following template schema. Do not skip any sections and do not use placeholders.

---
TEMPLATE SCHEMA FOR EACH PAGE:
{template_content}
---

Topic slug: {topic_slug}
Problem slug: {slug}
Status: {status}

Additional Instructions:
- Section 2 (Formal Setting) must define every quantity as it would actually be measured, with LaTeX formulations. State which assumptions are known to be violated in practice.
- Section 3 (State of the Art) must separate what is established from what is claimed but unablated. Where a result exists only as a benchmark number, say so.
- Section 4 (What Is Known) must give real numbers and name the scale they were measured at.
- Section 5 (What Is Not Known) must classify the gap as theoretically open, empirically open, or methodologically blocked.
- Section 6 (Why It Is Hard) must name a specific obstruction: compute cost, confounded measurement, absent ground truth, non-identifiability, or an evaluation that does not measure what it names. "Hard because it is important" is not an obstruction.
- Section 8 (Concrete Next Experiment) must state the scale, the control arm, and the single number that would decide the question. This section is what makes the page useful.
- Section 9 (Key References) must list real, verifiable papers with correct titles, authors, venues, and years. Never invent an arXiv ID or DOI; omit the link if unsure.
- Section 10 (Worked Example) must carry one concrete instance end to end with real numbers, and make the obstruction visible rather than only illustrating definitions.

LENGTH: target 1,400-1,900 words for the whole page. This is a catalog entry, not
a survey article. Uncapped, pages came back at 4,500 words and 400 seconds each,
which is 3x the length of the rest of the catalog and puts a 1,000-page run at
12 hours. Be dense and specific rather than expansive: name the theorem, state
the bound, cite the paper, move on.

Return only the markdown content, starting with the YAML frontmatter block."""

    started = time.time()
    markdown_content = await call_agy_cli(prompt, is_json=False)
    elapsed = time.time() - started
    markdown_content = normalize_markdown(
        markdown_content, topic_slug, slug, title, status)

    missing = [h for h in REQUIRED_SECTIONS if f"## {h}" not in markdown_content]
    if missing:
        PROGRESS["failed"] += 1
        note(f"    ! {slug}.md: missing sections {missing}, skipping write")
        return False

    
    # Post-process to fix math hash character parsing issues
    cleaned_content, n_escapes = fix_math_hash_transform(markdown_content)
    if n_escapes > 0:
        print(f"    Fixed {n_escapes} '#' in math mode for {slug}.md")
        
    # Write through a temp file in the same directory, then rename. A run killed
    # mid-write otherwise leaves a truncated page that the next pass skips as
    # "already exists", so the damage is permanent and silent.
    tmp = file_path.with_suffix(".md.tmp")
    tmp.write_text(cleaned_content, encoding="utf-8")
    tmp.replace(file_path)

    PROGRESS["done"] += 1
    note(f"  [{PROGRESS['done']}/{PROGRESS['total']}] {file_path.relative_to(ROOT)} "
         f"({len(cleaned_content) // 1024} KB, {elapsed:.0f}s)")
    return True


async def generate_interleaved(target_topics, limit_generate):
    """Generate across all topics round-robin instead of topic by topic.

    asyncio.gather queues per topic in order, and the semaphore hands out slots
    FIFO, so the first topic's whole backlog is served before the second topic
    gets a single slot. The first long run hit the agy quota wall after 124 files
    and 117 of them were Number Theory; eight topics had nothing at all. Round-robin
    means an interrupted run leaves the catalog evenly covered rather than deep in
    one field.
    """
    queues = []
    for slug, name in target_topics.items():
        f = ROOT / "topics" / slug / "candidates.json"
        if not f.exists():
            print(f"No candidates.json for {slug}; run --stage discovery first.")
            continue
        cands = sorted(json.loads(f.read_text(encoding="utf-8")), key=lambda c: c["slug"])
        queues.append((slug, name, cands[:limit_generate]))

    ordered = []
    for i in range(max((len(q[2]) for q in queues), default=0)):
        for slug, name, cands in queues:
            if i < len(cands):
                ordered.append((slug, name, cands[i]))
    PROGRESS["total"] = len(ordered)
    note(f"=== Generating {len(ordered)} problems, round-robin across "
         f"{len(queues)} topics ===")
    run_started = time.time()

    results = await asyncio.gather(*(
        generate_problem_file(slug, name, cand) for slug, name, cand in ordered
    ), return_exceptions=True)
    for r in results:
        if isinstance(r, QuotaExhausted):
            raise r
    errs = [r for r in results if isinstance(r, Exception)]
    if errs:
        PROGRESS["failed"] += len(errs)
        note(f"  {len(errs)} calls raised; first: {errs[0]}")

    mins = (time.time() - run_started) / 60
    note(f"=== pass done: {PROGRESS['done']} written, {PROGRESS['failed']} rejected, "
         f"{mins:.0f} min ({PROGRESS['done'] / max(mins, 1):.1f} files/min) ===")

    for slug, name, _ in queues:
        await generate_topic_readme(slug, name)


async def process_topic(topic_slug, topic_name, limit_candidates, limit_generate, stage):
    """Process a single topic (discovery and/or generation)."""
    topic_dir = ROOT / "topics" / topic_slug
    topic_dir.mkdir(parents=True, exist_ok=True)
    candidates_file = topic_dir / "candidates.json"
    
    candidates = []
    if stage == "topup":
        candidates = await topup_candidates(topic_slug, topic_name, limit_candidates)
    elif stage in ["discovery", "all"] or not candidates_file.exists():
        candidates = await discover_candidates(topic_slug, topic_name, limit_candidates)
    else:
        with open(candidates_file, "r", encoding="utf-8") as f:
            candidates = json.load(f)
            
    if stage in ["generation", "all"] and candidates:
        print(f"=== Generating up to {limit_generate} problems for {topic_name} ===")
        # Sort candidates to ensure deterministic ordering
        candidates = sorted(candidates, key=lambda x: x["slug"])
        
        # Limit to the requested generation limit
        candidates_to_gen = candidates[:limit_generate]
        
        results = await asyncio.gather(*(
            generate_problem_file(topic_slug, topic_name, cand)
            for cand in candidates_to_gen
        ), return_exceptions=True)
        for r in results:
            if isinstance(r, QuotaExhausted):
                raise r
            
    # Update the topic README.md index
    await generate_topic_readme(topic_slug, topic_name)

def topic_scope(topic_slug):
    """Pull the topic's scope blurb out of TAXONOMY.md, if present."""
    tax = ROOT / "TAXONOMY.md"
    if not tax.exists():
        return ""
    for line in tax.read_text(encoding="utf-8").splitlines():
        if f"`{topic_slug}`" in line and line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if cells:
                return cells[-1]
    return ""


def candidate_descriptions(topic_slug):
    """Map slug -> 1-line description from candidates.json, if present."""
    f = ROOT / "topics" / topic_slug / "candidates.json"
    if not f.exists():
        return {}
    try:
        return {c["slug"]: c.get("description", "") for c in json.loads(f.read_text(encoding="utf-8"))}
    except Exception:
        return {}


async def generate_topic_readme(topic_slug, topic_name):
    """Regenerate the topic's README.md based on generated markdown files.

    Preserves the hand-written intro paragraph when one already exists; falls
    back to the topic scope from TAXONOMY.md. Per-problem descriptions come
    from candidates.json.
    """
    topic_dir = ROOT / "topics" / topic_slug
    readme_path = topic_dir / "README.md"

    intro = ""
    if readme_path.exists():
        body = readme_path.read_text(encoding="utf-8").split("## Problems Index")[0]
        intro = "\n".join(l for l in body.splitlines() if l.strip() and not l.startswith("# ")).strip()
    if not intro:
        intro = topic_scope(topic_slug) or "Overview of the research topic and cataloged open problems."

    descriptions = candidate_descriptions(topic_slug)
    if readme_path.exists():
        # Keep descriptions already written into the index for files that have
        # no candidates.json entry (e.g. hand-authored problems).
        for line in readme_path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^\* \S+ \[.*?\]\(\./(.+?)\.md\)\s+\u2014\s+(.+)$", line)
            if m and not descriptions.get(m.group(1)):
                descriptions[m.group(1)] = m.group(2)

    problem_files = sorted(p for p in topic_dir.glob("*.md") if p.name != "README.md")

    lines = [f"# {topic_name}", "", intro, "", "## Problems Index", ""]

    for f in problem_files:
        text = f.read_text(encoding="utf-8")
        title_line = ""
        status = "open"

        for line in text.splitlines():
            if line.startswith("title:"):
                title_line = line.split(":", 1)[1].strip().strip('"').strip("'")
            elif line.startswith("status:"):
                status = line.split(":", 1)[1].strip()

        if not title_line:
            title_line = f.stem.replace("-", " ").title()

        status_emoji = {
            "partially-solved": "\U0001F7E1",
            "empirically-open": "\U0001F7E2",
            "solved-but-impractical": "\U0001F7E0",
            "methodologically-blocked": "\U0001F535",
        }.get(status, "\U0001F534")

        entry = f"* {status_emoji} [{title_line}](./{f.name})"
        desc = descriptions.get(f.stem, "").strip()
        if desc:
            entry += f" \u2014 {desc}"
        lines.append(entry)

    readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Updated README.md index for {topic_slug}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate the AI Research Catalog via the claude or agy CLI "
                    "(set CATALOG_BACKEND=claude|agy; default claude).")
    parser.add_argument("--topic", help="Specific topic slug to generate (default: all topics).")
    parser.add_argument("--stage", choices=["discovery", "topup", "generation", "all"], default="all",
                        help="Execution stage: discovery, generation, or all.")
    parser.add_argument("--limit-candidates", type=int, default=100,
                        help="Number of candidates to discover per topic (default: 100).")
    parser.add_argument("--limit-generate", type=int, default=100,
                        help="Number of files to generate per topic in this run (default: 100).")
    return parser.parse_args()

async def main():
    args = parse_args()
    
    target_topics = {}
    if args.topic:
        if args.topic in TOPICS:
            target_topics = {args.topic: TOPICS[args.topic]}
        else:
            print(f"Error: Unknown topic slug '{args.topic}'.")
            sys.exit(1)
    else:
        target_topics = TOPICS
        
    print(f"Starting AI Research Catalog generator (backend={BACKEND}, "
          f"concurrency={AGY_CONCURRENCY}). Target topics: {len(target_topics)}")
    
    async def _one(slug, name):
        try:
            await process_topic(slug, name, args.limit_candidates, args.limit_generate, args.stage)
        except QuotaExhausted:
            raise
        except Exception as e:
            print(f"Failed processing topic {slug}: {e}")

    try:
        if args.stage == "generation":
            await generate_interleaved(target_topics, args.limit_generate)
        else:
            await asyncio.gather(*(_one(s, n) for s, n in target_topics.items()))
    except QuotaExhausted as e:
        print(f"\nABORT: {BACKEND} quota exhausted -- {e}")
        print("Re-run tools/run_catalog.sh once it resets; generated files are skipped.")
        subprocess.run([str(ROOT / "gen_index.sh")], cwd=str(ROOT))
        sys.exit(2)
            
    print("Regenerating global INDEX.md...")
    subprocess.run([str(ROOT / "gen_index.sh")], cwd=str(ROOT))
    print("Done!")

if __name__ == "__main__":
    asyncio.run(main())
