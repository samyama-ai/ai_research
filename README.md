# AI Research — a first-principles catalog of open problems

A structured catalog of **open and hard research problems in AI and machine learning**, each worked from
first principles: what the problem actually is, the formal setting, what is genuinely known, what is not,
why it is hard, and the smallest experiment that would move it.

Sibling catalogs, same discipline and structure:
[`dbms_research`](https://github.com/samyama-ai/dbms_research) (databases) and
[`maths_research`](https://github.com/samyama-ai/maths_research) (mathematics).

## Layout
- **[`TAXONOMY.md`](./TAXONOMY.md)** — the topic map (35 topics).
- **[`INDEX.md`](./INDEX.md)** — flat, generated list of every problem.
- **[`TEMPLATE.md`](./TEMPLATE.md)** — the 10-section schema every problem page follows.
- **`topics/NN-topic-slug/<problem-slug>.md`** — one file per problem.

## What a page is for
Not a survey paragraph. Each page has to end with a **concrete next experiment**: the scale, the control
arm, and the number that would decide the question. A problem that cannot be stated that way is either not
yet a research problem or is really a different problem.

Pages separate three kinds of open:
- **theoretically open** — no proof either way;
- **empirically open** — the experiment is runnable, nobody has run it at the right scale;
- **methodologically blocked** — the measurement is not well defined yet.

That distinction is the point of the catalog. Most claimed open problems in ML are the second or third kind,
and saying which changes what to do about them.

## Rebuilt against the literature

This catalog has been carried, unchanged, under an **evidence layer** harvested from OpenAlex and
linked to these problems by typed, provenance-carrying edges — so that "who works on this now"
and "has this been settled" become queries. What that found about *this* catalog:

| | |
|---|---|
| Problems that gained at least one evidence paper | **592** of 1,000 |
| Problems reached by papers this catalog does not cite | **360**, carrying 821 such papers |
| Papers from 2026 reaching the problems | 71, of which **66 are not in this catalog** (its newest cited paper is 2025) |
| Extracted per-paper problems that ground to a catalog problem | 19% of 6,885 |

**One finding is about `TEMPLATE.md`, not about the model.** A refutation can only attach to a
claim that is written down. Of the 8 verdicts we judged correct, 5 were on the minority of
problems whose statement *names the claim under examination*; most statements here pose a
question instead, and a `refutes` verdict then has no target. If you are writing a problem page,
state the claim you believe is in doubt, explicitly.

**No status in this catalog was changed.** The system proposes a status change with its evidence
and never applies one — and the audit says why that matters. We hand-checked **every** `solves`
and `refutes` verdict both catalogs produced, all 41, not a sample: **9 correct, 30 partly
correct, 2 wrong.** No `refutes` (0 of 34) disproves an open problem; only 1 of 7 `solves` closes
the question its statement calls central. The most common error, 9 of 41, reads "current methods
fail" as refuting an open "can it be done?".

One data defect found on the way, since this catalog's citation rules are strict: OpenAlex carries
a **wrong abstract** for one well-known 2020 paper that 54 problems here cite. The error is
upstream, and it put six unrelated methods into a ranking before we caught it.

Everything above is at [`open-problems-as-data`](https://github.com/samyama-ai/open-problems-as-data)
(tag `v1.0-paper29`). The state of this catalog those numbers describe is tagged here as
**`v1.0-paper29`**.

## Honesty rules
- **Citations must be real.** No fabricated DOIs, arXiv IDs, or links. Where an identifier is uncertain,
  cite the metadata (authors, title, venue, year) and omit the link rather than guess.
- **Separate established from claimed.** A benchmark number that has never been ablated is reported as such.
- Frontier claims beyond confident knowledge carry *(frontier — verify)*.
- `tools/audit.py` enforces structure, frontmatter, index sync, and math escaping;
  `--links` resolves every URL and `--ours` checks claimed titles against the live arXiv API.

## Maintenance
```bash
./gen_index.sh                 # regenerate INDEX.md
python3 tools/audit.py         # structure + sync + safety
python3 tools/audit.py --links # also resolve every external URL (slow)
```
