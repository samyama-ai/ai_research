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
