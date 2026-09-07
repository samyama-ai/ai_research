---
id: <NN-topic-slug>/<problem-slug>        # stable identity — never change once assigned
title: "<Problem Title>"
topic: <NN-topic-slug>
status: open | partially-solved | empirically-open | solved-but-impractical | stale
first_added: <YYYY-MM>
last_reviewed: <YYYY-MM>
last_substantive_update: <YYYY-MM>
stale_since: ""
provenance: synthesized                    # synthesized | verified
---

# <Problem Title>

> **Topic:** <topic name> · **ID:** `<NN-topic-slug>/<problem-slug>` · **Status:** <status>

## 1. Problem Statement

Precise and self-contained: the input, the output, the objective or decision predicate, and what would count
as solving it. Distinguish the **measurement**, **method**, and **theory** variants where they differ — in
machine learning these often have very different difficulty, and conflating them is the usual reason a
"problem" resists progress.

## 2. Formal Setting

The formal model. Define the objects (tokens, distributions, parameter vectors, losses, compute budgets), the
notation, and each quantity **as it would actually be measured**. State the assumptions the problem rests on
and flag which are known to be violated in practice. Inline math `$...$`, display `$$...$$`.

## 3. State of the Art

Best-known methods and results — named, with venue and year. Separate **theory SOTA** from **systems/empirical
SOTA** when they differ. Separate what is *established* from what is *claimed but unablated*; where a result
exists only as a benchmark number, say so.

## 4. What Is Known

Established results: theorems, reliable empirical regularities, ablations that have been reproduced
independently. Give numbers where numbers exist, and name the scale they were measured at.

## 5. What Is Not Known

The genuine gap, classified:
- **theoretically open** — no proof either way;
- **empirically open** — the experiment is runnable but nobody has run it at the right scale;
- **methodologically blocked** — the measurement itself is not yet well defined.

## 6. Why It Is Hard

The specific obstruction. Compute cost, confounded measurement, absent ground truth, non-identifiability, or
an evaluation that does not measure the thing it names. "Hard because it is important" is not an obstruction.

## 7. Current Research (as of 2026)

Active directions and, where known, the groups pursuing them. Flag anything beyond confident knowledge with
*(frontier — verify)*.

## 8. Concrete Next Experiment

The smallest experiment that would move the problem, stated so someone could run it: the **scale**, the
**control arm**, and the **number that would decide it**. This section is what separates a catalog entry from
a survey paragraph.

## 9. Key References

- **[Foundational]** Author(s). *Title.* Venue, Year. — arXiv:NNNN.NNNNN
- **[SOTA]** Author(s). *Title.* Venue, Year.
- **[Survey]** Author(s). *Title.* Venue, Year.

References must be real. Where an identifier is uncertain, give authors/title/venue/year and omit the link
rather than guess.

## 10. Worked Example

A small, concrete instance carried through end to end — real numbers, a short calculation, or a minimal
experiment with its expected outcome. The example should make the *obstruction* visible, not just illustrate
the definitions.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*
