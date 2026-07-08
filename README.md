# AI Research — problems, worked from first principles

A structured catalog of **AI / LLM research problems**, worked from first principles. Seeded from the
**ERA V5** course (The School of AI): each assignment is captured as a *problem* — a precise formal
statement, the math it rests on, prior art, the approach and results, and — the point of this repo —
the **paper hooks / open questions** each one surfaces.

It mirrors the [`dbms_research`](https://github.com/samyama-ai/dbms_research) catalog in spirit, but where
that maps *open problems in databases*, this one turns *hands-on AI assignments* into a research pipeline:
solve the assignment honestly, then mine it for a genuine, un-scooped research question.

## Layout
- **[`TAXONOMY.md`](./TAXONOMY.md)** — the topic map (tokenization, attention, training, alignment, …).
- **[`INDEX.md`](./INDEX.md)** — flat list of all problems.
- **[`TEMPLATE.md`](./TEMPLATE.md)** — the schema every problem page follows.
- **`topics/NN-topic/<slug>.md`** — one file per problem.

## Provenance & honesty
Assignment prompts are paraphrased from ERA V5 for study; solutions and analysis are the author's own.
Numbers are **measured, not asserted** — every ratio/score is reproducible from the linked code in
`llm_ws/era-v5/`. Paper-hook claims are flagged with a novelty status and must clear a real gate before
they are treated as findings (same honesty-over-reach discipline as the daily-problem program).
