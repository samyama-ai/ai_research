---
id: <NN-topic-slug>/<problem-slug>        # stable identity — never change once assigned
title: "<Problem Title>"
topic: <NN-topic-slug>
status: open | in-progress | solved | submitted | graded
source: "<e.g. ERA V5 · Session N assignment>"   # where the problem came from
due: <YYYY-MM-DD or "">                    # if it is a graded assignment
score_metric: "<the exact objective, e.g. 1000/(X4-X1)>"
first_added: <YYYY-MM>
last_update: <YYYY-MM>
provenance: worked-from-first-principles
---

# <Problem Title>

> **Topic:** <topic name> · **Source:** <ERA V5 · Session N> · **Status:** <status> · **Metric:** `<score_metric>`

## 1. Problem Statement
Precise, self-contained statement: inputs, outputs, objective/score, constraints, and what counts as a
correct submission. Paraphrase the assignment; do not paste proprietary text verbatim.

## 2. First-Principles Formalization
Strip the wording to the underlying math. Define the quantities exactly as they will be *measured/graded*
(a definition mismatch is usually the top risk). Derive what the score actually rewards.

## 3. Prior Art / SOTA
The techniques and results this rests on — with venue/year. Separate what is standard from what is open.
Link related work in this repo and in `dbms_research` where relevant.

## 4. Approach
The method, derived — not a black box. Key design choices and *why*, each tied back to the objective.

## 5. Results
Measured numbers (reproducible from the linked code). Baselines → optimized. Honest about what failed.

## 6. Paper Hooks / Open Questions
The research questions this assignment surfaces. For each: a one-line claim, a novelty status
(RED conceded / YELLOW narrowed / GREEN open — *unverified until gated*), and why it might matter.
This is the reason the repo exists.

## 7. Artifact / Submission
Where the code lives (`llm_ws/…`), the deliverable (widget/URL/download), and the self-reported score.

## 8. References
Linked, verified or flagged `*(unverified)*`.
