---
id: 26-code-generation/learning-new-domain-specific-languages
title: "Data-Efficient Learning of New Domain-Specific Languages"
topic: 26-code-generation
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Data-Efficient Learning of New Domain-Specific Languages

> **Topic:** Code Generation & Program Synthesis · **ID:** `26-code-generation/learning-new-domain-specific-languages` · **Status:** open

## 1. Problem Statement

A domain-specific language (DSL — a small language with its own grammar and semantics, built for one domain: a query language, a hardware description language, a robot task language, an internal config language) appears in a company's repo with 40 pages of documentation, 200 example programs, and no presence in any pretraining corpus. How much of that material does a model need to write correct programs in it?

**Input.** A DSL $L$ given as a grammar, a reference interpreter, documentation of $B$ bytes, and $n$ worked examples. A held-out set of natural-language task specifications with executable tests.

**Output.** A program in $L$ per specification.

**Objective.** Minimize the acquisition budget $(B, n)$ needed to reach a target functional-correctness rate.

Three variants, routinely conflated:

- **Measurement.** Can we even build a benchmark whose DSL is provably absent from pretraining? Every published "novel DSL" result is confounded by unverifiable contamination and by transfer from syntactically similar known languages. This variant is the binding one.
- **Method.** Given a fixed budget, what beats the alternatives: in-context documentation, retrieval over docs, grammar-constrained decoding, fine-tuning on synthetic-but-tested programs, or library learning that induces the abstractions itself?
- **Theory.** What sample complexity is achievable for learning a semantics-carrying grammar from documentation plus examples, and does a pretrained prior provably reduce it?

Solved means: a stated bound or reliable empirical law relating documentation bytes and example count to correctness on a DSL verified novel, with a demonstrated method that hits a fixed correctness target at an order of magnitude less data than fine-tuning.

## 2. Formal Setting

A DSL is $L = (G, \Sigma, [\![\cdot]\!])$: grammar $G$ over terminals $\Sigma$, and a semantics $[\![\cdot]\!] : \mathcal{L}(G) \to (\mathcal{I} \to \mathcal{O})$ realized by an interpreter. Tasks are drawn as $t = (u, T)$ with $u$ a natural-language spec and $T$ a finite test set of input–output pairs.

**Correctness, as measured.** For a model $M$ conditioned on context $c$, sample $k$ programs and run the interpreter:

$$\text{pass@}k(M, c) = \mathbb{E}_{t \sim \mathcal{D}}\Big[\,\Pr_{p_{1..k} \sim M(\cdot \mid u, c)}\big[\exists i : \forall (x,y) \in T,\ [\![p_i]\!](x) = y\big]\Big]$$

estimated with the unbiased $n$-sample estimator of Chen et al. (2021). Two failure modes must be reported separately: *parse failure* ($p \notin \mathcal{L}(G)$) and *semantic failure* (parses, wrong output). Collapsing them hides which intervention is doing the work.

**Acquisition budget.** $B$ = bytes of DSL documentation in context or in the fine-tuning set; $n$ = number of (spec, program, tests) triples. Report both, plus tokens of compute, since fine-tuning arms and in-context arms are otherwise incomparable.

**Data efficiency.** The quantity of interest is the inverse curve:

$$n^*(\tau) = \min\{\, n : \text{pass@}1 \ge \tau \,\}, \qquad \tau = 0.5 \text{ by convention.}$$

**Novelty.** Define contamination-adjusted novelty of $L$ for model $M$ as the zero-shot rate with the DSL *named but not documented*: $\nu(L, M) = 1 - \text{pass@}1(M, \varnothing)$. $\nu = 1$ is necessary but not sufficient for novelty — a model can fail zero-shot and still absorb the language faster than a genuinely novel one because a near-isomorphic language is in its prior.

**Assumptions, and which are violated.**
- *Tests characterize the spec.* Violated: DSL test suites are small, so semantically wrong programs pass. Measured on general code benchmarks by Liu et al. (NeurIPS 2023), where a stronger test suite dropped reported pass@1 by up to ~13 points.
- *The DSL is novel.* Violated in every public benchmark; novelty is asserted, not verified.
- *Documentation is complete.* Violated: real DSLs have undocumented coercions and evaluation-order quirks recoverable only from the interpreter.
- *Tasks are i.i.d.* Violated: benchmark tasks are typically authored by one person from one mental template.

## 3. State of the Art

**Established (ablated, reproduced).**
- *Grammar-constrained decoding removes parse errors.* PICARD (Scholak et al., EMNLP 2021) and Synchromesh (Poesia et al., ICLR 2022) constrain generation to a grammar/schema at decode time and lift executable accuracy without touching weights. The mechanism is checkable: the parse-failure rate goes to zero by construction.
- *Library learning compresses a DSL from tasks alone.* DreamCoder (Ellis et al., PLDI 2021) alternates wake-phase search with sleep-phase abstraction, growing a DSL from primitives across eight domains. STITCH (Bowers et al., POPL 2023) replaces its compression step with a top-down search reported at up to ~3 orders of magnitude faster and ~2 orders less memory at equal or better compression — an ablation of one component, independently checkable.
- *Retrieval over docs beats parametric memory for unseen library calls.* DocPrompting (Zhou et al., ICLR 2023) shows gains on held-out-API splits of CoNaLa and a Bash benchmark.

**Claimed but under-ablated.**
- Grammar prompting (Wang et al., NeurIPS 2023) puts a BNF-style specialized grammar in the prompt and reports gains on semantic parsing (GeoQuery, SMCalFlow, Overnight), PDDL action generation, and molecule SMILES. The confound: how much comes from the grammar versus from the retrieved exemplars is not cleanly separated.
- MultiPL-T (Cassano et al., OOPSLA 2024) fine-tunes on synthetic-but-test-validated programs translated into low-resource languages (Racket, OCaml, Lua, R, Julia) and reports large pass@1 gains over base models. These are *low-resource*, not novel languages — the prior is present, just thin.

**Benchmark-number-only.** VerilogEval (Liu et al., ICCAD 2023) and similar per-DSL suites report pass@$k$ for a fixed model set. They give no acquisition curve: no arm varies $B$ or $n$.

## 4. What Is Known

- **Grammar constraints eliminate parse failure, not semantic failure.** Across the constrained-decoding literature, residual errors after constraining are overwhelmingly semantic. Scale: 7B–175B models, SQL/semantic-parsing benchmarks of $10^3$–$10^4$ tasks.
- **Library learning works at small scale and does not obviously scale.** DreamCoder's domains are on the order of $10^2$ tasks with programs of a few dozen tokens; LILO (Grand et al., ICLR 2024) adds an LLM proposer and documentation-writing step and beats DreamCoder+STITCH on REGEX, CLEVR, and LOGO. No result exists at the scale of a real 200-construct industrial DSL.
- **A grammar book substitutes for corpus data in the analogous NL case.** MTOB (Tanzer et al., ICLR 2024) has a model translate Kalamang from one grammar reference plus a small word list; the best long-context models reach chrF within roughly 5–7 points of a human learner given the same book. This is the strongest existing evidence that documentation-in-context is a real substitute for training data — in a *language* setting, not a DSL setting with an executable oracle.
- **Many-shot in-context learning continues improving to hundreds/thousands of examples** (Agarwal et al., NeurIPS 2024), including on low-resource translation and planning — so the in-context arm's ceiling is higher than the classic few-shot literature implied.
- **Negative theory carries over.** Gold (1967): superfinite language classes are not identifiable in the limit from positive data alone. Kearns & Valiant (1994): learning DFAs from examples is hard under cryptographic assumptions. Angluin (1987): $L^*$ learns regular languages in polynomial time — but requires an equivalence oracle, which an interpreter alone does not give you.

## 5. What Is Not Known

- **Methodologically blocked.** There is no accepted procedure to certify a DSL is absent from a frontier model's pretraining data. Without it, "data-efficient learning of a *new* DSL" cannot be distinguished from "recall of a rare DSL." This blocks the whole area, not one result.
- **Empirically open.** The acquisition curve itself: nobody has published $\text{pass@}1$ as a function of $(B, n)$ for a single DSL with matched in-context, retrieval, and fine-tuning arms on one model. The experiment is cheap. It has not been run.
- **Empirically open.** Whether the interpreter, used as a verifier in a self-training loop, beats an equal budget of human-written examples — and at what exchange rate (how many interpreter calls buy one labeled example).
- **Theoretically open.** No sample-complexity bound for grammar-plus-semantics acquisition in which the pretrained prior appears as a parameter. The classical hardness results assume no prior; the empirical regime is entirely about the prior.

## 6. Why It Is Hard

**The specific obstruction is a measurement confound with no available control.** The independent variable — novelty of the DSL — is unmeasurable on the models people care about, because pretraining corpora are closed and because near-isomorphism to a known language transfers even when the surface tokens are unseen. Two corollaries:

1. **Construction is not a fix.** Inventing a fresh DSL controls token-level novelty but not structural novelty: a freshly-named Lisp is not new to the model. Randomizing the syntax to break structural transfer produces a language nobody would ship, so the result stops generalizing to the practical case.
2. **The evaluation does not measure what it names.** pass@$k$ on 100–200 hand-written tasks with 3–5 tests each measures "can produce a program that passes weak tests," and grammar constraints inflate it by removing parse errors — a component of the score that has nothing to do with having learned the semantics.

Secondary obstruction: cost asymmetry across arms. A fine-tuning arm costs GPU-hours; an in-context arm costs a long-context forward pass. Comparing them at "equal data" is not comparing them at equal compute, and papers pick whichever normalization flatters the method.

## 7. Current Research (as of 2026)

- **Library learning with LLM proposers.** MIT (Ellis, Tenenbaum, Grand, Bowers lineage) — LILO-style loops where the model writes the abstraction *and its documentation*, so the induced DSL stays human-readable. Open question is scaling past toy domains.
- **Interpreter-in-the-loop RL for DSL correctness.** Execution feedback as reward for grammar-constrained generation; well-motivated but the published ablations are mostly on mainstream languages, not novel DSLs. *(frontier — verify)*
- **Long-context documentation ingestion.** MTOB-style "learn it from the manual" transferred to DSLs, using million-token contexts to hold the whole language reference. *(frontier — verify)*
- **Contamination-controlled benchmark construction.** Held-out-by-date and synthetically generated language suites; no community standard yet. *(frontier — verify)*
- **Industrial low-resource transfer.** MultiPL-T-style test-validated synthetic corpora extended to proprietary internal languages. *(frontier — verify)*

## 8. Concrete Next Experiment

**Question.** Does documentation in context substitute for labeled examples, and at what exchange rate?

**Scale.** One DSL of ~40 constructs with a reference interpreter, constructed so that its evaluation semantics (e.g. lazy, effect-tracked) differs from any mainstream language while its syntax is conventional. 300 tasks with $\ge 10$ tests each; 200 train / 100 test. One open-weights model at ~8B, one at ~70B, plus one frontier API model. Total under 2,000 GPU-hours.

**Arms.** Grid over $B \in \{0, 2\text{k}, 20\text{k}, 200\text{k}\}$ documentation bytes $\times$ $n \in \{0, 4, 16, 64, 200\}$ examples, each run twice: unconstrained decoding and grammar-constrained decoding. Fine-tuning arm at the same $n$, compute-matched by reporting FLOPs.

**Control arm.** The identical DSL with all identifiers renamed to a mainstream language's vocabulary and semantics reverted to standard eager evaluation — same task set, same tests. Its acquisition curve is the prior-transfer baseline; the gap between the two curves *is* the novelty effect, and it is measured rather than asserted.

**Deciding number.** $\rho = n^*(0.5)_{B=0} \big/ n^*(0.5)_{B=200\text{k}}$ — the factor by which 200 kB of documentation reduces the examples needed for 50% pass@1, on the novel DSL, with parse failures excluded from both. $\rho \ge 4$ says documentation is a genuine substitute for supervision and the field should push long-context ingestion. $\rho \le 1.5$ says documentation is decorative and only executed examples teach semantics. Report the same $\rho$ on the control arm; if the two are equal, the "novel DSL" framing is measuring nothing.

## 9. Key References

- **[Foundational]** E. M. Gold. *Language Identification in the Limit.* Information and Control, 1967.
- **[Foundational]** D. Angluin. *Learning Regular Sets from Queries and Counterexamples.* Information and Computation, 1987.
- **[Foundational]** M. Kearns, L. Valiant. *Cryptographic Limitations on Learning Boolean Formulae and Finite Automata.* Journal of the ACM, 1994.
- **[Foundational]** S. Gulwani. *Automating String Processing in Spreadsheets Using Input-Output Examples.* POPL, 2011.
- **[Foundational]** M. Chen et al. *Evaluating Large Language Models Trained on Code.* arXiv, 2021. — arXiv:2107.03374
- **[SOTA]** K. Ellis, C. Wong, M. Nye, M. Sablé-Meyer, L. Morales, L. Hewitt, L. Cary, A. Solar-Lezama, J. B. Tenenbaum. *DreamCoder: Bootstrapping Inductive Program Synthesis with Wake-Sleep Library Learning.* PLDI, 2021.
- **[SOTA]** M. Bowers, T. X. Olausson, L. Wong, G. Grand, J. B. Tenenbaum, K. Ellis, A. Solar-Lezama. *Top-Down Synthesis for Library Learning.* POPL, 2023.
- **[SOTA]** G. Grand, L. Wong, M. Bowers, T. X. Olausson, M. Liu, J. B. Tenenbaum, J. Andreas. *LILO: Learning Interpretable Libraries by Compressing and Documenting Code.* ICLR, 2024.
- **[SOTA]** B. Wang, Z. Wang, X. Wang, Y. Cao, R. A. Saurous, Y. Kim. *Grammar Prompting for Domain-Specific Language Generation with Large Language Models.* NeurIPS, 2023.
- **[SOTA]** T. Scholak, N. Schucher, D. Bahdanau. *PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding from Language Models.* EMNLP, 2021.
- **[SOTA]** G. Poesia, O. Polozov, V. Le, A. Tiwari, G. Soares, C. Meek, S. Gulwani. *Synchromesh: Reliable Code Generation from Pre-trained Language Models.* ICLR, 2022.
- **[SOTA]** S. Zhou, U. Alon, F. F. Xu, Z. Jiang, G. Neubig. *DocPrompting: Generating Code by Retrieving the Docs.* ICLR, 2023.
- **[SOTA]** F. Cassano, J. Gouwar, F. Lucchetti, C. Schlesinger, A. Freeman, C. J. Anderson, M. Q. Feldman, M. Greenberg, A. Jangda, A. Guha. *Knowledge Transfer from High-Resource to Low-Resource Programming Languages for Code LLMs.* OOPSLA, 2024.
- **[SOTA]** G. Tanzer, M. Suzgun, E. Visser, D. Jurafsky, L. Melas-Kyriazi. *A Benchmark for Learning to Translate a New Language from One Grammar Book.* ICLR, 2024.
- **[SOTA]** R. Agarwal et al. *Many-Shot In-Context Learning.* NeurIPS, 2024.
- **[Survey]** F. Chollet. *On the Measure of Intelligence.* arXiv, 2019. — arXiv:1911.01547
- **[Survey]** S. Gulwani, O. Polozov, R. Singh. *Program Synthesis.* Foundations and Trends in Programming Languages, 2017.

## 10. Worked Example

**The DSL.** `PIPE`, a 12-construct stream language. Syntax is Lisp-like. One non-standard rule: `(map f s)` on an infinite stream is lazy, and `(fold ...)` on an unbounded stream is a static error the interpreter refuses to run. Documentation: 18 kB. Tasks: 100, 8 tests each.

**Arm A — zero documentation, DSL named only.** A 70B open-weights model. Result pattern to expect: parse failure ~35%, and of the programs that parse, most compile a `fold` over an unbounded stream, because the model's Lisp prior says `fold` is total. Suppose pass@1 = 0.06.

**Arm B — full 18 kB doc in context, 0 examples.** Parse failure drops to ~5%. But the lazy/unbounded rule is stated once, in one sentence, on page 9. Suppose pass@1 = 0.19.

**Arm C — grammar-constrained decoding, full doc, 0 examples.** Parse failure = 0 by construction. pass@1 = 0.24. The headline improvement over Arm A is $0.24 - 0.06 = 0.18$, of which $0.35 - 0.05 = 0.30$ of the *attempt* mass was recovered purely by making outputs parseable — an effect that carries no evidence the model learned `fold`'s restriction.

**Arm D — full doc + 16 executed examples, 3 of which trip the `fold` error.** pass@1 = 0.55. The jump comes almost entirely from tasks whose reference solution needs `take` before `fold`.

**Where the obstruction becomes visible.** Two readings of the same table:

| Reading | Claim | Support |
|---|---|---|
| Method | Docs give $+0.13$, examples give $+0.31$ | Requires that `PIPE` be novel |
| Confound | Model already knows Lisp; docs only re-map surface names | Untested without a control |

Run the control: `PIPE` renamed to Scheme-standard identifiers with eager `fold`. If Arm B on the control is 0.62 and Arm B on `PIPE` is 0.19, the 0.43 gap is prior transfer, and the paper's real finding is that *one sentence of semantics on page 9 costs 16 examples to install* — not anything about learning a language. If the control's curve is also flat, the DSL was novel and the acquisition numbers mean what they say. No published DSL-acquisition paper reports this control. That is the gap.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*