---
id: 26-code-generation/learned-invariant-generation-loops
title: "Learned Invariant Generation for Loop Verification"
topic: 26-code-generation
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Learned Invariant Generation for Loop Verification

> **Topic:** Code Generation & Program Synthesis · **ID:** `26-code-generation/learned-invariant-generation-loops` · **Status:** partially-solved

## 1. Problem Statement

Given a program with a loop, a precondition, and a postcondition, produce a **loop invariant** — a formula that holds on entry, is preserved by the body, and is strong enough to imply the postcondition. The invariant is then checked by a symbolic verifier (Z3, CVC5, Frama-C/WP, SeaHorn), so a wrong guess costs a failed check, not an unsound proof. The learning problem is: can a model trained on programs propose invariants faster or more often than search-based invariant generators?

Three variants, different difficulty:

- **Method.** Build a proposer $M$ that maximizes the fraction of benchmark programs for which some proposal passes the verifier within a call budget. Largely solved on the standard suites, unsolved in the wild.
- **Measurement.** Define a benchmark whose solve rate predicts performance on unseen, non-curated code. Not solved — the standard suites are small, memorized, and syntactically stereotyped.
- **Theory.** Characterize the invariant classes for which learning gives an asymptotic advantage over complete search. Essentially untouched.

Solving it means: on held-out real-world loops (not SV-COMP), a learned proposer beats the best CHC solver (Spacer, FreqHorn) on solve rate at matched wall-clock, and the gap survives decontamination.

## 2. Formal Setting

A loop program is a transition system $P = (X, \mathit{Init}, T, \mathit{Bad})$ over program variables $X$, where $\mathit{Init}(X)$ and $\mathit{Bad}(X)$ are formulas and $T(X, X')$ is the body's transition relation. $I(X)$ is an **inductive invariant sufficient for safety** iff

$$\mathit{Init} \Rightarrow I, \qquad I \wedge T \Rightarrow I', \qquad I \wedge \mathit{Bad} \Rightarrow \bot .$$

All three are quantifier-free implications; each is one SMT call, and validity is decidable for linear integer arithmetic (LIA) and linear real arithmetic. **The check is cheap; the search is not.** Deciding whether *any* $I$ exists in an unrestricted language is undecidable (it subsumes reachability).

Measured quantities:

- **Solve rate** $\rho = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}[\exists k \le K: \text{verifier accepts } I_i^{(k)}]$, over $N$ programs, $K$ proposals. $K$ must be reported; $\rho$ at $K{=}1$ and $\rho$ at $K{=}32$ are different problems.
- **Verifier calls** $C$ — the real cost unit for LLM pipelines, because each rejected candidate is one Z3 invocation plus one model sample.
- **Wall clock** $\tau$ per program, with timeout $T_{\max}$ (commonly 60 s or 300 s). Solve rate without $T_{\max}$ is meaningless.
- **Grammar** $\mathcal{G}$: the fragment invariants are drawn from — conjunctions of octagons ($\pm x \pm y \le c$), general LIA, polynomial equalities of degree $d$, or arrays with quantifiers. Cross-paper comparison is invalid across different $\mathcal{G}$.
- **Contamination** $\kappa$: fraction of test programs whose source text or a near-duplicate appears in pretraining. Rarely measured; for SV-COMP-derived suites, publicly on GitHub since 2012, assume $\kappa$ is high.

Assumptions, and which fail in practice:

1. *Deterministic, side-effect-free loop body over scalars.* Violated by heap, aliasing, and calls — most learned systems either drop these benchmarks or assume separation-logic frames given.
2. *The verifier is complete for the invariant language.* Holds for LIA; **violated** for nonlinear integer arithmetic, where Z3 may return `unknown` and a correct invariant is scored as a failure.
3. *The postcondition is given.* Violated in real code, where the specification is the harder half of the problem.
4. *Benchmark i.i.d. with deployment.* Known violated: SV-COMP loop programs are hand-written verification puzzles, not application code.

## 3. State of the Art

**Theory SOTA.** For affine programs, Hrushovski, Ouaknine, Pouly and Worrell (LICS 2018) give an algorithm computing the *strongest polynomial (algebraic) invariant*, closing a long-open question. Kincaid et al. (POPL 2018) give compositional recurrence-based synthesis for nonlinear invariants. These are complete for their fragments and need no learning; they bound what a learned method can add.

**Search SOTA (non-learned).** Property-directed reachability / IC3 (Bradley, VMCAI 2011) and its Horn-clause form Spacer (Komuravelli et al., CAV 2014) inside Z3 remain the baseline that learned systems must beat and often do not report against at matched compute. ICE learning (Garg et al., CAV 2014) frames invariant inference as learning from implication counterexamples — the template every subsequent learned system instantiates.

**Learned SOTA (established).**
- **Code2Inv** (Si et al., NeurIPS 2018): graph neural network over the program AST plus reinforcement learning from counterexamples; solves 106 of 133 programs in its own benchmark suite.
- **CLN2INV** (Ryan et al., ICLR 2020): continuous logic networks make the invariant's truth value differentiable, so gradient descent fits coefficients directly. Solves all 124 of the 133 benchmarks that are theoretically solvable, with a reported ~40× speedup over Code2Inv, average solve time about 1.1 s.
- **G-CLN** (Yao et al., PLDI 2020): extends CLN2INV to nonlinear invariants; solves 26 of 27 polynomial loops in the NLA suite.

**Learned SOTA (claimed, thinly ablated).** LLM-based proposers — Pei et al. (ICML 2023) on predicting Daikon-style invariants, Chakraborty et al. (EMNLP Findings 2023) on *ranking* candidate invariants to cut verifier calls, and the Loopy line of work from Microsoft Research (Kamath, Lal et al., arXiv 2023) coupling GPT-4 proposals to Frama-C — report high solve rates on SV-COMP-derived suites. These are **benchmark numbers without decontamination**: no published run holds contamination fixed, and the ablation separating "the model recognized the program" from "the model reasoned about the transition relation" has not been done.

## 4. What Is Known

- The Code2Inv suite (133 programs, LIA, single loop, scalar variables, derived from SV-COMP and SyGuS) is **saturated**: CLN2INV solves every solvable instance. Scale: 133 programs, seconds each. Further gains on it measure nothing.
- Gradient-based coefficient fitting works when the invariant is a conjunction of linear inequalities over $\le 6$ variables; it degrades sharply with disjunction. Established across CLN2INV and G-CLN.
- Counterexample-guided loops converge: each rejected candidate yields a concrete state that prunes the hypothesis space, and empirically 5–20 rounds suffice on the standard suites.
- Ranking helps more than generating. Chakraborty et al. report that reordering an LLM's candidate list by a learned ranker solves more problems at a substantially smaller verifier-call budget than unranked sampling — the strongest reproduced effect in the LLM branch, measured on a few hundred programs.
- Daikon (Ernst et al., TSE 2001) infers *likely* invariants from traces at industrial scale but gives no soundness guarantee; roughly half of its output is typically not inductive.
- Hoare-triple checking for LIA invariants is decidable and fast — under 10 ms for typical benchmark formulas. The bottleneck is proposal, not checking.

## 5. What Is Not Known

- **Empirically open.** Does any learned proposer beat Spacer on *uncurated* loops from real repositories at matched wall clock? Runnable today: mine loops from a package ecosystem, generate specs, run both. Nobody has published it at the scale ($10^4$+ loops) that would settle it.
- **Empirically open.** How much of LLM invariant performance is retrieval of memorized SV-COMP solutions? The decontamination experiment (semantics-preserving renaming and restructuring, measure $\Delta\rho$) is cheap and unrun at scale.
- **Methodologically blocked.** There is no accepted difficulty metric for a loop-invariant instance. Solve rates are averaged over suites whose instances range from trivial to open, so a 5-point $\rho$ gain is uninterpretable.
- **Methodologically blocked.** Nonlinear benchmarks conflate "no invariant found" with "Z3 returned `unknown`". Until these are separated, nonlinear solve rates are not measurements.
- **Theoretically open.** No sample-complexity result for invariant learning: no bound of the form "with $n$ training programs from distribution $\mathcal{D}$, expected verifier calls on a fresh $P \sim \mathcal{D}$ is $O(\cdot)$". ICE learning gives convergence for a fixed hypothesis class, not generalization across programs.

## 6. Why It Is Hard

The specific obstruction is **an evaluation that does not measure what it names**, compounded by contamination. "Loop invariant generation" is scored on suites of 133–600 hand-written puzzle programs that have been on GitHub for a decade, in a syntactic style (`while (i < n) { x = x + y; }`, integer scalars, no heap) that occupies a vanishing fraction of real code. A model can score highly by pattern-matching the suite's idioms. Because the verifier accepts anything sound, high solve rate is *sound* — but it is not *evidence of generalization*, and the two get reported interchangeably.

Second obstruction: **absent ground truth for the specification**. In real code there is no postcondition, so the pipeline must invent one; a weak invented spec is trivially provable and inflates $\rho$. There is no accepted way to score spec quality, so the end-to-end task cannot currently be benchmarked at all.

Third: **the disjunction wall**. Invariants requiring case splits ($x \ge 0 \vee y < n$) blow up the hypothesis space multiplicatively, and neither gradient fitting nor next-token sampling has a mechanism that targets the split points.

## 7. Current Research (as of 2026)

- **LLM proposer + verifier feedback loops.** Microsoft Research (Lal's group) on Loopy and successors; the Frama-C/ACSL pipeline for C. Iterating counterexamples back into the prompt is now standard.
- **Learned ranking and repair over candidate pools** — cheaper than better generation, since the verifier is the oracle. Chakraborty, Lahiri and collaborators.
- **Neurosymbolic hybrids** feeding LLM guesses as lemma hints into Spacer/FreqHorn rather than as complete invariants, so the solver repairs partial guesses *(frontier — verify)*.
- **Proof-assistant transfer.** Groups working on Lean/Dafny invariant annotation are pushing toward richer logics with quantifiers and arrays *(frontier — verify)*.
- **Benchmark reconstruction.** Repository-mined, contamination-controlled loop suites are the obvious next artifact; as of this writing no such suite has become standard *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question:** is LLM invariant generation reasoning or retrieval?

**Scale.** Take the 133 Code2Inv programs plus ~400 SV-COMP `loops` programs (~533 total; hours of compute, not days). Produce two semantics-preserving transforms of each: (a) *rename* — all identifiers replaced with fresh opaque names, constants left intact; (b) *restructure* — rename, plus loop rewritten (`while`↔`for`), statements reordered where independent, and one dead variable introduced. Transforms verified equivalent by symbolic execution on the loop body.

**Arms.** (1) Original programs, LLM proposer, $K = 16$ samples, Z3 check, 60 s timeout. (2) Rename. (3) Restructure. **Control arm:** Spacer on all three variants at the same 60 s budget — Spacer is transform-invariant by construction, so its $\rho$ must be flat across arms; any drift exposes a broken transform rather than a model effect.

**Deciding number.** $\Delta\rho = \rho_{\text{original}} - \rho_{\text{restructure}}$, with 95% CI from a paired bootstrap over the 533 programs. $\Delta\rho \le 3$ points → the capability is semantic and the benchmark numbers are informative. $\Delta\rho \ge 15$ points → the reported solve rates are substantially retrieval, and every downstream comparison in the LLM branch needs re-running. Precision at $N=533$ is roughly $\pm 4$ points, enough to separate those hypotheses.

## 9. Key References

- **[Foundational]** Michael D. Ernst, Jake Cockrell, William G. Griswold, David Notkin. *Dynamically Discovering Likely Program Invariants to Support Program Evolution.* IEEE TSE, 2001.
- **[Foundational]** Pranav Garg, Christof Löding, P. Madhusudan, Daniel Neider. *ICE: A Robust Framework for Learning Invariants.* CAV, 2014.
- **[Foundational]** Aaron R. Bradley. *SAT-Based Model Checking Without Unrolling.* VMCAI, 2011.
- **[Theory SOTA]** Ehud Hrushovski, Joël Ouaknine, Amaury Pouly, James Worrell. *Polynomial Invariants for Affine Programs.* LICS, 2018.
- **[Theory]** Zachary Kincaid, Jason Breck, John Cyphert, Thomas Reps. *Closed Forms for Numerical Loops.* POPL, 2019.
- **[SOTA]** Xujie Si, Hanjun Dai, Mukund Raghothaman, Mayur Naik, Le Song. *Learning Loop Invariants for Program Verification.* NeurIPS, 2018.
- **[SOTA]** Gabriel Ryan, Justin Wong, Jianan Yao, Ronghui Gu, Suman Jana. *CLN2INV: Learning Loop Invariants with Continuous Logic Networks.* ICLR, 2020. — arXiv:1909.11542
- **[SOTA]** Jianan Yao, Gabriel Ryan, Justin Wong, Suman Jana, Ronghui Gu. *Learning Nonlinear Loop Invariants with Gated Continuous Logic Networks.* PLDI, 2020.
- **[LLM]** Kexin Pei, David Bieber, Kensen Shi, Charles Sutton, Pengcheng Yin. *Can Large Language Models Reason about Program Invariants?* ICML, 2023.
- **[LLM]** Saikat Chakraborty, Shuvendu K. Lahiri, Sarah Fakhoury, Madanlal Musuvathi, Akash Lal, Aseem Rastogi, Aditya Senthilnathan, Rahul Sharma, Nikhil Swamy. *Ranking LLM-Generated Loop Invariants for Program Verification.* Findings of EMNLP, 2023.
- **[Solver]** Anvesh Komuravelli, Arie Gurfinkel, Sagar Chaki. *SMT-Based Model Checking for Recursive Programs.* CAV, 2014.

## 10. Worked Example

Take the canonical benchmark loop:

```c
int x = 0, y = 0;
while (nondet()) { x = x + 1; y = y + 2; }
assert(y == 2 * x);
```

The invariant $I \equiv (y = 2x)$ discharges all three conditions. $\mathit{Init}$: $x{=}0 \wedge y{=}0 \Rightarrow y{=}2x$. Induction: $y{=}2x \wedge x'{=}x{+}1 \wedge y'{=}y{+}2 \Rightarrow y'{=}2x'$. Safety: immediate. Z3 discharges all three in under 5 ms. GPT-class models emit `y == 2*x` on the first sample; solve rate 1.0 at $K{=}1$.

Now perturb the constant: change `y = y + 2` to `y = y + 3` and the assertion to `3*x == y`. Semantically the same shape, one token different. Reported behavior is unchanged — models still succeed, because they are pattern-completing the affine relation.

Now restructure instead. Replace the counter pair with a single accumulator whose relation is $y = x(x{+}1)$:

```c
int x = 0, y = 0;
while (nondet()) { y = y + 2*x + 1 + 1; x = x + 1; }
assert(y == x*x + x);
```

The invariant $y = x^2 + x$ is degree 2. Three things break at once. (i) The induction check $y{=}x^2{+}x \wedge y'{=}y{+}2x{+}2 \wedge x'{=}x{+}1 \Rightarrow y'{=}x'^2{+}x'$ is nonlinear integer arithmetic; Z3 may answer `unknown` rather than `unsat` depending on tactic, so a **correct** invariant can be scored as a failure. (ii) CLN2INV's gradient fitting cannot reach it — the template is linear. (iii) An LLM that has memorized the linear family proposes $y = 2x$, $y = kx$, $y \ge 0$ across all 16 samples, burning 16 verifier calls for nothing.

The obstruction is visible in the accounting: the first program contributes $+1$ to solve rate at a cost of 5 ms; the third contributes $0$ at a cost of 16 samples plus 16 SMT calls, of which at least one failure is the *verifier's* incompleteness, not the proposer's. A suite dominated by instances of the first kind reports $\rho \approx 0.95$ and tells you nothing about the third.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*