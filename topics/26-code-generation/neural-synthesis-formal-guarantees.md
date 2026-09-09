---
id: 26-code-generation/neural-synthesis-formal-guarantees
title: "Neural Synthesis with Formal Correctness Guarantees"
topic: 26-code-generation
status: solved-but-impractical
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Neural Synthesis with Formal Correctness Guarantees

> **Topic:** Code Generation & Program Synthesis · **ID:** `26-code-generation/neural-synthesis-formal-guarantees` · **Status:** solved-but-impractical

## 1. Problem Statement

- **Input:** a task description $d$ — natural language, examples, or a formal specification — plus a target language and a verifier.
- **Output:** a triple $(P, \varphi, \pi)$: a program, a formal specification, and a proof certificate that a sound checker accepts as establishing $P \models \varphi$.
- **Predicate:** the artifact is *machine-checked correct* if the checker accepts, and *actually correct* if $\varphi$ also captures the intent behind $d$.

Three variants, routinely conflated:

- **Method variant (largely solved, impractical).** Build a neural system that emits verifier-accepted code at usable rates. Working systems exist for Dafny, Verus, F\*, and Lean. They are expensive, brittle outside their training distributions, and confined to small, self-contained functions.
- **Measurement variant (blocked).** Quantify how often an accepted $\varphi$ actually means what $d$ meant. There is no accepted metric for specification adequacy, so the headline "verified" numbers do not measure end-to-end correctness.
- **Theory variant (open in the relevant form).** Undecidability settles the worst case (Rice, 1953); what is open is whether verified-synthesis success rate scales predictably with model compute the way unverified `pass@1` does.

Solving it means: an unassisted system that takes an informal task, emits code plus specification plus proof, is accepted by a sound checker, and where the specification is independently audited to encode the intent — at a cost within a small constant factor of unverified generation.

## 2. Formal Setting

Let $\Sigma$ be the program space, $\Phi$ the specification language, $\Pi$ proof objects. A checker
$$\mathcal{C}: \Sigma \times \Phi \times \Pi \to \{0,1\}$$
is **sound** if $\mathcal{C}(P,\varphi,\pi)=1 \Rightarrow P \models \varphi$. Measured as: trusted computing base (TCB) in lines of code — Z3 plus the Dafny/Verus encoding is $\sim 10^5$ LOC and unverified; Lean's kernel is $\sim 10^3$ LOC.

A synthesizer is a conditional distribution $q_\theta(P,\varphi,\pi \mid d)$. Two quantities:

$$\mathrm{VR}@k(d) = \Pr_{(P_i,\varphi_i,\pi_i)\sim q_\theta^{\otimes k}}\!\left[\exists i:\ \mathcal{C}(P_i,\varphi_i,\pi_i)=1\right]$$

the **verification rate**, measured by running the checker under a fixed wall-clock budget $T$ per attempt (SMT timeouts make $\mathrm{VR}$ a function of $T$; papers report at $T \in [10,600]$ s and rarely hold it fixed across baselines), and

$$\mathrm{SA}(d) = \Pr\!\left[\ \varphi \equiv \varphi^\star_d \ \big|\ \mathcal{C}=1\right]$$

the **specification adequacy**: the chance the accepted specification is equivalent to the intended one $\varphi^\star_d$. End-to-end correctness is $\mathrm{VR}@k \cdot \mathrm{SA}$. Only the first factor is reported.

$\mathrm{SA}$ has no direct estimator, because $\varphi^\star_d$ does not exist as an artifact. Proxies: (i) **escape rate** — fraction of verified programs failing a held-out test oracle; (ii) **vacuity rate** — fraction where $\varphi$ is satisfied by a trivial program (`return 0`, or a body whose precondition is unsatisfiable); (iii) human audit agreement.

Cost: $\mathcal{K}(d) = N_{\text{tok}} + \lambda \cdot T_{\text{solver}}$, in generated tokens plus solver seconds. Verified pipelines run 10–100 attempts with repair loops, so $\mathcal{K}$ exceeds unverified generation by one to two orders of magnitude.

Assumptions known to be violated in practice:

- **Checker soundness.** Violated: SMT solvers and verifier encodings have had soundness bugs; `assume`, axioms, and `{:axiom}`/`external_body` escape hatches are reachable by a generator and are used by it.
- **Termination is proved.** Violated in Dafny only if decreases clauses are checked; models emit `assume {:axiom} false`-equivalent patterns that vacuously discharge goals.
- **Specification independence.** Violated: the same model writes $P$ and $\varphi$, so errors correlate. $\mathrm{SA}$ estimated under an independence assumption is biased upward.
- **Benchmark i.i.d.-ness.** Violated: Dafny/Verus/F\* corpora are small and heavily overlap public training data.

## 3. State of the Art

**Established (ablated, reproduced).**
- Constraint-based synthesis with a correctness oracle works and is old: Sketch (Solar-Lezama, 2008) and SyGuS/CEGIS (Alur et al., FMCAD 2013) produce provably-correct programs from formal specs. The guarantee is real; the bottleneck is that the human writes $\varphi$.
- LLMs as *proposal distributions* inside a sound loop are a robust win. Loop-invariant generation by LLM plus a checker (Chakraborty et al., EMNLP Findings 2023; Kamath et al., 2023) beats classical invariant inference on standard C benchmark suites, and the guarantee is unaffected because the checker filters.
- LLM whole-proof generation for Coq (Baldur, First et al., FSE 2023) and retrieval-augmented tactic prediction for Lean (LeanDojo, NeurIPS 2023) established that proof search benefits from language models; both include ablations on retrieval and repair.

**Claimed but weakly ablated.**
- End-to-end "verified code generation" systems — Clover (Sun et al., 2023/24), AutoVerus (Yang et al., 2024), AlphaVerus (Aggarwal, Parno, Welleck, 2024) — report high verification rates on curated suites of small functions. Sample sizes are tens to low hundreds; contamination controls are absent or informal; the specification-adequacy factor is either unmeasured or measured by the authors' own inspection.
- Fine-tuning on a large proof-oriented corpus (Chakraborty et al., *Towards Neural Synthesis for SMT-Assisted Proof-Oriented Programming*, ICSE 2025) shows the in-distribution/out-of-distribution gap explicitly: performance drops sharply on held-out projects. This is the most honest scaling evidence available.

**Benchmark-number-only.** DafnyBench (Loughridge et al., 2024) success rates, MBPP-DFY-50 (Misu et al., FSE 2024) verified-solution counts, and HumanEval-to-Verus ports are single-number results; almost none report solver-time budgets held constant across arms, and none report escape rate.

## 4. What Is Known

- **Undecidability is not the binding constraint.** Rice (1953) rules out a total decision procedure, but synthesis with a proof certificate sidesteps it: the checker only validates. Practical verifiers time out, they do not fail on undecidability in the common case.
- **Verified artifacts at scale are possible with human effort.** seL4 (Klein et al., SOSP 2009): $\sim 10^4$ LOC C kernel, $\sim 2\times10^5$ lines of Isabelle proof, roughly 20 person-years. CompCert (Leroy, POPL 2006): Csmith fuzzing (Yang et al., PLDI 2011) found hundreds of bugs across mainstream compilers and **zero** in CompCert's verified backend. Proof-to-code ratio of 10–20× is the measured human cost.
- **Verification collapses the false-positive rate of generation.** Where an unverified LLM solution passing tests still fails on adversarial inputs at double-digit rates, checker-accepted Dafny/Verus programs satisfy their stated specification by construction. Measured scale: functions of 5–50 LOC.
- **Rates on small curated sets are moderate, not high.** Across MBPP-DFY-50 (50 problems), CloverBench (~60 hand-written Dafny programs), and DafnyBench (782 programs), reported best-system verification rates cluster in the roughly 50–70% band for frontier models with repair loops, and fall well below that on unseen project code. Treat these as reported, not independently reproduced.
- **Repair loops carry most of the gain.** Feeding verifier error messages back to the model is worth more than better single-shot decoding in every system that ablates it (Baldur, AutoVerus, AlphaVerus).

## 5. What Is Not Known

- **Methodologically blocked — specification adequacy.** No agreed estimator for $\mathrm{SA}$, no benchmark reports escape rate or vacuity rate. "Verified" is currently a claim about $\varphi$, presented as a claim about $d$. This is the single largest gap and it is a measurement gap, not a modelling one.
- **Empirically open — scaling.** Whether $\mathrm{VR}@1$ improves with model compute at a rate comparable to unverified `pass@1`, or saturates because proof tokens are scarce. Runnable now: one model family, four sizes, one fixed solver budget. Nobody has published it with the budget held constant.
- **Empirically open — contamination.** How much of reported Dafny/Verus performance survives on specifications written after the training cutoff. No held-out-by-date suite exists.
- **Theoretically open — sample complexity.** No bound on how many verifier interactions are needed to reach a target $\mathrm{VR}$ for a program class, given a model with known per-token accuracy. CEGIS convergence bounds exist for finite hypothesis spaces; nothing analogous covers a neural proposal distribution.
- **Open — TCB under adversarial generation.** Whether a proof-search model trained against a solver systematically discovers unsoundness or vacuity paths. Anecdotes exist; no measurement.

## 6. Why It Is Hard

The specific obstruction is **absent ground truth for the specification, combined with an evaluation that does not measure what it names.**

Every reported number is $\mathrm{VR}$. The quantity users care about is $\mathrm{VR} \cdot \mathrm{SA}$. Estimating $\mathrm{SA}$ requires an oracle for intent, which is exactly what formal specification was introduced to supply — the regress is real, not rhetorical. Worse, the standard pipeline has the same model write $P$ and $\varphi$, so a misreading of $d$ produces a *consistent* pair that the checker accepts. The checker cannot detect this by construction: it verifies an implication, not a translation.

Two secondary obstructions:

- **Cost asymmetry.** SMT time is superlinear in specification complexity and non-monotone in irrelevant edits. A 10× token budget with a 100× solver budget buys perhaps 15 percentage points of $\mathrm{VR}$; the marginal cost per point rises.
- **Data scarcity.** Public Dafny, Verus, F\*, and Lean corpora total $10^5$–$10^6$ definitions, versus $10^{11}$ tokens of ordinary code. Proof data cannot be scraped, and synthetic bootstrapping (AlphaVerus) risks reward-hacking the checker's escape hatches.

## 7. Current Research (as of 2026)

- **Self-improving translation loops.** Translating verified libraries between proof languages and filtering by the checker — AlphaVerus (CMU: Aggarwal, Parno, Welleck). Extending this beyond function-level scope is the active frontier *(frontier — verify)*.
- **Proof-oriented fine-tuning at corpus scale.** Microsoft Research's F\*/PoPAI line (Chakraborty, Lahiri, Swamy) — the main source of honest OOD numbers.
- **Intent formalization as its own task.** `nl2postcond` (Endres, Fakhoury, Chakraborty, Lahiri, FSE 2024) measures whether LLM-generated postconditions catch real bugs — the closest existing operationalization of $\mathrm{SA}$, and the natural base for a proper metric.
- **Verifier-in-the-loop RL.** Using checker acceptance as a verifiable reward signal for code models; widely pursued, with vacuity/reward-hacking as the reported failure mode *(frontier — verify)*.
- **Lower-TCB backends.** Producing kernel-checkable Lean or Coq certificates rather than trusting SMT encodings.

## 8. Concrete Next Experiment

**Question:** what fraction of checker-accepted neural syntheses are semantically wrong relative to intent?

- **Scale:** 300 tasks — 150 from DafnyBench/MBPP-DFY plus 150 written after 2025-06 from a fresh source (e.g., competitive-programming statements with hidden test suites). One frontier model, repair loop capped at 20 verifier round-trips, solver budget fixed at 60 s per attempt across all arms.
- **Arms:**
  - *Treatment:* model writes both $\varphi$ and $P$; keep only checker-accepted triples.
  - *Control A:* human-written (or benchmark-provided) $\varphi$, model writes only $P$ and $\pi$. This isolates the specification-writing error.
  - *Control B:* unverified generation, tests only.
- **Decision number:** the **escape rate** $E$ = fraction of *checker-accepted* treatment outputs that fail the held-out hidden test suite, plus a vacuity check (does a stub body also verify against $\varphi$?).
  - $E < 3\%$ and vacuity $< 1\%$: current "verified" numbers are approximately end-to-end correctness numbers; the field can keep reporting $\mathrm{VR}$.
  - $E > 15\%$: $\mathrm{VR}$ is not a correctness metric, and every headline in Section 3 needs restating as $\mathrm{VR}\cdot\mathrm{SA}$.
- **Prediction (assumed, not measured):** $E$ between 10% and 30% on the post-cutoff half, with the gap to Control A giving the specification-error share directly.

Cost estimate: 300 tasks × 3 arms × 20 round-trips × ~2k tokens ≈ $4\times10^7$ tokens plus ~150 solver-hours. Runnable in days on one machine plus API access.

## 9. Key References

- **[Foundational]** H. G. Rice. *Classes of Recursively Enumerable Sets and Their Decision Problems.* Transactions of the AMS, 1953.
- **[Foundational]** Armando Solar-Lezama. *Program Synthesis by Sketching.* PhD thesis, UC Berkeley, 2008.
- **[Foundational]** Rajeev Alur, Rastislav Bodík, Garvit Juniwal, Milo Martin, Mukund Raghothaman, Sanjit Seshia, Rishabh Singh, Armando Solar-Lezama, Emina Torlak, Abhishek Udupa. *Syntax-Guided Synthesis.* FMCAD, 2013.
- **[Foundational]** Xavier Leroy. *Formal Certification of a Compiler Back-End, or: Programming a Compiler with a Proof Assistant.* POPL, 2006.
- **[Foundational]** Gerwin Klein et al. *seL4: Formal Verification of an OS Kernel.* SOSP, 2009.
- **[Foundational]** Xuejun Yang, Yang Chen, Eric Eide, John Regehr. *Finding and Understanding Bugs in C Compilers.* PLDI, 2011.
- **[SOTA]** Emily First, Markus Rabe, Talia Ringer, Yuriy Brun. *Baldur: Whole-Proof Generation and Repair with Large Language Models.* ESEC/FSE, 2023. — arXiv:2303.04910
- **[SOTA]** Kaiyu Yang et al. *LeanDojo: Theorem Proving with Retrieval-Augmented Language Models.* NeurIPS, 2023. — arXiv:2306.15626
- **[SOTA]** Md Rakib Hossain Misu, Cristina V. Lopes, Iris Ma, James Noble. *Towards AI-Assisted Synthesis of Verified Dafny Methods.* FSE, 2024. — arXiv:2402.00247
- **[SOTA]** Chuyue Sun, Ying Sheng, Oded Padon, Clark Barrett. *Clover: Closed-Loop Verifiable Code Generation.* 2023/2024. — arXiv:2310.17807
- **[SOTA]** Saikat Chakraborty, Gabriel Ebner, Siddharth Bhat, Sarah Fakhoury, Sakina Fatima, Shuvendu Lahiri, Nikhil Swamy. *Towards Neural Synthesis for SMT-Assisted Proof-Oriented Programming.* ICSE, 2025.
- **[SOTA]** Pranjal Aggarwal, Bryan Parno, Sean Welleck. *AlphaVerus: Bootstrapping Formally Verified Code Generation through Self-Improving Translation and Treefinement.* 2024.
- **[SOTA]** Chloe Loughridge et al. *DafnyBench: A Benchmark for Formal Software Verification.* 2024.
- **[Measurement]** Madeline Endres, Sarah Fakhoury, Saikat Chakraborty, Shuvendu Lahiri. *Can Large Language Models Transform Natural Language Intent into Formal Method Postconditions?* FSE, 2024.
- **[Measurement]** Saikat Chakraborty, Shuvendu Lahiri et al. *Ranking LLM-Generated Loop Invariants for Program Verification.* Findings of EMNLP, 2023.
- **[Systems]** Andrea Lattuada et al. *Verus: Verifying Rust Programs Using Linear Ghost Types.* OOPSLA, 2023.
- **[Survey]** Sumit Gulwani, Oleksandr Polozov, Rishabh Singh. *Program Synthesis.* Foundations and Trends in Programming Languages, 2017.

## 10. Worked Example

Task $d$: *"Return the index of the first occurrence of `key` in array `a`, or `-1` if absent."*

A model emits Dafny:

```dafny
method Find(a: array<int>, key: int) returns (i: int)
  ensures 0 <= i < a.Length ==> a[i] == key
  ensures i == -1 ==> forall k :: 0 <= k < a.Length ==> a[k] != key
{
  i := 0;
  while i < a.Length
    invariant 0 <= i <= a.Length
    invariant forall k :: 0 <= k < i ==> a[k] != key
  { if a[i] == key { return i; } i := i + 1; }
  return -1;
}
```

Dafny verifies this in about 1.2 s. $\mathrm{VR}@1 = 1$. Now audit $\varphi$:

- **"First" is never stated.** Nothing in `ensures` requires minimality of $i$. A body that returns the *last* match verifies against the identical specification.
- **The fall-through returns `a.Length`, not `-1`**, in a variant where the loop exits normally without the explicit `return -1` — and the second postcondition is then vacuous, since its guard $i = -1$ is false. Vacuity is invisible to the checker.
- **Stub test.** Replace the body with `i := -1;` and add `assume forall k :: 0 <= k < a.Length ==> a[k] != key;`. It verifies. The specification admits a program that is wrong on every input containing `key`.

Cost accounting for the honest version: adding `ensures 0 <= i ==> forall k :: 0 <= k < i ==> a[k] != key` fixes minimality, costs one extra line, and pushes solver time to ~2 s. Trivial here. But the model did not write it, and no benchmark in Section 3 would have penalized the omission — the program is counted as verified.

This is the obstruction in one function. The checker did its job perfectly. $\mathrm{VR} = 1$, $\mathrm{SA} < 1$, and the product — the only number a user cares about — was never computed.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*