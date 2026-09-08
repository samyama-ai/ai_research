---
id: 26-code-generation/autoformalization-specifications-contracts
title: "Autoformalization of Informal Specifications into Checkable Contracts"
topic: 26-code-generation
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Autoformalization of Informal Specifications into Checkable Contracts

> **Topic:** Code Generation & Program Synthesis · **ID:** `26-code-generation/autoformalization-specifications-contracts` · **Status:** open

## 1. Problem Statement

**Input:** an informal specification $d$ — a docstring, issue text, API comment, or requirements paragraph — together with a signature $\sigma$ (types of inputs and outputs) and optionally a program $P$.

**Output:** a machine-checkable contract $\varphi = (\mathrm{pre}, \mathrm{post}, \mathrm{inv})$ in a formal language with a decision procedure or proof assistant behind it: Dafny `requires`/`ensures`, JML, ACSL, Lean/Isabelle propositions, LTL for reactive systems, or refinement types.

**Objective:** $\varphi$ must be *sound* (every behaviour the author intended satisfies $\varphi$ — no correct program is rejected) and *complete* (every behaviour the author did not intend violates $\varphi$ — no incorrect program is accepted). Solving the problem means producing a system that, given $d$, emits $\varphi$ with a *calibrated* claim about which of these two properties holds and how strongly.

Three variants, with different difficulty:

- **Measurement variant.** Define a metric for "$\varphi$ faithfully captures $d$" that does not silently reduce to "$\varphi$ agrees with one reference implementation". This is the blocked variant.
- **Method variant.** Given a fixed metric, raise the fraction of specs that are sound *and* discriminating. Empirically open.
- **Theory variant.** Characterise when $d$ (a finite natural-language string) determines $\varphi$ up to logical equivalence. Natural language underdetermines edge cases, so the honest theory question is about the size of the residual equivalence class, not about exact recovery.

## 2. Formal Setting

Let $\Sigma$ be the input space induced by $\sigma$ and $B$ the set of total behaviours $b : \Sigma \rightharpoonup O$. A specification is a predicate on behaviours; write $\llbracket \varphi \rrbracket \subseteq B$ for the set of behaviours satisfying $\varphi$. Let $I(d) \subseteq B$ be the *intent set*: behaviours a competent human author of $d$ would accept. $I(d)$ is not a singleton — this is a modelling choice, not sloppiness.

- **Soundness (measured):** $\mathrm{Snd}(\varphi) = \mathbb{1}[I(d) \subseteq \llbracket\varphi\rrbracket]$. Operationally, run the reference implementation $P^\star$ (and any accepted alternates) on a test suite $T$ and check the runtime-checked contract never fires:
$$\widehat{\mathrm{Snd}}(\varphi) = \mathbb{1}\big[\forall x \in T:\ \mathrm{pre}(x) \Rightarrow \mathrm{post}(x, P^\star(x))\big].$$
- **Completeness / discriminating power (measured):** given a mutant set $M = \{P_1,\dots,P_m\}$ drawn from a corruption distribution $\mathcal{D}_{\text{bug}}$,
$$\widehat{\mathrm{Cmp}}(\varphi) = \frac{1}{m}\sum_{i=1}^{m} \mathbb{1}\big[\exists x \in T:\ \mathrm{pre}(x) \wedge \neg\,\mathrm{post}(x, P_i(x))\big].$$
This is *mutation kill rate under $\mathcal{D}_{\text{bug}}$*, not completeness. The gap between the two is the core measurement problem.
- **Checkability (measured):** $\mathrm{Chk}(\varphi) = \mathbb{1}[\text{verifier terminates with a verdict within budget } \tau]$ — for Dafny/Boogie, Z3 does not time out at $\tau$ seconds; for Lean, `#check` elaborates.
- **Non-triviality:** $\varphi \not\equiv \top$ and $\mathrm{pre} \not\equiv \bot$. A vacuous precondition makes $\widehat{\mathrm{Snd}} = 1$ and $\widehat{\mathrm{Cmp}} = 0$; both must be reported jointly.
- **Utility score:** $U(\varphi) = \widehat{\mathrm{Snd}}(\varphi)\cdot\widehat{\mathrm{Cmp}}(\varphi)\cdot\mathrm{Chk}(\varphi)$, with per-spec token/solver cost reported alongside.

**Assumptions, and which are violated:**

1. *$I(d)$ is well defined.* Violated — human annotators disagree on edge cases (empty input, overflow, aliasing) at rates high enough to dominate the metric.
2. *$P^\star$ realises intent.* Violated — reference implementations in HumanEval/MBPP contain behaviours nobody intended; measuring soundness against $P^\star$ launders implementation accidents into specification.
3. *$\mathcal{D}_{\text{bug}}$ resembles real bugs.* Violated — syntactic mutants over-represent off-by-one and operator swaps and under-represent missing-case and concurrency faults.
4. *Verification failure means the spec is wrong.* Violated — it often means the prover lacked a lemma or loop invariant, confounding spec quality with proof automation.

## 3. State of the Art

**Established.**
- Few-shot autoformalization of mathematics works at a low but non-zero rate: Wu et al. (NeurIPS 2022) report roughly a quarter of Codex formalizations of competition problems into Isabelle judged correct by human inspection.
- Postcondition generation from docstrings is *usually sound and rarely discriminating*. Endres et al. (FSE 2024, `nl2postcond`) show that on HumanEval/EvalPlus, GPT-4 postconditions are overwhelmingly consistent with the reference implementation, while only a minority are strong enough to reject subtly buggy variants; a small number nonetheless caught real Defects4J bugs.
- Closed-loop consistency checking is a real filter, not just a heuristic: Clover (Sun, Sheng, Padon, Barrett; SAIV 2024) triangulates code, docstring, and annotation, and on its 60-example Dafny benchmark accepts a large majority of correct instances while rejecting essentially all incorrect ones.
- End-to-end verified-code synthesis is far below code synthesis. Misu et al. (FSE 2024) report GPT-4 with retrieval-augmented few-shot prompting produces fully verified Dafny for roughly 58% of the 50-task MBPP-DFY-50 set — against >90% pass@1 on plain-Python MBPP for the same model class.

**Claimed but unablated.**
- Agentic repair loops ("re-prompt on verifier error until it verifies") report large gains, but almost never ablate whether the gain comes from better specifications or from the model weakening `ensures` clauses until they verify. Non-triviality is rarely reported.
- Specification-synthesis pipelines combining static analysis with LLMs — AutoSpec (Wen et al., CAV 2024) and SpecGen (Ma et al., 2024) — report high verification rates on SV-COMP-style and Java benchmarks. These are benchmark numbers; the counterfactual "would a trivial-but-provable spec score the same?" is generally not run.
- LTL translation (nl2spec, Cosler et al., CAV 2023) reports good accuracy on curated sentence sets. Accuracy is against a single gold formula, so semantically equivalent-but-different outputs are miscounted in both directions.

## 4. What Is Known

- **Soundness/strength asymmetry is robust.** Across Dafny, JML, and Python-postcondition settings, generated specs pass reference-implementation checks far more often than they kill mutants. Gaps of 30–60 percentage points between the two are typical at HumanEval scale (164 tasks).
- **Human "looks equivalent" judgements are unreliable.** Lahiri (2024) applies symbolic testing to LLM-generated Dafny specs and finds a substantial fraction of specs that pass human review are not equivalent to the intended one — the failure mode is silent weakening, not visible error.
- **Type-checking is not correctness.** ProofNet (Azerbayev et al., 2023) separates elaboration success from semantic correctness on 371 undergraduate problems; a large share of statements that compile are wrong. The same split holds for contracts that parse and verify.
- **Proof automation is the bottleneck in verified synthesis, not just spec writing.** DafnyBench (Loughridge et al., 2024; 782 programs) shows top models reconstruct verification hints for roughly two-thirds of programs; failures concentrate on loop invariants.
- **Invariant generation is helped by ranking.** Chakraborty et al. (Findings of EMNLP 2023) show reranking LLM-proposed loop invariants materially cuts calls to the verifier.

## 5. What Is Not Known

- **Methodologically blocked.** There is no accepted measure of "faithful to intent" independent of a reference implementation or a mutant distribution. $\widehat{\mathrm{Cmp}}$ is a function of $\mathcal{D}_{\text{bug}}$, and no community-standard $\mathcal{D}_{\text{bug}}$ exists. Two papers reporting "completeness" are usually not comparable.
- **Empirically open.** Whether scale, verifier-in-the-loop RL, or long-horizon reasoning raises *discriminating* spec rate — as opposed to verifiable-spec rate — has not been measured with non-triviality controls at repository scale (10k+ real functions with real bug histories).
- **Empirically open.** Whether autoformalized contracts reduce escaped defects in a real development workflow. No controlled deployment study exists.
- **Theoretically open.** No characterisation of the residual ambiguity class: given $d$ of length $n$ tokens, how large is $\{\varphi : \varphi \text{ consistent with } d\}$ modulo equivalence, and does interactive questioning shrink it at a provable rate? Also open: any sample-complexity result for learning specs from natural language plus finite behavioural probes.

## 6. Why It Is Hard

The obstruction is **absent ground truth combined with an optimisable degenerate solution**. $I(d)$ is unobservable, so every benchmark substitutes $P^\star$. But $\varphi$ derived from $P^\star$ scores perfectly on soundness while carrying zero independent information — the spec that says "output equals what the reference returns" is maximally sound, fully verifiable, and useless. The gradient of every automated metric therefore points toward weaker specs, and the only known counterweight (mutation kill rate) is defined relative to an arbitrary bug distribution.

Second obstruction: **confounded measurement in the verifier**. A failed Dafny proof does not distinguish "spec is wrong", "spec is right but too strong for this implementation", and "Z3 needed a lemma". Papers that report verification success rate are measuring a product of three variables and attributing it to one.

## 7. Current Research (as of 2026)

- **Verifier-in-the-loop training.** RL against Dafny/Verus/Lean signal, at Microsoft Research (Lahiri, Chakraborty and collaborators), Stanford/Amazon on Verus-style Rust verification, and DeepSeek/Kimi-scale Lean provers. Reward hacking via spec weakening is the acknowledged risk *(frontier — verify)*.
- **Triangulation and consistency.** Clover-style three-way checks (Barrett group, Stanford) extended beyond Dafny to Lean and Rust *(frontier — verify)*.
- **Benchmarks with adversarial mutants.** Successors to DafnyBench and `nl2postcond` that score specs on bug detection rather than verification success (Harvard/MIT, UC Irvine, Northeastern) *(frontier — verify)*.
- **Interactive disambiguation.** Systems that ask the author a bounded number of clarifying questions before emitting the contract; almost no quantitative results yet.
- **Industrial deployment.** AWS (Dafny, Kani), Meta (Infer-adjacent), and Galois-style assurance work on ACSL/Frama-C pipelines *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question:** does any current method produce contracts that are simultaneously sound and discriminating, or does apparent progress come entirely from spec weakening?

**Scale.** 500 functions sampled from real Python/Java repositories that have a linked bug-fix commit (mined from Defects4J, BugsInPy, and GitHub fix commits). For each: docstring $d$, pre-fix version $P^-$, post-fix version $P^+$, and the fix's regression test. Generate contracts from $d$ alone (the fixed code is never shown). Add 20 syntactic mutants per function from a published mutation operator set, giving $500 \times 21 = 10{,}500$ negative programs.

**Arms.**
1. Frontier LLM, single-shot, docstring only.
2. Same model with verifier-in-the-loop repair (≤10 rounds).
3. **Control A (degenerate):** contract that asserts equality with $P^+$'s output. Upper bound on soundness, zero information.
4. **Control B (trivial):** `ensures true`. Lower bound.
5. **Control C (human):** contracts written by two experienced engineers from $d$ alone, inter-annotator agreement reported.

**Deciding number.** Real-bug kill rate at fixed soundness: the fraction of the 500 functions where the contract (a) never fires on $P^+$ over the test suite and (b) *does* fire on $P^-$. Arms 3 and 4 score near 0 by construction. If the best LLM arm lands below the human arm's rate minus 15 points, the field's verification-success numbers are measuring proof automation, not autoformalization. Report mutation kill rate as a secondary number and $\Delta$ between arms 1 and 2 to isolate whether repair loops weaken specs — if arm 2's soundness rises while its real-bug kill rate falls, spec weakening is confirmed, quantitatively, for the first time.

Cost estimate: ~2M output tokens plus ~10 CPU-hours of test execution. Runnable in a week.

## 9. Key References

- **[Foundational]** Yuhuai Wu, Albert Q. Jiang, Wenda Li, Markus N. Rabe, Charles Staats, Mateja Jamnik, Christian Szegedy. *Autoformalization with Large Language Models.* NeurIPS, 2022. — arXiv:2205.12615
- **[Foundational]** Zhangir Azerbayev, Bartosz Piotrowski, Hailey Schoelkopf, Edward W. Ayers, Dragomir Radev, Jeremy Avigad. *ProofNet: Autoformalizing and Formally Proving Undergraduate-Level Mathematics.* 2023. — arXiv:2302.12433
- **[SOTA]** Madeline Endres, Sarah Fakhoury, Saikat Chakraborty, Shuvendu K. Lahiri. *Can Large Language Models Transform Natural Language Intent into Formal Method Postconditions?* FSE, 2024. — arXiv:2310.01831
- **[SOTA]** Chuyue Sun, Ying Sheng, Oded Padon, Clark Barrett. *Clover: Closed-Loop Verifiable Code Generation.* SAIV, 2024. — arXiv:2310.17807
- **[SOTA]** Md Rakib Hossain Misu, Cristina V. Lopes, Iris Ma, James Noble. *Towards AI-Assisted Synthesis of Verified Dafny Methods.* FSE, 2024.
- **[Benchmark]** Chloe Loughridge, Qinyi Sun, Seth Ahrenbach, Federico Cassano, Chuyue Sun, Ying Sheng, Anish Mudide, Md Rakib Hossain Misu, Nada Amin, Max Tegmark. *DafnyBench: A Benchmark for Formal Software Verification.* 2024. — arXiv:2406.08467
- **[Measurement]** Shuvendu K. Lahiri. *Evaluating LLM-driven User-Intent Formalization for Verification-Aware Languages.* 2024. — arXiv:2406.09757
- **[Method]** Cheng Wen, Jialun Cao, Jie Su, Zhiwu Xu, Shengchao Qin, Mengda He, Haokun Li, Shing-Chi Cheung, Cong Tian. *Enchanting Program Specification Synthesis by Large Language Models Using Static Analysis and Program Verification.* CAV, 2024.
- **[Method]** Saikat Chakraborty, Shuvendu K. Lahiri, Sarah Fakhoury, Madanlal Musuvathi, Akash Lal, Aseem Rastogi, Aditya Senthilnathan, Rahul Sharma, Nikhil Swamy. *Ranking LLM-Generated Loop Invariants for Program Verification.* Findings of EMNLP, 2023.
- **[Method]** Matthias Cosler, Christopher Hahn, Daniel Mendoza, Frederik Schmitt, Caroline Trippel. *nl2spec: Interactively Translating Unstructured Natural Language to Temporal Logics with Large Language Models.* CAV, 2023. — arXiv:2303.04864
- **[Survey]** Xinyi Hou, Yanjie Zhao, Yue Liu, Zhou Yang, Kailong Wang, Li Li, Xiapu Luo, David Lo, John Grundy, Haoyu Wang. *Large Language Models for Software Engineering: A Systematic Literature Review.* ACM TOSEM, 2024.

## 10. Worked Example

**Task.** Docstring: *"Return the list with duplicates removed, keeping the first occurrence of each element."* Signature: `dedup(xs: List[int]) -> List[int]`.

**Generated contract** (typical frontier-model output, Dafny-style, transcribed):

```
requires true
ensures forall i, j :: 0 <= i < j < |r| ==> r[i] != r[j]      // no duplicates
ensures forall x :: x in r <==> x in xs                        // same element set
```

**Soundness check.** Run against the reference implementation on 200 random lists of length ≤ 20: the contract never fires. $\widehat{\mathrm{Snd}} = 1$. A human reviewer reads it and marks it "correct" — it says exactly what the docstring says, twice over.

**Discriminating power.** Build 12 mutants:

| Mutant | Behaviour | Caught? |
|---|---|---|
| M1 | returns `sorted(set(xs))` | no |
| M2 | keeps *last* occurrence | no |
| M3 | reverses the output | no |
| M4 | returns `list(set(xs))` (arbitrary order) | no |
| M5 | drops the first element | yes (set clause) |
| M6 | appends `0` | yes |
| M7 | returns `[]` on non-empty input | yes |
| M8 | duplicates the last element | yes |
| M9 | off-by-one in the scan, skips index 0 | yes |
| M10 | returns `xs` unchanged | no (only when `xs` already unique) |
| M11 | drops elements appearing ≥3 times | yes |
| M12 | swaps two adjacent distinct outputs | no |

$\widehat{\mathrm{Cmp}} = 6/12 = 0.50$. $U(\varphi) = 1 \times 0.50 \times 1 = 0.50$.

**The obstruction, made visible.** M1–M4 and M12 are exactly the bugs the phrase *"keeping the first occurrence"* was written to exclude — order preservation. The contract omits the ordering conjunct
$$\forall i, j.\ 0 \le i < j < |r| \Rightarrow \mathrm{idx}_{xs}(r[i]) < \mathrm{idx}_{xs}(r[j])$$
and no automated check that uses only the reference implementation can notice, because the reference satisfies both the weak and the strong contract. Soundness saturates at 1.0; the human reviewer signs off; the verifier is happy. The only signal that the spec is half-strength comes from $\mathcal{D}_{\text{bug}}$ — and if the mutant set had contained only M5–M9, the same contract would have scored $\widehat{\mathrm{Cmp}} = 1.0$. The metric is a function of the adversary, and nobody has fixed the adversary.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*