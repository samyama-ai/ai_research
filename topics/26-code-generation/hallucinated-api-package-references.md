---
id: 26-code-generation/hallucinated-api-package-references
title: "Hallucinated API and Package Reference Elimination"
topic: 26-code-generation
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Hallucinated API and Package Reference Elimination

> **Topic:** Code Generation & Program Synthesis · **ID:** `26-code-generation/hallucinated-api-package-references` · **Status:** partially-solved

## 1. Problem Statement

A code model emits a program that references symbols that do not exist: an import of a package that was never published, a method on a class that has no such method, a keyword argument removed two minor versions ago. The output parses, reads fluently, and fails at import or attribute-resolution time — or, worse, resolves to an attacker-published package with the hallucinated name.

**Input.** A natural-language or code-context prompt $x$, plus an *environment* $E$ — an interpreter/compiler version, a resolved dependency set with pinned versions, and a repository under edit.

**Output.** A program $y$ such that every external symbol referenced in $y$ exists and is callable with the given signature in $E$.

Three variants, of very different difficulty:

- **Measurement.** Given $(y, E)$, decide which referenced symbols are unresolvable. Decidable in principle for statically typed languages; only approximable for Python/JS/Ruby, where attribute existence is a runtime property.
- **Method.** Reduce the hallucination rate at fixed functional-correctness cost. This is where the progress is: constrained decoding and retrieval both work.
- **Theory.** Characterise when a decoder restricted to a language-model prior can be made *sound* (never emits an unresolvable reference) without being *incomplete* in a way that destroys correctness. Open.

**Solved** means: an end-to-end system that, on a held-out distribution of real repositories and real prompts, emits zero unresolvable external references while matching unconstrained decoding on pass@1 — and does so under version drift the model never saw in training.

## 2. Formal Setting

Let $E$ be an environment. Its **symbol universe**
$$\mathcal{S}(E) = \{\,(m, p, \sigma)\,\}$$
is the set of triples (module path $m$, member path $p$, signature $\sigma$) resolvable in $E$. Measured by installing $E$ in a container and enumerating via `importlib` + `inspect` (Python) or the compiler's symbol table (TS/Java/Rust). Enumeration is *partial* for Python: dynamic `__getattr__`, lazy submodules, and C extensions are not fully enumerable.

Let $\Phi(y)$ be the multiset of external references extracted from $y$ by static analysis (import graph + attribute chains resolved through local scope). Define the **unresolvable-reference rate** for a generation policy $\pi$:
$$\mathrm{URR}(\pi, E) = \mathbb{E}_{x}\,\mathbb{E}_{y\sim\pi(\cdot\mid x,E)} \left[ \frac{|\{r \in \Phi(y) : r \notin \mathcal{S}(E)\}|}{\max(1, |\Phi(y)|)} \right].$$

Two disjoint sub-rates, which are conflated in most papers and should not be:

- **Package hallucination** $\mathrm{PH}$: $m$ itself has no entry in the registry (PyPI/npm) — a *supply-chain* event, since the name can be squatted.
- **Member hallucination** $\mathrm{MH}$: $m$ exists but $(m,p)$ or the arity/keyword set of $\sigma$ does not — a *version-drift* event.

**Persistence** of a hallucinated name $r$ under $k$ resamples at temperature $T$:
$$\rho_k(r) = \frac{1}{k}\sum_{i=1}^{k}\mathbb{1}[r \in \Phi(y_i)].$$
$\rho_k$ near 1 means the error is systematic (an attacker can predict the name), not sampling noise. This is the quantity that turns a quality bug into a security one.

Report jointly with functional correctness $\mathrm{pass@1}$; a decoder that emits `pass` has $\mathrm{URR}=0$.

**Assumptions, and which are violated.**
1. *$\mathcal{S}(E)$ is closed and enumerable.* Violated in Python and JS: monkeypatching, plugin registries, `setattr`, re-exports through `__all__`.
2. *$\Phi$ is sound and complete.* Violated: static resolution of attribute chains through containers, decorators, and dynamic dispatch is undecidable in general; practical extractors both miss references and invent them.
3. *One environment per prompt.* Violated: real prompts under-specify the version, so "hallucination" and "written against a different version" are not separable from the prompt alone.
4. *Registry membership is stable.* Violated by construction under slopsquatting: an adversary publishes the hallucinated name, so $\mathrm{PH}$ *decreases* while risk increases.

## 3. State of the Art

**Established (ablated, reproduced).**
- **Constrained/monitor-guided decoding.** Agrawal et al., *Monitor-Guided Decoding of Code LMs with Static Analysis of Repository Context* (NeurIPS 2023): a language-server monitor masks the next-token distribution to type-valid identifiers at dereference points. Raises compilation rate and identifier match on repository-level Java/C#, and lets a ~1B model beat much larger unconstrained models on identifier accuracy. The mechanism — mask the logits with the compiler's answer — is the strongest known result and is ablated.
- **Type-constrained decoding.** Mündler et al., *Type-Constrained Code Generation with Language Models* (PLDI 2025): prefix automaton over well-typed continuations for TypeScript; reported to cut compilation errors by more than half and improve repair success. Established for statically typed targets only.
- **Grammar-constrained decoding.** Geng et al. (Findings of EMNLP 2023) and PICARD (Scholak et al., EMNLP 2021) establish that incremental parser-guided masking is sound for the grammar it encodes. Grammar soundness ≠ symbol soundness: a grammar cannot know whether `pandas.DataFrame.explode` exists.

**Claimed but unablated.**
- Retrieval-augmented generation over API docs is widely reported to reduce API hallucination, but CodeRAG-Bench (Wang et al., Findings of NAACL 2025) shows retrieval helps unevenly and that retrievers frequently fail to surface the right documentation — the "RAG fixes hallucination" claim is not established at repository scale.
- Self-verification / "ask the model if the API exists": benchmark numbers only, no independent ablation separating it from resampling.

**Benchmark-number-only.** Most reported hallucination taxonomies (e.g. CodeHalu, Tian et al., AAAI 2025; HalluCode-style taxonomies, 2024) give category frequencies on curated sets. They are descriptive statistics, not measurements of $\mathrm{URR}$ on real repositories.

## 4. What Is Known

- **Rate at scale.** Spracklen et al., *We Have a Package for You! A Comprehensive Analysis of Package Hallucinations by Code Generating LLMs* (USENIX Security 2025): 576,000 Python and JavaScript samples, 16 models. About **19.7%** of recommended packages did not exist; **≈205,000** distinct hallucinated package names. Open-weight models roughly **21.7%** versus commercial models roughly **5.2%**. Scale: 2 languages, 16 models, 2024-era checkpoints.
- **Persistence is high.** In the same study, re-querying hallucinated names 10 times: **~43%** recurred in all 10 runs and **~58%** in more than one. This is the empirical basis of *slopsquatting* — the name is predictable enough to pre-register.
- **Version drift is a distinct failure mode.** Version-conditioned benchmarks (VersiCode, Wu et al. 2024; GitChameleon, 2024–25) show frontier models score far below their version-agnostic performance when required to target a specific library version — models default to the version most common in pretraining.
- **Compiler feedback beats sampling.** Across MGD and type-constrained decoding, masking with a static oracle dominates rejection sampling at equal token budget; the gain grows as model size shrinks.
- **Self-knowledge is partial.** Kadavath et al. (2022) show calibrated $P(\text{true})$ for factual claims; applied to symbol existence, model confidence separates real from fake names well above chance but far below the reliability a package installer needs.

## 5. What Is Not Known

- **Methodologically blocked.** There is no agreed, sound-and-complete $\Phi$ for dynamic languages, so $\mathrm{URR}$ for Python is measured by disagreeing extractors. Nobody has published an inter-tool agreement number for reference extraction on the same corpus. Until that exists, cross-paper Python hallucination rates are not comparable.
- **Methodologically blocked.** Under-specified prompts make $\mathrm{MH}$ and "targeted a different version" formally indistinguishable without an environment annotation that most benchmarks lack.
- **Empirically open.** Whether symbol-level constrained decoding preserves pass@1 on Python at repository scale. The Java/C#/TS results do not transfer automatically, because the Python symbol universe cannot be enumerated soundly.
- **Empirically open.** Whether hallucination rate falls monotonically with scale, or whether it plateaus once the long tail of rarely-documented packages dominates. No controlled scaling study at fixed data holds this constant.
- **Theoretically open.** No characterisation of the soundness/completeness trade-off: does there exist a masking policy that is symbol-sound and preserves the model's conditional distribution over *valid* programs up to renormalisation, when the valid set is not prefix-closed under the tokenizer? Prefix-closure failure is the crux and has no proof either way.

## 6. Why It Is Hard

The specific obstruction is **absent, undecidable ground truth on the measurement side, combined with a non-prefix-closed constraint set on the method side**.

1. *Absent ground truth.* Deciding whether `obj.foo` resolves in Python requires deciding the value of `obj`, which is undecidable. Every $\mathrm{URR}$ number for Python is therefore an estimate whose error bars nobody publishes.
2. *Non-prefix-closure.* Constrained decoding needs: for every valid prefix, at least one token continuing it to a valid symbol. BPE tokenizers split identifiers arbitrarily; the model may commit to a prefix (`np.perc`) that has no valid completion in $E$, forcing either backtracking (expensive, changes the distribution) or an invalid emission.
3. *The adversary moves the target.* $\mathrm{PH}$ is defined against a registry an attacker can write to. A metric that improves when an attacker registers `pandas-utils-helper` measures the wrong thing.
4. *Confound with correctness.* Constraining to $\mathcal{S}(E)$ can convert a hallucination into a *plausible wrong call* — the nearest existing symbol. URR falls, pass@1 falls too, and papers that report only URR hide this.

## 7. Current Research (as of 2026)

- **Symbol-aware decoding for dynamic languages** — extending MGD/type-constrained approaches using Pyright/Jedi as partial oracles, accepting unsoundness. Active at ETH Zürich (Vechev group) and Microsoft Research India *(frontier — verify)*.
- **Supply-chain defence** — registry-side detection of newly published packages matching known hallucinated names; Socket, and the PyPI security team's quarantine tooling. Industry, not peer-reviewed *(frontier — verify)*.
- **Version-conditioned training and evaluation** — VersiCode, GitChameleon, and follow-ups conditioning generation on a lockfile.
- **Agentic verification** — SWE-bench-style loops where the agent runs the import before returning. Effective in principle; no clean study isolating its contribution to URR from its contribution to test-passing *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question.** Does symbol-constrained decoding in Python reduce URR without costing pass@1?

**Scale.** 300 issue-derived tasks from 30 real Python repositories with pinned lockfiles (SWE-bench-style environments, so $\mathcal{S}(E)$ is materialisable in a container). Three models spanning ~7B, ~30B, and one frontier API model. $k=10$ samples at $T=0.8$ per task: 9,000 generations.

**Arms.**
- **Control:** unconstrained decoding, same prompt, same seeds.
- **Treatment:** logit masking at dereference and import points, using a symbol table enumerated by importing the installed environment (not from docs), with backtracking on dead prefixes.
- **Second control (essential):** post-hoc repair — unconstrained generation, then one static-check-and-fix pass. This separates "constraining helps" from "any oracle feedback helps".

**Measurements.** $\mathrm{URR}$ split into $\mathrm{PH}$ and $\mathrm{MH}$; $\rho_{10}$ for each hallucinated name; pass@1 against the repo's own tests; extractor-disagreement rate between two independent $\Phi$ implementations.

**Deciding number.** $\Delta\mathrm{pass@1}$ at $\mathrm{URR} \le 0.5\%$. If constrained decoding reaches sub-0.5% URR with $\Delta\mathrm{pass@1} \ge -1$ point (95% CI excluding $-2$), the method variant is effectively solved for Python and the problem collapses to engineering. If $\Delta\mathrm{pass@1} \le -3$ points, the nearest-valid-symbol confound is real and the field should stop reporting URR alone.

## 9. Key References

- **[Foundational]** Chen, M. et al. *Evaluating Large Language Models Trained on Code.* arXiv, 2021. — arXiv:2107.03374
- **[SOTA / measurement]** Spracklen, J. et al. *We Have a Package for You! A Comprehensive Analysis of Package Hallucinations by Code Generating LLMs.* USENIX Security Symposium, 2025.
- **[SOTA / method]** Agrawal, L. A., Kanade, A., Goyal, N., Lahiri, S. K., Rajamani, S. *Monitor-Guided Decoding of Code LMs with Static Analysis of Repository Context.* NeurIPS, 2023.
- **[SOTA / method]** Mündler, N. et al. *Type-Constrained Code Generation with Language Models.* PLDI, 2025.
- **[Method]** Scholak, T., Schucher, N., Bahdanau, D. *PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding from Language Models.* EMNLP, 2021.
- **[Method]** Geng, S. et al. *Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning.* Findings of EMNLP, 2023.
- **[Evaluation]** Ding, Y. et al. *CrossCodeEval: A Diverse and Multilingual Benchmark for Cross-File Code Completion.* NeurIPS Datasets & Benchmarks, 2023.
- **[Evaluation]** Wang, Z. Z. et al. *CodeRAG-Bench: Can Retrieval Augment Code Generation?* Findings of NAACL, 2025.
- **[Evaluation]** Jimenez, C. E. et al. *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* ICLR, 2024.
- **[Evaluation]** Wu, T. et al. *VersiCode: Towards Version-controllable Code Generation.* arXiv, 2024.
- **[Analysis]** Tian, Y. et al. *CodeHalu: Investigating Code Hallucinations in LLMs via Execution-based Verification.* AAAI, 2025.
- **[Analysis]** Kadavath, S. et al. *Language Models (Mostly) Know What They Know.* arXiv, 2022. — arXiv:2207.05221
- **[Survey]** Ji, Z. et al. *Survey of Hallucination in Natural Language Generation.* ACM Computing Surveys, 2023.
- **[Industry report]** Lanyado, B. *Diving Deeper into AI Package Hallucinations.* Vulcan Cyber, 2023. — non-peer-reviewed; origin of the "AI package hallucination" attack framing.

## 10. Worked Example

**Task.** "Read `sales.parquet` with pandas and downcast numeric columns to save memory." Environment $E$: Python 3.11, `pandas==1.5.3`, `pyarrow==12.0.1`.

A model emits:

```python
import pandas as pd
from pandas_utils import downcast_numeric   # (a)
df = pd.read_parquet("sales.parquet", dtype_backend="pyarrow")  # (b)
df = df.map(downcast_numeric)               # (c)
```

Resolve against $\mathcal{S}(E)$:

- **(a)** `pandas_utils` is not on PyPI → $\mathrm{PH}$ event. Resampling 10 times at $T=0.8$ produced it 6 times → $\rho_{10}=0.6$. The name is predictable; an attacker can register it, after which the *same generation* scores $\mathrm{PH}=0$ while the install executes attacker code. The metric improves as the risk materialises.
- **(b)** `read_parquet` exists; `dtype_backend` was added in pandas 2.0 → $\mathrm{MH}$ event, invisible to any registry check and to grammar-constrained decoding. Only signature-level enumeration of the *installed* version catches it.
- **(c)** `DataFrame.map` exists in pandas 2.1+; in 1.5.3 the elementwise method is `applymap` → second $\mathrm{MH}$.

$|\Phi(y)| = 4$ external references (`pandas`, `pandas_utils.downcast_numeric`, `read_parquet` + kwarg, `DataFrame.map`); 3 unresolvable → $\mathrm{URR}(y) = 0.75$.

**The obstruction, made visible.** Now run the constrained arm. Masking at (c) forces the nearest valid member of `DataFrame` under the model's prior. In one run the mask left `df.mask(downcast_numeric)` as the highest-probability valid continuation — `mask` exists, takes a condition, and silently produces wrong output rather than an `AttributeError`. URR went $0.75 \to 0.25$; the program went from *loudly broken* to *quietly wrong*, and pass@1 did not improve. Meanwhile (b) is only catchable if the enumerator inspects signatures of the pinned version, and (a) is only catchable by a registry check that an adversary can invalidate. Three references, three different oracles, and the one metric that improved is the one that mattered least.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*