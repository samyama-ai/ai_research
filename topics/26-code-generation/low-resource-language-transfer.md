---
id: 26-code-generation/low-resource-language-transfer
title: "Low-Resource Programming Language Transfer"
topic: 26-code-generation
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Low-Resource Programming Language Transfer

> **Topic:** Code Generation & Program Synthesis · **ID:** `26-code-generation/low-resource-language-transfer` · **Status:** empirically-open

## 1. Problem Statement

A code LLM trained mostly on Python, JavaScript, Java, C++ writes those languages far better than it writes Racket, OCaml, Lua, R, Julia, Fortran, COBOL, Verilog, or a proprietary in-house DSL. The question is whether that gap closes by *transfer* — reusing algorithmic and reasoning competence already in the model — or only by *more data in the target language*.

Three variants, different difficulty:

- **Measurement.** Given a model $M$ and languages $\ell_{\text{hi}}, \ell_{\text{lo}}$, is the observed gap $\Delta = \text{pass@1}(M, \ell_{\text{hi}}) - \text{pass@1}(M, \ell_{\text{lo}})$ a property of the model or of the benchmark? Benchmarks in low-resource languages are usually machine-translated from Python and carry translation artifacts. Currently the weakest link.
- **Method.** Find a procedure — data synthesis, curriculum, upsampling, adapters, retrieval of language documentation — that raises $\text{pass@1}(M,\ell_{\text{lo}})$ by more than spending the same compute and tokens on generic pretraining would.
- **Theory.** Is there a transfer coefficient? Does $\text{pass@1}(\ell_{\text{lo}})$ depend on $N_{\ell_{\text{lo}}}$ (target-language tokens) through a scaling law whose exponent or offset is set by the *rest* of the mixture, and does that dependence saturate?

**Solved** would mean: a stated recipe that takes any language with $\le 10^7$ tokens of public code and reaches within 10 points of pass@1 of a high-resource language of comparable expressiveness, on a benchmark validated as not favouring the source language, reproduced by an independent group.

## 2. Formal Setting

Let $\mathcal{L}$ be a set of languages. Training mixture $\mathcal{D} = \{(\ell, N_\ell)\}$, where $N_\ell$ is **tokens after the model's own tokenizer**, not bytes and not lines — tokenizer fertility (tokens per byte) varies 1.5–2.5× across languages, so a byte-matched mixture is not a token-matched one. Total budget $N = \sum_\ell N_\ell$, mixture weight $w_\ell = N_\ell/N$, model size $P$, compute $C \approx 6PN$.

Evaluation: benchmark $B_\ell = \{(p_i, T_i)\}_{i=1}^n$ of prompts and test suites. With $s$ samples at temperature $\tau$ and $c_i$ passing,

$$\text{pass@}k(\ell) = \frac{1}{n}\sum_{i=1}^{n}\left[1 - \binom{s-c_i}{k}\big/\binom{s}{k}\right]$$

(the unbiased estimator of Chen et al., 2021). Measured with $s=200$, $\tau=0.2$ for $k=1$ in the MultiPL-E protocol.

**Transfer gain.** Fix $P$, $N$, and $N_{\ell_{\text{lo}}}$. Let $M_{\text{solo}}$ be trained only on $\ell_{\text{lo}}$ plus natural language, $M_{\text{mix}}$ on the full mixture. Then

$$G(\ell_{\text{lo}}) = \text{pass@1}(M_{\text{mix}}, \ell_{\text{lo}}) - \text{pass@1}(M_{\text{solo}}, \ell_{\text{lo}}).$$

$G>0$ is the claim that other languages help. Almost nobody measures $G$; papers report $\text{pass@1}(M_{\text{mix}}, \ell_{\text{lo}})$ alone.

**Data-equivalence.** Fit a per-language curve $\text{pass@1}(\ell) = a_\ell - b_\ell N_\ell^{-\alpha_\ell}$ and define the **effective multiplier** $\rho_\ell = \tilde N_\ell / N_\ell$, where $\tilde N_\ell$ is the solo-training token count reaching the same pass@1. $\rho_\ell = 8$ means the mixture bought an 8× data discount. This is the quantity the field lacks.

Assumptions, and where they break:

1. *Benchmark parallelism.* $B_{\ell_{\text{lo}}}$ is assumed a faithful translation of $B_{\ell_{\text{hi}}}$. Violated: MultiPL-E translates HumanEval prompts and tests by compiler-based rewriting; idiomatic solutions in Racket or R differ in signature and return type, so the prompt encodes Python's shape.
2. *Test-suite equal strength.* Assumed identical semantics; violated by float tolerance, integer width, and collection-ordering differences.
3. *Contamination independence.* HumanEval derivatives are widely on the web; low-resource splits leak too, unevenly.
4. *$N_\ell$ counts distinct content.* Violated: near-duplicate and vendored code inflates $N_\ell$ for small ecosystems more than for large ones.
5. *Single-file function completion is the task.* Real low-resource work is API-heavy and repo-scoped.

## 3. State of the Art

**Empirical / systems SOTA.**

- **MultiPL-E** (Cassano, Gouwar, Nguyen, Guha et al., *IEEE TSE* 2023) — HumanEval/MBPP translated by compiler into 18+ languages. It is the de facto measurement instrument. *Established:* the harness runs and the rankings are stable. *Unablated:* whether score differences across languages measure model competence or translation fidelity.
- **MultiPL-T** (Cassano et al., *OOPSLA* 2024, "Knowledge Transfer from High-Resource to Low-Resource Programming Languages for Code LLMs") — generate Python data, translate to Racket/OCaml/Lua/R/Julia, keep only examples whose translated unit tests pass, fine-tune. Reported roughly doubled pass@1 on those languages for StarCoderBase-scale models. *Established:* test-validated translation beats untested translation. *Claimed but unablated:* that the gain is transfer rather than domain-matching the benchmark — the synthetic data descends from the same Python distribution HumanEval came from.
- **BabelCode / "Measuring the Impact of Programming Language Distribution"** (Orlanski, Xiao, Garcia et al., *ICML* 2023) — upsampling low-resource languages helps them at modest cost to high-resource ones. Closest thing to a controlled mixture ablation, at ~<1B–2B parameter scale.
- **StarCoder 2 / The Stack v2** (Lozhkov, Li, Allal, et al., 2024) — 600+ languages, and it folds MultiPL-T data in. Benchmark numbers only; no per-language counterfactual.
- **HumanEval-X / CodeGeeX** (Zheng et al., *KDD* 2023) — hand-checked multilingual set, 5 languages, all high-resource.

**Theory SOTA.** Essentially absent for cross-language transfer specifically. The nearest transferable results are Chinchilla-style compute-optimal scaling (Hoffmann et al., 2022) and data-constrained repetition (Muennighoff et al., *NeurIPS* 2023: up to ~4 epochs nearly as good as fresh data, decaying after). Neither is fit per-language.

## 4. What Is Known

- **The mixture is extreme.** In The Stack, the top handful of languages hold the large majority of permissively licensed code; languages like Racket or OCaml sit 3–4 orders of magnitude below Python by volume. Measured at corpus scale (~1–3 TB, 300+ languages).
- **Score gaps are large and consistent.** On MultiPL-E, open models of the 7B–34B class typically score 30–50 pass@1 on Python and 5–20 on Racket, OCaml, R, D. Measured at $s=200$, $\tau=0.2$.
- **Test-filtered synthetic transfer works.** MultiPL-T's reported roughly 2× pass@1 improvements on five low-resource languages, at 1B–15B parameters, is the strongest positive result.
- **Upsampling is not free but is cheap.** BabelCode/ICML 2023: raising low-resource weight improves those languages with small high-resource loss, at sub-2B scale.
- **Repetition is tolerable.** ~4 epochs of repeated data ≈ fresh data (Muennighoff et al., 2023) — which is why a $10^7$-token language is not hopeless.
- **Syntax is learned before semantics.** Low-resource outputs fail more on API/type errors than on algorithm choice; reported as error taxonomies in MultiPL-E-derived analyses, not as an independently reproduced law.

## 5. What Is Not Known

- **Empirically open.** The value of $G(\ell_{\text{lo}})$ and $\rho_\ell$. Nobody has trained matched $M_{\text{solo}}$ and $M_{\text{mix}}$ at equal $P$, $N$, $N_{\ell_{\text{lo}}}$ and compared. The experiment is runnable today for under $10^5$ GPU-hours at 1–3B scale. This is the central gap.
- **Empirically open.** Whether the MultiPL-T gain survives on a benchmark with no Python ancestry (e.g. Racket problems written by Racket programmers).
- **Empirically open.** Whether transfer is stronger between paradigm-siblings (OCaml←Haskell) than between high-volume distant languages (OCaml←Python). Plausible, untested with controls.
- **Methodologically blocked.** "Low-resource" has no agreed unit. Bytes, deduplicated bytes, tokens, distinct repositories, and distinct API surface give different orderings. Without a unit, no scaling law can be fit.
- **Methodologically blocked.** Separating *language competence* from *library knowledge*. A model failing R is often failing `dplyr`, not R.
- **Theoretically open.** Whether a per-language scaling law with a mixture-dependent offset exists at all, or whether transfer is discontinuous in mixture composition.

## 6. Why It Is Hard

The obstruction is **confounded measurement compounded by an absent control arm**.

Every headline number is $\text{pass@1}(M_{\text{mix}}, \ell_{\text{lo}})$ on a benchmark that was translated out of Python. The quantity of interest, $G$, is a difference between two training runs; the quantity reported is a single number from one training run on an instrument built from the source language. Improving $\text{pass@1}(\ell_{\text{lo}})$ by making training data look more like translated-Python raises the score whether or not any transfer happened. That is not a subtle bias — it is the same mechanism the method exploits.

Second, the control arm is expensive in exactly the regime that matters. $M_{\text{solo}}$ must be trained from scratch, so the comparison costs two pretraining runs per language and cannot be done by fine-tuning an existing checkpoint — the checkpoint already contains the mixture.

## 7. Current Research (as of 2026)

- **Test-validated data translation** as the standard recipe; MultiPL-T-style pipelines are now folded into open pretraining corpora (Northeastern/Roblox/Wellesley collaboration around Cassano, Guha and the BigCode community).
- **Compiler- and type-checker-in-the-loop generation** — rejection sampling against the target-language toolchain, which is stronger for statically typed low-resource languages (OCaml, F#, Ada) than for dynamically typed ones.
- **Repo- and API-grounded evaluation**, moving off translated HumanEval toward execution benchmarks built natively per language *(frontier — verify)*.
- **Hardware and legacy DSLs** — Verilog/VHDL and COBOL as the commercially motivated instances of the same problem; benchmark-only results so far.
- **Massively multilingual execution benchmarks** covering 20–40 languages *(frontier — verify)*; coverage has grown faster than validation of translation fidelity.

## 8. Concrete Next Experiment

**Measure $G$ and $\rho$ for one language.** Target: OCaml.

- **Scale.** Six pretraining runs at $P = 1.4$B, $N = 60$B tokens each (~$5\times10^{21}$ FLOPs per run; feasible on 64 A100s in days).
- **Arms.** Three mixtures — (A) OCaml + English only; (B) OCaml + English + Python; (C) OCaml + English + 20-language mixture — each at two OCaml budgets, $N_{\text{OCaml}} \in \{5\times10^{8}, 4\times10^{9}\}$ tokens, with total $N$ held at 60B and repetition capped at 4 epochs. **Control arm is (A)**, the solo model; it is what the literature omits.
- **Evaluation.** Two benchmarks: MultiPL-E OCaml (translated) and a held-out set of 150 OCaml problems written natively by OCaml developers with their own tests, never derived from Python. $s=200$, $\tau=0.2$.
- **Deciding number.** $G = \text{pass@1}(C) - \text{pass@1}(A)$ on the *native* benchmark at the low OCaml budget. If $G \ge 10$ points, cross-language transfer is real and worth optimizing. If $G \le 3$ points while the translated benchmark shows $G \ge 10$, the field's headline gains are benchmark-translation artifacts and the measurement must be rebuilt before any method claim stands.

Secondary readout: fit $\rho_{\text{OCaml}}$ from the two budgets in arms A and C.

## 9. Key References

- **[Foundational]** Mark Chen, Jerry Tworek, Heewoo Jun, et al. *Evaluating Large Language Models Trained on Code.* arXiv, 2021. — arXiv:2107.03374
- **[Foundational]** Denis Kocetkov, Raymond Li, Loubna Ben Allal, et al. *The Stack: 3 TB of Permissively Licensed Source Code.* TMLR, 2023. — arXiv:2211.15533
- **[SOTA]** Federico Cassano, John Gouwar, Daniel Nguyen, Sydney Nguyen, Luna Phipps-Costin, Donald Pinckney, Ming-Ho Yee, Yangtian Zi, Carolyn Jane Anderson, Molly Q Feldman, Arjun Guha, Michael Greenberg, Abhinav Jangda. *MultiPL-E: A Scalable and Polyglot Approach to Benchmarking Neural Code Generation.* IEEE Transactions on Software Engineering, 2023.
- **[SOTA]** Federico Cassano, John Gouwar, Francesca Lucchetti, Claire Schlesinger, Anders Freeman, Carolyn Jane Anderson, Molly Q Feldman, Michael Greenberg, Abhinav Jangda, Arjun Guha. *Knowledge Transfer from High-Resource to Low-Resource Programming Languages for Code LLMs.* OOPSLA, 2024.
- **[SOTA]** Gabriel Orlanski, Kefan Xiao, Xavier Garcia, Jeffrey Hui, Joshua Howland, Jonathan Malmaud, Jacob Austin, Rishabh Singh, Michele Catasta. *Measuring the Impact of Programming Language Distribution.* ICML, 2023.
- **[SOTA]** Anton Lozhkov, Raymond Li, Loubna Ben Allal, et al. *StarCoder 2 and The Stack v2: The Next Generation.* 2024. — arXiv:2402.19173
- **[Foundational]** Niklas Muennighoff, Alexander M. Rush, Boaz Barak, et al. *Scaling Data-Constrained Language Models.* NeurIPS, 2023. — arXiv:2305.16264
- **[Foundational]** Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, et al. *Training Compute-Optimal Large Language Models.* NeurIPS, 2022. — arXiv:2203.15556
- **[Survey]** Ben Athiwaratkun, Sanjay Krishna Gouda, Zijian Wang, et al. *Multi-lingual Evaluation of Code Generation Models.* ICLR, 2023. — arXiv:2210.14868
- **[Survey]** Qinkai Zheng, Xiao Xia, Xu Zou, et al. *CodeGeeX: A Pre-Trained Model for Code Generation with Multilingual Benchmarking on HumanEval-X.* KDD, 2023.

## 10. Worked Example

Take HumanEval problem 0, `has_close_elements(numbers, threshold)`. Python solution: nested loop, `abs(a-b) < threshold`.

MultiPL-E's OCaml translation produces roughly:

```ocaml
let has_close_elements (numbers: float list) (threshold: float) : bool =
```

A 15B open model of the StarCoder generation typically emits a nested `List.iter` with a mutable `ref` — a direct transliteration of the Python loop. It compiles and passes. An OCaml programmer would write `List.exists` over pairs, or sort first for $O(n\log n)$.

Now the numbers. Suppose the model scores 12.0 pass@1 on MultiPL-E OCaml. MultiPL-T fine-tuning on test-validated Python→OCaml data raises it to about 24. The paper's claim: transfer works.

Here is the obstruction, made concrete. The fine-tuning data was generated from Python, translated, and kept only if the *translated Python tests* passed. So the fine-tuning distribution is "OCaml written in Python's shape with Python's type signatures" — exactly the distribution MultiPL-E's prompts sample from. Part of the 12-point gain is the model learning to transliterate rather than learning OCaml.

Split the measurement:

| Benchmark | Before | After | Gain |
|---|---|---|---|
| MultiPL-E OCaml (Python-derived) | 12.0 | 24.0 | +12.0 |
| Native OCaml problems (functors, `Result`, `Seq`, `Map`) | ? | ? | ? |

The bottom row has never been filled in for any low-resource language with a matched solo-trained control. Until it is, the field cannot distinguish "the model transferred algorithmic competence into OCaml" from "the model learned to write Python in OCaml syntax, and the benchmark was written in Python."

A cheap partial probe: take the 150 problems where the reference OCaml solution uses a language feature with no Python analogue (functors, GADTs, polymorphic variants). If post-fine-tuning pass@1 on that subset moves less than 3 points while the full benchmark moves 12, the headline number is measuring transliteration.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*