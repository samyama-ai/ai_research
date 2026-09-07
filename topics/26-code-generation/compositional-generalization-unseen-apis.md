---
id: 26-code-generation/compositional-generalization-unseen-apis
title: "Compositional Generalization to Unseen Library APIs"
topic: 26-code-generation
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Compositional Generalization to Unseen Library APIs

> **Topic:** Code Generation & Program Synthesis · **ID:** `26-code-generation/compositional-generalization-unseen-apis` · **Status:** open

## 1. Problem Statement

A code model is given a natural-language task and a specification of a library it has never seen in training — signatures, docstrings, type stubs, maybe a few usage examples. It must emit a program that composes several of that library's functions correctly on the first try.

- **Input:** task description $x$, plus an API context $A = \{(s_i, d_i)\}_{i=1}^m$ of signatures $s_i$ and documentation $d_i$.
- **Output:** program $\hat{y}$.
- **Predicate:** $\hat{y}$ passes a held-out test suite $T$, and every call in $\hat{y}$ resolves against $A$.

Three variants, routinely conflated:

- **Measurement:** build an evaluation where "unseen" is verifiable, not assumed. Pretraining corpora for frontier models are undisclosed, so any public library may be contaminated. This variant is currently the binding constraint.
- **Method:** get accuracy on novel-API composition close to accuracy on familiar-API composition of matched difficulty, via retrieval, in-context docs, tool use, or execution feedback.
- **Theory:** state conditions on the model class and data distribution under which held-out *compounds* of seen atoms are learned. Open even for the toy setting of SCAN (Lake & Baroni, ICML 2018).

Solving it means: the gap between seen-API and unseen-API pass rate, at matched compositional depth and matched task difficulty, is small and does not grow with the number of composed calls.

## 2. Formal Setting

Let $\mathcal{A}$ be the universe of API symbols. A program $y$ has an **atom set** $\mathrm{At}(y) \subseteq \mathcal{A}$ (the symbols it calls) and a **compound multiset** $\mathrm{Cp}(y)$ (subtrees of its call graph up to depth $k$, e.g. `f(g(·), ·)` with argument roles). Training distribution $\mathcal{D}_{\text{tr}}$, evaluation distribution $\mathcal{D}_{\text{ev}}$.

Following Keysers et al. (ICLR 2020), with $\mathcal{F}_V$ the frequency distribution of a feature set under split $V$ and $C_\alpha(P\|Q) = \sum_k P_k^\alpha Q_k^{1-\alpha}$ the Chernoff coefficient:

$$\mathcal{D}_A = 1 - C_{0.5}(\mathcal{F}^{\mathrm{At}}_{\text{ev}} \,\|\, \mathcal{F}^{\mathrm{At}}_{\text{tr}}), \qquad \mathcal{D}_C = 1 - C_{0.1}(\mathcal{F}^{\mathrm{Cp}}_{\text{ev}} \,\|\, \mathcal{F}^{\mathrm{Cp}}_{\text{tr}}).$$

The classical compositional split holds $\mathcal{D}_A \approx 0$ and pushes $\mathcal{D}_C \to 1$. The **unseen-API** setting is the harder regime $\mathcal{D}_A \to 1$: the atoms themselves are new and are supplied only through $A$ at inference.

Measured quantities:

- **Pass rate:** $\mathrm{pass}@1 = \mathbb{E}_{(x,A,T)}\big[\mathbb{1}[\hat{y} \models T]\big]$, one sample, temperature fixed and reported, tests executed in a sandbox.
- **Composition depth:** $c(y) = $ number of distinct library calls in $y$'s reference solution. Report pass@1 stratified by $c$.
- **Generalization gap:** $\Delta(c) = \mathrm{pass}@1_{\text{seen}}(c) - \mathrm{pass}@1_{\text{unseen}}(c)$, with the seen arm matched on $c$, prompt length, and test-suite strength.
- **Grounding rate:** fraction of emitted calls resolving to a symbol in $A$ with a type-compatible arity. Ungrounded calls are API hallucinations.
- **Novelty, measured not assumed:** $n(A) = $ number of $n$-gram hits ($n \ge 8$) of each signature and docstring in the training corpus. If the corpus is unavailable, $n(A)$ is unmeasured and every claim about "unseen" is an assumption.

Assumptions and their status: (i) *the library is absent from pretraining* — violated for every public library and unverifiable for closed models; (ii) *tests are strong enough that passing implies correct* — violated; EvalPlus (Liu et al., NeurIPS 2023) showed HumanEval tests admit wrong programs; (iii) *docs are complete and correct* — violated in real ecosystems; (iv) *seen and unseen arms are difficulty-matched* — rarely enforced, and it is the main confound.

## 3. State of the Art

**Empirical SOTA (established).** Retrieval of documentation into the prompt helps and has been ablated. DocPrompting (Zhou et al., ICLR 2023) explicitly targets unseen functions: retrieve docs for library/CLI commands not in training, condition generation on them; it reports gains on CoNaLa (~+2.9 pass@1) and tldr (~+4.4 exact match) over non-retrieval baselines of the same size. Repository-level retrieval (RepoCoder, Zhang et al., EMNLP 2023; CrossCodeEval, Ding et al., NeurIPS 2023) shows the same shape for private in-repo APIs: retrieval closes part of the gap, never all of it.

**Empirical SOTA (benchmark number only, unablated).** Frontier models on BigCodeBench (Zhuo et al., ICLR 2025), which requires composing calls across 139 libraries: at release, no model exceeded roughly 60% pass@1 on the Complete split against a ~97% human-estimated ceiling. This is a leaderboard number, not a controlled seen-vs-unseen contrast — the libraries are all public and plausibly in pretraining, so it measures difficult composition, not novelty.

**Tool/agent framings.** Gorilla (Patil et al., NeurIPS 2024) and ToolLLM (Qin et al., ICLR 2024) fine-tune on API documentation and report large gains in correct API selection, including for held-out APIs. These are single-call or shallow-chain settings; deep composition of a novel API surface is largely untested there.

**Theory SOTA.** Nothing specific to APIs. The general result set is negative-to-descriptive: seq2seq models fail systematic splits (Lake & Baroni 2018); pre-training helps more than architecture on CFQ (Furrer et al., 2020); transformers solve compositional tasks by matching subgraph patterns rather than composing, and degrade sharply with depth (Dziri et al., "Faith and Fate", NeurIPS 2023).

## 4. What Is Known

- **Compound splits break models at small scale.** SCAN add-jump: standard seq2seq at ~99% on the random split falls to ~1% on add-jump (Lake & Baroni, ICML 2018, ~20k examples).
- **Realistic compound divergence hurts too.** CFQ MCD splits: baseline accuracy ~18% mean versus ~95% on a random split of the same data (Keysers et al., ICLR 2020, ~240k queries).
- **Depth is the failure axis.** GPT-4 on multi-digit multiplication drops from high accuracy at 2×2 digits to near zero at 4×4 (Dziri et al., NeurIPS 2023), with errors localized to composed intermediate steps rather than single operations.
- **Weak tests inflate everything.** EvalPlus adds ~80× more tests to HumanEval and drops reported pass@1 by roughly 10–15 points for models across the 2023 frontier (NeurIPS 2023). Any unseen-API gap measured with thin tests is measured with an instrument of that error size.
- **Private-library generation is measurably worse.** Zan et al. (Findings of EMNLP 2022; CERT, IJCAI 2022) built evaluations on libraries released after model training cutoffs and found large drops versus public libraries at the ~350M–13B parameter scale, partially recovered by doc-conditioned prompting.

## 5. What Is Not Known

- **Methodologically blocked (primary).** Whether an API is truly unseen cannot be checked for closed frontier models; training corpora are undisclosed. Without $n(A)$, "unseen API" is an unverified label, and the headline quantity $\Delta(c)$ is not well defined for the models people care about.
- **Empirically open.** Does $\Delta(c)$ grow with $c$ at frontier scale? The experiment is runnable — synthesize a novel library, difficulty-match a seen arm — and has not been run with matched controls at the 10^2-task, multi-model scale.
- **Empirically open.** Do agentic loops (execute, read traceback, retry) collapse $\Delta$, or only trade it for token spend? Reported agent numbers on SWE-bench (Jimenez et al., ICLR 2024) do not decompose into novelty versus iteration.
- **Theoretically open.** No characterization of when a transformer trained on programs learns a symbol-invariant composition operator versus memorized call templates. No separation theorem, no matching lower bound.

## 6. Why It Is Hard

The obstruction is **confounded measurement plus absent ground truth about the training set**. Every naturally "unseen" library differs from seen libraries on axes other than novelty: it is newer, less documented, less idiomatic, and its tasks are drawn from a different domain. So a measured drop is a sum of novelty, difficulty, doc quality, and domain shift, with no way to attribute it. The clean fix — synthesize a library that provably did not exist at training time — buys verifiability at the cost of ecological validity: synthetic APIs are unnaturally regular, and a model that succeeds on them may still fail on a real library whose semantics contradict its name. A second obstruction is that pass@1 with weak tests has a systematic error (~10–15 points, §4) of the same magnitude as the effect being estimated.

## 7. Current Research (as of 2026)

- **Version- and update-aware benchmarks.** VersiCode (Wu et al., 2024) and CodeUpdateArena (Liu et al., 2024) evaluate whether models track API changes over library versions; both find substantial drops when the target version postdates the model. *(frontier — verify current leaderboard state.)*
- **Doc-conditioned retrieval at long context.** Whether 100k+ token API dumps substitute for retrieval, or degrade via position effects, is being tested by several groups. *(frontier — verify.)*
- **Continual / knowledge-editing approaches** that inject a new library's semantics into weights rather than context (successors to CERT). *(frontier — verify.)*
- **Execution-feedback agents** (SWE-bench-lineage systems) as the practical answer: substitute iteration for generalization.
- **Synthetic-library probes** with controlled compound divergence, following the CFQ/COGS (Kim & Linzen, EMNLP 2020) methodology transplanted to code. Thinly published so far.

## 8. Concrete Next Experiment

**Question:** does the unseen-API penalty grow with composition depth, once difficulty is matched?

- **Scale.** Machine-generate two libraries, $L_{\text{novel}}$ and $L_{\text{mirror}}$, with identical semantics under different names — 60 functions each. Publish neither before the run; hold $L_{\text{novel}}$ private (never on the public web). Draw 600 tasks, 150 at each depth $c \in \{1,2,3,4\}$, each with $\ge 40$ property-based tests. Evaluate 5 models spanning ~7B open weights to frontier closed, pass@1 at $T=0$, 5 seeds of task ordering.
- **Control arm.** The same 600 tasks re-expressed against a *widely-seen* library (e.g. `itertools` + `pathlib` + `numpy`) with identical call graphs, identical prompt length, and identical test count. Second control: $L_{\text{mirror}}$ with names alpha-renamed to seen library names but unseen semantics — separates name-prior from structure.
- **Deciding number.** The slope $\beta$ in $\Delta(c) = \alpha + \beta c$, fit across the four depths. $\beta \le 2$ points/depth with 95% CI excluding 5 ⇒ the failure is per-symbol lookup, and better retrieval suffices. $\beta \ge 8$ points/depth ⇒ the failure is compositional, and retrieval cannot fix it. Report grounding rate alongside; if ungrounded calls explain more than half of $\Delta$, the problem is name recall, not composition.

## 9. Key References

- **[Foundational]** Brenden Lake, Marco Baroni. *Generalization without Systematicity: On the Compositional Skills of Sequence-to-Sequence Recurrent Networks.* ICML, 2018. — arXiv:1711.00350
- **[Foundational]** Daniel Keysers et al. *Measuring Compositional Generalization: A Comprehensive Method on Realistic Data.* ICLR, 2020. — arXiv:1912.09713
- **[Foundational]** Najoung Kim, Tal Linzen. *COGS: A Compositional Generalization Challenge Based on Semantic Interpretation.* EMNLP, 2020.
- **[SOTA]** Shuyan Zhou et al. *DocPrompting: Generating Code by Retrieving the Docs.* ICLR, 2023. — arXiv:2207.05987
- **[SOTA]** Terry Yue Zhuo et al. *BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions.* ICLR, 2025. — arXiv:2406.15877
- **[SOTA]** Shishir G. Patil et al. *Gorilla: Large Language Model Connected with Massive APIs.* NeurIPS, 2024. — arXiv:2305.15334
- **[SOTA]** Yizhong Qin et al. *ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs.* ICLR, 2024. — arXiv:2307.16789
- **[Evaluation]** Jiawei Liu et al. *Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code.* NeurIPS, 2023. — arXiv:2305.01210
- **[Evaluation]** Carlos E. Jimenez et al. *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* ICLR, 2024. — arXiv:2310.06770
- **[Evaluation]** Yangruibo Ding et al. *CrossCodeEval: A Diverse and Multilingual Benchmark for Cross-File Code Completion.* NeurIPS, 2023.
- **[Method]** Fengji Zhang et al. *RepoCoder: Repository-Level Code Completion Through Iterative Retrieval and Generation.* EMNLP, 2023. — arXiv:2303.12570
- **[Method]** Daoguang Zan et al. *CERT: Continual Pre-training on Sketches for Library-oriented Code Generation.* IJCAI, 2022.
- **[Method]** Daoguang Zan et al. *When Language Model Meets Private Library.* Findings of EMNLP, 2022.
- **[Analysis]** Nouha Dziri et al. *Faith and Fate: Limits of Transformers on Compositionality.* NeurIPS, 2023. — arXiv:2305.18654
- **[Survey]** Dieuwke Hupkes et al. *Compositionality Decomposed: How do Neural Networks Generalise?* JAIR, 2020. — arXiv:1908.08351

## 10. Worked Example

A synthetic tensor library `zx` with three functions: `zx.fold(t, axis)` (cumulative sum along `axis`), `zx.gate(t, lo, hi)` (clamp), `zx.braid(a, b)` (elementwise interleave). Docs for all three go in the prompt. Task: *"clamp `t` to [0,1], cumulatively sum along axis 0, then interleave with the original."* Reference: `zx.braid(zx.fold(zx.gate(t,0,1),0), t)`, depth $c=3$.

Observed failure modes across a typical 10-sample run, and what each implies:

| Failure | Count | Diagnosis |
|---|---|---|
| Argument order swapped in `braid` | 3 | compositional — atoms grounded, structure wrong |
| `zx.clip` emitted (not in $A$) | 2 | name prior from NumPy; grounding failure |
| `fold` applied before `gate` | 2 | compositional ordering |
| Correct | 3 | — |

pass@1 $= 0.30$. The matched seen-library arm — `np.cumsum(np.clip(t,0,1),0)` interleaved — passes at $0.85$, so $\Delta(3) = 55$ points.

The obstruction shows up when you try to attribute those 55 points. Two of ten failures are pure name recall (`clip` for `gate`) and would vanish with better retrieval; five are structural and would not. But `zx` is regular by construction — its three functions are renamed NumPy primitives — so the seen arm's advantage is partly that the *model already knows the semantics* and only needs the name. Swap in a genuinely novel semantic (say `braid` interleaves along the last axis for odd ranks and the first for even) and the seen arm has no analogue to be matched against at all. Difficulty-matching and novelty are in tension: the more faithfully novel the library, the less the control arm controls. That is why $\Delta(c)$, the quantity everyone quotes, is not yet a well-defined measurement.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*