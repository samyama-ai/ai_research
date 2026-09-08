---
id: 30-synthetic-data/leakage-free-synthetic-benchmarks
title: "Synthetic Benchmarks Free of Train-Test Leakage by Construction"
topic: 30-synthetic-data
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Synthetic Benchmarks Free of Train-Test Leakage by Construction

> **Topic:** Synthetic Data · **ID:** `30-synthetic-data/leakage-free-synthetic-benchmarks` · **Status:** open

## 1. Problem Statement

Build a benchmark generator whose instances provably cannot have appeared in any model's training corpus, and show that scores on it still measure the capability the benchmark names.

- **Input:** a capability specification $C$ (e.g. multi-step arithmetic word problems, code repair, deductive entailment) and a random seed $s$.
- **Output:** a generator $G$ that emits instances $(x, y) = G(s)$ with verified labels, plus a certificate that $x$ is absent from the training corpus of the model under test.
- **Decision predicate:** a benchmark is *leakage-free by construction* if instance-level absence holds by argument rather than by detection, **and** the score is not inflated by the generator's own regularity being memorized.

Three variants, routinely conflated:

- **Measurement variant.** Define a statistic that separates "the model saw this instance" from "the model saw this instance *family*". Currently ill-posed.
- **Method variant.** Build $G$ so novelty is cheap and refreshable (procedural generation, post-cutoff harvesting, formal-verifier-checked instances). Partially solved.
- **Theory variant.** Prove that no polynomially-bounded contamination of the training corpus can raise expected accuracy on $G$'s output by more than $\epsilon$ without the model also solving the underlying task. No such theorem exists.

## 2. Formal Setting

Let $\mathcal{D}_{\text{train}}$ be a model's training corpus (a multiset of documents), $M_\theta$ the trained model, and $G: \mathcal{S} \to \mathcal{X} \times \mathcal{Y}$ a seeded generator over seed space $\mathcal{S}$.

**Instance leakage.** For similarity $\mathrm{sim}$ and threshold $\tau$,
$$L_{\text{inst}}(\tau) = \Pr_{s \sim \mathcal{S}}\big[\exists\, d \in \mathcal{D}_{\text{train}} : \mathrm{sim}(x_s, d) > \tau \big].$$
*Measured as:* 13-gram or 50-character substring overlap (GPT-4 report protocol), or embedding cosine over a shard index. Requires corpus access; unavailable for closed models, so in practice it is replaced by a proxy (membership-inference score, or the exchangeability test of Oren et al.).

**Family leakage.** With $T$ the template/program that $G$ instantiates,
$$L_{\text{fam}} = I(T; \theta),$$
the mutual information between the generator's structure and the weights. Not directly measurable; the operational surrogate is the *transfer gap*
$$\Delta_{\text{fam}} = \mathbb{E}_{s}[\mathrm{acc}(M_\theta, G(s))] - \mathbb{E}_{s'}[\mathrm{acc}(M_\theta, G'(s'))]$$
for $G'$ a same-capability generator with a disjoint surface grammar.

**Contamination inflation.** With $\theta_0$ trained on $\mathcal{D}_{\text{train}} \setminus \{\text{benchmark-adjacent docs}\}$,
$$\Delta_{\text{leak}} = \mathrm{acc}(M_\theta) - \mathrm{acc}(M_{\theta_0}).$$
*Measured as:* a full retraining ablation. Cost is the reason it is almost never run; public-vs-fresh score deltas (GSM8k vs. GSM1k) are used instead and are only a lower bound.

**Effective entropy.** $H(G) = H(x_s)$ in bits. A necessary condition for construction-level guarantees is $H(G) \gg \log_2 |\mathcal{D}_{\text{train}}|$, i.e. the instance space is too large for any corpus to cover it. For $|\mathcal{D}_{\text{train}}| \approx 10^{13}$ tokens, $\log_2 \approx 43$ bits — so a generator with $\geq 80$ bits of instance entropy makes exact-instance collision negligible by counting alone.

**Assumptions, and where they break:**

1. *High instance entropy implies no leakage.* **Violated:** GSM-Symbolic templates have large numeric entropy yet accuracy tracks the surface template, not the instance (Mirzadeh et al., 2024).
2. *$\mathrm{sim}$ captures semantic reuse.* **Violated:** rephrased samples evade 13-gram and embedding filters while still inflating scores (Yang et al., 2023).
3. *Post-cutoff data is uncontaminated.* **Violated in the limit:** models are continuously retrained; a benchmark's cutoff advantage decays within months.
4. *Verified labels are correct.* Approximately holds for formally checked instances (SAT, theorem provers, unit tests); fails for LLM-generated labels.

## 3. State of the Art

**Established (reproduced, with ablations):**

- **Fresh re-collection.** GSM1k (Zhang et al., 2024, arXiv:2405.00332) re-authored 1,250 GSM8k-style problems by humans after the fact. Establishes a real gap for some model families and none for others — the discrimination itself is the result.
- **Live, rolling benchmarks.** LiveCodeBench (Jain et al., ICLR 2025) and LiveBench (White et al., ICLR 2025) timestamp every instance and score models only on post-release windows. Establishes measurable drops for specific models across the cutoff boundary.
- **Procedural generation with graph-controlled difficulty.** DyVal (Zhu et al., ICLR 2024) generates reasoning instances from directed acyclic graphs, with complexity knobs.
- **Contamination detection.** Oren et al. (ICLR 2024) give a provable-under-exchangeability test: if a benchmark's canonical ordering is not memorized, log-likelihood is invariant to shuffling; a significant difference certifies contamination (one-sided).

**Claimed but unablated:** that any current synthetic benchmark is leakage-free *by construction*. Every deployed system establishes only instance-level novelty; none controls $L_{\text{fam}}$. "Contamination-free" in LiveCodeBench/LiveBench names a timestamp discipline, not a proof.

**Benchmark-number-only results:** ARC-AGI-2 (Chollet et al., 2025) scores, GSM-Symbolic drop magnitudes, and most "our synthetic set is uncontaminated" claims in model cards are single numbers with no retraining control arm.

## 4. What Is Known

- **GSM1k (Zhang et al., 2024):** accuracy drops up to **13 percentage points** vs. GSM8k on some open model families; frontier models (GPT-4 class, Claude, Gemini at the time) show near-zero gap. Scale: 1,250 problems, ~50 models.
- **GSM-Symbolic (Mirzadeh et al., ICLR 2025):** changing only proper names and numeric values within a template moves accuracy by several points; adding one irrelevant clause (GSM-NoOp) drops accuracy by up to **~65%** relative on some models. Scale: 100 templates × 50 instantiations, ~20 models.
- **Rephrasing defeats n-gram filters (Yang et al., 2023, arXiv:2311.04850):** a 13B model fine-tuned on rephrased test items reaches near-perfect benchmark scores while passing standard decontamination.
- **Membership inference on LLMs is near-chance (Duan et al., COLM 2024, arXiv:2402.07841):** AUC ≈ **0.5–0.55** across Pythia 160M–12B on Pile members vs. non-members, once the candidate sets are distribution-matched. So MIA cannot serve as the leakage certificate.
- **Memorization scales (Carlini et al., ICLR 2023, arXiv:2202.07646):** extractable memorization grows log-linearly in model size, example duplication count, and prompt-prefix length — GPT-Neo 6B memorizes an order of magnitude more than 125M under the same protocol.
- **Functional variants expose a "reasoning gap"** (Srivastava et al., 2024, arXiv:2402.19450): parameterized MATH problems yield 58–80% relative gaps for several models at the time of measurement.
- **Counterfactual task variants** (Wu et al., NAACL 2024): performance drops sharply under base-9 arithmetic or shifted keyboard layouts, above-chance but far below the default condition.

## 5. What Is Not Known

- **Theoretically open.** No theorem bounding $\Delta_{\text{leak}}$ as a function of $H(G)$ and corpus size. No formal separation between "solved by generalization" and "solved by memorizing the generator". Whether such a separation is even well-posed for a generator whose source code is public is unclear.
- **Empirically open.** The retraining ablation — pretrain two matched models, one with the generator's code/templates in-corpus, one without, at ≥7B and ≥300B tokens — has not been run for any synthetic benchmark. Runnable today for roughly $10^5$ GPU-hours.
- **Methodologically blocked.** $L_{\text{fam}}$ has no accepted estimator. "Same capability, different surface form" has no operational definition, so the transfer gap $\Delta_{\text{fam}}$ is uninterpretable: a drop could mean family leakage or genuine difficulty change. Construct validity of any procedurally generated task is likewise undefined — there is no ground truth for what "multi-step reasoning" is beyond the benchmark.

## 6. Why It Is Hard

**The core obstruction is non-identifiability plus an absent control arm.**

- *Non-identifiability.* Instance novelty and skill novelty are not separable from behaviour alone. A fresh instance drawn from a template the model has seen 10,000 times tests pattern application, not the named capability; a fresh instance from an unseen template may simply be harder. Both produce a score drop, and no observable distinguishes them.
- *Absent control arm.* The only clean estimator of $\Delta_{\text{leak}}$ requires retraining with the benchmark removed. Nobody retrains a frontier model to validate one benchmark, so the field substitutes proxies — and the proxies (n-gram overlap, MIA) are measured at AUC ≈ 0.5 and fail against paraphrase.
- *Evaluation does not measure what it names.* "Contamination-free" in deployed benchmarks names timestamp hygiene, which decays with each retraining cycle, not a construction-level guarantee.
- *The generator is itself the leak.* Publishing $G$ so results are reproducible puts the solution procedure into the next crawl. Withholding $G$ makes the benchmark unauditable. There is no known construction that is both public and immune.

## 7. Current Research (as of 2026)

- **Rolling/live evaluation:** LiveBench (NYU/Abacus/Nvidia consortium), LiveCodeBench (Berkeley/MIT), monthly refresh cycles.
- **Formally verified generators:** Lean/Isabelle-checked synthetic theorem sets and compiler-checked program tasks, where labels are certified rather than annotated *(frontier — verify current scale)*.
- **Template-perturbation suites:** GSM-Symbolic (Apple), functional MATH() (Consequent AI), platinum-quality re-labelled benchmarks (Vendrow et al., 2025, arXiv:2502.03461).
- **Provable contamination tests:** exchangeability tests (Stanford, Oren et al.), performance-based detection with reference distributions (ConStat, Dekoninck et al., NeurIPS 2024).
- **Canary/tripwire strings** embedded in benchmark releases (BIG-bench precedent) to detect ingestion after the fact — detection, not prevention.
- **Hardness-controlled combinatorial generators** revisited: planted-solution SAT near the $m/n \approx 4.26$ phase transition, and the Chomsky-hierarchy formal-language suites (Delétang et al., ICLR 2023), as capability probes whose entropy is unbounded by construction.

## 8. Concrete Next Experiment

**Question:** does instance-level novelty suffice, or does family-level leakage inflate scores on procedurally generated benchmarks?

- **Scale.** Pretrain two 1.4B-parameter decoder models on 300B tokens of a fixed open corpus (e.g. a Pile/FineWeb slice). Cost: roughly 6,000 A100-hours per arm.
- **Treatment arm.** Inject into the corpus 200M tokens of *solved* instances from generator $G$ (a DyVal-style DAG reasoning generator with $H(G) > 100$ bits), sampled from seed space $\mathcal{S}_A$ — chain-of-thought solutions included.
- **Control arm.** Identical corpus with the 200M tokens replaced by length- and domain-matched text containing no $G$-family content.
- **Evaluation.** Both models on 5,000 instances from disjoint seed space $\mathcal{S}_B$ (zero instance overlap, verified by exact match and 13-gram check), plus 5,000 instances from $G'$, a semantically equivalent generator with a different surface grammar.
- **Deciding number:** $\Delta_{\text{fam}} = \mathrm{acc}_{\text{treat}}(\mathcal{S}_B) - \mathrm{acc}_{\text{ctrl}}(\mathcal{S}_B)$, with 95% CI from 5 seeds. **If $\Delta_{\text{fam}} \le 2$ points, instance-level novelty is sufficient and "leakage-free by construction" is achievable by high-entropy generation. If $\Delta_{\text{fam}} \ge 10$ points, it is not — and every current synthetic benchmark that publishes its generator is inflated by an unmeasured amount.**
- **Secondary readout:** whether the inflation transfers to $G'$. If $\Delta_{\text{fam}}(G') \approx \Delta_{\text{fam}}(G)$, the model learned the capability; if $\Delta_{\text{fam}}(G') \approx 0$, it learned the grammar.

## 9. Key References

- **[Foundational]** Carlini, Ippolito, Jagielski, Lee, Tramèr, Zhang. *Quantifying Memorization Across Neural Language Models.* ICLR 2023. — arXiv:2202.07646
- **[Foundational]** Selman, Mitchell, Levesque. *Hard and Easy Distributions of SAT Problems.* AAAI 1992. — the original hardness-controlled synthetic benchmark.
- **[SOTA]** Zhang, Da, Lee, et al. *A Careful Examination of Large Language Model Performance on Grade School Arithmetic.* NeurIPS 2024. — arXiv:2405.00332
- **[SOTA]** Mirzadeh, Alizadeh, Shahrokhi, Tuzel, Bengio, Farajtabar. *GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models.* ICLR 2025. — arXiv:2410.05229
- **[SOTA]** Jain, Han, Gu, Li, Yan, Zhang, Wang, Solar-Lezama, Sen, Stoica. *LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code.* ICLR 2025. — arXiv:2403.07974
- **[SOTA]** White, Dooley, Roberts, et al. *LiveBench: A Challenging, Contamination-Free LLM Benchmark.* ICLR 2025. — arXiv:2406.19314
- **[SOTA]** Zhu, Chen, Wang, et al. *DyVal: Dynamic Evaluation of Large Language Models for Reasoning Tasks.* ICLR 2024. — arXiv:2309.17167
- **[Method]** Oren, Meister, Chatterji, Ladhak, Hashimoto. *Proving Test Set Contamination in Black Box Language Models.* ICLR 2024. — arXiv:2310.17623
- **[Method]** Yang, Chiang, Zheng, Gonzalez, Stoica. *Rethinking Benchmark and Contamination for Language Models with Rephrased Samples.* 2023. — arXiv:2311.04850
- **[Method]** Duan, Suri, Mireshghallah, et al. *Do Membership Inference Attacks Work on Large Language Models?* COLM 2024. — arXiv:2402.07841
- **[Method]** Dekoninck, Müller, Vechev. *ConStat: Performance-Based Contamination Detection in Large Language Models.* NeurIPS 2024. — arXiv:2405.16281
- **[Related]** Wu, Zhang, Zhang, et al. *Reasoning or Reciting? Exploring the Capabilities and Limitations of Language Models Through Counterfactual Tasks.* NAACL 2024. — arXiv:2307.02477
- **[Related]** Srivastava, Anand, et al. *Functional Benchmarks for Robust Evaluation of Reasoning Performance, and the Reasoning Gap.* 2024. — arXiv:2402.19450
- **[Related]** Delétang, Ruoss, Grau-Moya, et al. *Neural Networks and the Chomsky Hierarchy.* ICLR 2023. — arXiv:2207.02098
- **[Survey]** Chollet. *On the Measure of Intelligence.* 2019. — arXiv:1911.01547

## 10. Worked Example

Take a DyVal-style arithmetic-DAG generator. Each instance is a DAG with $n = 7$ internal nodes, each an operator drawn from $\{+, -, \times, \max, \min\}$, leaves drawn uniformly from $[0, 999]$.

**Entropy count.** Operators: $7 \log_2 5 \approx 16.3$ bits. Leaves (8 of them): $8 \log_2 1000 \approx 79.7$ bits. Topology: $\approx 20$ bits. Total $H(G) \approx 116$ bits, so $|\mathcal{X}| \approx 2^{116} \approx 8 \times 10^{34}$.

A 15-trillion-token corpus holds at most $\sim 10^{12}$ such instances if it were nothing else. Collision probability for a 5,000-instance benchmark:
$$L_{\text{inst}} \le 5000 \times \frac{10^{12}}{8\times10^{34}} \approx 6 \times 10^{-20}.$$

By counting, instance leakage is zero. The construction looks solved.

**Where it breaks.** Now measure. A 7B model scores 61% on this generator. Re-render the *same DAGs* into a different surface form — infix expressions instead of natural-language "node A is the sum of B and C" prose — and it scores 38%. Same semantic content, same entropy, 23-point gap.

Two explanations fit identically:

1. The prose grammar (or near-variants of it, published with DyVal in 2023) is in the corpus; the model pattern-matches the phrasing. Score inflation of 23 points.
2. Infix notation with 7-deep nesting is genuinely harder to parse; the model's arithmetic ability is unchanged and the prose score is honest.

No observable in the benchmark separates these. $L_{\text{inst}} = 6 \times 10^{-20}$ says nothing about which holds, because the quantity that matters is $I(T;\theta)$ — mutual information with the *template* — and it is not estimated by any deployed method. The 116-bit entropy guarantee is real and irrelevant. That is the obstruction: instance-level leakage is easy to drive to zero and is not the leakage that inflates scores.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*