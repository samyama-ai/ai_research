---
id: 20-interpretability/circuit-completeness-verification
title: "Completeness of Discovered Circuits"
topic: 20-interpretability
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Completeness of Discovered Circuits

> **Topic:** Interpretability · **ID:** `20-interpretability/circuit-completeness-verification` · **Status:** open

## 1. Problem Statement

A *circuit* is a subgraph of a network's computational graph claimed to implement a behavior. Circuit discovery reports subgraphs; the open problem is certifying that a reported subgraph is **complete** — that it contains *every* component the model actually uses for the behavior, not merely enough components to reproduce the output on the evaluation distribution.

- **Input:** a model $M$, a task distribution $\mathcal{D}$, a behavioral metric $F$, an ablation operator, and a candidate subgraph $C$.
- **Output:** a decision or a certified bound on the incompleteness of $C$.
- **Predicate:** no set of components outside $C$ matters for $F$ on $\mathcal{D}$, under any pattern of interventions inside $C$.

Three variants, of very different difficulty:

- **Measurement:** define an incompleteness score that is invariant to ablation choice and does not reward memorized shortcuts. Currently unsettled.
- **Method:** compute or bound that score without enumerating $2^{|C|}$ subsets. Currently intractable exactly.
- **Theory:** prove that any faithful circuit of minimal size is complete, or exhibit a model class where faithful-but-incomplete circuits are generic. Open.

Faithfulness ("$C$ alone reproduces the behavior") and completeness ("nothing outside $C$ is needed") are different claims. Nearly all published circuits report the first.

## 2. Formal Setting

Let $M$ be a transformer with component set $N$ (attention heads, MLPs, or SAE features), $|N| = n$. A circuit is $C \subseteq N$. Let $\mathcal{D}$ be a distribution over prompts with a paired *corrupted* distribution $\mathcal{D}'$ (e.g. name-swapped prompts).

**Metric.** $F(x)$ is a scalar readout, in practice the logit difference
$$F(x) = z_{\text{correct}}(x) - z_{\text{incorrect}}(x),$$
measured as the raw pre-softmax difference averaged over $N_{\text{prompts}}$ samples; the reported number is $\mathbb{E}_{x\sim\mathcal{D}}[F(x)]$ with standard error $\sigma/\sqrt{N_{\text{prompts}}}$.

**Ablation.** $M_{|K \to a}$ denotes $M$ with components $K$ replaced by a value $a$: zero, dataset mean, resample from $\mathcal{D}'$, or optimal-noise. The choice is a free parameter and changes results materially (§4).

**Faithfulness.**
$$\Phi(C) = \frac{\mathbb{E}[F(M_{|N\setminus C \to a})]}{\mathbb{E}[F(M)]},$$
i.e. keep $C$, ablate its complement. Reported as a percentage.

**Incompleteness** (Wang et al., ICLR 2023):
$$\mathrm{IC}(C) = \max_{K \subseteq C}\;\bigl|\;\mathbb{E}[F(M_{|K \to a})] - \mathbb{E}[F(M_{|(N\setminus C)\cup K \to a})]\;\bigr|.$$
Read: for every subset $K$ knocked out of the circuit, the full model and the circuit must degrade the same way. $C$ is complete iff $\mathrm{IC}(C) \le \epsilon$ for a stated tolerance $\epsilon$.

**Causal scrubbing** (Chan et al., 2022) replaces the max over $K$ with a hypothesis $h$ mapping graph nodes to input-invariance classes, and scores the expected metric under all resampling consistent with $h$.

**Assumptions, and their status in practice:**

1. *Ablation values are off-distribution-safe.* Violated: zero-ablation pushes activations far outside the training manifold; resample ablation is standard precisely because of this.
2. *The max over $K$ is computable.* Violated: $2^{|C|}$ terms. Every published $\mathrm{IC}$ is a sampled lower bound.
3. *Components are the right unit.* Violated under superposition — a head can carry several features, so head-level completeness does not imply feature-level completeness.
4. *$\mathcal{D}$ covers the behavior.* Violated: circuits are validated on templated prompt sets, and $\mathrm{IC}$ is only ever a statement about $\mathcal{D}$.
5. *$F$ is the behavior.* Logit difference is a proxy; KL to the full model's distribution gives different rankings.

## 3. State of the Art

**Established.**
- The IOI circuit in GPT-2 small (Wang, Variengien, Conmy, Shlegeris, Steinhardt, ICLR 2023): 26 attention heads of 144, recovering roughly 87% of the full model's logit difference; the paper is the origin of the completeness test and reports it on sampled $K$.
- ACDC (Conmy et al., NeurIPS 2023) automates subgraph search by iterative edge pruning against a KL threshold; it recovers the IOI heads at competitive precision/recall on small tasks.
- EAP / attribution patching (Syed, Rager, Conmy, 2023) approximates every edge's effect with two forward passes and one backward pass, making circuit search $O(1)$ in graph size rather than $O(|E|)$.
- Sparse feature circuits (Marks, Rager, Michaud, Belinkov, Bau, Mueller, ICLR 2025) move the unit from heads to SAE features, addressing assumption 3.

**Claimed but unablated / benchmark-only.**
- Completeness claims for any circuit above GPT-2 scale. No published circuit in a model $\ge$ 7B parameters reports an $\mathrm{IC}$ score with stated subset coverage.
- InterpBench (Gupta, Jenner et al., NeurIPS 2024 Datasets & Benchmarks) and MIB (Mueller et al., 2025) give ground-truth or leaderboard scores for *recovery* of known circuits. These are benchmark numbers under a fixed ablation and fixed distribution; they do not certify completeness on natural models.
- Transcoder circuits (Dunefsky, Chlenski, Nanda, NeurIPS 2024) report interpretable feature-level graphs; completeness of those graphs is not tested.

## 4. What Is Known

- **Circuits are sparse but not tiny.** IOI: 26/144 heads, 7 functional classes, ~87% logit-difference recovery — GPT-2 small, 124M params, templated prompts ($\sim$10$^3$ examples).
- **Faithfulness metrics are not robust.** Miller, Chughtai, Saunders (COLM 2024) show that on the same IOI circuit, changing ablation type, the corrupted distribution, or whether ablation is by node or edge changes faithfulness scores by tens of percentage points, and can rank a random subgraph above the published circuit. Scale: GPT-2 small.
- **Patching can produce illusions.** Makelov, Lange, Nanda (ICLR 2024) show subspace activation patching can restore behavior through a *dormant* pathway not used by the clean model — a "faithful" intervention on a component the model does not use. Scale: GPT-2 small, IOI.
- **Circuit hypotheses partially fail statistical tests.** Shi, Beltran-Velez, Zheng, Blei et al. (NeurIPS 2024) formalize circuit criteria as hypothesis tests; published circuits pass some and fail others, notably tests of sufficiency versus necessity.
- **Components are reused across tasks.** Merullo, Eickhoff, Pavlick (ICLR 2024) show the same induction/mover heads serve several tasks in GPT-2 medium — so "outside the circuit" is task-relative, not absolute.
- **Ground truth exists only in synthetic models.** Tracr (Lindner et al., NeurIPS 2023) compiles RASP programs to exact weights, giving true circuits; InterpBench extends this with interchange-intervention training.

## 5. What Is Not Known

- **Methodologically blocked:** whether $\mathrm{IC}$ measures completeness at all. It is defined relative to an ablation whose choice is known to swing the number (Miller et al.), and there is no ablation-invariant definition. This blocks the other two variants.
- **Theoretically open:** no theorem relating faithfulness to completeness. It is unproven whether, for transformers, a minimal $\Phi \ge 1-\epsilon$ subgraph must have $\mathrm{IC} \le g(\epsilon)$ for any $g$; no counterexample construction is published either.
- **Theoretically open:** hardness. Exact completeness verification is plausibly NP-hard by reduction from subset selection, but no reduction is in the literature.
- **Empirically open:** the exhaustive $\mathrm{IC}$ sweep for one small circuit has never been run (§10 shows it costs ~30 GPU-days, i.e. it is affordable).
- **Empirically open:** whether sampled $\mathrm{IC}$ underestimates true $\mathrm{IC}$ by a little or by a lot. Unknown at every scale.

## 6. Why It Is Hard

Three named obstructions.

1. **Exponential search with a max, not a mean.** $\mathrm{IC}$ is a maximum over $2^{|C|}$ subsets. Sampling estimates means well and maxima badly; every published score is a lower bound with no upper-bound guarantee. A circuit can look complete under $10^3$ sampled $K$ and fail on one adversarial $K$.
2. **Absent ground truth.** Outside Tracr/InterpBench, no model has a known true circuit, so completeness methods cannot be validated — only compared to each other.
3. **Confounded measurement.** Ablation moves activations off-distribution; the model's degradation then mixes "this component mattered" with "this input is now nonsense". The dormant-pathway illusion (Makelov et al.) is the sharp case: the metric names causal necessity and measures reachability instead.

Non-identifiability compounds all three: under superposition, the set of components carrying a feature is not unique, so "complete" is not well defined at head granularity.

## 7. Current Research (as of 2026)

- **Feature-level circuits.** Anthropic, EleutherAI, Bau Lab, and Northeastern/MIT groups building SAE- and transcoder-based graphs, replacing heads with features to make the unit well defined. Attribution-graph work at Anthropic (2025) traces feature-level circuits in production-scale models but reports completeness qualitatively.
- **Benchmarking.** MIB (Mueller et al., 2025) standardizes circuit-localization scoring across models and tasks; the field is converging on it as the comparison surface *(frontier — verify current leaderboard composition)*.
- **Robustness of metrics.** Follow-ups to Miller et al. on ablation-invariant faithfulness; proposals for optimal-ablation baselines.
- **Synthetic ground truth.** Extensions of InterpBench to larger compiled models to give completeness methods something to be wrong about *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question:** does sampled incompleteness underestimate true incompleteness enough to invalidate published completeness claims?

- **Scale:** GPT-2 small (124M), IOI task, the published 26-head circuit, $N_{\text{prompts}} = 1000$ per evaluation, mean-resample ablation from the ABC distribution.
- **Procedure:** compute $\mathbb{E}[F]$ for **all** $2^{26} = 6.7\times10^7$ subsets $K \subseteq C$ under both the full model and the circuit, giving the exact $\mathrm{IC}(C)$.
- **Control arm:** the same exhaustive sweep for 5 random 26-head subgraphs matched on layer distribution, plus the ACDC-recovered circuit at matched size.
- **Deciding number:** the ratio
  $$\rho = \frac{\mathrm{IC}_{\text{exact}}(C)}{\mathrm{IC}_{\text{sampled}}(C,\;10^3\text{ subsets})}.$$
  $\rho < 1.5$: sampled scores are trustworthy and the practice stands. $\rho > 5$, or $\mathrm{IC}_{\text{exact}}(C)$ exceeding the *sampled* score of the random controls: every published completeness claim is uninformative and the metric needs replacing.

Cost estimate in §10. Secondary readout: the empirical distribution of $\mathbb{E}[F(M_{|K})] - \mathbb{E}[F(C_{|K})]$ over all $K$, which tells whether incompleteness is carried by a few adversarial subsets or is diffuse.

## 9. Key References

- **[Foundational]** Wang, Variengien, Conmy, Shlegeris, Steinhardt. *Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 Small.* ICLR 2023. — arXiv:2211.00593
- **[Foundational]** Elhage, Nanda, Olsson, et al. *A Mathematical Framework for Transformer Circuits.* Transformer Circuits Thread, 2021.
- **[Foundational]** Chan, Garriga-Alonso, Goldowsky-Dill, et al. *Causal Scrubbing: A Method for Rigorously Testing Interpretability Hypotheses.* Alignment Forum, 2022.
- **[SOTA]** Conmy, Mavor-Parker, Lynch, Heimersheim, Garriga-Alonso. *Towards Automated Circuit Discovery for Mechanistic Interpretability.* NeurIPS 2023. — arXiv:2304.14997
- **[SOTA]** Syed, Rager, Conmy. *Attribution Patching Outperforms Automated Circuit Discovery.* 2023. — arXiv:2310.10348
- **[SOTA]** Marks, Rager, Michaud, Belinkov, Bau, Mueller. *Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models.* ICLR 2025. — arXiv:2403.19647
- **[Critique]** Miller, Chughtai, Saunders. *Transformer Circuit Faithfulness Metrics Are Not Robust.* COLM 2024. — arXiv:2407.08734
- **[Critique]** Makelov, Lange, Nanda. *Is This the Subspace You Are Looking For? An Interpretability Illusion for Subspace Activation Patching.* ICLR 2024. — arXiv:2311.17030
- **[Critique]** Zhang, Nanda. *Towards Best Practices of Activation Patching in Language Models: Metrics and Methods.* ICLR 2024. — arXiv:2309.16042
- **[Benchmark]** Gupta, Jenner, et al. *InterpBench: Semi-Synthetic Transformers for Evaluating Mechanistic Interpretability Techniques.* NeurIPS 2024 Datasets & Benchmarks.
- **[Benchmark]** Lindner, Kramár, Farquhar, Rahtz, McGrath, Mikulik. *Tracr: Compiled Transformers as a Laboratory for Interpretability.* NeurIPS 2023. — arXiv:2301.05062
- **[Survey]** Sharkey, Chughtai, et al. *Open Problems in Mechanistic Interpretability.* 2025. — arXiv:2501.16496

## 10. Worked Example

Take the IOI circuit, $|C| = 26$ heads in GPT-2 small.

**Cost of the exact sweep.** GPT-2 small is $1.24\times10^8$ parameters, so a forward pass costs $\approx 2P = 2.5\times10^8$ FLOPs per token. IOI prompts are ~15 tokens:
$$3.7\times10^9 \text{ FLOPs/prompt} \times 10^3 \text{ prompts} = 3.7\times10^{12} \text{ FLOPs per subset evaluation}.$$
Two evaluations per subset (full model, circuit) over $2^{26}$ subsets:
$$2 \times 6.7\times10^7 \times 3.7\times10^{12} \approx 5\times10^{20}\ \text{FLOPs}.$$
At $10^{14}$ effective FLOP/s on one A100, that is $5\times10^6$ s $\approx$ **58 GPU-days**, or under two days on 32 GPUs. The decisive experiment is affordable and has not been run.

**Where the obstruction becomes visible.** Now scale the same calculation. A feature-level circuit in a 7B model with $|C| = 100$ nodes needs $2^{100} \approx 1.3\times10^{30}$ subset evaluations, each $\sim$56× more expensive per prompt. That is $\sim10^{43}$ FLOPs — roughly $10^{17}$ times the compute used to train the model. No hardware trajectory closes that gap.

So the practical protocol samples, say, $10^3$ subsets and reports the max. The failure mode is arithmetic, not philosophical: sampling $10^3$ of $6.7\times10^7$ subsets covers $1.5\times10^{-5}$ of the space. If incompleteness is carried by a single 4-head subset — one backup name-mover configuration that the circuit reproduces and the full model does not — the chance of drawing it is $\sim10^{-5}$. The published circuit and a circuit missing a genuinely load-bearing component return the same sampled score.

That is the whole problem: completeness is a statement about a maximum over an exponential set, and every reported number is a lower bound whose gap to the true value is itself unmeasured, at any scale, in any model.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*