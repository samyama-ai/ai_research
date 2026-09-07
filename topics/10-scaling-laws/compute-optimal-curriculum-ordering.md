---
id: 10-scaling-laws/compute-optimal-curriculum-ordering
title: "Compute-Optimal Curriculum Ordering"
topic: 10-scaling-laws
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Compute-Optimal Curriculum Ordering

> **Topic:** Scaling Laws & Compute Allocation · **ID:** `10-scaling-laws/compute-optimal-curriculum-ordering` · **Status:** empirically-open

## 1. Problem Statement

Chinchilla-style scaling laws allocate a compute budget $C$ across two axes: parameters $N$ and tokens $D$. They say nothing about a third axis that costs nothing extra — **the order in which the $D$ tokens are presented**. The question:

- **Measurement variant.** Given a fixed token pool and a fixed $(N, D)$, how large is the gap between the best and worst orderings of the same data, measured as final loss or as compute-equivalent speedup? Is that gap a function of $C$?
- **Method variant.** Produce an ordering policy that is *provably compute-optimal in the ordering axis*: for each budget $C$, an order whose final loss no other order of the same pool beats by more than $\epsilon$.
- **Theory variant.** Prove or refute that the optimal ordering is budget-dependent — i.e. that the ordering that minimises loss at $10^{21}$ FLOPs differs from the one that minimises it at $10^{24}$ FLOPs, not just in magnitude of benefit but in the *ranking of policies*.

Solving it means: a predictive law $\hat{L}(N, D, \pi)$ that ranks orderings $\pi$ correctly at scales above where it was fit, validated by held-out extrapolation.

## 2. Formal Setting

Let $\mathcal{P} = \{(x_i, d_i)\}_{i=1}^{M}$ be a token pool, each document $x_i$ tagged with a domain/quality label $d_i \in \{1,\dots,K\}$. An **ordering policy** $\pi$ is a map from training step $t \in [T]$ to a mixture over domains,
$$\pi: [T] \to \Delta^{K-1}, \qquad \pi_t = (w_{t,1},\dots,w_{t,K}),$$
with the marginal constraint $\sum_t \pi_t / T = \bar{w}$ fixed. Fixing $\bar w$ is what separates *ordering* from *mixture selection*: two policies with identical $\bar w$ differ only in schedule. Batch $B_t$ is sampled i.i.d. from $\sum_k w_{t,k} \, \mathcal{D}_k$.

Compute is $C \approx 6ND$ (forward+backward FLOPs per token, Kaplan et al. 2020). Loss is measured on a held-out set $\mathcal{V}$ disjoint from $\mathcal{P}$ by document hash, reported as mean token NLL in nats:
$$L(N, D, \pi) = \frac{1}{|\mathcal{V}|}\sum_{(x)\in\mathcal{V}} -\log p_{\theta_T}(x).$$

Define the **curriculum advantage** against the shuffled control $\pi^{\text{iid}}_t \equiv \bar w$:
$$A(C) = L(N^*, D^*, \pi^{\text{iid}}) - \min_{\pi} L(N^*, D^*, \pi),$$
evaluated at the Chinchilla-optimal $(N^*, D^*)$ for that $C$. The practically useful conversion is the **compute-equivalent gain**: the factor $\rho$ such that $L(\rho C, \pi^{\text{iid}}) = L(C, \pi^\star)$, obtained by inverting a fitted $L(C) = E + A C^{-\alpha}$.

Assumptions, with the ones known to be violated marked:

1. $L$ is a deterministic function of $(N, D, \pi)$ — **violated**: seed variance at 1B scale is roughly $\pm 0.005$–$0.01$ nats, comparable to reported curriculum effects.
2. Order affects only the optimisation path, not the reachable loss floor — **untested**; loss-of-plasticity results suggest early data can permanently constrain later learning.
3. The order effect is separable from the learning-rate schedule — **violated**: WSD/annealing schedules make the final decay phase's data disproportionately influential (Hu et al. 2024), so $\pi$ and $\eta_t$ interact.
4. Fixed $\bar w$ makes the comparison order-only — **partially violated** under multi-epoch training, where repetition counts differ per domain (Muennighoff et al. 2023).
5. Held-out loss ranks policies the same way downstream benchmarks do — **violated**: domain upsampling can raise benchmark scores while raising held-out loss on the general distribution.

## 3. State of the Art

**Established (mixture, not ordering).** DoReMi (Xie et al., NeurIPS 2023) fits domain weights with a 280M proxy model via group-DRO and transfers them to 8B; reported ~2.6× fewer steps to reach baseline Pile perplexity. RegMix (Liu et al., ICLR 2025) fits a regression from mixture to loss over many small runs. Data Mixing Laws (Ye et al., 2024) fit an explicit functional form for loss vs. mixture. All three optimise a *stationary* $\bar w$; none isolates schedule.

**Claimed but unablated (ordering).** Two-phase pretraining — generic web early, curated/high-quality late — is now standard practice (MiniCPM's WSD decay phase, Hu et al. 2024; OLMo 2's Dolmino mid-training, 2025; Blakeney et al., COLM 2024 on domain upsampling in the last 10–20% of training). These runs report gains, but almost none holds $\bar w$ fixed, so the reported effect confounds *ordering* with *more high-quality data overall*. Blakeney et al. is the closest to a controlled comparison and is a small number of 7B runs, not a scaling sweep.

**Benchmark-number-only.** Rho-1 (Lin et al., NeurIPS 2024) selects tokens online by excess loss against a reference model; the 1B math result (matching much larger token budgets on GSM8K/MATH) is a benchmark number in a narrow domain, with no held-out-loss scaling curve.

**Negative result, treated as established.** Wu, Dyer & Neyshabur (ICLR 2021) swept curriculum, anti-curriculum, and random ordering across CIFAR-10/100 and ImageNet-scale vision training: no reliable benefit under standard budgets; benefit appeared only under short training time or label noise.

## 4. What Is Known

- **Order does not matter much at convergence in vision.** Wu et al. (2021), thousands of runs, ResNet-scale: curriculum vs. random ordering differences within noise for standard budgets; benefits appear at heavily truncated budgets (the regime where models are far from converged).
- **Mixture matters, and its optimum is budget-dependent.** Goyal et al. (CVPR 2024) showed for CLIP-style training that aggressive data filtering is optimal at small compute and harmful at large compute (repetition penalty dominates) — the same curated pool flips sign across roughly an order of magnitude in budget.
- **Late-training data placement has measurable effect.** Blakeney et al. (COLM 2024), 7B models on ~1T tokens: upsampling curated domains only in the final ~10–20% of tokens improved downstream averages by a few points versus uniform mixing, at equal total compute.
- **Repetition costs are quantified.** Muennighoff et al. (NeurIPS 2023): up to ~4 epochs of repeated data is nearly as good as fresh data; value decays to near-zero by ~16 epochs. This bounds how much reordering-with-repetition can buy.
- **Skill dependencies exist.** Skill-it! (Chen et al., NeurIPS 2023) showed measurable ordered prerequisite structure between skills at 125M–1.3B scale, and that exploiting it beats random sampling on those axes.
- **Seed noise floor.** At ~1B parameters, independent seeds differ by $\sim 0.005$–$0.01$ nats on held-out loss — the same order as most claimed curriculum gains.

## 5. What Is Not Known

- **Empirically open.** Whether $A(C)$ grows, shrinks, or vanishes with $C$ under a strict fixed-$\bar w$ control. The experiment is runnable today at $10^{19}$–$10^{22}$ FLOPs; nobody has published the sweep with matched marginals and multiple seeds.
- **Empirically open.** Whether ordering gains survive as *compute-equivalent* gains rather than fixed loss offsets. A constant $-0.01$ nat offset is worth progressively less as $C$ grows; a change in the exponent $\alpha$ would be worth a great deal. No published fit resolves offset vs. exponent for any curriculum.
- **Theoretically open.** No proof that the optimal $\pi$ is budget-dependent for any nontrivial model class. Convex-case results (SGD on convex objectives with i.i.d. sampling) give order-independent rates; the nonconvex case has no analogue.
- **Methodologically blocked.** There is no accepted definition of "difficulty" or "quality" that is model-independent. Difficulty measured by a reference model's loss is circular — the ordering depends on which reference model you picked, and no ground truth adjudicates.
- **Methodologically blocked.** Separating $\pi$ from the learning-rate schedule. Under WSD, moving data later is confounded with training it at lower $\eta$; nobody has a design that decouples the two.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability plus a noise floor near the effect size**. Any observed order effect decomposes into at least four terms: (a) true ordering effect, (b) change in effective per-domain repetition, (c) interaction with $\eta_t$, and (d) evaluation-set affinity to the late-phase domains. Published two-phase results vary all four at once. Isolating (a) requires holding $\bar w$, epoch counts per domain, and $\eta_t$ fixed — and then the residual effect at 1B scale is roughly the size of seed variance, so $k \geq 3$ seeds per arm are needed, multiplying cost by 3. A meaningful scaling sweep is 4 budgets × 3 policies × 3 seeds = 36 runs. At the top budget ($\sim 10^{22}$ FLOPs), that is a several-hundred-GPU-month program to answer one question with no product deliverable — which is why labs run one 7B two-phase ablation instead.

## 7. Current Research (as of 2026)

- **Mid-training as a named phase.** AI2 (OLMo 2 / Dolmino), and similar staged recipes elsewhere, treat the final 5–15% of tokens as a distinct curated stage. Ablations exist per-release but are not scaling sweeps.
- **Online data selection.** Follow-ons to Rho-1 and RHO-Loss (Mindermann et al., ICML 2022) that score tokens against a reference model during training; Stanford, Meta, and CMU groups active *(frontier — verify)*.
- **Mixture-schedule laws.** Extensions of Data Mixing Laws / RegMix to time-varying $w_t$ — fitting $L$ as a functional of the whole trajectory rather than the marginal *(frontier — verify)*.
- **Loss-of-plasticity work** (Dohare et al., *Nature* 2024, on continual backprop) supplies the mechanism by which early data could permanently constrain a network; connecting it to pretraining order is not yet done.

## 8. Concrete Next Experiment

**Scale.** Four compute budgets $C \in \{3\times10^{19}, 3\times10^{20}, 3\times10^{21}, 3\times10^{22}\}$ FLOPs, each at Chinchilla-optimal $(N^*, D^*)$: roughly 110M/2.2B, 340M/6.8B, 1.1B/22B, 3.5B/70B tokens.

**Arms** (all share identical $\bar w$, identical per-domain epoch counts, identical cosine LR schedule, 3 seeds each):
1. **Control:** $\pi^{\text{iid}}$ — fully shuffled.
2. **Curriculum:** easy→hard, difficulty = per-document NLL under a fixed 110M reference model trained once on the shuffled pool.
3. **Anti-curriculum:** the reverse order (guards against "any non-random order helps").
4. **Quality-late:** curated domains concentrated in the final 20% of steps, generic early, marginals matched by down-weighting curated data earlier.

**The deciding number.** Fit $L(C) = E + A\,C^{-\alpha}$ separately per arm on held-out loss. Report $\Delta\alpha = \alpha_{\text{best}} - \alpha_{\text{control}}$ with bootstrap CI over seeds. If the 95% CI for $\Delta\alpha$ excludes 0, ordering changes the scaling exponent and the problem is live. If $\Delta\alpha$'s CI contains 0 but the offset $\Delta E$ is significant, ordering is a fixed, vanishing-value bonus. If both CIs contain 0 across the full 3-decade range, the null (order-invariance at fixed marginals) stands and practice should stop attributing two-phase gains to ordering.

Cost estimate: total $\approx 4\times(3\times10^{19}+\dots) \times 4 \text{ arms} \times 3 \text{ seeds} \approx 4\times10^{23}$ FLOPs, dominated by the top budget — roughly 60–80 H100-months.

## 9. Key References

- **[Foundational]** Yoshua Bengio, Jérôme Louradour, Ronan Collobert, Jason Weston. *Curriculum Learning.* ICML, 2009.
- **[Foundational]** Jeffrey L. Elman. *Learning and development in neural networks: the importance of starting small.* Cognition, 1993.
- **[Foundational]** Jared Kaplan et al. *Scaling Laws for Neural Language Models.* 2020 — arXiv:2001.08361.
- **[Foundational]** Jordan Hoffmann et al. *Training Compute-Optimal Large Language Models.* NeurIPS, 2022 — arXiv:2203.15556.
- **[SOTA / negative result]** Xiaoxia Wu, Ethan Dyer, Behnam Neyshabur. *When Do Curricula Work?* ICLR, 2021 — arXiv:2012.03107.
- **[SOTA]** Sang Michael Xie et al. *DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining.* NeurIPS, 2023 — arXiv:2305.10429.
- **[SOTA]** Qian Liu et al. *RegMix: Data Mixture as Regression for Language Model Pre-training.* ICLR, 2025 — arXiv:2407.01492.
- **[SOTA]** Jiasheng Ye et al. *Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance.* 2024 — arXiv:2403.16952.
- **[SOTA]** Cody Blakeney et al. *Does Your Data Spark Joy? Performance Gains from Domain Upsampling at the End of Training.* COLM, 2024 — arXiv:2406.03476.
- **[SOTA]** Zhenghao Lin et al. *Rho-1: Not All Tokens Are What You Need.* NeurIPS, 2024 — arXiv:2404.07965.
- **[SOTA]** Sachin Goyal et al. *Scaling Laws for Data Filtering — Data Curation Cannot be Compute Agnostic.* CVPR, 2024 — arXiv:2404.07177.
- **[Related]** Niklas Muennighoff et al. *Scaling Data-Constrained Language Models.* NeurIPS, 2023 — arXiv:2305.16264.
- **[Related]** Mayee F. Chen et al. *Skill-it! A Data-Driven Skills Framework for Understanding and Training Language Models.* NeurIPS, 2023 — arXiv:2307.14430.
- **[Related]** Shengding Hu et al. *MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies.* 2024 — arXiv:2404.06395.
- **[Survey]** Alon Albalak et al. *A Survey on Data Selection for Language Models.* TMLR, 2024 — arXiv:2402.16827.
- **[Survey]** Petru Soviany, Radu Tudor Ionescu, Paolo Rota, Nicu Sebe. *Curriculum Learning: A Survey.* IJCV, 2022.

## 10. Worked Example

Take $C = 3\times10^{21}$ FLOPs: $N^* \approx 1.1$B, $D^* \approx 22$B tokens. Pool: 80% generic web, 20% curated (code, papers, textbooks), so $\bar w = (0.8, 0.2)$.

Run two arms with **identical marginals**:
- **Shuffled:** every batch is 80/20.
- **Quality-late:** first 80% of steps at 87.5/12.5, last 20% at 50/50. Check: $0.8(0.125) + 0.2(0.5) = 0.2$. Marginals match.

Suppose the observed held-out losses are $2.412$ (shuffled) and $2.398$ (quality-late), a gap of $0.014$ nats. Convert to compute-equivalence using a fitted $L(C) = 1.85 + 1.6\times10^{3} C^{-0.075}$: near $C = 3\times10^{21}$, $dL/d\ln C \approx -0.075 \times 0.562 \approx -0.042$ nats per e-fold. So $0.014$ nats $\approx \exp(0.014/0.042) = 1.40\times$ compute — a 40% saving. That is the headline someone would publish.

Now the obstruction. Three seeds per arm give per-arm standard deviations of about $0.008$ nats, so the standard error of the difference is $0.008\sqrt{2/3} \approx 0.0065$ and the gap is $2.2\sigma$ — barely significant, and only at one budget. Worse, the fitted quantity is an *offset*, not an exponent change. Carry the same $0.014$-nat offset to $C = 3\times10^{23}$: the local slope steepens in absolute terms only slowly, and $dL/d\ln C \approx -0.035$, so the same offset buys $\exp(0.014/0.035) = 1.49\times$ — but only if the offset persists. If instead the gap shrinks like $C^{-0.02}$ (entirely consistent with four noisy points), the offset at $3\times10^{23}$ is $0.008$ nats, worth $1.26\times$, and at $10^{25}$ it is inside the seed noise.

The visible obstruction: **a single-budget curriculum result cannot distinguish "40% compute saving forever" from "a bonus that decays to nothing two orders of magnitude up."** Only the multi-budget fit of $\Delta\alpha$ in §8 separates them, and that is the experiment nobody has run.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*