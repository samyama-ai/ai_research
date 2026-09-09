---
id: 20-interpretability/developmental-phase-change-detection
title: "Developmental Interpretability of Training Phase Changes"
topic: 20-interpretability
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Developmental Interpretability of Training Phase Changes

> **Topic:** Interpretability · **ID:** `20-interpretability/developmental-phase-change-detection` · **Status:** open

## 1. Problem Statement

Neural network training is not uniform. Loss curves that look smooth hide discrete events: a circuit forms, a representation reorganizes, a capability appears. **Developmental interpretability** asks whether these events can be *detected from the training trajectory alone* and then *explained mechanistically*.

Input: a checkpoint sequence $\{w_t\}_{t=0}^{T}$ from one training run, plus the data distribution and the optimizer state. Output: a set of times $\hat{S} = \{t_1, \dots, t_k\}$ marked as phase changes, each annotated with a structural claim (what changed inside the network) that survives an intervention test.

Three variants, of different difficulty:

- **Measurement.** Given $\{w_t\}$, produce $\hat S$ with a detector that does not read the behavior you are trying to explain. Solving means: the detector fires at the same steps under reseeding, and fires *before* or *at* the behavioral change, not after.
- **Method.** Given a detected $t_i$, localize the change to a subset of parameters or a circuit, and verify by ablation that the localized component causes the behavioral shift.
- **Theory.** Prove that the detected events correspond to a well-defined structural object — e.g. a change in the geometry of the loss landscape around $w_t$ — rather than to an artifact of the parameterization, the learning-rate schedule, or the detector's hyperparameters.

The measurement variant is the bottleneck. Everything downstream inherits its false-positive rate.

## 2. Formal Setting

Let $p(x)$ be the data distribution, $q_w(y \mid x)$ the model, and the population loss
$$L(w) = -\mathbb{E}_{x \sim p}\left[\log q_w(y \mid x)\right],$$
estimated in practice by $L_n(w)$ on a held-out set of $n$ tokens. All quantities below are functions of a checkpoint $w_t$ and are measured per checkpoint.

**Behavioral trajectory.** $B_t = (\ell_t^{(1)}, \dots, \ell_t^{(m)})$, per-task losses on $m$ probe tasks. Measured: run each checkpoint on a fixed held-out set, no fine-tuning. A *behavioral* phase change at $t^\ast$ is a local extremum of $\partial^2 \ell^{(j)}_t / \partial (\log t)^2$ exceeding a threshold — a bend in log-time, not a discontinuity, since no finite-step trajectory has one.

**Local learning coefficient (LLC).** From singular learning theory (Watanabe 2009), the asymptotic free energy of a singular model near $w^\ast$ is
$$F_n \approx n L_n(w^\ast) + \lambda(w^\ast)\log n + O(\log\log n),$$
where $\lambda(w^\ast)$ is the real log canonical threshold, an effective-dimension count that is $d/2$ only for regular models. Measured (Lau et al. 2023) by the SGLD estimator
$$\hat\lambda(w^\ast) = n\beta^\ast\left(\mathbb{E}_{w \sim \pi}\left[L_n(w)\right] - L_n(w^\ast)\right),$$
with $\pi \propto \exp\left(-n\beta^\ast L_n(w) - \tfrac{\gamma}{2}\lVert w - w^\ast\rVert^2\right)$ sampled by stochastic-gradient Langevin dynamics. Free hyperparameters: localization strength $\gamma$, inverse temperature $\beta^\ast$, step size $\epsilon$, chain length. The estimate is *not* invariant to these; it is a family of curves indexed by $(\gamma,\epsilon)$.

**Detector.** $D: \{w_t\} \to \hat S$. Candidates: sign changes in $d\hat\lambda_t/d\log t$; changepoints in the essential-dynamics PCA of $\{w_t\}$; changepoints in the per-head refined LLC $\hat\lambda_t^{(h)}$.

**Assumptions, and which fail.**
1. *Near-convergence.* The free-energy expansion assumes $w^\ast$ is a local minimum. **Violated** — checkpoints mid-training are not minima, and the SGLD chain drifts. Practice patches this with $\gamma$, which reintroduces hyperparameter dependence.
2. *Fixed distribution.* $p(x)$ is stationary. **Violated** in any run with curriculum, data-mixture change, or repeated epochs.
3. *Fixed optimization geometry.* Detectors assume the trajectory is intrinsic. **Violated** — learning-rate warmup and decay alone induce curvature in $\hat\lambda_t$ with no structural change.
4. *One run is representative.* **Violated** — seed variance in phase-change timing is large and rarely reported.

## 3. State of the Art

**Established (replicated, with ablations).**
- Induction heads: Olsson et al. (2022) show a visible bump in the loss derivative co-occurring with induction-head formation and with the onset of in-context learning, across models from 13M to 13B parameters. Backed by ablation, not just correlation.
- Grokking on modular arithmetic: Nanda et al. (ICLR 2023) reverse-engineer a one-layer transformer on addition mod 113 into a discrete Fourier / trig-identity algorithm, and define progress measures that separate three phases — memorization, circuit formation, cleanup. The circuit is confirmed by direct weight reading and ablation. This is the only case where a phase change is understood end to end.
- Circuit stability across scale: Tigges et al. (2024) track the indirect-object-identification circuit across Pythia checkpoints from 70M to 12B and find the algorithm emerges early and persists, while the specific heads implementing it churn.

**Claimed but not fully ablated.**
- LLC-based stage discovery. Hoogland et al. (2024) identify five developmental stages in a ~3M-parameter attention-only transformer learning linear regression in context, using $\hat\lambda_t$ plus essential dynamics. The stage boundaries are reported; a control arm showing the same detector is *silent* on a run with no structural change is not.
- Refined per-head LLC as a specialization signal (Wang et al., ICLR 2025) — attention-head differentiation in a 3M-parameter language model. Compelling visualizations; the causal link between $\hat\lambda^{(h)}$ changes and head function is correlational.
- LLC at scale (Furman & Lau, 2024): estimator scaled to billion-parameter Pythia models. What is established is *feasibility and internal consistency*, not that the resulting curve tracks structure.

**Benchmark-number-only.** Claims that "emergent abilities" occur at particular parameter counts (Wei et al., TMLR 2022) rest on downstream accuracy curves; Schaeffer et al. (NeurIPS 2023) show many vanish under continuous metrics. Those are metric artifacts, not established phase changes.

## 4. What Is Known

- The induction bump appears in a narrow window, roughly $2.5$–$5$ billion tokens, consistently across the Anthropic model sweep (13M–13B), and in-context-learning score jumps in the same window.
- Grokking on mod-113 addition: test loss falls by orders of magnitude around $\sim 10^4$ optimizer steps, thousands of steps after train loss saturates; restricted-loss and excluded-loss progress measures move *before* the test-loss drop.
- Masked LMs show a sudden drop in loss coincident with acquisition of syntactic attention structure (Chen et al., ICLR 2024), with the transition reproducible across seeds but varying in timing by a large multiple.
- $\hat\lambda$ is not $d/2$: in the toy model of superposition, measured LLC tracks the number of distinct feature-geometry configurations, and Bayesian-predicted transitions and SGD-observed transitions do *not* coincide (Chen et al., 2023). Scale: models with $d \le 10^3$.
- Pythia (Biderman et al., ICML 2023) provides 154 public checkpoints per model, 70M–12B — the standard substrate. Most developmental results at $>1$B parameters are descriptive.

## 5. What Is Not Known

- **Methodologically blocked.** There is no accepted definition of a phase change in a finite non-equilibrium trajectory. Every operational definition (loss-curve bend, LLC extremum, PCA changepoint) is threshold- and hyperparameter-dependent, and no two agree on a common test set. Until a detector has a published false-positive rate on a negative control, "we found five stages" is not falsifiable.
- **Theoretically open.** Whether SGD trajectory transitions correspond to Bayesian posterior phase transitions. Known to fail in at least one toy model; no theorem characterizing when they agree.
- **Empirically open.** Seed variance of phase-change *timing* at $\ge 1$B parameters. Runnable today with 8–16 seeded runs; nobody has published it. Also open: whether LLC-detected stages in $10^6$-parameter models have counterparts at $10^{10}$.
- **Empirically open.** Whether any detector fires *before* the behavioral change with enough lead time to be predictive rather than post-hoc.

## 6. Why It Is Hard

Three specific obstructions.

1. **Confounded measurement.** The learning-rate schedule injects its own curvature into every trajectory statistic. A cosine decay produces a smooth sweep in $\hat\lambda_t$ that is indistinguishable, without a control, from gradual structural consolidation. Nobody publishes the constant-LR control.
2. **Absent ground truth.** Outside modular arithmetic and induction heads, there is no independently known list of structural events to score a detector against. Precision and recall are uncomputable, so detectors are evaluated by whether their output looks interesting.
3. **Non-identifiability.** $\hat\lambda$ depends on $(\gamma, \beta^\ast, \epsilon, \text{chain length})$, and the dependence is not a scale factor — different settings reorder local extrema. Two labs can compute "the LLC" of the same checkpoint sequence and disagree on how many stages exist.

Compute is a secondary cost, not the binding constraint: $\hat\lambda$ at one checkpoint of a 1B model needs a few hundred SGLD steps, so 150 checkpoints × 8 seeds is a few thousand GPU-hours — affordable.

## 7. Current Research (as of 2026)

- **Timaeus** (Murfet, Lau, Hoogland, van Wingerden) — the LLC program: refined and per-component LLCs, essential dynamics, scaling the estimator to Pythia-class models.
- **Anthropic interpretability** — circuit formation over training as a downstream question of attribution-graph and crosscoder work; developmental claims mostly qualitative *(frontier — verify)*.
- **EleutherAI / Pythia ecosystem** — open checkpoint suites; the reproducible substrate everyone else uses.
- **Grokking / phase-transition theory** (academic groups incl. MIT, Northeastern, Tel Aviv) — quantization and multi-task-decomposition accounts of why capability curves look discrete (Michaud et al., NeurIPS 2023).
- Emerging *(frontier — verify)*: negative-control benchmarks for developmental detectors, and per-datapoint loss-trajectory clustering as a cheaper alternative to LLC.

## 8. Concrete Next Experiment

**Question.** Does any developmental detector have a false-positive rate below 20% on a run with no structural change?

**Scale.** Pythia-410M architecture, 8 seeds, 40B tokens of the Pile, checkpoints every 1000 steps (~140 per run). Estimated 3,000–5,000 A100-hours total.

**Arms.**
- *Treatment:* standard schedule, standard data.
- *Control arm 1 (null):* identical run, but the model is trained on a synthetic i.i.d. bigram corpus with a **constant** learning rate. The model can only learn one thing (the bigram table), monotonically. Any detected multi-stage structure here is a false positive.
- *Control arm 2 (schedule confound):* real data, constant LR, no warmup. Compare detected $\hat S$ against the cosine-schedule run.

**Detectors scored.** SGLD-LLC extrema (3 hyperparameter settings), essential-dynamics PCA changepoints, loss-curvature bends.

**The deciding number.** $\mathrm{FPR} = |\hat S_{\text{control-1}}| / |\hat S_{\text{treatment}}|$, plus the seed-agreement rate $A$ = fraction of treatment-arm detections that fall within $\pm 10\%$ of $\log t$ across at least 6 of 8 seeds. A detector is usable if $\mathrm{FPR} < 0.2$ and $A > 0.75$. Current published evidence supports no value for either.

## 9. Key References

- **[Foundational]** Sumio Watanabe. *Algebraic Geometry and Statistical Learning Theory.* Cambridge University Press, 2009.
- **[Foundational]** Catherine Olsson et al. *In-context Learning and Induction Heads.* Transformer Circuits Thread, Anthropic, 2022. — arXiv:2209.11895
- **[Foundational]** Neel Nanda, Lawrence Chan, Tom Lieberum, Jess Smith, Jacob Steinhardt. *Progress Measures for Grokking via Mechanistic Interpretability.* ICLR 2023. — arXiv:2301.05217
- **[Foundational]** Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, Vedant Misra. *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets.* 2022. — arXiv:2201.02177
- **[SOTA]** Edmund Lau, Zach Furman, George Wang, Daniel Murfet, Susan Wei. *The Local Learning Coefficient: A Singularity-Aware Complexity Measure.* 2023. — arXiv:2308.12108
- **[SOTA]** Jesse Hoogland, George Wang, Matthew Farrugia-Roberts, Liam Carroll, Susan Wei, Daniel Murfet. *The Developmental Landscape of In-Context Learning.* 2024. — arXiv:2402.02364
- **[SOTA]** George Wang, Jesse Hoogland, Stan van Wingerden, Zach Furman, Daniel Murfet. *Differentiation and Specialization of Attention Heads via the Refined Local Learning Coefficient.* ICLR 2025. — arXiv:2410.02984
- **[SOTA]** Zhongtian Chen, Edmund Lau, Jake Mendel, Susan Wei, Daniel Murfet. *Dynamical versus Bayesian Phase Transitions in a Toy Model of Superposition.* 2023. — arXiv:2310.06301
- **[SOTA]** Curt Tigges, Michael Hanna, Qinan Yu, Stella Biderman. *LLM Circuit Analyses Are Consistent Across Training and Scale.* NeurIPS 2024. — arXiv:2407.10827
- **[Empirical]** Angelica Chen, Ravid Shwartz-Ziv, Kyunghyun Cho, Matthew L. Leavitt, Naomi Saphra. *Sudden Drops in the Loss: Syntax Acquisition, Phase Transitions, and Simplicity Bias in MLMs.* ICLR 2024. — arXiv:2309.07311
- **[Empirical]** Stella Biderman et al. *Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling.* ICML 2023. — arXiv:2304.01373
- **[Counterpoint]** Rylan Schaeffer, Brando Miranda, Sanmi Koyejo. *Are Emergent Abilities of Large Language Models a Mirage?* NeurIPS 2023. — arXiv:2305.15891
- **[Survey]** Eric J. Michaud, Ziming Liu, Uzay Girit, Max Tegmark. *The Quantization Model of Neural Scaling.* NeurIPS 2023. — arXiv:2303.13506

## 10. Worked Example

Take the one-layer transformer on addition mod 113 (Nanda et al.), where ground truth exists: the trained network computes $a+b \bmod 113$ by mapping inputs to frequencies $\omega_k$, forming $\cos(\omega_k(a+b))$ via trig identities, and reading off the argmax. Three structural events are *known*: circuit formation, memorization decay, cleanup.

Now run a detector on this trajectory and ask what it would have told you without the ground truth.

- Train loss reaches $\sim 10^{-5}$ by step $\sim 10^3$. Test loss stays near chance until $\sim 10^4$ steps, then collapses. On the loss curve alone, a bend-detector reports **one** event, at the test-loss drop.
- Restricted loss (ablating all but the key Fourier frequencies) starts falling around step $2\times10^3$ — thousands of steps *earlier*. So the true circuit-formation event is invisible to the loss-curve detector.
- Run the SGLD-LLC estimator across the same checkpoints. With localization $\gamma$ large, the chain barely moves, $\hat\lambda_t$ is nearly flat, and **zero** events are reported. With $\gamma$ small, the chain escapes the basin, $\hat\lambda_t$ is dominated by SGLD noise, and a naive extremum-finder reports **six to ten** events, most of them at random steps. There is an intermediate band where the curve shows a rise-then-fall matching formation and cleanup — but that band was located by knowing the answer.

That is the obstruction in one instance. On the only problem where the ground truth is known, the detector's output is a function of a hyperparameter chosen after seeing the ground truth, and the loss curve — the thing everyone actually looks at — under-reports the number of structural events by a factor of at least two. Scale this to a 12B-parameter model where no ground truth exists, and there is no way to tell which of the two failure modes you are in. Section 8's control arm is the cheapest way to find out.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*