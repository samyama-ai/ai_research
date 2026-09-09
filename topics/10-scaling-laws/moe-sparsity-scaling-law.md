---
id: 10-scaling-laws/moe-sparsity-scaling-law
title: "Scaling Laws for Mixture-of-Experts Sparsity"
topic: 10-scaling-laws
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Scaling Laws for Mixture-of-Experts Sparsity

> **Topic:** Scaling Laws & Compute Allocation · **ID:** `10-scaling-laws/moe-sparsity-scaling-law` · **Status:** empirically-open

## 1. Problem Statement

Given a training compute budget $C$, a dense transformer has two free variables: parameters $N$ and tokens $D$, tied by $C \approx 6ND$. Chinchilla fixes their ratio. A Mixture-of-Experts (MoE) model adds at least two more: how many parameters are *active* per token, and how the inactive ones are partitioned into experts. The problem is to give the loss as a function of all of them, and to invert it into an allocation rule.

Three variants, with different difficulty:

- **Measurement.** Fit $L(N_{\text{tot}}, N_{\text{act}}, D, G)$ — total parameters, active parameters, tokens, expert granularity — over a grid wide enough that the sparsity exponent is identified, not aliased onto the parameter exponent. Empirically open.
- **Method.** Produce a rule "at budget $C$ and deployment constraint $R$ (memory, or tokens/sec), use sparsity $S^\star(C,R)$" that transfers across routers, data mixtures, and post-training. Empirically open, and confounded by the fact that different papers hold different things fixed.
- **Theory.** Explain *why* loss should depend on $N_{\text{act}}$ and $N_{\text{tot}}$ through a particular functional form. No derivation exists from any data-distribution model. Theoretically open.

A solution is a law that predicts held-out loss within its own fit error at a scale $\geq 10\times$ beyond the fitting grid, and whose optimal-sparsity prediction is confirmed by a model trained at that scale.

## 2. Formal Setting

A sparse layer has $E$ experts, top-$k$ routing, and expert hidden width $d_{\text{ff}}/G$ where $G$ is **granularity** (number of experts a dense FFN is split into; $G=1$ is standard, $G>1$ is fine-grained). Measured quantities:

- $N_{\text{tot}}$ — all trainable parameters, counted from the checkpoint.
- $N_{\text{act}}$ — parameters touched per token, counted analytically: embeddings + attention + $k/E \cdot$ (expert parameters). Router parameters are $O(d\cdot E)$ and usually dropped.
- **Sparsity** $S = 1 - N_{\text{act}}/N_{\text{tot}} \in [0,1)$. Note $S$ is *not* $1-k/E$: attention and embeddings are always active, so $S$ saturates below 1 and depends on depth and vocabulary size. Papers that report $E$ instead of $S$ are not comparable across architectures.
- **Training compute** $C = 6 N_{\text{act}} D$ + routing/all-to-all overhead. The overhead is a systems quantity, not a FLOP count, and is the main place accounting diverges between papers. Hardware-measured $C$ (chip-hours) and analytic $C$ differ by 1.2–2$\times$ for MoE at $E \geq 64$.
- $L$ — cross-entropy in nats/token on a held-out split of the *training* distribution.

The candidate form, following Clark et al. (2022) and Krajewski et al. (2024):

$$L(N_{\text{act}}, N_{\text{tot}}, D, G) = c + \frac{a}{N_{\text{act}}^{\alpha}} + \frac{b}{D^{\beta}} + \frac{g}{(N_{\text{tot}}/N_{\text{act}})^{\gamma} G^{\delta}} + \text{(interaction terms)}$$

with the practically important question being whether $\gamma > 0$ is scale-independent or decays, i.e. whether $\partial^2 L / \partial \log N_{\text{act}} \, \partial \log S$ vanishes.

Assumptions, and which fail:

1. *Loss is a smooth function of parameter counts.* Fails at expert-count boundaries where capacity factor causes token dropping — a discontinuity, not a smooth term.
2. *Routing is at its optimum.* Violated: load-balancing losses trade loss against balance, and the trade weight $\alpha_{\text{aux}}$ is a hyperparameter almost never re-tuned per sparsity level.
3. *Optimal hyperparameters transfer across $S$.* Known violated — optimal learning rate and batch size shift with $N_{\text{act}}$ *and* with $E$ independently.
4. *Upstream loss orders downstream capability.* Violated for MoE specifically: sparse models with matched loss are repeatedly reported worse at reasoning-heavy fine-tuning (ST-MoE, Zoph et al. 2022).

## 3. State of the Art

**Established.**
- *Unified Scaling Laws for Routed Language Models* (Clark et al., ICML 2022) fits a bilinear law in $\log N$ and $\log E$ over 269 models to $\sim$900M dense-equivalent parameters and $E \le 512$, across three routers (Sinkhorn, hash, RL). Result: the MoE advantage shrinks with scale and the fit extrapolates to zero benefit near $\sim$1.3B. Established *as a fit on its grid*.
- *Scaling Laws for Fine-Grained Mixture of Experts* (Krajewski et al., ICML 2024) adds $G$ and shows Clark's vanishing-gain conclusion is an artifact of fixing $G=1$ and holding tokens fixed rather than compute-optimal. With $G$ free, the MoE-over-dense gain does not close on their grid.
- *Training Compute-Optimal LLMs* (Hoffmann et al., NeurIPS 2022) supplies the dense control that all MoE comparisons are measured against.

**Claimed but unablated.**
- "Optimal sparsity increases with compute budget" (Abnar et al., ICML 2025, *Parameters vs FLOPs*). The direction is plausible and consistent with several groups; the claim is unablated against router choice and against re-tuned learning rates at each sparsity.
- Fine-grained + shared-expert designs (DeepSeekMoE, Dai et al., ACL 2024) are justified by ablations at 2B and 16B, not by a fitted law.

**Benchmark numbers only, not laws.** Switch Transformer's "7$\times$ speedup over T5-Base"; GLaM matching GPT-3 quality at $\sim$1/3 training energy (1.2T total / 97B active); Mixtral 8x7B (46.7B total / 12.9B active) matching Llama-2 70B on several suites. Each is one point, with one router and one data mixture — none constrains an exponent.

## 4. What Is Known

- Sparsity buys loss at fixed FLOPs. Switch-Base reaches T5-Base's pretraining loss in $\sim$1/7 the steps at 128 experts (Fedus et al., JMLR 2022), measured at $\sim$7B total parameters.
- The gain is sublinear and saturating in $E$ at fixed granularity: Clark et al. measure most of the benefit by $E \approx 64$ at $\le$900M scale.
- Granularity matters and is separately identified: Krajewski et al. report best $G$ in the range 4–16 across budgets up to $\sim$10$^{20}$ FLOPs, with $G=1$ strictly suboptimal.
- Fine-grained + shared experts: DeepSeekMoE 16B (2.8B active) roughly matches LLaMA2-7B on standard benchmarks — a $\sim$2.5$\times$ active-FLOP reduction at matched quality.
- Sparse models are more sensitive to fine-tuning than dense ones at matched pretraining loss; ST-MoE (Zoph et al., 2022) documents overfitting on small SuperGLUE tasks at 269B total parameters and prescribes dropout and selective updates as mitigation.
- Auxiliary-loss-free balancing (DeepSeek, 2024) removes one confound: balance can be enforced by bias updates rather than a gradient term, so the loss/balance trade weight need not be tuned.

## 5. What Is Not Known

- **Empirically open (dominant).** No public fit spans $\geq 3$ decades of compute *with $S$, $G$, and $D$ all varied and hyperparameters re-tuned per cell*. Every existing grid is small in at least one axis. Consequence: $\gamma$ and $\delta$ are known only jointly with the grid's aliasing.
- **Empirically open.** Whether $S^\star$ keeps rising with $C$ or turns over. Both Clark (turnover) and Abnar (monotone rise) are consistent with their own data.
- **Theoretically open.** No derivation of why $N_{\text{tot}}$ should enter as a power law at all. Frantar et al. (ICLR 2024) give the parallel result for *weight* sparsity — a multiplicative capacity term — but no generative model predicts either exponent.
- **Methodologically blocked.** "Optimal sparsity" is undefined without stating the deployment constraint. Under a memory cap, a FLOP cap, and a tokens/sec cap the optima differ by more than the effect being measured. Papers frequently do not state which.
- **Methodologically blocked.** Downstream-capability scaling for MoE. There is no measurement protocol that separates "sparse models have less depth of computation per token" from "sparse models were fine-tuned with dense hyperparameters".

## 6. Why It Is Hard

The specific obstruction is **non-identifiability under compute cost**. $N_{\text{act}}$, $N_{\text{tot}}$ and $D$ are three axes; a compute-matched sweep only affords $O(10)$ points per decade, so the design matrix is near-collinear and the fitted $\alpha$ and $\gamma$ trade off almost freely. Clark's and Krajewski's opposite conclusions are the same data-shape problem, not a disagreement about facts — one held $G$ and $D$ fixed, and the sparsity exponent absorbed the difference.

Compounding it: hyperparameter non-transfer means the "effect of sparsity" measured on a grid with fixed learning rate is partly the effect of a mistuned learning rate, and that mistuning grows with $E$. And $C$ itself is not agreed on — analytic FLOPs ignore all-to-all, so two labs can report opposite sparsity optima from identical checkpoints.

## 7. Current Research (as of 2026)

- Joint memory/FLOP laws — treating total parameters as a priced resource rather than free (Ludziejewski et al., 2025, on memory-efficient MoE scaling). *(frontier — verify)*
- Inference-aware allocation: optimising over training *and* serving cost jointly, which moves $S^\star$ down when serving dominates.
- Router-invariance studies: showing exponents are stable across sigmoid/softmax gating and loss-free balancing (DeepSeek, Qwen, Moonshot lines). *(frontier — verify)*
- Fine-grained + shared-expert as default (DeepSeek-V3, Qwen3-MoE, OLMoE), pushing $G$ high and $k/E$ low; whether the law's $\delta$ is stable at $E>256$ is untested publicly.
- Muon/second-order optimisers changing the effective $D$-exponent, which would shift MoE optima. *(frontier — verify)*

## 8. Concrete Next Experiment

**Question decided:** is $\partial S^\star / \partial \log C > 0$, holding the deployment constraint fixed?

- **Scale.** Four compute decades: $C \in \{3\times10^{18}, 10^{19}, 3\times10^{19}, 10^{20}, 3\times10^{20}\}$ analytic FLOPs. At each $C$, five sparsity levels $S \in \{0, 0.5, 0.75, 0.875, 0.9375\}$ at fixed $G=8$, top-$k$ with $k/E$ fixed, loss-free balancing, Chinchilla-optimal $D$ recomputed from $N_{\text{act}}$. 25 runs; largest $\sim$3$\times10^{20}$ FLOPs, about 2k H100-hours. Total budget $\approx$ 15–20k H100-hours.
- **Control arm.** The $S=0$ dense column at every $C$, trained with the identical pipeline, and — critically — a **re-tuned** arm: at $C = 10^{19}$ and $C = 10^{20}$, sweep learning rate over $\{0.5,1,2\}\times$ per $(C,S)$ cell. If the sparsity optimum moves when the LR is re-tuned, the uncontrolled grid is measuring optimiser mistuning.
- **Deciding number.** Fit $S^\star(C)$ by quadratic interpolation of held-out loss over $S$ at each $C$, then report the slope
  $$\hat{\theta} = \frac{d\,\text{logit}(S^\star)}{d \log_{10} C}$$
  with a bootstrap CI over seeds. **$\hat\theta$'s 95% CI excluding 0 from above** settles the direction; a CI containing 0 across four decades falsifies the "sparsity should keep rising" prescription as currently stated. Secondary readout: $\hat\theta$ from the re-tuned arm minus $\hat\theta$ from the fixed-LR arm — if that difference exceeds the CI width, the whole literature's grids are confounded.

## 9. Key References

- **[Foundational]** Shazeer, Mirhoseini, Maziarz, Davis, Le, Hinton, Dean. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* ICLR 2017. — arXiv:1701.06538
- **[Foundational]** Kaplan, McCandlish, Henighan, et al. *Scaling Laws for Neural Language Models.* 2020. — arXiv:2001.08361
- **[Foundational]** Hoffmann, Borgeaud, Mensch, et al. *Training Compute-Optimal Large Language Models.* NeurIPS 2022. — arXiv:2203.15556
- **[SOTA]** Clark, de las Casas, Guy, et al. *Unified Scaling Laws for Routed Language Models.* ICML 2022. — arXiv:2202.01169
- **[SOTA]** Krajewski, Ludziejewski, Adamczewski, et al. *Scaling Laws for Fine-Grained Mixture of Experts.* ICML 2024. — arXiv:2402.07871
- **[SOTA]** Abnar, Shah, Busbridge, et al. *Parameters vs FLOPs: Scaling Laws for Optimal Sparsity for Mixture-of-Experts Language Models.* ICML 2025.
- **[SOTA]** Frantar, Riquelme, Houlsby, Alistarh, Evci. *Scaling Laws for Sparsely-Connected Foundation Models.* ICLR 2024. — arXiv:2309.08520
- **[Systems]** Fedus, Zoph, Shazeer. *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.* JMLR 2022. — arXiv:2101.03961
- **[Systems]** Lepikhin, Lee, Xu, et al. *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding.* ICLR 2021. — arXiv:2006.16668
- **[Systems]** Du, Huang, Dai, et al. *GLaM: Efficient Scaling of Language Models with Mixture-of-Experts.* ICML 2022. — arXiv:2112.06905
- **[Systems]** Zoph, Bello, Kumar, et al. *ST-MoE: Designing Stable and Transferable Sparse Expert Models.* 2022. — arXiv:2202.08906
- **[Systems]** Dai, Deng, Zhao, et al. *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models.* ACL 2024. — arXiv:2401.06066
- **[Open model]** Muennighoff, Soldaini, Groeneveld, et al. *OLMoE: Open Mixture-of-Experts Language Models.* ICLR 2025. — arXiv:2409.02060

## 10. Worked Example

Take $C = 10^{20}$ FLOPs and ask for the compute-optimal configuration under two readings of the literature.

**Dense control.** Chinchilla: $N \approx \sqrt{C/(6\cdot 20)} \cdot \sqrt{20} $ — concretely $N_{\text{act}} \approx 1.0\times10^{9}$, $D \approx 1.7\times10^{10}$ tokens, since $6ND = 6(10^9)(1.7\times10^{10}) \approx 1.0\times10^{20}$.

**Reading A (Clark et al., extrapolated).** The MoE-over-dense gain closes near $\sim$1.3B dense-equivalent. At $N_{\text{act}} = 10^9$ we are at the crossover, so the prescription is $E$ small — say $E=8$, $S \approx 0.7$ — and the expected loss gain over dense is under 0.01 nats.

**Reading B (Krajewski et al., $G$ free).** With $G=8$ the gain does not close. Prescription: $E = 64$, $k=8$, $S \approx 0.93$, so $N_{\text{tot}} \approx 1.4\times10^{10}$ with $N_{\text{act}}$ still $10^9$, and an expected gain of order 0.05–0.10 nats.

The two prescriptions differ by **14$\times$ in total parameters** — a serving-memory difference between one accelerator and eight — at the same budget, from the same public evidence. Now make the obstruction visible: both readings fit their own grids with $R^2 > 0.99$. Clark's grid tops out near 900M with $G=1$; Krajewski's varies $G$ but tops out near $10^{20}$ FLOPs. At $C=10^{20}$ the models' predicted losses differ by $\sim$0.08 nats, while the *fit* residual of each on its own grid is $\sim$0.01 nats and the seed-to-seed noise on a single 1B run is $\sim$0.005 nats. So the disagreement is 8–16$\times$ larger than either paper's stated uncertainty — the error bars are conditional on a functional form, and the forms disagree in the region where nobody has data. That is what "empirically open" means here: not that the effect is small, but that the design matrix never separated $\log N_{\text{act}}$ from $\log S$, and the extrapolation inherits the whole ambiguity.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*