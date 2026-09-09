---
id: 15-mixture-of-experts/moe-dense-frontier-crossing
title: "MoE Versus Dense Compute-Optimal Frontier Crossing"
topic: 15-mixture-of-experts
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# MoE Versus Dense Compute-Optimal Frontier Crossing

> **Topic:** Mixture of Experts · **ID:** `15-mixture-of-experts/moe-dense-frontier-crossing` · **Status:** empirically-open

## 1. Problem Statement

Sparse mixture-of-experts (MoE) models beat dense models at equal training FLOPs across every scale anyone has measured. The open question is whether that advantage **persists indefinitely** or **crosses over** — whether there exists a finite compute budget $C^\ast$ beyond which the best dense model reaches lower loss than the best MoE model at the same budget.

Three variants, with different difficulty:

- **Measurement.** Given a fixed training-compute budget $C$, is $\min_{\text{MoE}} L(C) < \min_{\text{dense}} L(C)$ for all $C$, or does the sign flip? Requires both arms optimized independently, not one arm tuned and the other inherited.
- **Method.** Does the answer depend on architecture choices inside "MoE" — expert granularity, shared experts, router type — such that a crossing observed for one MoE family is not a crossing for the class?
- **Theory.** Is there a functional form for $L(N_{\text{total}}, N_{\text{active}}, D)$ whose fitted parameters *imply* a crossing, and are the parameters identifiable from data at attainable scales?

A solution is a scaling law fit over a compute range spanning $\geq 3$ decades in which the MoE-versus-dense loss gap, extrapolated, either (a) has a confidently positive lower bound at $C \to \infty$, or (b) crosses zero at an identified $C^\ast$ with a confidence interval narrower than one decade.

## 2. Formal Setting

Let a model be described by:

- $N_{\text{total}}$ — all trainable non-embedding parameters, counted directly from the checkpoint.
- $N_{\text{active}}$ — parameters touched per token: dense layers plus $k$ of $E$ experts. Measured, not nominal; a top-2 router with a dropped token has lower realized $N_{\text{active}}$.
- $D$ — training tokens seen (sequence count $\times$ context length, counting repeats).
- $C$ — training compute. The standard proxy is $C \approx 6 N_{\text{active}} D$ FLOPs; the honest measurement is wall-clock GPU-seconds $\times$ achieved FLOP/s, which for MoE is 20–50% below the proxy because of all-to-all communication and expert-capacity padding.
- $S = 1 - N_{\text{active}}/N_{\text{total}}$ — sparsity. $G$ — granularity, the number of expert slices per equivalent dense FFN (Krajewski et al., 2024).
- $L$ — cross-entropy in nats/token on a held-out set drawn from the *training* distribution, evaluated with identical tokenizer and identical held-out shards across arms.

The compute-optimal frontier for a family $\mathcal{F}$:

$$L^\ast_{\mathcal{F}}(C) \;=\; \min_{\theta \in \mathcal{F},\; 6N_{\text{active}}D \le C} L(\theta, D)$$

and the crossing predicate: $\exists\, C^\ast$ such that $L^\ast_{\text{MoE}}(C) > L^\ast_{\text{dense}}(C)$ for all $C > C^\ast$.

Clark et al. (2022) fit a trilinear law at fixed $D$:

$$\log L \;=\; a \log N + b \log \hat{E} + c \log N \log \hat{E} + d, \qquad \hat{E} = \min(E, E_{\max})$$

The crossing is entirely carried by the interaction term $c$: $c < 0$ in their fit means the benefit of experts shrinks as $N$ grows, and $\partial \log L / \partial \log \hat{E} = b + c\log N = 0$ locates $C^\ast$.

Assumptions, with the violated ones flagged:

1. $C = 6N_{\text{active}}D$ — **violated**: ignores attention, router, and all-to-all cost; the gap widens with $E$.
2. $D$ held fixed while $N$ varies — **violated as a modeling choice**: Chinchilla-optimal $D$ grows with $C$, so a law fit at fixed $D$ confounds the interaction term with over-training of the large models.
3. Both arms independently hyperparameter-optimized — **usually violated**: MoE runs typically inherit the dense learning rate and batch schedule.
4. Loss is the objective — **violated in use**: downstream capability, not held-out perplexity, is what the deployment cares about, and the MoE-dense loss-to-benchmark map is not the same function for both families.

## 3. State of the Art

**Theory SOTA.** No proof in either direction. The only law that predicts a crossing is Clark et al., *Unified Scaling Laws for Routed Language Models* (ICML 2022), whose fit places the vanishing point at roughly $9 \times 10^{11}$ dense parameters. This is *established as a fit* and *unestablished as a prediction*: it was fit at a fixed token budget, and the extrapolation is one to two decades beyond the largest model in the fit.

**Empirical SOTA (rebuttal side).** Krajewski et al., *Scaling Laws for Fine-Grained Mixture of Experts* (2024), refit with granularity $G$ as a free variable and with $D$ scaled with compute, and report that the Clark crossing disappears — the MoE advantage grows rather than shrinks, provided $G$ is tuned. Established: the granularity axis matters and the fixed-$D$ protocol biases the interaction term. Claimed but unablated: that no crossing exists at any $C$; their fit also extrapolates.

**Empirical SOTA (sparsity side).** Abnar et al. (Apple, 2025) report that compute-optimal sparsity *increases* with budget — bigger budgets prefer sparser models — which is the opposite sign to Clark. Established at their fitted range; the trend's continuation is extrapolation.

**Benchmark-only results.** Mixtral 8x7B (47B total / 13B active) matching or beating Llama 2 70B, DeepSeek-V3 (671B total / 37B active, 14.8T tokens, ~2.79M H800 GPU-hours), and OLMoE-1B-7B (6.9B total / 1.3B active, 5T tokens) are *benchmark numbers*, not frontier measurements: neither arm is compute-optimal and the dense comparator was trained by a different group on different data.

## 4. What Is Known

- **The gap is real and large at small-to-mid scale.** Switch Transformer (Fedus et al., JMLR 2022) reports ~7$\times$ pre-training speedup over T5-Base at matched FLOPs, at the sub-1B active-parameter scale, 32–2048 experts.
- **The gap narrows with $N$ under fixed $D$.** Clark et al. measured 15M–1.3B dense-equivalent models, up to 512 experts, at a fixed ~130B-token budget; the routed-vs-dense gap shrinks monotonically in $N$ over that range. This narrowing is reproduced; its *cause* is disputed.
- **Granularity is a first-order axis.** Fine-grained experts with a shared expert (DeepSeekMoE, Dai et al., 2024) beat coarse top-2 MoE at matched active parameters at the 2B and 16B total-parameter scale.
- **Chinchilla token scaling ($D/N \approx 20$) is dense-fit** (Hoffmann et al., 2022) and is not the MoE optimum; MoE optima sit at higher $D/N_{\text{active}}$.
- **MoE is memory-bound at inference.** At a fixed *serving* memory budget rather than a training-FLOP budget, dense wins far earlier — this is a different frontier and is not in dispute.

## 5. What Is Not Known

- **Theoretically open.** No result derives the sign of the interaction term $c$ from any model of representation capacity. There is no argument that sparse conditional computation must, or must not, saturate.
- **Empirically open (the main gap).** Nobody has run both arms, independently hyperparameter-tuned, along their own Chinchilla-style token-scaling ladders, over $\geq 3$ decades of compute reaching $\geq 10^{23}$ FLOPs. The experiment is runnable — it costs perhaps $10^4$–$10^5$ GPU-hours for a credible ladder — and has not been run publicly.
- **Methodologically blocked.** "Equal compute" is not well defined for MoE. FLOP-matching, wall-clock-matching, and dollar-matching give different answers and can order the arms differently. Until the catalog fixes one, the crossing question has no single truth value.

## 6. Why It Is Hard

The obstruction is **non-identifiability of the interaction term under attainable compute**. The crossing is determined by $c$ — the curvature of the loss surface in $(\log N, \log E)$ — which is a second-order effect on a quantity that moves by ~0.02 nats between arms at the scales anyone can afford. Fitting $c$ to a precision that locates $C^\ast$ within one decade requires either a longer compute lever than exists in any public run, or a seed-noise floor below the effect size. Run-to-run loss variance at 1B scale is roughly 0.005–0.01 nats; the gap being extrapolated is a few multiples of that.

Compounding it: **the measurement is confounded by protocol**. Fixed-$D$ versus compute-scaled-$D$ flips the sign of the reported trend (Clark vs Krajewski) using overlapping architectures. The disagreement in the literature is not about data, it is about which held-fixed variable makes the comparison fair — and that choice is not empirically determined.

## 7. Current Research (as of 2026)

- **Joint sparsity–granularity–token laws.** Follow-ups to Krajewski/Ludziejewski (IDEAS NCBR / University of Warsaw) fit $L(N_{\text{total}}, N_{\text{active}}, D, G)$ jointly, including memory-cost terms *(frontier — verify)*.
- **Optimal-sparsity trends** (Apple) — whether $S^\ast(C)$ is monotone increasing, and where it asymptotes.
- **Inference-aware frontiers** — reframing the objective as total lifetime cost (training + serving), where the crossing is much closer and better posed.
- **Upcycling laws** — dense→MoE conversion, which makes the two families non-independent and muddies "which family is better" as a question *(frontier — verify)*.
- Frontier-lab internal ladders almost certainly exist at $\geq 10^{24}$ FLOPs; none are published with both arms.

## 8. Concrete Next Experiment

**Scale.** Six compute budgets, log-spaced from $3\times10^{19}$ to $3\times10^{22}$ FLOPs (3 decades). At each budget, sweep $(N_{\text{active}}, D)$ over 5 points to find each family's own optimum — do not transfer the dense optimum.

**Arms.**
- *Control:* dense decoder-only Transformer, per-budget $\mu$P-transferred learning rate, its own $D/N$ optimum.
- *Treatment:* fine-grained MoE, granularity $G \in \{1, 4, 16\}$, one shared expert, sparsity re-optimized per budget, its own independently tuned LR and batch size.
- Identical data (a fixed 500B-token corpus, no repeats within a run), tokenizer, and held-out shards. Three seeds at the two smallest budgets to establish the noise floor.

**Deciding number.** Fit $\Delta(C) = L^\ast_{\text{dense}}(C) - L^\ast_{\text{MoE}}(C)$ as a power law $\Delta(C) = \alpha C^{-\beta} + \gamma$ and report $\hat{\gamma}$ with a bootstrap 95% CI.

- $\hat{\gamma} > 0$ with CI excluding 0 → no crossing; the MoE advantage is asymptotically bounded below.
- $\hat{\gamma} < 0$ with CI excluding 0 → crossing exists; report $C^\ast$ where $\hat{\Delta} = 0$.
- CI straddling 0 → the question is not decidable at 3 decades, which is itself the publishable result and sets the lever length needed.

## 9. Key References

- **[Foundational]** Shazeer et al. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* ICLR 2017. — arXiv:1701.06538
- **[Foundational]** Lepikhin et al. *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding.* ICLR 2021. — arXiv:2006.16668
- **[Foundational]** Fedus, Zoph, Shazeer. *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.* JMLR 23, 2022. — arXiv:2101.03961
- **[SOTA / crossing claim]** Clark et al. *Unified Scaling Laws for Routed Language Models.* ICML 2022. — arXiv:2202.01169
- **[SOTA / rebuttal]** Krajewski, Ludziejewski et al. *Scaling Laws for Fine-Grained Mixture of Experts.* 2024. — arXiv:2402.07871
- **[SOTA]** Abnar et al. *Parameters vs FLOPs: Scaling Laws for Optimal Sparsity for Mixture-of-Experts Language Models.* Apple, 2025.
- **[Baseline law]** Hoffmann et al. *Training Compute-Optimal Large Language Models.* NeurIPS 2022. — arXiv:2203.15556
- **[Architecture]** Dai et al. *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models.* ACL 2024. — arXiv:2401.06066
- **[Open artifact]** Muennighoff et al. *OLMoE: Open Mixture-of-Experts Language Models.* 2024. — arXiv:2409.02060
- **[Survey]** Cai et al. *A Survey on Mixture of Experts in Large Language Models.* IEEE TKDE, 2025.

## 10. Worked Example

Take Clark et al.'s fit at face value and ask what it says about a real model.

Their law implies the routing benefit vanishes near $N \approx 9\times10^{11}$ dense parameters. DeepSeek-V3 has $N_{\text{total}} = 671$B and $N_{\text{active}} = 37$B — right at the predicted crossing in *total* parameters. If the law's variable is total parameters, DeepSeek-V3 should be roughly matched by a dense model of comparable total size. But the compute-matched dense comparator is set by $N_{\text{active}}$, not $N_{\text{total}}$: at $6 N_{\text{active}} D$ with $D = 14.8$T, DeepSeek-V3's budget is

$$C \approx 6 \times 3.7\times10^{10} \times 1.48\times10^{13} \approx 3.3\times10^{24}\ \text{FLOPs},$$

which buys a Chinchilla-optimal dense model of about $\sqrt{C/6/20} \approx 1.7\times10^{11}$ — roughly 170B parameters on 3.4T tokens. Nobody has trained that control.

Now the obstruction. Plug $N = 1.7\times10^{11}$ into the two competing fits. Clark's says the routed advantage at that scale is within noise of zero. Krajewski's, refit with $G$ free and $D$ compute-scaled, says the advantage is still worth several times the FLOP budget. The two fits differ by less than 0.03 nats over the entire range where *both* were measured ($\leq 10^{20}$ FLOPs) — well inside the region where seed variance is 0.005–0.01 nats and data-order effects are comparable. Four orders of magnitude later they disagree about whether to build a 671B sparse model or a 170B dense one, a difference of roughly $10^6$ GPU-hours.

The two hypotheses are separated by a quantity smaller than the measurement noise at every scale where the measurement has been made. That is the problem, stated exactly.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*