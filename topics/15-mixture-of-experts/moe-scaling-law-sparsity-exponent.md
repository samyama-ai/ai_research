---
id: 15-mixture-of-experts/moe-scaling-law-sparsity-exponent
title: "Scaling Law for Sparse Mixture-of-Experts Loss"
topic: 15-mixture-of-experts
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Scaling Law for Sparse Mixture-of-Experts Loss

> **Topic:** Mixture of Experts · **ID:** `15-mixture-of-experts/moe-scaling-law-sparsity-exponent` · **Status:** partially-solved

## 1. Problem Statement

Dense transformer loss is predicted to a few percent by $L(N, D)$ with two power-law terms (Kaplan 2020; Hoffmann 2022). Sparse MoE breaks the single-$N$ parameterization: total parameters $N_{\text{tot}}$, active parameters $N_{\text{act}}$, expert count $E$, top-$k$, and expert width all move independently. The problem: find the smallest sufficient parameterization of MoE loss and the exponent governing the sparsity direction.

Three variants, different difficulty:

- **Measurement.** Given a fitted family, does $L$ extrapolate one order of magnitude in compute with error comparable to dense fits (≈1–2% relative on held-out loss)? Runnable today; largely unrun at frontier scale.
- **Method.** Given budget $C$ and a deployment constraint (memory, or inference FLOPs, or serving latency), output the loss-minimizing $(N_{\text{tot}}, N_{\text{act}}, E, G, D)$. Partially solved, with mutually inconsistent published optima.
- **Theory.** Explain *why* the sparsity term takes the functional form it does — derive the exponent from a data/routing model rather than fitting it. Open.

Solving it means: a closed form with parameters fitted below $10^{20}$ FLOPs whose compute-optimal prescription, executed at $10^{22}$ FLOPs, lands within its stated confidence interval, and whose loss ordering survives a change of tokenizer, corpus, and router.

## 2. Formal Setting

Tokens $x_{1:D}$ drawn i.i.d. from corpus distribution $\mathcal{P}$; loss is next-token cross-entropy in nats/token on a *held-out split of the same $\mathcal{P}$* — not a downstream benchmark.

Per MoE layer with model width $d_{\text{model}}$ and dense-equivalent hidden width $d_{\text{ff}}$:

- $E$ = experts per layer; $k$ = experts routed per token (top-$k$).
- **Granularity** $G = d_{\text{ff}} / d_{\text{expert}}$ — how many times each expert is narrower than one dense FFN (Krajewski et al. 2024). $G=1$ recovers classic Switch/GShard experts.
- $N_{\text{tot}}$ = all parameters excluding embeddings; $N_{\text{act}}$ = parameters touched per token, again excluding embeddings. **Measured, not derived:** count them from the module list, because shared experts, router matrices, and attention are counted inconsistently across papers.
- **Sparsity** $S = 1 - N_{\text{act}}/N_{\text{tot}}$.
- Training compute $C \approx 6 N_{\text{act}} D$ — this approximation ignores routing, all-to-all, and attention, and is the single largest source of cross-paper disagreement.

Clark et al. (2022) fit a bilinear form in logs,

$$\log L = a\log N + b\log \hat E + c\,(\log N)(\log \hat E) + d,$$

with $\hat E$ a capped effective expert count; the interaction term $c$ forces the routing gain to shrink as $N$ grows. Krajewski et al. (2024) fit an additively separable form with a granularity-dependent coefficient,

$$L(N, D, G) \;=\; c \;+\; \Big(\frac{g}{G^{\gamma}} + a\Big)\frac{1}{N^{\alpha}} \;+\; \frac{b}{D^{\beta}},$$

so that increasing $G$ shrinks the parameter-term coefficient rather than interacting multiplicatively with $N$. The two forms disagree qualitatively about the large-$N$ limit, and the disagreement is the problem.

Assumptions, with those known to be violated marked:

1. Single-epoch, fully converged training with tuned LR. **Violated** — Clark et al. trained all models on a fixed token count, off the Chinchilla-optimal ray.
2. $C = 6ND$ ignores routing overhead. **Violated** — all-to-all cost grows with $E$ and with expert-parallel degree; wall-clock and FLOP-optimal prescriptions diverge.
3. Loss is a function of $(N_{\text{tot}}, N_{\text{act}}, \ldots)$ alone, independent of router algorithm and load-balance coefficient. **Violated** — auxiliary-loss weight and dropless vs. capacity-factor routing shift loss by an amount comparable to a $2\times$ compute change.
4. $E$ and $G$ vary continuously. **Violated** — hardware forces $E$ to powers of two and $d_{\text{expert}}$ to multiples of 128.

## 3. State of the Art

**Theory SOTA.** There is none for the sparsity exponent. Every published MoE law is a curve fit. The nearest analytic result is Frantar et al. (ICLR 2024) for *weight*-sparse (not conditionally-routed) models, which derives a multiplicative capacity factor $(1-S)^{\gamma}$ from a fixed-capacity argument; it does not transfer to routing, where sparsity adds parameters instead of removing them.

**Empirical SOTA.** Three fits, in tension:

- **Clark et al., ICML 2022** — established at $\le$ 1.3B dense-equivalent, fixed ~130B tokens: routing gain decays with scale, extrapolating to parity near ~900M parameters. *This conclusion is now understood as an artifact* of fixed expert size ($G=1$) and off-optimal token budget.
- **Krajewski et al., ICML 2024** — granularity $G$ is a first-class variable; optimal $G$ grows with compute; the Clark saturation disappears. Established within their fitted range ($\le \sim$10$^{20}$ FLOPs); the claim that no saturation occurs at frontier scale is **extrapolation, unablated**.
- **Abnar et al., ICML 2025 (Apple)** — at fixed $N_{\text{tot}}$, optimal sparsity $S^\star$ *increases* with training budget. Established as a fit over a large sweep; the downstream-task corollary is reported as benchmark numbers only, without matched-loss controls.

Production systems (DeepSeekMoE, ACL 2024; OLMoE, ICLR 2025) confirm the direction — many fine-grained experts plus a shared expert beat few coarse experts at matched active FLOPs — but their configurations were chosen by ablation, not by solving any law.

## 4. What Is Known

- **Sparsity helps at matched active FLOPs, robustly.** GShard (ICLR 2021) and Switch Transformer (JMLR 2022) report ~4–7$\times$ speedups to fixed dense quality at matched FLOPs/token; GLaM (ICML 2022) matched GPT-3 quality at roughly 1/3 the training energy with 1.2T total / 97B active parameters.
- **Fine granularity beats coarse.** DeepSeekMoE 16B (2.8B active) matched LLaMA2-7B on most benchmarks; OLMoE-1B-7B (1.3B active, 64 experts, top-8) beat all open ~1B-active models at 5T tokens. Both use $G \gg 1$.
- **The Clark saturation is not a law of nature.** Krajewski et al. reproduce Clark's decaying gain only when $G$ is pinned to 1; with $G$ free, MoE advantage over dense continues to grow across their range.
- **Optimal sparsity is budget-dependent.** Abnar et al.: for a fixed $N_{\text{tot}}$, loss is monotone decreasing in $S$ over the range tested once $D$ is large, with $S^\star$ shifting upward as $C$ grows — measured over models up to the low-billions of total parameters.
- **Memory-constrained optima differ.** Ludziejewski et al. (2025) report MoE configurations beating dense at equal *total* parameters and equal training FLOPs, with up to ~5$\times$ fewer active parameters — contradicting the folk claim that MoE only wins when memory is free.

## 5. What Is Not Known

- **Theoretically open.** No derivation of the granularity exponent $\gamma$, or of why the sparsity direction is a power law at all, from any model of the data distribution or the router. $\gamma$ is a fitted constant with no known interpretation.
- **Empirically open.** Whether any published law extrapolates past $10^{22}$ FLOPs. Every fit was calibrated below $\sim$10$^{20}$; frontier MoE runs are single points, not sweeps, and their configurations are not disclosed with matched dense controls. Also open: whether the law's coefficients are corpus-invariant.
- **Methodologically blocked.** The unit of "compute" itself. FLOP-optimal, memory-optimal, and wall-clock-optimal sparsity give different answers, and no accepted normalization exists that makes the three commensurate. Until that is fixed, "the MoE scaling law" names three different functions.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability under confounding by the token budget**. $L$ depends on at least $(N_{\text{tot}}, N_{\text{act}}, E, G, D, k)$, of which only four are independent given the architecture, and the published sweeps vary them along correlated rays. Clark's decaying-gain interaction term and Krajewski's separable granularity term fit the *same* small-scale data within noise; they diverge only where nobody has measured. That is not a resolvable ambiguity at $10^{19}$ FLOPs — it requires a sweep at the scale where the predictions separate, which is exactly the scale nobody runs sweeps at. Compounding it: routing hyperparameters (auxiliary-loss weight, capacity factor) shift loss by amounts comparable to the effect being measured, so an unablated router change can flip which law fits better.

## 7. Current Research (as of 2026)

- **Granularity-aware joint laws** (Warsaw/IDEAS–NCBR group: Krajewski, Ludziejewski, Jaszczur) — extending the fit to memory- and inference-constrained optima.
- **Inference-aware optima** (Apple; Yun et al.) — replacing training FLOPs with a serving-cost objective, which moves $S^\star$ down sharply because total parameters set the memory bill.
- **Router-invariant parameterizations** *(frontier — verify)* — attempts to define an "effective active parameter" count that absorbs routing quality, so that expert-choice, sigmoid-gated, and auxiliary-loss-free routers land on one curve.
- **Upcycling laws** *(frontier — verify)* — scaling behavior when experts are initialized from a trained dense checkpoint rather than from scratch; the from-scratch laws are not expected to hold.

## 8. Concrete Next Experiment

**Question.** Does the granularity term saturate, as Clark's interaction form predicts, or stay separable, as Krajewski's form predicts?

**Scale.** A 3$\times$3$\times$2 grid: $N_{\text{act}} \in \{$0.3B, 1B, 3B$\}$, $G \in \{1, 8, 32\}$, $E$ chosen to hold $N_{\text{tot}}$ fixed at 8$\times$ $N_{\text{act}}$; each trained Chinchilla-optimal ($D = 20 N_{\text{act}}$) on one fixed corpus, one tokenizer, one router (dropless top-$k$, auxiliary-loss weight swept over 3 values at the smallest point only and then held fixed). Total ≈ $2\times10^{21}$ FLOPs — roughly one frontier pretraining run.

**Control arm.** Dense models at the same three $N_{\text{act}}$ and same $D$, same corpus, same LR schedule.

**Deciding number.** Fit both functional forms to the six points with $N_{\text{act}} \le 1$B, then predict held-out loss at $N_{\text{act}} = 3$B, $G=32$. Report the signed relative error of each. The forms' predictions separate by $\ge$0.03 nats/token there; run-to-run seed noise at this scale is $\approx$0.005 nats. **If Clark's form errs by $>$2% and Krajewski's by $<$1%, separability is established over a decade of compute; if both err $>$2%, neither parameterization is sufficient and the missing variable is the finding.**

## 9. Key References

- **[Foundational]** Kaplan, McCandlish, Henighan, et al. *Scaling Laws for Neural Language Models.* 2020. — arXiv:2001.08361
- **[Foundational]** Hoffmann, Borgeaud, Mensch, et al. *Training Compute-Optimal Large Language Models.* NeurIPS 2022. — arXiv:2203.15556
- **[Foundational]** Lepikhin, Lee, Xu, et al. *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding.* ICLR 2021. — arXiv:2006.16668
- **[Foundational]** Fedus, Zoph, Shazeer. *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.* JMLR 2022. — arXiv:2101.03961
- **[SOTA]** Clark, de las Casas, Guy, et al. *Unified Scaling Laws for Routed Language Models.* ICML 2022. — arXiv:2202.01169
- **[SOTA]** Krajewski, Ludziejewski, Adamczewski, et al. *Scaling Laws for Fine-Grained Mixture of Experts.* ICML 2024. — arXiv:2402.07871
- **[SOTA]** Abnar, Shah, Busbridge, et al. *Parameters vs FLOPs: Scaling Laws for Optimal Sparsity for Mixture-of-Experts Language Models.* ICML 2025. — arXiv:2501.12370
- **[SOTA]** Ludziejewski, Pióro, Krajewski, et al. *Joint MoE Scaling Laws: Mixture of Experts Can Be Memory Efficient.* 2025. — arXiv:2502.05172
- **[Related]** Frantar, Riquelme, Houlsby, Alistarh, Evci. *Scaling Laws for Sparsely-Connected Foundation Models.* ICLR 2024. — arXiv:2309.08520
- **[Systems]** Du, Huang, Dai, et al. *GLaM: Efficient Scaling of Language Models with Mixture-of-Experts.* ICML 2022. — arXiv:2112.06905
- **[Systems]** Dai, Deng, Zhao, et al. *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models.* ACL 2024. — arXiv:2401.06066
- **[Systems]** Muennighoff, Soldaini, Groeneveld, et al. *OLMoE: Open Mixture-of-Experts Language Models.* ICLR 2025. — arXiv:2409.02060
- **[Survey]** Yun, Zhuang, Fu, Xing, Zhang. *Toward Inference-Optimal Mixture-of-Expert Large Language Models.* 2024. (identifier uncertain; link omitted)

## 10. Worked Example

Take a budget $C = 6\times10^{20}$ FLOPs and ask for the compute-optimal MoE.

Dense Chinchilla baseline: $C = 6ND$ with $D = 20N$ gives $N = \sqrt{C/120} \approx 2.2\times10^{9}$, i.e. 2.2B parameters on 45B tokens.

Now hold active parameters at $N_{\text{act}} = 2.2\times10^{9}$ and set $E=64$, $k=8$, so $N_{\text{tot}} \approx 1.7\times10^{10}$ and $S = 1 - 2.2/17 = 0.87$.

- **Clark's form** predicts the routing gain has largely decayed by $N \approx 10^9$: the interaction term $c(\log N)(\log \hat E)$ has the same magnitude and opposite sign to $b\log\hat E$ near there, so predicted MoE-vs-dense improvement is a few hundredths of a nat and shrinking.
- **Krajewski's form** at $G = 8$ ($d_{\text{expert}} = d_{\text{ff}}/8$) multiplies the parameter-term coefficient by $(g/G^{\gamma} + a)$; with the fitted $\gamma$, that coefficient falls by roughly a third relative to $G=1$, and the gain *grows* with $N$ because the $N^{-\alpha}$ term still dominates the irreducible constant $c$.

Both were fitted on runs at or below $\sim$10$^{19}$ FLOPs, where they agree to within about 0.01 nats/token — smaller than the shift caused by changing the load-balancing auxiliary-loss weight from 0.01 to 0.001. **That is the obstruction, made concrete:** at 60$\times$ the fitting compute the two laws disagree about whether to build a 2.2B dense model or a 17B-total/2.2B-active MoE, and the only evidence that would discriminate them is buried under a routing hyperparameter that neither paper held fixed across the other's sweep. The prescription is not compute-limited in principle — it is one frontier run — but no published sweep varies $G$ and $N$ orthogonally with the router frozen.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*