---
id: 06-data-pipeline/data-mixture-weight-scale-transfer
title: "Transfer of Data-Mixture Weights Across Model Scales"
topic: 06-data-pipeline
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Transfer of Data-Mixture Weights Across Model Scales

> **Topic:** Data Pipelines & Curation · **ID:** `06-data-pipeline/data-mixture-weight-scale-transfer` · **Status:** empirically-open

## 1. Problem Statement

Pre-training corpora are partitioned into $k$ domains (web, code, arXiv, books, …). A **mixture** is a sampling weight vector $\alpha \in \Delta^{k-1}$. Tuning $\alpha$ directly at target scale is unaffordable, so every production pipeline tunes it on small proxy models and reuses the result. The problem is whether that reuse is valid.

- **Measurement variant.** Given a proxy scale $(N_p, D_p)$ and a target scale $(N_t, D_t)$, how large is the loss penalty from using $\alpha^\star(N_p, D_p)$ instead of $\alpha^\star(N_t, D_t)$? Nobody has published this penalty as a curve in $N_t/N_p$ with confidence intervals.
- **Method variant.** Construct a transfer rule $T$ with $T(\alpha^\star(N_p,D_p); N_p,D_p,N_t,D_t) \approx \alpha^\star(N_t,D_t)$ — the data-mixture analogue of $\mu$P's zero-shot hyperparameter transfer.
- **Theory variant.** Prove conditions on the per-domain loss surface under which $\alpha^\star$ is scale-invariant, or exhibit a family where it provably is not.

Solved means: a rule that predicts target-scale optimal weights to within the run-to-run seed noise of the target-scale loss, verified at $N_t/N_p \geq 100$.

## 2. Formal Setting

Domains $i = 1..k$ with token pools $\mathcal{D}_i$. Model size $N$ (non-embedding parameters), token budget $D$. Training samples domain $i$ with probability $\alpha_i$; measured as the realized token fraction in the actual dataloader, not the configured weight — these differ when a domain is exhausted and repeated.

Validation loss on held-out domain $j$:
$$L_j(\alpha, N, D) = -\frac{1}{|V_j|}\sum_{x \in V_j} \log p_{\theta(\alpha,N,D)}(x)$$
in nats/token, with $V_j$ a fixed held-out shard never seen in training, and $\theta(\alpha,N,D)$ the weights from one seed at a fixed LR schedule.

The objective is a scalarization over domains with **evaluation weights** $w \in \Delta^{k-1}$ — a separate object from the sampling weights $\alpha$, and the usual source of confusion:
$$\alpha^\star(N,D) = \arg\min_{\alpha \in \Delta^{k-1}} \sum_j w_j\, L_j(\alpha, N, D).$$

**Transfer gap**, the quantity the page is about:
$$\Delta(N_p \to N_t) = \sum_j w_j\Big[L_j(\alpha^\star(N_p,D_p), N_t, D_t) - L_j(\alpha^\star(N_t,D_t), N_t, D_t)\Big] \geq 0,$$
measured in nats/token, and reported against the seed-noise floor $\sigma_{\text{seed}}$ (std of $\sum_j w_j L_j$ across $\geq 3$ seeds at fixed $\alpha$). Transfer "holds" iff $\Delta \lesssim 2\sigma_{\text{seed}}$.

Assumptions, with those known to be violated marked:
1. $\alpha^\star$ is unique and interior. **Violated:** loss is often near-flat over a large simplex region, so $\arg\min$ is poorly conditioned.
2. Domains are disjoint. **Violated:** web crawl contains code and arXiv; deduplication is partial.
3. Data is not repeated. **Violated at target scale** — up-weighting a small domain forces epochs over it, and the repetition penalty is itself scale-dependent (Muennighoff et al., NeurIPS 2023).
4. Validation loss ranks downstream ability. **Violated across domains:** loss is only comparable within a fixed tokenizer and text distribution; a mixture that lowers total loss can lower reasoning accuracy.
5. Other hyperparameters (LR, batch, schedule) are held at their own optima. **Usually violated** — proxy and target runs rarely both sit at optimal LR, so mixture effects and LR effects are confounded.

## 3. State of the Art

**Empirical SOTA — established.**
- **DoReMi** (Xie et al., NeurIPS 2023): group-DRO over domains on a 280M proxy; the resulting weights, applied to an 8B model on The Pile, reach baseline perplexity with **2.6× fewer steps** and gain **+6.5%** average few-shot accuracy. This is the strongest published demonstration that *some* proxy-tuned mixture transfers up ~30×.
- **RegMix** (Liu et al., ICML 2025): fit a regression from mixture to loss using 512 models of 1M parameters trained on 1B tokens, then apply the predicted mixture to a 1B model on 25B tokens. Reported to match or beat DoReMi at ~10% of its compute.
- **Data Mixing Laws** (Ye et al., ICML 2024): loss is an explicit parametric function of $\alpha$, $N$, and $D$ (exponential-of-linear in mixture proportions), fitted on small runs and *extrapolated* to predict the optimum for a 1B/100B-token run.

**Claimed but unablated.** Almost every mixture paper compares against a hand-tuned baseline at one target scale with one seed. Seed noise is rarely reported, so a 0.01-nat win and a null result are not distinguishable from the tables. **Aioli** (Chen et al., ICLR 2025) re-ran the main methods under a unified parameterization and found existing mixing methods frequently fail to beat a simple stratified baseline on average — the single most important negative control in the literature.

**Benchmark-number-only results.** Industrial reports (Llama 3, DeepSeek-V2/V3, Qwen technical reports) state that mixtures were chosen by small-scale ablation and then scaled. None publishes $\Delta(N_p \to N_t)$; the transfer claim exists only as a downstream benchmark score at one scale.

**Theory SOTA.** No scale-transfer theorem for $\alpha$. The nearest analogue is $\mu$P (Yang et al., NeurIPS 2021), which proves optimal LR transfer under a specific parameterization — mixture weights have no equivalent parameterization result.

## 4. What Is Known

- **The optimum moves with compute.** Goyal et al. (CVPR 2024) show for CLIP-style filtering that the utility of aggressively filtered data *inverts* with compute budget: high-quality-only subsets win at small budgets and lose at large ones because repetition dominates. Measured at up to ~640M samples-seen on DataComp pools.
- **The optimum moves with $N$ and $D$ separately.** AutoScale (Kang et al., 2024/2025) reports that mixtures optimal at ~10^8-token scale are measurably sub-optimal at 10^10 tokens, with the direction of shift domain-dependent (web up, curated-small-domain down as $D$ grows).
- **Repetition penalty is quantified.** Muennighoff et al. (NeurIPS 2023): up to ~4 epochs of repeat is nearly free; beyond ~16 epochs, added compute returns approximately nothing. This bounds how far a small domain can be up-weighted at large $D$ — a constraint absent at proxy scale.
- **Transfer works over ~30× at least once.** DoReMi 280M → 8B (Section 3). It is a single corpus (The Pile), a single $k=22$ partition, and a single objective.
- **Loss is smooth in $\alpha$.** Regression and parametric-law fits (RegMix, Ye et al.) achieve high rank correlation ($r$ near 0.9 on held-out mixtures at 1M-param scale), so the surface is learnable — at that scale.

## 5. What Is Not Known

- **Empirically open (primary).** The transfer gap $\Delta(N_p \to N_t)$ as a function of $N_t/N_p$, with seed-noise error bars, on a fixed corpus. Every ingredient exists; the sweep — a target-scale mixture search — has not been published because it costs a full mixture optimization at target scale.
- **Empirically open.** Whether $D$-transfer or $N$-transfer fails first. Repetition arguments predict $D$; nobody has separated them.
- **Methodologically blocked.** The objective itself. $w$ (evaluation weights over domains) is chosen by hand and is not identified by any measurement; "the optimal mixture" is undefined until someone fixes $w$ or fixes a downstream metric. Two labs reporting different optima may simply be optimizing different objectives.
- **Theoretically open.** No result of the form "under conditions C, $\alpha^\star(N,D)$ is independent of $N$." Not even for a two-domain linear-regression toy model with a proof.

## 6. Why It Is Hard

The specific obstruction is **the absent ground truth at target scale, made absent by cost**. Computing $\Delta$ requires $\alpha^\star(N_t, D_t)$, which requires a search over the simplex *at target scale* — tens to hundreds of full pre-training runs. A single 8B/1T-token run is $\sim\!10^{23}$ FLOPs; a 50-point simplex search is $5\times10^{24}$, comparable to a frontier lab's entire pre-training budget. So the reference point the field needs is exactly the thing nobody can afford, and every claim of transfer is therefore a claim without a control.

Two secondary obstructions compound it: (a) **flat optima** — if $\nabla_\alpha L$ is small near the optimum, $\alpha^\star$ is non-identifiable and two very different weight vectors give indistinguishable loss, so "the optimum moved" and "the optimum was never located" look alike; (b) **the evaluation does not measure what it names** — mixture papers optimize aggregate validation loss and then report few-shot benchmarks, and these disagree in sign for domains like code and math.

## 7. Current Research (as of 2026)

- **Parametric mixture laws** joint in $(\alpha, N, D)$, extending Ye et al. — Shanghai AI Lab / Fudan lineage; Sailor/RegMix line at Sea AI Lab.
- **Unified analysis and negative controls** — Stanford (Hazy Research) Aioli line, arguing much of the reported gain is mis-parameterization rather than better mixtures.
- **Compute-aware curation** — CMU/Bosch line following Goyal et al., treating filtering strength as a function of budget rather than a fixed threshold.
- **Online/adaptive mixing** that sidesteps transfer by re-weighting during the target run (Albalak et al., Online Data Mixing; Skill-It, Chen et al., NeurIPS 2023). *(frontier — verify: whether any frontier lab now runs adaptive mixing in production rather than a fixed proxy-tuned vector.)*
- **$\mu$P-style parameterization for data** — searching for a reparameterization of $\alpha$ that is scale-invariant by construction. *(frontier — verify; no published theorem.)*

## 8. Concrete Next Experiment

**Scale.** Fix one corpus (SlimPajama, $k=7$ domains) and one tokenizer. Run mixture searches at three scales: 70M/1.4B tokens, 400M/8B tokens, 1.4B/28B tokens (all Chinchilla-optimal, $D \approx 20N$). At each scale, evaluate 32 mixtures drawn from a fixed Dirichlet design (identical design at all three scales), fit a quadratic surrogate, and take its minimizer as $\hat\alpha^\star$.

**Control arm.** Three seeds at the *uniform-by-token-count* mixture at each scale, to estimate $\sigma_{\text{seed}}$; plus the natural-proportion mixture as the human baseline.

**Cost.** $3 \times 35$ runs; the 1.4B tier dominates at ~$35 \times 2.4\times10^{20} \approx 8\times10^{21}$ FLOPs — roughly 2k H100-days, feasible for an academic consortium.

**The deciding number.** The ratio
$$\rho = \frac{\Delta(70\text{M} \to 1.4\text{B})}{\sigma_{\text{seed}}(1.4\text{B})}.$$
$\rho < 2$: proxy tuning at 20× is sound, and the field's default practice is vindicated. $\rho > 5$: proxy-tuned mixtures are measurably wrong at 20×, and every published transfer claim at 30× or more is uncontrolled. Report $\rho$ for $70\text{M}\to400\text{M}$ as well to get the first two points of the $\Delta$-vs-scale curve.

## 9. Key References

- **[Foundational]** Hoffmann et al. *Training Compute-Optimal Large Language Models.* NeurIPS, 2022. — arXiv:2203.15556
- **[Foundational]** Yang, Hu, Babuschkin, et al. *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer.* NeurIPS, 2021. — arXiv:2203.03466
- **[SOTA]** Xie, Pham, Dong, et al. *DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining.* NeurIPS, 2023. — arXiv:2305.10429
- **[SOTA]** Ye, Liu, Zhang, et al. *Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance.* ICML, 2024. — arXiv:2403.16952
- **[SOTA]** Liu, Zeng, He, et al. *RegMix: Data Mixture as Regression for Language Model Pre-training.* ICML, 2025. — arXiv:2407.01492
- **[Negative control]** Chen, Chen, Ré, et al. *Aioli: A Unified Optimization Framework for Language Model Data Mixing.* ICLR, 2025.
- **[Related]** Goyal, Souri, Kirchhof, et al. *Scaling Laws for Data Filtering — Data Curation cannot be Compute Agnostic.* CVPR, 2024.
- **[Related]** Muennighoff, Rush, Barak, et al. *Scaling Data-Constrained Language Models.* NeurIPS, 2023. — arXiv:2305.16264
- **[Related]** Kang, Sun, Wang, et al. *AutoScale: Scale-Aware Data Mixing for Pre-Training LLMs.* 2024/2025 preprint.
- **[Survey]** Albalak, Elazar, Xie, et al. *A Survey on Data Selection for Language Models.* TMLR, 2024. — arXiv:2402.16827

## 10. Worked Example

Two domains: web $\mathcal{D}_1$ (500B tokens available) and arXiv $\mathcal{D}_2$ (20B tokens available). $\alpha = (1-a, a)$. Objective $w = (0.5, 0.5)$.

At the proxy scale $N_p = 70$M, $D_p = 1.4$B: with $a = 0.30$, arXiv consumes $0.42$B tokens — 0.02 epochs, no repetition. The surrogate fit gives $\hat a^\star_p = 0.30$ with $\sum_j w_j L_j = 3.412$ nats and $\sigma_{\text{seed}} = 0.004$ nats.

Now transfer to $N_t = 7$B, $D_t = 140$B. At $a = 0.30$, arXiv supplies 42B tokens from a 20B pool — **2.1 epochs**. At $a = 0.60$ it would be 4.2 epochs, near the point where Muennighoff et al. measure returns starting to decay. The proxy run never touched this regime: at proxy scale every $a \in [0,1]$ is single-epoch, so the loss surface it fit contains no repetition term at all.

Take a plausible penalty of $+0.02$ nats on the arXiv component at 2.1 epochs. The target-scale objective at $a=0.30$ picks up $0.5 \times 0.02 = 0.01$ nats, and the true minimizer slides down to roughly $a^\star_t \approx 0.22$. Then $\Delta(70\text{M}\to7\text{B}) \approx 0.008$ nats against $\sigma_{\text{seed}}(7\text{B}) \approx 0.002$, so $\rho \approx 4$ — a real, four-sigma miss.

The obstruction is visible in the last step: the $0.02$-nat repetition penalty was **assumed**, not measured, because measuring it needs $a^\star_t$, which needs the target-scale search. The proxy cannot even in principle see the effect that breaks it, since the constraint it violates (finite arXiv pool) is inactive at proxy scale. That is why the problem is empirically open rather than merely unstudied: the error term lives in a regime the cheap experiment structurally cannot reach.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*