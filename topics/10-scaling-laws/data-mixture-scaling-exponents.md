---
id: 10-scaling-laws/data-mixture-scaling-exponents
title: "Data Mixture Exponents"
topic: 10-scaling-laws
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Data Mixture Exponents

> **Topic:** Scaling Laws & Compute Allocation · **ID:** `10-scaling-laws/data-mixture-scaling-exponents` · **Status:** open

## 1. Problem Statement

Pretraining corpora are mixtures: web text, code, books, math, multilingual, scientific. The mixture is a simplex point $r \in \Delta^{k-1}$. Chinchilla-style scaling laws fit loss as $L(N,D) = E + A N^{-\alpha} + B D^{-\beta}$ for a *fixed* corpus. The question is what $r$ does to that fit.

Three variants, different difficulty:

- **Measurement.** Given a family of runs over $(N, D, r)$, can $\alpha(r)$ and $\beta(r)$ be estimated with error bars tight enough to reject $\alpha(r) = \text{const}$? Purely a fitting and identifiability question.
- **Method.** Given a target evaluation, predict the compute-optimal $r^\star(C)$ at a budget $C$ far above any budget where you can afford to search. Solved approximately by DoReMi/RegMix/Data Mixing Laws; not solved with guarantees.
- **Theory.** Is there a model in which the loss exponent is a computable functional of the data distribution, so that reweighting domains provably moves $\alpha$ rather than only $A$ and $E$?

**Decision predicate.** Mixture affects only the *coefficients* $(A, B, E)$ iff $r^\star(C)$ is independent of $C$. If the compute-optimal mixture shifts with scale, at least one exponent is mixture-dependent. That equivalence is the sharpest operational form of the problem.

## 2. Formal Setting

Domains $i = 1..k$ with distributions $\mathcal{D}_i$; mixture $r$, $\sum_i r_i = 1$. Sampling is *with replacement from finite pools*: domain $i$ has $T_i$ unique tokens, so training $D$ tokens repeats domain $i$ approximately $\epsilon_i = r_i D / T_i$ times.

Per-domain validation loss, measured as mean next-token cross-entropy in nats on a held-out shard of $\mathcal{D}_i$ never seen in training:

$$\hat{L}_i = -\frac{1}{|V_i|}\sum_{t \in V_i} \log p_\theta(t \mid t_{<t}).$$

Aggregate objective is a *second* weight vector $w$ (evaluation weights), generally $\neq r$:

$$L(N,D,r) = \sum_i w_i\, \hat{L}_i(N,D,r).$$

Conflating $w$ and $r$ is the single most common error in this literature.

The hypothesis under test, per domain:

$$\hat{L}_i(N,D,r) = E_i(r) + A_i(r)\,N^{-\alpha_i(r)} + B_i(r)\,D^{-\beta_i(r)}.$$

- $N$ = non-embedding parameters (count them; embedding inclusion shifts $\alpha$ by ~0.02 at 100M–1B).
- $D$ = tokens *consumed*, not unique tokens.
- $C \approx 6ND$ FLOPs, forward+backward, ignoring attention quadratic term.
- $\alpha_i(r)$ = local log-log slope $-\partial \log(\hat{L}_i - E_i)/\partial \log N$ at fixed $D/N$. Only estimable if $E_i$ is estimable, which is the crux (§6).

**Assumptions, and their violations.**

1. *Power law + constant floor is the true functional form.* Violated: loss curves show breaks and a "descent" phase; Chinchilla's own fit is sensitive to the Huber $\delta$ and the $E$ initialization (Besiroglu et al., 2024).
2. *Domains are disjoint.* Violated badly. Common Crawl contains code, math, and books; measured domain "transfer" terms are large and asymmetric.
3. *Loss is separable in $N$ and $D$.* Violated under repetition: at $\epsilon_i \gtrsim 4$ the $D$-term saturates (Muennighoff et al., 2023), so $\beta_i$ becomes a function of $r_i D / T_i$, not of $D$ alone.
4. *Fixed architecture, optimizer, and LR schedule across the sweep.* Routinely violated in practice; a mis-tuned LR at large $N$ biases $\alpha$ downward.

## 3. State of the Art

**Empirical / systems SOTA (established).**

- **DoReMi** (Xie et al., NeurIPS 2023): group-DRO on a 280M proxy yields mixture weights transferred to 8B; reported 2.6× fewer steps to baseline perplexity on The Pile and +6.5% average few-shot accuracy. Established that a small proxy's weights transfer usefully; *not* established that they are optimal at 8B.
- **RegMix** (Liu et al., 2024): 512 models of ~1M parameters × 1B tokens, regression from $r$ to loss, extrapolate. High rank correlation between predicted and realized ordering at 1B scale. This is a *ranking* result at one target scale.
- **Data Mixing Laws** (Ye et al., 2024): exponential-of-linear form $\hat L_i(r) = c_i + k_i \exp(\sum_j t_{ij} r_j)$, nested with $N$ and $D$ laws to predict a 1B/100B RedPajama run from small runs. The nesting *assumes* $\alpha, \beta$ are mixture-independent and only the prefactors move — the assumption this problem asks about.
- **D-CPT Law** (Que et al., NeurIPS 2024): fits $L(N,D,r)$ with an explicit mixture-ratio term for continual pretraining; reports domain-specific fitted exponents that differ across domains.

**Claimed but unablated.** That proxy-derived mixtures remain optimal two or more orders of magnitude up in compute. Every paper above validates one hop (e.g. 280M → 8B) against a *human-chosen* baseline, not against a mixture search at the target scale — because that search is unaffordable.

**Negative result, established.** **Aioli** (Chen, Hu, Lourie, Cho, Ré, ICLR 2025) unified DoReMi/DoGE/Skill-It as instances of one linear mixing optimization and found that existing methods do not consistently beat plain stratified (proportional) sampling across their six evaluation settings; the gains are within run-to-run noise in several. Aioli's own improvement is reported as a small average perplexity gain, on the order of 0.3 nats-equivalent — real, but small next to the headline speedup numbers.

**Theory SOTA.** No theorem gives $\alpha(r)$ for realistic text. Two models make the exponent distribution-dependent: (i) $\alpha \approx 4/d$ with $d$ the intrinsic data-manifold dimension (Sharma & Kaplan, JMLR 2022; Bahri et al., PNAS 2024); (ii) the **Quantization Model** (Michaud, Liu, Niklasson, Tegmark, NeurIPS 2023), where the loss exponent is a deterministic function of the Zipf exponent of subtask frequencies. Both imply mixture *should* move $\alpha$, since reweighting domains reweights the frequency distribution.

## 4. What Is Known

- Chinchilla refit (Besiroglu, Erdil, Barnett, You, 2024) on Hoffmann et al.'s data: $\alpha \approx 0.35$, $\beta \approx 0.37$, $E \approx 1.82$, with confidence intervals excluding the originally reported $\alpha = 0.34, \beta = 0.28$ pair. Scale: 70M–16B, MassiveText.
- Data pruning can change the *functional form*, not just constants: on ImageNet/CIFAR, keeping the hardest examples under a good scoring rule gives error decaying faster than the power law, approaching exponential (Sorscher et al., NeurIPS 2022). Existence proof that distribution choice moves the exponent — in vision, at ≤ ResNet-50 scale.
- Repetition: up to ~4 epochs is nearly as good as fresh tokens; by ~16 epochs added compute is nearly worthless (Muennighoff et al., NeurIPS 2023; up to 9B params, 900B tokens). So $\beta$ is repetition-dependent and therefore $r$-dependent through $\epsilon_i$.
- Optimal filtering strength is compute-dependent: aggressive filtering wins at small budgets and loses at large ones (Goyal et al., CVPR 2024, LAION/DataComp CLIP training). Direct evidence that $r^\star$ moves with $C$.
- Downstream metrics can be *non-monotone* in data when the pretraining distribution is misaligned with the task, while upstream loss keeps improving (Isik et al., 2024, machine translation, up to 3B).

## 5. What Is Not Known

- **Methodologically blocked (primary).** $\alpha_i(r)$ is not separately identifiable from $E_i(r)$ at feasible sweep ranges. Over 2–3 decades of $N$, a fit with larger $E$ and larger $\alpha$ is nearly indistinguishable from smaller $E$ and smaller $\alpha$; the ridge in the likelihood is nearly flat. No published mixture study reports a confidence region for $\alpha$ jointly with $E$.
- **Empirically open.** Whether $r^\star(C)$ shifts by more than fit noise across ≥ 3 decades of compute for language pretraining. Runnable today at 10M → 3B; nobody has published the full $(N, D, r)$ grid with replicate seeds.
- **Empirically open.** Whether proxy-model mixture weights degrade monotonically with the proxy/target scale gap, and at what gap they become worse than stratified.
- **Theoretically open.** Whether any exponent-changing mechanism (manifold dimension, Zipf reweighting) predicts the *sign* of $\Delta\alpha$ from measurable corpus statistics. No proof either way.

## 6. Why It Is Hard

**Non-identifiability of $(E, \alpha)$**, compounded by **confounded measurement**.

The irreducible loss $E_i$ is the entropy of domain $i$ plus tokenizer overhead — unknown and mixture-dependent, since a mixture changes the tokenizer's effective coverage. With three decades of $N$ and 1–3% run-to-run loss noise, the fitted $\alpha$ moves by ±0.05 when $E$ is perturbed by 0.05 nats, which is the same size as any plausible mixture effect. Fixing $E$ by assumption (what nested mixture laws do) *defines away* the question.

Second, the compute cost of the honest experiment. Distinguishing $\alpha(r_1) \neq \alpha(r_2)$ requires the full grid at several $r$, several $N$, several seeds — a multiplicative blow-up over a single scaling ladder.

Third, the evaluation does not measure what it names: "loss on the mixture" is $\sum_i r_i \hat L_i$, which *improves* by upweighting easy, low-entropy domains. Optimizing it optimizes the corpus, not the model.

## 7. Current Research (as of 2026)

- Regression-and-extrapolate mixture prediction (RegMix line, Sea AI Lab; CLIMB-style iterative bootstrapping at NVIDIA) — scaling the proxy sweep rather than fixing the identifiability problem. *(frontier — verify)*
- Unified/diagnostic framing after Aioli (Ré group, Stanford; Cho group, NYU): asking whether any online method beats stratified once noise is accounted for.
- Domain-specific continual-pretraining laws with explicit ratio terms (D-CPT line, Alibaba).
- Mechanistic exponent theory: quanta/Zipf accounts (Tegmark group, MIT) and multi-power-law / broken-law forms. No published attempt to *derive* $\alpha(r)$ from corpus statistics. *(frontier — verify)*

## 8. Concrete Next Experiment

**Question.** Does $r^\star$ shift with compute by more than fit noise?

**Scale.** Two mixtures only, on a two-domain corpus (web + code), $r \in \{0.9/0.1,\ 0.5/0.5\}$ plus a third at $0.7/0.3$. Model sizes $N \in \{20\text{M}, 60\text{M}, 200\text{M}, 600\text{M}, 1.8\text{B}\}$ non-embedding, each at Chinchilla ratio $D = 20N$, **3 seeds each**. 45 runs; roughly $3 \times 3 \times 6 \times (20\text{M} \cdot 20 \cdot 20\text{M}) \dots$ — total under $5 \times 10^{21}$ FLOPs, i.e. a few thousand H100-hours. Unique-token pools sized so $\epsilon_i \le 1$ for every cell, removing the repetition confound.

**Control arm.** The *same* 45 runs re-fit under the constraint $\alpha_{\text{web}} = \alpha_{\text{code}} = \alpha$, $E$ shared across mixtures. This is the null model that every nested mixture law assumes.

**Deciding number.** The likelihood-ratio statistic between the constrained fit and the free fit, calibrated against the seed-to-seed variance. Report $\Delta\alpha = \alpha(0.9/0.1) - \alpha(0.5/0.5)$ with a bootstrap 95% CI computed *jointly with $E$*. **If the CI on $\Delta\alpha$ excludes 0 and $|\Delta\alpha| > 0.02$, mixture exponents are real and every nested mixture law needs re-derivation. If the CI contains 0 at this range, publish the width** — that width is the number the field currently does not have.

## 9. Key References

- **[Foundational]** Hoffmann, Borgeaud, Mensch, et al. *Training Compute-Optimal Large Language Models.* NeurIPS 2022. — arXiv:2203.15556
- **[Foundational]** Kaplan, McCandlish, Henighan, et al. *Scaling Laws for Neural Language Models.* 2020. — arXiv:2001.08361
- **[SOTA]** Xie, Pham, Dong, et al. *DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining.* NeurIPS 2023. — arXiv:2305.10429
- **[SOTA]** Ye, Liu, Zhang, Yin, Qiu. *Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance.* 2024. — arXiv:2403.16952
- **[SOTA]** Liu, Zheng, Do, et al. *RegMix: Data Mixture as Regression for Language Model Pre-training.* 2024. — arXiv:2407.01492
- **[SOTA / negative]** Chen, Hu, Lourie, Cho, Ré. *Aioli: A Unified Optimization Framework for Language Model Data Mixing.* ICLR 2025.
- **[Foundational]** Muennighoff, Rush, Barak, et al. *Scaling Data-Constrained Language Models.* NeurIPS 2023. — arXiv:2305.16264
- **[Foundational]** Sorscher, Geirhos, Shekhar, Ganguli, Morcos. *Beyond Neural Scaling Laws: Beating Power Law Scaling via Data Pruning.* NeurIPS 2022. — arXiv:2206.14486
- **[Theory]** Michaud, Liu, Niklasson, Tegmark. *The Quantization Model of Neural Scaling.* NeurIPS 2023. — arXiv:2303.13506
- **[Theory]** Bahri, Dyer, Kaplan, Lee, Sharma. *Explaining Neural Scaling Laws.* PNAS, 2024.
- **[Replication]** Besiroglu, Erdil, Barnett, You. *Chinchilla Scaling: A Replication Attempt.* 2024. — arXiv:2404.10102
- **[Related]** Goyal, Maini, Lipton, Raghunathan, Kolter. *Scaling Laws for Data Filtering — Data Curation Cannot Be Compute Agnostic.* CVPR 2024.
- **[Related]** Que, Liu, Zhang, et al. *D-CPT Law: Domain-specific Continual Pre-Training Scaling Law for Large Language Models.* NeurIPS 2024.
- **[Survey]** Albalak, Elazar, Xie, et al. *A Survey on Data Selection for Language Models.* TMLR, 2024. — arXiv:2402.16827

## 10. Worked Example

Two domains, web (w) and code (c), evaluated with equal weights $w_i = 0.5$. Suppose the true per-domain laws at fixed $D = 20N$ are

$$\hat L_w = 1.80 + 12\,N^{-0.32}, \qquad \hat L_c = 0.90 + 40\,N^{-0.38}.$$

Code has a lower floor and a steeper exponent. At $N = 10^8$: $\hat L_w = 1.80 + 12 \cdot 10^{-8 \cdot 0.32} = 1.80 + 0.0323 = 1.832$; $\hat L_c = 0.90 + 40 \cdot 10^{-8 \cdot 0.38} = 0.90 + 0.0344 = 0.934$. At $N = 10^{10}$: $\hat L_w = 1.80 + 0.00738$, $\hat L_c = 0.90 + 0.00570$. The *reducible* part of code shrinks 6.0× while web shrinks 4.4×. Marginal return per token has flipped ordering: at $10^8$ the reducible losses are nearly equal (0.0323 vs 0.0344), at $10^{10}$ code's is 23% smaller. A greedy allocator that upweighted code at the proxy scale is upweighting the domain with less headroom left at the target scale — $r^\star$ moved.

Now the obstruction. Fit the *web* curve using only $N \in [2\times10^7, 2\times10^9]$ with 1.5% multiplicative noise. Try $E = 1.85$ instead of the true 1.80. The residual reducible loss at $N=10^8$ is now $0.032 - 0.05 < 0$ at the large-$N$ end, so the optimizer compensates by pushing the fitted exponent down; sweeping $E \in [1.75, 1.85]$ moves the best-fit $\alpha$ across roughly $0.27$–$0.38$ with essentially indistinguishable RMSE. The exponent gap we are trying to detect, $0.38 - 0.32 = 0.06$, is *smaller than the range $\alpha$ spans under a 0.05-nat uncertainty in $E$*.

This is the whole problem in one calculation: the effect is real and it is economically decisive at the scales where models are actually trained, but the estimator that would detect it is not identified over the range of $N$ anyone can afford to sweep. Hence the deciding number in §8 is a *jointly bootstrapped* CI on $\Delta\alpha$, never a point estimate.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*