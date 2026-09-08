---
id: 13-parameter-efficient-adaptation/dp-cost-of-adapter-tuning
title: "Differential Privacy Cost of Adapter Tuning"
topic: 13-parameter-efficient-adaptation
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Differential Privacy Cost of Adapter Tuning

> **Topic:** Parameter-Efficient Adaptation · **ID:** `13-parameter-efficient-adaptation/dp-cost-of-adapter-tuning` · **Status:** partially-solved

## 1. Problem Statement

Under DP-SGD, does restricting the trainable set to an adapter (LoRA, bottleneck adapter, prefix, bias-only) buy utility at fixed privacy budget — and if so, how much, and as a function of what?

Three variants, routinely conflated:

- **Measurement.** At fixed $(\varepsilon,\delta)$, dataset, backbone, and *tuned* hyperparameters, what is the accuracy gap between adapter tuning and full fine-tuning? Solving it means a gap curve over trainable-parameter count spanning several orders of magnitude, with hyperparameter search cost held equal and its own privacy cost accounted.
- **Method.** Does there exist an adapter parameterization whose DP-SGD utility strictly dominates full fine-tuning at every $\varepsilon$ in $[0.5, 8]$? Solving it means such a construction plus an ablation isolating *why* (rank, placement, conditioning, or clipping geometry).
- **Theory.** DP-ERM excess risk scales as $\tilde{O}(\sqrt{d}/(n\varepsilon))$ in the ambient dimension $d$ (Bassily–Smith–Thakurta, FOCS 2014). Empirically the gap between $d\!\approx\!10^5$ and $d\!\approx\!10^9$ trainable parameters is near zero on many tasks. Prove a bound whose dimension term is replaced by a measurable property of the fine-tuning gradient distribution (rank, trace of covariance, or a Gaussian-width term) that predicts the observed insensitivity.

## 2. Formal Setting

Backbone $f_{\theta_0}$, $\theta_0 \in \mathbb{R}^{d}$ pretrained on public data. Adapter map $\phi: \mathbb{R}^{p} \to \mathbb{R}^{d}$, $p \ll d$, with $\phi(0)=\theta_0$; trainable vector $w \in \mathbb{R}^{p}$. Private dataset $D = \{x_i\}_{i=1}^{n}$, loss $\ell(w;x)$.

**DP-SGD step.** Poisson sample $B_t$ at rate $q=B/n$; per-example gradient $g_i = \nabla_w \ell(w_t;x_i)$; clip to $\bar g_i = g_i \cdot \min(1, C/\|g_i\|_2)$; update

$$w_{t+1} = w_t - \eta_t \frac{1}{B}\Big(\sum_{i \in B_t} \bar g_i + \mathcal{N}(0, \sigma^2 C^2 I_p)\Big).$$

**Measured quantities.**

- $\varepsilon$: computed by the accountant actually used — Rényi (Mironov 2017) or numerical PLD — over $T$ steps at $(q,\sigma)$, with $\delta = 1/(10n)$ or smaller. Report the accountant; RDP and PLD differ by 10–30% in $\varepsilon$ at the same $(q,\sigma,T)$.
- **Signal-to-noise per step:** $\mathrm{SNR}_t = \|\frac{1}{B}\sum_i \bar g_i\|_2 \big/ (\sigma C \sqrt{p}/B)$. The $\sqrt{p}$ is the entire hypothesised adapter advantage; measure it, do not assume it.
- **Clipping bias:** $\beta_t = \|\mathbb{E}[\bar g] - \mathbb{E}[g]\|_2 / \|\mathbb{E}[g]\|_2$, estimated on a held-out public split. Rises with $p$ because $\|g_i\|_2$ grows with dimension.
- **Effective gradient dimension:** $d_{\text{eff}} = \big(\mathrm{tr}\,\Sigma_g\big)^2 / \mathrm{tr}(\Sigma_g^2)$ (participation ratio of the per-example gradient covariance $\Sigma_g$), the candidate replacement for $p$ in the risk bound.
- **Utility gap:** $\Delta(\varepsilon) = A_{\text{full}}(\varepsilon) - A_{\text{adapter}}(\varepsilon)$, both arms tuned under an identical hyperparameter-search budget.
- **Cost:** peak activation memory (bytes) and wall-clock per epoch. Per-example clipping removes the usual PEFT memory win unless ghost clipping / book-keeping is used.

**Assumptions, and which fail.** (i) Convexity — false; all bounds transfer only heuristically. (ii) Public pretraining data is non-private — false when the backbone is web-scraped and the private set overlaps it. (iii) Poisson sampling — implementations use shuffled fixed-size batches, so reported $\varepsilon$ is not the $\varepsilon$ realized (Chua et al., ICML 2024). (iv) Hyperparameters are free — they are selected on the private data in nearly every published run, so headline $\varepsilon$ values understate the true budget.

## 3. State of the Art

**Established (reproduced, ablated).**

- DP fine-tuning of large public backbones closes most of the private/non-private gap at $\varepsilon \in [3,8]$: Li et al., *Large Language Models Can Be Strong Differentially Private Learners* (ICLR 2022); Yu et al., *Differentially Private Fine-tuning of Language Models* (ICLR 2022). Both show LoRA and full fine-tuning land within roughly a point of each other on GLUE at matched $\varepsilon$ — the parameter-count advantage is small, not the dominant term.
- Large batches and many epochs, not small $p$, drive DP utility (De et al., 2022; Sander et al., *TAN Without a Burn*, ICML 2023). Scaling $B$ with $\sigma$ held at constant noise-to-signal reproduces DP-SGD accuracy at ~1/100 the compute, and the scaling law is parameter-count-insensitive over the ranges tested.
- Ghost clipping makes per-example clipping cost close to non-private backprop in memory (Li et al., ICLR 2022; Bu et al., *Differentially Private Optimization on Large Model at Small Cost*, ICML 2023), so the "PEFT is needed for DP because of memory" argument is largely obsolete.

**Claimed but unablated.**

- That adapter rank $r$ trades off privacy noise against expressivity with an interior optimum. Reported as benchmark numbers at one or two ranks per paper; no paper sweeps $r$ across a decade with matched tuning budget and reports the curve.
- DP-BiTFiT (Bu et al., ICML 2024) trains ~0.1% of parameters and reports accuracy comparable to full DP fine-tuning with 2–30× speedups. The efficiency claim is well supported; the *accuracy parity* claim rests on benchmark tables, not on an ablation separating dimension effects from clipping-bias effects.
- Prefix tuning under DP is reported as unstable relative to LoRA. This is folklore-plus-tables, not an isolated result.

## 4. What Is Known

- **Dimension term is loose in practice.** Bassily et al. (FOCS 2014) give tight $\Theta(\sqrt{d}/(n\varepsilon))$ minimax excess risk for convex DP-ERM, yet moving from $p\!\approx\!3\times10^5$ (LoRA $r{=}4$ on GPT-2) to $p\!\approx\!3.5\times10^8$ (full RoBERTa-large) does not degrade utility by the predicted $\sim\!30\times$ factor. Established at the $10^8$-parameter scale on GLUE and E2E.
- **Reference numbers.** RoBERTa-large, $\varepsilon=3$, $\delta=10^{-6}$: SST-2 $\approx 93\%$, MNLI-m $\approx 87\%$, against $\approx 96\%$ / $\approx 90\%$ non-private (Li et al., ICLR 2022). GPT-2 on E2E, $\varepsilon\approx 6.8$ with LoRA: BLEU $\approx 64$ vs $\approx 69$ non-private (Yu et al., ICLR 2022). CIFAR-10 from scratch, $\varepsilon=8$: 81.4% top-1 (De et al., 2022) — the no-public-data control.
- **Public pretraining is doing the work.** Ganesh et al., *Why Is Public Pretraining Necessary for Private Model Training?* (ICML 2023), show a separation: the private phase need only find a nearby basin, which is a low-dimensional problem regardless of $p$.
- **Auditing works and reveals slack.** Steinke, Nasr, Jagielski, *Privacy Auditing with One (1) Training Run* (NeurIPS 2023) and Nasr et al. (USENIX Security 2023) give empirical $\varepsilon$ lower bounds; for realistic DP-SGD these sit well below the analytic $\varepsilon$, so "cost at $\varepsilon=8$" is an upper bound on the true cost, unevenly so across parameterizations.

## 5. What Is Not Known

- **Theoretically open.** No bound in which the dimension factor is replaced by $d_{\text{eff}}$ or a gradient-rank quantity that provably tracks fine-tuning. No proof that any adapter family dominates full fine-tuning at fixed $\varepsilon$ — nor that none can.
- **Empirically open.** The gap curve $\Delta(\varepsilon)$ as a joint function of $p$ (six decades), $\varepsilon \in [0.5,8]$, and $n$, at $\ge 7$B-parameter backbones. Runnable today; nobody has run it with matched tuning budgets. Sub-$\varepsilon{=}1$ behavior is nearly unmeasured for LLMs.
- **Methodologically blocked.** The reported $\varepsilon$ is not the realized $\varepsilon$: private hyperparameter selection is unaccounted, and shuffling-vs-Poisson mismatch inflates the claimed guarantee. Until the search cost is priced, "adapter A beats full FT at $\varepsilon=8$" is not a well-defined comparison — the arms may sit at different true budgets.

## 6. Why It Is Hard

**Confounded measurement, primarily.** Three variables move together and no published protocol pins them: (1) trainable-parameter count $p$; (2) clipping bias, which depends on the *distribution* of $\|g_i\|$ and hence on parameterization and initialization, not just $p$; (3) the hyperparameter budget, since DP-SGD utility is far more sensitive to $(C, \eta, B, T)$ than non-private SGD, so the better-tuned arm wins regardless of the mechanism under test. A gap of one accuracy point is inside the tuning noise of most published comparisons.

**Non-identifiability, secondarily.** LoRA at rank $r$ changes $p$, the loss landscape, and the gradient-norm scale simultaneously. Any observed $\Delta$ can be attributed to noise dimension or to conditioning; nothing in the standard experiment separates them.

**Compute cost is real but not the blocker.** A full $p \times \varepsilon$ grid at 7B is roughly $10^3$ fine-tuning runs — expensive, not prohibitive.

## 7. Current Research (as of 2026)

- Low-cost per-example clipping (ghost clipping, book-keeping) at 7B+ scale — Bu, Wang and collaborators; largely a systems success. *(frontier — verify current scale records.)*
- Correct accounting for shuffled batches (Chua et al., ICML 2024) and for private hyperparameter tuning (Papernot & Steinke, ICLR 2022, random-stopping tuning) — the machinery to fix the methodological block exists; it is not yet standard practice in PEFT papers.
- One-run auditing applied per-parameterization, to measure whether adapters have *smaller realized* $\varepsilon$ than the analytic bound. *(frontier — verify.)*
- DP-adjacent alternatives: private synthetic-data generation then non-private adapter tuning; DP in-context learning with per-example privacy (Wu et al., 2023). *(frontier — verify.)*

## 8. Concrete Next Experiment

**Question.** At fixed $\varepsilon$ and matched tuning budget, is trainable-parameter count a first-order driver of DP fine-tuning utility?

**Scale.** One backbone (Llama-3.1-8B or Pythia-6.9B), one task with $n \approx 50{,}000$ private examples (e.g. MNLI or a de-identified clinical-note classification set). Five arms spanning $p$: bias-only ($\sim\!3\times10^5$), LoRA $r{=}2$ ($\sim\!2\times10^6$), LoRA $r{=}32$ ($\sim\!3\times10^7$), bottleneck adapter ($\sim\!10^8$), full fine-tuning ($\sim\!8\times10^9$). Budgets $\varepsilon \in \{0.5, 2, 8\}$, $\delta = 10^{-6}$, PLD accountant, Poisson sampling implemented as Poisson.

**Control arms.** (a) Identical grid of 32 hyperparameter trials per arm, with the tuning cost itself accounted via random-stopping private selection, so every arm reports the *same total* $\varepsilon$. (b) A dimension-matched sham: full fine-tuning with noise variance rescaled to $\sigma^2 C^2 p_{\text{LoRA}}/d$ — if this recovers LoRA's accuracy, the effect is noise dimension; if not, it is conditioning.

**Deciding number.** $\Delta(\varepsilon{=}2) = A_{\text{full}} - A_{\text{LoRA-}r{=}2}$ in accuracy points, with a 95% CI over 5 seeds. If $|\Delta| < 1.0$ point at all three $\varepsilon$, parameter count is not first-order and the field should stop citing $\sqrt{d}$ as the reason to use adapters under DP. If $\Delta > 3$ points at $\varepsilon = 0.5$ but not at $\varepsilon = 8$, the advantage is real and confined to the high-noise regime — the useful, currently unmeasured claim.

## 9. Key References

- **[Foundational]** M. Abadi, A. Chu, I. Goodfellow, H. B. McMahan, I. Mironov, K. Talwar, L. Zhang. *Deep Learning with Differential Privacy.* ACM CCS, 2016. — arXiv:1607.00133
- **[Foundational]** R. Bassily, A. Smith, A. Thakurta. *Private Empirical Risk Minimization: Efficient Algorithms and Tight Error Bounds.* FOCS, 2014. — arXiv:1405.7085
- **[SOTA]** X. Li, F. Tramèr, P. Liang, T. Hashimoto. *Large Language Models Can Be Strong Differentially Private Learners.* ICLR, 2022. — arXiv:2110.05679
- **[SOTA]** D. Yu, S. Naik, A. Backurs, S. Gopi, H. A. Inan, G. Kamath, J. Kulkarni, Y. T. Lee, A. Manoel, L. Wutschitz, S. Yekhanin, H. Zhang. *Differentially Private Fine-tuning of Language Models.* ICLR, 2022. — arXiv:2110.06500
- **[SOTA]** Z. Bu, Y.-X. Wang, S. Zha, G. Karypis. *Differentially Private Bias-Term Fine-tuning of Foundation Models.* ICML, 2024. — arXiv:2210.00036
- **[SOTA]** S. De, L. Berrada, J. Hayes, S. L. Smith, B. Balle. *Unlocking High-Accuracy Differentially Private Image Classification through Scale.* 2022. — arXiv:2204.13650
- **[Method]** T. Sander, P. Stock, A. Sablayrolles. *TAN Without a Burn: Scaling Laws of DP-SGD.* ICML, 2023. — arXiv:2210.03403
- **[Method]** N. Papernot, T. Steinke. *Hyperparameter Tuning with Renyi Differential Privacy.* ICLR, 2022. — arXiv:2110.03620
- **[Method]** T. Steinke, M. Nasr, M. Jagielski. *Privacy Auditing with One (1) Training Run.* NeurIPS, 2023. — arXiv:2305.08846
- **[Theory]** A. Ganesh, M. Haghifam, M. Nasr, S. Oh, T. Steinke, O. Thakkar, A. Thakurta, L. Wang. *Why Is Public Pretraining Necessary for Private Model Training?* ICML, 2023. — arXiv:2302.09483
- **[Accounting]** L. Chua, B. Ghazi, P. Kamath, R. Kumar, P. Manurangsi, A. Sinha, C. Zhang. *How Private are DP-SGD Implementations?* ICML, 2024. — arXiv:2403.17673
- **[Survey]** F. Tramèr, D. Boneh. *Differentially Private Learning Needs Better Features (or Much More Data).* ICLR, 2021. — arXiv:2011.11660

## 10. Worked Example

RoBERTa-large, SST-2, $n = 67{,}349$, $\varepsilon = 3$, $\delta = 10^{-6}$, $B = 1024$ ($q = 0.0152$), $E = 3$ epochs so $T \approx 197$ steps. RDP accounting gives $\sigma \approx 0.75$ at this $(q,T,\varepsilon)$.

Compare full fine-tuning ($p = 3.55\times10^{8}$) against LoRA $r{=}4$ ($p \approx 8\times10^{5}$). Per-step noise magnitude added to the averaged gradient is $\sigma C \sqrt{p}/B$. With $C=1$:

- Full: $0.75 \cdot \sqrt{3.55\times10^{8}} / 1024 \approx 13.8$.
- LoRA: $0.75 \cdot \sqrt{8\times10^{5}} / 1024 \approx 0.65$.

A 21× difference in injected noise norm. The naive prediction is that full fine-tuning is destroyed. Measured accuracy at $\varepsilon=3$ differs by roughly one point (Li et al., ICLR 2022).

**Where the naive calculation fails.** The isotropic noise norm is not the quantity that matters; only the component of noise inside the subspace where the clipped gradient signal lives degrades the update. If the per-example gradient covariance has $d_{\text{eff}} \approx 10^{3}$ — plausible for fine-tuning a pretrained backbone — the *in-subspace* noise is $\sigma C\sqrt{d_{\text{eff}}}/B \approx 0.023$ for both arms, and the 21× gap vanishes.

**The obstruction, made visible.** Nothing in the standard experiment measures $d_{\text{eff}}$. Both the "adapters help by $\sqrt{p/p'} = 21\times$" story and the "dimension is irrelevant because $d_{\text{eff}}$ is shared" story fit the same one-point gap. Distinguishing them needs the sham control of §8 — inject LoRA-magnitude noise into full fine-tuning — and a direct estimate of $\mathrm{tr}\,\Sigma_g$ and $\mathrm{tr}\,\Sigma_g^2$ from per-example gradients on a public split. Neither is standard, which is why the problem is only partially solved: the empirical answer ("adapters help little at $\varepsilon \ge 3$ with a public backbone") is fairly firm, while the reason, and hence the extrapolation to $\varepsilon < 1$ and to 100B-parameter backbones, is not.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*