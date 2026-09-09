---
id: 23-privacy-memorization/dpsgd-analysis-empirical-leakage-gap
title: "The Gap Between DP-SGD Analysis and Empirical Leakage"
topic: 23-privacy-memorization
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# The Gap Between DP-SGD Analysis and Empirical Leakage

> **Topic:** Privacy & Memorization · **ID:** `23-privacy-memorization/dpsgd-analysis-empirical-leakage-gap` · **Status:** open

## 1. Problem Statement

DP-SGD (Abadi et al., CCS 2016) ships two numbers that disagree. The **analytical** $\varepsilon$ comes from composing per-step Gaussian mechanisms under an assumed Poisson-subsampled, full-batch-gradient-visible adversary. The **empirical** $\varepsilon_{\mathrm{emp}}$ comes from running an attack and converting its true/false positive rates into a differential-privacy lower bound. In realistic training runs $\varepsilon_{\mathrm{emp}} \ll \varepsilon$, often by an order of magnitude. The problem is to explain and close that gap.

Three variants, with different difficulty:

- **Measurement.** Given a fixed training pipeline, produce a *tight* $\varepsilon_{\mathrm{emp}}$ — a lower bound close to the true privacy loss of that pipeline, not of its worst-case abstraction. Solved would mean: an auditing procedure whose output moves when the real leakage moves, and whose gap to the truth is bounded.
- **Method.** Produce an accountant that is tight for the *deployed* algorithm — shuffled (not Poisson) batches, fixed batch size, no gradient release, correlated noise — so that the shipped $\varepsilon$ is not simultaneously loose (against real adversaries) and unsound (against the shuffling assumption).
- **Theory.** Prove upper bounds on the privacy loss of DP-SGD under restricted adversaries (final-weights-only, black-box query access, natural-data neighbouring pairs) that are provably below the worst-case composition bound, or prove no such improvement is possible.

## 2. Formal Setting

Dataset $D = \{z_1,\dots,z_n\}$. DP-SGD runs $T$ steps; at step $t$ it draws a batch $B_t$, computes per-example gradients $g_i = \nabla \ell(\theta_t, z_i)$, clips them to $\ell_2$ norm $C$ via $\bar g_i = g_i \cdot \min(1, C/\|g_i\|_2)$, and updates

$$\theta_{t+1} = \theta_t - \eta \Big( \tfrac{1}{B}\textstyle\sum_{i \in B_t} \bar g_i + \tfrac{1}{B}\,\mathcal{N}(0, \sigma^2 C^2 I_d) \Big).$$

**Analytical $\varepsilon$ as measured.** Not measured at all — computed. Fix $\delta$ (usually $10^{-5}$ or $1/n$), sampling rate $q = B/n$, noise multiplier $\sigma$, steps $T$; feed to a numerical accountant (Gopi–Lee–Wutschitz, NeurIPS 2021, or a Rényi accountant, Mironov CSF 2017) to get $\varepsilon(\delta)$ for the *sequence of released iterates* $(\theta_1,\dots,\theta_T)$.

**Empirical $\varepsilon_{\mathrm{emp}}$ as measured.** Choose neighbouring datasets $D, D'$ differing in one canary $z^*$. Train $K$ models, half on each. Run a membership test producing scores; at threshold $\tau$ measure $\mathrm{TPR}(\tau)$ and $\mathrm{FPR}(\tau)$, then

$$\varepsilon_{\mathrm{emp}} = \max_\tau \ \log \frac{\mathrm{TPR}^-(\tau) - \delta}{\mathrm{FPR}^+(\tau)},$$

where $\mathrm{TPR}^-$ / $\mathrm{FPR}^+$ are Clopper–Pearson confidence limits at level $1-\alpha$ (typically $\alpha = 0.05$). With $K$ runs the *statistically attainable* ceiling is roughly $\log K$ — you cannot audit $\varepsilon = 8$ with 100 runs. The one-run auditor (Steinke, Nasr, Jagielski, NeurIPS 2023) replaces $K$ trainings with $m$ independently-included canaries in one training run and applies the same construction to per-canary scores.

The gap is $\Delta = \varepsilon - \varepsilon_{\mathrm{emp}}$, and it decomposes into: (i) statistical slack from finite $K$ or $m$; (ii) adversary weakness (the attack is not the likelihood-ratio optimal test); (iii) threat-model slack (final weights released, not all iterates; natural neighbours, not adversarially chosen); (iv) genuine looseness of the accountant.

**Assumptions known to be violated in practice.**
- *Poisson subsampling.* Real pipelines shuffle and take fixed-size batches. Chua et al. (ICML 2024) show the shuffled algorithm's true $\varepsilon$ can substantially exceed the Poisson-accounted number in some regimes — so the shipped $\varepsilon$ is not merely loose, it can be **unsound**.
- *Full iterate release.* The accountant charges for every $\theta_t$; deployments release only $\theta_T$.
- *Worst-case neighbours.* The bound holds for a maximally-influential inserted example; audits with natural data are far weaker.
- *Exact arithmetic.* Floating-point Gaussian sampling has known discretization attacks (Mironov, CCS 2012).
- *No side channels.* Deduplication, retrieval caches, and system-level state break the unit-of-privacy accounting (Debenedetti et al., USENIX Security 2024).

## 3. State of the Art

**Theory SOTA (established).** Numerical accountants (Gopi–Lee–Wutschitz 2021; Doroshenko et al. PLD accounting) are tight *for the composition of subsampled Gaussians with all iterates released* — the looseness is not in the composition arithmetic. Hidden-state analyses (privacy amplification by iteration, Feldman et al. FOCS 2018; Altschuler & Talwar, 2022) give improved bounds when only $\theta_T$ is released, but require convexity/smoothness that deep networks violate. This is the sharpest statement of the gap: for convex losses we can prove the final-iterate bound is better; for the non-convex case used in practice, we cannot.

**Empirical SOTA (established).** Nasr et al. (S&P 2021) instantiate adversaries at increasing strength and show that only the *gradient-canary, all-iterates, adversarially-crafted-dataset* adversary approaches the analytical $\varepsilon$; every weaker adversary falls far short. Nasr et al. (USENIX Security 2023) achieve essentially tight audits under that strongest model. Annamalai & De Cristofaro (2024) push black-box auditing considerably closer to the analytical bound by crafting worst-case initializations — showing much of the reported black-box slack was adversary weakness, not a real privacy surplus.

**Claimed but unablated.** The recurring practitioner claim that "$\varepsilon = 8$ is fine because attacks only recover $\varepsilon_{\mathrm{emp}} \approx 1$" is unablated in the direction that matters: it has not been shown that the attack is near-optimal for that pipeline. Aerni, Zhang & Tramèr (CCS 2024) show that privacy-defense evaluations reporting low leakage are systematically misleading when averaged over examples rather than measured on the most vulnerable ones.

**Benchmark-number-only results.** Utility claims — e.g. high-accuracy private ImageNet classification at moderate $\varepsilon$ (De et al., 2022) — are single-setting benchmark numbers; they say nothing about the audit gap.

## 4. What Is Known

- **Composition is not the culprit.** PLD/Rényi accountants match the true worst-case privacy loss of the released-iterate mechanism to numerical precision (Gopi et al., NeurIPS 2021).
- **The audit gap is adversary-dependent, at CIFAR-10 / WRN-16-4 scale.** In white-box, gradient-canary settings, audits recover a large fraction of the analytical $\varepsilon$; in final-model black-box settings with natural canaries, reported $\varepsilon_{\mathrm{emp}}$ is commonly a factor of $\sim$2–5 below the analytical $\varepsilon$ at $\varepsilon \in [4,10]$ (Jagielski et al., NeurIPS 2020; Steinke et al., NeurIPS 2023; Annamalai & De Cristofaro, 2024). Treat the exact ratios as setting-specific, not universal constants.
- **One-run auditing works.** Steinke et al. (2023) obtain non-trivial lower bounds from a single training run with $m \approx 1000$ canaries, removing the $K$-runs compute barrier that previously capped audits at small $\varepsilon$.
- **Memorization without DP is severe and scale-increasing.** Carlini et al. (ICLR 2023) find extractable memorization grows log-linearly in model size, data duplication, and prompt-context length, measured on GPT-Neo 125M–6B over the Pile.
- **Some memorization is necessary.** Feldman (STOC 2020) proves that for long-tailed label distributions, near-optimal generalization *requires* memorizing singleton examples — so any DP guarantee tight enough to forbid it costs accuracy.
- **The deployed algorithm is not the analyzed one.** Chua et al. (ICML 2024): shuffling-based DP-SGD, accounted as if Poisson-subsampled, can have a materially larger true $\varepsilon$.

## 5. What Is Not Known

- **Theoretically open.** Whether the final-iterate privacy loss of DP-SGD on non-convex objectives is asymptotically smaller than the released-iterate composition bound. No proof either way; the convex results do not transfer, and no matching lower bound (a non-convex loss where the composition bound is tight for final weights) is known.
- **Theoretically open.** A tight accountant for shuffled fixed-size batching at practical $q, T$ — only bounds and regime-specific separations exist.
- **Empirically open.** Whether the audit gap persists at LLM pretraining scale ($10^9$+ parameters, $10^{11}$+ tokens, one epoch). Every tight audit to date is at image-classification scale. The experiment is runnable; it has not been run at that scale.
- **Methodologically blocked.** "Empirical leakage" for generative models is not a well-defined quantity. Extraction rate, verbatim $n$-gram overlap, and membership AUC are different objects, and none converts to a DP $\varepsilon$ without assuming the attack is optimal. There is no measurement that is simultaneously threat-model-faithful and computable.

## 6. Why It Is Hard

The obstruction is **non-identifiability of the gap's cause**, compounded by an **evaluation that does not measure what it names**.

Observing $\varepsilon_{\mathrm{emp}} = 1$ under an analytical $\varepsilon = 8$ is consistent with two incompatible worlds: (a) the pipeline genuinely leaks like $\varepsilon \approx 1$, or (b) it leaks like $\varepsilon \approx 8$ and the attack is weak. An audit produces only a *lower* bound. No amount of attacking distinguishes the cases, because the missing object is an upper bound under the restricted threat model — precisely the thing that is theoretically open. Annamalai & De Cristofaro's result is the empirical demonstration: gaps previously read as world (a) collapsed once the adversary improved.

Secondary obstructions, both concrete: statistical slack caps $\varepsilon_{\mathrm{emp}}$ at $\approx \log K$, so auditing $\varepsilon = 8$ classically needs $K \gtrsim 10^4$ trainings; and audits report *average* leakage while the quantity of interest is the worst-case example, which the average understates by orders of magnitude on long-tailed data.

## 7. Current Research (as of 2026)

- **One-run and few-run auditing** — Steinke, Nasr, Jagielski (Google DeepMind / Google Research), extended to federated and black-box settings.
- **Accounting for the deployed algorithm** — shuffling-vs-Poisson accounting and DP-FTRL-style correlated-noise mechanisms (Google; Kairouz, McMahan and collaborators).
- **Adversarial auditing that closes the black-box gap** — Annamalai & De Cristofaro (UCL); Tramèr's group (ETH Zürich) on evaluation validity and worst-case-example reporting.
- **Hidden-state analysis beyond convexity** — Altschuler, Talwar, Feldman and collaborators. *(frontier — verify)* No non-convex final-iterate improvement has been established.
- **Auditing DP fine-tuning of production-scale LLMs** *(frontier — verify)*: reported informally; no peer-reviewed tight audit at $\ge 7$B parameters known to this catalog.

## 8. Concrete Next Experiment

**Question.** At LLM fine-tuning scale, is the black-box audit gap real privacy surplus or adversary weakness?

**Scale.** Fine-tune a 1.4B-parameter open model (e.g. Pythia-1.4B) on a 200M-token instruction corpus with DP-SGD at analytical $\varepsilon = 8$, $\delta = 10^{-6}$, clipping $C = 1.0$, batch size $2^{18}$ tokens, $T \approx 2{,}000$ steps. Insert $m = 2{,}000$ canary documents, each included independently with probability $1/2$, per the one-run auditor. One training run, roughly 3–5k A100-hours including arms.

**Arms.**
1. *Natural canaries*, black-box scoring (per-token loss on the canary given a prefix).
2. *Crafted canaries* — out-of-distribution high-gradient-norm sequences — same black-box scoring.
3. **Control arm:** identical pipeline with $\sigma = 0$ (no noise, clipping only), same canaries, same auditor. This calibrates the auditor's statistical ceiling and the attack's power on a mechanism with no privacy guarantee at all.

**Deciding number.** The ratio $\rho = \varepsilon_{\mathrm{emp}}^{\text{arm 2}} / 8$.
- $\rho \ge 0.5$: the black-box gap at this scale is adversary weakness. $\varepsilon = 8$ should be read as a real, near-attained bound; practitioner reliance on the audit gap is unsafe.
- $\rho \le 0.15$ **while arm 3 attains $\varepsilon_{\mathrm{emp}} \ge 6$** (proving the auditor is not the bottleneck): there is a genuine, scale-persistent surplus, and the theory target is a final-iterate upper bound for non-convex DP-SGD.
- $\rho \le 0.15$ with arm 3 also low: the result is uninformative — the auditor, not the mechanism, is the limit. This is the outcome most existing published gaps have not excluded.

## 9. Key References

- **[Foundational]** M. Abadi, A. Chu, I. Goodfellow, H. B. McMahan, I. Mironov, K. Talwar, L. Zhang. *Deep Learning with Differential Privacy.* ACM CCS, 2016. — arXiv:1607.00133
- **[Foundational]** C. Dwork, F. McSherry, K. Nissim, A. Smith. *Calibrating Noise to Sensitivity in Private Data Analysis.* TCC, 2006.
- **[Foundational]** V. Feldman. *Does Learning Require Memorization? A Short Tale about a Long Tail.* STOC, 2020. — arXiv:1906.05271
- **[SOTA]** T. Steinke, M. Nasr, M. Jagielski. *Privacy Auditing with One (1) Training Run.* NeurIPS, 2023. — arXiv:2305.08846
- **[SOTA]** M. Nasr, J. Hayes, T. Steinke, B. Balle, F. Tramèr, M. Jagielski, N. Carlini, A. Terzis. *Tight Auditing of Differentially Private Machine Learning.* USENIX Security, 2023. — arXiv:2302.07956
- **[SOTA]** M. Nasr, S. Songi, A. Thakurta, N. Papernot, N. Carlini. *Adversary Instantiation: Lower Bounds for Differentially Private Machine Learning.* IEEE S&P, 2021. — arXiv:2101.04535
- **[SOTA]** M. A. Annamalai, E. De Cristofaro. *Nearly Tight Black-Box Auditing of Differentially Private Machine Learning.* NeurIPS, 2024. — arXiv:2405.14106
- **[SOTA]** L. Chua, B. Ghazi, P. Kamath, R. Kumar, P. Manurangsi, A. Sinha, C. Zhang. *How Private are DP-SGD Implementations?* ICML, 2024. — arXiv:2403.17673
- **[Key]** M. Jagielski, J. Ullman, A. Oprea. *Auditing Differentially Private Machine Learning: How Private is Private SGD?* NeurIPS, 2020. — arXiv:2006.07709
- **[Key]** S. Gopi, Y. T. Lee, L. Wutschitz. *Numerical Composition of Differential Privacy.* NeurIPS, 2021. — arXiv:2106.02848
- **[Key]** N. Carlini, D. Ippolito, M. Jagielski, K. Lee, F. Tramèr, C. Zhang. *Quantifying Memorization Across Neural Language Models.* ICLR, 2023. — arXiv:2202.07646
- **[Key]** N. Carlini, S. Chien, M. Nasr, S. Song, A. Terzis, F. Tramèr. *Membership Inference Attacks From First Principles.* IEEE S&P, 2022. — arXiv:2112.03570
- **[Key]** M. Aerni, J. Zhang, F. Tramèr. *Evaluations of Machine Learning Privacy Defenses are Misleading.* ACM CCS, 2024. — arXiv:2404.17399
- **[Survey]** N. Ponomareva, H. Hazimeh, A. Kurakin, Z. Xu, C. Denison, H. B. McMahan, S. Vassilvitskii, S. Chien, A. Thakurta. *How to DP-fy ML: A Practical Guide to Machine Learning with Differential Privacy.* JAIR, 2023. — arXiv:2303.00654

## 10. Worked Example

Fix a CIFAR-10-scale run: $n = 50{,}000$, $B = 500$ so $q = 0.01$, $T = 2{,}000$ steps (20 epochs), $\sigma = 1.1$, $\delta = 10^{-5}$. A numerical accountant returns roughly $\varepsilon \approx 8$ for the released-iterate mechanism.

Now audit it classically with $K = 500$ paired training runs. Suppose the attack achieves $\mathrm{TPR} = 0.80$ at $\mathrm{FPR} = 0.20$ — a strong-looking attack. Point estimate:

$$\log\frac{0.80}{0.20} = \log 4 \approx 1.39.$$

Apply Clopper–Pearson at $\alpha = 0.05$ with 250 positives and 250 negatives: $\mathrm{TPR}^- \approx 0.745$, $\mathrm{FPR}^+ \approx 0.256$, giving

$$\varepsilon_{\mathrm{emp}} \approx \log\frac{0.745 - 10^{-5}}{0.256} \approx 1.07.$$

Reported as "$\varepsilon = 8$ but only $\varepsilon_{\mathrm{emp}} = 1.07$", an 8× gap.

**Where the obstruction becomes visible.** Ask what $\varepsilon_{\mathrm{emp}}$ *could* have been. With 250 samples per arm, even a perfect attack ($\mathrm{TPR} = 1$, $0$ false positives) yields $\mathrm{FPR}^+ \approx 3/250 = 0.012$ and $\mathrm{TPR}^- \approx 0.988$, so

$$\varepsilon_{\mathrm{emp}}^{\max} \approx \log\frac{0.988}{0.012} \approx 4.4.$$

The audit could never have reported more than 4.4 against an analytical 8. So of the 8× gap, a factor of $\approx 1.8$× is pure statistical slack from $K$, and the remainder is unattributable between adversary weakness and genuine privacy surplus — the two worlds of §6. Reaching $\varepsilon_{\mathrm{emp}} = 8$ needs $\mathrm{FPR}^+ \lesssim e^{-8} \approx 3 \times 10^{-4}$, hence $K \gtrsim 10^4$ runs, roughly $20\times$ the compute of the experiment above and about $10^4$ full trainings for a single point on a single hyperparameter setting.

That is the whole problem in one calculation: the headline gap is partly an artifact of the audit's sample size, the residual is non-identifiable without an upper bound nobody has proved, and the brute-force route to identifying it costs $10^4$ trainings per configuration.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*