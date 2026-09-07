---
id: 23-privacy-memorization/amplification-shuffling-realistic-sampling
title: "Privacy Amplification by Shuffling Under Realistic Sampling"
topic: 23-privacy-memorization
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Privacy Amplification by Shuffling Under Realistic Sampling

> **Topic:** Privacy & Memorization · **ID:** `23-privacy-memorization/amplification-shuffling-realistic-sampling` · **Status:** partially-solved

## 1. Problem Statement

Shuffling amplifies privacy: if $n$ users each send the output of an $\varepsilon_0$-local randomizer to a trusted shuffler, the multiset of messages satisfies central $(\varepsilon, \delta)$-DP with $\varepsilon \approx \varepsilon_0\sqrt{\log(1/\delta)/n}$ — a $\sqrt{n}$ gain. The theorems establishing this assume an idealized sampling model: a fixed, known population, exactly one contribution per user, and independent participation.

Real deployments violate all three. DP-SGD implementations shuffle a dataset once per epoch and cut it into fixed-size batches, then account for privacy using the Poisson-subsampling amplification bound because that is what the accountant library implements. Federated systems sample clients by device availability (charging, on Wi-Fi, idle), which is neither uniform nor independent across rounds, and clients participate repeatedly across an unknown number of rounds.

Three variants, in increasing difficulty:

- **Measurement.** Given a deployed pipeline (shuffle-and-batch DP-SGD, or FL with availability-driven check-in), what is the *true* $(\varepsilon, \delta)$? Reported vs. audited.
- **Method.** Build an accountant that is tight for the sampling scheme actually used — deterministic batch sizes, correlated participation, multiple contributions — rather than for Poisson subsampling.
- **Theory.** Prove matching upper and lower bounds on shuffle amplification when the participation process is adversarially correlated but has known marginals.

Solving it means: a certified $\varepsilon$ for shuffle-based training whose gap to an empirical audit lower bound is under a factor of 2, at ImageNet/LLM fine-tuning scale.

## 2. Formal Setting

**Objects.** Dataset $D = (x_1,\dots,x_N)$. Local randomizer $R: \mathcal{X} \to \mathcal{Y}$ satisfying $\varepsilon_0$-LDP. Shuffler $\mathcal{S}$ applies a uniform random permutation to a message multiset. The shuffled mechanism is $M(D) = \mathcal{S}(R(x_{i_1}),\dots,R(x_{i_n}))$ over a participating set $I$ of size $n$.

**Participation process.** The realistic object is a random set-valued process $\{I_t\}_{t=1}^{T}$ over rounds. Define per-user marginal $p_i = \Pr[i \in I_t]$ and the batch-size law $|I_t|$. Three regimes:

$$\text{Poisson: } I_t \sim \mathrm{Bern}(q)^{\otimes N},\quad \text{Shuffle-batch: } |I_t| = B \text{ exactly},\quad \text{Availability: } \Pr[i \in I_t] = p_i(t,\text{device state}).$$

**Amplification bound (Feldman–McMillan–Talwar, FOCS 2021).** For $n$ users each running an $\varepsilon_0$-LDP randomizer, the shuffled output is $(\varepsilon,\delta)$-DP with

$$\varepsilon \le \log\!\left(1 + \frac{e^{\varepsilon_0}-1}{e^{\varepsilon_0}+1}\left(\frac{8\sqrt{e^{\varepsilon_0}\log(4/\delta)}}{\sqrt{n}} + \frac{8 e^{\varepsilon_0}}{n}\right)\right).$$

**Measured quantities.**
- $\varepsilon_{\text{rep}}$: the number the accountant prints. Measured by running the library (Poisson RDP/PLD accountant) with the configured $q = B/N$, noise $\sigma$, and $T$ steps.
- $\varepsilon_{\text{true}}$: the tightest provable bound for the *actual* sampler. Usually unavailable; substituted by a numerically computed dominating pair when one exists.
- $\hat\varepsilon_{\text{emp}}$: an audit lower bound. Measured by inserting a canary, running $K$ paired trials, computing a Clopper–Pearson confidence interval on (TPR, FPR) of the membership test, and inverting $\varepsilon \ge \log\frac{\mathrm{TPR}-\delta}{\mathrm{FPR}}$.
- The decision statistic is the **gap ratio** $G = \hat\varepsilon_{\text{emp}} / \varepsilon_{\text{rep}}$. $G > 1$ means the reported guarantee is false, not merely loose.

**Assumptions known to be violated.**
1. *One contribution per user per mechanism.* Violated: $T$ epochs $\Rightarrow$ $T$ contributions; composition over rounds assumes independence that shuffling breaks.
2. *Poisson-independent inclusion.* Violated by fixed-size batches (negative correlation) and by device availability (positive correlation across rounds — the same reliable devices check in).
3. *Known population $n$.* Violated in FL: cohort size is realized, not chosen, and $\varepsilon \propto 1/\sqrt{n}$ makes the bound sensitive to it.
4. *Trusted shuffler with no timing leakage.* Violated by network metadata; addressed separately by shuffling-without-shuffling protocols.

## 3. State of the Art

**Theory SOTA (established).**
- Erlingsson, Feldman, Mironov, Raghunathan, Talwar, Thakurta (SODA 2019): first $O(\varepsilon_0\sqrt{\log(1/\delta)/n})$ amplification.
- Balle, Bell, Gascón, Nissim (CRYPTO 2019): "privacy blanket" decomposition, tight constants for the single-message model.
- Feldman, McMillan, Talwar (FOCS 2021): near-optimal analysis, the bound quoted above, with matching lower bound up to constants.
- Feldman, McMillan, Talwar (SODA 2023): Rényi-DP and approximate-DP amplification, which is what composition over $T$ rounds actually needs.

**Systems/empirical SOTA (established).**
- Chua, Ghazi, Kamath, Kumar, Manurangsi, Sinha, Zhang, *How Private are DP-SGD Implementations?* (ICML 2024): formalizes adaptive batch linear queries (ABLQ) under Deterministic, Shuffle, and Poisson batch samplers, and shows $\varepsilon_{\mathcal{S}} \gg \varepsilon_{\mathcal{P}}$ — shuffle-batch DP-SGD accounted as Poisson understates $\varepsilon$, by orders of magnitude in the high-privacy regime.
- Chua et al., *Scalable DP-SGD: Shuffling vs. Poisson Subsampling* (NeurIPS 2024): lower bounds on the shuffle-sampler privacy loss plus a scalable Poisson implementation, showing the fix costs utility.
- Annamalai, Balle, De Cristofaro, Hayes, *To Shuffle or not to Shuffle: Auditing DP-SGD with Shuffling* (2024): empirical audits of shuffle-based DP-SGD produce lower bounds exceeding the Poisson-reported $\varepsilon$ — the first $G > 1$ demonstration on a real training pipeline.

**Claimed but unablated.** Federated amplification results that combine client subsampling with shuffling (Girgis, Data, Diggavi, Kairouz, Suresh, AISTATS 2021) assume independent client sampling with a known rate; no deployment ablation shows the bound survives availability-correlated check-in. Vendor accountants that expose a "shuffle" flag typically apply a single-round shuffle bound and then compose — this composition step is asserted, not proven tight, and exists only as a library default.

**Benchmark-number-only results.** Reported CIFAR-10/ImageNet DP-SGD accuracies at $\varepsilon = 8$ are almost all produced by shuffle-batch loaders with Poisson accounting. They are utility numbers attached to an $\varepsilon$ that does not describe the mechanism run.

## 4. What Is Known

- **The $\sqrt{n}$ gain is real and near-optimal in the idealized model.** FMT (FOCS 2021) is tight up to constant factors; for $\varepsilon_0 = 4$, $n = 10^6$, $\delta = 10^{-8}$, the bound gives central $\varepsilon \approx 0.3$ — roughly a $13\times$ reduction.
- **The gain degrades as $\varepsilon_0$ grows.** Amplification requires $\varepsilon_0 \lesssim \log(n/\log(1/\delta))$; beyond that, the $e^{\varepsilon_0}/n$ term dominates and shuffling buys nothing.
- **Poisson and shuffle samplers are not interchangeable.** Chua et al. (ICML 2024) prove separation for ABLQ: the two samplers have genuinely different dominating pairs, and the shuffle one is worse in the regimes DP-SGD is run in (small $q$, many steps).
- **Audits confirm the direction of the error.** Shuffle-trained models audit above their Poisson-reported $\varepsilon$; Poisson-trained models audit below it. Measured at CIFAR-10 / WideResNet scale with thousands of paired canary runs (Annamalai et al., 2024). Exact ratios are setting-dependent — read the paper's tables rather than a single headline figure.
- **Amplification can be avoided.** DP-FTRL (Kairouz, McMahan, Song, Thakkar, Thakurta, Xu, ICML 2021) attains competitive utility with no sampling assumption at all, and is what Google deployed for Gboard — evidence that the assumption-free path is practical, not just safe.

## 5. What Is Not Known

- **Theoretically open.** A tight $T$-round composition bound for the shuffle sampler with repeated participation. Single-round shuffle amplification is settled; the composed object under fixed-size batches and epoch-level shuffling has upper and lower bounds separated by an unquantified gap. Also open: amplification under adversarially correlated participation with fixed marginals $\{p_i\}$ — no bound either way.
- **Empirically open.** The size of $G$ at frontier scale. Every audit is at CIFAR-scale with small models. Whether $G$ at billion-parameter LLM fine-tuning is 1.1 or 5 is runnable and unrun — it costs GPU-months, not new theory.
- **Methodologically blocked.** "Realistic sampling" has no agreed formal model. Device availability traces are proprietary; there is no public benchmark participation process. Without one, an accountant cannot be evaluated against the distribution it claims to cover, and two papers claiming to handle "realistic" sampling need not be comparable.

## 6. Why It Is Hard

The obstruction is **non-identifiability of the dominating pair combined with audit cost**.

Privacy accounting reduces to finding a dominating pair of distributions $(P,Q)$ whose hockey-stick divergence upper-bounds the mechanism's. For Poisson subsampling this pair is a two-component mixture with a closed form. For shuffle-and-batch with $T$ epochs, the loss depends on the joint law of *which* batch each example landed in across all epochs; the induced pair is a high-dimensional mixture with no closed form and no known small sufficient statistic. Numerical PLD accountants need a dominating pair as input, so the tooling cannot be pointed at the problem.

The empirical fallback is auditing, but audits produce *lower* bounds and their tightness scales as $O(1/\sqrt{K})$ in the number of trials $K$. Distinguishing $G = 1.0$ from $G = 1.5$ at $\varepsilon_{\text{rep}} = 8$ needs $K \sim 10^4$ full training runs unless one uses one-run auditing with many canaries, which itself relies on an independence assumption that shuffling breaks. So the theory cannot compute the number and the experiment that would measure it is confounded by the same correlation it is trying to detect.

## 7. Current Research (as of 2026)

- **Tight samplers-aware accounting.** Google Research (Chua, Ghazi, Kumar, Manurangsi et al.) continues on ABLQ lower bounds and scalable Poisson implementations that remove the discrepancy by changing the sampler rather than the analysis.
- **Auditing under correlated participation.** UCL / Google DeepMind (Annamalai, Balle, De Cristofaro, Hayes) on shuffle-aware audits and one-run auditing validity.
- **Shuffling without a trusted shuffler.** Balle, Bell, Gascón (CCS 2023) — amplification from secure aggregation primitives, removing the trust assumption while inheriting the sampling one.
- **Assumption-free training.** DP-FTRL / matrix-factorization mechanisms; the practical answer if the gap cannot be closed. *(frontier — verify)* Correlated-noise mechanisms with amplification claims under fixed-size batches are appearing; treat their amplification factors as unaudited.

## 8. Concrete Next Experiment

**Question.** At production scale, is $G = \hat\varepsilon_{\text{emp}}/\varepsilon_{\text{rep}} > 1$ for shuffle-batch DP-SGD accounted as Poisson?

**Scale.** Fine-tune a 1.4B-parameter decoder on a 500k-example instruction set, $B = 1024$, $T = 3$ epochs, $\sigma$ chosen so the Poisson accountant reports $\varepsilon_{\text{rep}} = 8$ at $\delta = 10^{-6}$. Use one-run auditing with 2,000 inserted canaries plus 50 paired full runs for calibration.

**Control arm.** Identical training with a true Poisson sampler (variable batch size, mean $B$), same $\sigma$, same canaries, same audit. This is the arm where the accountant is correct, so it calibrates audit slack.

**Deciding number.** $\Delta = \hat\varepsilon_{\text{emp}}^{\text{shuffle}} - \hat\varepsilon_{\text{emp}}^{\text{Poisson}}$, with 95% Clopper–Pearson intervals. If the interval on $\Delta$ excludes 0 and $\hat\varepsilon_{\text{emp}}^{\text{shuffle}} > 8$, then every published shuffle-batch DP-SGD $\varepsilon$ at this scale is invalid, and the fix is a sampler change. If $\Delta \le 0.5$ and $\hat\varepsilon_{\text{emp}}^{\text{shuffle}} < 8$, the discrepancy is a small-$\varepsilon$ artifact and current practice is defensible at $\varepsilon = 8$.

Cost estimate: ~60 fine-tuning runs, roughly 3–5 GPU-months on A100-class hardware.

## 9. Key References

- **[Foundational]** Erlingsson, Feldman, Mironov, Raghunathan, Talwar, Thakurta. *Amplification by Shuffling: From Local to Central Differential Privacy via Anonymity.* SODA, 2019. — arXiv:1811.12469
- **[Foundational]** Cheu, Smith, Ullman, Zeber, Zhilyaev. *Distributed Differential Privacy via Shuffling.* EUROCRYPT, 2019. — arXiv:1808.01394
- **[Foundational]** Balle, Bell, Gascón, Nissim. *The Privacy Blanket of the Shuffle Model.* CRYPTO, 2019. — arXiv:1903.02837
- **[SOTA]** Feldman, McMillan, Talwar. *Hiding Among the Clones: A Simple and Nearly Optimal Analysis of Privacy Amplification by Shuffling.* FOCS, 2021. — arXiv:2012.12803
- **[SOTA]** Feldman, McMillan, Talwar. *Stronger Privacy Amplification by Shuffling for Rényi and Approximate Differential Privacy.* SODA, 2023.
- **[SOTA]** Chua, Ghazi, Kamath, Kumar, Manurangsi, Sinha, Zhang. *How Private are DP-SGD Implementations?* ICML, 2024. — arXiv:2403.17673
- **[SOTA]** Chua, Ghazi, Kamath, Kumar, Manurangsi, Sinha, Zhang. *Scalable DP-SGD: Shuffling vs. Poisson Subsampling.* NeurIPS, 2024.
- **[SOTA]** Annamalai, Balle, De Cristofaro, Hayes. *To Shuffle or not to Shuffle: Auditing DP-SGD with Shuffling.* 2024.
- **[Method]** Kairouz, McMahan, Song, Thakkar, Thakurta, Xu. *Practical and Private (Deep) Learning without Sampling or Shuffling.* ICML, 2021. — arXiv:2103.00039
- **[Method]** Balle, Barthe, Gaboardi. *Privacy Amplification by Subsampling: Tight Analyses via Couplings and Divergences.* NeurIPS, 2018. — arXiv:1807.01647
- **[Survey]** Kairouz, McMahan, et al. *Advances and Open Problems in Federated Learning.* FnTML, 2021. — arXiv:1912.04977

## 10. Worked Example

Take $N = 50{,}000$ (CIFAR-10), $B = 500$ so $q = 0.01$, $T = 5{,}000$ steps (50 epochs), $\sigma = 1.0$, $\delta = 10^{-5}$.

**Step 1 — what the accountant says.** A PLD accountant for Poisson subsampled Gaussian at $q=0.01$, $\sigma=1.0$, $T=5000$ returns roughly $\varepsilon_{\text{rep}} \approx 7$. This is the number that appears in the paper.

**Step 2 — what was actually run.** The loader called `torch.randperm(50000)` once per epoch and sliced 100 contiguous batches. Every example appears exactly once per epoch, in exactly one batch. Inclusion across the 100 batches of an epoch is a *negatively correlated* multivariate hypergeometric draw, not 100 independent $\mathrm{Bern}(0.01)$ draws.

**Step 3 — where the bound breaks.** Poisson accounting's strength comes from the event "the target is absent," probability $1-q = 0.99$ per step, and independence across steps means the absence pattern over 5,000 steps carries $\approx 5000 \cdot H(0.01) \approx 400$ bits of hiding entropy. Under shuffling the target is present exactly 50 times — once per epoch — with certainty. The adversary knows the *count*; only the *positions* are hidden, and only $100^{50}$ position choices exist, i.e. $\approx 332$ bits. The mechanism has strictly less randomness to hide in than the accountant assumed, and the deficit is not a constant — it grows with $T$.

**Step 4 — the obstruction made visible.** Now try to convert that into a number. To compute $\varepsilon_{\text{true}}$ you need the hockey-stick divergence of the mixture over all $100^{50}$ placement patterns, where the loss in each pattern depends on the gradient state at those specific steps under adaptive composition. There is no closed form, no known low-dimensional sufficient statistic, and Monte Carlo over $10^{100}$ patterns is not a plan. So you audit instead — and the audit's own confidence interval at $K = 100$ runs is roughly $\pm 1.5$ in $\varepsilon$ near $\varepsilon = 7$, which is wider than the effect you are trying to detect.

That is the problem in one instance: an entropy deficit you can see and count in bits, that neither the theory can price nor the affordable experiment can resolve.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*