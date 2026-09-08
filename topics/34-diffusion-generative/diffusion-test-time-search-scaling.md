---
id: 34-diffusion-generative/diffusion-test-time-search-scaling
title: "Test-Time Scaling Laws for Generative Search Over Noise"
topic: 34-diffusion-generative
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Test-Time Scaling Laws for Generative Search Over Noise

> **Topic:** Diffusion & Generative Modeling · **ID:** `34-diffusion-generative/diffusion-test-time-search-scaling` · **Status:** empirically-open

## 1. Problem Statement

A diffusion or flow model turns a noise draw $z \sim \mathcal{N}(0, I)$ into a sample $x = \Phi(z)$. Two knobs spend test-time compute: the number of denoising steps (NFE per sample), and the number of noise seeds searched over with a verifier that scores candidates. The first saturates — past roughly 100–250 steps, sample quality is flat. The second, "generative search over noise," does not obviously saturate.

The problem: **does search over noise obey a scaling law, and in what quantity?**

- **Measurement variant.** Given a budget $N$ (total denoising forward passes), a search algorithm $\mathcal{A}$, and a *quality functional the verifier does not define*, does $\mathbb{E}[Q]$ follow a power law or log-law in $N$, and does the exponent depend on the model, the verifier, or the task? Nobody has measured this on a held-out quality metric across two decades of $N$.
- **Method variant.** Which allocation of a fixed budget — more steps, more seeds, more search depth within a seed's trajectory — is compute-optimal? Is there a diffusion analogue of the Chinchilla frontier?
- **Theory variant.** Prove an upper bound on achievable quality gain from selecting among $n$ i.i.d. model samples, as a function of the model's tail behaviour and the verifier's noise. Is the gain $\Theta(\sqrt{\log n})$, $\Theta(\log n)$, or bounded?

Solving it means: a fitted law with stated validity range, an ablation showing the exponent is a property of the model–verifier pair rather than of the reward being optimized, and a compute-optimal allocation rule.

## 2. Formal Setting

Let $p_\theta$ be the model, $p^\star$ the target data distribution, and $\Phi_T: \mathbb{R}^d \to \mathbb{R}^d$ the deterministic (ODE) or stochastic (SDE) sampler at $T$ steps.

**Compute.** Measured in NFE: $N = n \cdot T \cdot c$, where $n$ is candidates evaluated, $T$ steps, and $c = 2$ under classifier-free guidance (two network calls per step). Verifier cost $\kappa$ per candidate adds $n\kappa$; report it separately, since $\kappa / (Tc) \approx 10^{-2}$ for a CLIP-scale verifier but $\approx 1$ for a VLM judge.

**Search.** An algorithm $\mathcal{A}_n$ selects $\hat{z} = \arg\max_{i \le n} r(\Phi_T(z_i))$ (best-of-$n$), or refines seeds locally (zero-order search over a ball around the incumbent), or resamples partial trajectories (particle filtering / Feynman–Kac steering). Induced output law $q_n$.

**Verifier vs. quality.** $r: \mathcal{X} \to \mathbb{R}$ is the *selection* signal. $Q: \mathcal{X} \to \mathbb{R}$ is the *held-out* quality. The measured object must be
$$\Delta(n) \;=\; \mathbb{E}_{x \sim q_n}[Q(x)] - \mathbb{E}_{x \sim p_\theta}[Q(x)],$$
with $Q \perp r$ in training data and architecture. A "scaling law" is a fit $\Delta(n) \approx a (\log n)^{\beta}$ or $a n^{\alpha}$ with reported $R^2$ over $n$ spanning $\ge 2$ decades.

**Distributional cost.** Selection is not free: for best-of-$n$ with continuous $r$,
$$\mathrm{KL}(q_n \Vert p_\theta) \;=\; \log n - \frac{n-1}{n},$$
so at $n=16$ the output law sits $1.83$ nats from the model. FID must be reported alongside precision/recall, since search trades recall for precision and FID conflates the two.

**Assumptions, and which fail.**
1. *Candidates are i.i.d.* — holds for random search, violated by zero-order and path search (dependent proposals; the KL identity above no longer applies).
2. *The verifier is an unbiased noisy view of $Q$* — violated: CLIP/aesthetic/HPS scores share encoders and training data with common evaluation metrics, so $\mathrm{Cov}(r, Q)$ contains a shared-artifact term.
3. *Quality is scalar and monotone* — violated: prompt-faithfulness and photorealism trade off, and a single $Q$ hides the trade.
4. *The sampler is a fixed map* — violated under SDE sampling, where $\Phi$ is itself stochastic and "search over noise" and "search over paths" are not separable.

## 3. State of the Art

**Empirical SOTA.** Ma et al., *Inference-Time Scaling for Diffusion Models beyond Scaling Denoising Steps* (2025, arXiv:2501.09732), is the reference point: it frames test-time compute as search over noise with a verifier, on SiT-XL (ImageNet 256) and FLUX.1-dev (DrawBench, T2I-CompBench). Three search algorithms — random search, zero-order search, search over paths — and multiple verifiers (aesthetic score, CLIPScore, ImageReward, an ensemble). *Established*: quality keeps improving well past the point where extra denoising steps do nothing, across several orders of magnitude of search budget. *Claimed but unablated*: that this constitutes a scaling **law**; no exponent is fit with a validity range, and the reported curves largely use metrics correlated with the selection verifier. The paper itself names "verifier hacking" and shows verifier–metric mismatch, which is the honest form of the caveat.

**Sequential-Monte-Carlo line.** Wu et al., *Practical and Asymptotically Exact Conditional Sampling in Diffusion Models* (NeurIPS 2023, arXiv:2306.17775) gives twisted SMC with asymptotic exactness in particle count $P$ — a *theory* result about the target, not a rate in compute. Singhal et al., *A General Framework for Inference-time Scaling and Steering of Diffusion Models* (2025), unifies these as Feynman–Kac steering; reported as benchmark numbers on text-to-image reward metrics, without held-out-verifier ablation.

**Theory SOTA.** Beirami et al., *Theoretical guarantees on the best-of-$n$ alignment policy* (2024, arXiv:2401.01879) shows the $\log n - (n-1)/n$ KL expression is exact for continuous rewards and that the common $\log n$ formula is an upper bound. This is distribution-level, not a quality-vs-compute rate. No diffusion-specific bound on $\Delta(n)$ exists.

**Adjacent, transferable.** Snell et al. (arXiv:2408.03314) established for LLMs that compute-optimal test-time allocation is question-difficulty-dependent, and that the best strategy shifts with budget. The analogous statement for diffusion — that optimal $(n, T)$ split depends on prompt difficulty — is untested.

## 4. What Is Known

- **Denoising steps saturate.** EDM (Karras et al., NeurIPS 2022): CIFAR-10 unconditional FID $1.97$ at 35 NFE, with no gain from more; ImageNet-64 FID $1.36$ at 511 NFE. Adding steps past this is measurably worthless.
- **Search does not saturate at the same point.** Ma et al. (2025) report continued improvement over roughly two orders of magnitude of additional search NFE on SiT-XL/2 (baseline FID $\approx 2.1$ with CFG on ImageNet 256) and on FLUX.1-dev at 12B parameters.
- **Selection cost is exact.** $\mathrm{KL}(q_n \Vert p_\theta) = \log n - (n-1)/n$: $0.31$ nats at $n=2$, $1.83$ at $n=16$, $4.91$ at $n=1024$.
- **Reward over-optimization is real in diffusion.** DDPO (Black et al., ICLR 2024) and DRaFT (Clark et al., ICLR 2024) both show reward-model gains that decouple from human-judged quality; Diffusion-DPO (Wallace et al., CVPR 2024) reports the same failure mode when optimizing PickScore-style rewards hard.
- **Verifier ensembling helps but does not remove the bias.** Ma et al. use a verifier ensemble specifically because single verifiers are gamed; no independent reproduction of the ensemble result exists.

## 5. What Is Not Known

- **Empirically open.** The core one: $\Delta(n)$ measured on a verifier disjoint from the selection verifier, over $n \in [1, 10^3]$, at a fixed model, with human or held-out-model quality. Runnable today for under 1,000 H100-hours (§8). Nobody has published it with a fitted exponent and confidence interval.
- **Empirically open.** The compute-optimal $(n, T)$ frontier. Is $T=250, n=64$ better or worse than $T=25, n=640$ at equal NFE? Cheap steps plus wide search is the untested hypothesis.
- **Theoretically open.** No bound of the form $\Delta(n) \le f(n; \text{model tail}, \text{verifier noise})$. Extreme-value theory predicts $\Theta(\sqrt{\log n})$ under Gaussian-tailed quality and $\Theta(n^{1/\gamma})$ under Pareto tails — which regime real diffusion models occupy is unproven.
- **Methodologically blocked.** "Quality" itself. FID falls when precision rises and recall falls; a search curve that lowers FID may be shrinking the support. There is no accepted single scalar that is both search-invariant and sensitive to the gains search actually delivers.

## 6. Why It Is Hard

**The measurement is confounded by construction.** The search maximizes $r$; the reported number is almost always $r$ or a metric sharing $r$'s encoder. Under a simple noise model the *shape* of the curve is identical whether the verifier is perfect or half noise — only the amplitude changes (§10). So the fitted exponent is **non-identifiable** from the search curve alone: you cannot tell "the model has rich untapped tails" from "the verifier is well-calibrated," and both produce the same $\sqrt{\log n}$.

Second obstruction: **cost asymmetry against the honest control.** Establishing a law needs $\ge 2$ decades of $n$ *and* enough samples per point for a stable distributional metric (FID wants ~50k). The product is what deters the experiment, not the physics.

Third: **held-out ground truth is scarce.** Human preference at $10^4$ pairwise comparisons per budget point is the only uncontaminated $Q$, and it costs more than the GPUs.

## 7. Current Research (as of 2026)

- **Search-as-inference for diffusion** — NYU (Xie group) and Google DeepMind, following arXiv:2501.09732; extensions to video and to verifier ensembles *(frontier — verify)*.
- **Feynman–Kac / SMC steering** — NYU (Ranganath, Ren) and Columbia; the reweighting view subsumes best-of-$n$ and gives a principled particle-count knob.
- **Test-time alignment without fine-tuning** — KAIST and others on SMC-based reward alignment claiming reduced over-optimization relative to reward fine-tuning *(frontier — verify the over-optimization claim; it is a benchmark number, not an ablation)*.
- **Compute-optimal allocation transfer from LLMs** — the Snell et al. difficulty-conditional allocation applied to prompt difficulty in T2I; discussed, not published at scale *(frontier — verify)*.

## 8. Concrete Next Experiment

**Scale.** SiT-XL/2 or DiT-XL/2, ImageNet 256×256, class-conditional, CFG on. $T = 250$, so 500 forward passes per candidate at $118.6$ GFLOPs each $\approx 5.9 \times 10^{13}$ FLOPs per candidate. Sweep $n \in \{1,2,4,\dots,1024\}$ (11 points), 10,000 prompts/classes per point. Largest arm: $10^4 \times 1024 \times 5.9\times10^{13} = 6.0\times10^{20}$ FLOPs $\approx 420$ H100-hours at 40% MFU; geometric sum over all arms $\approx 850$ H100-hours per condition.

**Arms.**
1. *Treatment:* select with verifier $r$ = ImageReward.
2. *Control arm A (the one that decides it):* identical samples, scored post hoc with a **disjoint** verifier $Q$ — a model with no shared encoder, no shared training corpus with $r$ (e.g. a from-scratch-trained ImageNet classifier-confidence + a 2k-pair human preference subsample at $n \in \{1, 32, 1024\}$).
3. *Control arm B (compute-matched, no search):* $n=1$ with $T$ scaled up so NFE matches each treatment point. Establishes the saturation floor.
4. *Control arm C (random selection):* select uniformly among the $n$ candidates. Isolates the "more samples, luckier draw" artifact from real verifier signal.

**The deciding number.** Fit $\Delta_Q(n) = a(\log_2 n)^\beta$ on arm 2 and report $\hat{\beta}$ with a bootstrap 95% CI, plus the ratio
$$\rho_{\text{eff}} \;=\; \Delta_Q(1024) \,/\, \Delta_r(1024).$$
If $\rho_{\text{eff}} \ge 0.7$ and the CI on $\hat\beta$ excludes 0, generative search over noise is a real scaling axis. If $\rho_{\text{eff}} < 0.3$ — i.e. held-out gain is under a third of verifier-measured gain — the reported curves are mostly verifier hacking and the "law" is a law about the verifier. Report FID *and* precision/recall at every point; a precision rise with recall fall of matching magnitude means the gain is support-shrinkage, not quality.

## 9. Key References

- **[SOTA]** Nanye Ma, Shangyuan Tong, Haolin Jia, Hexiang Hu, Yu-Chuan Su, Mingda Zhang, Xuan Yang, Yandong Li, Tommi Jaakkola, Xuhui Jia, Saining Xie. *Inference-Time Scaling for Diffusion Models beyond Scaling Denoising Steps.* 2025. — arXiv:2501.09732
- **[Foundational]** Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole. *Score-Based Generative Modeling through Stochastic Differential Equations.* ICLR 2021. — arXiv:2011.13456
- **[Foundational]** Jonathan Ho, Ajay Jain, Pieter Abbeel. *Denoising Diffusion Probabilistic Models.* NeurIPS 2020. — arXiv:2006.11239
- **[Foundational]** Tero Karras, Miika Aittala, Timo Aila, Samuli Laine. *Elucidating the Design Space of Diffusion-Based Generative Models.* NeurIPS 2022. — arXiv:2206.00364
- **[SOTA]** Luhuan Wu, Brian L. Trippe, Christian A. Naesseth, David M. Blei, John P. Cunningham. *Practical and Asymptotically Exact Conditional Sampling in Diffusion Models.* NeurIPS 2023. — arXiv:2306.17775
- **[Theory]** Ahmad Beirami, Alekh Agarwal, Jonathan Berant, Alexander D'Amour, Jacob Eisenstein, Chirag Nagpal, Ananda Theertha Suresh. *Theoretical guarantees on the best-of-n alignment policy.* 2024. — arXiv:2401.01879
- **[SOTA]** Charlie Snell, Jaehoon Lee, Kelvin Xu, Aviral Kumar. *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters.* 2024. — arXiv:2408.03314
- **[SOTA]** Raghav Singhal, Zachary Horvitz, Ryan Teehan, Mengye Ren, Zhou Yu, Kathleen McKeown, Rajesh Ranganath. *A General Framework for Inference-time Scaling and Steering of Diffusion Models.* 2025.
- **[Related]** Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, Sergey Levine. *Training Diffusion Models with Reinforcement Learning.* ICLR 2024. — arXiv:2305.13301
- **[Related]** Kevin Clark, Paul Vicol, Kevin Swersky, David J. Fleet. *Directly Fine-Tuning Diffusion Models on Differentiable Rewards.* ICLR 2024. — arXiv:2309.17400
- **[Related]** William Peebles, Saining Xie. *Scalable Diffusion Models with Transformers.* ICCV 2023. — arXiv:2212.09748

## 10. Worked Example

Take true quality $Q \sim \mathcal{N}(0,1)$ over model samples, and a verifier $r = Q + \varepsilon$, $\varepsilon \sim \mathcal{N}(0, \sigma^2)$, independent. Best-of-$n$ selects $\arg\max r$. The correlation is $\rho = 1/\sqrt{1+\sigma^2}$, and

$$\mathbb{E}[Q \mid \text{selected}] = \rho \cdot \mathbb{E}[\max_n \mathcal{N}(0,1)] \approx \rho\sqrt{2\log n}.$$

At $n = 16$, $\mathbb{E}[\max_{16}] = 1.766$.

| verifier noise $\sigma$ | $\rho$ | true gain $\Delta_Q(16)$ | measured gain $\Delta_r(16)$ | $\rho_{\text{eff}}$ |
|---|---|---|---|---|
| $0$ (perfect) | $1.00$ | $1.77$ | $1.77$ | $1.00$ |
| $1$ | $0.71$ | $1.25$ | $2.50$ | $0.50$ |
| $2$ | $0.45$ | $0.79$ | $3.95$ | $0.20$ |

Two things are visible. First, at $\sigma = 2$ the paper-style curve (column 4) climbs *faster* than the perfect-verifier curve while delivering less than half the real gain — a worse verifier produces a more impressive-looking scaling plot. Second, all three rows have the identical functional form $a\sqrt{\log n}$; only $a$ differs. So fitting $\beta$ on the selection verifier gives $\beta = 0.5$ regardless of whether the verifier is informative. **The exponent carries no information about whether search is working.** Only $\rho_{\text{eff}}$ does, and computing it requires the disjoint $Q$ of §8 arm 2.

Scale it: at $n = 1024$, $\mathbb{E}[\max] = 3.24$, so the $\sigma=1$ true gain is $2.29$ against a measured $4.58$ — the gap in absolute terms widens with budget even as the ratio stays fixed at $0.5$. Meanwhile the distributional cost is $\mathrm{KL} = 4.91$ nats. A practitioner reading only the verifier curve concludes that $64\times$ more compute bought $4.58$ units of quality; the honest answer is $2.29$ units and a distribution nearly $5$ nats from the model it was trained to sample.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*