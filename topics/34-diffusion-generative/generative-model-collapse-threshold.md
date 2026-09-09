---
id: 34-diffusion-generative/generative-model-collapse-threshold
title: "Model Collapse Threshold Under Synthetic Data Recycling"
topic: 34-diffusion-generative
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Model Collapse Threshold Under Synthetic Data Recycling

> **Topic:** Diffusion & Generative Modeling · **ID:** `34-diffusion-generative/generative-model-collapse-threshold` · **Status:** empirically-open

## 1. Problem Statement

Generative models are increasingly trained on corpora that contain the output of earlier generative models. **Does there exist a critical synthetic fraction $\lambda^\*$ below which repeated recycling is harmless, and above which the model family degrades without bound?**

Three variants, of different difficulty:

- **Measurement variant.** Given a training corpus, estimate what fraction of it is model-generated, and estimate the distributional damage attributable to that fraction. Currently the weakest link: synthetic-text detection is unreliable at scale, and the damage estimator (FID, test NLL) is confounded.
- **Method variant.** Given a fixed real-data budget and an unbounded synthetic budget, choose a mixing/curation/verification policy that keeps a quality functional non-increasing across generations. Partially solved: *accumulate rather than replace* works.
- **Theory variant.** Prove existence, or non-existence, of a sharp phase transition in $\lambda$ for a realistic estimator class (deep diffusion models, autoregressive transformers) with finite sample size $n$ per generation. Open.

"Solving it" means: a predicate on $(\lambda, n, \text{model class}, \text{curation policy})$ that decides whether $\lim_{t\to\infty}$ of the chosen divergence is finite, plus an experiment that confirms the predicate's location to within a factor of two.

## 2. Formal Setting

Let $p_0$ be the real data distribution over $\mathcal{X}$. A **self-consuming loop** is the iteration

$$\hat p_{t+1} = \mathcal{F}_n\!\left(\lambda\, \mathcal{D}^{\text{real}}_n \;\cup\; (1-\lambda)\, \mathcal{D}^{\text{syn}}_n(\hat p_t)\right),$$

where $\mathcal{F}_n$ is the fit operator (architecture + optimizer + stopping rule) applied to $n$ samples, $\mathcal{D}^{\text{real}}_n \sim p_0$, and $\mathcal{D}^{\text{syn}}_n \sim \hat p_t$. Three regimes in the literature:

- **replace:** generation $t{+}1$ sees only generation-$t$ output ($\lambda = 0$);
- **mix (fixed $\lambda$):** a constant real fraction, corpus size held at $n$;
- **accumulate:** all past generations are retained, so the real fraction decays as $1/(t{+}1)$ but real data is never discarded.

**Quantities, as measured.**

- Divergence $d_t = d(\hat p_t, p_0)$. In images: $\mathrm{FID}$ on $50{,}000$ samples against a fixed reference set; better, $\mathrm{FD}_{\text{DINOv2}}$. In text: held-out negative log-likelihood on a *pre-2020* corpus (to avoid synthetic contamination of the eval set), in nats/token.
- Diversity: recall $\rho_t$ (Kynkäänniemi et al., NeurIPS 2019) or per-class entropy; tail mass $T_t = \Pr_{\hat p_t}[x \in \text{lowest-}1\%\text{-density region of } p_0]$.
- **Threshold.** $\lambda^\* = \inf\{\lambda : \limsup_{t\to\infty} \mathbb{E}[d_t] < \infty\}$. A *sharp* threshold requires $\mathbb{E}[d_t]$ to be discontinuous in $\lambda$ at $\lambda^\*$ in the $n\to\infty$ limit.

**Assumptions, and which fail.**

1. $\mathcal{F}_n$ is the same operator every generation — *violated*: real pipelines change architecture, data mix and filtering each cycle.
2. Synthetic samples are i.i.d. draws from $\hat p_t$ — *violated*: real deployments sample with temperature $<1$, top-$p$, CFG guidance scale $>1$, and best-of-$k$ selection, all of which are density-sharpening or reward-maximizing operators, not sampling.
3. Uncurated recycling — *violated*: web-scale corpora are filtered, and humans discard bad generations. Curation changes the fixed point from $p_0$ to a reward-tilted distribution (Ferbach et al., NeurIPS 2024).
4. $\lambda$ is known — *violated*: nobody can measure the synthetic fraction of Common Crawl.

## 3. State of the Art

**Theory SOTA (established).**
- Bertrand et al., *On the Stability of Iterative Retraining of Generative Models on their Own Data* (ICLR 2024): if the initial model is close enough to $p_0$ **and** the real fraction $\lambda$ exceeds a model-dependent constant, the iteration is locally stable — a contraction argument around $p_0$. This is a *sufficient* condition, not a threshold: the constant is not computable for a real diffusion model.
- Gerstgrasser et al., *Is Model Collapse Inevitable?* (COLM 2024): under **accumulate**, linear-regression test error is bounded by a finite constant independent of $t$, whereas under **replace** it grows linearly in $t$. This is the strongest clean separation in the field.
- Dohmatob, Feng, Kempe et al., *A Tale of Tails: Model Collapse as a Change of Scaling Laws* (ICML 2024) and *Model Collapse Demystified: The Case of Regression* (NeurIPS 2024): collapse appears as a **modified scaling law** with a plateau — added data stops helping beyond a synthetic-data-determined ceiling.
- Seddik et al., *How Bad Is Training on Synthetic Data?* (COLM 2024): bounds the tolerable synthetic fraction as a function of sample count for a Gaussian/mixture setting.

**Claimed but unablated.** *Strong Model Collapse* (Dohmatob et al., 2024/2025) reports that even a $\sim 1\%$ synthetic fraction can flatten the scaling curve. The mechanism is proved in random-projection/regression models; the transformer evidence is small-scale and has not been reproduced independently at $\ge 1$B parameters.

**Empirical SOTA (benchmark numbers only).** Alemohammad et al., *Self-Consuming Generative Models Go MAD* (ICLR 2024) — FID and recall trajectories for StyleGAN2/DDPM on FFHQ across $\sim 5$–$10$ generations. Shumailov et al., *AI Models Collapse When Trained on Recursively Generated Data* (Nature, July 2024) — OPT-125M on wikitext2 and a VAE/GMM demonstration. Both are *existence proofs of degradation under replace*, at scales far below production, and neither locates $\lambda^\*$.

## 4. What Is Known

- **Replace-only recycling degrades, reliably.** Shumailov et al. (Nature 2024): OPT-125M fine-tuned recursively on wikitext2 shows monotone perplexity increase and visible tail loss by generation $\sim 5$–$9$; by generation 9 outputs are largely off-distribution. Scale: 125M params, $\sim10^5$-sample corpora.
- **Diversity dies before fidelity.** In Alemohammad et al. (ICLR 2024), fully synthetic StyleGAN2 loops on FFHQ ($n \approx 70$k, $1024^2$) show recall collapsing within a handful of generations while FID initially moves little — precision can even *improve*. Measured at single-GPU-days scale.
- **A fixed real anchor changes the qualitative outcome.** With a persistent real set in every generation, FID degradation is slowed or arrested rather than divergent (Alemohammad et al. 2024; Bertrand et al. 2024).
- **Accumulate beats replace.** Gerstgrasser et al. (COLM 2024): GPT-2 and Llama-2-scale language modeling plus diffusion and VAE experiments show flat or near-flat error under accumulation, against divergence under replacement. Kazdan et al., *Collapse or Thrive?* (2024/2025), replicate the accumulate-vs-replace split and show the outcome depends on whether real data is *added to* or *swapped out*.
- **Curated loops do not converge to $p_0$.** Ferbach et al. (NeurIPS 2024) prove that self-consuming loops with preference-based curation converge to a reward-maximizing distribution — a *different* fixed point, not a collapsed one.

## 5. What Is Not Known

- **Theoretically open.** Whether $\lambda^\*$ is a genuine phase transition (order parameter discontinuous) or a smooth crossover, for any estimator class richer than linear/Gaussian. All existing thresholds are sufficient conditions with unevaluable constants.
- **Theoretically open.** The interaction of $\lambda$ with $n$: is the relevant control parameter $\lambda$, or the *absolute* real-token count $\lambda n$? The Gaussian analysis in §10 suggests the latter, which would mean no scale-free threshold exists.
- **Empirically open.** Nobody has run a multi-generation self-consuming loop with a $\lambda$-sweep at $\ge 1$B parameters and $\ge 10$ generations with an honest compute-matched control. Cost, not difficulty, is the barrier.
- **Empirically open.** Whether the *sharpening* operators used in practice (CFG guidance $>1$, temperature $<1$, best-of-$k$) accelerate or arrest collapse. These are known to reduce entropy per generation but also to raise per-sample quality; the sign of the net effect over $t$ generations is unmeasured.
- **Methodologically blocked.** The measurement variant. There is no validated estimator of the synthetic fraction of a web corpus, and no divergence estimator with known sensitivity to tail loss at the $10^{-3}$ mass scale where collapse begins.

## 6. Why It Is Hard

The binding obstruction is **an evaluation that does not measure the thing it names**, compounded by **confounded measurement**.

- FID is a Gaussian fit in Inception feature space. It is dominated by bulk-mode agreement and is nearly blind to the loss of $10^{-3}$-mass tails — exactly the first casualty of recycling. FID also has a sample-size bias of order the estimation noise for $N < 50$k (Chong & Forsyth, CVPR 2020), so early-generation drift sits inside the error bar.
- Test NLL on a modern eval corpus is contaminated: post-2023 text is itself partly synthetic, so a collapsing model can score *better* on it.
- **Absent ground truth on $\lambda$.** For any real corpus the independent variable is unobserved, so field observations of "collapse in the wild" cannot be tied to a fraction.
- **Compute cost.** A credible $\lambda$-sweep is $|\Lambda| \times T$ full pretraining runs. At 5 values of $\lambda$ and 10 generations that is 50 runs; at 1B params and 20B tokens each, roughly $10^{22}$–$10^{23}$ FLOPs total. This is why the question stays at 125M–350M.
- **Non-identifiability.** Degradation from recycling is not separable from degradation from repeated fine-tuning, distribution shift in the fit operator, or reduced effective dataset size, unless every arm is compute- and token-matched.

## 7. Current Research (as of 2026)

- **Accumulation and verification.** Stanford (Gerstgrasser, Kazdan, Koyejo and collaborators) on accumulate-vs-replace; Feng, Dohmatob, Kempe et al. on *Beyond Model Collapse: Scaling Up with Synthesized Data Requires Verification* — a verifier/oracle that filters synthetic samples restores scaling.
- **Scaling-law formulations.** NYU/Meta (Dohmatob, Feng, Kempe) treating collapse as a modification of the Chinchilla-style exponent rather than a divergence.
- **Curated loops as implicit preference optimization.** Ferbach, Bertrand, Gidel and collaborators (Mila).
- **Provenance infrastructure.** C2PA content credentials and statistical watermarking as the route around the measurement blockage — the only path to observing $\lambda$ in the wild rather than assuming it. *(frontier — verify: adoption rates and whether watermarks survive the re-encoding typical of web scraping.)*
- **Diffusion-specific loops.** Self-consuming behaviour under classifier-free guidance and under latent-space rather than pixel-space training is being probed but not yet settled. *(frontier — verify.)*

## 8. Concrete Next Experiment

**Question decided:** does a fixed real fraction $\lambda$ admit a threshold, and does it depend on $\lambda$ or on $\lambda n$?

- **Scale.** Latent diffusion, $\sim 400$M params, trained from scratch on ImageNet-1k at $256^2$. $T = 10$ generations. Sweep $\lambda \in \{0, 0.01, 0.05, 0.20, 1.0\}$ at fixed corpus size $n = 1.28$M, plus a second row at $n = 128$k with $\lambda \in \{0.1, 1.0\}$ to separate $\lambda$ from $\lambda n$. 12 arms × 10 generations ≈ 120 training runs at $\sim 200$ A100-hours each ≈ 24k A100-hours.
- **Control arm.** For every $(\lambda, t)$, a **compute- and token-matched real-only** run: same optimizer steps, same number of *unique* real images ($\lambda n$), no synthetic data. This isolates recycling damage from small-dataset damage — the single control most existing papers omit.
- **Deciding number.** Tail recall at the $1\%$ level, $\rho_t^{(1\%)}$: the fraction of held-out real images whose $k$-NN neighbourhood in DINOv2 feature space contains a generated sample, restricted to the $1\%$ lowest-density real images. Report $\Delta_t = \rho_t^{(1\%)} - \rho_t^{(1\%),\text{control}}$ over $t$.
  - If $\Delta_{10} > -0.02$ for $\lambda \ge \lambda_c$ and $\Delta_{10} < -0.20$ for $\lambda < \lambda_c$, with the crossover at the same $\lambda_c$ for both $n$ values → a real threshold in $\lambda$.
  - If the crossover instead lands at the same $\lambda n$ across the two rows → no scale-free threshold; the control parameter is absolute real-sample count, and the "collapse threshold" framing is wrong.
  - If $|\Delta_{10}| < 0.02$ everywhere → degradation is attributable to effective dataset size, not recycling.

Report FID alongside, but do not decide on it: FID's null-arm variance at $N = 50$k is the reason the field cannot currently resolve early generations.

## 9. Key References

- **[Foundational]** Ilia Shumailov, Zakhar Shumaylov, Yiren Zhao, Nicolas Papernot, Ross Anderson, Yarin Gal. *AI models collapse when trained on recursively generated data.* Nature 631, 2024. (earlier as *The Curse of Recursion: Training on Generated Data Makes Models Forget*, arXiv:2305.17493)
- **[Foundational]** Sina Alemohammad, Josue Casco-Rodriguez, Lorenzo Luzi, Ahmed Imtiaz Humayun, Hossein Babaei, Daniel LeJeune, Ali Siahkoohi, Richard G. Baraniuk. *Self-Consuming Generative Models Go MAD.* ICLR 2024. — arXiv:2307.01850
- **[SOTA]** Matthias Gerstgrasser, Rylan Schaeffer, Apratim Dey, Rafael Rafailov, et al. *Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data.* COLM 2024. — arXiv:2404.01413
- **[SOTA]** Elvis Dohmatob, Yunzhen Feng, Pu Yang, Francois Charton, Julia Kempe. *A Tale of Tails: Model Collapse as a Change of Scaling Laws.* ICML 2024. — arXiv:2402.07043
- **[SOTA]** Quentin Bertrand, Avishek Joey Bose, Alexandre Duplessis, Marco Jiralerspong, Gauthier Gidel. *On the Stability of Iterative Retraining of Generative Models on their own Data.* ICLR 2024. — arXiv:2310.00429
- **[Theory]** Elvis Dohmatob, Yunzhen Feng, Julia Kempe. *Model Collapse Demystified: The Case of Regression.* NeurIPS 2024. — arXiv:2402.07712
- **[Theory]** Mohamed El Amine Seddik, Suei-Wen Chen, Soufiane Hayou, Pierre Youssef, Merouane Debbah. *How Bad is Training on Synthetic Data? A Statistical Analysis of Language Model Collapse.* COLM 2024. — arXiv:2404.05090
- **[Theory]** Damien Ferbach, Quentin Bertrand, Avishek Joey Bose, Gauthier Gidel. *Self-Consuming Generative Models with Curated Data Provably Optimize Human Preferences.* NeurIPS 2024. — arXiv:2407.09499
- **[Method]** Yunzhen Feng, Elvis Dohmatob, Pu Yang, Francois Charton, Julia Kempe. *Beyond Model Collapse: Scaling Up with Synthesized Data Requires Verification.* 2024. — arXiv:2406.07515
- **[Measurement]** Min Jin Chong, David Forsyth. *Effectively Unbiased FID and Inception Score and Where to Find Them.* CVPR 2020. — arXiv:1911.07023
- **[Measurement]** Tuomas Kynkäänniemi, Tero Karras, Samuli Laine, Jaakko Lehtinen, Timo Aila. *Improved Precision and Recall Metric for Assessing Generative Models.* NeurIPS 2019. — arXiv:1904.06991

## 10. Worked Example

Take the simplest loop where everything is closed-form: fit a 1-D Gaussian by maximum likelihood to $n$ samples, of which $\lambda n$ are real ($p_0 = \mathcal{N}(0,1)$) and $(1-\lambda)n$ come from $\hat p_t = \mathcal{N}(\mu_t, \sigma_t^2)$.

The MLE variance is biased low by $(n-1)/n$. Writing $a = 1 - 1/n$ and ignoring mean drift:

$$\mathbb{E}[\sigma_{t+1}^2] \approx a\left(\lambda \cdot 1 + (1-\lambda)\,\sigma_t^2\right).$$

For $\lambda = 0$ this is $\sigma_t^2 \to \sigma_0^2 (1-1/n)^t \to 0$: **replace collapses**, with a time constant of $n$ generations. For $\lambda > 0$ there is a stable fixed point:

$$\sigma_\*^2 = \frac{a\lambda}{1 - a(1-\lambda)} = \frac{a\lambda}{1/n + a\lambda}, \qquad 1 - \sigma_\*^2 \approx \frac{1}{\lambda n}.$$

Now put numbers in. With $n = 10^4$ and $\lambda = 0.1$: $\sigma_\*^2 = 0.99900$, a $0.1\%$ variance deficit. With $\lambda = 0.01$: $\sigma_\*^2 = 0.99010$, a $1\%$ deficit. With $\lambda = 0.001$: $\sigma_\*^2 = 0.9091$, a $9\%$ deficit.

**What this makes visible.** There is no $\lambda^\*$ here at all. Every $\lambda > 0$ is stable; the steady-state damage is a smooth $1/(\lambda n)$, and only $\lambda = 0$ diverges. The control parameter is the **absolute real-sample count $\lambda n$**, not the fraction. A "threshold" appears only when you fix a detection floor: the smallest deficit your metric can resolve. If your FID estimator's null-arm standard deviation at $N = 50$k samples is $\pm 0.1$ FID and a $1\%$ second-moment deficit moves FID by less than that, then every $\lambda > 1/(0.01 n)$ is *reported* as "no collapse" — and the measured threshold is a property of the instrument, not of the loop.

That is the obstruction in one line: the field is measuring the noise floor of FID and calling it a phase transition. The experiment in §8 is designed to replace that instrument with a tail statistic whose null distribution is known.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*