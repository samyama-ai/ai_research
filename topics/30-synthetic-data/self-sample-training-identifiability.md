---
id: 30-synthetic-data/self-sample-training-identifiability
title: "Identifiability of Generative Models Trained on Their Own Samples"
topic: 30-synthetic-data
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Identifiability of Generative Models Trained on Their Own Samples

> **Topic:** Synthetic Data · **ID:** `30-synthetic-data/self-sample-training-identifiability` · **Status:** open

## 1. Problem Statement

A model is trained on a corpus that contains samples from an earlier generative model — possibly an earlier version of itself. **Input:** the resulting model $\hat p_T$, available either as weights plus a sampler, or as a black box emitting samples. **Output:** the *training provenance* — the number of self-training generations $T$, the per-generation synthetic fraction $\alpha$, and the identity of the ancestor generators.

Three variants that are routinely conflated:

- **Theory variant.** Is the map $\Phi: (p_0, \alpha, T, \text{ancestry}) \mapsto \hat p_T$ injective, up to a stated equivalence class? If two distinct histories induce the same limiting distribution, no estimator can separate them regardless of sample size. This is a non-identifiability question, not a statistical-efficiency one.
- **Measurement variant.** Given $m$ samples from $\hat p_T$ and no access to $p_0$, estimate $\alpha$ with a confidence interval. Requires that the estimand be defined without ground-truth provenance labels, which is exactly what a scraped web corpus lacks.
- **Method variant.** Build an estimator $\hat\alpha(\cdot)$ that beats the trivial baseline (predict the population base rate) on held-out model families it never saw during calibration.

**Solved** means: a theorem giving necessary and sufficient conditions on the model class $\mathcal{P}$ and the fitting map under which $(\alpha, T)$ is identifiable from $\hat p_T$, plus an estimator meeting the stated rate on a benchmark with audited provenance labels.

## 2. Formal Setting

Let $p_0$ be the real-data distribution on $\mathcal{X}$. Let $\mathcal{P}_\theta$ be a parametric family and $\mathcal{A}: \mathcal{X}^n \to \Theta$ the fitting algorithm (SGD to a fixed budget, not the exact MLE — this matters).

Generation $t$ trains on $n$ draws from the mixture
$$q_t = (1-\alpha)\, p_0 + \alpha\, \hat p_{t-1}, \qquad \hat p_t = p_{\mathcal{A}(X_{1:n} \sim q_t)}.$$

**Quantities, as measured:**

- $\alpha \in [0,1]$ — synthetic fraction, measured as the token-weighted share of training documents whose provenance audit marks them machine-generated. In practice audits are missing; $\alpha$ is then *defined* only through a detector, which makes it detector-relative.
- $T$ — generation depth. Observable only as a model-card claim; not measurable from the corpus.
- Tail mass $M_\tau(p) = \Pr_{x\sim p}[p(x) < \tau]$, estimated by importance-weighted sampling with a reference scorer. Tail loss is the most-cited collapse signature (Shumailov et al. 2024).
- Entropy rate $H(\hat p_T)$, estimated as mean negative log-likelihood on a held-out *real* corpus — this is a cross-entropy and conflates entropy loss with distribution shift.
- Sample diversity: Vendi score, or Precision/Recall (Kynkäänniemi et al., NeurIPS 2019) on a fixed embedder. Both are embedder-relative.

**Identifiability predicate.** $(\alpha, T)$ is identifiable at $p_0$ iff
$$\Phi(p_0,\alpha,T) = \Phi(p_0,\alpha',T') \implies (\alpha,T)=(\alpha',T').$$

**Assumptions, with violation status:**

1. $\hat p_t$ depends on the past only through $\hat p_{t-1}$ (Markov). **Violated:** real pipelines accumulate rather than replace data (Gerstgrasser et al., COLM 2024).
2. No curation between generations. **Violated:** every deployed pipeline filters, rejects, and RLHF-selects; curation reverses the sign of the drift (Ferbach et al., NeurIPS 2024).
3. Fixed $n$, fixed architecture across $t$. **Violated:** frontier retrains change all three.
4. $p_0$ is stationary. **Violated:** the web itself now contains generation-$t$ text, so $p_0$ is a function of $T$.

## 3. State of the Art

**Theory SOTA (established).** Bertrand et al. (ICLR 2024) prove that iterative retraining on a mixture is *locally stable* around $p_0$ provided the initial model is close enough and $\alpha$ is below a threshold set by the fitting map's contraction constant. Dohmatob, Feng & Kempe (NeurIPS 2024) give exact closed-form collapse for ridge/OLS regression: test error acquires an additive term that does not vanish with $n$. Dohmatob et al. (ICML 2024) show synthetic data truncates Pareto tails and thereby *changes the exponent* of the scaling law. Gerstgrasser et al. (COLM 2024) prove that under data *accumulation* the test error is bounded by a finite constant for linear models — collapse is not inevitable.

**Empirical SOTA (established).** Shumailov et al. (Nature 2024) show OPT-125M fine-tuned recursively on wikitext2 degrades over ~9 generations with tail events disappearing first. Alemohammad et al. (ICLR 2024) show StyleGAN-2 and diffusion models drift in FID and lose diversity within ~5 self-consuming loops unless fresh data is injected each loop.

**Claimed but unablated.** That collapse signatures are *diagnostic* of self-training. No paper shows that the measured signature (tail loss, diversity drop) distinguishes self-training from the confounds — aggressive dedup, quality filtering, temperature-sampled decoding, or an RLHF stage — each of which produces the same low-entropy, short-tailed output.

**Benchmark-number-only results.** Machine-text detection accuracy. Sun et al. (2025, *Idiosyncrasies in Large Language Models*) report that a simple classifier on LLM outputs identifies the source model with high accuracy in-distribution. These numbers are held-out-split numbers on curated generations; there is no audited, provenance-labeled web corpus on which any of them has been validated, and none of them estimates $\alpha$.

## 4. What Is Known

- **Variance decay is exact for Gaussians.** Fitting a Gaussian by MLE on $n$ own-samples per round multiplies expected variance by $(n-1)/n$ per generation. Shumailov et al. report the corresponding closed form; collapse to a point is almost sure as $T\to\infty$ at $\alpha=1$.
- **Perplexity degradation at 125M scale.** Shumailov et al. (Nature 2024): OPT-125M on wikitext2, generation 0 perplexity rises monotonically through generation 9, with the largest shift in the upper perplexity percentiles.
- **Accumulation bounds error.** Gerstgrasser et al. (COLM 2024): with accumulated data, linear-regression test error is bounded by $\frac{\pi^2}{6}$ times the single-generation error; validated empirically on GPT-2-scale transformers (up to ~125M) and diffusion models on CelebA.
- **A stability threshold exists.** Bertrand et al. (ICLR 2024): stability holds for $\alpha$ below a model-dependent constant; measured empirically on FFHQ/CIFAR-10 diffusion models.
- **Verification changes the sign.** Feng et al. (2024) show a verifier over synthetic data restores scaling in a transformer arithmetic setting.
- **Deconvolution is logarithmically hard.** Fan (Ann. Statist., 1991): recovering a density through a smooth convolution attains only $(\log n)^{-\beta}$ rates. Inverting a generation step is a deconvolution.

## 5. What Is Not Known

- **Theoretically open.** No injectivity result. Nobody has shown whether distinct $(\alpha,T)$ pairs can produce identical $\hat p_T$ in any nontrivial family — the Gaussian case (§10) suggests they *can*, but there is no theorem stating the equivalence classes, and no characterization of which fitting maps $\mathcal{A}$ break the degeneracy. Also open: whether weight access (not just samples) restores identifiability.
- **Empirically open.** Whether collapse signatures survive at frontier scale. Every published self-consuming loop is at $\le$ 1.5B parameters with $\le$ 10 generations. The experiment at 7B with realistic curation is runnable today and unrun.
- **Methodologically blocked.** $\alpha$ for a web corpus. There is no gold provenance label, so every reported "synthetic fraction of the web" is a detector output, and detectors are known to be unreliable under paraphrase (Sadasivan et al., 2023). The estimand is not defined independently of the estimator.

## 6. Why It Is Hard

**The primary obstruction is a dimension mismatch that makes the parameters non-identifiable.** The observables from black-box access — entropy, tail mass, diversity — are a handful of scalars that all move monotonically in the *same* direction under self-training. The unknowns are $(\alpha, T, n, \text{curation strength}, \text{decoding temperature})$, at least five. Contraction dynamics make matters worse: mixtures with $\alpha<1$ converge to a fixed point, so once converged, $T$ leaves no trace at all, and $\alpha$ appears only through a fixed-point ratio that is also a function of $n$.

**Second: absent ground truth.** No audited corpus with per-document provenance exists at pretraining scale, so no estimator can be calibrated, only benchmarked against its own synthetic construction.

**Third: confounded measurement.** Quality filtering and RLHF both reduce output entropy. An "is this model self-trained?" test that fires on any low-entropy model is not measuring what it names.

## 7. Current Research (as of 2026)

- **Statistical model-collapse theory** — Kempe/Dohmatob (NYU) on scaling-law modification; Seddik et al. (MBZUAI, COLM 2024) on collapse bounds for token distributions.
- **Curated self-consumption** — Ferbach, Bertrand, Gidel (Mila) proving curated loops optimize a reward rather than collapse; the direct antagonist to the naive-collapse literature.
- **Provenance infrastructure** — the Data Provenance Initiative (Longpre et al.) auditing dataset licensing and origin; C2PA-style content credentials for images. Neither yet yields a labeled text corpus.
- **Model fingerprinting from outputs** — Sun, Yin, Xu, Kolter, Liu (2025) on idiosyncratic per-model signatures. *(frontier — verify)* whether these signatures survive one round of retraining, which is the crux for ancestry recovery.
- **Radioactive-data-style tracing** — extending Sablayrolles et al. (ICML 2020) to generative ancestry, i.e. deliberately marking outputs so descendants are detectable. This converts the problem from inference to design and is the most likely near-term "solution".

## 8. Concrete Next Experiment

**Question:** is $(\alpha, T)$ identifiable from black-box samples at all?

**Scale.** Pythia-1.4B, 8 fitting generations, $n = 2 \times 10^9$ tokens per generation, on a grid of $\alpha \in \{0.25, 0.5, 0.75, 1.0\}$ — 32 model-checkpoints, roughly 4,000 A100-hours. Sample $m = 10^6$ generations per checkpoint.

**Control arm.** Two controls, both essential. (a) $\alpha = 0$ retrained for 8 generations on fresh real shards — isolates run-to-run drift. (b) A *confound arm*: a single-generation $\alpha=0$ model trained with a quality filter tuned to match the entropy of the $\alpha=1, T=8$ model. If the estimator cannot separate the confound arm from the collapsed arm, the signature is not diagnostic.

**Deciding number.** Train a classifier on the $10^6$ samples to predict the $(\alpha, T)$ cell, and report **balanced accuracy over the 32 cells against the 1/32 = 3.1% chance baseline, plus its accuracy on the confound arm.** If the pairs $(\alpha=1,T=2)$ and $(\alpha=0.5,T=8)$ are confused above 40% — the degeneracy predicted in §10 — the parameters are empirically non-identifiable from samples, and the field should stop reporting scalar collapse metrics as provenance evidence.

## 9. Key References

- **[Foundational]** Shumailov, Shumaylov, Zhao, Papernot, Anderson, Gal. *AI models collapse when trained on recursively generated data.* Nature 631, 2024. (earlier: *The Curse of Recursion*, arXiv:2305.17493)
- **[Foundational]** Alemohammad, Casco-Rodriguez, Luzi, Humayun, Babaei, LeJeune, Siahkoohi, Baraniuk. *Self-Consuming Generative Models Go MAD.* ICLR 2024. — arXiv:2307.01850
- **[SOTA]** Gerstgrasser, Schaeffer, Dey, Rafailov, et al. *Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data.* COLM 2024. — arXiv:2404.01413
- **[SOTA]** Dohmatob, Feng, Yang, Charton, Kempe. *A Tale of Tails: Model Collapse as a Change of Scaling Laws.* ICML 2024. — arXiv:2402.07043
- **[SOTA]** Bertrand, Bose, Duplessis, Jiralerspong, Gidel. *On the Stability of Iterative Retraining of Generative Models on their own Data.* ICLR 2024. — arXiv:2310.00429
- **[SOTA]** Ferbach, Bertrand, Bose, Gidel. *Self-Consuming Generative Models with Curated Data Provably Optimize Human Preferences.* NeurIPS 2024. — arXiv:2407.09499
- **[Theory]** Fan. *On the Optimal Rates of Convergence for Nonparametric Deconvolution Problems.* Annals of Statistics 19(3), 1991.
- **[Theory]** Hyvärinen, Pajunen. *Nonlinear independent component analysis: Existence and uniqueness results.* Neural Networks 12(3), 1999.
- **[Related]** Khemakhem, Kingma, Monti, Hyvärinen. *Variational Autoencoders and Nonlinear ICA: A Unifying Framework.* AISTATS 2020. — arXiv:1907.04809
- **[Related]** Sablayrolles, Douze, Schmid, Jégou. *Radioactive data: tracing through training.* ICML 2020. — arXiv:2002.00937
- **[Related]** Sadasivan, Kumar, Balasubramanian, Wang, Feizi. *Can AI-Generated Text be Reliably Detected?* 2023. — arXiv:2303.11156
- **[Survey]** Locatello, Bauer, Lucic, Rätsch, Gelly, Schölkopf, Bachem. *Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations.* ICML 2019. — arXiv:1811.12359

## 10. Worked Example

Take the simplest case where everything is computable: $\mathcal{P} = \{\mathcal{N}(0,\sigma^2)\}$, MLE fit, $n = 100$ samples per generation, mixture fraction $\alpha$. Let $u_t = \mathbb{E}[\hat\sigma_t^2]/\sigma_0^2$ and $c = (n-1)/n = 0.99$. Then
$$u_t = c\big[(1-\alpha) + \alpha\, u_{t-1}\big], \qquad u_0 = 1, \qquad u^\star = \frac{c(1-\alpha)}{1-c\alpha}.$$

Two histories:

| history | computation | $u_T$ |
|---|---|---|
| $\alpha = 1.0$, $T = 2$ | $0.99^2$ | **0.9801** |
| $\alpha = 0.5$, $T = 8$ | fixed point $0.495/0.505$, reached to 4 d.p. by $T{=}8$ (rate $c\alpha = 0.495$) | **0.9802** |

Pure self-training for two rounds and half-synthetic training for eight rounds are **indistinguishable to 4 decimal places** in the only observable this family has. The map $\Phi$ is not injective, and no amount of data from $\hat p_T$ fixes it.

Now add sampling noise. The relative standard error of $\hat\sigma^2$ from $m$ samples is $\sqrt{2/m}$. To resolve even a genuinely different pair — say $u = 0.980$ versus $u = 0.990$, a 1% gap — at $2\sigma$ separation needs $\sqrt{2/m} < 0.0025$, i.e. $m > 320{,}000$ samples, *and* exact knowledge of $\sigma_0^2$, which for a web corpus is unknown.

The obstruction is visible in one line: the observable is one number, the unknowns are three ($\alpha$, $T$, $n$), and contraction to $u^\star$ destroys the $T$ coordinate outright. Scaling from Gaussians to transformers adds parameters and adds confounds; it does not add observables.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*