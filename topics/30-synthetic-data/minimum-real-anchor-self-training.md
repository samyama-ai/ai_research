---
id: 30-synthetic-data/minimum-real-anchor-self-training
title: "Minimum Real-Data Anchor for Stable Iterative Self-Training"
topic: 30-synthetic-data
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Minimum Real-Data Anchor for Stable Iterative Self-Training

> **Topic:** Synthetic Data · **ID:** `30-synthetic-data/minimum-real-anchor-self-training` · **Status:** open

## 1. Problem Statement

A model is retrained across generations on a mixture of human data and its own outputs. Web-scale corpora already contain model output, so this loop is running whether or not anyone designed it.

**Question.** What is the smallest amount of real data $\lambda$ — as a fraction of each generation's training mix, or as an absolute token count — that keeps the loop stable for $T$ generations, and what does $\lambda^\star$ depend on?

Three variants, routinely conflated:

- **Measurement.** Define "stable". Is it bounded divergence $\sup_t D(p^\star \Vert p_{\theta_t}) < \infty$, non-increasing downstream error, or preserved tail coverage? These orderings differ: a loop can hold cross-entropy flat while losing the rare-event tail.
- **Method.** Given a budget of real tokens, how should they be spent — uniform mixing every round, one fixed anchor set replayed, fresh i.i.d. real draws per round, or verification/curation of synthetic data instead of mixing?
- **Theory.** Prove a threshold: an $\lambda^\star(\text{model class}, n, T)$ below which divergence grows without bound and above which the retraining map has a stable fixed point near $p^\star$.

**Solved** would mean: a predictive formula for $\lambda^\star$, validated at $\geq 1$B parameters over $\geq 10$ generations, that transfers to a held-out model scale without refitting.

## 2. Formal Setting

Real distribution $p^\star$ over sequences. Generation $t$ model $p_{\theta_t}$, learned by an operator $\mathcal{T}$ (fixed architecture, optimizer, token budget $n$) from dataset $\mathcal{D}_t$:

$$\mathcal{D}_t = \underbrace{\{x_i \sim p^\star\}_{i=1}^{\lambda n}}_{\text{anchor}} \;\cup\; \underbrace{\{\tilde x_j \sim p_{\theta_{t-1}}\}_{j=1}^{(1-\lambda) n}}_{\text{synthetic}}, \qquad \theta_t = \mathcal{T}(\mathcal{D}_t).$$

Measured quantities:

- **Anchor fraction** $\lambda \in [0,1]$: real tokens ÷ total tokens in $\mathcal{D}_t$, counted after dedup. **Anchor mass** $m = \lambda n$ in tokens — the two decouple when $n$ grows across generations.
- **Anchor freshness**: *replay* (same $\lambda n$ tokens every round) vs *fresh* ($\lambda n$ new i.i.d. draws). Replay is what practice can afford; every stability proof assumes fresh.
- **Divergence proxy** $L_t = -\frac{1}{|V|}\sum_{x \in V} \log p_{\theta_t}(x)$, cross-entropy in nats/token on a held-out real set $V$ that is frozen at $t=0$ and never enters any $\mathcal{D}_t$. $L_t - H(p^\star)$ estimates $\mathrm{KL}(p^\star \Vert p_{\theta_t})$ up to an unknown constant, so only *differences* across $t$ are interpretable.
- **Tail coverage** $C_t(\alpha)$: fraction of the lowest-$\alpha$-probability quantile of $p^\star$ assigned $\log p_{\theta_t} > \tau$. Collapse hits $C_t$ before it hits $L_t$.
- **Stability predicate**: $\hat s = \mathrm{slope}_t(L_t)$ over generations $t \in [T/2, T]$, in nats/generation. Stable iff the upper confidence bound on $\hat s$ is below a tolerance $\epsilon$.

Assumptions the theory rests on, and their status in practice:

| Assumption | Practice |
|---|---|
| $\mathcal{T}$ returns a (near-)maximum-likelihood fit | **Violated** — one-epoch SGD, early stopping, LR schedule |
| Synthetic data sampled i.i.d. from $p_{\theta_{t-1}}$ | **Violated** — temperature $<1$, top-$p$, rejection sampling, prompt distribution $\neq p^\star$ |
| No filtering between generations | **Violated** — dedup, quality classifiers, RLHF-shaped generators |
| Anchor draws are fresh each round | **Violated** — the same human corpus is replayed |
| Model class contains $p^\star$ (well-specified) | **Violated** at every practical scale |

## 3. State of the Art

**Theory (established).** Bertrand et al. (ICLR 2024) prove that iterative retraining on a mix is locally stable if the initial model is close enough to $p^\star$ *and* the real-data proportion exceeds a threshold set by the Lipschitz constant of the retraining map; below it, the fixed point loses stability. This is the cleanest existing statement of the problem, but the threshold is an abstract constant, not a computable number for a transformer. Gerstgrasser et al. (COLM 2024) prove that *accumulating* data — appending synthetic to all prior data rather than replacing it — bounds the linear-regression test error by a finite constant ($\propto \pi^2/6$ times the noise term) instead of letting it grow linearly in generations. Seddik et al. (COLM 2024) give a finite-sample bound on how many synthetic samples a fixed real corpus can support before collapse. Dohmatob et al. (ICML 2024) show synthetic data changes the *exponent* of the scaling law, producing a finite-loss plateau: more compute stops buying accuracy.

**Empirical (established).** Shumailov et al. (Nature 2024) show monotone degradation over ~9 generations of OPT-125M on wikitext2 under full replacement, with tail events disappearing first. Alemohammad et al. (ICLR 2024) show, for StyleGAN2/diffusion on FFHQ, that loops with a *fixed* real set still go "MAD" — precision and recall collapse — while loops with *fresh* real data each generation do not. Kazdan et al. (2024/2025) systematically compare replace / accumulate / accumulate-subsample.

**Claimed but unablated.** That "a few percent real data suffices" — the figure circulates from single-scale, single-task runs and has no cross-scale ablation. That contemporary synthetic pipelines (phi-style textbooks, Cosmopedia, WRAP-style rephrasing) are safe because filtering breaks the loop: the training-set quality numbers are benchmark scores at one generation, not multi-generation stability measurements. Shumailov's own "preserve 10% of original data" variant slows but does not eliminate degradation — one model, one dataset, benchmark number only.

## 4. What Is Known

- **Full replacement collapses.** OPT-125M / wikitext2, 9 generations: perplexity on real data rises monotonically and output diversity falls sharply (Nature 2024, 125M scale).
- **Accumulation does not collapse** in linear regression and in transformer language models up to ~125M–1B params over 5–10 generations; test error saturates rather than diverging (COLM 2024).
- **Fresh real data > fixed real data.** FFHQ StyleGAN2 loops: fixed anchor → FID rises across generations; fresh anchor per generation → FID stable (ICLR 2024, image scale).
- **Synthetic data bends scaling laws.** Under a synthetic mixture the loss-vs-compute curve plateaus at a nonzero floor; a small synthetic fraction is enough to induce it in the "strong collapse" analysis of Dohmatob et al. (2024), in linear/random-feature models and small transformers.
- **Tails go first.** Rare-token and tail-quantile coverage degrades several generations before aggregate perplexity moves (multiple studies, 125M–1.4B scale).

No published work reports a *measured* $\lambda^\star$ with an error bar, at more than one model scale, for a fixed protocol.

## 5. What Is Not Known

- **Theoretically open.** Whether a threshold $\lambda^\star > 0$ exists for over-parameterized, misspecified models under replay (non-fresh) anchors. Bertrand's result is local and assumes a near-MLE operator; nothing rules out $\lambda^\star \to 0$ or $\lambda^\star \to 1$ as parameter count grows. No proof either way.
- **Theoretically open.** Whether $\lambda^\star$ scales with parameters $N$, tokens $n$, or generations $T$ — and whether the right invariant is the fraction $\lambda$ or the absolute anchor mass $m$.
- **Empirically open.** The multi-scale, multi-generation sweep is runnable today at 160M–1.4B for a few thousand GPU-hours. Nobody has published it with a shared protocol, so the "$\approx$ a few percent" folklore is untested.
- **Methodologically blocked.** "Stability" has no agreed operational definition. Cross-entropy on frozen held-out real data, tail coverage, and downstream benchmark accuracy give different orderings of the same runs; a loop can improve on MMLU while losing $\log p$ on rare text. Until the predicate is fixed, $\lambda^\star$ is not a well-posed number.

## 6. Why It Is Hard

**Confounded measurement, compounded by cost.** Each generation is a full training run, so measuring one $\lambda$ over $T=10$ generations costs $10\times$ a pretraining run; a sweep over 6 values of $\lambda$ and 3 model scales is 180 runs. That cost forces short loops and small models — exactly the regime where the effect is weakest and the extrapolation to frontier scale is unjustified.

**Non-identifiability of the cause.** When $L_t$ rises, the increase mixes at least four sources: (a) statistical error from finite resampling, (b) approximation error from misspecification, (c) sampling-policy distortion (temperature, top-$p$) which is not $p_{\theta_{t-1}}$ at all, and (d) optimizer path dependence from warm-starting. Existing experiments do not separate them, so a measured $\lambda^\star$ is a property of the pipeline, not of the loop.

**The evaluation does not measure what it names.** Benchmark accuracy is the reported outcome, but collapse is a tail phenomenon. A pipeline can score flat on MMLU while $C_t(0.01)$ halves.

## 7. Current Research (as of 2026)

- **Curation and verification as the stabilizer.** Ferbach et al. (NeurIPS 2024) prove that curated self-consuming loops optimize the curator's preference — meaning the loop converges, but to the *verifier's* distribution, not $p^\star$. Feng et al. (2024) argue verification, not real-data mass, is the binding constraint. This reframes $\lambda^\star$ as a trade against verifier accuracy. *(frontier — verify: whether verifier bias substitutes for or merely disguises collapse.)*
- **Contamination measurement at web scale** — estimating the synthetic fraction already present in Common Crawl snapshots, which sets an involuntary lower bound on $1-\lambda$. Estimates vary widely by method; treat any single number as unreliable. *(frontier — verify)*
- **Self-correcting loops** (Gillman et al., ICML 2024) — projecting synthetic samples onto an expert-corrected manifold each generation.
- **Groups.** Kempe/Dohmatob (NYU/Meta) on scaling-law collapse; Gidel/Bertrand/Ferbach (Mila) on iterative-retraining stability; Koyejo/Schaeffer/Kazdan (Stanford) on accumulate-vs-replace; Shumailov/Anderson (Oxford/Cambridge) on the collapse phenomenon.

## 8. Concrete Next Experiment

**Scale.** Pythia-style decoder-only models at 160M and 410M params. Per generation: 4B tokens, one epoch, identical hyperparameters, cold restart from init (no warm start — removes confound (d)). $T = 10$ generations. Anchor from C4; held-out real validation set $V$ frozen at $t=0$, 50M tokens, excluded from all $\mathcal{D}_t$.

**Arms.** $\lambda \in \{0, 0.01, 0.03, 0.10, 0.30\}$, each in two conditions: *replay* (same anchor tokens every generation) and *fresh* (new C4 shard each generation). Synthetic data sampled at temperature 1.0, unfiltered, from prompts drawn from $V$-disjoint real prefixes.

**Control arm.** $\lambda = 1.0$ (all-real, fresh 4B tokens per generation, same 10 restarts). This is the null: it isolates run-to-run variance in $\hat s$ from loop-induced drift. Report all effects as differences against it.

**Deciding number.** $\hat s_\lambda = \mathrm{slope}$ of $L_t$ over $t \in \{5,\dots,10\}$, nats/generation, with a 95% CI from 3 seeds. Define $\lambda^\star$ as the smallest $\lambda$ with $\hat s_\lambda - \hat s_{1.0} < 0.002$ nats/gen at the CI upper bound. **The single decisive comparison:** $\lambda^\star(160\text{M})$ vs $\lambda^\star(410\text{M})$ under replay. If they agree within one grid step, the fraction is the invariant and the folklore constant is defensible; if $\lambda^\star$ rises with $N$, every frontier-scale extrapolation from small-model collapse studies is invalid.

**Cost.** $6\lambda \times 2$ conditions $\times 10$ generations $\times 3$ seeds at 410M, $\approx 6ND = 1.0\times10^{19}$ FLOPs per run-generation, $\approx 3.6\times10^{21}$ FLOPs total — order $10^3$ A100-hours. Log $C_t(0.01)$ alongside $L_t$ every generation; report both, since the sections above predict they disagree.

## 9. Key References

- **[Foundational]** Ilia Shumailov, Zakhar Shumaylov, Yiren Zhao, Nicolas Papernot, Ross Anderson, Yarin Gal. *AI models collapse when trained on recursively generated data.* Nature 631, 2024. (Earlier version: *The Curse of Recursion: Training on Generated Data Makes Models Forget*, arXiv:2305.17493)
- **[Foundational/Theory]** Quentin Bertrand, Avishek Joey Bose, Alexandre Duplessis, Marco Jiralerspong, Gauthier Gidel. *On the Stability of Iterative Retraining of Generative Models on their own Data.* ICLR 2024. — arXiv:2310.00429
- **[SOTA]** Matthias Gerstgrasser, Rylan Schaeffer, Apratim Dey, Rafael Rafailov, et al. *Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data.* COLM 2024. — arXiv:2404.01413
- **[SOTA]** Elvis Dohmatob, Yunzhen Feng, Pu Yang, Francois Charton, Julia Kempe. *A Tale of Tails: Model Collapse as a Change of Scaling Laws.* ICML 2024. — arXiv:2402.07043
- **[SOTA]** Mohamed El Amine Seddik, Suei-Wen Chen, Soufiane Hayou, Pierre Youssef, Merouane Debbah. *How Bad is Training on Synthetic Data? A Statistical Analysis of Language Model Collapse.* COLM 2024. — arXiv:2404.05090
- **[SOTA]** Sina Alemohammad, Josue Casco-Rodriguez, Lorenzo Luzi, Ahmed Imtiaz Humayun, Hossein Babaei, Daniel LeJeune, Ali Siahkoohi, Richard G. Baraniuk. *Self-Consuming Generative Models Go MAD.* ICLR 2024. — arXiv:2307.01850
- **[SOTA]** Damien Ferbach, Quentin Bertrand, Avishek Joey Bose, Gauthier Gidel. *Self-Consuming Generative Models with Curated Data Provably Optimize Human Preferences.* NeurIPS 2024.
- **[SOTA]** Joshua Kazdan, Rylan Schaeffer, Apratim Dey, Matthias Gerstgrasser, et al. *Collapse or Thrive? Perils and Promises of Synthetic Data in a Self-Generating World.* arXiv:2410.16713, 2024.
- **[Related]** Nate Gillman, Michael Freeman, Daksh Aggarwal, Chia-Hong Hsu, et al. *Self-Correcting Self-Consuming Loops for Generative Model Training.* ICML 2024.
- **[Related]** Elvis Dohmatob, Yunzhen Feng, Julia Kempe. *Strong Model Collapse.* arXiv:2410.04840, 2024.

## 10. Worked Example

Take the well-specified Gaussian mean case, the only setting where $\lambda^\star$ can be computed by hand. $p^\star = \mathcal{N}(\mu, \sigma^2)$; each generation fits $\hat\mu_t$ by the sample mean of $n$ points, $\lambda n$ real and $(1-\lambda)n$ drawn from $\mathcal{N}(\hat\mu_{t-1}, \sigma^2)$. Then

$$\hat\mu_t = \lambda \bar{x}^{\text{real}}_t + (1-\lambda)\hat\mu_{t-1} + \xi_t, \qquad \mathrm{Var}(\xi_t) \approx \frac{(1-\lambda)\sigma^2}{n}.$$

With **fresh** real draws each round, $\hat\mu_t$ is an AR(1) process contracting toward $\mu$ with factor $a = 1-\lambda$, and

$$\lim_{t\to\infty}\mathrm{Var}(\hat\mu_t) = \frac{\sigma^2/n}{1-(1-\lambda)^2} \approx \frac{\sigma^2}{2\lambda n}.$$

Finite for any $\lambda > 0$. Put in numbers: $n = 10^4$, $\sigma = 1$, $\lambda = 0.03$ gives a stationary standard error of $\sqrt{1/(2\cdot0.03\cdot10^4)} = 0.041$, versus $0.010$ for all-real. Cost of a 3% anchor: $4\times$ the error, not divergence. This is the calculation behind "a few percent suffices."

Now switch to **replay** — the same $\lambda n = 300$ real points every round, which is what any real pipeline does. The contraction now pulls toward $\bar{x}^{\text{real}}$, not $\mu$, and the fixed point carries a permanent bias of variance $\sigma^2/(\lambda n) = 1/300$, s.e. $0.058$, that no number of generations removes. Worse, the injected noise $\xi_t$ is no longer averaged away against fresh signal: the process is a random walk with a weak restoring force whose *variance* is bounded but whose realized draw is locked in at generation 1.

The obstruction is visible here. The two protocols differ by a factor of $\approx 1.4$ in stationary error at $\lambda = 0.03$ — well inside the seed-to-seed noise of any 10-generation transformer experiment at 410M. So the quantity the theory says matters most (anchor freshness) is, at achievable experiment scale, smaller than the measurement noise. And this is the *easy* case: well-specified, one parameter, exact MLE. In the transformer case, misspecification adds a bias term that is not identified from $L_t$ alone, which is why Section 8 insists on logging tail coverage $C_t$ next to cross-entropy rather than trusting a single scalar.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*