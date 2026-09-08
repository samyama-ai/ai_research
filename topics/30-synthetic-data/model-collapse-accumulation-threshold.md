---
id: 30-synthetic-data/model-collapse-accumulation-threshold
title: "Model Collapse Thresholds Under Data Accumulation"
topic: 30-synthetic-data
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Model Collapse Thresholds Under Data Accumulation

> **Topic:** Synthetic Data · **ID:** `30-synthetic-data/model-collapse-accumulation-threshold` · **Status:** open

## 1. Problem Statement

Successive generations of models are trained on corpora that contain the outputs of earlier models. Two data policies bracket practice:

- **Replace** — generation $t$ trains only on data sampled from model $t-1$. Degradation is unbounded.
- **Accumulate** — generation $t$ trains on the union of all real data and all synthetic data produced so far. Degradation appears bounded.

Real web-scale training sits between them: the real fraction of the corpus shrinks each generation but never hits zero, and compute limits force subsampling of the accumulated pool. The problem is to locate the boundary.

- **Measurement variant.** Define a statistic on the generation-$t$ model that is monotone in collapse, computable at scale, and not confounded by the fact that later models train on more total tokens. Test loss on a fixed held-out *real* corpus is the current default; it is insensitive to tail loss until tails are already gone.
- **Method variant.** Given a fixed compute budget per generation and a real-data injection rate, choose a mixing/curation policy that keeps degradation bounded as $t \to \infty$.
- **Theory variant.** Prove a sharp threshold: a condition on the real-data schedule $\{\alpha_t\}$ separating $\sup_t \mathrm{Err}_t < \infty$ from $\mathrm{Err}_t \to \infty$.

Solving it means: a stated condition on $\{\alpha_t\}$, proved for a nontrivial model class, and empirically confirmed at $\geq 10$ generations of a $\geq 10^9$-parameter language model.

## 2. Formal Setting

Let $p_0$ be the true data distribution over $\mathcal{X}$. Generation $t \in \{1,\dots,T\}$ fits $\hat p_t = \mathcal{A}(D_t)$ where $\mathcal{A}$ is a fixed training procedure and

$$D_t \;=\; \underbrace{R_t}_{\text{real, } n\alpha_t \text{ samples}} \;\cup\; \underbrace{\bigcup_{s<t} S_s}_{\text{synthetic}},\qquad S_s \sim \hat p_s^{\otimes m_s},\qquad |D_t| = n.$$

**Measured quantities.**

- $\alpha_t = |R_t|/|D_t| \in [0,1]$ — real fraction. Measured by provenance tagging; in practice unmeasurable on web crawls (§6).
- $\mathrm{Err}_t = \mathbb{E}_{x\sim p_0}[-\log \hat p_t(x)] - H(p_0)$ — excess cross-entropy on a **frozen, pre-2022 held-out real corpus**. Estimated by mean token NLL over $\geq 10^7$ held-out tokens; $H(p_0)$ is unknown, so only differences $\mathrm{Err}_t - \mathrm{Err}_0$ are identified.
- Tail coverage $C_t(\varepsilon) = \hat p_t(\{x : p_0(x) < \varepsilon\})$ — measured by the mass $\hat p_t$ assigns to held-out documents in the bottom $\varepsilon$-quantile of a reference model's likelihood.
- Diversity $V_t$ — distinct-$n$-gram rate, or eigenvalue entropy of the embedding covariance of $10^4$ samples.

**Decision predicate.** For a schedule $\alpha_t \sim c\,t^{-\gamma}$, is
$$\gamma^\star := \inf\{\gamma : \limsup_{t\to\infty}\mathrm{Err}_t = \infty\}$$
finite, and what is its value? Pure accumulation is $\gamma = 1$ (real fraction $1/(t+1)$); replace is $\gamma = \infty$.

**Conjecture (variance-summation).** Under isotropic linear regression, per-generation injected noise enters additively with weight $\propto \alpha_t^{-2}$-free terms, giving boundedness iff $\sum_t t^{-2} < \infty$ — the same $\pi^2/6$ constant that appears in the accumulate analysis. If the mechanism generalises, $\gamma^\star = 1/2$ for $\alpha_t \sim t^{-\gamma}$ with error growth $\Theta(t^{2\gamma - 1})$. Unproved beyond linear models.

**Assumptions, with violations flagged.**

1. $\mathcal{A}$ is fixed across generations — **violated**: architectures, tokenizers and data filters change every 6–18 months.
2. Synthetic samples are i.i.d. from $\hat p_t$ — **violated**: real deployments sample at temperature $<1$, apply top-$p$, RLHF, and rejection sampling. Curation is a selection operator, not a sampler, and provably changes the fixed point (Ferbach et al., 2024).
3. Provenance is observable — **violated**: no reliable synthetic-text detector exists at web scale.
4. $p_0$ is stationary — **violated**: the real distribution drifts, and post-2022 "real" text is partly synthetic.

## 3. State of the Art

**Theory SOTA (established).**
- Gerstgrasser et al. (COLM 2024): for linear regression with accumulation, test error is bounded uniformly in $t$ by a constant of order $\sigma^2\pi^2/6$; under replacement it grows linearly in the number of generations.
- Dohmatob, Feng, Kempe (NeurIPS 2024, *Model Collapse Demystified*): exact expressions for ridge/random-projection regression showing test error under replace grows as $\Theta(t)$ and a modified scaling law under synthetic-data mixing.
- Dohmatob et al. (*Strong Model Collapse*, ICLR 2025): in high-dimensional regression, a synthetic fraction as small as $\sim 1\%$ of the corpus can flatten the scaling curve — more data stops helping. **Established for the analysed model class only**; the extrapolation to transformers is claimed, not ablated.
- Bertrand et al. (ICLR 2024): iterative retraining is locally stable if the initial model is close enough to $p_0$ *and* the real-data fraction exceeds a model-dependent constant — an existence result, not a computable threshold.

**Empirical SOTA.**
- Shumailov et al. (*Nature*, 2024): recursive fine-tuning of OPT-125M on wikitext-2 over 9 generations; perplexity on real held-out text rises monotonically and low-probability tails vanish. Retaining 10% real data slows but does not stop the rise.
- Gerstgrasser et al.: transformers on TinyStories, diffusion models on CelebA-HQ, VAEs on CIFAR — accumulate keeps test loss roughly flat over 5–10 generations; replace degrades in all three.
- Kazdan et al. (2024, *Collapse or Thrive?*): accumulate-subsample (fixed corpus size, uniform subsample of the accumulated pool) interpolates between the two regimes.

**Benchmark-number-only.** All "no collapse under accumulation" claims rest on $\leq 10$ generations at $\leq 10^9$ parameters with a *fixed* generator recipe. Nothing establishes the $t\to\infty$ behaviour, and no published run holds compute fixed while varying $\alpha_t$ on a schedule.

## 4. What Is Known

| Result | Scale measured | Number |
|---|---|---|
| Replace-loop error growth | linear regression, analytic | $\Theta(t)$, exact constants |
| Accumulate bounded | linear regression, analytic | $\leq \sigma^2\pi^2/6 \approx 1.645\,\sigma^2$ |
| Accumulate flat empirically | GPT-2-scale (9M–125M) on TinyStories, 5–10 gens | test cross-entropy change within noise |
| Replace collapses in LMs | OPT-125M, wikitext-2, 9 gens | perplexity rises monotonically; tail $n$-grams disappear by gen ~5 |
| Small synthetic fraction hurts scaling | high-dim regression, analytic + toy nets | $\sim 1\%$ synthetic suffices to flatten the curve |
| Verification restores scaling | Feng et al. (2024), synthetic-data pipelines with a verifier | oracle/weak verifier converts collapse into improvement |

Reliable regularity: **variance collapses before mean shifts**. Diversity metrics ($V_t$, tail coverage) move 2–4 generations before held-out cross-entropy does, across GMMs, VAEs, diffusion and LMs.

## 5. What Is Not Known

- **Theoretically open.** Whether a sharp threshold $\gamma^\star$ exists for schedules $\alpha_t \sim t^{-\gamma}$ in any model class beyond linear/ridge regression. No proof either way for transformers, or even for a two-layer network with nonlinearity. The conjecture $\gamma^\star = 1/2$ is untested.
- **Theoretically open.** Behaviour under *curated* self-consumption with a non-oracle verifier: Ferbach et al. show curation optimises the reward, but no result bounds the drift of $\hat p_t$ from $p_0$ when the verifier itself is a collapsing model.
- **Empirically open.** No run exceeds ~10 generations at $\geq 1$B parameters with a fixed FLOP budget per generation. The experiment is runnable today for roughly $10^{21}$–$10^{22}$ FLOP total; nobody has run it.
- **Methodologically blocked.** Collapse is not operationally separable from ordinary distribution shift, because (a) $\alpha_t$ is unmeasurable on real corpora and (b) held-out "real" evaluation sets after 2022 are themselves contaminated with model output. There is no accepted metric for "tail loss" that is comparable across model scales.

## 6. Why It Is Hard

Four named obstructions.

1. **Confounded measurement.** In accumulate, $|D_t|$ grows with $t$. A flat loss curve could mean "no collapse" or "collapse exactly cancelled by more tokens". No published run separates them by holding tokens-seen fixed while varying $\alpha_t$.
2. **Absent ground truth on $\alpha_t$.** Every real-world estimate of the synthetic fraction of the web is an extrapolation from detectors with unmeasured false-positive rates. The independent variable of the experiment is unobservable in the deployment setting the problem is about.
3. **Non-identifiability of collapse vs. curation.** Temperature $<1$ sampling and RLHF are entropy-reducing operators applied deliberately. A drop in $V_t$ is consistent with both "collapse" and "successful alignment". The two have opposite normative signs and the same measurement signature.
4. **Compute cost of the asymptotic regime.** The interesting claim is about $t \to \infty$. Thresholds in linear theory only separate at $t \gtrsim 20$–50 generations; at 1B parameters and 20B tokens per generation, 30 generations is $\sim 3.6\times10^{21}$ FLOP of training plus a comparable amount of *sampling* to regenerate the corpus, and sampling is the dominant unbudgeted cost.

## 7. Current Research (as of 2026)

- **Sharp-threshold theory.** Kempe's group (NYU) and collaborators (Dohmatob, Feng) continue the exact-asymptotics line from regression toward random-feature and two-layer models. *(frontier — verify)*
- **Verification-gated synthesis.** The dominant applied direction: filter synthetic data with a verifier (unit tests, proof checkers, reward models) before recycling. Established to work where verification is cheap and sound (code, math); open where the verifier is itself learned.
- **Provenance and watermarking** as a way to make $\alpha_t$ measurable — C2PA-style metadata for images, watermark detection for text. Fragile to paraphrase; not deployed at corpus scale.
- **Accumulate-subsample under fixed budget** (Stanford / Kazdan et al. line) — the closest existing work to the threshold question. *(frontier — verify)*
- **Multi-model ecosystems**: what happens when generation $t$ trains on the pooled output of $k$ different model families, which may de-correlate errors. Largely unstudied.

## 8. Concrete Next Experiment

**Question.** Does bounded error under accumulation survive when the real-data fraction decays faster than $1/t$, at fixed compute?

**Setup.** One 1.3B-parameter decoder-only LM, fixed architecture, tokenizer and hyperparameters. Fixed budget per generation: 26B training tokens (Chinchilla-optimal, $\approx 2\times10^{20}$ FLOP), $T = 25$ generations. Real pool: a frozen 300B-token pre-2021 corpus (e.g. a Pile/C4 snapshot), which is never exhausted. At generation $t$, build a 26B-token corpus with real fraction $\alpha_t = \min(1, t^{-\gamma})$, the remainder sampled uniformly from the accumulated synthetic pool $\bigcup_{s<t}S_s$; generate 26B new synthetic tokens per generation at temperature 1.0, no filtering.

**Arms.** $\gamma \in \{0.5,\,1.0,\,1.5\}$, plus two controls: (a) $\alpha_t \equiv 1$ (real-only, same token budget, isolates the effect of re-sampling a fixed pool), and (b) $\alpha_t \equiv 0$ for $t \geq 1$ (replace). Five arms $\times$ 25 generations $\approx 2.5\times10^{22}$ FLOP training plus sampling; ~3–5 M GPU-hours on H100-class hardware, or run the identical design at 160M/3.3B tokens for a ~60× cheaper pilot.

**Deciding number.** Fit $\mathrm{Err}_t - \mathrm{Err}_0 = a\,t^{b}$ over $t \in [10,25]$ on frozen pre-2021 held-out text ($10^8$ tokens). The single decision statistic is $b$ for the $\gamma = 0.5$ arm, with a bootstrap 95% CI over held-out shards. $b$ CI containing $0$ and $|a| < 0.01$ nats ⇒ bounded, conjecture $\gamma^\star \geq 0.5$ falsified downward. $b > 0$ with CI excluding 0 for $\gamma=1.5$ but not $\gamma=0.5$ ⇒ threshold lies in $(0.5, 1.5)$, and the $2\gamma-1$ prediction is checked by comparing $b$ against $2\gamma - 1$.

## 9. Key References

- **[Foundational]** Shumailov, Shumaylov, Zhao, Papernot, Anderson, Gal. *AI models collapse when trained on recursively generated data.* Nature 631, 2024. (Earlier version: *The Curse of Recursion: Training on Generated Data Makes Models Forget*, arXiv:2305.17493.)
- **[Foundational]** Alemohammad, Casco-Rodriguez, Luzi, Humayun, Babaei, LeJeune, Siahkoohi, Baraniuk. *Self-Consuming Generative Models Go MAD.* ICLR 2024 — arXiv:2307.01850.
- **[SOTA]** Gerstgrasser, Schaeffer, Dey, Rafailov, Sleight, Hughes, Korbak, Agrawal, Pai, Gromov, Roberts, Yang, Donoho, Koyejo. *Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data.* COLM 2024 — arXiv:2404.01413.
- **[SOTA]** Dohmatob, Feng, Kempe. *Model Collapse Demystified: The Case of Regression.* NeurIPS 2024 — arXiv:2402.07712.
- **[SOTA]** Dohmatob, Feng, Subramonian, Kempe. *Strong Model Collapse.* ICLR 2025 — arXiv:2410.04840.
- **[Theory]** Bertrand, Bose, Duplessis, Jiralerspong, Gidel. *On the Stability of Iterative Retraining of Generative Models on their own Data.* ICLR 2024 — arXiv:2310.00429.
- **[Theory]** Ferbach, Bertrand, Bose, Gidel. *Self-Consuming Generative Models with Curated Data Provably Optimize Human Preferences.* NeurIPS 2024 — arXiv:2407.09499.
- **[Method]** Feng, Dohmatob, Yang, Charton, Kempe. *Beyond Model Collapse: Scaling Up with Synthesized Data Requires Verification.* 2024 — arXiv:2406.07515.
- **[Empirical]** Kazdan, Schaeffer, Dey, Gerstgrasser, Rafailov, Donoho, Koyejo. *Collapse or Thrive? Perils and Promises of Synthetic Data in a Self-Generating World.* 2024 — arXiv:2410.16713.
- **[Analysis]** Seddik, Chen, Hayou, Youssef, Debbah. *How Bad is Training on Synthetic Data? A Statistical Analysis of Language Model Collapse.* COLM 2024 — arXiv:2404.05090.

## 10. Worked Example

Take the simplest case where the arithmetic is exact: estimating the mean of a 1-D Gaussian. $p_0 = \mathcal{N}(0,1)$. Each generation fits $\hat\mu_t$ from $n$ samples and emits $\mathcal{N}(\hat\mu_t, 1)$.

**Replace.** $\hat\mu_t = \hat\mu_{t-1} + \varepsilon_t$, $\varepsilon_t \sim \mathcal{N}(0, 1/n)$. Then $\mathrm{Var}(\hat\mu_t) = t/n$ — a random walk, unbounded. With $n = 10^4$, after $t=100$ generations the mean has drifted $\sim 0.1\sigma$; after $10^4$ generations, $1\sigma$. Excess KL is $t/2n$, linear in $t$.

**Accumulate.** Generation $t$ averages $n$ real samples with all prior synthetic samples. The real samples never lose weight relative to any single later cohort, and the induced increments decay: $\mathrm{Var}(\hat\mu_t) \approx \frac{1}{n}\sum_{s=1}^{t} s^{-2} \leq \frac{\pi^2}{6n} = \frac{1.645}{n}$. With $n=10^4$: standard deviation $0.0128$ **forever**. This is the $\pi^2/6$ constant in its bare form.

**Decayed injection, $\alpha_t = t^{-\gamma}$.** Now generation $t$'s estimate is dominated by synthetic data whenever $\gamma > 1$, and the increment variance scales as $\alpha_t^{-1}/n$ contributions accumulating: $\mathrm{Var}(\hat\mu_t) \approx \frac{1}{n}\sum_{s\leq t} s^{2\gamma-2}$, which converges iff $\gamma < 1/2$ and grows as $t^{2\gamma-1}$ otherwise. At $\gamma = 1$ this gives $\mathrm{Var} \approx t/n$ — the same random walk as replace, despite $\alpha_t > 0$ at every step.

**Where the obstruction becomes visible.** In this toy model the three arms are trivially distinguishable because $\hat\mu_t$ is directly observed. In a language model you observe only $\mathrm{Err}_t$, and at $n$-equivalent scale the predicted separation between $\gamma=0.5$ and $\gamma=1.0$ after 25 generations is $\Delta\mathrm{Err} \approx 0.02$ nats/token — smaller than the seed-to-seed variance of a 1.3B pretraining run (typically 0.01–0.03 nats). The threshold is therefore *not* detectable without multiple seeds per arm, which multiplies the $2.5\times10^{22}$ FLOP budget by three. That, and not the theory, is why the question is still open.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*