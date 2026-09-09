---
id: 10-scaling-laws/intrinsic-dimension-scaling-exponent
title: "Data-Dimension Scaling Exponent Theory"
topic: 10-scaling-laws
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Data-Dimension Scaling Exponent Theory

> **Topic:** Scaling Laws & Compute Allocation · **ID:** `10-scaling-laws/intrinsic-dimension-scaling-exponent` · **Status:** partially-solved

## 1. Problem Statement

Neural loss curves fall as power laws: $L \approx L_\infty + A N^{-\alpha_N}$ in parameters and $L \approx L_\infty + B D^{-\alpha_D}$ in data. The exponents $\alpha$ are small (0.05–0.4) and stubborn — they set how much compute buys how much loss, and therefore the entire economics of pretraining. **Data-dimension scaling exponent theory** claims the exponent is set by the intrinsic dimension $d$ of the data manifold, canonically $\alpha \approx 4/d$ (Sharma & Kaplan, JMLR 2022).

Three variants, of very different difficulty:

- **Measurement.** Given a trained model and a dataset, estimate $d$ and $\alpha$ independently and test whether $\alpha d \approx 4$ (or whatever constant the theory names). Runnable today; the estimators disagree.
- **Method.** Given a target exponent, change $d$ — by curating, pruning, or re-representing data — and observe the predicted change in $\alpha$. This is the causal claim, and it is the one with money attached.
- **Theory.** Prove that for a stated function class, architecture, and optimizer, the resolution-limited exponent is $\alpha = f(d, s)$ with $s$ a smoothness index, and that the proof's constants are not vacuous at realistic $d \sim 10$–$50$.

Solving it means: a measurement of $d$ from data alone that predicts the observed $\alpha$ within stated error, plus at least one intervention on $d$ that moves $\alpha$ in the predicted direction and magnitude.

## 2. Formal Setting

Data are i.i.d. draws $x \sim \mu$ supported on (or concentrated near) a set $\mathcal{M} \subset \mathbb{R}^{D_{\text{amb}}}$, with target $y = f^*(x)$.

**Intrinsic dimension, as measured.** Two operational definitions, and they are not the same number.

1. *Nearest-neighbour / TwoNN* (Facco et al., Sci. Rep. 2017): for each point let $r_1, r_2$ be distances to its first two neighbours, $\rho = r_2/r_1$. Under a locally uniform Poisson model $\Pr(\rho > t) = t^{-d}$, so
$$\hat d_{\text{2NN}} = \frac{n}{\sum_{i=1}^{n} \log \rho_i}.$$
2. *MLE / Levina–Bickel* (NIPS 2004), with $k$ neighbours: $\hat d_k(x)^{-1} = \frac{1}{k-1}\sum_{j=1}^{k-1}\log \frac{T_k(x)}{T_j(x)}$, $T_j$ the distance to the $j$-th neighbour.

Both estimate the exponent of the small-ball mass function $\mu(B(x,r)) \sim r^{d}$ — a *local* quantity, at a scale set by $n$. Neither measures a global manifold dimension.

**Exponent, as measured.** Fit $L(N) = L_\infty + A N^{-\alpha_N}$ by nonlinear least squares (or Huber loss on $\log L$, as in Hoffmann et al. 2022) over a model-size sweep at fixed data, with $L$ the held-out cross-entropy in nats/token. $\alpha_N$ is identified only jointly with $L_\infty$; the fitted $\alpha$ is strongly sensitive to the assumed irreducible loss.

**The claim under test.** Resolution-limited regime, smooth target, interpolating predictor:
$$\alpha \;=\; \frac{4}{d} \quad\text{(Sharma \& Kaplan)}, \qquad\text{or more generally}\qquad \alpha \;=\; \frac{2s}{d} \ \text{or}\ \frac{2s}{2s+d}$$
with $s$ the target's smoothness — the classical Stone (1982) minimax rate $n^{-2s/(2s+d)}$ recovers the second form.

**Assumptions, and which are violated.**
- *Manifold support with a single $d$.* Violated: measured ID of text and image data varies by orders of magnitude across regions and by a factor of 2–4 across scales; the distribution is multi-fractal, not a manifold.
- *Noiseless target, $L_\infty$ known.* Violated: for language, entropy of the data is unknown, and $L_\infty$ is a free fit parameter that trades off against $\alpha$.
- *Interpolation / near-zero training error.* Violated: frontier LLMs are trained at or near one epoch, far from interpolation.
- *Isotropy of the metric.* Violated: ID estimators in $\mathbb{R}^{D_{\text{amb}}}$ on token embeddings depend on the embedding, which is itself learned.

## 3. State of the Art

**Theory SOTA (established).** Bahri, Dyer, Kaplan, Lee & Sharma, *Explaining Neural Scaling Laws* (PNAS 2024; arXiv:2102.06701) separate a **variance-limited** regime (exponent 1, independent of $d$) from a **resolution-limited** regime where the exponent is set by the decay of the kernel/feature spectrum, which for a $d$-dimensional manifold and a smooth target gives $\alpha \propto 1/d$. This is a theorem in the kernel-regression setting, not for trained transformers. Havrilla & Liao (NeurIPS 2024, arXiv:2411.06646) prove approximation-plus-estimation bounds for transformers on data with low intrinsic dimension, giving rates in $d$ rather than $D_{\text{amb}}$ — the strongest statement that transformers *do* adapt to low-dimensional structure.

**Competing theory (established, incompatible mechanism).** Hutter's *Learning Curve Theory* (arXiv:2102.04074) and Michaud et al.'s *Quantization Model of Neural Scaling* (NeurIPS 2023, arXiv:2303.13506) derive the same power-law shape from a **Zipf** distribution over discrete skills or features, with $\alpha$ set by the Zipf exponent and no reference to dimension. Maloney, Roberts & Sully (arXiv:2210.16859) get it from a power-law latent covariance spectrum. All three fit the data.

**Empirical SOTA (established).** Sharma & Kaplan (JMLR 2022) verify $\alpha \approx 4/d$ in teacher–student settings where $d$ is *set by construction*, and report agreement within tens of percent on CNN image tasks with $d$ measured by nearest-neighbour estimators.

**Claimed but unablated.** That the $4/d$ relation holds for language modelling. Sharma & Kaplan's own inversion of Kaplan et al.'s $\alpha_N = 0.076$ gives $d \approx 53$ — a number with no independent measurement confirming it. No published work has changed the measured ID of a pretraining corpus and reported the resulting change in $\alpha_N$.

**Benchmark-number-only results.** Reported IDs of LLM hidden representations (Ansuini et al., NeurIPS 2019, for CNNs; Tulchinskii et al., NeurIPS 2023, for text embeddings, $\hat d \sim 9$–$12$) are estimator outputs on specific layers, not validated against any scaling exponent.

## 4. What Is Known

- **Exponents are small and reproducible within a setup.** Kaplan et al. (2020): $\alpha_N = 0.076$, $\alpha_D = 0.095$, measured 768 to $1.5\times10^9$ non-embedding parameters on WebText2.
- **They are not stable across setups.** Hoffmann et al. (2022, Chinchilla), over 400 models from 70M to 16B params and 5B–500B tokens, fit $a = 0.34$, $b = 0.28$ — a 4.5$\times$ larger parameter exponent on the same modality. The difference is attributed to LR-schedule and fitting methodology, not to the data.
- **Data ID is measurable and matters for sample complexity.** Pope et al. (ICLR 2021, arXiv:2104.08894) measure ImageNet ID $\approx 26$–$43$ (MLE, $k$-dependent), and show on GAN-generated data with *controlled* latent dimension that classification sample complexity grows with ID and is insensitive to ambient dimension.
- **The exponent is not a property of the data alone.** Sorscher et al. (NeurIPS 2022) beat power-law scaling by pruning: with a good example-difficulty score, test error on ImageNet/CIFAR falls faster than any fixed power law over the retained-fraction sweep. Same underlying distribution, different exponent.
- **Two regimes exist.** The variance-limited exponent 1 predicted by Bahri et al. is observed in the small-$D$/wide-model corner; the resolution-limited regime is where $d$ could enter.

## 5. What Is Not Known

- **Theoretically open.** Whether $\alpha = 4/d$, or any $d$-only law, holds for a trained transformer under SGD on a non-manifold, heavy-tailed distribution. Existing proofs (Havrilla & Liao; Nakada & Imaizumi, JMLR 2020; Chen et al., *Information and Inference* 2022) bound rates for ERM over a function class, not for the optimizer's actual output, and their constants are exponential in $d$.
- **Theoretically open.** Whether the dimension mechanism and the Zipf/quanta mechanism are the same statement in different coordinates. No reduction either way exists.
- **Empirically open.** No one has run the causal experiment: hold architecture, tokenizer, and token budget fixed, vary corpus intrinsic dimension by a controlled factor, and measure $\Delta \alpha_N$. It is runnable at 100M–1B scale for well under $10^{21}$ FLOPs.
- **Methodologically blocked.** "The" intrinsic dimension of a text corpus is not well defined. TwoNN and Levina–Bickel estimates depend on the embedding, the neighbourhood scale, and $n$; both are known to underestimate badly for $d \gtrsim 20$ at achievable $n$, since $n$ must grow roughly exponentially in $d$.

## 6. Why It Is Hard

**Non-identifiability, and it is structural.** In $L = L_\infty + A N^{-\alpha}$, the triple $(L_\infty, A, \alpha)$ is jointly fit over at most 2–3 decades of $N$. Shifting $L_\infty$ by 0.05 nats moves the fitted $\alpha$ by tens of percent. Kaplan-vs-Chinchilla is exactly this: the same modality, exponents differing 4.5$\times$, implied $d$ differing 4.5$\times$. A theory that predicts $d$ from $\alpha$ inherits that whole uncertainty.

**Estimator bias in the regime that matters.** ID estimators need $n \sim e^{cd}$ samples. At $d \approx 50$ — the value $\alpha_N = 0.076$ implies — no achievable $n$ gives an unbiased estimate; every estimator returns something smaller. So the theory's prediction lands precisely where its measurement instrument fails, and a "disagreement" between $4/\hat\alpha$ and $\hat d$ is unfalsifiable.

**Absent ground truth.** For synthetic teacher–student data $d$ is known and the theory works. For real corpora there is no ground-truth $d$ to compare against, only competing estimators.

## 7. Current Research (as of 2026)

- Dynamical-mean-field and random-feature derivations of scaling exponents from spectra rather than dimension: Bordelon, Atanasov & Pehlevan (ICML 2024); Paquette et al., *4+3 Phases of Compute-Optimal Neural Scaling Laws* (NeurIPS 2024). These treat $d$ as downstream of a spectral exponent, which is the more defensible primitive.
- Statistical-physics models deriving power laws from data structure directly — e.g. percolation-style accounts of feature-frequency distributions *(frontier — verify)*.
- Approximation-theory groups (Liao and collaborators, Georgia Tech/Duke) extending low-intrinsic-dimension rates from ReLU nets to transformers and to next-token prediction.
- Data-curation work at frontier labs testing whether pruning/filtering changes $\alpha$ rather than only the offset $A$ — largely unpublished *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question.** Does a controlled change in corpus intrinsic dimension change $\alpha_N$, or only the prefactor $A$?

**Scale.** Six model sizes, 30M–1.4B non-embedding params, decoder-only transformer, fixed tokenizer, 20 tokens/param (Chinchilla-optimal), cosine schedule fully decayed at each size — 36 runs total, roughly $10^{21}$ FLOPs, days on 64 H100s.

**Arms.** Build three corpora of identical token count from the same source pool, differing only in measured ID of a fixed frozen sentence-embedder's representations:
- **Control:** uniform random subsample. Measure $\hat d_{\text{2NN}}$ and $\hat d_{\text{MLE}}$.
- **Low-$d$:** subsample restricted to a few semantic clusters, target $\hat d$ at $0.5\times$ control.
- **High-$d$:** maximally diverse subsample, target $\hat d$ at $1.5\times$ control.

Fit $\alpha_N$ per arm with $L_\infty$ free, and again with $L_\infty$ pinned across arms; report both, with bootstrap CIs over the six-point sweep.

**Deciding number.** $\alpha_N^{\text{low}} / \alpha_N^{\text{high}}$. The $4/d$ theory predicts $3.0$ (inverse of the $0.5$:$1.5$ ID ratio). A pure-offset account predicts $1.0$. The experiment decides if the bootstrap CI on that ratio excludes one of $\{1.0, 3.0\}$ — which needs the CI half-width below $\pm 0.5$, achievable only if per-arm $\alpha_N$ is determined to $\pm 8\%$. If it is not, the honest output is that the measurement is blocked, which is itself the finding.

## 9. Key References

- **[Foundational]** Utkarsh Sharma, Jared Kaplan. *A Neural Scaling Law from the Dimension of the Data Manifold.* JMLR, 2022 (arXiv:2004.10802).
- **[Foundational]** Jared Kaplan et al. *Scaling Laws for Neural Language Models.* 2020 — arXiv:2001.08361.
- **[SOTA — theory]** Yasaman Bahri, Ethan Dyer, Jared Kaplan, Jaehoon Lee, Utkarsh Sharma. *Explaining Neural Scaling Laws.* PNAS, 2024 — arXiv:2102.06701.
- **[SOTA — theory]** Alex Havrilla, Wenjing Liao. *Understanding Scaling Laws with Statistical and Approximation Theory for Transformer Neural Networks on Intrinsically Low-dimensional Data.* NeurIPS, 2024 — arXiv:2411.06646.
- **[SOTA — empirical]** Jordan Hoffmann et al. *Training Compute-Optimal Large Language Models.* NeurIPS, 2022 — arXiv:2203.15556.
- **[Competing mechanism]** Eric J. Michaud, Ziming Liu, Uzay Girit, Max Tegmark. *The Quantization Model of Neural Scaling.* NeurIPS, 2023 — arXiv:2303.13506.
- **[Competing mechanism]** Marcus Hutter. *Learning Curve Theory.* 2021 — arXiv:2102.04074.
- **[Measurement]** Phillip Pope, Chen Zhu, Ahmed Abdelkader, Micah Goldblum, Tom Goldstein. *The Intrinsic Dimension of Images and Its Impact on Learning.* ICLR, 2021 — arXiv:2104.08894.
- **[Measurement]** Elena Facco, Maria d'Errico, Alex Rodriguez, Alessandro Laio. *Estimating the intrinsic dimension of datasets by a minimal neighborhood information.* Scientific Reports, 2017.
- **[Measurement]** Elizaveta Levina, Peter Bickel. *Maximum Likelihood Estimation of Intrinsic Dimension.* NIPS, 2004.
- **[Counterexample]** Ben Sorscher, Robert Geirhos, Shashank Shekhar, Surya Ganguli, Ari Morcos. *Beyond neural scaling laws: beating power law scaling via data pruning.* NeurIPS, 2022 — arXiv:2206.14486.
- **[Classical bound]** Charles J. Stone. *Optimal Global Rates of Convergence for Nonparametric Regression.* Annals of Statistics, 1982.

## 10. Worked Example

Invert the theory on the two best-known language-model fits.

| Fit | $\alpha_N$ | $d = 4/\alpha_N$ |
|---|---|---|
| Kaplan et al. 2020 | 0.076 | 52.6 |
| Hoffmann et al. 2022 | 0.34 | 11.8 |

Both are fits to autoregressive English text with transformers. The theory therefore assigns the *same modality* two intrinsic dimensions differing by 4.5$\times$. The discrepancy is not a data property: Hoffmann et al. attribute it to fully decaying the learning-rate schedule at each model size, which Kaplan et al. did not do.

Now bring in a measurement. TwoNN on frozen sentence embeddings of natural English returns $\hat d \approx 9$–$12$ (Tulchinskii et al., NeurIPS 2023). Taken at face value this endorses Chinchilla ($11.8$) and refutes Kaplan ($52.6$).

But the estimator cannot adjudicate. To resolve $d = 50$ with TwoNN you need roughly $n \sim e^{cd}$ points at the relevant radius; at any $n$ reachable on a corpus, the estimator's small-ball fit saturates and returns a number in the low tens regardless of the truth. So $\hat d \approx 10$ is exactly what you would observe whether the true $d$ is 10 or 50.

The chain closes on itself: $\alpha$ is under-determined because $L_\infty$ is free, $d$ is under-determined because the estimator is biased downward in precisely the range at issue, and the only bridge between them is the theory being tested. That is the obstruction — not compute, but two unidentified quantities linked by the hypothesis they are supposed to test. Section 8's design breaks it by making $d$ an *intervention* with a known ratio rather than an estimated absolute value.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*