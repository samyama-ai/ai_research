---
id: 10-scaling-laws/data-quality-filtering-exponent
title: "Data Quality Filtering Exponent Shift"
topic: 10-scaling-laws
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Data Quality Filtering Exponent Shift

> **Topic:** Scaling Laws & Compute Allocation · **ID:** `10-scaling-laws/data-quality-filtering-exponent` · **Status:** open

## 1. Problem Statement

Pretraining data filtering — dedup, classifier-based quality scoring, perplexity pruning, prototype pruning — reliably lowers loss at fixed compute. The open question is *which parameter of the scaling law it moves*.

Write the data-side loss as $L(D) = E + B D^{-\beta}$. A filter $f$ maps a raw pool to a subset and induces $(E_f, B_f, \beta_f)$. Three mutually exclusive claims:

- **Coefficient shift:** $\beta_f = \beta$, $B_f < B$. The filter buys a *constant* data multiplier at every scale. Nice, but bounded: it never changes the exponent of the compute–loss curve.
- **Exponent shift:** $\beta_f > \beta$. The advantage *grows* without bound with scale. This is the claim implicit in "quality beats quantity" narratives and in the pruning theory of Sorscher et al. (2022).
- **Offset shift:** $E_f < E$ with $\beta$ unchanged — the filter changes the target distribution's irreducible entropy, which is a change of task, not an improvement.

**Measurement variant:** given a fixed pool and a filter, estimate $\Delta\beta = \beta_f - \beta$ with a confidence interval that excludes 0. **Method variant:** construct a filter for which $\Delta\beta > 0$ is demonstrated, not assumed. **Theory variant:** prove for some non-trivial data model whether any measurable, pool-fraction-bounded filter can raise $\beta$, or prove it cannot. Solving the page means resolving the measurement variant for at least one production-grade filter at $\geq 3$ decades of $D$.

## 2. Formal Setting

- **Pool.** $\mathcal{P} = \{x_1,\dots,x_M\}$, $M$ tokens, i.i.d. from crawl distribution $q$. Measured as deduplicated token count under a fixed tokenizer.
- **Filter.** $f: \mathcal{X} \to \{0,1\}$ with keep rate $\rho = |f(\mathcal{P})|/M$, measured as the surviving token fraction. Induced distribution $q_f(x) \propto q(x) f(x)$.
- **Training set.** $D$ tokens *seen*, with epoch count $R = D/(\rho M)$. $R$ is the confound: filtering shrinks the pool, so at fixed $D$ a filtered run repeats data more.
- **Loss.** $L$ = mean next-token cross-entropy in nats on a **held-out evaluation distribution $p$ that is fixed across arms** and drawn from neither $q$ nor $q_f$ (e.g. held-out Wikipedia + arXiv + code, or an aggregated benchmark log-likelihood). Measured with the same tokenizer and the same sequence length in both arms.
- **Joint law** (Chinchilla parameterization, Hoffmann et al. 2022):

$$L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}$$

- **Filtered law**, with repetition decay in the style of Muennighoff et al. (2023):

$$L_f(N,D) = E_f + \frac{A}{N^{\alpha}} + \frac{B_f}{D_{\mathrm{eff}}^{\beta_f}}, \qquad D_{\mathrm{eff}} = \rho M \cdot U(R), \quad U(R) = \sum_{r=0}^{R-1} e^{-r/R_D^*}$$

- **Decision quantity.** $\Delta\beta = \beta_f - \beta$. Equivalent operational form: the *data multiplier* $m(D)$ solving $L_f(D) = L(m^{-1}D)$. Coefficient shift $\Rightarrow m$ constant; exponent shift $\Rightarrow m(D) \propto D^{\Delta\beta/\beta}$.

**Assumptions, and which are violated.**
1. *Single power law over the fit range* — violated: fits drift with range (Choshen et al. 2024 find exponent estimates move materially when the smallest models are dropped).
2. *$E$ common across arms* — violated whenever the filter changes the effective task mix; $E_f \neq E$ is exactly what makes $\Delta\beta$ non-identifiable.
3. *i.i.d. pool, no epoch effects* — violated: aggressive filters force $R > 1$ at frontier $D$.
4. *Filter independent of the evaluation set* — violated in practice; most production filters are classifier-trained on data resembling the benchmarks used to score them.

## 3. State of the Art

**Theory SOTA.** Sorscher et al. (NeurIPS 2022) give the only clean exponent result: in a perceptron teacher–student model with an oracle example-difficulty ordering, pruning at a scale-dependent rate turns power-law scaling into *exponential* scaling in the pruned-set size. This is **established as a theorem in that model**, and it requires (i) an oracle margin, (ii) a pool that grows with the target, (iii) the optimal keep rate shrinking as data becomes abundant. None hold for web pretraining. Bahri et al. (PNAS 2024) tie $\beta$ to the intrinsic dimension of the data manifold, which predicts filtering changes $\beta$ only if it changes the manifold dimension — a prediction never tested.

**Empirical SOTA.** DataComp (Gadre et al., NeurIPS 2023 D&B) and DataComp-LM (Li et al., 2024) established filtering as the dominant lever at fixed compute: DataComp-1B trains CLIP ViT-L/14 to 79.2% zero-shot ImageNet vs. 75.5% for OpenAI CLIP with ~2.7× less compute; DCLM-Baseline trains a 7B model to 64% 5-shot MMLU, ~6.6 points over the comparable open baseline with ~40% less compute. **These are benchmark numbers at a single compute point, not exponent estimates** — neither paper fits $\beta_f$ against $\beta$ on a matched ladder.

Goyal et al. (CVPR 2024, *Scaling Laws for Data Filtering*) is the closest to an ablation of the question: they show utility of a filtered subset decays with repeated epochs and that the compute-optimal filtering threshold *loosens* as compute grows. That is evidence against a pure exponent gain and for a coefficient gain that erodes — but it is measured in the CLIP/DataComp regime, not for LLM pretraining.

**Claimed but unablated:** every "quality beats quantity" claim descending from phi-1 (Gunasekar et al. 2023) and FineWeb-Edu (Penedo et al., NeurIPS 2024 D&B). They report large single-point wins with no $\beta$ estimate and no pool-exhaustion control.

## 4. What Is Known

- Unfiltered LLM data exponent: $\beta \approx 0.28$ ($E{=}1.69$, $A{=}406.4$, $B{=}410.7$), fit on 400+ models, 70M–16B params, up to 500B tokens (Hoffmann et al. 2022). Kaplan et al. (2020) fit $\beta \approx 0.095$ on a different corpus and eval — the exponent is corpus- and eval-dependent, at the same order as any claimed filtering effect.
- Repetition: up to ~4 epochs is near-free; value decays with half-life $R_D^* \approx 15$ epochs; beyond ~44 epochs additional compute is worthless (Muennighoff et al. 2023, up to 9B params, 900B tokens).
- Perplexity-based pruning keeping ~50% of the pool matches or beats the full pool downstream at 1B and 6.7B params (Marion et al. 2023) — a *coefficient-scale* result, single-scale.
- ImageNet pruning of ~20% of examples costs no top-1 accuracy for ResNet-50; self-supervised prototype metrics recover most of the supervised difficulty ordering (Sorscher et al. 2022).
- No published LLM study fits $\beta_f$ and $\beta$ on the same evaluation with matched pools and reports a confidence interval on $\Delta\beta$.

## 5. What Is Not Known

- **Theoretically open.** Whether any filter computable from the data alone (no oracle margin) can raise $\beta$ for a heavy-tailed skill distribution. The quantization model (Michaud et al. 2023) predicts $\beta$ is set by the Zipf exponent of skill frequency; whether filtering can change that exponent rather than only remove non-skill tokens is unproven either way.
- **Empirically open.** $\Delta\beta$ for a production filter (FineWeb-Edu classifier, DCLM fastText filter) over $\geq 3$ decades of $D$ with $R \leq 1$ in both arms. The experiment is runnable at ~$10^{22}$ FLOP total; nobody has published it.
- **Methodologically blocked.** Separating $E_f$ from $\beta_f$. If the filter changes the achievable entropy on the eval set, $(E_f, B_f, \beta_f)$ are jointly unidentifiable from loss curves over 2 decades — see §10. Also blocked: any $\Delta\beta$ measured on benchmarks the filter's classifier was trained toward.

## 6. Why It Is Hard

**Non-identifiability over the affordable range, compounded by pool exhaustion.**

Over 2 decades of $D$ — what a normal lab affords — a 10% coefficient reduction and a $\Delta\beta = +0.015$ exponent gain differ by ~0.01 nats in predicted loss (§10). Seed-to-seed val-loss variance at 1B params is of the same order. Distinguishing them requires ~4 decades. But the filter's own pool caps the range: a filter with $\rho = 0.1$ on a 20T-token crawl exhausts at 2T tokens, so the top decade of the filtered arm is run at $R > 1$, where repetition decay $U(R)$ *lowers* the apparent $\beta_f$ — biasing the measurement toward "no exponent gain" for exactly the aggressive filters most likely to have one. The two dominant effects are confounded by construction.

Second obstruction: absent ground truth on $E$. There is no independent estimate of the irreducible entropy of any eval set, so $E$ is a free parameter absorbing exponent differences.

## 7. Current Research (as of 2026)

- **Filtering-aware scaling laws.** Goyal/Maini/Lipton/Raghunathan/Kolter (CMU) on utility-vs-diversity decomposition; extension of the CVPR 2024 model to text pretraining is the natural next step *(frontier — verify)*.
- **Open data ladders.** HuggingFace (FineWeb / FineWeb-Edu) and the DataComp-LM consortium (UW, Apple, TRI, Columbia) now release pools plus multi-scale ladders, which makes the matched-arm fit cheap for the first time.
- **Data mixing laws.** Ye et al. (2024) and DoReMi (Xie et al., NeurIPS 2023) fit loss as a function of mixture weights; these are coefficient-level models by construction and do not currently estimate an exponent shift.
- **Scaling-law estimation methodology.** Choshen, Zhang, Andreas (2024) quantify how many models and what scale range are needed for a stable exponent — the direct prerequisite for any $\Delta\beta$ claim.

## 8. Concrete Next Experiment

**Scale.** One crawl pool of 20T deduplicated tokens. Two arms sharing tokenizer, architecture, schedule, and eval:
- **Control arm:** uniform random subsample of the raw pool at keep rate $\rho$ (same token count as the filtered arm — this is the arm most papers omit, and it controls for pool size and for repetition).
- **Treatment arm:** DCLM/FineWeb-Edu-style classifier filter at the same $\rho = 0.2$ (4T tokens).

Train a ladder at $N \in \{150\text{M}, 400\text{M}, 1\text{B}, 3\text{B}\}$ and, at each $N$, $D \in \{2, 20, 200, 2000\}\times 10^{9}$ tokens — 4 decades, $R \leq 1$ everywhere in both arms. 32 runs per arm, 3 seeds at the two smallest $N$. Total ≈ $1.2\times10^{22}$ FLOP, roughly 3k H100-days.

**Evaluation.** Held-out cross-entropy on a frozen mixture (Wikipedia, arXiv, GitHub, books, news) sampled *before* the filter classifier was trained and never used in filter development.

**The deciding number.** Fit $L = E + A N^{-\alpha} + B D^{-\beta}$ per arm by Huber loss on log-loss residuals; bootstrap over runs. Report the 95% CI on $\Delta\beta = \beta_f - \beta_{\text{ctrl}}$.
- CI excludes 0 and lower bound $> 0.01$ → **exponent shift confirmed**; report the implied multiplier growth $m(D) \propto D^{\Delta\beta/\beta}$.
- CI contains 0 with width $< 0.02$ → **coefficient shift**; filtering is a constant multiplier and its frontier value is capped at $(B/B_f)^{1/\beta}$.

## 9. Key References

- **[Foundational]** J. Kaplan, S. McCandlish, T. Henighan, et al. *Scaling Laws for Neural Language Models.* 2020. — arXiv:2001.08361
- **[Foundational]** J. Hoffmann, S. Borgeaud, A. Mensch, et al. *Training Compute-Optimal Large Language Models.* NeurIPS 2022. — arXiv:2203.15556
- **[Foundational/Theory]** B. Sorscher, R. Geirhos, S. Shekhar, S. Ganguli, A. Morcos. *Beyond neural scaling laws: beating power law scaling via data pruning.* NeurIPS 2022. — arXiv:2206.14486
- **[SOTA]** S. Goyal, P. Maini, Z. Lipton, A. Raghunathan, J. Z. Kolter. *Scaling Laws for Data Filtering — Data Curation cannot be Compute Agnostic.* CVPR 2024. — arXiv:2404.07177
- **[SOTA]** S. Gadre, G. Ilharco, A. Fang, et al. *DataComp: In search of the next generation of multimodal datasets.* NeurIPS 2023 Datasets & Benchmarks. — arXiv:2304.14108
- **[SOTA]** J. Li, A. Fang, G. Smyrnis, et al. *DataComp-LM: In search of the next generation of training sets for language models.* 2024. — arXiv:2406.11794
- **[SOTA]** N. Muennighoff, A. M. Rush, B. Barak, et al. *Scaling Data-Constrained Language Models.* NeurIPS 2023. — arXiv:2305.16264
- **[Empirical]** M. Marion, A. Üstün, L. Pozzobon, A. Wang, M. Fadaee, S. Hooker. *When Less is More: Investigating Data Pruning for Pretraining LLMs at Scale.* 2023. — arXiv:2309.04564
- **[Empirical]** G. Penedo, H. Kydlíček, L. von Werra, T. Wolf, et al. *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale.* NeurIPS 2024 Datasets & Benchmarks. — arXiv:2406.17557
- **[Theory]** Y. Bahri, E. Dyer, J. Kaplan, J. Lee, U. Sharma. *Explaining Neural Scaling Laws.* PNAS, 2024. — arXiv:2102.06701
- **[Theory]** E. Michaud, Z. Liu, U. Girit, M. Tegmark. *The Quantization Model of Neural Scaling.* NeurIPS 2023. — arXiv:2303.13506
- **[Methodology]** L. Choshen, Y. Zhang, J. Andreas. *A Hitchhiker's Guide to Scaling Law Estimation.* 2024. — arXiv:2410.11840
- **[Related]** S. M. Xie, H. Pham, X. Dong, et al. *DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining.* NeurIPS 2023. — arXiv:2305.10429

## 10. Worked Example

Take the Chinchilla data term as the control: $L(D) = 1.69 + 410.7\,D^{-0.28}$.

Two hypotheses about a filter that lowers reducible loss by 10% at $D = 10^{10}$:

| $D$ (tokens) | control | H1 coefficient ($B_f{=}369.6$, $\beta{=}0.28$) | H2 exponent ($B_f{=}523.3$, $\beta_f{=}0.295$) |
|---|---|---|---|
| $10^{10}$ | 2.342 | 2.277 | 2.277 |
| $10^{11}$ | 2.032 | 1.998 | 1.988 |
| $10^{12}$ | 1.870 | 1.851 | 1.841 |
| $10^{14}$ | — | 1.7345 | 1.7287 |

Over the two decades most labs can afford, H1 and H2 differ by **0.010 nats** — inside typical seed-to-seed variance for a 1B-parameter run. Both fit the same data. Yet they say opposite things about the frontier: H1 is a fixed data multiplier of $(410.7/369.6)^{1/0.28} = 1.46\times$ at every scale; H2's multiplier grows as $D^{0.054}$, reaching $\approx 2.0\times$ at $10^{14}$ tokens and unbounded thereafter.

Worse, the absolute gap *shrinks* at large $D$ (0.0058 nats at $10^{14}$) because both curves are crushed toward $E = 1.69$. So the discriminating signal lives in the reducible term, whose size depends on an $E$ that must be fit from the same short range. Add the pool constraint — a $\rho = 0.2$ filter on a 20T pool cannot reach $10^{13}$ tokens at $R = 1$ — and the top decade, the only one where H1 and H2 separate, is precisely the decade where repetition decay contaminates the filtered arm.

That is the obstruction in one line: **the regime where the exponent question has an answer is the regime where the filtered pool has run out.**

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*