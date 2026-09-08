---
id: 06-data-pipeline/curation-versus-training-compute-split
title: "Compute-Optimal Split Between Data Curation and Training"
topic: 06-data-pipeline
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Compute-Optimal Split Between Data Curation and Training

> **Topic:** Data Pipelines & Curation · **ID:** `06-data-pipeline/curation-versus-training-compute-split` · **Status:** empirically-open

## 1. Problem Statement

Given one fixed compute budget that must pay for **both** selecting the training data and training on it, what fraction should go to curation?

- **Input:** a total budget $C$ (FLOPs, or dollars/GPU-hours), a raw token pool $\mathcal{P}$ of size $M$, a family of curation operators (dedup, classifier filtering, LLM annotation, clustering, synthetic rewriting), and a model family.
- **Output:** a split $(C_{\text{cur}}, C_{\text{tr}})$ with $C_{\text{cur}}+C_{\text{tr}}\le C$, plus the induced $(N, D)$ and filter aggressiveness.
- **Objective:** minimise end-task loss at fixed $C$.

Three variants, routinely conflated:

- **Measurement:** what is the exchange rate — how many training FLOPs does one curation FLOP buy back? Blocked mainly by the fact that almost nobody reports $C_{\text{cur}}$.
- **Method:** an allocation rule that is cheap to fit and transfers across scales, in the spirit of Chinchilla's $D \approx 20N$.
- **Theory:** conditions under which curation moves the *exponent* of the scaling law rather than its constant. If curation only shifts the constant, its optimal share $\to 0$ as $C\to\infty$.

Solved means: a published rule predicting the optimal split within a stated tolerance, validated by held-out runs at a scale at least $10\times$ above the fit.

## 2. Formal Setting

Budget accounting, all in FLOPs, measured as they would actually be billed:

$$C = \underbrace{C_{\text{score}} + C_{\text{annot}} + C_{\text{synth}} + C_{\text{dedup}}}_{C_{\text{cur}}} + \underbrace{6ND}_{C_{\text{tr}}}$$

- $C_{\text{score}} = 2 N_s M$ — a scorer of $N_s$ params run over all $M$ pool tokens (forward pass $\approx 2$ FLOPs/param/token).
- $C_{\text{annot}} = 2 N_a m \bar{\ell}$ — an annotator LLM of $N_a$ params labelling $m$ documents of mean length $\bar{\ell}$ to train the scorer.
- $C_{\text{synth}} = 2 N_g D_{\text{gen}}$ — generated tokens cost a forward pass per token from a generator of $N_g$ params.
- $C_{\text{dedup}}$ — MinHash/embedding dedup, not FLOP-dominated; measured in CPU-hours and converted at the cluster's realised FLOPs/dollar. This conversion is the weakest link in the accounting.
- $6ND$ — standard training estimate (Kaplan et al., 2020).

Curation is a keep-rate $\rho\in(0,1]$ applied to $\mathcal{P}$, yielding $D_{\text{avail}} = \rho M$. If $D > D_{\text{avail}}$, tokens repeat; model epoch count $R = D/D_{\text{avail}}$ with the repeat-decay of Muennighoff et al. (2023).

Loss model, per Hoffmann et al. (2022), made curation-dependent:

$$L(N, D, \rho) = E(\rho) + \frac{A(\rho)}{N^{\alpha}} + \frac{B(\rho)}{D_{\text{eff}}^{\beta}}, \qquad D_{\text{eff}} = D_{\text{avail}}\big(1 + (R-1)e^{-(R-1)/R^\star}\big)$$

with Chinchilla's fitted $\alpha=0.34$, $\beta=0.28$. Under a pure compute constraint and $\rho$ fixed, reducible loss falls as $C_{\text{tr}}^{-\gamma}$ with

$$\gamma = \frac{\alpha\beta}{\alpha+\beta} = \frac{0.34 \times 0.28}{0.62} \approx 0.154 .$$

The decision predicate: spend a marginal FLOP on curation iff $-\partial L/\partial C_{\text{cur}} > \gamma L_{\text{red}}/C_{\text{tr}}$.

**Assumptions, and which are violated.**
1. *Curation changes $A,B,E$ but not $\alpha,\beta$.* Contradicted by Sorscher et al. (2022), who report exponent-breaking pruning in a solvable perceptron setting.
2. *$\rho$ is scale-independent.* Contradicted by Goyal et al. (2024): optimal filter aggressiveness falls as budget rises.
3. *Curation cost is a one-off amortised over all downstream runs.* True for a lab shipping one corpus, false for the single-run comparison the scaling law describes. Which convention is used changes the answer by an order of magnitude and is usually left unstated.
4. *Loss is the target.* End tasks (MMLU, code) respond to curation far more than perplexity does; perplexity can move the wrong way while MMLU improves.

## 3. State of the Art

**Established.**
- Chinchilla (Hoffmann et al., NeurIPS 2022): compute-optimal $D \approx 20N$, verified by the 70B/1.4T Chinchilla beating 280B Gopher at equal compute. Data is treated as free.
- Data-constrained scaling (Muennighoff et al., NeurIPS 2023): up to ~4 epochs of repeated data are worth roughly fresh data; value decays to near zero by ~16 epochs. Fit over 400+ runs up to 9B params / 900B tokens.
- DataComp (Gadre et al., NeurIPS 2023 D&B): a controlled benchmark where the model and budget are fixed and only the filter varies. Best CLIP filters beat unfiltered LAION-style pools by tens of ImageNet accuracy points at fixed compute.

**Claimed but unablated with respect to curation cost.**
- DCLM-Baseline (Li et al., NeurIPS 2024 D&B): 7B params / 2.6T tokens reaching 64% 5-shot MMLU, +6.6 pp over MAP-Neo with 40% less *training* compute. The classifier-scoring pass over the ~240T-token Common Crawl extraction is not in the denominator.
- FineWeb-Edu (Penedo et al., NeurIPS 2024 D&B): a classifier distilled from ~500k Llama-3-70B annotations lifts MMLU/ARC substantially at 1.8B scale. Annotation and inference FLOPs are described but never folded into a compute-matched comparison.
- Synthetic "textbook" data (Gunasekar et al., 2023, phi-1): 1.3B params, ~7B tokens, 50.6% HumanEval. Generation used GPT-3.5 at a cost never reported; the headline is a benchmark number, not a compute-matched result.

**Theory SOTA.** Sorscher et al. (NeurIPS 2022) give the only clean statement that pruning can convert power-law to exponential scaling — in a student-teacher perceptron with an oracle difficulty score. No transfer to transformers is proved.

## 4. What Is Known

- $\gamma \approx 0.154$: diverting 20% of a budget from training costs about $0.8^{-0.154}-1 = 3.5\%$ of reducible loss. The bar curation must clear is small. (Chinchilla fits, 70M–16B params.)
- Filter utility decays with repetition. Goyal et al. (CVPR 2024) show on LAION/DataComp pools that a filter optimal at a 32M-sample budget is *harmful* at 640M samples; the compute-optimal keep-rate rises with budget.
- Pruning gains are non-monotone in pool quality. Sorscher et al.: on ImageNet (ResNet-50) keeping the hardest examples helps when data is abundant and hurts when scarce; the correct pruning direction flips.
- Perplexity-selection and benchmark-selection disagree. Marion et al. (2023) find mid-perplexity selection beats both tails at 1B–7B scale; DSIR (Xie et al., NeurIPS 2023) shows n-gram importance resampling at ~10⁻⁴ of the cost of neural scoring recovers much of the gain.
- Scoring cost is not negligible. A 110M-param classifier over a 1T-token pool costs $2\times1.1{\times}10^{8}\times10^{12} = 2.2\times10^{20}$ FLOPs — comparable to fully training a 1B model on 35B tokens.

## 5. What Is Not Known

- **Empirically open (the core gap).** No published study sweeps $C_{\text{cur}}/C$ at fixed total $C$ with everything else held constant. The sweep is runnable today — perhaps 20–40 runs at ≤1B params — and nobody has run it at a scale where the extrapolation is credible.
- **Theoretically open.** Whether curation can change $\alpha$ or $\beta$ for autoregressive transformers on natural text, or only $A,B,E$. This determines whether the optimal curation share is a constant or decays as $C^{-\delta}$.
- **Methodologically blocked.** There is no agreed accounting convention for $C_{\text{cur}}$: whether to amortise the annotator LLM's own pretraining, how to price CPU-hours of dedup in FLOPs, and whether a filter reused across ten runs is charged once or ten times. Without a convention, two labs' numbers are not comparable.
- **Empirically open.** Whether the optimal split transfers across modality and across the raw-pool quality axis (Common Crawl vs curated code vs licensed text).

## 6. Why It Is Hard

The obstruction is **confounded measurement compounded by a denominator nobody reports**. Every published curation win varies at least three things at once — keep-rate, token count, and the deduplication/mixture recipe — and reports only training FLOPs. A "2× compute efficiency" claim where the scoring pass cost 25% of the training budget is really ~1.6×, but the reader cannot compute this because $M$, $N_s$ and the annotation volume are usually absent.

Second obstruction: **the decisive experiment is scale-fragile**. The gap to be resolved is a few percent of reducible loss (§4). At 1B params, run-to-run seed variance on MMLU is of comparable size, so the sweep must be replicated — multiplying its cost — and its extrapolation to 100B+ is exactly the regime where assumption 2 (scale-independent $\rho$) is known to fail.

## 7. Current Research (as of 2026)

- **Compute-aware filtering laws.** Goyal, Ramanujan and collaborators (CMU/AI2) extend filtering scaling laws to the repeat-aware regime. Direct descendant of the CVPR 2024 result.
- **Cheap proxies for expensive scorers.** DSIR, Ask-LLM/Density sampling (Sachdeva et al., 2024), and small-model perplexity signals aim to cut $C_{\text{score}}$ by 2–4 orders of magnitude at small quality loss.
- **Curation as an amortised asset.** Labs increasingly treat corpora as capital reused across model generations, making the amortised split the operative question; no public accounting standard exists. *(frontier — verify)*
- **Synthetic-data budget allocation.** How many generator FLOPs per training FLOP, and whether generated tokens obey the same repeat-decay. Actively studied; no compute-matched public result. *(frontier — verify)*

## 8. Concrete Next Experiment

**Scale.** Fix $C = 10^{21}$ FLOPs total. Fix the pool: 1T tokens of raw Common Crawl text. Fix one curation operator: a 110M-param quality classifier applied at keep-rate $\rho$.

**Arms.** Six splits, $C_{\text{cur}}/C \in \{0, 0.02, 0.05, 0.11, 0.22, 0.44\}$, realised by scoring a $\rho$-dependent subsample of the pool. Each arm re-derives its own Chinchilla-optimal $(N,D)$ from its residual $C_{\text{tr}}$, so $N$ ranges ~2.9B down to ~2.2B. Three seeds per arm, 18 runs.

**Control arm.** $C_{\text{cur}}=0$: train the Chinchilla-optimal $N=2.9\times10^{9}$, $D=5.8\times10^{10}$ model on unfiltered pool tokens, using the full $10^{21}$ FLOPs.

**Deciding number.** The argmin over arms of mean validation loss on a fixed held-out mixture, reported with its seed standard error. If the argmin is at $C_{\text{cur}}/C = 0$ or the curve is flat within $\pm 2\sigma$, curation-under-budget is a wash at this scale; if the argmin is interior, its location *is* the first measured split. Secondary readout: MMLU delta, to expose loss/benchmark disagreement.

Then repeat the whole sweep at $C=10^{22}$ (one seed) to test whether the argmin moves — the single fact that decides between "constant share" and "share $\to 0$".

## 9. Key References

- **[Foundational]** J. Hoffmann, S. Borgeaud, A. Mensch, et al. *Training Compute-Optimal Large Language Models.* NeurIPS 2022. — arXiv:2203.15556
- **[Foundational]** J. Kaplan, S. McCandlish, T. Henighan, et al. *Scaling Laws for Neural Language Models.* 2020. — arXiv:2001.08361
- **[Foundational]** B. Sorscher, R. Geirhos, S. Shekhar, S. Ganguli, A. Morcos. *Beyond neural scaling laws: beating power law scaling via data pruning.* NeurIPS 2022. — arXiv:2206.14486
- **[SOTA]** S. Goyal, P. Maini, Z. Lipton, A. Raghunathan, J. Z. Kolter. *Scaling Laws for Data Filtering — Data Curation cannot be Compute Agnostic.* CVPR 2024. — arXiv:2404.07177
- **[SOTA]** N. Muennighoff, A. M. Rush, B. Barak, et al. *Scaling Data-Constrained Language Models.* NeurIPS 2023. — arXiv:2305.16264
- **[SOTA]** J. Li, A. Fang, G. Smyrnis, et al. *DataComp-LM: In search of the next generation of training sets for language models.* NeurIPS 2024 Datasets & Benchmarks. — arXiv:2406.11794
- **[SOTA]** G. Penedo, H. Kydlíček, L. von Werra, T. Wolf, et al. *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale.* NeurIPS 2024 Datasets & Benchmarks. — arXiv:2406.17557
- **[SOTA]** S. Y. Gadre, G. Ilharco, A. Fang, et al. *DataComp: In search of the next generation of multimodal datasets.* NeurIPS 2023 Datasets & Benchmarks. — arXiv:2304.14108
- **[SOTA]** S. M. Xie, S. Santurkar, T. Ma, P. Liang. *Data Selection for Language Models via Importance Resampling.* NeurIPS 2023. — arXiv:2302.03169
- **[Related]** M. Marion, A. Üstün, L. Pozzobon, A. Wang, M. Fadaee, S. Hooker. *When Less is More: Investigating Data Pruning for Pretraining LLMs at Scale.* 2023. — arXiv:2309.04564
- **[Related]** N. Sardana, J. Frankle, et al. *Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws.* ICML 2024. — arXiv:2401.00448
- **[Survey]** A. Albalak, Y. Elazar, S. M. Xie, et al. *A Survey on Data Selection for Language Model Pretraining.* 2024. — arXiv:2402.16827

## 10. Worked Example

Budget $C = 10^{21}$ FLOPs, pool $M = 10^{12}$ tokens.

*Control.* Chinchilla: $C = 6ND$ with $D=20N$ gives $N = \sqrt{C/120} = 2.9\times10^{9}$, $D = 5.8\times10^{10}$. All $10^{21}$ FLOPs train.

*Filtered arm.* Score the whole pool with a 110M classifier:
$$C_{\text{score}} = 2 \times 1.1\times10^{8} \times 10^{12} = 2.2\times10^{20}\ \text{FLOPs} = 22\%\ \text{of}\ C .$$
Residual $C_{\text{tr}} = 7.8\times10^{20}$, so $N = 2.55\times10^{9}$, $D = 5.1\times10^{10}$. Cost of the diversion in reducible loss:
$$\Delta = 0.78^{-0.154} - 1 = e^{0.154 \times 0.248} - 1 = 3.9\% .$$
So filtering must buy back **more than 3.9% of reducible loss** to pay for itself. At $\rho=0.1$ the filtered corpus holds $10^{11}$ tokens and $D=5.1\times10^{10}$ fits in under one epoch — no repeat penalty. Clearing 3.9% is plausible; DCLM-scale filtering gains look far larger than that.

*Where it breaks.* Now use an LLM-based scorer instead — a 7B model over the same pool: $C_{\text{score}} = 2\times7\times10^{9}\times10^{12} = 1.4\times10^{22}$, **14× the entire budget**. The same filter that is a bargain at 110M is impossible at 7B, at this budget. And the crossover moves with $C$: at $C=10^{24}$ the 7B scorer costs only 1.4% of budget while the training-side bar it must clear is unchanged at $\gamma=0.154$.

The obstruction is visible in one line: the published DCLM and FineWeb-Edu wins are reported *without* $C_{\text{score}}$, so from the papers alone one cannot tell whether their filters sit on the profitable or the ruinous side of this crossover — and the crossover is where the whole problem lives.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*