---
id: 03-training-dynamics/compute-optimal-model-shape
title: "Compute-Optimal Model Shape Beyond Parameter Count"
topic: 03-training-dynamics
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Compute-Optimal Model Shape Beyond Parameter Count

> **Topic:** Training Dynamics & Optimization · **ID:** `03-training-dynamics/compute-optimal-model-shape` · **Status:** empirically-open

## 1. Problem Statement

Chinchilla-style scaling laws answer *how many parameters* and *how many tokens* for a compute budget $C$. They do not answer *what shape those parameters take*. Depth $L$, width $d$, MLP expansion ratio, head count and head dimension, vocabulary size $V$, and MoE granularity are all free once $N$ is fixed. The standard practice is to interpolate an aspect-ratio table from GPT-3 and move on.

- **Measurement variant.** Given fixed $(C, N, D)$, does validation loss depend on shape at all once the learning rate is retuned per shape? Kaplan et al. (2020) said *barely*; nobody has rerun that sweep with modern optimizers, at $\geq 10^{21}$ FLOPs, with per-shape hyperparameter transfer.
- **Method variant.** Produce $\mathrm{shape}^*(C)$ — a function from compute budget to $(L, d, \text{ffw}, V, \ldots)$ — fit from small-scale runs and validated by extrapolation to a held-out larger budget.
- **Theory variant.** Prove a depth–width tradeoff for the loss of a trained transformer on a realistic data distribution, not just an expressivity separation for a fixed function class.

Solving it means: a published $\mathrm{shape}^*(C)$ that beats the standard aspect-ratio heuristic by a stated margin at a budget at least $30\times$ above the largest fitting run.

## 2. Formal Setting

A decoder transformer is specified by $\theta = (L, d, d_{\text{ffw}}, n_h, d_h, V, s)$: layers, model width, MLP hidden width, heads, head dim, vocabulary, context length. Non-embedding parameters:

$$N(\theta) \;=\; L\left(4d^2 + 2 d\, d_{\text{ffw}}\right), \qquad N_{\text{emb}} = V d .$$

**Measured as:** count of trainable tensors excluding embedding and unembedding, reported explicitly — the embedding/non-embedding split is the single largest source of cross-paper disagreement.

Training compute over $D$ tokens, including the attention term Kaplan et al. drop at large $d$:

$$C(\theta, D) \;=\; 6 N D \;+\; 6\, L\, s\, d\, D .$$

**Measured as:** analytic FLOPs, *not* hardware counters. The two differ by the model FLOP utilization $\mu(\theta) \in (0,1)$, giving wall-clock $T = C / (\mu(\theta) \cdot F_{\text{peak}})$. $\mu$ is shape-dependent — this is where the problem bites.

The objective, with $\mathcal{H}$ the optimizer hyperparameters (peak LR, schedule, warmup, weight decay, init scale, batch size):

$$\theta^*(C) \;=\; \arg\min_{\theta, D \,:\, C(\theta,D) = C} \;\; \min_{\mathcal{H}} \; \mathcal{L}_{\text{val}}(\theta, D, \mathcal{H}).$$

The inner $\min_\mathcal{H}$ is what makes the problem expensive and what most published shape comparisons omit.

**Assumptions, and which are violated.**
1. *Loss is a function of $(N, D)$ alone* — the Chinchilla parametrization $\mathcal{L} = E + A N^{-\alpha} + B D^{-\beta}$ has no shape term. Violated: Tay et al. (2022) show shape changes downstream quality at fixed $N$.
2. *Optimal $\mathcal{H}$ transfers across shape* — μP gives width transfer; depth-μP gives depth transfer for residual nets. Violated in practice for joint depth×width×batch-size moves and for anything with a non-standard normalization placement.
3. *FLOPs are the budget* — violated whenever the deployment cost is latency or memory. A depth-79 and a depth-20 model at equal FLOPs are not equal products.
4. *Single epoch, uncapped data* — violated in data-constrained regimes (Muennighoff et al., 2023), where the shape/data tradeoff changes.

## 3. State of the Art

**Established.**
- Chinchilla (Hoffmann et al., NeurIPS 2022): $N \propto C^{0.5}$, $D \propto C^{0.5}$, $D/N \approx 20$. Shape held near-fixed throughout; the law makes no shape claim.
- Kaplan et al. (2020): at fixed non-embedding $N$ around $10^7$–$10^8$, varying aspect ratio $d/L$ over roughly $1$–$2$ orders of magnitude moves loss by only a few percent, with a shallow optimum. This is the load-bearing empirical claim behind current practice, and it was measured two model generations ago.
- Alabdulmohsin et al. (NeurIPS 2023), "Getting ViT in Shape": fits per-dimension scaling exponents for ViT width, depth and MLP separately, then extrapolates. Yields SoViT-400m/14, which matches ViT-g/14 quality at roughly half the parameters. This is the strongest existing instance of a fitted $\mathrm{shape}^*(C)$ — for vision encoders, not autoregressive LMs.

**Claimed but unablated.**
- "Deeper is better for small models" (MobileLLM, Liu et al., ICML 2024) — reported at sub-1B scale on downstream averages, without per-shape LR retuning reported at the same fidelity for each arm.
- "Larger models deserve larger vocabularies" (Tao et al., NeurIPS 2024): predicts Llama-2-70B's optimal $V \approx 216$k versus its actual $32$k. Fit from runs far below 70B; the extrapolation has not been tested by training a 70B model at $V=216$k.

**Benchmark-number-only.** Tay et al. (ICLR 2022) show upstream perplexity and downstream fine-tuned quality rank shapes *differently* — their DeepNarrow T5 matches T5-Base with about 50% of the parameters. The mechanism is not isolated; it is a table of SuperGLUE scores.

## 4. What Is Known

- Depth-1 attention has provable expressivity limits; Levine et al. (NeurIPS 2020) derive a depth-efficiency threshold in which useful depth grows roughly logarithmically in width, predicting saturation of deep-narrow gains — an expressivity result about representable functions, not trained loss.
- μP (Yang et al., NeurIPS 2021) transfers optimal LR across width: tuned at $d=256$, transferred to 6.7B, beating the published 6.7B baseline. Depthwise transfer for residual nets is established separately (Bordelon et al., ICLR 2024).
- MoE shape has a *fitted* law: Ludziejewski et al. (ICML 2024) add expert granularity $G$ as an explicit variable and report the compute-optimal $G$ rising with budget; Clark et al. (ICML 2022) fit routed-LM laws to 900+ models up to ~$10^{10}$ params but find routing gains fading by that scale.
- Discrepancies between Kaplan and Chinchilla exponents ($N \propto C^{0.73}$ vs $C^{0.5}$) were traced by Porian et al. (NeurIPS 2024) to three causes: embedding-parameter counting, LR tuning, and warmup. Two of the three are shape-coupled. This is the clearest evidence that shape effects and tuning effects are entangled in the existing record.

## 5. What Is Not Known

- **Empirically open (primary).** Whether the residual shape effect at fixed $(N,D)$, *after* per-shape hyperparameter tuning, is above 1% of loss at $\geq 10^{22}$ FLOPs. The experiment is runnable today for roughly $10^{22}$–$10^{23}$ FLOPs total; nobody has published it with a proper inner tuning loop.
- **Empirically open.** Whether $\mathrm{shape}^*$ extrapolates. Every fitted shape law is validated within $\sim10\times$ of its fitting range.
- **Methodologically blocked.** The objective itself. "Compute-optimal" mixes FLOPs, wall-clock and inference cost; shape changes the conversion between them by up to $2\times$. Until the page's currency is fixed, two correct papers can give opposite answers.
- **Theoretically open.** No lower bound on trained loss as a function of $(L,d)$ at fixed $Ld^2$ for any realistic data distribution. Expressivity separations exist; optimization-and-generalization ones do not.

## 6. Why It Is Hard

**Non-identifiability between shape and hyperparameters.** The measured quantity is $\min_\mathcal{H} \mathcal{L}$, but every published shape sweep reports $\mathcal{L}$ at *some* $\mathcal{H}$. Deep-narrow models want smaller LR and longer warmup; if you sweep shape at a fixed LR tuned for the wide arm, you measure LR sensitivity and label it depth. Porian et al. show this exact mechanism accounts for a chunk of the Kaplan–Chinchilla gap. Removing it multiplies the experiment cost by the size of the $\mathcal{H}$ grid — the reason the sweep does not exist at scale.

**Compounding this:** the effect size is small (few percent of loss) while run-to-run seed variance at fixed shape is of the same order, so each cell needs replicates; and the analytic-FLOP budget is not the real budget, since $\mu(\theta)$ falls with depth at fixed FLOPs on any pipeline- or tensor-parallel layout.

## 7. Current Research (as of 2026)

- **Shape-explicit scaling laws.** Extending the Chinchilla form with per-axis exponents, following the ViT recipe, to decoder LMs. Google DeepMind (Alabdulmohsin, Zhai, Beyer lineage) is the group with the method. *(frontier — verify)*
- **Granularity laws for MoE.** IDEAS NCBR / University of Warsaw (Ludziejewski, Krutul, Jaszczur), DeepSeek, and the OLMoE line at AI2 are all fitting expert-count/expert-size tradeoffs. Shape here has more degrees of freedom and larger effects than in dense models.
- **Depth-μP and infinite-depth limits.** Hayou, Yang, Bordelon, Pehlevan — making the inner $\min_\mathcal{H}$ cheap enough that a clean shape sweep becomes affordable. This is the enabling technology for the experiment in §8.
- **Inference-aware shape.** Fitting $\mathrm{shape}^*$ against a serving-cost objective rather than training FLOPs. *(frontier — verify)*

## 8. Concrete Next Experiment

**Scale.** Fix $N = 1.0\times10^9$ non-embedding parameters, $D = 20N = 2\times10^{10}$ tokens, $s = 4096$, $V = 32{,}768$ held constant. That is $C \approx 1.3\times10^{20}$ FLOPs per arm — under 400 H100-hours each.

**Arms.** Four aspect ratios $d/L \in \{13, 51, 102, 819\}$, realized as $(d, L) = (1024, 79), (2048, 20)$ — plus $(1448, 40)$ and $(4096, 5)$ — with $d_{\text{ffw}} = 4d$ and $d_h = 128$ fixed. Each arm gets an independent 5-point peak-LR sweep seeded by μP/depth-μP transfer from a $d=256$ proxy, plus 3 seeds at the winning LR. Total: 4 shapes × (5 LR + 2 extra seeds) = 28 runs, ~$3.7\times10^{21}$ FLOPs.

**Control arm.** $(d,L) = (2048, 20)$, aspect ratio $102$ — the GPT-3-family heuristic — at its own tuned LR. All comparisons are against this.

**Decision number.** $\Delta \mathcal{L} = \mathcal{L}_{\text{control}} - \min_{\text{shapes}} \mathcal{L}$, in nats, with a seed-variance error bar. Using the local Chinchilla slope, $0.01$ nats $\approx$ 10% of compute at this budget. **If $\Delta\mathcal{L} < 0.01$ nats with $2\sigma$ bars excluding $0.01$, shape is confirmed second-order under FLOPs and the field should stop tuning it; if $\Delta\mathcal{L} > 0.03$ nats, every published Chinchilla-optimal model is leaving a measurable fraction of its budget on the table.** Report the same $\Delta\mathcal{L}$ a second time under a wall-clock-matched budget to expose the currency problem.

## 9. Key References

- **[Foundational]** Kaplan, McCandlish, Henighan, Brown, et al. *Scaling Laws for Neural Language Models.* Preprint, 2020. — arXiv:2001.08361
- **[Foundational]** Hoffmann, Borgeaud, Mensch, et al. *Training Compute-Optimal Large Language Models.* NeurIPS, 2022. — arXiv:2203.15556
- **[SOTA]** Alabdulmohsin, Zhai, Kolesnikov, Beyer. *Getting ViT in Shape: Scaling Laws for Compute-Optimal Model Design.* NeurIPS, 2023.
- **[SOTA]** Tay, Dehghani, Rao, et al. *Scale Efficiently: Insights from Pre-training and Fine-tuning Transformers.* ICLR, 2022. — arXiv:2109.10686
- **[SOTA]** Ludziejewski, Krajewski, Adamczewski, et al. *Scaling Laws for Fine-Grained Mixture of Experts.* ICML, 2024.
- **[Method]** Yang, Hu, Babuschkin, et al. *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer.* NeurIPS, 2021. — arXiv:2203.03466
- **[Method]** Bordelon, Noci, Li, Hanin, Pehlevan. *Depthwise Hyperparameter Transfer in Residual Networks.* ICLR, 2024.
- **[Theory]** Levine, Wies, Sharir, Bata, Shashua. *Limits to Depth Efficiencies of Self-Attention.* NeurIPS, 2020.
- **[Diagnostic]** Porian, Wortsman, Jitsev, Schmidt, Carmon. *Resolving Discrepancies in Compute-Optimal Scaling of Language Models.* NeurIPS, 2024.
- **[Related]** Tao, Liu, Kocetkov, et al. *Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies.* NeurIPS, 2024.
- **[Related]** Liu, Chang, Wu, et al. *MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases.* ICML, 2024. — arXiv:2402.14905
- **[Survey]** Clark, de las Casas, Guy, et al. *Unified Scaling Laws for Routed Language Models.* ICML, 2022. — arXiv:2202.01169

## 10. Worked Example

Take the §8 arms and check whether they are actually FLOP-matched. With $d_{\text{ffw}} = 4d$, $N = 12 L d^2$:

| $d$ | $L$ | $N$ | attention/param FLOP ratio $= s/(6d)$ | $C$ rel. to control |
|---|---|---|---|---|
| 1024 | 79 | $0.99\times10^9$ | 0.67 | **+41%** |
| 1448 | 40 | $1.01\times10^9$ | 0.47 | +21% |
| 2048 | 20 | $1.01\times10^9$ | 0.33 | 1.00 (control) |
| 4096 | 5 | $1.01\times10^9$ | 0.17 | $-12\%$ |

The four arms have equal $N$ to within 2% and equal $6ND$, but true training FLOPs at $s=4096$ span a $1.6\times$ range, because the $6Lsd D$ attention term scales as $L d = N/(12d)$ and therefore grows as the model gets narrower. A naive "equal $N$, equal tokens" shape sweep is a $1.6\times$ compute-unequal comparison, biased *toward* deep-narrow arms. At the local Chinchilla slope, a 41% compute advantage is worth roughly $0.03$–$0.04$ nats — three to four times the decision threshold.

Now the second half. Measure wall-clock instead: at fixed FLOPs, the $L=79$ arm runs about 79 sequential residual blocks per step versus 5 for the $L=5$ arm, so its per-step latency at matched batch and its parallel-layout MFU are both worse; observed $\mu$ typically drops from ~0.5 to ~0.35 across this depth range on standard 3D-parallel setups. Converting to wall-clock therefore *reverses* part of the bias the FLOP accounting introduced.

The obstruction is visible in these two paragraphs: the same four models rank differently under $6ND$, under full $C(\theta,D)$, and under wall-clock, and the spread between those rankings ($\sim$0.04 nats) exceeds the effect the experiment is trying to detect ($\sim$0.01–0.03 nats). Any shape law that does not state its currency and include the attention FLOP term is reporting an accounting artifact.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*