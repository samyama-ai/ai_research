---
id: 10-scaling-laws/emergence-metric-artifact
title: "Emergence as Metric Artifact"
topic: 10-scaling-laws
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Emergence as Metric Artifact

> **Topic:** Scaling Laws & Compute Allocation · **ID:** `10-scaling-laws/emergence-metric-artifact` · **Status:** partially-solved

## 1. Problem Statement

"Emergent abilities" are capabilities reported as absent in smaller models and present in larger ones, with a sharp transition rather than a smooth improvement (Wei et al., TMLR 2022). The competing claim is that the sharpness lives in the *scoring function*, not in the model: a discontinuous or superlinear metric (exact string match, multiple-choice grade) applied to a smoothly improving latent quantity manufactures a cliff (Schaeffer et al., NeurIPS 2023).

Three variants, different difficulty:

- **Measurement.** Given a task and a model ladder, decide whether the sharp curve survives replacing the metric with a continuous surrogate of the same capability. *Largely settled for the BIG-Bench cases; unsettled in general because "surrogate of the same capability" has no agreed definition.*
- **Method.** Produce, before training the large model, a calibrated prediction of its downstream score on the metric people actually deploy against. *Open.*
- **Theory.** Determine whether any true non-analyticity — a discontinuity in a smooth functional of model weights as compute grows — exists in the scaling limit, or whether all observed sharpness is finite-ladder resolution plus metric nonlinearity. *Theoretically open.*

Solving it means: a rule that, given task, metric, and ladder, says in advance whether the curve will be extrapolable, and a prediction interval that holds out-of-sample.

## 2. Formal Setting

A model family indexed by compute $C$ (FLOPs), trained to a compute-optimal allocation (Hoffmann et al., 2022). For a task with example distribution $\mathcal{D}$, the model emits $\hat{y} = f_C(x)$.

**Measured quantities.**

- Pretraining loss, measured on a held-out corpus: $L(C) = -\frac{1}{N}\sum_{i=1}^{N}\log p_{f_C}(t_i \mid t_{<i})$, in nats/token.
- Per-token correctness on the target task: $p(C) = \Pr[\hat{y}_j = y_j]$, estimated by teacher-forced token accuracy over $N \cdot L$ tokens.
- Exact match on an $L$-token answer: $\mathrm{EM}(C) = \mathbb{E}_{\mathcal{D}}\big[\prod_{j=1}^{L}\mathbb{1}[\hat{y}_j = y_j]\big]$. Under token independence, $\mathrm{EM} \approx p^{L}$.
- A continuous surrogate: token edit distance $\mathrm{ED}(C)$, or the Brier score $\mathbb{E}[(\Pr[y] - 1)^2 + \sum_{y'\neq y}\Pr[y']^2]$ over answer options.
- Sampling-resolution-free score (PassUntil, Hu et al., ICLR 2024): $\log \pi(C)$, where $\pi$ is the per-sample pass probability estimated from $K$ samples, resolving down to $\pi \sim 10^{-5}$ at $K \sim 10^5$.

**Emergence predicate.** Do not define it as "a kink." Define it operationally as a *predictability gap*. Fit $\log M(C) = a \log C + b$ (or a saturating form) on $C \le C_0$; predict $\hat{M}(C_1)$ for $C_1 = 10 C_0$. Then

$$R(M, C_0) \;=\; \frac{\big|\,M(C_1) - \hat{M}(C_1)\,\big|}{M(C_1)} .$$

A task/metric pair is *emergent at $C_0$* if $R > \tau$ (e.g. $\tau = 0.25$). The mirage hypothesis is the claim that $R(\mathrm{EM}) \gg \tau$ while $R(\mathrm{ED}) < \tau$ on the same ladder.

**Assumptions, and where they break.**

- *Token independence* in $\mathrm{EM} \approx p^L$ — violated: errors are correlated within an answer, so $p^L$ understates $\mathrm{EM}$.
- *Single scaling axis* — violated: published "emergence" plots mix parameter count, data, tokenizer, and data mixture across families; $C$ is not a sufficient statistic.
- *A monotone functional relation from $L$ to task score* — violated across data mixtures; the same loss gives different downstream scores under different corpora.
- *Fixed prompt and decoding* — violated: prompt format, few-shot count, and chain-of-thought move the transition point by $\sim$1 order of magnitude in $C$.

## 3. State of the Art

**Established.** Schaeffer, Miranda and Koyejo (NeurIPS 2023, best paper) showed that in BIG-Bench, emergent behaviour is concentrated in discontinuous metrics: over 92% of reported emergent task–model instances used Multiple Choice Grade or Exact String Match. They also produced the constructive converse — inducing apparent emergence in a convolutional autoencoder on MNIST purely by choosing a nonlinear metric. Both directions are ablated and reproduced.

**Established, opposing.** Du et al. (2024) evaluated 1.5B–32B models on 12 English and Chinese benchmarks using *continuous* metrics (including Brier score) and still found a threshold: performance stays near chance until pretraining loss drops below roughly 2.2 nats/token, then rises. This is emergence with respect to loss, not with respect to a discontinuous metric — so the mirage explanation does not cover it.

**Claimed but unablated.** That emergent abilities are reducible to in-context learning plus instruction tuning (Lu et al., ACL 2024) — tested on a specific task set, not on the arithmetic/GSM8K cases that anchor the original claim. That the quantization model (Michaud et al., NeurIPS 2023) explains real emergence — the discrete-quanta mechanism is demonstrated on toy and small-LM settings, not shown to generate the specific BIG-Bench curves.

**Benchmark-number-only.** GPT-4's report of HumanEval pass rate predicted from models trained with $10^{4}\times$ less compute (OpenAI, 2023) is a single unreproducible point: no ablation, no released ladder, mean-log-pass-rate metric chosen post hoc.

**Systems SOTA for prediction.** Observational scaling laws (Ruan et al., NeurIPS 2024) fit a low-dimensional capability space over ~100 public models and predict downstream metrics, including some previously called emergent, with $R^2 > 0.9$ — but this is interpolation across an existing model population, not forward prediction of an untrained model.

## 4. What Is Known

- Sharpness is metric-dependent. Schaeffer et al.: on GPT-3 3-digit arithmetic, accuracy jumps from near 0 to $>0.7$ across two model sizes while per-token edit distance falls smoothly and monotonically over the same range.
- Resolution is a confound. Hu et al. (ICLR 2024) with $\sim10^{5}$ samples per problem resolved pass rates near $10^{-5}$ on code generation and predicted a 2.4B model's performance from models $\sim100\times$ smaller with fitting error under 0.05% — but reported at least one task showing genuine "accelerated emergence" not fit by the standard power law.
- Loss thresholds survive continuous metrics. Du et al.: ~2.2 nats/token onset, models 1.5B–32B, 12 benchmarks.
- Some transitions are real and mechanistic. Induction-head formation in 2–13 layer transformers produces a visible bump in the *loss derivative* during training (Olsson et al., 2022) — a sharp change in a continuous, non-thresholded quantity.
- Memorization is partly predictable: Biderman et al. (NeurIPS 2023) predicted which sequences a Pythia-12B model memorizes from smaller runs, but with low recall at usable precision.
- Discontinuous metrics remain the deployed default. BIG-Bench (Srivastava et al., TMLR 2023) ships 204 tasks whose canonical scoring is dominated by exact match and multiple-choice grade.

## 5. What Is Not Known

- **Theoretically open.** Whether any true non-analyticity in $C \mapsto \mathbb{E}[\text{smooth score}]$ exists for transformers, or whether all sharpness is finite-resolution artifact. No proof either way. The quantization model supplies a mechanism for discreteness but no theorem excluding smooth alternatives.
- **Empirically open.** A single controlled ladder — one architecture, one corpus, one tokenizer, $\geq 8$ scales — evaluated under matched discontinuous and continuous metrics, with prediction residuals reported. Runnable at $<10^{22}$ FLOPs. Not run publicly.
- **Methodologically blocked.** "Continuous surrogate of the same capability" is undefined. Edit distance rewards partial credit that no downstream user gets; a system that returns 8 of 9 correct digits of a bank transfer is worth zero. There is no criterion for when a smooth proxy is measuring the capability versus measuring a correlate of it.

## 6. Why It Is Hard

The specific obstruction is **an evaluation that does not measure what it names, compounded by non-identifiability**. The deployed metric (exact match) is the one that matters economically and is the one that is unpredictable; the predictable metric (edit distance, log pass rate) is not the deployed one. Choosing between "sharpness is in the metric" and "sharpness is in the model" requires distinguishing $\phi(g(C))$ from $g'(C)$ where only the composition is observed — the decomposition is not identified without an independent handle on $\phi$, and none exists. Compute is *not* the binding constraint: the deciding ladder costs under $10^{22}$ FLOPs. Confounded measurement is: published emergence plots pool model families that differ in data, tokenizer and instruction tuning.

## 7. Current Research (as of 2026)

- Predictability-first evaluation: continuous, high-resolution scoring (PassUntil-style massive sampling; per-token log-likelihood scoring of benchmark options) as the default for frontier-model reporting — Tsinghua/OpenBMB, Stanford (Koyejo group), EleutherAI.
- Loss-as-the-x-axis: replacing compute with pretraining loss or a latent capability index (Ruan et al.; Du et al.) so cross-family curves become comparable. *(frontier — verify: adoption in frontier-lab system cards.)*
- Mechanistic accounts of genuine phase changes: circuit formation, induction heads, quanta acquisition — Anthropic interpretability, MIT (Tegmark group).
- Fine-tuning-based early warning: measuring how much task-specific fine-tuning a small model needs to reach a threshold, and extrapolating the "emergence point" (Snell et al., 2024). *(frontier — verify.)*
- Survey coverage: Berti, Giorgi and Kasneci, *Emergent Abilities in Large Language Models: A Survey* (2025).

## 8. Concrete Next Experiment

**Scale.** Train 8 decoder-only models, 20M → 3B non-embedding parameters, spaced $\sim0.36$ dex, Chinchilla-optimal ($\approx 20$ tokens/parameter), identical corpus, identical data order prefix, identical tokenizer. Total $\approx 1.5\times10^{21}$ FLOPs — a few thousand H100-hours.

**Task suite.** 12 tasks: 4 previously flagged emergent (3-digit multiplication, modular arithmetic, IPA transliteration, word unscrambling), 4 flagged smooth, 4 multiple-choice. Each scored four ways: exact match; token edit distance; Brier score over options; log pass rate at $K=10^{4}$ samples.

**Control arm.** The same four metrics on the *same* checkpoints for the smooth tasks — this controls for ladder resolution rather than metric shape. Second control: a random-init ladder with the metrics applied to a synthetic $p(C)$ known to be a clean power law, verifying the fitting pipeline recovers $R \approx 0$.

**Deciding number.** Fit each metric's curve on the smallest 6 models, predict models 7 and 8, and report the median relative extrapolation error $R$ per metric. **The question is decided by $R(\mathrm{EM}) / R(\text{best continuous metric})$ on the four emergent tasks.** If that ratio exceeds 5 with $R(\text{continuous}) < 0.25$, emergence there is a metric artifact. If the ratio is below 2 — both metrics fail to extrapolate — the sharpness is in the model or the ladder, and the mirage account is insufficient.

## 9. Key References

- **[Foundational]** Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, et al. *Emergent Abilities of Large Language Models.* TMLR, 2022. — arXiv:2206.07682
- **[SOTA]** Rylan Schaeffer, Brando Miranda, Sanmi Koyejo. *Are Emergent Abilities of Large Language Models a Mirage?* NeurIPS, 2023 (Outstanding Paper). — arXiv:2304.15004
- **[SOTA]** Shengding Hu, Xin Liu, Xu Han, Xinrong Zhang, Chaoqun He, Weilin Zhao, et al. *Predicting Emergent Abilities with Infinite Resolution Evaluation.* ICLR, 2024. — arXiv:2310.03262
- **[SOTA]** Zhengxiao Du, Aohan Zeng, Yuxiao Dong, Jie Tang. *Understanding Emergent Abilities of Language Models from the Loss Perspective.* 2024. — arXiv:2403.15796
- **[SOTA]** Yangjun Ruan, Chris J. Maddison, Tatsunori Hashimoto. *Observational Scaling Laws and the Predictability of Language Model Performance.* NeurIPS, 2024. — arXiv:2405.10938
- **[Theory]** Eric J. Michaud, Ziming Liu, Uzay Girit, Max Tegmark. *The Quantization Model of Neural Scaling.* NeurIPS, 2023. — arXiv:2303.13506
- **[Mechanism]** Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, et al. *In-context Learning and Induction Heads.* Transformer Circuits Thread, 2022. — arXiv:2209.11895
- **[Benchmark]** Aarohi Srivastava et al. *Beyond the Imitation Game: Quantifying and Extrapolating the Capabilities of Language Models.* TMLR, 2023. — arXiv:2206.04615
- **[Context]** Deep Ganguli, Danny Hernandez, Liane Lovitt, et al. *Predictability and Surprise in Large Generative Models.* ACM FAccT, 2022. — arXiv:2202.07785
- **[Related]** Stella Biderman, USVSN Sai Prashanth, Lintang Sutawika, et al. *Emergent and Predictable Memorization in Large Language Models.* NeurIPS, 2023. — arXiv:2304.11158
- **[Related]** Sheng Lu, Irina Bigoulaeva, Rachneet Sachdeva, Harish Tayyar Madabushi, Iryna Gurevych. *Are Emergent Abilities in Large Language Models just In-Context Learning?* ACL, 2024.
- **[Related]** Rylan Schaeffer, Hailey Schoelkopf, Brando Miranda, Gabriel Mukobi, Varun Madan, Adam Ibrahim, Herbie Bradley, Stella Biderman, Sanmi Koyejo. *Why Has Predicting Downstream Capabilities of Frontier AI Models with Scale Remained Elusive?* 2024.
- **[Survey]** Leonardo Berti, Flavio Giorgi, Gjergji Kasneci. *Emergent Abilities in Large Language Models: A Survey.* 2025.
- **[Baseline]** Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, et al. *Training Compute-Optimal Large Language Models.* NeurIPS, 2022. — arXiv:2203.15556

## 10. Worked Example

Three-digit multiplication, 6-token answers, four ladder rungs at $C, 10C, 10^2C, 10^3C$.

| Rung | per-token acc $p$ | per-token error $\varepsilon = 1-p$ | $\mathrm{EM} \approx p^{6}$ |
|---|---|---|---|
| 1 | 0.60 | 0.40 | 0.047 |
| 2 | 0.80 | 0.20 | 0.262 |
| 3 | 0.93 | 0.07 | 0.647 |
| 4 | 0.99 | 0.01 | 0.941 |

**Step 1 — the artifact.** Between rungs 1 and 3, $p$ rises $1.55\times$; $\mathrm{EM}$ rises $13.8\times$. Plotted against $\log C$, $\mathrm{EM}$ looks like a cliff and $p$ looks like a ramp. This is the mirage result, exactly: no change in the model's behaviour is needed to make the curve sharp, only $\phi(p) = p^{6}$.

**Step 2 — where the fix fails.** Fit a power law to the *smooth* surrogate on rungs 1–2: $\log_{10}\varepsilon = -0.398, -0.699$, slope $-0.301$ per dex. Extrapolate to rung 4: predicted $\log_{10}\varepsilon = -1.30$, so $\hat\varepsilon = 0.050$. Observed $\varepsilon = 0.010$. Relative error $R = |0.010 - 0.050| / 0.010 = 4.0$.

The observed slopes are $-0.301$, $-0.456$, $-0.845$ — accelerating, not constant. The continuous metric removed the cliff but did **not** make the curve extrapolable: $R = 4.0$ is far above $\tau = 0.25$.

**What this makes visible.** The mirage argument and the predictability argument are separate claims, and the literature routinely conflates them. Schaeffer et al. proved the first (sharpness is metric-induced). Nobody has proved the second (that a continuous metric restores forward prediction), and this arithmetic shows it can fail on numbers consistent with real ladders. That is why Section 8's deciding quantity is a *ratio* of residuals rather than a yes/no on curve shape — the ratio separates "the metric made it sharp" from "the underlying capability was already unpredictable."

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*