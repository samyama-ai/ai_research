---
id: 09-model-design/diffusion-vs-autoregressive-language-scaling
title: "Diffusion Language Models Versus Autoregressive Scaling"
topic: 09-model-design
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Diffusion Language Models Versus Autoregressive Scaling

> **Topic:** Architecture & Model Design · **ID:** `09-model-design/diffusion-vs-autoregressive-language-scaling` · **Status:** empirically-open

## 1. Problem Statement

Autoregressive (AR) transformers factor $p(x) = \prod_i p(x_i \mid x_{<i})$ and train on one exact log-likelihood. Discrete diffusion language models (DLMs) — in practice, masked/absorbing-state diffusion — train on a variational bound over random masking ratios and decode many positions per step. The question is whether DLMs are a better use of compute than AR at scale, and if not, where the crossover lies.

Three variants, which are routinely conflated:

- **Measurement.** Given matched compute $C$ and matched data $D$, does the DLM Pareto frontier of (loss, $C$) ever cross the AR frontier? At what $(C, D)$? Blocked partly by the fact that DLM loss is a bound, not a likelihood, so the two curves are not on the same axis.
- **Method.** Is there a training or sampling recipe (any-order objectives, block decoding, distillation, AR initialization) that closes the reported compute gap without giving up the parallel-decoding advantage?
- **Theory.** Is the gap intrinsic — a lower bound in bits/parameter/FLOP for any-order modeling under a fixed architecture — or an artifact of the current objective, whose slack is removable?

A solution to the measurement variant is a pair of fitted scaling laws with stated confidence intervals on the crossover compute $C^\star$, using an evaluation both families can be scored on.

## 2. Formal Setting

Vocabulary $\mathcal{V}$, $|\mathcal{V}| = V$, sequence $x \in \mathcal{V}^L$, data distribution $q$.

**AR loss (measured).** Sum of per-token cross-entropies on a held-out corpus, divided by token count:
$$\mathcal{L}_{\mathrm{AR}}(\theta) = -\tfrac{1}{L}\,\mathbb{E}_{q}\sum_{i=1}^{L}\log p_\theta(x_i \mid x_{<i}).$$
This is an exact negative log-likelihood (NLL) per token, in nats. One forward pass per sequence.

**Masked-diffusion loss (measured).** With absorbing state $\mathbf{m}$ and mask schedule $\alpha_t$ decreasing from $1$ to $0$, the continuous-time bound (Sahoo et al. 2024; Shi et al. 2024; Ou et al. 2025) reduces to a reweighted masked-token cross-entropy:
$$\mathcal{L}_{\mathrm{DLM}}(\theta) = \mathbb{E}_{t\sim U[0,1]}\ \frac{1}{t}\ \mathbb{E}_{x_t \sim q_t(\cdot\mid x)} \Big[\tfrac{1}{L}\sum_{i:\,x_t^i = \mathbf{m}} -\log p_\theta(x^i \mid x_t)\Big] \ \ge\ \mathcal{L}_{\mathrm{AR-oracle}} .$$
In practice: sample a masking ratio, mask, predict masked positions, weight by $1/t$. Monte Carlo over $t$ makes the gradient estimator higher-variance than AR's.

**Key inequality.** $\mathcal{L}_{\mathrm{DLM}}$ is an *upper bound* on NLL, equal to the expected NLL under a uniformly random decoding order. Comparing it to $\mathcal{L}_{\mathrm{AR}}$ compares a bound to an exact value, so any DLM "loss gap" is an upper bound on the true gap.

**Compute (measured).** $C \approx 6ND$ FLOPs for both families at training ($N$ non-embedding parameters, $D$ tokens seen). This is the first assumption that is violated: DLMs see each token at multiple noise levels and typically need more epochs, so $D$ (unique tokens) and $D$ (tokens processed) diverge. Report both.

**Inference compute.** AR: $L$ sequential forwards. DLM with $K$ denoising steps over $L$ positions: $K$ forwards, $K \ll L$ possible, so throughput advantage $\approx L/K$ — but per-step quality falls as $K$ drops, so the correct axis is *quality at fixed inference FLOPs*, not steps.

**Scaling law form.** Fit $\mathcal{L}(N, D) = E + A N^{-\alpha} + B D^{-\beta}$ (Hoffmann et al. 2022) separately per family, then define the crossover
$$C^\star = \inf\{C : \mathcal{L}^{\mathrm{DLM}}_{\min}(C) \le \mathcal{L}^{\mathrm{AR}}_{\min}(C)\}.$$

**Assumptions known to be violated:** (i) both families use the same tokenizer and data — often false across published comparisons; (ii) hyperparameters are equally tuned — DLMs have had far less tuning history; (iii) the loss bound is tight — it is not, and its slack is unmeasured; (iv) $6ND$ holds — bidirectional attention without KV-cache changes the inference constant sharply.

## 3. State of the Art

**Established.**
- Masked/absorbing diffusion is the surviving discrete formulation. D3PM (Austin et al., NeurIPS 2021) introduced the family; SEDD (Lou, Meng, Ermon, ICML 2024) reached GPT-2-scale perplexity bounds via score-entropy; MDLM (Sahoo et al., NeurIPS 2024) and MD4 (Shi et al., NeurIPS 2024) simplified the objective to weighted masked cross-entropy and reproduced each other independently.
- RADD / time-agnostic results (Ou et al., ICLR 2025) established that the absorbing-diffusion network need not condition on $t$; the loss is a reweighted any-order autoregressive objective. This is a theorem, reproduced.
- Nie et al. (ICLR 2025, *Scaling up Masked Diffusion Models on Text*) fit scaling laws for both families and report MDMs following clean power laws with an AR-matching compute ratio around $16\times$ on likelihood-adjacent tasks. This is the closest thing to a direct answer.

**Claimed but unablated.**
- LLaDA-8B (Nie et al. 2025) matches LLaMA3-8B on many benchmarks while trained on ~2.3T tokens. Benchmark numbers only — no matched-compute AR control trained on the identical 2.3T corpus was released, so the comparison is against a model with a different data mixture.
- Dream-7B (2025) is initialized from AR weights (Qwen2.5), which makes it evidence about adaptation cost, not about diffusion scaling from scratch.
- Commercial DLMs (Inception Labs *Mercury*, 2025; Google *Gemini Diffusion*, 2025) report large throughput gains (hundreds to >1000 tok/s). Latency numbers, not loss-versus-compute evidence. *(frontier — verify)*
- Prabhudesai et al. (2025) report diffusion overtaking AR in data-constrained settings — when unique data is fixed and epochs are many. Single lab, small-to-mid scale.

## 4. What Is Known

- **The bound gap is real at GPT-2 scale.** SEDD reports perplexity bounds within roughly $1.1$–$1.4\times$ of GPT-2 small/medium on standard zero-shot sets; it does not beat a matched AR model on exact likelihood. Scale: 90M–400M params.
- **Compute multiplier ~16×.** Nie et al. (ICLR 2025) fit MDM and AR scaling laws over roughly $10^{18}$–$10^{20}$ FLOPs and find MDMs need on the order of $16\times$ the compute to reach equal likelihood-based performance, while the *exponents* are similar — the curves are near-parallel, not converging, in that window.
- **Likelihood-based DLMs beat GPT-2 on some zero-shot sets.** Plaid 1B (Gulrajani & Hashimoto, NeurIPS 2023) is the first DLM to exceed GPT-2 zero-shot likelihood, at roughly an order of magnitude more training compute.
- **Order matters.** Kim et al. (ICML 2025) show masked diffusion is trained on the worst-case token orderings but can be *planned* over at inference; adaptive decoding order recovers substantial accuracy on tasks like Sudoku, from near-chance to high accuracy. Scale: small task-specific models.
- **Repetition tolerance.** AR models degrade after ~4 epochs of repeated data (Muennighoff et al., NeurIPS 2023); DLMs are reported to keep improving over far more repeats, because each epoch presents a different masking pattern.
- **Inference is not free.** Reducing $K$ below ~$L/4$ degrades generation quality measurably in every published DLM; the $L/K$ speedup is bought with loss.

## 5. What Is Not Known

- **Empirically open.** Whether the ~16× multiplier shrinks, holds, or grows above $10^{22}$ FLOPs. Nobody has trained a matched pair (same tokenizer, same corpus, same tuning budget) at $\ge 10^{22}$ FLOPs and published both curves. Runnable today for roughly a few hundred thousand GPU-hours.
- **Empirically open.** Whether the data-constrained crossover (Prabhudesai et al. 2025) survives at 8B+ parameters and trillion-token unique corpora, which is the regime that matters if data, not compute, becomes binding.
- **Theoretically open.** No lower bound showing any-order modeling must cost more parameters or FLOPs than fixed-order modeling for a natural language class. Conversely, no proof the ELBO slack is removable.
- **Methodologically blocked.** There is no accepted way to compare a bound to a likelihood. Nor is there an agreed *quality-at-fixed-inference-FLOPs* metric that both families are scored on; benchmark accuracy at unspecified $K$ is not one.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by compute cost**. Three confounds ride together in every published comparison: (i) the DLM number is an upper bound and the AR number is exact, so a tie in reported loss is a DLM win of unknown size; (ii) DLM inference has a free parameter $K$ that trades quality for FLOPs and is chosen differently in every paper, so benchmark tables do not fix the compute axis; (iii) AR has ~8 years of hyperparameter, initialization, and data-curation tuning that DLMs do not, so any gap mixes "intrinsic" with "under-tuned". Deconfounding requires the matched-pair run — and at the scale where the answer changes policy ($\ge 10^{22}$ FLOPs, $\times$ at least 4 points per family for a scaling fit, $\times$ the ~16× DLM handicap), the experiment costs on the order of a frontier pretraining run. That is why it has not been done outside a handful of labs, and those labs ship benchmark tables instead of curves.

## 7. Current Research (as of 2026)

- **Objective tightening:** any-order and time-agnostic reformulations (Ou et al.; Zheng et al.), which remove the $t$-conditioning and reduce estimator variance. Tsinghua, Stanford, Cornell.
- **AR-to-diffusion adaptation:** Dream-7B-style initialization from an AR checkpoint, sidestepping the pretraining multiplier entirely. HKU / Huawei Noah's Ark.
- **Block / semi-autoregressive hybrids:** block diffusion (Cornell, ICLR 2025 oral) interpolating between AR and diffusion to recover KV-caching and arbitrary length.
- **Few-step distillation** of the denoiser, targeting $K \le 8$ without quality loss. Inception Labs, Google DeepMind *(frontier — verify)*.
- **Data-constrained scaling:** CMU-led work on diffusion as the better repetition-tolerant learner.

## 8. Concrete Next Experiment

**Question decided:** does the DLM/AR compute multiplier at equal validation loss shrink with scale?

- **Scale:** four matched pairs at $N \in \{150\text{M}, 400\text{M}, 1.3\text{B}, 3\text{B}\}$ non-embedding parameters, each trained to Chinchilla-optimal $D$ for the AR arm and to $16\times$ compute for the DLM arm. Top pair ≈ $2\times10^{21}$ FLOPs; total ≈ 40k–60k H100-hours.
- **Control arm:** identical tokenizer, identical token stream and ordering, identical transformer trunk (only attention mask and head differ), and an equal hyperparameter search budget per arm (same number of LR/warmup trials).
- **Deconfound the bound:** for each DLM checkpoint, additionally estimate exact NLL by importance-weighted multi-order evaluation with $M=128$ sampled orders, and report both the ELBO and the tightened estimate. This makes the slack a measured number rather than an assumption.
- **The deciding number:** the fitted compute multiplier $R(N) = C_{\mathrm{DLM}}(\mathcal{L}) / C_{\mathrm{AR}}(\mathcal{L})$ at matched tightened NLL, regressed against $\log N$. If $d\log R / d\log N < -0.1$ with a 95% CI excluding zero, DLMs close the gap and $C^\star$ is finite and extrapolable. If the CI contains zero, the multiplier is scale-invariant and the DLM case rests entirely on inference throughput and data-constrained regimes, not on pretraining efficiency.

## 9. Key References

- **[Foundational]** Austin, Johnson, Ho, Tarlow, van den Berg. *Structured Denoising Diffusion Models in Discrete State-Spaces.* NeurIPS 2021. — arXiv:2107.03006
- **[Foundational]** Li, Thickstun, Gulrajani, Liang, Hashimoto. *Diffusion-LM Improves Controllable Text Generation.* NeurIPS 2022. — arXiv:2205.14217
- **[SOTA]** Lou, Meng, Ermon. *Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution.* ICML 2024. — arXiv:2310.16834
- **[SOTA]** Sahoo, Arriola, Schiff, Gokaslan, Marroquin, Chiu, Rush, Kuleshov. *Simple and Effective Masked Diffusion Language Models.* NeurIPS 2024. — arXiv:2406.07524
- **[SOTA]** Shi, Han, Wang, Doucet, Titsias. *Simplified and Generalized Masked Diffusion for Discrete Data.* NeurIPS 2024. — arXiv:2406.04329
- **[SOTA]** Nie, Zhu, Han, Wang, Li, Zhang. *Large Language Diffusion Models.* 2025. — arXiv:2502.09992
- **[Key result]** Nie, Zhu, Du, Li, Zhang. *Scaling up Masked Diffusion Models on Text.* ICLR 2025. — arXiv:2410.18514
- **[Key result]** Gulrajani, Hashimoto. *Likelihood-Based Diffusion Language Models.* NeurIPS 2023. — arXiv:2305.18619
- **[Theory]** Ou, Nie, Xue, Liu, Zhang, Lin, Li. *Your Absorbing Discrete Diffusion Secretly Models the Conditional Distributions of Clean Data.* ICLR 2025.
- **[Analysis]** Kim, Shah, Kontonis, Oymak, Vishwanath. *Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions.* ICML 2025.
- **[Comparison]** Prabhudesai et al. *Diffusion Beats Autoregressive in Data-Constrained Settings.* 2025.
- **[Baseline]** Hoffmann et al. *Training Compute-Optimal Large Language Models.* NeurIPS 2022. — arXiv:2203.15556
- **[Baseline]** Muennighoff et al. *Scaling Data-Constrained Language Models.* NeurIPS 2023. — arXiv:2305.16264

## 10. Worked Example

Take a 1.3B-parameter budget with $D = 26$B tokens (Chinchilla-optimal, $20\times N$). AR training compute:
$$C_{\mathrm{AR}} = 6 \times 1.3\times10^9 \times 2.6\times10^{10} \approx 2.0\times10^{20}\ \text{FLOPs}.$$
Suppose the AR arm reaches validation NLL $2.05$ nats/token. Applying the reported $16\times$ multiplier, matching that with a DLM needs $C_{\mathrm{DLM}} \approx 3.2\times10^{21}$ FLOPs — the same 1.3B model over $\approx 416$B tokens processed, i.e. 16 epochs of the 26B-token corpus.

Now the obstruction. The DLM's reported $2.05$ is an ELBO. Tightening it with $M$ sampled decoding orders typically recovers some slack; call the recovered amount $\delta$ nats. Published tightening experiments at small scale suggest $\delta$ in the range $0.02$–$0.10$ nats — but $\delta$ has never been measured at 1.3B. Local loss-versus-compute slope in this window is roughly $\Delta \mathcal{L} \approx -0.03$ nats per compute doubling. So:

| assumed $\delta$ | doublings of AR compute it is worth | implied true multiplier |
|---|---|---|
| $0.02$ | $0.7$ | $\approx 16 / 1.6 \approx 10\times$ |
| $0.06$ | $2.0$ | $\approx 4\times$ |
| $0.10$ | $3.3$ | $\approx 1.6\times$ |

The headline "16×" therefore spans a factor of ten depending on a quantity nobody has measured at scale. The experiment in §8 costs less than one frontier run and collapses that column to a single number — which is why the ELBO-tightening arm, not the extra parameters, is the part of the design that actually decides the problem.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*