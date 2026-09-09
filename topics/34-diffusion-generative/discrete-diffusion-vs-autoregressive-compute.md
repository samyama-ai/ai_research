---
id: 34-diffusion-generative/discrete-diffusion-vs-autoregressive-compute
title: "Discrete Diffusion versus Autoregression at Matched Compute"
topic: 34-diffusion-generative
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Discrete Diffusion versus Autoregression at Matched Compute

> **Topic:** Diffusion & Generative Modeling · **ID:** `34-diffusion-generative/discrete-diffusion-vs-autoregressive-compute` · **Status:** empirically-open

## 1. Problem Statement

Given a fixed training compute budget $C$ (FLOPs), a fixed token budget $D$, and a fixed inference budget per generated sequence, does a masked/absorbing discrete diffusion language model reach lower loss and better downstream accuracy than an autoregressive (AR) transformer trained under the same constraints? If not everywhere, where is the crossover?

Three variants, with different difficulty:

- **Measurement.** Diffusion models report an *upper bound* on negative log-likelihood (an ELBO), AR models report exact NLL. Comparing them as if they were the same quantity is invalid in the direction that matters. The measurement problem is to define one number that is comparable and that both families can be scored on.
- **Method.** Build a discrete diffusion model that dominates a well-tuned AR baseline at matched $C$, $D$, and inference FLOPs on at least one non-cherry-picked axis (perplexity, reasoning accuracy, infilling, or throughput at fixed quality).
- **Theory.** Prove a separation: a family of distributions learnable by any-order/parallel denoising at compute $C$ that no left-to-right factorized model learns below $\omega(C)$, or the converse.

A solution states the crossover surface in $(N, D, T)$ — parameters, tokens, sampling steps — not a single benchmark win.

## 2. Formal Setting

Sequences $x = (x_1,\dots,x_L)$ over vocabulary $\mathcal{V}$, $|\mathcal{V}| = V$, with a mask symbol $\mathbf{m} \notin \mathcal{V}$.

**AR objective.** $\mathcal{L}_{\text{AR}}(\theta) = -\sum_{i=1}^{L} \log p_\theta(x_i \mid x_{<i})$, exact and tight.

**Masked diffusion objective.** With forward process $q(x_t \mid x_0)$ masking each token independently with probability $1-\alpha_t$, $\alpha_0 = 1$, $\alpha_1 = 0$, the continuous-time bound (Sahoo et al. 2024; Shi et al. 2024) is

$$-\log p_\theta(x_0) \;\le\; \mathcal{L}_{\text{MD}}(\theta) = \mathbb{E}_{t\sim U[0,1]}\,\mathbb{E}_{x_t\sim q}\left[\frac{\alpha_t'}{1-\alpha_t}\sum_{i:\,x_t^i = \mathbf{m}} \log p_\theta(x_0^i \mid x_t)\right].$$

Only masked positions contribute. Under uniform $t$, the expected number of supervised positions per forward pass is $L/2$, versus $L$ for AR.

**Compute, as measured.** Training: $C_{\text{train}} \approx 6ND$ for both families ($N$ non-embedding parameters, $D$ tokens *seen*, counting every token in the context whether or not it is supervised). Report also **supervised-token efficiency** $\eta = C_{\text{train}} / (\text{gradient-carrying token count})$: for AR $\eta \approx 6N$, for uniform-$t$ masked diffusion $\eta \approx 12N$.

Inference for $L$ tokens: AR with KV cache $\approx 2NL$; diffusion with $T$ denoising steps over the full sequence and no reusable cache $\approx 2NLT$. Ratio $T$. Semi-autoregressive block decoding with block size $B$ gives $\approx 2NL\,(T/B_{\text{steps}})$ with caching across blocks (Arriola et al. 2025).

**The comparable number.** Define the matched-compute frontier
$$\mathcal{F}_f(C) = \min_{N,D:\,6ND=C} \; \mathbb{E}_{x\sim\mathcal{D}_{\text{test}}}\big[\hat{\mathcal{L}}_f(x)\big], \quad f \in \{\text{AR}, \text{MD}\},$$
where $\hat{\mathcal{L}}_{\text{AR}}$ is exact NLL and $\hat{\mathcal{L}}_{\text{MD}}$ is the ELBO. The decision predicate is $\mathcal{F}_{\text{MD}}(C) < \mathcal{F}_{\text{AR}}(C)$.

**Assumptions, and which are violated.**
1. *The ELBO gap is small.* Unquantified for text; no tight two-sided estimate exists at scale. **Violated or unknown.**
2. *$6ND$ is a fair cost model for both.* Diffusion training typically uses random-length or full-length packing and Monte-Carlo $t$ sampling with high gradient variance; equal FLOPs do not mean equal gradient SNR. **Violated in the relevant direction.**
3. *Generative perplexity under an external scorer measures sample quality.* It is minimized by low-entropy degenerate samplers. **Known violated** (Zheng et al. 2025).
4. *Both families are equally tuned.* AR transformer hyperparameters have had eight years of optimization; diffusion noise schedules, $t$-sampling, and unmasking order have not. **Violated.**

## 3. State of the Art

**Established (ablated, reproduced).**
- Simplified masked diffusion — MDLM (Sahoo et al., NeurIPS 2024) and MD4 (Shi et al., NeurIPS 2024) — reduces D3PM/SEDD to a weighted masked-token cross-entropy and closes most of the gap to AR at GPT-2-small scale on OpenWebText. MDLM: test PPL bound $\le 23.21$ against $17.54$ for a matched AR transformer.
- RADD (Ou et al., ICLR 2025): absorbing discrete diffusion's denoiser is time-independent; the ELBO reduces to an expectation of conditional cross-entropies. This is a theorem, and it removes the time-conditioning design space from the comparison.
- Zheng et al. (ICLR 2025): a substantial part of earlier reported generative-perplexity gains came from 32-bit Gumbel categorical sampling, which acts as an unintended temperature reduction. A measurement artifact, not a modeling gain.

**Claimed but not fully ablated.**
- LLaDA-8B (Nie et al. 2025), trained on 2.3T tokens, is reported competitive with LLaMA3-8B on several benchmarks and better on reversal tasks. There is no matched-compute AR control trained by the same team on the same data; the comparison is against an externally trained model.
- "Diffusion beats autoregressive in data-constrained settings" (Prabhudesai et al. 2025): diffusion overtakes AR when unique data is fixed and epochs are large (tens to hundreds of repeats). Ablated within its own grid; the grid is $\le$ 1B parameters.

**Benchmark-number-only.** Dream-7B and comparable open diffusion LLMs (2025) report downstream scores without a compute-matched AR arm. Treat as existence proofs that the training recipe works at 7B, not as evidence about the frontier.

## 4. What Is Known

- **Compute penalty at fixed likelihood.** Nie et al. (ICLR 2025), fitting scaling laws over roughly $10^{18}$–$10^{21}$ FLOPs and models up to ~1B parameters, find masked diffusion needs about **16× the compute** of AR to match validation likelihood, while the two loss curves are parallel in $\log C$ — the exponent is similar, the constant is not.
- **Both families are power-law in compute.** Diffusion loss curves fit $L(C) = a C^{-b} + c$ with $b$ comparable to Kaplan/Chinchilla AR exponents. No published evidence of a crossing exponent in the compute-rich regime.
- **Data-constrained crossover is real.** Prabhudesai et al. (2025) report that with unique-token budgets in the $10^8$ range, diffusion continues improving past ~100 epochs where AR has already overfit, and the crossover epoch count grows with unique data.
- **Order matters, and averaging hides it.** Kim et al. (ICML 2025) show masked diffusion trains against the *worst-case* token subset ordering but can plan a good order at inference; on Sudoku, adaptive-order inference lifts accuracy from below 10% to above 85% with the same weights.
- **Inference cost is $T$-linear.** With $T = L$ steps, diffusion generation costs about $L\times$ an AR decode with KV cache. Quality degrades as $T$ falls; the quality-vs-$T$ curve is model- and dataset-specific and has no closed form.

## 5. What Is Not Known

- **Empirically open.** The crossover point in $C$, if any, at $\ge 10^{22}$ FLOPs with a same-team, same-data, same-tokenizer AR control. Every large diffusion LM to date compares against an external AR model. The experiment is runnable — it costs roughly one 7B-scale pretraining run per arm — and has not been run publicly.
- **Empirically open.** Whether the 16× penalty is a constant or shrinks with scale. Two points on a log-log plot at 1B do not extrapolate to 100B.
- **Methodologically blocked.** How to compare an ELBO against an exact NLL. Nobody has published a tight importance-weighted or annealed lower-bound-on-the-gap estimate for a text diffusion LM at scale, so "diffusion perplexity 23.2" and "AR perplexity 17.5" are not on the same axis; the true diffusion NLL is somewhere at or below 23.2.
- **Methodologically blocked.** A quality metric for parallel decoding that is not gameable by entropy reduction. Generative perplexity under an external scorer is not it.
- **Theoretically open.** No separation theorem in either direction under matched compute. Feng et al. (2025) give conditional-independence-based limitations on parallel decoding and an efficiency benefit for some tasks, but not a compute-matched separation.

## 6. Why It Is Hard

The specific obstruction is **an objective mismatch compounded by an incomparable metric**.

Diffusion's reported loss is an upper bound with an unmeasured gap; AR's is exact. Any observed advantage for AR could be entirely the bound gap, and any observed advantage for diffusion is a fortiori real but under-credited. There is no cheap estimator of the gap: the standard trick — importance sampling over the latent trajectory — has variance that scales with $L$ masked positions, so it is uninformative at $L = 1024$.

Second obstruction: **confounded compute accounting**. At equal FLOPs, masked diffusion supervises about half the tokens with a $1/(1-\alpha_t)$-weighted loss whose variance blows up near $t\to 0$. So "matched compute" silently varies gradient SNR between arms. Correcting for it (low-discrepancy $t$ sampling, clipped weights) changes the answer by an unpublished amount.

Third: cost. Resolving a 16× constant requires arms that differ by 16× in compute, so the decisive experiment is not one pretraining run but a grid spanning more than a decade in $C$.

## 7. Current Research (as of 2026)

- **Interpolation.** Block Diffusion (Arriola et al., ICLR 2025) makes block size $B$ a dial between AR ($B=1$) and full diffusion ($B=L$), restoring KV caching and variable-length generation. The compute-matched sweep over $B$ is the most direct route to the crossover surface. *(frontier — verify whether a $\ge$7B sweep has been published.)*
- **Inference-order planning.** Learned or entropy-based unmasking schedules (Kim et al. 2025 and successors).
- **Distillation.** Self-distillation through time (Deschenaux & Gulcehre, ICLR 2025) to cut $T$ by one to two orders of magnitude — this attacks the inference-side term, not the training-side 16×.
- **Data-constrained scaling.** CMU/Prabhudesai line of work; the strongest current case for diffusion is repeated-data regimes, which is where frontier training is heading.
- **Groups.** Stanford (Ermon), Cornell (Kuleshov), Google DeepMind, Tsinghua/RUC (LLaDA), EPFL. *(frontier — verify current affiliations and unreleased runs.)*

## 8. Concrete Next Experiment

**Scale.** Six-point compute ladder, $C \in \{3\times10^{19}, 10^{20}, 3\times10^{20}, 10^{21}, 3\times10^{21}, 10^{22}\}$ FLOPs, Chinchilla-optimal $(N, D)$ per point for the AR arm and the *same* $(N, D)$ for the diffusion arm. Identical tokenizer, identical data order-seed pool, $L = 2048$, single codebase. Roughly 12 runs, topping out near a 3B model on 200B tokens — about 3,000 A100-days total.

**Control arm.** A tuned AR transformer at each $C$. Plus a *second* control: Block Diffusion at $B = 16$, so the comparison is not binary.

**Deciding number.** Fit $L_f(C) = a_f C^{-b_f} + c_f$ for $f \in \{\text{AR}, \text{MD}, \text{BD}_{16}\}$ on held-out C4 and report the **compute multiplier** $\rho(C) = C_{\text{MD}}/C_{\text{AR}}$ at equal loss. The decision: is $\rho$ falling in $C$? If $\rho(10^{22}) \le 8$ against $\rho(3\times10^{19}) \approx 16$, the penalty is scale-decaying and a crossover is plausible; if $\rho$ is flat within $\pm 20\%$ across two decades, discrete diffusion does not win on likelihood at any reachable scale and the case must be made on inference throughput or data reuse instead.

**Required side-measurement.** On the largest checkpoint, run a 1,000-sample importance-weighted estimate of the ELBO gap. If the gap exceeds 0.05 nats/token, the headline comparison is not decidable from the ELBO and the paper must say so.

## 9. Key References

- **[Foundational]** Austin, J., Johnson, D., Ho, J., Tarlow, D., van den Berg, R. *Structured Denoising Diffusion Models in Discrete State-Spaces.* NeurIPS, 2021. — arXiv:2107.03006
- **[Foundational]** Lou, A., Meng, C., Ermon, S. *Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution.* ICML, 2024. — arXiv:2310.16834
- **[SOTA]** Sahoo, S., Arriola, M., Schiff, Y., Gokaslan, A., Marroquin, E., Chiu, J., Rush, A., Kuleshov, V. *Simple and Effective Masked Diffusion Language Models.* NeurIPS, 2024. — arXiv:2406.07524
- **[SOTA]** Shi, J., Han, K., Wang, Z., Doucet, A., Titsias, M. *Simplified and Generalized Masked Diffusion for Discrete Data.* NeurIPS, 2024. — arXiv:2406.04329
- **[SOTA]** Nie, S., Zhu, F., Du, C., Pang, T., Liu, Q., Zeng, G., Lin, M., Li, C. *Scaling up Masked Diffusion Models on Text.* ICLR, 2025. — arXiv:2410.18514
- **[SOTA]** Nie, S., Zhu, F., You, Z., Zhang, X., Ou, J., Hu, J., Zhou, J., Lin, Y., Wen, J.-R., Li, C. *Large Language Diffusion Models.* 2025. — arXiv:2502.09992
- **[SOTA]** Arriola, M., Gokaslan, A., Chiu, J., Yang, Z., Qi, Z., Han, J., Sahoo, S., Kuleshov, V. *Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models.* ICLR, 2025. — arXiv:2503.09573
- **[Measurement]** Zheng, K., Chen, Y., Mao, H., Liu, M.-Y., Zhu, J., Zhang, Q. *Masked Diffusion Models are Secretly Time-Agnostic Masked Models and Exploit Inaccurate Categorical Sampling.* ICLR, 2025. — arXiv:2409.02908
- **[Theory]** Ou, J., Nie, S., Xue, K., Zhu, F., Sun, J., Li, Z., Li, C. *Your Absorbing Discrete Diffusion Secretly Models the Conditional Distributions of Clean Data.* ICLR, 2025. — arXiv:2406.03736
- **[Empirical]** Prabhudesai, M., Wu, M., Zadeh, A., Fragkiadaki, K., Pathak, D. *Diffusion Beats Autoregressive in Data-Constrained Settings.* 2025. — arXiv:2507.15857
- **[Empirical]** Kim, J., Shah, K., Kontonis, V., Kakade, S., Chen, S. *Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions.* ICML, 2025.
- **[Foundational]** Hoffmann, J., et al. *Training Compute-Optimal Large Language Models.* NeurIPS, 2022. — arXiv:2203.15556
- **[Theory]** Feng, G., Geng, Y., Guan, J., Wu, W., Wang, L., He, D. *Theoretical Benefit and Limitation of Diffusion Language Model.* 2025.

## 10. Worked Example

Take the MDLM setting: GPT-2-small architecture, $N \approx 8.5\times10^7$ non-embedding parameters, $L = 1024$, OpenWebText, $D \approx 3.3\times10^{10}$ tokens. Training compute per arm $\approx 6ND = 6 \times 8.5{\times}10^7 \times 3.3{\times}10^{10} \approx 1.7\times10^{19}$ FLOPs. The AR arm reaches test perplexity $17.54$; the diffusion arm reaches an ELBO-perplexity of $\le 23.21$.

Naive reading: AR is ahead by $\log(23.21/17.54) = 0.28$ nats/token.

Now make the two obstructions visible.

1. **The bound.** $23.21$ is an upper bound. If the ELBO gap is $0.28$ nats/token, the diffusion model's true NLL is $17.54$ and the arms are tied. Is $0.28$ nats plausible? At $L=1024$ that is 287 nats per sequence of slack — small relative to the ~18,000-nat total. Nobody has measured it. **The headline number cannot distinguish "AR wins by 0.28 nats" from "tie".**

2. **The compute accounting.** Under uniform $t$, the diffusion arm backpropagates through roughly $L/2 = 512$ supervised positions per sequence versus $1024$ for AR at identical FLOPs. Equalizing supervised tokens rather than FLOPs would give the diffusion arm $3.4\times10^{19}$ FLOPs. Using the fitted 16× multiplier from Nie et al., a 2× compute increase buys about $2^{-b}$ loss reduction with $b\approx0.15$, i.e. roughly 10% of the gap — not enough to close it, but enough that "matched compute" and "matched supervision" give different rank orders in the *small*-gap regime near the crossover.

3. **Inference.** At $T = 1024$ steps, generating one 1024-token sequence costs $2NLT \approx 1.8\times10^{14}$ FLOPs versus $2NL \approx 1.7\times10^{11}$ for AR with a KV cache — a 1,000× gap. Cutting to $T = 32$ gives 32×, and the quality loss at $T=32$ is reported as a generative-perplexity increase — a metric that Zheng et al. showed is confounded by sampler precision.

Every leg of the comparison — train loss, train compute, inference quality — has a confound of the same order as the effect. That, not the size of the runs, is why the question stays open.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*