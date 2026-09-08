---
id: 10-scaling-laws/critical-batch-size-growth
title: "Critical Batch Size Growth Law"
topic: 10-scaling-laws
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Critical Batch Size Growth Law

> **Topic:** Scaling Laws & Compute Allocation · **ID:** `10-scaling-laws/critical-batch-size-growth` · **Status:** empirically-open

## 1. Problem Statement

The critical batch size $B_{\mathrm{crit}}$ is the batch size beyond which increasing parallelism stops buying proportional reductions in optimizer steps. Below it, doubling the batch nearly halves steps-to-target; above it, tokens are burned for little wall-clock gain. Every large pre-training run picks a batch size, and that pick determines how much data parallelism is usable before the run becomes token-inefficient.

The open question is what $B_{\mathrm{crit}}$ is a function of. Three candidate laws are in circulation and are **not** equivalent:

$$B_{\mathrm{crit}} \propto L^{-1/\alpha_B}, \qquad B_{\mathrm{crit}} \propto D^{\beta}, \qquad B_{\mathrm{crit}} \propto C^{\gamma},$$

with $L$ the loss reached, $D$ the tokens consumed, $N$ the parameter count, and $C \approx 6ND$ the compute. They coincide along a Chinchilla-optimal ray (where $L$, $D$, $N$, $C$ move together) and diverge off it — exactly the regime of over-trained production models.

Three variants, with different difficulty:

- **Measurement.** Given a fixed architecture, data distribution, and optimizer, estimate $B_{\mathrm{crit}}$ at a specified $(N, D)$ point with stated error bars. Requires a batch-size sweep with per-batch-size learning-rate retuning.
- **Method.** Predict $B_{\mathrm{crit}}$ at a target scale from cheap proxies (gradient noise scale, small-scale sweeps) without running the sweep at target scale.
- **Theory.** Derive the exponent from an optimization model of the loss landscape, rather than fitting it.

Solved means: a law with held-out predictive accuracy — predicting $B_{\mathrm{crit}}$ at a scale $\geq 10\times$ above the fitting range to within a factor of ~1.3, under a stated learning-rate rule.

## 2. Formal Setting

Fix architecture family, tokenizer, data distribution $\mathcal{D}$, and optimizer. Train with batch size $B$ (tokens per optimizer step) and per-$B$ tuned peak learning rate $\eta^\star(B)$, warmup, and decay schedule.

**Steps-to-target.** For a target loss $L_\tau$ define
$$S(B; L_\tau) = \min\{ s : L_s \le L_\tau \}, \qquad E(B; L_\tau) = B \cdot S(B; L_\tau).$$
$S$ is measured by running to $L_\tau$ and reading the step index; $E$ is tokens consumed. Both are random variables over seed and data order; report medians over $\geq 3$ seeds.

**Pareto branch definition (McCandlish et al., 2018).** The empirical $(S, E)$ frontier is well fit by
$$\left(\frac{S}{S_{\min}} - 1\right)\left(\frac{E}{E_{\min}} - 1\right) = 1, \qquad B_{\mathrm{crit}} \equiv \frac{E_{\min}}{S_{\min}}.$$
$B_{\mathrm{crit}}$ is thus a *fitted* hyperbola parameter, not a directly observed threshold. At $B = B_{\mathrm{crit}}$ the run pays $2\times$ the minimum steps and $2\times$ the minimum tokens.

**Gradient noise scale (the cheap proxy).**
$$\mathcal{B}_{\text{noise}} = \frac{\operatorname{tr}\Sigma}{|G|^2}, \qquad \Sigma = \operatorname{Cov}_{x\sim\mathcal{D}}\big[\nabla_\theta \ell(x;\theta)\big],$$
measured in practice as $\mathcal{B}_{\text{simple}} = \operatorname{tr}\Sigma / |G|^2$ estimated from the difference of squared gradient norms at two batch sizes $B_{\text{small}}, B_{\text{big}}$ across data-parallel workers, with EMA smoothing.

**The law to be fit.** Candidates:
$$B_{\mathrm{crit}}(L) = \frac{B_*}{L^{1/\alpha_B}}, \qquad \log B_{\mathrm{crit}} = a + \beta \log D + \delta \log N .$$
The decision predicate is whether $\delta$ is statistically distinguishable from $0$ at fixed $D$.

**Assumptions, and which are violated.**
- *Optimal LR at every $B$.* Violated routinely: sweeps use a linear or square-root rule instead. Under-tuned $\eta$ at large $B$ inflates $S$ and biases $B_{\mathrm{crit}}$ **downward**.
- *Hyperbolic Pareto form.* Approximate; residuals are rarely reported.
- *Target loss is scale-free.* Violated: $B_{\mathrm{crit}}$ estimated at $L_\tau$ far above a model's final loss measures early-training dynamics, not the run.
- *Optimizer-independence.* Violated: Adam, Muon, and Shampoo do not share a $B_{\mathrm{crit}}$.
- *Fixed schedule.* Violated: cosine decay couples $S$ to the pre-declared horizon, so $S(B)$ is not measurable without either constant-LR or per-$B$ re-scheduling (WSD helps).

## 3. State of the Art

**Established.**
- McCandlish et al. (2018) — the hyperbolic trade-off and the noise-scale predictor, validated across image classification, RL (Atari, Dota), and language modeling; $\mathcal{B}_{\text{noise}}$ predicts $B_{\mathrm{crit}}$ within roughly an order of magnitude and tracks it *within* a run as training proceeds.
- Shallue et al. (JMLR 2019) — perfect scaling, then diminishing returns, then saturation, across 6 architectures and 7 datasets, with exhaustive per-$B$ hyperparameter search. Key negative result: the saturation point varied by orders of magnitude across workload and was **not** predicted by simple metrics they tested.
- Zhang et al. (ICLR 2025), *How Does Critical Batch Size Scale in Pre-training?* — with LR and weight decay retuned per $B$, CBS scales primarily with **data size**, showing little dependence on model size at fixed data. Measured on decoder-only LMs up to ~1.2B parameters.

**Claimed but unablated.**
- Kaplan et al. (2020) fit $B_{\mathrm{crit}}(L) = B_*/L^{1/\alpha_B}$ with $B_* \approx 2\times10^8$ tokens and $\alpha_B \approx 0.21$. The fit is loss-parameterized; because $L$, $N$, $D$ were varied jointly, the law cannot separate a data effect from a model-size effect. It is widely cited as if it could.
- DeepSeek LLM (2024) reports $B_{\mathrm{opt}} \propto C^{\approx 0.33}$, $\eta_{\mathrm{opt}} \propto C^{\approx -0.12}$, fitted on IsoFLOP grids. This is *optimal* batch size under a wall-clock-agnostic generalization-error criterion, not the Pareto branch point — the two are conflated in downstream citations.
- Bergsma et al. (2025), *Power Lines*, gives joint power laws for batch size and weight decay, with batch size dependent on data rather than parameters — consistent with Zhang et al., fitted at sub-billion scale.

**Benchmark-number-only.** Production batch sizes (Llama 3's ~16M-token batches, GPT-4-class runs) are cited as evidence of a growth law. They are not: they are systems-throughput choices with no accompanying $S(B)$ sweep.

## 4. What Is Known

- Kaplan et al. (2020), LMs $10^3$–$10^9$ params on WebText2: $\alpha_B \approx 0.21$, $B_* \approx 2\times10^8$ tokens. At $L = 3.0$ nats this yields $B_{\mathrm{crit}} \approx 4\times10^6$ tokens.
- Shallue et al. (2019), up to ResNet-50/ImageNet and Transformer/LM1B: perfect scaling extends to $B\sim10^3$–$10^4$ depending on workload; saturation batch sizes differ by $>100\times$ across workloads at similar model size.
- Zhang et al. (ICLR 2025): at fixed model size, increasing training tokens raises CBS; at fixed tokens, increasing $N$ from ~85M to ~1.2B moves CBS little. Fitted exponent on data is sub-linear.
- McCandlish et al. (2018): $\mathcal{B}_{\text{noise}}$ grows by roughly an order of magnitude over a single training run as loss falls — the growth is *within*-run, not only across-scale.
- Goyal et al. (2017): linear LR scaling with warmup holds to $B = 8192$ images on ImageNet with no accuracy loss; breaks past that.
- Adaptive-optimizer caveat: Li et al. (NeurIPS 2024) report a *surge* — optimal LR rising then falling with $B$ under Adam — contradicting the monotone square-root rule assumed in most CBS sweeps.

## 5. What Is Not Known

- **Empirically open (primary).** Whether $B_{\mathrm{crit}}$ depends on $N$ at fixed $D$. The experiment is a $4\times4$ $(N, D)$ grid with per-cell LR retuning; it is runnable today at $\leq 10^{21}$ FLOP and has been run only to ~1.2B params on a narrow data range. No published result reaches the 10B+ parameter, 10T+ token regime where production runs sit.
- **Empirically open.** Whether the exponent transfers across optimizers (AdamW vs Muon) and across data-repetition regimes.
- **Methodologically blocked.** There is no agreed estimator of $B_{\mathrm{crit}}$. The hyperbola fit, the "90%-of-linear-speedup" threshold, and the IsoFLOP argmin give different numbers on the same data, sometimes by $2$–$4\times$. Papers report $B_{\mathrm{crit}}$ without stating the estimator or its error bar.
- **Theoretically open.** No derivation of $\alpha_B$ (or $\beta$) from a loss-landscape model. Quadratic/NQM analyses (Zhang et al., NeurIPS 2019) reproduce the *shape* of the curve but predict the location only given a spectrum that must itself be measured.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement under a nested tuning loop**. $B_{\mathrm{crit}}$ is defined only relative to *optimal* $\eta(B)$, $\lambda(B)$, warmup, and schedule. Estimating it therefore costs an inner hyperparameter sweep per batch size per $(N,D)$ cell — an $O(10^2)$ multiplier on an already expensive grid. Almost every published estimate short-circuits this loop with a heuristic LR rule, and the bias is one-signed: under-tuning at large $B$ makes $B_{\mathrm{crit}}$ look smaller. Because the reported growth exponents come from studies with different short-circuits, disagreement between $\alpha_B \approx 0.21$ (loss) and $\gamma \approx 0.33$ (compute) is not decidable from published numbers.

Second obstruction: **non-identifiability along the Chinchilla ray**. Historic scaling studies varied $N$ and $D$ together, so $\log B_{\mathrm{crit}} = a + \beta\log D + \delta\log N$ has a collinear design matrix; $\beta$ and $\delta$ are not separately estimable from that data. Only a deliberately off-ray grid breaks it.

## 7. Current Research (as of 2026)

- **Data-driven CBS.** Zhang, Kakade, and collaborators (Harvard/Kempner) pushing the "CBS scales with data, not parameters" result to larger $N$ *(frontier — verify)*.
- **Joint HP power laws.** Cerebras (Bergsma et al.) fitting batch size, LR, and weight decay jointly; the $\lambda \cdot \eta$ timescale is the claimed invariant.
- **µP-style transfer.** Extending µTransfer (Yang et al., 2021) beyond width to batch size, so $B_{\mathrm{crit}}$ transfers from proxy models. Whether $B$ admits a µP-like invariant is unresolved *(frontier — verify)*.
- **Non-Adam optimizers.** Muon and related orthogonalized-momentum methods are reported to tolerate larger batches; ablations against a retuned AdamW baseline are thin *(frontier — verify)*.
- **Systems pressure.** Multi-datacenter and asynchronous training make large $B$ attractive independent of token efficiency, raising demand for a reliable law.

## 8. Concrete Next Experiment

**Question.** Is $\delta$ (the exponent on $N$ at fixed $D$) zero?

**Scale.** Decoder-only transformers at $N \in \{0.3\mathrm{B}, 1\mathrm{B}, 3\mathrm{B}, 8\mathrm{B}\}$, each trained on $D \in \{20N, 100N_{\text{ref}}, 500N_{\text{ref}}\}$ tokens with $N_{\text{ref}} = 1\mathrm{B}$ — i.e. a deliberately **off-Chinchilla-ray** grid where $D$ is set independently of $N$. Fixed data (a single deduplicated web corpus), fixed tokenizer, AdamW, WSD schedule so runs can be branched at any horizon. Total ~$3\times10^{22}$ FLOP for the main grid.

**Per cell.** Batch sizes $B \in \{0.25, 0.5, 1, 2, 4, 8\}\times$ a per-cell anchor, spanning $\geq 5$ octaves. For each $B$, tune $\eta$ over a 5-point log grid and $\lambda$ over 3 points, selecting by final loss. 3 seeds at the selected point.

**Control arm.** Same grid with the *heuristic* rule $\eta \propto \sqrt{B}$, no per-$B$ retuning — reproducing standard practice. The control quantifies the tuning-induced bias directly.

**Estimator.** Fit the hyperbola $(S/S_{\min}-1)(E/E_{\min}-1)=1$ at three target losses per cell; report $B_{\mathrm{crit}}$ with bootstrap CIs, and report the 90%-linear-speedup threshold alongside so estimator sensitivity is visible.

**Deciding number.** The fitted $\delta$ in $\log B_{\mathrm{crit}} = a + \beta \log D + \delta \log N$, with its 95% CI. **$|\delta| < 0.05$ with CI excluding $0.15$** falsifies model-size dependence and confirms the data-only law. $\delta > 0.15$ with CI excluding $0$ restores it. A secondary number: the ratio $B_{\mathrm{crit}}^{\text{tuned}} / B_{\mathrm{crit}}^{\text{control}}$, which quantifies how much of the literature's disagreement is tuning bias.

## 9. Key References

- **[Foundational]** Sam McCandlish, Jared Kaplan, Dario Amodei, OpenAI Dota Team. *An Empirical Model of Large-Batch Training.* arXiv preprint, 2018. — arXiv:1812.06162
- **[Foundational]** Christopher J. Shallue, Jaehoon Lee, Joseph Antognini, Jascha Sohl-Dickstein, Roy Frostig, George E. Dahl. *Measuring the Effects of Data Parallelism on Neural Network Training.* JMLR 20(112), 2019. — arXiv:1811.03600
- **[Foundational]** Jared Kaplan, Sam McCandlish, Tom Henighan, et al. *Scaling Laws for Neural Language Models.* arXiv preprint, 2020. — arXiv:2001.08361
- **[SOTA]** Hanlin Zhang, Depen Morwani, Nikhil Vyas, Jingfeng Wu, Difan Zou, Udaya Ghai, Dean Foster, Sham Kakade. *How Does Critical Batch Size Scale in Pre-training?* ICLR, 2025. — arXiv:2410.21676
- **[SOTA]** Shane Bergsma, Nolan Dey, Gurpreet Gosal, Gavia Gray, Daria Soboleva, Joel Hestness. *Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training.* 2025. — arXiv:2505.13738
- **[Supporting]** Priya Goyal, Piotr Dollár, Ross Girshick, et al. *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour.* arXiv preprint, 2017. — arXiv:1706.02677
- **[Supporting]** Guodong Zhang, Lala Li, Zachary Nado, James Martens, Sushant Sachdeva, George E. Dahl, Christopher J. Shallue, Roger Grosse. *Which Algorithmic Choices Matter at Which Batch Sizes? Insights from a Noisy Quadratic Model.* NeurIPS, 2019. — arXiv:1907.04164
- **[Supporting]** DeepSeek-AI. *DeepSeek LLM: Scaling Open-Source Language Models with Longtermism.* arXiv preprint, 2024. — arXiv:2401.02954
- **[Supporting]** Greg Yang, Edward J. Hu, Igor Babuschkin, et al. *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer.* NeurIPS, 2021. — arXiv:2203.03466

## 10. Worked Example

Take a 7B model targeted at $L_\tau = 2.0$ nats, and ask what batch size to use.

**Route A — Kaplan's loss law.** $B_{\mathrm{crit}} = B_*/L^{1/\alpha_B}$ with $B_* = 2\times10^8$ tokens, $\alpha_B = 0.21$, so $1/\alpha_B \approx 4.76$:
$$B_{\mathrm{crit}} = \frac{2\times10^8}{2.0^{4.76}} = \frac{2\times10^8}{27.1} \approx 7.4\times10^6 \text{ tokens}.$$

**Route B — a data law.** Suppose $B_{\mathrm{crit}} \propto D^{0.5}$, anchored at Zhang et al.'s regime: $B_{\mathrm{crit}} \approx 1\times10^6$ tokens at $D = 100$B. A 7B model trained Chinchilla-style ($D = 140$B) gives $1\times10^6\sqrt{1.4} \approx 1.2\times10^6$ tokens. The same 7B model over-trained to $D = 2.8$T ($20\times$ more data, roughly Llama-3-8B territory) gives $1\times10^6\sqrt{28} \approx 5.3\times10^6$ tokens.

**Where the obstruction shows.** At the Chinchilla point the two routes differ by $6\times$ ($7.4$M vs $1.2$M tokens). At the over-trained point they nearly agree ($7.4$M vs $5.3$M) — because Route A's $L$ has fallen too, and $L$, $D$ move together. So the disagreement is largest exactly where the data is thinnest, and vanishes exactly where the historic fits were made. That is the non-identifiability in Section 6, made numerical: no published dataset separates the two, because no published grid puts a fixed $D$ against widely varying $N$.

**Cost of the error.** Choosing $B = 7.4$M when the true $B_{\mathrm{crit}}$ is $1.2$M puts the run at $B/B_{\mathrm{crit}} \approx 6$. On the hyperbola, $E/E_{\min} = 1 + B/B_{\mathrm{crit}} = 7$: the run consumes about $7\times$ the minimum tokens for that loss, buying only $S/S_{\min} = 1 + B_{\mathrm{crit}}/B \approx 1.17$ — a 14% step reduction versus the $B_{\mathrm{crit}}$ setting for roughly $3.5\times$ the tokens. A factor-of-6 error in a hyperparameter nobody sweeps is a multi-million-dollar error in a frontier run.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*