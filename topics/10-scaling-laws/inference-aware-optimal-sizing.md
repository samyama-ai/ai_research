---
id: 10-scaling-laws/inference-aware-optimal-sizing
title: "Inference-Aware Compute-Optimal Sizing"
topic: 10-scaling-laws
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Inference-Aware Compute-Optimal Sizing

> **Topic:** Scaling Laws & Compute Allocation · **ID:** `10-scaling-laws/inference-aware-optimal-sizing` · **Status:** empirically-open

## 1. Problem Statement

Chinchilla-style sizing minimises pretraining loss at a fixed *training* compute budget. Deployed models are paid for twice: once to train, then once per served token, for the life of the model. The problem is to choose $(N, D)$ — parameters and training tokens — that minimises **total lifetime cost** at fixed delivered quality, given an inference demand that is not known when the decision is made.

Three variants, of different difficulty:

- **Measurement.** What is the true cost per served token as a function of $N$ and architecture, on a real serving stack? FLOPs is a proxy that is known to be wrong for autoregressive decode. Currently ill-posed as usually stated.
- **Method.** Given a demand forecast $D_{\text{inf}}$ and a cost model, compute $(N^\star, D^\star)$. Solved as an optimisation; its inputs are the disputed part.
- **Theory.** Does an iso-loss contour of $L(N,D)$ correspond to an iso-*capability* contour? If not, "fixed delivered quality" is undefined and the whole allocation argument rests on a substitution that has never been validated.

Solving it means: a sizing rule that, prospectively, predicts the measured lifetime-cost crossover point in $D_{\text{inf}}$ within a factor of two on a fixed serving stack.

## 2. Formal Setting

Let $N$ be non-embedding parameters, $D$ pretraining tokens, and adopt the Chinchilla parametric loss

$$L(N,D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}},$$

measured as mean next-token cross-entropy in nats on a held-out sample of the *training* distribution (not a downstream benchmark).

**Training cost.** $C_{\text{train}} \approx 6ND$ FLOPs — measured as wall-clock GPU-hours times achieved FLOP/s, not the nominal count.

**Inference cost.** With $D_{\text{pre}}$ prompt tokens and $D_{\text{gen}}$ generated tokens over the model's life, the arithmetic model is $C_{\text{inf}} \approx 2N(D_{\text{pre}} + D_{\text{gen}})$. What is actually paid is

$$\mathcal{C}_{\text{serve}} = \int \big(p_{\text{gpu}} \cdot t_{\text{token}}\big),\quad t_{\text{token}} \approx \max\!\left(\frac{b_w N \cdot s}{\text{BW}},\ \frac{2NB}{\text{FLOP/s}}\right),$$

with $b_w$ bytes per weight (quantisation), $s$ the fraction of weights read per token (1 for dense, $<1$ for MoE), $B$ the achieved batch size, and BW the HBM bandwidth. Decode at small $B$ is memory-bandwidth-bound, so cost scales with *bytes moved*, not FLOPs.

**Decision problem.**
$$\min_{N,D} \; 6ND + 2N D_{\text{inf}} \quad \text{s.t.} \quad L(N,D) \le \ell .$$
The constraint binds, so $D(N)$ is determined by $\ell$ and the problem is one-dimensional in $N$; the solution moves to smaller $N$ and larger $D$ monotonically in $D_{\text{inf}}$.

**Assumptions, and which are violated.**
1. *FLOPs are the cost currency.* Violated: decode is bandwidth-bound below the roofline knee; quantisation changes $b_w$ by $4\times$ without changing FLOPs.
2. *Iso-loss implies iso-capability.* Not established; violated at least for reasoning tasks where sampling budget substitutes for parameters.
3. *$L(N,D)$ extrapolates far off the fitted ratio.* Chinchilla fits cover roughly $D/N \in [5, 100]$; production models run at $D/N > 1000$.
4. *$D_{\text{inf}}$ is known.* It is a forecast, usually wrong by an order of magnitude.
5. *One epoch, no distillation, no test-time search.* All three are standard in 2026 practice.

## 3. State of the Art

**Established.**
- Hoffmann et al. (NeurIPS 2022) — training-only optimum, $N \propto C^{0.46}$, $D \propto C^{0.54}$, giving $\approx 20$ tokens/parameter. Reproduced qualitatively many times.
- Sardana et al. (ICML 2024), *Beyond Chinchilla-Optimal* — the first explicit inference-aware formulation, with the modified optimum above. The *direction* (smaller, longer-trained models as $D_{\text{inf}}$ grows) is robust and follows from the constrained optimisation; the *magnitude* is a consequence of the FLOP cost model, not a measurement.
- Gadre et al. (ICLR 2025) — loss remains predictable at up to $32\times$ the Chinchilla token ratio, and average downstream top-1 error is an approximately exponential function of validation loss over their model suite.

**Claimed but unablated.**
- That inference-aware sizing yields the reported dollar savings. Sardana et al. report FLOP savings; no published study takes the two arms to a production serving stack and measures $/1M tokens at matched quality.
- That test-time compute substitutes for parameters. Snell et al. (2024) report a smaller model plus PRM-guided search beating a $\sim 14\times$ larger model at matched FLOPs on easy/medium MATH problems, and *failing* on the hardest bucket. This is a benchmark number on one task family with an oracle-quality verifier, not an ablated law.

**Benchmark-number-only.** Brown et al. (2024) report DeepSeek-Coder-V2-Instruct rising from 15.9% to 56% on SWE-bench Lite between 1 and 250 samples. Coverage, not pass@1; it needs a verifier to cash out.

## 4. What Is Known

- **The training optimum.** $\approx 20$ tokens/param at Chinchilla scales ($70$M–$16$B, up to $\sim 10^{24}$ FLOPs).
- **The fit is shakier than reported.** Besiroglu et al. (2024) refit Hoffmann's Approach 3 data and recover exponents near $\alpha \approx 0.35$, $\beta \approx 0.37$, with confidence intervals far wider than the original; the published parametric fit is not self-consistent with the published 20:1 rule.
- **Over-training is cheap in loss and expensive in tokens.** Llama-3-8B: 15T tokens $\approx 1875$ tokens/param, roughly $90\times$ Chinchilla — an industry revealed preference for inference-aware sizing, taken without a public cost model.
- **Quantisation moves the cost curve without moving FLOPs.** Dettmers & Zettlemoyer (ICML 2023) find 4-bit weights near-Pareto-optimal for zero-shot accuracy per bit across 19M–176B parameters; Kumar et al. (ICLR 2025) show post-training-quantisation degradation *grows* with training tokens per parameter — directly penalising the over-trained small models that inference-aware sizing recommends.
- **Sparsity changes the cost axes.** Abnar et al. (2025) fit optimal MoE sparsity as a function of budget; DeepSeek-V3 (2024) serves 671B total / 37B active parameters — FLOP cost tracks 37B, memory-bandwidth and capacity cost track 671B.

## 5. What Is Not Known

- **Methodologically blocked:** whether iso-loss is iso-capability. Downstream error is a smooth function of loss *on average over benchmarks*, but no one has shown that two models at equal loss and unequal $N$ need equal inference tokens to reach equal task success. Until that is measured, "fixed quality" is not a well-defined constraint.
- **Methodologically blocked:** the serving cost currency. There is no agreed measured cost function $\mathcal{C}(N, \text{arch}, \text{stack})$; papers report FLOPs, vendors report $/1M tokens, and the two disagree by more than the effect being estimated.
- **Empirically open:** the prospective test. Train two models to the same loss at $N$ and $N/2$, serve both, measure the crossover $D_{\text{inf}}^\star$. Runnable at 1–8B for a few hundred thousand dollars. Not published.
- **Theoretically open:** the joint optimum over $(N, D, \text{sparsity}, \text{precision}, k_{\text{samples}})$. Each axis has a separate scaling law; no result gives conditions under which they compose, or a counterexample where they do not.

## 6. Why It Is Hard

The obstruction is **confounded measurement plus non-identifiability of the objective**.

- The quality constraint is measured in one currency (cross-entropy) and the decision is paid in another (task success per dollar). The map between them is fitted on benchmark averages and is not known to be $N$-invariant — exactly the invariance the sizing argument needs.
- The cost model is stack-dependent and the stack changes after the training decision is irreversible. A model sized under a bf16, batch-32 assumption is served two years later at 4-bit with speculative decoding and 10x the batch — a $6$–$10\times$ shift in cost per token, larger than the $N$ effect being optimised.
- $D_{\text{inf}}$ enters the objective linearly and is unknown to an order of magnitude. The optimiser's answer is more sensitive to this forecast than to the scaling-law exponents, so improving the exponents does not improve the decision.
- Settling it needs *pairs* of models trained to matched loss at different $N$ — several full pretraining runs per data point, at scales large enough that bandwidth-bound decode behaves as it does in production.

## 7. Current Research (as of 2026)

- **Inference-aware and over-training laws.** Databricks/MosaicML (Sardana, Frankle) and the Gadre et al. group (UW/TTIC/Apple) on over-trained scaling and downstream prediction.
- **Test-time compute allocation.** Berkeley/Google DeepMind (Snell, Kumar) and CMU (Wu et al., *Inference Scaling Laws*, 2024) on the compute-optimal split between model size and sampling budget.
- **Distillation as a sizing tool.** Busbridge et al. (Apple, 2025), *Distillation Scaling Laws* — gives conditions under which distilling into a small serving model beats pretraining it directly; a direct competitor to shrinking $N$ under the Chinchilla constraint.
- **Sparsity/precision co-optimisation.** Abnar et al. (Apple) on optimal MoE sparsity; Kumar et al. (Harvard/Stanford) on precision-aware laws. *(frontier — verify)* Reports that frontier labs now size against an internal serving-cost model rather than a FLOP model are credible but not documented publicly.

## 8. Concrete Next Experiment

**Scale.** Four models on one fixed corpus, matched to a single iso-loss target $\ell$: $N \in \{1.4\text{B}, 2.8\text{B}, 5.6\text{B}, 11\text{B}\}$, with $D$ set by solving $L(N,D)=\ell$ (roughly 1.5T down to 200B tokens). Total $\approx 3\times10^{22}$ training FLOPs — order 100k H100-hours.

**Control arm.** The Chinchilla-optimal member of the ladder (the $(N,D)$ pair minimising $6ND$ at $L=\ell$), served on the identical stack: vLLM, H100, bf16 and 4-bit variants, 1024-token prompt / 256-token generation, batch swept 1–256.

**Measurements.** (a) $/1M served tokens at $p99$ latency $\le 100$ ms/token; (b) task success on a held-out suite at fixed sampling budget, and the sampling budget $k(N)$ each model needs to hit a common success threshold.

**The deciding number.** The measured lifetime-cost crossover $\hat{D}_{\text{inf}}^\star$ — the served-token volume at which the small-and-over-trained arm becomes cheaper than the control — divided by the FLOP-model prediction $D_{\text{inf}}^\star$. If $\hat{D}^\star/D^\star \in [0.5, 2]$, the FLOP formulation is adequate and the problem reduces to demand forecasting. If it falls outside — the likely outcome if $k(N)$ rises as $N$ falls — the constraint must be restated in capability terms and the current literature's recommendations do not transfer.

## 9. Key References

- **[Foundational]** J. Kaplan, S. McCandlish, T. Henighan, et al. *Scaling Laws for Neural Language Models.* 2020. — arXiv:2001.08361
- **[Foundational]** J. Hoffmann, S. Borgeaud, A. Mensch, et al. *Training Compute-Optimal Large Language Models.* NeurIPS, 2022. — arXiv:2203.15556
- **[SOTA]** N. Sardana, J. Portes, S. Doubov, J. Frankle. *Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws.* ICML, 2024. — arXiv:2401.00448
- **[SOTA]** S. Y. Gadre, G. Smyrnis, V. Shankar, et al. *Language Models Scale Reliably with Over-Training and on Downstream Tasks.* ICLR, 2025. — arXiv:2403.08540
- **[SOTA]** C. Snell, J. Lee, K. Xu, A. Kumar. *Scaling LLM Test-Time Compute Optimally Can Be More Effective Than Scaling Model Parameters.* 2024. — arXiv:2408.03314
- **[SOTA]** T. Dettmers, L. Zettlemoyer. *The Case for 4-bit Precision: k-bit Inference Scaling Laws.* ICML, 2023. — arXiv:2212.09720
- **[SOTA]** T. Kumar, Z. Ankner, B. F. Spector, et al. *Scaling Laws for Precision.* ICLR, 2025. — arXiv:2411.04330
- **[SOTA]** D. Busbridge, A. Shidani, F. Weers, et al. *Distillation Scaling Laws.* 2025. — arXiv:2502.08606
- **[Replication]** T. Besiroglu, E. Erdil, M. Barnett, J. You. *Chinchilla Scaling: A Replication Attempt.* 2024. — arXiv:2404.10102
- **[Systems]** R. Pope, S. Douglas, A. Chowdhery, et al. *Efficiently Scaling Transformer Inference.* MLSys, 2023. — arXiv:2211.05102
- **[Empirical]** B. Brown, J. Juravsky, R. Ehrlich, et al. *Large Language Monkeys: Scaling Inference Compute with Repeated Sampling.* 2024. — arXiv:2407.21787

## 10. Worked Example

Use the Chinchilla Approach-3 fit: $E=1.69$, $A=406.4$, $\alpha=0.34$, $B=410.7$, $\beta=0.28$.

**Arm A (control, 20:1).** $N=7\times10^9$, $D=1.4\times10^{11}$.
$A/N^{\alpha} = 406.4/2226 = 0.183$; $B/D^{\beta} = 410.7/1319 = 0.311$. So $L_A = 2.184$.

**Arm B (inference-aware, half the parameters).** $N=3.5\times10^9$. Then $A/N^\alpha = 0.231$, so matching $L=2.184$ requires $B/D^\beta = 0.263$, i.e. $D = 2.55\times10^{11}$ — 73 tokens/param.

**Costs.** $C_{\text{train}}^A = 6\cdot 7\text{e}9\cdot 1.4\text{e}11 = 5.88\times10^{21}$; $C_{\text{train}}^B = 5.36\times10^{21}$. Per served token, $2N$: $1.4\times10^{10}$ vs $7.0\times10^{9}$. At $D_{\text{inf}}=10^{12}$ tokens: A $= 1.99\times10^{22}$, B $= 1.24\times10^{22}$ — a **38% saving**, and B is cheaper to train too.

**Where it breaks.** Two places, both invisible in the arithmetic.

1. The optimality condition at the fixed-compute optimum is $\alpha A N^{-\alpha} = \beta B D^{-\beta}$. At arm A that reads $0.34\times0.183 = 0.062$ against $0.28\times0.311 = 0.087$. They are not equal, so the published parametric fit says the published 20:1 control point is *already* under-trained — the baseline the field compares against is not a stationary point of the law it comes from.
2. Suppose the served task needs $k=2$ samples with majority voting for arm B to match arm A's accuracy, despite equal cross-entropy. B's serving cost doubles to $1.4\times10^{10}$ FLOPs/token, and the 38% saving becomes a 4% loss.

The decision therefore turns on $k(N)$, a quantity the scaling law does not contain and that no published experiment measures at matched loss. That is the obstruction, not the algebra.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*