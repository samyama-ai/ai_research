---
id: 30-synthetic-data/compute-split-generation-training
title: "Optimal Compute Split Between Generation and Training"
topic: 30-synthetic-data
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Optimal Compute Split Between Generation and Training

> **Topic:** Synthetic Data · **ID:** `30-synthetic-data/compute-split-generation-training` · **Status:** empirically-open

## 1. Problem Statement

A synthetic-data pipeline spends a fixed compute budget $C$ on two activities: **generation** (sampling candidate data from one or more generator models, plus filtering/verification) and **training** (gradient steps on the surviving data). Every FLOP spent generating is a FLOP not spent training. The question: given $C$, what split $\alpha = C_{\text{gen}}/C$ minimizes final loss or maximizes downstream accuracy, and how does $\alpha^\star$ move with $C$, with generator size, with filter strictness, and with the difficulty of the target task?

Three variants, of very different difficulty:

- **Measurement.** Given a fixed pipeline, estimate $\alpha^\star$ empirically to within a resolvable confidence interval. Blocked mainly by evaluation noise, not by compute.
- **Method.** Produce a scheduler that picks $\alpha$ (and its internal knobs: samples per prompt $k$, generator size $N_g$, filter threshold $\tau$, epochs $E$) online, beating a tuned fixed split at held-out compute scales.
- **Theory.** Derive a Chinchilla-style scaling law in which generation compute enters as a first-class term, and prove where the optimum lies. No such law exists with the status of Hoffmann et al. (2022).

Solving it means: a formula $\alpha^\star(C, N_g, N_t, \tau)$ that predicts the empirical optimum at a compute scale it was not fit on, with the prediction error smaller than the loss gap between the best and worst splits.

## 2. Formal Setting

Budget accounting, in FLOPs, using the standard transformer estimates:

$$C = C_{\text{gen}} + C_{\text{ver}} + C_{\text{train}}, \qquad \alpha = \frac{C_{\text{gen}} + C_{\text{ver}}}{C}.$$

- **Generation.** $C_{\text{gen}} = 2 N_g \, k \, P \, (\ell_{\text{in}} + \ell_{\text{out}})$, where $N_g$ is generator non-embedding parameters, $P$ the number of prompts, $k$ samples per prompt, $\ell$ token counts. Measured as: wall-clock GPU-seconds × achieved FLOP/s, not the nominal $2N$ figure — batched decoding is memory-bandwidth bound and typically realizes 5–30% of peak, so the FLOP identity understates the true cost by 3–20×. **This assumption is routinely violated**; papers that report "compute-matched" arms in nominal FLOPs are not cost-matched in dollars or in GPU-hours.
- **Verification.** $C_{\text{ver}} = P k c_v$, where $c_v$ is the cost of one check. For a code/math task with an executable oracle, $c_v \approx 0$ in FLOPs but nonzero in wall-clock. For an LLM judge of size $N_j$, $c_v = 2N_j(\ell_{\text{out}} + \ell_{\text{judge}})$ and is often the dominant term.
- **Training.** $C_{\text{train}} = 6 N_t D_{\text{eff}}$, with $D_{\text{eff}} = E \cdot \rho \cdot k P \ell_{\text{out}}$ tokens, $\rho \in [0,1]$ the retention rate after filtering and dedup, $E$ epochs.
- **Yield.** Define coverage $\mathrm{cov}(k) = \mathbb{E}_p[\mathbf{1}\{\exists i \le k: \text{correct}\}]$, the fraction of prompts solved at least once in $k$ draws. Retention $\rho(k,\tau)$ is measured post-dedup at exact-match or embedding threshold; the two differ by 2–5× on math corpora.
- **Objective.** $\alpha^\star(C) = \arg\min_\alpha \mathbb{E}[\mathcal{L}(\theta(\alpha, C))]$ where $\mathcal{L}$ is held-out log-loss on human data, or $-\mathrm{acc}$ on a benchmark. The expectation is over generation seeds, data order, and init — **in practice it is estimated from one seed**, which is the central measurement failure (§6).

Assumptions known to be violated: (i) that synthetic tokens and human tokens are interchangeable in the data term $D$ — they are not, repeated and near-duplicate synthetic tokens decay in value (Muennighoff et al., 2023); (ii) that filtered samples are i.i.d. draws from a fixed distribution — filtering induces selection on the generator's own errors; (iii) that generator and trainee are independent — in self-training they are the same model, so $\alpha$ shifts the generator too.

## 3. State of the Art

**Established.**
- Chinchilla (Hoffmann et al., NeurIPS 2022): for training-only budgets, $N^\star \propto C^{0.5}$, $D^\star \propto C^{0.5}$, fit over 400+ models from 70M to 16B. This is the reference frame the generation term must be added to; it says nothing about $\alpha$.
- Sardana et al. (ICML 2024, arXiv:2401.00448) extend Chinchilla to include *inference* compute, showing optima shift to smaller-and-longer-trained models when inference demand is large. This is the closest existing formalism, but the inference there is deployment serving, not data generation feeding back into training.
- Bansal et al., *Smaller, Weaker, Yet Better* (ICLR 2025, arXiv:2408.16737): at fixed sampling FLOPs, data from a weaker/cheaper generator (Gemma2-9B) beats data from a stronger one (Gemma2-27B) for finetuning, across knowledge-distillation, self-improvement and weak-to-strong setups. Directly a statement about how to spend $C_{\text{gen}}$ — established with compute-matched arms, though matched in nominal FLOPs.

**Claimed but unablated.**
- That repeated sampling ("more $k$") is a good use of marginal compute: Brown et al., *Large Language Monkeys* (arXiv:2407.21787) show coverage rising log-linearly in $k$ to $k=10^4$, but coverage is a generation-side quantity; the transfer to post-training gain at matched budget is not ablated.
- Phi-family claims (Li et al., *Textbooks Are All You Need*, arXiv:2306.11644) that curated synthetic data buys large parameter-efficiency. Reported as benchmark numbers with an undisclosed generation budget; no compute-matched control arm exists, so it cannot be read as evidence about $\alpha$.

**Benchmark-number-only.** Nearly all industrial "synthetic data at scale" reports (open-weight post-training recipes) give final scores without disclosing $C_{\text{gen}}$. They are unusable as evidence on this problem.

## 4. What Is Known

- **Coverage scales, accuracy does not follow.** Brown et al.: on GSM8K with Llama-3-8B-Instruct, pass@1 $\approx$ 0.79 rises to pass@10 000 $\approx$ 0.99; on SWE-bench Lite, resolve rate rises 15.9% → 56% from 1 to 250 samples, but with a verifier-limited realized rate far below coverage. Scale: 8B models, $10^2$–$10^4$ samples.
- **Weaker generators win at fixed FLOPs.** Bansal et al.: finetuning Gemma-7B on Gemma2-9B data beat Gemma2-27B data by 6–8 points on MATH at matched sampling FLOPs, because the 9B model produces ~3× more samples per FLOP with higher coverage and only modestly worse false-positive rate.
- **Self-training saturates in rounds, not in data.** ReST$^{EM}$ (Singh et al., TMLR 2024, arXiv:2312.06585) on PaLM 2: gains saturate after 1–3 iterations on MATH and HumanEval, and further generation compute in later rounds buys almost nothing. This bounds how much $\alpha$ can usefully grow.
- **Repeated tokens decay.** Muennighoff et al. (NeurIPS 2023): up to ~4 epochs, repeated data is nearly as good as fresh; by 16 epochs, marginal value is near zero. Scale: up to 9B params, 900B tokens. This caps the value of the training arm when $\rho k P \ell$ is small.
- **Recursive-only training degrades; accumulation does not.** Shumailov et al. (Nature, 2024) show collapse when each generation replaces the previous corpus; Gerstgrasser et al. (COLM 2024, arXiv:2404.01413) show the collapse disappears when synthetic data is *accumulated* alongside real data. Dohmatob et al. (ICML 2024) give the corresponding change in scaling-law exponents.

## 5. What Is Not Known

- **Theoretically open.** No scaling law of the form $\mathcal{L}(C_{\text{gen}}, C_{\text{train}})$ with fitted exponents and a derived $\alpha^\star$. Not even for a toy setting (linear regression with a self-generated design matrix) is the optimal split characterized.
- **Empirically open.** The direct experiment — sweep $\alpha \in \{0.1, \dots, 0.9\}$ at three total budgets an order of magnitude apart, holding the pipeline fixed — is runnable today for under ~$10^{22}$ FLOPs and has not been published. Whether $\alpha^\star$ is scale-invariant or drifts with $C$ is unknown.
- **Methodologically blocked.** The FLOP accounting itself. Generation and training have different arithmetic intensity, so "compute-matched" is ambiguous between nominal FLOPs, GPU-hours, and dollars, and the three orderings can disagree. Until the field fixes one, $\alpha^\star$ values are not comparable across papers.

## 6. Why It Is Hard

The obstruction is **resolution, not cost**. The loss surface in $\alpha$ is flat near the optimum — plausibly under 1 point of benchmark accuracy across $\alpha \in [0.3, 0.7]$ — while the measurement noise is large. GSM8K has 1319 test items; at 80% accuracy the binomial standard error alone is 1.1 points, before seed-to-seed variance in generation and finetuning, which is typically 1–2 points more. Distinguishing $\alpha = 0.4$ from $\alpha = 0.6$ therefore needs many seeds per arm, multiplying an already expensive sweep by 5–10×.

Compounding it: **confounded knobs**. $\alpha$ is not a single dial. Raising it can mean more $k$, a bigger $N_g$, or a more expensive judge, and these have opposite signs (Bansal et al.: bigger $N_g$ hurts, bigger $k$ helps). A sweep over $\alpha$ that does not hold the internal composition fixed measures the knob, not the split. And the objective is often a benchmark that does not measure the named thing: verifier-filtered synthetic data raises benchmark scores partly by narrowing the output distribution toward the benchmark's format, which shows up as accuracy without corresponding held-out log-loss improvement.

## 7. Current Research (as of 2026)

- Inference-aware scaling laws extending Sardana et al. to training-feedback loops (*frontier — verify*); the open question is whether the generation term enters as an additive compute cost or as a modifier of the data exponent.
- Compute-optimal sampling: Google DeepMind's line following Bansal/Agarwal on weak-generator sampling, and Snell et al. (arXiv:2408.03314) on test-time compute allocation — the latter is about serving, but the allocation math is the same shape.
- Verifier economics: whether to spend the marginal FLOP on a larger generator or a larger reward model/judge. Open across academic and industrial labs (*frontier — verify*).
- Accumulation-vs-replacement corpus policy, following Gerstgrasser et al., now framed as a budget question: how much real-data training must be retained per unit of synthetic generation.

## 8. Concrete Next Experiment

**Scale.** Total budget $C = 3 \times 10^{20}$ FLOPs per arm, ~1500 H100-hours. Trainee: Llama-3.1-8B or Qwen-3-8B base. Generator: the same model (self-improvement arm) plus a fixed 3B generator (asymmetric arm). Task: MATH + GSM8K training prompts, $P = 15{,}000$, executable answer-match verification so $c_v \approx 0$.

**Design.** Seven arms, $\alpha \in \{0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 0.95\}$. Within each arm, $\alpha$ is realized *only* by varying $k$ (generator size, temperature, filter, and dedup threshold held fixed), so the split is not confounded with the generator knob. Remaining budget goes to SFT, epochs capped at 4 (Muennighoff bound). **Five seeds per arm** — the seed count is the point of the design, not an afterthought.

**Control arm.** $\alpha = 0$: spend the entire $3\times10^{20}$ FLOPs training on human data (the MATH/GSM8K solutions plus a fixed open corpus) with no generation. Any synthetic arm that does not beat this is evidence the split question is moot at this scale.

**Deciding number.** The seed-averaged held-out accuracy on a *withheld* math set (MATH500 + a fresh olympiad slice, not the tuning set), reported with a 95% CI. The question is settled at this scale if $\hat\alpha^\star$'s arm beats the $\alpha=0.5$ arm by more than $2\,\mathrm{SE}$ — roughly **1.5 accuracy points**. If no arm separates from any other by that margin, the finding is that $\alpha$ is flat over $[0.1, 0.85]$ and the field should stop tuning it, which is itself the useful result. Repeat at $3\times10^{21}$ FLOPs to test scale-invariance of $\hat\alpha^\star$.

## 9. Key References

- **[Foundational]** Hoffmann, Borgeaud, Mensch, et al. *Training Compute-Optimal Large Language Models.* NeurIPS 2022. — arXiv:2203.15556
- **[Foundational]** Kaplan, McCandlish, Henighan, et al. *Scaling Laws for Neural Language Models.* 2020. — arXiv:2001.08361
- **[SOTA]** Bansal, Hosseini, Agarwal, Tran, Kazemi. *Smaller, Weaker, Yet Better: Training LLM Reasoners via Compute-Optimal Sampling.* ICLR 2025. — arXiv:2408.16737
- **[SOTA]** Sardana, Portes, Doubov, Frankle. *Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws.* ICML 2024. — arXiv:2401.00448
- **[SOTA]** Brown, Juravsky, Ehrlich, et al. *Large Language Monkeys: Scaling Inference Compute with Repeated Sampling.* 2024. — arXiv:2407.21787
- **[SOTA]** Snell, Lee, Xu, Kumar. *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters.* 2024. — arXiv:2408.03314
- **[Method]** Singh, Co-Reyes, Agarwal, et al. *Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models.* TMLR 2024. — arXiv:2312.06585
- **[Method]** Zelikman, Wu, Mu, Goodman. *STaR: Bootstrapping Reasoning With Reasoning.* NeurIPS 2022. — arXiv:2203.14465
- **[Constraint]** Muennighoff, Rush, Barak, et al. *Scaling Data-Constrained Language Models.* NeurIPS 2023. — arXiv:2305.16264
- **[Constraint]** Shumailov, Shumaylov, Zhao, et al. *AI models collapse when trained on recursively generated data.* Nature 631, 2024.
- **[Constraint]** Gerstgrasser, Schaeffer, Dey, et al. *Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data.* COLM 2024. — arXiv:2404.01413
- **[Theory]** Dohmatob, Feng, Yang, Charton, Kempe. *A Tale of Tails: Model Collapse as a Change of Scaling Laws.* ICML 2024. — arXiv:2402.07043

## 10. Worked Example

Fix $N_g = N_t = 8\times10^9$, $P = 10^5$ prompts, $\ell_{\text{out}} = 500$ tokens, executable verifier ($C_{\text{ver}}=0$), retention $\rho = 0.2$ after correctness filtering and dedup.

Generation cost per sample: $2 N_g \ell_{\text{out}} = 2(8\times10^9)(500) = 8\times10^{12}$ FLOPs. Over all prompts:

$$C_{\text{gen}}(k) = 8\times10^{12} \cdot 10^5 \cdot k = 8\times10^{17}\,k.$$

Surviving tokens: $D_{\text{eff}}(k) = 0.2 \cdot k \cdot 10^5 \cdot 500 = 10^7 k$. One epoch of training:

$$C_{\text{train}}(k) = 6 N_t D_{\text{eff}} = 6(8\times10^9)(10^7 k) = 4.8\times10^{17}\,k.$$

So at one epoch the split is pinned by the pipeline, not chosen: $\alpha = 8/(8+4.8) = 0.62$, independent of $k$. To move $\alpha$ you must change $k$ **and** epochs jointly: at $k=100$ and $E=4$, $C_{\text{gen}} = 8\times10^{19}$, $C_{\text{train}} = 1.9\times10^{20}$, $\alpha = 0.29$, total $2.7\times10^{20}$ FLOPs.

Now the obstruction. Compare that $\alpha=0.29$ arm against $k=400, E=1$ — same total ($3.2\times10^{20}$), $\alpha = 0.62$. Both train on the same *distinct* prompt set; the first sees each token 4 times, the second sees 4× more distinct samples once. Muennighoff's result says 4 epochs costs almost nothing in value; Brown's says coverage from $k{=}100$ to $k{=}400$ rises maybe 3–5 points. The predicted accuracy difference between the two arms is therefore on the order of **1–3 points**, and the empirical standard error on GSM8K (1319 items, ~80% accuracy) is 1.1 points from sampling alone plus ~1.5 points across seeds. A single-seed comparison has roughly 50/50 power to get the ordering right. That is the problem: the two arms are cheap to run, and the answer they produce is not, at one seed, distinguishable from a coin flip.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*