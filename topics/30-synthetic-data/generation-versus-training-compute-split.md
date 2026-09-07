---
id: 30-synthetic-data/generation-versus-training-compute-split
title: "Compute-Optimal Allocation Between Generation and Training"
topic: 30-synthetic-data
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Compute-Optimal Allocation Between Generation and Training

> **Topic:** Synthetic Data · **ID:** `30-synthetic-data/generation-versus-training-compute-split` · **Status:** open

## 1. Problem Statement

Given a fixed total compute budget $C$ (FLOPs, or dollars), a synthetic-data pipeline spends it on three things: sampling candidate data from a generator, verifying/filtering those candidates, and training the student on what survives. The question is how to split $C$.

- **Measurement variant.** For a given pipeline, what is the empirical loss surface $L(C_{\text{gen}}, C_{\text{train}})$ subject to $C_{\text{gen}} + C_{\text{filter}} + C_{\text{train}} = C$? This is runnable and mostly unrun at scale.
- **Method variant.** Given a budget, choose generator size $N_g$, samples per prompt $k$, prompt-set size $P$, filter threshold, and student token count $D$ — before spending the budget, from small-scale probes.
- **Theory variant.** Is there a Chinchilla-style scaling law for synthetic pipelines, i.e. exponents $a, b$ with $C_{\text{gen}}^\star \propto C^{a}$, $C_{\text{train}}^\star \propto C^{b}$? Nothing of this form is proved.

Solving it means: a predictor that, given $C$ and a target task, outputs an allocation whose downstream loss is within a stated tolerance of the best allocation found by grid search, and that extrapolates across at least one order of magnitude in $C$.

## 2. Formal Setting

Let the generator have $N_g$ non-embedding parameters and the student $N_t$. Measured costs, using the standard counting conventions:

- **Generation.** Autoregressive decoding costs $\approx 2 N_g$ FLOPs per token (prefill of a prompt of length $\ell_p$ costs $2N_g\ell_p$). For $P$ prompts, $k$ samples each, mean completion length $\bar\ell$:
$$C_{\text{gen}} = 2 N_g P k (\ell_p + \bar\ell).$$
In practice measure it as wall-clock GPU-seconds × achieved FLOP/s, because decoding is memory-bandwidth-bound and realized utilization is often 5–25% of the training-time figure.
- **Verification.** $C_{\text{filter}} = P k \cdot c_v$, where $c_v$ is per-sample verifier cost: $\approx 0$ for a sandboxed unit test, $2N_v \ell$ for an LLM judge of size $N_v$.
- **Training.** $C_{\text{train}} = 6 N_t D$ with $D$ the number of *processed* tokens (epochs included).

Define the **yield** $\rho \in (0,1]$: the fraction of samples passing the filter, and the **true yield** $\rho^\star$: the fraction that are both passing and correct. The **verifier false-positive rate** is $\phi = 1 - \rho^\star/\rho$. Accepted unique tokens are $D_u = \rho P k \bar\ell$, and the amortized cost of one accepted token is
$$\kappa \;=\; \frac{2N_g(\ell_p/k + \bar\ell)}{\rho\,\bar\ell} \;+\; \frac{c_v}{\rho\,\bar\ell} \;+\; 6N_t .$$
The optimization is $\min L(N_t, D)$ over $\{N_g, k, P, \tau, N_t, D\}$ subject to $\kappa$-weighted budget $\le C$.

Assumptions the standard treatment makes, and their status:

1. **$L$ depends on data only through token count** — violated. Synthetic corpora have far lower entropy per token than web text; diversity, not count, is the binding constraint.
2. **Yield $\rho$ is constant in $k$** — violated. Coverage saturates: extra samples on already-solved prompts add duplicates, and pass@$k$ grows roughly log-linearly then flattens (Brown et al. 2024).
3. **The verifier is sound ($\phi = 0$)** — violated. Answer-matching on MATH admits wrong-reasoning-right-answer solutions; Bansal et al. (2024) measured this rate as materially higher for weaker generators.
4. **Repeated epochs are free of penalty** — partially violated; up to ~4 epochs repeated data is near-equivalent to fresh data, then decays (Muennighoff et al. 2023).
5. **Generator is fixed** — violated in self-improvement loops, where the student becomes the next generator and $\rho$ drifts.

## 3. State of the Art

**Established (ablated, multiple settings).**
- *Smaller, Weaker, Yet Better* (Bansal, Hu, Singh, et al., ICLR 2025). The one direct attack on this problem: at **fixed sampling FLOPs**, a weaker/cheaper generator (Gemma2-9B) beats a stronger one (Gemma2-27B) as a data source, because the price ratio buys $\approx 3\times$ more samples and higher coverage, despite higher false-positive rate. Held across knowledge-distillation, self-improvement, and weak-to-strong setups.
- *Scaling Data-Constrained Language Models* (Muennighoff et al., NeurIPS 2023). Fits a scaling law with explicit decay for repeated tokens and excess parameters; up to 4 epochs ≈ fresh data, ~16 epochs ≈ worthless. This is the only piece of the allocation problem with a fitted functional form.

**Claimed but unablated.**
- Phi-series claims (*Textbooks Are All You Need*, Gunasekar et al. 2023; phi-2/3 reports) that generated textbook data buys 5–25× parameter efficiency. Generation compute is not reported, so the claim is not a compute-normalized one; benchmark contamination was never fully excluded.
- Nemotron-4 340B (NVIDIA, 2024) reports >98% of alignment data being synthetic, but no generation-vs-training FLOP accounting.

**Benchmark-number-only.** Most "synthetic data works" results (Self-Instruct, Evol-Instruct, Cosmopedia, Magpie) report downstream benchmark deltas against a no-synthetic baseline, not against an equal-FLOPs baseline that spent the generation budget on more training instead. That control is the crux of this problem and is nearly always missing.

## 4. What Is Known

- **Coverage scales as a power law in $k$.** Brown et al. (2024), *Large Language Monkeys*: on MiniF2F/GSM8K/CodeContests, pass@$k$ (coverage) is near log-linear over 4 orders of magnitude in $k$ (1 → $10^4$); DeepSeek-Coder-V2-Instruct rises from 15.9% (1 sample) to 56% (250 samples) on SWE-bench Lite. Scale: 7B–70B generators.
- **Weak-generator advantage is measured, not universal.** Bansal et al. report relative pass@1 gains for the weak-generator arm ranging from a few percent to ~30% depending on setup, at Gemma-7B/9B/27B scale on MATH and GSM8K, at matched sampling FLOPs.
- **Test-time compute substitutes for parameters within limits.** Snell et al. (2024): optimal test-time scaling can beat a $14\times$ larger model on easy/medium problems, but the advantage inverts on the hardest bins.
- **Repetition penalty.** Muennighoff et al.: at up to 9B parameters and 900B tokens, repeating data 4× costs almost nothing; return decays to ~0 by 16 epochs.
- **Iterated self-training saturates.** ReST$^{\text{EM}}$ (Singh et al., TMLR 2024) at PaLM 2-S/M/L: gains concentrate in iterations 1–2, then overfit on train-set-derived data.
- **Collapse is an accumulation question, not a generation question.** Shumailov et al. (Nature, 2024) show degeneration when each generation *replaces* its predecessor's data; Gerstgrasser et al. (2024) show that *accumulating* real + synthetic data bounds the error instead of compounding it.

## 5. What Is Not Known

- **Theoretically open.** No scaling law of the form $C_{\text{gen}}^\star/C \to f(C)$ exists. It is unproven whether the optimal generation fraction is asymptotically constant, rising, or falling in $C$. No theorem relates generator entropy and verifier soundness to student loss.
- **Empirically open.** The equal-FLOPs control — spend $C_{\text{gen}}$ on additional real-data training instead of on sampling — has not been run at $\ge 10^{22}$ FLOPs with a swept allocation grid. Everything needed exists; only the compute has not been spent by anyone who published the surface.
- **Methodologically blocked.** "Data quality" has no measurement that predicts downstream loss across pipelines. Diversity proxies (self-BLEU, embedding dispersion, $n$-gram entropy) are not known to correlate with the loss reduction they are used to justify, so the $\rho$ vs. diversity trade-off cannot currently be priced.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement through the verifier**. The allocation optimum depends on $\rho$, but every measured $\rho$ is a verifier reading contaminated by $\phi$, and $\phi$ is itself a function of the generator you are trying to choose. Weaker generators produce more false positives; a grid search that maximizes accepted-token throughput therefore systematically over-selects weak generators, and the bias is in the same direction as the effect being reported. Separating them needs ground-truth grading of a sample of accepted data — human or formal-verifier — which exists for Lean-style targets and essentially nowhere else.

Secondary: generation FLOPs and training FLOPs are not the same currency in practice. Decoding runs at low arithmetic intensity, so a FLOP-matched comparison and a dollar-matched comparison can invert. Papers rarely say which they matched.

## 7. Current Research (as of 2026)

- Compute-optimal sampling for reasoning data: Google DeepMind / CMU line following Bansal et al.; extensions to multi-round and to code.
- Verifier-cost-aware pipelines — spending part of the budget on a better reward model or process verifier instead of more samples *(frontier — verify)*.
- Formal-verification-anchored generation (Lean/Coq autoformalization), where $\phi = 0$ by construction, making the allocation surface cleanly measurable on a narrow domain.
- Data-constrained scaling laws extended to mixed real/synthetic corpora; open-data groups (HuggingFace, AI2, EleutherAI) are the plausible venue for the full grid because they publish negative arms.
- Model-collapse theory: bounding student error under accumulation rather than replacement.

## 8. Concrete Next Experiment

**Scale.** Total budget $C = 3\times10^{21}$ FLOPs per arm, student $N_t = 1.4$B, task = MATH + GSM8K + HumanEval. Generators: 2B, 9B, 27B from one family.

**Grid.** Sweep the generation fraction $g = C_{\text{gen}}/C \in \{0, 0.1, 0.25, 0.5, 0.75, 0.9\}$, with the residual spent on training tokens (real data top-up when synthetic runs out, epochs capped at 4).

**Control arm.** $g = 0$: the same student trained on real data only for the full $C$. This is the arm the literature omits.

**Instrumentation.** Grade 500 accepted samples per generator by hand or by an independent stronger verifier to estimate $\phi$; report $\rho^\star$, not $\rho$.

**Deciding number.** The location of $\arg\min_g L_{\text{val}}(g)$ and whether it moves when $C$ is raised $10\times$ to $3\times10^{22}$. If $g^\star$ is stable within $\pm0.1$ across the decade, an allocation law exists and can be fit; if $g^\star$ shifts monotonically, the constant-fraction heuristics used in current pipelines are wrong and the exponent must be measured. Secondary decider: whether the $\phi$-corrected ranking of generators matches the uncorrected one.

## 9. Key References

- **[Foundational]** Hoffmann et al. *Training Compute-Optimal Large Language Models.* NeurIPS 2022. — arXiv:2203.15556
- **[Foundational]** Kaplan et al. *Scaling Laws for Neural Language Models.* 2020. — arXiv:2001.08361
- **[SOTA]** Bansal, Hu, Singh, et al. *Smaller, Weaker, Yet Better: Training LLM Reasoners via Compute-Optimal Sampling.* ICLR 2025. — arXiv:2408.16737
- **[SOTA]** Muennighoff et al. *Scaling Data-Constrained Language Models.* NeurIPS 2023. — arXiv:2305.16264
- **[SOTA]** Brown et al. *Large Language Monkeys: Scaling Inference Compute with Repeated Sampling.* 2024. — arXiv:2407.21787
- **[SOTA]** Snell, Lee, Xu, Kumar. *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters.* 2024. — arXiv:2408.03314
- **[Related]** Singh et al. *Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models.* TMLR 2024. — arXiv:2312.06585
- **[Related]** Zelikman et al. *STaR: Bootstrapping Reasoning With Reasoning.* NeurIPS 2022. — arXiv:2203.14465
- **[Related]** Gunasekar et al. *Textbooks Are All You Need.* 2023. — arXiv:2306.11644
- **[Related]** Shumailov et al. *AI models collapse when trained on recursively generated data.* Nature 631, 2024.
- **[Related]** Gerstgrasser et al. *Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data.* COLM 2024. — arXiv:2404.01413

## 10. Worked Example

Target: a 1B student, Chinchilla-optimal at $D = 20$B tokens. Training cost $6 \times 10^9 \times 2\times10^{10} = 1.2\times10^{20}$ FLOPs.

Two generator choices, ignoring prompt prefill, $\bar\ell = 500$:

| | 8B generator | 1B generator |
|---|---|---|
| FLOPs per emitted token $2N_g$ | $1.6\times10^{10}$ | $2\times10^{9}$ |
| Measured yield $\rho$ | 0.20 | 0.05 |
| Gen FLOPs per accepted token | $8.0\times10^{10}$ | $4.0\times10^{10}$ |
| Gen FLOPs for 20B accepted tokens | $1.6\times10^{21}$ | $8.0\times10^{20}$ |
| Generation share of total | 93% | 87% |

Two things fall out. First, the split is not near 50/50 — generation eats 87–93% of the budget, so the entire optimization is really about generation, and the practice of reporting only $6N_tD$ understates pipeline cost by more than an order of magnitude. Second, the 1B generator looks $2\times$ cheaper per accepted token, matching the Bansal et al. direction.

Now make the obstruction visible. Suppose ground-truth grading shows $\phi = 0.15$ for the 8B generator and $\phi = 0.45$ for the 1B. True yields are $\rho^\star = 0.17$ and $0.0275$. Cost per *correct* accepted token becomes $9.4\times10^{10}$ (8B) versus $7.3\times10^{10}$ (1B) — the gap narrows from $2.0\times$ to $1.3\times$. Push $\phi$ for the weak generator to 0.6 and the ranking inverts: $1.0\times10^{11}$ versus $9.4\times10^{10}$, and the strong generator wins.

The conclusion is not "weak generators are better." It is that the published conclusion is a function of an unmeasured quantity, $\phi$, that nobody reports and that moves the answer across the decision boundary within its plausible range. Until $\phi$ is measured per generator, the allocation problem is not merely unsolved — the measurement that would solve it is not being taken.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*