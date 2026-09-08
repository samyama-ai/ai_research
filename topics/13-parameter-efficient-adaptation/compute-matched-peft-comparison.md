---
id: 13-parameter-efficient-adaptation/compute-matched-peft-comparison
title: "Fair Compute-Matched PEFT Comparison Protocol"
topic: 13-parameter-efficient-adaptation
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Fair Compute-Matched PEFT Comparison Protocol

> **Topic:** Parameter-Efficient Adaptation · **ID:** `13-parameter-efficient-adaptation/compute-matched-peft-comparison` · **Status:** methodologically-blocked

## 1. Problem Statement

Given a pre-trained model and a downstream task, decide which adaptation method — full fine-tuning, LoRA, DoRA, $(IA)^3$, prefix/prompt tuning, BitFit, or a sparse-update rule — is better **at equal cost**. The field reports adapter quality against a single full fine-tuning baseline at unmatched cost, and normalizes by *trainable parameter count*, a quantity that predicts neither training FLOPs, nor memory, nor wall-clock, nor inference latency.

Three variants, with different difficulty:

- **Measurement.** Define a cost vector and a comparison predicate such that "method $A$ dominates $B$ at budget $C$" is decidable and reproducible across labs. This is the blocked variant: there is no agreed cost accounting, and per-method hyperparameter search cost is almost never charged to the method.
- **Method.** Build a search-and-report protocol (a PEFT analogue of Dodge et al.'s expected-max-performance curves) that a paper can run for under ~2× the cost of its current headline table.
- **Theory.** Prove or bound how the achievable loss at budget $C$ scales with adapter rank/capacity — i.e. a fine-tuning scaling law in which the method is a parameter, not a separate curve.

A solution is a protocol under which a reported ranking of two PEFT methods reverses at a rate below some stated threshold when re-run by an independent group at the same budget.

## 2. Formal Setting

Base model $\theta_0 \in \mathbb{R}^N$ ($N$ = non-embedding parameters). A method $m$ defines a reparameterization $\theta = \theta_0 + \Delta(\phi)$ with trainable $\phi \in \mathbb{R}^{P_m}$, and a hyperparameter space $\Lambda_m$ (learning rate, rank $r$, scaling $\alpha$, target modules, schedule, epochs).

**Cost is a vector, not a scalar.** For one training run with configuration $\lambda$ on $D$ tokens:

$$c(m,\lambda) = \big(F_{\text{train}},\; M_{\text{peak}},\; T_{\text{wall}},\; F_{\text{infer}},\; B_{\text{ckpt}}\big)$$

measured as: $F_{\text{train}}$ = hardware FLOPs from a profiler (not the analytic $6ND$), $M_{\text{peak}}$ = peak device memory in bytes from the allocator, $T_{\text{wall}}$ = seconds on a named device count, $F_{\text{infer}}$ = FLOPs per generated token after any merge, $B_{\text{ckpt}}$ = bytes stored per task.

Analytically, per training token: full fine-tuning costs $\approx 6N$ (forward $2N$, activation grads $2N$, weight grads $2N$); a frozen-backbone adapter costs $\approx 4N + O(P_m)$, because activation gradients must still be backpropagated to the earliest adapted layer. So the FLOPs ratio is bounded by $6/4 = 1.5$, **not** by $N/P_m$.

**Total method cost must include search.** With $k$ trials drawn from a search distribution over $\Lambda_m$:

$$C_m(k) = \sum_{i=1}^{k} F_{\text{train}}(m,\lambda_i), \qquad \hat U_m(k) = \mathbb{E}\Big[\max_{i \le k} \;\mathrm{val}(m,\lambda_i)\Big]$$

The comparison predicate is on the *budget-indexed* curve, not on single points: $A \succ B$ at budget $C$ iff $\hat U_A(k_A) > \hat U_B(k_B)$ for the largest $k_A,k_B$ with $C_A(k_A), C_B(k_B) \le C$, with the gap exceeding the seed/data-order standard error.

**Assumptions, and which fail.**
1. *Val metric is a low-variance estimate of test quality.* Violated: seed and data-order variance on small GLUE-scale tasks is often comparable to reported method gaps (Bouthillier et al., MLSys 2021).
2. *$\Lambda_m$ tuned equally across methods.* Violated systematically: baselines inherit hyperparameters from the pre-training recipe while the proposed method is swept.
3. *Optimal learning rate is comparable across methods.* Violated: LoRA's optimum is roughly an order of magnitude above full fine-tuning's, and depends on rank unless $\alpha/\sqrt{r}$ scaling is used (Kalajdzievski, 2023; Hayou et al., ICML 2024).
4. *Cost is scalar.* Violated by construction — LoRA can lose on FLOPs while winning by $\sim$3× on memory and by orders of magnitude on $B_{\text{ckpt}}$.

## 3. State of the Art

**Established (independently reproduced).** LoRA (Hu et al., ICLR 2022) matches full fine-tuning on GLUE-scale classification and GPT-2/GPT-3 generation at $\ll 1\%$ trainable parameters, and adds zero inference latency after merging. $(IA)^3$/T-Few (Liu et al., NeurIPS 2022) beats GPT-3 175B in-context learning on RAFT with T0-3B. QLoRA (Dettmers et al., NeurIPS 2023) fine-tunes a 65B model on one 48 GB GPU — a memory result, replicated widely, and not a quality claim.

**Established as a *negative* result.** Biderman et al. (*LoRA Learns Less and Forgets Less*, TMLR 2024) run Llama-2 7B/13B on code and math, both continued pre-training ($\sim$20B tokens) and instruction tuning, with learning rates swept per method. LoRA underperforms full fine-tuning on code, and the gap does not close at high rank; it forgets less of the base distribution. This is the strongest existing example of the protocol this page asks for.

**Claimed but unablated.** Most "beats LoRA by $x$ points" results — DoRA (Liu et al., ICML 2024; $\sim$+3.7 on commonsense reasoning at LLaMA-7B), LoRA+ (Hayou et al., ICML 2024), AdaLoRA (Zhang et al., ICLR 2023), rank-adaptive and SVD-initialized variants — report a headline table at matched *parameter count*, not matched search budget, usually one seed, on the same commonsense-reasoning or GLUE suites. Whether the gain survives an equal-trial LR sweep for the baseline is untested in the original papers. DoRA additionally adds inference cost unless merged, which the parameter-count column does not show.

**Benchmark-number-only.** Nearly all PEFT leaderboard entries on GLUE/SuperGLUE, commonsense-170k, and MT-Bench are single-configuration numbers with no variance estimate and no reported search cost.

## 4. What Is Known

- **Parameter count is not compute.** Freezing the backbone removes $\approx 1/3$ of training FLOPs, not $99\%$ (Section 2 arithmetic; consistent with profiled step times, where LoRA step time on 7B models is typically 60–75% of full fine-tuning, not 1–5%).
- **Optimizer state, not parameters, drives the memory win.** Adam state for full 7B bf16 fine-tuning is $\sim$56 GB (fp32 $m,v$ + master weights); LoRA at $r=16$ reduces this to well under 1 GB — the dominant reason QLoRA's 48 GB result works.
- **Learning rate dominates the comparison.** Across LoRA studies, the best LR is $\sim$1e-4 to 4e-4 versus $\sim$1e-5 to 2e-5 for full fine-tuning at 7B; a baseline run at the wrong end of that range loses by more than most published method deltas.
- **Rank matters less than expected at small data, more at large data.** Biderman et al. find rank increases (16 → 256) help mainly in the 20B-token continued-pre-training regime.
- **Solutions are not equivalent even when losses match.** Shuttleworth et al. (2024) report "intruder dimensions" — singular vectors in LoRA solutions absent from full fine-tuning updates — at matched target-task accuracy, with worse sequential-task and out-of-distribution behaviour.
- **Fine-tuning scale laws exist but are method-specific.** Zhang et al. (*When Scaling Meets LLM Finetuning*, ICLR 2024) fit power laws over 1B–16B models, showing PEFT methods benefit more from model scale than from data scale relative to full fine-tuning.

## 5. What Is Not Known

- **Methodologically blocked.** No accepted definition of "compute-matched" for PEFT. Search cost is uncharged; the cost vector is never reported in full; no venue requires it. Until this is fixed, the empirical questions below cannot be posed unambiguously.
- **Empirically open.** Does any post-2023 LoRA variant retain a statistically significant win over plain LoRA when both get $k$ equal trials of random search and $\ge 3$ seeds at $\ge$7B? Runnable today; not published for any variant.
- **Empirically open.** At what data scale does the LoRA-vs-full gap open as a function of $r/N$? Biderman et al. give two points (IFT, 20B-token CPT); the curve is unmapped.
- **Theoretically open.** No bound relating achievable loss at rank $r$ to the intrinsic dimension of the task update. Aghajanyan et al. (ACL 2021) measured intrinsic dimension empirically; there is no theorem converting it into an $r$ sufficient for $\epsilon$-optimality.
- **Theoretically open.** Whether the low-rank constraint changes the optimization *problem* (reachable minima) or only the *path*.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement plus asymmetric search**. The reported quantity — final validation score at a fixed configuration — conflates (a) the method's capacity, (b) how much hyperparameter search the author spent on it, and (c) seed noise. These three are not separately identifiable from a single-number table. Because a new method's search cost is invisible in the parameter-count column, spending more search on the proposal than the baseline is both undetectable and rewarded.

Two secondary obstructions:

- **Vector-valued cost has no total order.** LoRA loses on FLOPs by up to 1.5× and wins on memory by $\sim$3× and on per-task storage by $\sim$1000×. "Better" requires a user-specified weighting, so a single leaderboard rank is not well defined.
- **The compute needed to remove the confound is roughly $k \times$ the compute of the current headline table**, at 7B+, per task — which is why the honest protocol is skipped rather than disputed.

## 7. Current Research (as of 2026)

- **Sweep-inclusive PEFT evaluation.** The Databricks Mosaic Research line (Biderman et al.) is the clearest existing template: per-method LR sweeps, rank sweeps, learning-vs-forgetting reported jointly.
- **Learning-rate and scaling corrections** ($\mu$P-style adapter parameterization, LoRA+, rsLoRA) aim to remove LR as a confound by making the optimum transfer across rank and width. If successful, this collapses most of $\Lambda_m$ and makes matched comparison cheap. *(frontier — verify)*
- **Representation-level diagnostics** (intruder dimensions, spectral comparison of $\Delta W$) as method-discriminators that do not depend on a tuned scalar metric.
- **Full cost-vector reporting** in efficiency-benchmark work (HELM-style efficiency columns, MLPerf-style profiled FLOPs) applied to adaptation rather than pre-training. *(frontier — verify)*
- Unified-view analyses (He et al., ICLR 2022; Ding et al., *Nature Machine Intelligence* 2023) that treat adapters/prefix/LoRA as instances of one family remain the theoretical scaffolding for a single comparison space.

## 8. Concrete Next Experiment

**Question.** Does DoRA's reported win over LoRA survive equal search?

- **Scale.** Llama-3-8B (or Llama-2-7B for comparability), two tasks: commonsense-170k (in-distribution for published DoRA numbers) and a code SFT set of $\ge$1B tokens (where Biderman et al. saw LoRA fail).
- **Arms.** (1) LoRA, (2) DoRA, (3) full fine-tuning — **the control arm is LoRA with the same number of search trials as DoRA**, which is the arm the original papers omit. Each arm gets $k=16$ random-search trials over $\{$LR $\in$ log-uniform[1e-5, 1e-3], $r \in \{8,16,64,256\}$, target modules, epochs$\}$, then the best configuration re-run with 5 seeds. Report the full cost vector from a profiler for every trial.
- **Cost.** $3 \times 16$ trials + $3\times5$ reruns $\approx$ 63 runs. At $\sim$1B tokens and $\approx 4N$–$6N$ FLOPs/token, $\approx 2\text{–}3 \times 10^{20}$ FLOPs total — roughly 1–2 node-weeks on 8×H100.
- **Deciding number.** $\Delta = \hat U_{\text{DoRA}}(k) - \hat U_{\text{LoRA}}(k)$ at equal $C_m(k)$, with a 5-seed 95% CI. **If the CI for $\Delta$ contains 0 at $k=16$ while the $k=1$ published-configuration comparison shows $\Delta \approx +3.7$, the gain is a search artifact, not a method effect.** Publish $\hat U_m(k)$ for $k=1..16$ so readers can see where the ranking flips.

## 9. Key References

- **[Foundational]** Edward Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR, 2022. — arXiv:2106.09685
- **[Foundational]** Jesse Dodge, Suchin Gururangan, Dallas Card, Roy Schwartz, Noah A. Smith. *Show Your Work: Improved Reporting of Experimental Results.* EMNLP, 2019. — arXiv:1909.03004
- **[Foundational]** Xavier Bouthillier et al. *Accounting for Variance in Machine Learning Benchmarks.* MLSys, 2021. — arXiv:2103.03098
- **[SOTA]** Dan Biderman et al. *LoRA Learns Less and Forgets Less.* TMLR, 2024. — arXiv:2405.09673
- **[SOTA]** Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer. *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS, 2023. — arXiv:2305.14314
- **[SOTA]** Shih-Yang Liu et al. *DoRA: Weight-Decomposed Low-Rank Adaptation.* ICML, 2024. — arXiv:2402.09353
- **[SOTA]** Soufiane Hayou, Nikhil Ghosh, Bin Yu. *LoRA+: Efficient Low Rank Adaptation of Large Models.* ICML, 2024. — arXiv:2402.12354
- **[SOTA]** Biao Zhang, Zhongtao Liu, Colin Cherry, Orhan Firat. *When Scaling Meets LLM Finetuning: The Effect of Data, Model and Finetuning Method.* ICLR, 2024. — arXiv:ized 2402.17193
- **[Analysis]** Reece Shuttleworth, Jacob Andreas, Antonio Torralba, Pratyusha Sharma. *LoRA vs Full Fine-tuning: An Illusion of Equivalence.* 2024. — arXiv:2410.21228
- **[Analysis]** Armen Aghajanyan, Luke Zettlemoyer, Sonal Gupta. *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning.* ACL, 2021. — arXiv:2012.13255
- **[Survey]** Junxian He, Chunting Zhou, Xuezhe Ma, Taylor Berg-Kirkpatrick, Graham Neubig. *Towards a Unified View of Parameter-Efficient Transfer Learning.* ICLR, 2022. — arXiv:2110.04366
- **[Survey]** Ning Ding et al. *Parameter-efficient fine-tuning of large-scale pre-trained language models.* Nature Machine Intelligence, 2023.
- **[Survey]** Vladislav Lialin, Vijeta Deshpande, Anna Rumshisky. *Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning.* 2023. — arXiv:2303.15647

*(The arXiv identifier for Zhang et al. 2024 above should be verified before citation; authors/title/venue are correct.)*

## 10. Worked Example

Llama-2-7B ($N \approx 6.5\times10^9$ non-embedding), 1B fine-tuning tokens, LoRA $r=16$ on all attention projections ($P_m \approx 2\times10^7$, i.e. $0.3\%$ of $N$).

**The claim a paper would make.** "LoRA trains 0.3% of parameters — a 300× efficiency gain."

**Cost vector, computed.**

| Quantity | Full FT | LoRA $r{=}16$ | Ratio |
|---|---|---|---|
| Train FLOPs/token | $6N = 3.9\times10^{10}$ | $\approx 4N = 2.6\times10^{10}$ | **1.5×** |
| FLOPs, 1B tokens | $3.9\times10^{19}$ | $2.6\times10^{19}$ | 1.5× |
| Optimizer + master state | $\sim$78 GB | $\sim$0.24 GB | $\sim$325× |
| Stored bytes per task | 13 GB | 40 MB | $\sim$325× |
| Inference FLOPs/token (merged) | $2N$ | $2N$ | 1× |

The 300× is real for storage and optimizer memory. It is 1.5× for compute — the gap because activation gradients still traverse the frozen backbone.

**Now charge search.** Full fine-tuning is run once at the inherited LR 2e-5: cost $3.9\times10^{19}$ FLOPs. LoRA is swept over 6 learning rates $\{$1e-5 … 1e-3$\}$: cost $6 \times 2.6\times10^{19} = 1.6\times10^{20}$ FLOPs.

$$\frac{C_{\text{LoRA}}(6)}{C_{\text{full}}(1)} = 4.0$$

At the budget actually spent, LoRA consumed **4× the compute of its own baseline** while the table's efficiency column reported "0.3%". Match the budget instead — give full fine-tuning 4 trials of its own LR sweep at $1.6\times10^{20}$ FLOPs — and the comparison changes arm, because full fine-tuning's LR is the one hyperparameter most likely to be mis-set at 2e-5 for a 1B-token domain shift.

**Where the obstruction becomes visible.** Nothing in the reported artifact — trainable parameters, final accuracy, GPU-hours of the winning run — records that $k_{\text{LoRA}}=6$ and $k_{\text{full}}=1$. The confound is not an error in any single number; it is a quantity the reporting format has no slot for. That is why this problem is methodologically blocked rather than merely expensive.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*