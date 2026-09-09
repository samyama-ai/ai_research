---
id: 32-hardware-and-kernels/hardware-lottery-measurement
title: "Hardware Lottery Measurement for Novel Architectures"
topic: 32-hardware-and-kernels
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Hardware Lottery Measurement for Novel Architectures

> **Topic:** Hardware & Kernels · **ID:** `32-hardware-and-kernels/hardware-lottery-measurement` · **Status:** methodologically-blocked

## 1. Problem Statement

Sara Hooker's *hardware lottery* (CACM 2021) says a research idea wins because it fits the available hardware and software, not because it is better. The claim is widely repeated. It has never been measured.

Give a candidate architecture $a$ (a new attention variant, a state-space model, a sparse or dynamic-routing network), the current accelerator-plus-kernel stack $h_0$, and a reference architecture $a_0$ (dense Transformer on GPU). Produce a number: how much of $a$'s observed deficit against $a_0$ is intrinsic to $a$, and how much is the stack.

Three variants, different difficulty:

- **Measurement.** Define and estimate a *lottery gap* $\Delta(a)$ — quality-per-unit-cost that $a$ would attain under a stack tuned as hard as $a_0$'s, minus what it attains today. Blocked: the counterfactual stack is unobserved.
- **Method.** Build a cheap estimator (roofline ceiling, autotuned-kernel bound, compiler search) that predicts $\Delta(a)$ without paying the multi-year engineering cost. Partly runnable today.
- **Theory.** Prove that $\Delta$ is identifiable at all from single-stack observations, or prove it is not. Open.

Solved means: a procedure that, applied to an architecture in year $t$, ranks it as it would be ranked in year $t+3$ after the kernels arrive — validated retrospectively on architectures whose outcome we now know.

## 2. Formal Setting

Architecture $a \in \mathcal{A}$: a family of computation graphs with parameter count $N$. Stack $h = (\text{silicon}, \text{compiler}, \text{kernels}, \text{engineer-hours})$.

**Attained quality.** For budget $C$ in accelerator-seconds on stack $h$,
$$L(a, h, C) = \min_{\theta,\, \text{cost}(a,h,\theta) \le C} \; \mathbb{E}_{x \sim \mathcal{D}}\!\left[-\log p_{a,\theta}(x)\right].$$
Measured as: validation cross-entropy in nats/token on a held-out slice of the training distribution, with $C$ the wall-clock GPU-seconds of the training run, logged from the job scheduler — not FLOPs.

**Hardware efficiency.** Model FLOPs utilization
$$\mathrm{MFU}(a,h) = \frac{6ND / T}{F_{\text{peak}}},$$
$D$ tokens, $T$ seconds, $F_{\text{peak}}$ the vendor dense peak (A100 BF16: $3.12\times10^{14}$ FLOP/s; H100 SXM: $9.9\times10^{14}$). Measured with `nsys`/`ncu` counters, not from the vendor datasheet alone.

**Roofline ceiling.** With arithmetic intensity $I = \text{FLOPs} / \text{bytes moved}$ and bandwidth $B$,
$$T_{\min} = \max\!\left(\frac{\text{FLOPs}}{F_{\text{peak}}}, \; \frac{\text{bytes}}{B}\right), \qquad \rho(a,h) = \frac{T_{\min}}{T_{\text{realized}}} \in (0,1].$$
$\rho$ is the *realization ratio*: how much of the physically attainable speed the current kernel gets.

**Lottery gap.** With $h^\star(a)$ the stack $a$ would receive under equal engineering investment,
$$\Delta(a) \;=\; \big[L(a_0, h_0, C) - L(a, h_0, C)\big] \;-\; \big[L(a_0, h^\star(a_0), C) - L(a, h^\star(a), C)\big].$$
$\Delta(a) > 0$ means $a$ is a lottery loser: it looks worse than it would under a fair stack.

**Assumptions, and which fail.**

1. *$F_{\text{peak}}$ is a real ceiling.* Fails: sparse-tensor-core peaks (H100 quotes $2\times$ with 2:4 sparsity) are unreachable for most graphs, so MFU denominators are not comparable across architectures.
2. *Loss at fixed $C$ is comparable across architectures.* Fails: architectures have different loss-scale offsets and different optimal hyperparameters; a shared $\eta$ favors the tuned incumbent (Tay et al., EMNLP Findings 2023).
3. *$h^\star(a)$ exists and is unique.* Fails: the counterfactual stack depends on which silicon generation gets built, which is itself endogenous to which architectures are popular. This is the core defect.
4. *Roofline is tight.* Fails for kernels bound by launch overhead, warp divergence, or recurrent serialization — $T_{\min}$ then understates by $2$–$10\times$.

## 3. State of the Art

**Conceptual (established, not quantified).** Hooker (CACM 2021) states the thesis with historical cases — LSTMs over other recurrent forms, backprop's 30-year delay. No estimator is proposed.

**Measurement critique (established).** Dehghani et al., *The Efficiency Misnomer* (ICLR 2022), show FLOPs, parameters, and throughput rank models differently and that single-metric claims are not reliable. Established via re-measurement across vision and NLP models; not a lottery estimator.

**Kernel-side existence proofs (established).** FlashAttention (Dao et al., NeurIPS 2022) raised attention from memory-bound to near-roofline, giving $3\times$ end-to-end GPT-2 speedup with *no* change to the math. This is a lower bound on stack-induced measurement error: pre-2022 attention benchmarks understated attention by roughly that factor. Mamba (Gu & Dao, COLM 2024) is the same story for selective SSMs: a hardware-aware parallel scan turns an unimplementable recurrence into a competitive one.

**Claimed but unablated.** Papers reporting a novel architecture "matches Transformers at lower cost" almost never report $\rho$ for both arms, so it is not determinable whether the gain is algorithmic or kernel-level. Long Range Arena (Tay et al., ICLR 2021) rankings are widely cited as architecture comparisons but are benchmark numbers on one task suite at one scale; several LRA leaders did not transfer to language modeling.

**Benchmark-number-only.** MLPerf training/inference results are stack-and-architecture jointly optimized by vendors; they measure the pair, not either factor.

## 4. What Is Known

- **Kernel work alone moves end-to-end training time by $2$–$4\times$ at identical FLOPs.** FlashAttention: $3\times$ on GPT-2 (1.5B-scale codebase, A100), $15\%$ on BERT-large vs. the MLPerf 1.1 record.
- **MFU spread across well-engineered systems is $\sim 20$–$55\%$.** PaLM 540B reported $46.2\%$ model FLOPs utilization on TPU v4 (Chowdhery et al., JMLR 2023); typical unoptimized research training runs sit near $20\%$.
- **Structured sparsity delivers well under its nominal factor.** NVIDIA 2:4 sparsity has a $2\times$ math peak; reported end-to-end inference gains are commonly $1.3$–$1.5\times$ (Mishra et al., 2021). Unstructured pruning at $90\%$ sparsity typically gives no dense-GPU speedup at all (Blalock et al., MLSys 2020).
- **Architecture rankings are scale-dependent.** Tay et al. (2023) trained ~10 architectures across scales and found rank order changes with $N$; upstream perplexity ranking does not reliably transfer downstream. Scale: up to ~few-billion parameters.
- **Data movement, not math, dominates.** Ivanov et al. (MLSys 2021) attribute ~$40\%$ of Transformer training time to non-tensor operations on a V100 baseline.

## 5. What Is Not Known

- **Methodologically blocked (the primary gap).** $h^\star(a)$ has no operational definition. "Equal engineering investment" is not a measurable quantity — engineer-hours, compiler passes, and silicon feature requests are not on a common scale, and the effort actually spent is confounded with the architecture's perceived promise. Without $h^\star$, $\Delta(a)$ is a difference of one observed and one undefined term.
- **Theoretically open.** Whether $\Delta(a)$ is identifiable from any finite set of single-stack observations. No impossibility proof and no identification result exists. Plausibly a causal-inference problem with an unobserved confounder (research attention) that drives both stack maturity and reported quality.
- **Empirically open.** Whether $\rho(a,h_0)$ — cheap to compute — predicts future ranking change. The retrospective study over 2017–2025 architectures is runnable now on public artifacts and has not been run.
- **Empirically open.** Whether hand-tuned kernels still beat compiler autotuning (TVM, Triton, Mosaic) by enough to matter; if the gap were small, autotuned $\rho$ would be a usable proxy for $h^\star$.

## 6. Why It Is Hard

**Non-identifiability under an unobserved confounder.** Architecture popularity causes kernel investment, and kernel investment causes measured performance. An architecture that looks bad may be bad, or may be unfunded; both produce the same single-stack observation. There is no natural experiment assigning engineering effort at random.

**Absent ground truth.** Validating an estimator needs architectures whose "fair-stack" performance is known. The only such cases are ones that already won the lottery — a selected sample, biased toward architectures whose fair-stack value was high.

**Confounded measurement.** The comparison needs matched hyperparameter tuning budgets, matched data, matched precision, and matched parallelism strategy. Almost no published pairwise comparison controls all four.

**Compute cost is secondary but real.** A credible arm is a $\ge$1B-parameter, $\ge$100B-token run per architecture per stack condition; four conditions is a few thousand A100-days.

## 7. Current Research (as of 2026)

- **Kernel-neutral architecture search.** Mechanistic architecture design (Poli et al., 2024) screens architectures on small synthetic tasks before committing to kernels — a partial answer to "evaluate before engineering." *(frontier — verify current scaling of this line.)*
- **Compiler-first evaluation.** Triton (Tillet et al., MAPL 2019) and successor DSLs lower the cost of a competent kernel, shrinking the effort asymmetry. Whether autotuned kernels close enough of the hand-tuned gap to serve as $h^\star$ proxy is unsettled.
- **Hybrid SSM–attention models** (Jamba, Zamba, Nemotron-H lines) are commercial evidence that a formerly-losing family became viable once kernels existed. *(frontier — verify.)*
- **Non-GPU silicon** (Cerebras wafer-scale, Groq/Etched-style dataflow parts) provides genuinely different $h$, and is the only route to a real second observation of the same architecture under a different stack. *(frontier — verify which architectures have been trained comparably on two substrates.)*

## 8. Concrete Next Experiment

**Retrospective $\rho$-prediction study.** Decide whether the realization ratio measured at publication time predicts later ranking movement.

- **Scale.** 12 architectures published 2018–2023 with public reference implementations (vanilla attention, Performer, Linformer, S4, H3, Hyena, RWKV, Mamba, MoE-switch, 2:4-sparse dense, MQA, sliding-window). For each, two implementations: the *original author code* and the *best public kernel as of 2026*. Train each at $N = 350$M on 7B tokens of a fixed corpus, identical tokenizer, identical data order, per-arch learning-rate sweep over 5 points. About 24 runs $\times$ ~150 A100-hours $\approx$ 3,600 A100-hours ($\sim$\$10k spot).
- **Control arm.** Dense Transformer with FlashAttention, whose $\rho$ was already near-ceiling in 2022 — it should show near-zero ranking movement between the two implementation conditions. Any estimator that assigns it a large $\Delta$ is falsified.
- **Deciding number.** Spearman $r_s$ between $\rho_{\text{original}}(a)$ (roofline realization ratio of the author code, measured with `ncu` on one A100) and $\Delta_{\text{rank}}(a)$ (change in loss-per-GPU-second rank between original and 2026 kernels). **$r_s \le -0.6$ with $p < 0.05$ over $n=12$** makes $\rho$ a usable pre-registration filter: low $\rho$ at publication predicts later gain. $|r_s| < 0.3$ says $\rho$ is not a lottery estimator and the problem stays blocked.

## 9. Key References

- **[Foundational]** Sara Hooker. *The Hardware Lottery.* Communications of the ACM 64(12), 2021. — arXiv:2009.06489
- **[Foundational]** Samuel Williams, Andrew Waterman, David Patterson. *Roofline: An Insightful Visual Performance Model for Multicore Architectures.* CACM 52(4), 2009.
- **[SOTA]** Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS 2022. — arXiv:2205.14135
- **[SOTA]** Albert Gu, Tri Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* COLM 2024. — arXiv:2312.00752
- **[SOTA]** Mostafa Dehghani, Anurag Arnab, Lucas Beyer, Ashish Vaswani, Yi Tay. *The Efficiency Misnomer.* ICLR 2022. — arXiv:2110.12894
- **[SOTA]** Yi Tay et al. *Scaling Laws vs Model Architectures: How Does Inductive Bias Influence Scaling?* Findings of EMNLP 2023. — arXiv:2207.10551
- **[Empirical]** Davis Blalock, Jose Javier Gonzalez Ortiz, Jonathan Frankle, John Guttag. *What Is the State of Neural Network Pruning?* MLSys 2020. — arXiv:2003.03033
- **[Empirical]** Andrei Ivanov, Nikoli Dryden, Tal Ben-Nun, Shigang Li, Torsten Hoefler. *Data Movement Is All You Need: A Case Study on Optimizing Transformers.* MLSys 2021. — arXiv:2007.00072
- **[Systems]** Philippe Tillet, H. T. Kung, David Cox. *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations.* MAPL 2019.
- **[Systems]** Aakanksha Chowdhery et al. *PaLM: Scaling Language Modeling with Pathways.* JMLR 24(240), 2023. — arXiv:2204.02311
- **[Survey]** John Hennessy, David Patterson. *A New Golden Age for Computer Architecture.* CACM 62(2), 2019.

## 10. Worked Example

**Selective SSM, 2021 vs. 2024.** Take the S4/selective-scan family and ask what a 2021 evaluator would have concluded.

A selective scan over sequence length $L=4096$, batch 8, model width $d=1024$, state $n=16$. A naive materializing implementation writes the full hidden state: $B \cdot L \cdot d \cdot n \cdot 2$ bytes in BF16 $= 8 \cdot 4096 \cdot 1024 \cdot 16 \cdot 2 \approx 1.07$ GB per layer, read and written. On an A100 at $B = 1.55$ TB/s that is
$$T_{\text{mem}} \approx \frac{2 \times 1.07\ \text{GB}}{1.55\ \text{TB/s}} \approx 1.4\ \text{ms}.$$
The math is roughly $B L d n \cdot \mathcal{O}(1)$ FLOPs $\approx 5.4 \times 10^8$, which at $3.12\times10^{14}$ FLOP/s costs $\approx 0.002$ ms. Arithmetic intensity $I \approx 0.25$ FLOP/byte — deep in the memory-bound regime. $\rho \approx 0.002/1.4 \approx 1.5\times10^{-3}$: the naive kernel runs at about a thousandth of its roofline ceiling.

The 2022–2024 fix is not algorithmic. Keep the state in SRAM, fuse the scan, never write $h$ to HBM. The bytes moved drop to the $B L d$ input/output traffic, $\approx 67$ MB, and the kernel becomes latency-bound instead — reported as $20$–$40\times$ faster than the materializing version in the Mamba paper, and turning a family that had lost the lottery into a shipped production architecture.

**Where the obstruction becomes visible.** $\rho \approx 10^{-3}$ was computable in 2021 with a spreadsheet and `ncu`. It correctly flagged "this is a kernel problem, not an architecture problem." But it does not give $\Delta$. To get $\Delta$ you must know what quality the architecture reaches *after* the kernel exists, and the only way anyone has learned that is by spending two years writing the kernel. $\rho$ says the ceiling is far away; it says nothing about whether the architecture is any good once it gets there. Linformer also had a low realization ratio, got competent kernels, and still lost — on quality. That is exactly the discrimination the retrospective study in §8 is built to test, and until it runs, "hardware lottery" remains a diagnosis available only in hindsight.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*