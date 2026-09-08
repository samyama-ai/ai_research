---
id: 13-parameter-efficient-adaptation/continual-adapter-accumulation
title: "Continual Adapter Accumulation Without Growth"
topic: 13-parameter-efficient-adaptation
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Continual Adapter Accumulation Without Growth

> **Topic:** Parameter-Efficient Adaptation · **ID:** `13-parameter-efficient-adaptation/continual-adapter-accumulation` · **Status:** open

## 1. Problem Statement

A frozen base model is adapted to a stream of tasks $t = 1, 2, \dots, T$, one at a time, with no access to earlier tasks' data. Each adaptation is parameter-efficient (LoRA, adapter, prefix). The naive solution keeps one adapter per task: storage and routing cost grow $\Theta(T)$.

**The problem:** produce an adaptation state of size $O(1)$ in $T$ — a fixed parameter budget $B$ that does not grow as tasks arrive — while retaining per-task accuracy within $\epsilon$ of the per-task-adapter oracle.

Three variants, different difficulty:

- **Measurement.** Define "does not grow." Total bytes? Bytes per task amortized? Inference-time active parameters? A method that stores $T$ rank-1 factors but activates one has $O(1)$ compute and $\Theta(T)$ storage. Current papers report whichever is favorable.
- **Method.** Build a fixed-budget update $\Delta_T \in \mathbb{R}^{d \times k}$ with $\mathrm{rank}(\Delta_T) \le r$ fixed, such that accuracy on all $T$ tasks stays within $\epsilon$. Empirically open.
- **Theory.** Determine whether a fixed-rank subspace can represent $T$ arbitrary task updates at all, and characterize the capacity–interference trade-off. Theoretically open; partial negative results exist.

Solved means: at $T = 100$ with $B$ fixed, average accuracy over all seen tasks is within 2 points of the $T$-separate-adapters oracle, with no data replay.

## 2. Formal Setting

Base model weights $W_0 \in \mathbb{R}^{d\times k}$, frozen. Task $t$ has data $D_t \sim \mathcal{D}_t$ and loss $\ell_t$. A LoRA update is $\Delta_t = B_t A_t$ with $A_t \in \mathbb{R}^{r\times k}$, $B_t \in \mathbb{R}^{d\times r}$, so $|\Delta_t| = r(d+k)$ trainable scalars.

**Accumulated state** after $T$ tasks: $S_T$, with **budget** measured three ways, all reportable separately:

$$B_{\text{store}}(T) = |S_T| \cdot b \ \text{bytes},\quad B_{\text{act}}(T) = \max_t |\{\theta \in S_T : \partial y_t/\partial\theta \ne 0\}|,\quad B_{\text{train}}(T)$$

where $b$ is bits per parameter after quantization. "No growth" means $\lim_T B_{\text{store}}(T)/T = 0$ — the strict form is $B_{\text{store}}(T) = B$ constant.

**Retention.** Let $a_t(\tau)$ be accuracy on task $t$ measured after training through task $\tau$, on a held-out set of $\ge 1000$ examples with a fixed decoding rule (greedy, fixed prompt). Standard quantities:

$$\mathrm{ACC}(T) = \frac{1}{T}\sum_{t=1}^{T} a_t(T), \qquad \mathrm{BWT}(T) = \frac{1}{T-1}\sum_{t=1}^{T-1}\big(a_t(T) - a_t(t)\big)$$

**Oracle.** $a_t^{\ast}$ = accuracy of a fresh rank-$r$ adapter trained on $D_t$ alone from $W_0$. The gap to close is $\epsilon(T) = \frac{1}{T}\sum_t (a_t^{\ast} - a_t(T))$.

**Capacity bound (elementary).** If task updates are constrained mutually orthogonal — $\mathrm{col}(\Delta_i) \perp \mathrm{col}(\Delta_j)$ — then $\sum_t \mathrm{rank}(\Delta_t) \le d$, so at most $\lfloor d/r \rfloor$ tasks fit exactly. Merging without the orthogonality constraint gives $\mathrm{rank}(\sum_t \Delta_t) \le \min(d, Tr)$, and truncating back to rank $r$ incurs Eckart–Young error $\|\Delta_{1:T} - \Pi_r \Delta_{1:T}\|_F^2 = \sum_{i>r}\sigma_i^2$.

**Assumptions, and which fail.**
1. *Tasks are discrete and boundaries are known.* Violated in deployment streams; boundary-free variants are much harder.
2. *Task identity is available at inference.* Assumed by most routing methods; violated in the realistic setting, and this single assumption accounts for much of the reported gap.
3. *Updates are low-rank.* Established for single-task adaptation at moderate ranks; not established for the *accumulated* update, whose spectrum grows.
4. *Losses are locally quadratic near $W_0$* (needed for orthogonal-projection arguments). Approximately true for small $\|\Delta\|$, false when adaptation moves the model far.
5. *$\mathcal{D}_t$ are non-overlapping.* False for instruction-tuning streams, which share massive support — this inflates ACC without any real retention.

## 3. State of the Art

**Growing methods (the honest baseline).** Per-task LoRA + task identity at inference is near-oracle by construction and is the arm every no-growth method must beat. Progressive Prompts (Razdaibiedina et al., ICLR 2023) concatenates a new soft prompt per task, forward transfer positive, storage $\Theta(T)$. AdapterFusion (Pfeiffer et al., EACL 2021) keeps all adapters and learns an attention-based combiner — established as a strong composition mechanism, but the fusion layer itself grows with the adapter pool.

**Constrained-subspace methods.** O-LoRA (Wang et al., *Orthogonal Subspace Learning for Language Model Continual Learning*, EMNLP Findings 2023) trains each task's LoRA orthogonal to the span of previous ones and merges into the base — inference cost is $O(1)$, but the *training* state retains previous subspaces, and capacity is bounded by $\lfloor d/r\rfloor$. InfLoRA (Liang & Li, CVPR 2024) picks a per-task subspace that eliminates interference with old tasks by construction; reported strong on class-incremental vision benchmarks. Both are established as beating replay-free baselines on their benchmarks; neither has been ablated at $T \gg 20$.

**Merging methods.** Task arithmetic (Ilharco et al., ICLR 2023), TIES-Merging (Yadav et al., NeurIPS 2023), and DARE (Yu et al., ICML 2024) collapse many task vectors into one fixed-size model. These are genuinely $O(1)$ in storage. They are *offline* — all task vectors available at once — so they are not continual methods, and their sequential variants are largely unstudied.

**Claimed but unablated.** MoE-of-LoRA and adapter-routing systems report both "efficiency" and "no forgetting"; the router and expert pool grow with $T$, so the $O(1)$ claim is about active parameters only. Papers rarely report $B_{\text{store}}$ and $B_{\text{act}}$ in the same table.

**Benchmark-number-only results.** Most reported CL-for-LLM ACC values come from the standard 15-task text benchmark with T5-large or Llama-2-7B, three task orders. Cross-paper comparison there is unreliable: prompt format, decoding, and per-task epoch count differ.

## 4. What Is Known

- **LoRA cost.** Hu et al. (ICLR 2022) reduced trainable parameters on GPT-3 175B by $10^4\times$ versus full fine-tuning with no quality loss on the reported tasks. Rank $r{=}4$–$16$ suffices for single-task adaptation at 7B scale.
- **Hardness.** Knoblauch, Husain & Diethe (ICML 2020) prove that optimal continual learning requires perfect memory of a set-intersection structure and that the resulting problem is NP-hard. This is the sharpest known negative result and it applies directly: a fixed-budget state cannot in general reconstruct the optimal joint solution.
- **Orthogonal projection works, up to capacity.** Gradient Projection Memory (Saha et al., ICLR 2021) and Orthogonal Gradient Descent (Farajtabar et al., AISTATS 2020) show near-zero backward transfer while the free subspace lasts; both report degradation as the projected space fills — the mechanism, not just the accuracy, is established.
- **Interference is spectral.** Merging error is exactly the discarded singular mass; for $T$ random rank-$r$ updates in $d$ dimensions, the accumulated spectrum flattens, so rank-$r$ truncation error grows roughly linearly in $T$ until saturation.
- **Regularization alone is insufficient at scale.** EWC (Kirkpatrick et al., PNAS 2017) established the Fisher-penalty approach; subsequent replay-free evaluations on long task sequences show it lagging replay and architecture methods by wide margins.
- **Task-ID removal is the dominant cost.** Across prompt-based CL (L2P, CVPR 2022; DualPrompt, ECCV 2022; CODA-Prompt, CVPR 2023), performance is substantially higher when the correct prompt is selected than when the query-key retrieval picks it. Retrieval error, not capacity, is the leading term at small $T$.

## 5. What Is Not Known

- **Theoretically open.** No bound on the number of tasks representable by a fixed rank-$r$ update at target error $\epsilon$, as a function of task-gradient geometry. Knoblauch et al. rules out optimality, not $\epsilon$-approximation under realistic task-similarity assumptions. No known lower bound of the form "$B = \Omega(f(T,\epsilon))$ bytes are necessary."
- **Empirically open.** No published run of any no-growth PEFT method at $T \ge 100$ tasks on a $\ge$ 7B model with a per-task-adapter oracle in the same table. The experiment is runnable today for well under $10^4$ GPU-hours. This is the single largest gap.
- **Methodologically blocked.** "Growth" has no agreed operational definition (Section 2 gives three), and benchmark task streams have unmeasured overlap, so ACC conflates retention with generalization from later tasks. Until stream overlap is quantified, retention numbers are not comparable across papers.

## 6. Why It Is Hard

**Non-identifiability of the accumulated update.** Many $(B, A)$ factorizations produce the same $\Delta$, and many $\Delta$ produce the same behavior on the tasks seen so far but differ arbitrarily on tasks not yet seen. The learner has no signal distinguishing them, so "which fixed-budget state to keep" is underdetermined by the observed stream.

**Confounded measurement.** Task streams share support. A method can score well on task 3 after task 40 because task 40 taught overlapping skill, not because anything was retained. No standard benchmark reports the mutual information between task distributions, so BWT cannot be attributed.

**Evaluation names the wrong thing.** "Parameter-efficient" is reported as trainable parameters per task; the continual quantity is total retained bytes. A method with 0.06% trainable parameters per task and 100 tasks has stored 6% of the model.

Compute is *not* the obstruction: the deciding experiment is affordable.

## 7. Current Research (as of 2026)

- Orthogonal/null-space LoRA extended to long streams, with adaptive rank allocation per task (successors to O-LoRA and InfLoRA) *(frontier — verify)*.
- Sequential merging: applying TIES/DARE-style trimming online, one task vector at a time, with a fixed-size running state. Promising because it is the only family that is honestly $O(1)$ in storage *(frontier — verify)*.
- Compressing an adapter *library* — sharing bases across tasks so per-task cost is coefficients only, giving $o(T)$ rather than $O(1)$ growth *(frontier — verify)*.
- Task-ID-free routing via learned keys, and the question of whether retrieval error or capacity dominates at large $T$.
- Groups active in the area include the CL-for-LLM lines at Tsinghua/Fudan (O-LoRA lineage), Nanjing (InfLoRA lineage), and the model-merging line at UW/AI2 and UNC *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question:** at fixed storage $B$, how does $\epsilon(T)$ scale with $T$, and where does it cross 2 points?

- **Scale.** Llama-2-7B (or Qwen-2.5-7B), frozen. $T = 128$ tasks drawn from a mixed pool (Super-NaturalInstructions task clusters + 15-task standard CL benchmark), each with $\ge 2000$ train and $\ge 1000$ eval examples, three task orders. Cost estimate: 128 tasks $\times$ ~0.3 GPU-hour/task $\times$ 3 orders $\times$ 4 arms $\approx$ 500 A100-hours.
- **Arms.** (a) Per-task LoRA $r{=}8$ with oracle task ID — the **control** and upper bound. (b) O-LoRA at fixed total $r{=}8$. (c) Sequential TIES-merge into a single $r{=}8$ state. (d) Sequential full-rank merge then rank-8 SVD truncation.
- **Report.** $B_{\text{store}}$, $B_{\text{act}}$, ACC, BWT, and the pairwise task-overlap matrix (n-gram/embedding similarity), so retention is separable from transfer.
- **Deciding number.** $T^{\ast} = \min\{T : \epsilon(T) > 2 \text{ points}\}$ for the best fixed-budget arm. $T^{\ast} \ge 128$ means fixed-budget accumulation is practical at deployment scale. $T^{\ast} \le 16$ — the value the rank-capacity bound and current 20-task benchmarks suggest — means the field's $O(1)$ claims are artifacts of short streams, and the useful target becomes sublinear growth, not zero.

## 9. Key References

- **[Foundational]** Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022. — arXiv:2106.09685
- **[Foundational]** Kirkpatrick et al. *Overcoming catastrophic forgetting in neural networks.* PNAS 114(13), 2017.
- **[Theory]** Knoblauch, Husain, Diethe. *Optimal Continual Learning has Perfect Memory and is NP-hard.* ICML 2020.
- **[Theory]** Saha, Garg, Roy. *Gradient Projection Memory for Continual Learning.* ICLR 2021. — arXiv:2103.09762
- **[SOTA]** Wang, Chen, Jiang, Song, Gu, Xu, Wang, et al. *Orthogonal Subspace Learning for Language Model Continual Learning.* Findings of EMNLP 2023. — arXiv:2310.14152
- **[SOTA]** Liang, Li. *InfLoRA: Interference-Free Low-Rank Adaptation for Continual Learning.* CVPR 2024. — arXiv:2404.00228
- **[SOTA]** Razdaibiedina, Mao, Hou, Khabsa, Lewis, Almahairi. *Progressive Prompts: Continual Learning for Language Models.* ICLR 2023. — arXiv:2301.12314
- **[SOTA]** Yadav, Tam, Raffel, Bansal. *TIES-Merging: Resolving Interference When Merging Models.* NeurIPS 2023. — arXiv:2306.01708
- **[Related]** Ilharco, Ribeiro, Wortsman, Schmidt, Hajishirzi, Farhadi. *Editing Models with Task Arithmetic.* ICLR 2023. — arXiv:2212.04089
- **[Related]** Pfeiffer, Kamath, Rücklé, Cho, Gurevych. *AdapterFusion: Non-Destructive Task Composition for Transfer Learning.* EACL 2021. — arXiv:2005.00247
- **[Related]** Wang, Zhang, Lee, Zhang, Sun, Ren, Su, Perot, Dy, Pfister. *Learning to Prompt for Continual Learning.* CVPR 2022. — arXiv:2112.08654
- **[Survey]** Wang, Zhang, Su, Zhu. *A Comprehensive Survey of Continual Learning: Theory, Method and Application.* IEEE TPAMI, 2024. — arXiv:2302.00487

## 10. Worked Example

Llama-2-7B: 32 layers, hidden $d = 4096$. LoRA $r{=}8$ on $W_q$ and $W_v$ only.

Per task: $2 \text{ matrices} \times 32 \text{ layers} \times 8 \times (4096+4096) = 4.19$M parameters $= 8.4$ MB at fp16. Base model = 13.5 GB fp16.

**Growth arm.** 1,000 tasks $\Rightarrow$ 8.4 GB of adapters, 62% of the base model. At 1,600 tasks the adapter library exceeds the model it adapts. Per-task cost of 0.06% is irrelevant; the accumulated cost is the number that matters.

**Orthogonal arm.** Strict orthogonality allows $\lfloor 4096/8 \rfloor = 512$ tasks per matrix — comfortably above 128, so the *nominal* bound is not binding. But the constraint is on the input activations' span, and the empirical activation covariance of $W_q$ inputs is heavily concentrated: typically ~90% of variance in a few hundred directions. Once each task's projection is restricted to the shrinking complement of the *high-energy* subspace, adaptation quality degrades long before dimension 4096 is used. Suppose the effective usable dimension is 512: capacity drops to 64 tasks, and $T^{\ast} \approx 64$, not 512.

**Merging arm.** Sum 128 rank-8 updates: $\mathrm{rank}(\sum \Delta_t) \le \min(4096, 1024) = 1024$. Truncating to rank 8 keeps 8 of 1024 singular directions. If the spectrum were flat, retained energy is $8/1024 = 0.8\%$; real spectra decay, so retained energy is higher, perhaps 10–20% — still discarding most of the accumulated update.

**The obstruction, made visible.** The two fixed-budget mechanisms fail for different reasons that current benchmarks cannot separate. At $T = 20$ — where every published method is evaluated — orthogonal capacity is untouched and merging error is small, so both look like they work. The failure mode only appears past $T \approx 50$, and the number that distinguishes "capacity exhausted" from "router picked the wrong subspace" is not reported by any existing paper. That is why the problem is open: not that the experiment is expensive, but that nobody has run it far enough out, with the oracle arm in the same table.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*