---
id: 13-parameter-efficient-adaptation/effective-rank-of-learned-updates
title: "Rank Deficiency of Learned LoRA Updates"
topic: 13-parameter-efficient-adaptation
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Rank Deficiency of Learned LoRA Updates

> **Topic:** Parameter-Efficient Adaptation · **ID:** `13-parameter-efficient-adaptation/effective-rank-of-learned-updates` · **Status:** empirically-open

## 1. Problem Statement

A LoRA adapter allocates rank $r$ to a weight matrix. The learned update $\Delta W = \frac{\alpha}{r} BA$ has algebraic rank at most $r$, but its singular value spectrum is usually far from flat: most of the Frobenius energy sits in a handful of directions. The question is whether that concentration means the allocated rank was wasted, or whether the low-mass directions do work that the loss depends on.

Three variants, routinely conflated:

- **Measurement.** Given a trained adapter, report a scalar $\hat{r}$ that predicts what rank the task actually needed. No agreed estimator exists; stable rank, entropy-based effective rank, and energy-threshold rank disagree by a factor of 3 on the same matrix (§10).
- **Method.** Given a task and budget, allocate rank per layer so that no direction is wasted. AdaLoRA and its successors do this by pruning during training; whether the gains come from better allocation or from the extra regularization is not ablated.
- **Theory.** Characterize the minimum rank $r^\star(\epsilon)$ such that some rank-$r$ update reaches loss within $\epsilon$ of full fine-tuning on a given task and pretrained model. Open even for two-layer networks with data-dependent targets.

Solved would mean: an estimator computable from a single trained adapter whose value predicts, within $\pm 1$ rank step, the smallest $r$ that retrains to the same loss.

## 2. Formal Setting

Pretrained weight $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, adapters $B \in \mathbb{R}^{d_{\text{out}} \times r}$, $A \in \mathbb{R}^{r \times d_{\text{in}}}$, $B$ initialized to zero, $A$ Gaussian. The deployed weight is $W_0 + \gamma BA$ with scaling $\gamma = \alpha/r$ (LoRA) or $\alpha/\sqrt{r}$ (rsLoRA). Write $\Delta W = \gamma BA$, singular values $\sigma_1 \ge \dots \ge \sigma_r > 0$, $p_i = \sigma_i / \sum_j \sigma_j$.

Candidate estimators, each computed from one SVD of a $d_{\text{out}} \times d_{\text{in}}$ matrix:

$$\text{srank}(\Delta W) = \frac{\|\Delta W\|_F^2}{\|\Delta W\|_2^2}, \qquad \text{erank}(\Delta W) = \exp\!\Big(-\sum_{i=1}^r p_i \log p_i\Big), \qquad r_\tau = \min\Big\{k : \sum_{i \le k}\sigma_i^2 \ge \tau \|\Delta W\|_F^2\Big\}.$$

`erank` is Roy & Vetterli's effective rank (EUSIPCO 2007). These are *spectral* quantities. The quantity that matters is *functional*:

$$r^\star(\epsilon) = \min\{ r : \min_{B,A} \mathcal{L}(W_0 + \gamma BA) \le \mathcal{L}^{\text{full}} + \epsilon \},$$

with $\mathcal{L}$ the held-out task loss and $\mathcal{L}^{\text{full}}$ the full fine-tuning loss under matched budget. A distinct, weaker quantity is the *truncation rank* $r_{\text{trunc}}(\epsilon)$: the smallest $k$ such that replacing $\Delta W$ by its rank-$k$ SVD truncation, in every layer at once, keeps loss within $\epsilon$.

Alignment to the pretrained spectrum matters separately. For left singular vector $u_i$ of $\Delta W$ and the top-$k$ left singular vectors $U_0^{(k)}$ of $W_0$, define $\text{align}(u_i) = \|U_0^{(k)\top} u_i\|_2$. Shuttleworth et al. call a direction with high $\sigma_i$ and low alignment an *intruder dimension*.

Assumptions, with the ones known to fail marked:

1. Per-matrix independence — the spectra of $\Delta W$ in different layers are treated as separable. **Violated**: attention output depends on the product of $Q$ and $K$ updates.
2. $\hat{r}$ is scale-invariant. `srank`, `erank`, $r_\tau$ are invariant to $\gamma$, so they cannot see the $\alpha/r$ collapse — but the *trained* spectrum does depend on $\gamma$, so cross-$r$ comparisons remain confounded. **Violated in practice** whenever $\alpha$ is held fixed while $r$ sweeps.
3. Spectrum implies function, i.e. $r_{\text{trunc}} \approx r^\star$. **Not established**; the two differ whenever the optimizer's path, not just its endpoint, determines the reachable loss.

## 3. State of the Art

**Established.** LoRA (Hu et al., ICLR 2022) reports GPT-3 175B matching or beating full fine-tuning on WikiSQL and MNLI with adapters on $W_q, W_v$ only, at ranks as low as 1–4. Their Grassmann-distance analysis shows the top singular directions of $A_{r=8}$ and $A_{r=64}$ largely coincide, while the tail directions of the $r{=}64$ run do not reappear across seeds — evidence the tail is noise-dominated. Aghajanyan et al. (ACL 2021) measured intrinsic dimension $d_{90}$ directly: RoBERTa-base needs $\approx 896$ random directions to reach 90% of full fine-tuning on MRPC, RoBERTa-large $\approx 207$ — larger models need *fewer*.

**Claimed but unablated.** AdaLoRA (Zhang et al., ICLR 2023) allocates rank by importance-scored SVD pruning and reports gains over fixed-rank LoRA on GLUE/SQuAD; the gain is not separated from the regularization effect of the pruning schedule and the orthogonality penalty. DoRA (Liu et al., ICML 2024) attributes its gain to magnitude/direction decomposition, with the rank contribution not isolated. PiSSA (Meng et al., NeurIPS 2024) and LoRA-GA (Wang et al., NeurIPS 2024) initialize in the principal or gradient-aligned subspace and report faster convergence; whether they change $r^\star$ or only the path to it is untested.

**Benchmark-number-only.** Most "rank $r{=}8$ suffices" claims are single-task accuracy deltas on GLUE or GSM8K, at one seed, with $\alpha$ fixed across the sweep. Under the $\alpha/r$ scaling, higher $r$ shrinks the effective learning rate; Kalajdzievski (2023) shows $\alpha/\sqrt{r}$ removes the collapse and that higher ranks then keep improving. Any rank sweep that did not use $\alpha/\sqrt{r}$ measures scaling, not capacity.

**Theory SOTA.** Zeng & Lee (ICLR 2024) give expressivity: a frozen network of depth $L$ can be adapted to any target of comparable width by LoRA of rank about $\lceil \text{width}/L \rceil$ per layer — an existence result, silent on what SGD finds. Jang, Lee & Ryu (ICML 2024) show LoRA training in the NTK regime has no spurious local minima for $r \gtrsim \sqrt{N}$ with $N$ samples.

## 4. What Is Known

- **Full fine-tuning updates are high-rank.** Biderman et al. (TMLR 2024), on Llama-2-7B and 13B over code and math, report the spectra of full fine-tuning $\Delta W$ carry rank 10–100× higher than typical LoRA configurations ($r = 16$). At the same scale LoRA "learns less and forgets less": worse on target-domain code, better retention of source-domain benchmarks.
- **Low-rank adaptation is not a low-rank *approximation* of full fine-tuning.** Shuttleworth et al. (2024, arXiv:2410.21228), on RoBERTa and Llama-family models, find LoRA solutions contain high-singular-value directions nearly orthogonal to the pretrained spectrum — intruder dimensions — that full fine-tuning does not produce, at equal target-task accuracy. Their prevalence rises as $r$ falls and as $\alpha/r$ rises, and models carrying them degrade more under sequential task training.
- **Intrinsic dimension shrinks with model scale**, measured by random-subspace projection at RoBERTa-base/large scale (Aghajanyan et al., ACL 2021).
- **Rank can be pushed to 1 with shared random bases.** VeRA (Kopiczko et al., ICLR 2024) freezes random $A, B$ and trains only diagonal scaling vectors, matching LoRA $r{=}16$ on GLUE at ~10× fewer trained parameters — evidence that *trained* rank and *expressed* rank are different quantities.
- **High-rank behavior recoverable from low-rank steps.** ReLoRA (Lialin et al., ICLR 2024) reaches higher cumulative rank by periodically merging and restarting adapters, at 350M–1.3B pretraining scale.

## 5. What Is Not Known

- **Empirically open.** Whether $r_{\text{trunc}}$ predicts $r^\star$. The experiment — train at $r{=}64$ with $\alpha/\sqrt{r}$, measure the spectrum, retrain at the predicted rank — is a few thousand GPU-hours at 7B and has not been published as a controlled sweep with seeds.
- **Empirically open.** Whether the low-mass tail contributes to generalization rather than target-task fit. No published test truncates the tail and measures out-of-distribution and forgetting metrics, which is exactly where intruder dimensions bite.
- **Methodologically blocked.** There is no agreed definition of "the rank the task needed". `srank`, `erank`, $r_\tau$ are not monotone transforms of each other (§10), and none is validated against a retraining ground truth.
- **Theoretically open.** A data-dependent lower bound on $r^\star$. Zeng & Lee bound expressivity from above; no matching lower bound ties task structure to required rank, even for a two-layer teacher–student setup.

## 6. Why It Is Hard

The obstruction is **non-identifiability compounded by a confounded control**.

Non-identifiability: $\Delta W = BA$ is invariant under $B \mapsto BG$, $A \mapsto G^{-1}A$ for any invertible $G \in \mathbb{R}^{r \times r}$. The spectra of $A$ and $B$ separately are meaningless; only $\Delta W$'s spectrum is well defined, and it is an endpoint statistic that says nothing about which directions the optimizer needed en route. A direction with $\sigma_i \approx 0$ at convergence may have carried gradient early.

Confounded control: the standard rank sweep changes three things at once — parameter count, effective learning rate via $\alpha/r$, and initialization variance of $A$ (which is $r$-dependent under common schemes). Hayou et al. (NeurIPS 2024) show initialization alone flips which of $A$ or $B$ dominates the dynamics. So "rank $r{=}64$ did not beat $r{=}8$" is not evidence that rank 8 sufficed.

Absent ground truth: $r^\star$ is defined by a minimum over training runs, so establishing it requires a full retraining sweep per layer group per task. The cost is the reason no one has run it at 7B with seeds.

## 7. Current Research (as of 2026)

- **Rank-stabilized and spectrum-aware initialization.** PiSSA, LoRA-GA, and rsLoRA-style scaling are converging into default recipes in PEFT libraries; the open question is whether they lower $r^\star$ or only accelerate convergence.
- **Low-rank optimizer subspaces rather than low-rank weights.** GaLore (Zhao et al., ICML 2024) projects gradients, not weights, allowing full-rank updates under a low-rank memory budget — a direct test of whether the constraint that hurts is on $\Delta W$ or on the optimizer state.
- **Forgetting-as-diagnostic.** Following Shuttleworth et al. and Biderman et al., using source-domain retention rather than target accuracy as the signal that rank is mis-allocated *(frontier — verify)*.
- **Per-layer allocation learned from spectra** at 7B–70B scale, in industrial fine-tuning stacks; results are mostly unablated internal reports *(frontier — verify)*.

## 8. Concrete Next Experiment

**Scale.** Llama-3-8B (or Llama-2-7B for comparability with Biderman et al.), instruction tuning on MetaMathQA (395k) and a code SFT set, adapters on all linear layers. Ranks $r \in \{4, 8, 16, 32, 64, 128\}$, 3 seeds. ~40 runs, roughly 2–4k A100-hours.

**Control arm.** Every run uses $\gamma = \alpha/\sqrt{r}$ with $\alpha$ tuned once at $r{=}16$ and held fixed, so the effective learning rate does not drift with $r$. Second control: full fine-tuning at matched token budget. Third control: a rank-$r$ *random-direction* adapter (VeRA-style, frozen bases) at each $r$, to separate trained rank from expressed rank.

**Procedure.** From the $r{=}128$ run, compute per-layer `erank`, `srank`, $r_{0.9}$, $r_{0.99}$. Take each estimator's per-layer value as a rank allocation, retrain from scratch at that allocation, and measure GSM8K/HumanEval plus source-domain retention (MMLU, HellaSwag).

**Deciding number.** The rank prediction error $|\hat{r}_{\text{est}} - r^\star|$ in rank-doubling steps, where $r^\star$ is the smallest swept rank within 0.5 points of the $r{=}128$ target-task score. If any estimator achieves median $|\hat{r}_{\text{est}} - r^\star| \le 1$ step across layers and both tasks, the measurement variant is solved. If all estimators exceed 1 step — in particular if `srank` and $r_{0.99}$ point to different sides of $r^\star$ — the spectrum is confirmed non-predictive and the field should stop reporting effective rank as evidence.

## 9. Key References

- **[Foundational]** Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022. — arXiv:2106.09685
- **[Foundational]** Armen Aghajanyan, Sonal Gupta, Luke Zettlemoyer. *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning.* ACL 2021. — arXiv:2012.13255
- **[Foundational]** Chunyuan Li, Heerad Farkhoor, Rosanne Liu, Jason Yosinski. *Measuring the Intrinsic Dimension of Objective Landscapes.* ICLR 2018. — arXiv:1804.08838
- **[Foundational]** Olivier Roy, Martin Vetterli. *The Effective Rank: A Measure of Effective Dimensionality.* EUSIPCO 2007.
- **[SOTA]** Dan Biderman et al. *LoRA Learns Less and Forgets Less.* TMLR 2024. — arXiv:2405.09673
- **[SOTA]** Reece Shuttleworth, Jacob Andreas, Antonio Torralba, Pratyusha Sharma. *LoRA vs Full Fine-tuning: An Illusion of Equivalence.* 2024. — arXiv:2410.21228
- **[SOTA]** Qingru Zhang, Minshuo Chen, Alexander Bukharin, Pengcheng He, Yu Cheng, Weizhu Chen, Tuo Zhao. *AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning.* ICLR 2023. — arXiv:2303.10512
- **[SOTA]** Damjan Kalajdzievski. *A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA.* 2023. — arXiv:2312.03732
- **[Theory]** Yuchen Zeng, Kangwook Lee. *The Expressive Power of Low-Rank Adaptation.* ICLR 2024. — arXiv:2310.17513
- **[Theory]** Uijeong Jang, Jason D. Lee, Ernest K. Ryu. *LoRA Training in the NTK Regime has No Spurious Local Minima.* ICML 2024. — arXiv:2402.11867
- **[Method]** Dawid J. Kopiczko, Tijmen Blankevoort, Yuki M. Asano. *VeRA: Vector-based Random Matrix Adaptation.* ICLR 2024. — arXiv:2310.11454
- **[Method]** Vladislav Lialin, Sherin Muckatira, Namrata Shivagunde, Anna Rumshisky. *ReLoRA: High-Rank Training Through Low-Rank Updates.* ICLR 2024. — arXiv:2307.05695
- **[Method]** Jiawei Zhao, Zhenyu Zhang, Beidi Chen, Zhangyang Wang, Anima Anandkumar, Yuandong Tian. *GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection.* ICML 2024. — arXiv:2403.03507
- **[Method]** Fanxu Meng, Zhaohui Wang, Muhan Zhang. *PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models.* NeurIPS 2024. — arXiv:2404.02948
- **[Survey]** Zeyu Han, Chao Gao, Jinyang Liu, Jeff Zhang, Sai Qian Zhang. *Parameter-Efficient Fine-Tuning for Large Models: A Comprehensive Survey.* TMLR 2024. — arXiv:2403.14608

## 10. Worked Example

Take one $4096 \times 4096$ projection with an $r{=}16$ adapter. Suppose the trained $\Delta W$ has a geometrically decaying spectrum, $\sigma_i = \sigma_1 \rho^{i-1}$ with $\rho = 0.85$ — a mild decay, far flatter than what LoRA runs typically show. Compute all three estimators on the *same* matrix.

Sum of singular values: $\sum_i \rho^{i-1} = (1 - 0.85^{16})/0.15 = (1 - 0.0743)/0.15 = 6.172$.

Effective rank: $H = \log S - (\log\rho)\sum_i p_i (i{-}1) = 1.820 + 0.1625 \times 4.383 = 2.532$, so

$$\text{erank} = e^{2.532} = 12.6.$$

Stable rank: $\sum_i \rho^{2(i-1)} = (1 - 0.7225^{16})/(1 - 0.7225) = 0.9945/0.2775$, so

$$\text{srank} = 3.6.$$

Energy ranks: $0.7225^k \le 0.105 \Rightarrow k = 7$, so $r_{0.9} = 7$; $0.7225^k \le 0.0155 \Rightarrow k = 13$, so $r_{0.99} = 13$.

One matrix, one spectrum, four answers: **3.6, 7, 12.6, 13** out of a budget of 16. `srank` says three-quarters of the rank is wasted; `erank` says almost none is. The two are not monotone rescalings — flatten $\rho$ to 0.95 and `srank` rises to 10.2 while `erank` rises to 15.0, compressing the gap; sharpen to $\rho = 0.6$ and `srank` falls to 1.6 while `erank` falls to 5.0. The ratio `erank`/`srank` moves from 3.5 to 1.5 across a plausible range of real spectra, so a paper reporting "effective rank 12 at $r{=}16$, so the budget is well used" and a paper reporting "stable rank 3.6 at $r{=}16$, so 4 would do" can describe the identical adapter.

The obstruction is visible here: neither number is anchored to a retraining outcome. To know whether $r{=}4$ suffices you must train at $r{=}4$ — and if you do that with $\alpha$ fixed and $\gamma = \alpha/r$, the $r{=}4$ run gets $4\times$ the effective step size of the $r{=}16$ run, so a win at $r{=}4$ is not evidence about rank at all. That is why the experiment in §8 fixes $\gamma = \alpha/\sqrt{r}$ before touching the spectrum.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*