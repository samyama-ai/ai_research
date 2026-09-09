---
id: 16-state-space-models/parallel-scan-numerical-stability
title: "Numerical Stability of Long-Horizon Parallel Scans"
topic: 16-state-space-models
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Numerical Stability of Long-Horizon Parallel Scans

> **Topic:** State-Space & Recurrent Models · **ID:** `16-state-space-models/parallel-scan-numerical-stability` · **Status:** partially-solved

## 1. Problem Statement

Modern linear-recurrent sequence models (S4, S5, LRU, Mamba, GLA, mLSTM, DeltaNet) are trained by evaluating a first-order linear recurrence with an **associative scan** — a work-efficient tree, or a chunked matmul reformulation — instead of a sequential loop. The tree changes the *order* of floating-point operations. Floating-point addition and multiplication are not associative, so the tree and the loop compute different numbers from the same inputs.

The problem has three variants that are routinely conflated:

- **Measurement.** Given a recurrence and a precision, what is the forward and backward error of the scan as a function of horizon $T$? No standard metric or benchmark exists; papers report loss curves, not errors.
- **Method.** Build a scan whose relative error is bounded independently of $T$ (or grows at most polylogarithmically) at bf16/fp8 throughput. Log-space accumulation, running-max stabilizers, and fp32 chunk accumulators each fix part of this; none is known to fix all of it.
- **Theory.** Prove a backward-stability result for the *gated* scan $h_t = a_t h_{t-1} + b_t$ with data-dependent $a_t$, matching or beating the sequential bound. The known negative result (Mathias 1995) is for matrix prefix products; the diagonal-gated case has no tight published bound.

**Solved** would mean: a scan with a proven relative-error bound $O(\text{polylog}(T)\,u\,\kappa)$ ($u$ = unit roundoff, $\kappa$ = a condition number of the recurrence), realized in a kernel within 15% of the current bf16 throughput, with the bound verified empirically at $T \ge 10^6$.

## 2. Formal Setting

The scan primitive. For $t = 1,\dots,T$ with state $h_t \in \mathbb{R}^{d}$, diagonal gate $a_t \in \mathbb{C}^{d}$, input $b_t \in \mathbb{R}^{d}$:

$$h_t = a_t \odot h_{t-1} + b_t, \qquad h_0 = 0.$$

Closed form, per channel:

$$h_T = \sum_{s=1}^{T} \Big(\prod_{r=s+1}^{T} a_r\Big) b_s .$$

The scan is the prefix operation under $(a,b) \circ (a',b') = (a a',\, a' b + b')$, associative and computable in $O(\log T)$ depth / $O(T)$ work (Blelloch, 1990).

**Quantities as measured.**

- *Unit roundoff* $u$: $u = 2^{-24} \approx 5.96\times10^{-8}$ (fp32), $2^{-11} \approx 4.88\times10^{-4}$ (fp16), $2^{-8} \approx 3.91\times10^{-3}$ (bf16). Measured by inspecting the accumulate dtype of the kernel, not the storage dtype.
- *Forward error* $E(T) = \|\hat h_T - h_T^{(64)}\|_2 / \|h_T^{(64)}\|_2$, where $h^{(64)}$ is the **sequential** recurrence in float64 on the same inputs. This reference is the only available ground truth and is itself only $O(Tu_{64}\kappa)$-accurate; at $T=10^6$, $Tu_{64}\approx 10^{-10}$, so it is safe as a reference down to $E \sim 10^{-8}$.
- *Condition number*: $\kappa_T = \dfrac{\sum_{s} \big|\prod_{r>s} a_r\big|\,|b_s|}{\big|\sum_{s} \prod_{r>s} a_r\, b_s\big|}$ — the standard summation condition number. Measured in float64 alongside the reference. $\kappa_T \gg 1$ exactly when contributions cancel.
- *Dynamic range*: $R_T = \max_{s\le T}\big|\prod_{r>s}a_r\big| \big/ \min_{s\le T}\big|\prod_{r>s}a_r\big|$. Overflow/underflow, not roundoff, is the binding constraint when $\log_2 R_T$ exceeds the exponent range (255 for bf16/fp32; ~35 for fp16 normals).
- *Backward error*: the smallest $\epsilon$ such that $\hat h_T$ is exact for gates $a_t(1+\delta_t)$, $|\delta_t|\le\epsilon$. Estimated numerically by solving a small least-squares perturbation problem; not routinely reported.
- *Gradient fidelity*: $\cos\big(\nabla_\theta \hat L, \nabla_\theta L^{(64)}\big)$ over the whole layer. This is the quantity training actually depends on.

**Assumptions and their violations.**

1. *$|a_t| \le 1$ for all $t$.* Enforced in Mamba/GLA/LRU by $a_t = \exp(-\Delta_t\,\mathrm{softplus}(\cdot))$ or a sigmoid gate. **Holds by construction** in these models; **violated** in unconstrained linear RNNs and in DeltaNet-style transitions where the effective per-step operator is $I - \beta_t k_t k_t^\top$, whose product is not diagonal and not norm-decreasing channel-wise.
2. *Diagonal transition.* **Violated** by DeltaNet, gated DeltaNet, and any non-diagonal SSM; the relevant object is then a matrix prefix product, where Mathias's instability result applies directly.
3. *No cancellation ($\kappa_T = O(1)$).* **Violated in practice**: $b_t$ has mixed signs and $|a_t|\to 1$ on retrieval-heavy channels, which is precisely the regime long-context models are trained into.
4. *Errors are independent across steps.* **Violated**: gates are produced by a shared projection, so $\log a_t$ errors are correlated along $t$, and the cumulative-sum error does not average out as $\sqrt{T}$.

## 3. State of the Art

**Systems/empirical SOTA (established).**

- **fp32 scan state with recomputation** — Mamba (Gu & Dao, COLM 2024). The selective scan keeps $\Delta$, $A$, and the running state in fp32 in SRAM regardless of the bf16 parameter dtype, and recomputes the forward states in the backward pass. This is the de-facto standard and is ablated only in the sense that the authors report it is necessary; no error curve is published.
- **Chunked / "state-space duality" form** — Mamba-2 (Dao & Gu, ICML 2024). Intra-chunk decays are formed as $\exp(\text{cumsum}(\log a))$ over a chunk of 64–256 steps, with an fp32 accumulator; inter-chunk state is carried in fp32. Capping the cumsum length at the chunk size bounds the exponent range, which is a stability mechanism as much as a speed one. The stability role is stated but **not ablated** — reported throughput and loss, not error.
- **Explicit running-max stabilizer** — RWKV (Peng et al., EMNLP Findings 2023) subtracts a running maximum inside the WKV recurrence; xLSTM's mLSTM (Beck et al., NeurIPS 2024) carries a dedicated stabilizer state $m_t = \max(\log f_t + m_{t-1}, \log i_t)$. This is the only mechanism in wide use with an explicit, documented purpose of preventing overflow in the exponentiated gate accumulation.
- **Log-space / exponential parameterization** — LRU (Orvieto et al., ICML 2023) parameterizes eigenvalues as $\lambda = \exp(-\exp(\nu) + i\exp(\theta))$, guaranteeing $|\lambda|<1$ and moving the product to a sum of logs. Established as necessary for stable training at $T \sim 10^4$ on Long Range Arena; the paper's ablations are on task accuracy, not measured roundoff.
- **Chunkwise fp32 accumulation in GLA** (Yang et al., ICML 2024) and DeltaNet (Yang et al., NeurIPS 2024): secondary-level chunking splits the recurrence so that matmuls run on tensor cores in bf16 while the state accumulates in fp32. Presented as a throughput result; the precision split is a design choice with **no published error ablation**.

**Theory SOTA.** Higham's summation analysis (1993; *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM 2002) gives $|\hat s - s| \le \gamma_{n}\sum|x_i|$ with $\gamma_n = nu/(1-nu)$ for sequential summation and $\gamma_{\lceil\log_2 n\rceil}$ for pairwise — the $O(\log T)$ tree is *better* than the loop for pure summation. Mathias (SIAM J. Sci. Comput., 1995) proved the opposite for **parallel prefix matrix products**: the tree algorithm is not backward stable in general, and he constructs examples where its error exceeds the sequential algorithm's substantially. The gated scan sits between these two results and has no tight published bound.

## 4. What Is Known

- Pairwise/tree summation error grows as $\log_2 T$, not $T$: at $T = 2^{20}$, $\gamma_{20} \approx 20u$ against $\gamma_{10^6}\approx 10^6 u$ — a factor $5\times10^4$ in favor of the tree, for the additive part alone.
- The multiplicative part is the problem. Mathias (1995) showed the prefix-product tree loses the sequential algorithm's backward-stability guarantee for general matrices.
- **Exponent range is binding before roundoff is.** With $|a|=0.99$ and $T=8192$, $\prod a = e^{-82.3} = 1.8\times10^{-36}$: representable in fp32 (min normal $1.18\times10^{-38}$) and bf16, but ~31 orders of magnitude below fp16's min normal $6.10\times10^{-5}$. This is the direct reason fp16 scans fail and bf16 (same 8-bit exponent as fp32) does not.
- Mamba (2.8B params, sequences to 8k) and Mamba-2 (to 2.7B) train stably in bf16 **only** with fp32 scan state; this is reported by the authors and reproduced widely in the open `mamba-ssm` and `flash-linear-attention` kernels.
- LRU: without the exponential eigenvalue parameterization, training on Long Range Arena tasks at $T$ up to 16{,}384 degrades or diverges (Orvieto et al., ICML 2023, ablation tables).
- Non-linear recurrences can be parallelized by fixed-point iteration (DEER; Lim et al., ICLR 2024, arXiv:2309.12252), but convergence and its numerical behavior at long $T$ are demonstrated at $T \sim 10^3$–$10^5$, not at LM scale.

## 5. What Is Not Known

- **Theoretically open.** No published tight forward- or backward-error bound for the *diagonal gated* prefix scan as a function of $T$, $\kappa_T$, and $R_T$. Specifically, whether $E(T) = O(u\,\kappa_T \log T)$ holds when $|a_t|\le 1$, or whether a Mathias-type $T$-dependent counterexample exists inside that constraint.
- **Theoretically open.** Whether the log-space scan (cumsum of $\log a_t$, then exponentiate) is backward stable. The cumsum has error $O(u\log T)$ in the *exponent*, which exponentiates to a relative error $O(u \log T \cdot |\log \prod a|)$ — this grows with the total decay and is not obviously bounded.
- **Empirically open.** No published curve of $E(T)$ or gradient cosine versus $T$ for any production scan kernel, at any scale. The experiment costs a few GPU-hours; it has not been run and reported.
- **Empirically open.** Whether fp8 scan accumulation is viable at $T \ge 10^5$ with a stabilizer, or whether the exponent range collapses first.
- **Methodologically blocked.** "Stability" in this literature is measured by *training didn't diverge*. That conflates scan roundoff with optimizer, init, and data effects, and gives no signal on models that converge to a slightly worse optimum because of scan error. There is no agreed metric, no reference implementation of the float64 oracle at scale, and no benchmark.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement with an expensive ground truth**.

- The float64 sequential reference is $O(T)$ depth and has no fast GPU path. At $T = 10^6$, $d = 4096$, one reference forward pass is a serial loop of $10^6$ iterations — minutes per layer per batch element, versus microseconds for the kernel. So the oracle cannot be run inside a training loop, only on isolated probes, and the probes may not sample the states that matter.
- The observable everyone uses (training loss) is dominated by other effects. A scan with $E = 10^{-2}$ and one with $E = 10^{-5}$ can produce indistinguishable loss curves for $10^4$ steps and diverge afterwards, so the ablation must be long to be informative — which makes it expensive, which is why it is not run.
- **Non-identifiability of the failure**: a bf16 loss spike is attributable to scan roundoff, gate saturation ($a_t \to 1$ making the state a near-integrator), or optimizer state precision. These three share the same symptom, and the standard fix (fp32 everything) suppresses all three at once, so nobody learns which one mattered.

## 7. Current Research (as of 2026)

- **Kernel libraries as the venue for the work.** `flash-linear-attention` (Songlin Yang, Yu Zhang and collaborators) and `mamba-ssm` (Gu, Dao) encode the current stability practice — fp32 chunk accumulators, log-space decay, chunk-size caps. The design rationale lives in code and issues, not papers. *(frontier — verify)*
- **Low-precision linear attention.** Ongoing work on fp8 chunkwise kernels for gated linear attention and gated DeltaNet, where the state matrix is the precision-critical object. *(frontier — verify)*
- **Non-diagonal transitions.** Gated DeltaNet (Yang, Kautz, Hatamizadeh, ICLR 2025) and its successors use rank-1-perturbed identity transitions; these fall squarely into the Mathias regime and are the most likely place for a genuine instability result.
- **Numerical-analysis crossover.** Reproducible/compensated summation (Kahan, Neumaier) and error-free transformations are well developed in the HPC literature but have not been ported into scan kernels, where the two-word accumulator would cost register pressure. No published attempt at LM scale.

## 8. Concrete Next Experiment

**Question.** At what horizon does the standard bf16-storage/fp32-accumulate chunked scan lose gradient fidelity, and does the chunk size control it?

**Scale.** One gated linear recurrence layer, $d = 1024$ channels, batch 8, horizons $T \in \{2^{10}, 2^{12}, \dots, 2^{20}\}$. Gates from a trained Mamba-2 130M checkpoint's $\Delta$ distribution (so $|a_t|$ is realistic, not synthetic), inputs $b_t$ from the same checkpoint's activations. Cost: under 20 A100-hours, dominated by the float64 references at $T=2^{20}$.

**Arms.**
- **Control:** sequential recurrence in float64 (the oracle).
- **A:** sequential fp32 loop — isolates scan-order effect from precision effect.
- **B:** Blelloch tree, bf16 inputs, fp32 accumulate.
- **C:** chunked scan, chunk $\in \{64, 256\}$, bf16 matmul + fp32 state.
- **D:** log-space chunked scan (cumsum of $\log a$ in fp32, exponentiate per chunk).

**Deciding number.** $T^{*}$ = the smallest horizon at which the median gradient cosine $\cos(\nabla_\theta \hat L, \nabla_\theta L^{(64)})$ over channels drops below $0.99$.

**Interpretation.** If $T^{*} > 2^{20}$ for arms B, C, D, the problem is empirically closed for diagonal gates and attention should move to non-diagonal transitions. If $T^{*}$ for arm C at chunk 256 is more than $4\times$ smaller than at chunk 64, chunk size is a stability knob, not just a throughput knob — which contradicts how it is currently tuned. Report $E(T)$ and $\kappa_T$ alongside, so the result is attributable to cancellation rather than to horizon per se.

## 9. Key References

- **[Foundational]** Guy E. Blelloch. *Prefix Sums and Their Applications.* Technical Report CMU-CS-90-190, Carnegie Mellon University, 1990.
- **[Foundational]** Nicholas J. Higham. *The Accuracy of Floating Point Summation.* SIAM Journal on Scientific Computing, 14(4), 1993.
- **[Foundational]** Nicholas J. Higham. *Accuracy and Stability of Numerical Algorithms*, 2nd edition. SIAM, 2002.
- **[Theory SOTA]** Roy Mathias. *The Instability of Parallel Prefix Matrix Multiplication.* SIAM Journal on Scientific Computing, 16(4), 1995.
- **[Foundational]** Albert Gu, Karan Goel, Christopher Ré. *Efficiently Modeling Long Sequences with Structured State Spaces.* ICLR 2022. — arXiv:2111.00396
- **[Foundational]** Jimmy T.H. Smith, Andrew Warrington, Scott W. Linderman. *Simplified State Space Layers for Sequence Modeling.* ICLR 2023. — arXiv:2208.04933
- **[SOTA]** Albert Gu, Tri Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* COLM 2024. — arXiv:2312.00752
- **[SOTA]** Tri Dao, Albert Gu. *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.* ICML 2024. — arXiv:2405.21060
- **[SOTA]** Antonio Orvieto, Samuel L. Smith, Albert Gu, Anushan Fernando, Caglar Gulcehre, Razvan Pascanu, Soham De. *Resurrecting Recurrent Neural Networks for Long Sequences.* ICML 2023. — arXiv:2303.06349
- **[SOTA]** Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, Yoon Kim. *Gated Linear Attention Transformers with Hardware-Efficient Training.* ICML 2024. — arXiv:2312.06635
- **[SOTA]** Maximilian Beck, Korbinian Pöppel, Markus Spanring, Andreas Auer, Oleksandra Prudnikova, Michael Kopp, Günter Klambauer, Johannes Brandstetter, Sepp Hochreiter. *xLSTM: Extended Long Short-Term Memory.* NeurIPS 2024. — arXiv:2405.04517
- **[SOTA]** Bo Peng et al. *RWKV: Reinventing RNNs for the Transformer Era.* Findings of EMNLP 2023. — arXiv:2305.13048
- **[Related]** Yi Heng Lim, Qi Zhu, Joshua Selfridge, Muhammad Firmansyah Kasim. *Parallelizing Non-Linear Sequential Models over the Sequence Length.* ICLR 2024. — arXiv:2309.12252

## 10. Worked Example

Take $d=1$, $T = 2^{17} = 131{,}072$, gate constant $a = 0.9995$, and inputs $b_t = (-1)^t$ — alternating signs, the cancellation regime.

**Exact value.** $h_T = \sum_{s=1}^{T} a^{T-s}(-1)^s$. This is a geometric series with ratio $-a$:
$$h_T \approx \frac{-1}{1+a} \cdot \big(1 - (-a)^{T}\big) \approx -0.50012 .$$

**Condition number.** $\sum_s |a^{T-s}| = \frac{1-a^{T}}{1-a} \approx \frac{1}{5\times10^{-4}} = 2000$. So
$$\kappa_T \approx \frac{2000}{0.5} = 4000 .$$

**Error budget.**

| Arm | Bound | Predicted $E$ |
|---|---|---|
| Sequential fp32 | $\gamma_T \kappa_T \approx Tu\kappa_T = 1.31\times10^{5}\cdot 5.96\times10^{-8}\cdot 4000$ | $\approx 31$ — no correct digits |
| Tree fp32 | $\gamma_{\log_2 T}\kappa_T \approx 17\cdot 5.96\times10^{-8}\cdot 4000$ | $\approx 4\times10^{-3}$ |
| Tree bf16 accumulate | $17\cdot 3.91\times10^{-3}\cdot 4000$ | $\approx 266$ — meaningless |

Two things are visible that the folklore misses.

1. The **tree beats the loop by four orders of magnitude here**, because the additive part dominates and pairwise summation is the better algorithm. The worst-case sequential bound is pessimistic — real fp32 sequential error is nearer $10^{-3}$ because roundoff signs are not adversarial — but the *ordering* holds and is measurable.
2. The bf16 row is the real constraint, and it is why every shipped kernel accumulates in fp32. The mechanism is $\kappa_T \approx 1/(1-a)$: as gates approach 1 to extend memory, the condition number grows as the inverse of the decay rate. **Longer memory and numerical conditioning trade off directly**, and no choice of scan order changes that.

Now the obstruction. Replace $b_t = (-1)^t$ with the $b_t$ from a real checkpoint. $\kappa_T$ is no longer computable in closed form; it must be measured in float64, which requires the $O(T)$ serial oracle — 131k dependent steps, no GPU parallelism. Do it for $d = 4096$ channels, 8 batch elements, at every layer, and the oracle costs more than the training run it is meant to audit. That is why $E(T)$ curves do not exist: not because the analysis is hard, but because the ground truth is serial and the model is not.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*