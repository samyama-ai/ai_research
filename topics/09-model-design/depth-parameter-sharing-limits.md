---
id: 09-model-design/depth-parameter-sharing-limits
title: "Universal Transformers and Parameter Sharing Across Depth"
topic: 09-model-design
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Universal Transformers and Parameter Sharing Across Depth

> **Topic:** Architecture & Model Design · **ID:** `09-model-design/depth-parameter-sharing-limits` · **Status:** empirically-open

## 1. Problem Statement

A standard Transformer of depth $L$ holds $L$ distinct blocks of weights. A **depth-shared** (Universal / recursive / looped) Transformer applies one block, or a small bank of $K \ll L$ blocks, repeatedly for $L$ steps. The question: **when does tying weights across depth trade favourably, and against which budget?**

Three variants, routinely conflated:

- **Measurement.** Fix a budget and compare. Which budget — parameters, training FLOPs, inference FLOPs, or memory bandwidth? Sharing wins on parameters by construction and never wins on FLOPs by construction. A claim of "better" without naming the budget is empty.
- **Method.** Find an architecture that shares depth weights and is Pareto-dominant on *at least one* axis (params at equal loss, or loss at equal training FLOPs) at $\geq 1$B parameters, without paying elsewhere.
- **Theory.** Characterise the function classes separating $K=1$ from $K=L$. Recurrence in depth buys adaptive computation and length generalisation in principle; does it buy anything a non-shared network of the same *compute* cannot get?

Solving it means: a scaling law with the sharing factor as an explicit variable, fit over $\geq 2$ orders of magnitude, that predicts when sharing is on the compute-optimal frontier.

## 2. Formal Setting

Let $f_\theta: \mathbb{R}^{n \times d} \to \mathbb{R}^{n \times d}$ be one Transformer block with parameters $\theta$. Define the sharing schedule $\sigma: \{1,\dots,L\} \to \{1,\dots,K\}$, so layer $\ell$ uses $\theta_{\sigma(\ell)}$. $K=L$ with $\sigma=\mathrm{id}$ is the standard Transformer; $K=1$ is the Universal Transformer.

$$h^{(\ell)} = f_{\theta_{\sigma(\ell)}}\!\left(h^{(\ell-1)} + p^{(\ell)}\right), \qquad \ell = 1,\dots,L$$

with $p^{(\ell)}$ a per-step depth embedding — cheap, unshared, and necessary: without it the shared block cannot know its position in the stack.

**Measured quantities.**

- $N = K \cdot |\theta| + N_{\text{emb}}$ — non-embedding parameters counted from the checkpoint, embeddings reported separately (they dominate at small $N$ and distort every sharing comparison).
- $C_{\text{train}} \approx 6 \cdot N_{\text{act}} \cdot D$, where $N_{\text{act}} = L\cdot|\theta|$ is *activated* parameters per token and $D$ is tokens. Sharing reduces $N$, not $N_{\text{act}}$.
- $\mathcal{L}(N, C)$ — validation cross-entropy in nats/token on a held-out corpus, same tokenizer and same data order across arms.
- **Sharing ratio** $\rho = L/K \in [1, L]$. The object to fit is $\mathcal{L}(N_{\text{act}}, \rho, D)$, not $\mathcal{L}(N,D)$.
- $T_{\text{tok}}$ — wall-clock ms/token at batch 1 and at batch 256, measured, not derived. This is where sharing's real payoff lives: weights fit in cache.

**Adaptive depth.** With a halting unit (ACT) producing $q^{(\ell)} \in [0,1]$, the model halts at $\tau = \min\{\ell: \sum_{j\le\ell} q^{(j)} \ge 1-\epsilon\}$, giving per-token depth $\tau_i$ and expected compute $\mathbb{E}_i[\tau_i]$.

**Assumptions known to be violated.**
1. *That one block is expressive enough to be all layers.* Probing shows standard Transformers specialise by depth (early = surface/syntax, late = task); tying forces a single fixed point of behaviour.
2. *That the scaling law is separable in $N$ and $\rho$.* Unverified; no published fit includes $\rho$.
3. *That parameters are the binding constraint.* True for on-device serving, false for frontier training, where FLOPs and data bind.
4. *That depth $L$ can be extrapolated at inference.* Empirically it degrades outside the training distribution of $L$ unless explicitly trained with random $L$.

## 3. State of the Art

**Theory (established).** Pérez, Barceló & Marinković, *Attention is Turing-Complete* (JMLR 2021) — a fixed-size Transformer with unbounded decoding steps is Turing-complete under arbitrary precision and hard attention; the assumptions are strong and not met by real models. Merrill & Sabharwal, *The Expressive Power of Transformers with Chain of Thought* (ICLR 2024) and *A Little Depth Goes a Long Way* (2025) place constant-depth Transformers inside uniform $\mathsf{TC}^0$ and show log-depth suffices for problems (e.g. graph connectivity) believed outside it. This is the cleanest argument for depth recurrence: *more steps* changes the complexity class; *more parameters* does not.

**Empirical (established).** Dehghani et al., *Universal Transformers* (ICLR 2019) — gains on bAbI, LAMBADA, and algorithmic tasks; the LM-scale claim was never made. Lan et al., *ALBERT* (ICLR 2020) — full cross-layer sharing, ALBERT-xxlarge 235M vs BERT-large 334M. Csordás et al., *MoEUT* (NeurIPS 2024) — the first depth-shared LM competitive with a standard Transformer at matched parameters up to ~1B, by making the shared block a mixture-of-experts so width scales without quadratic compute.

**Claimed but unablated.** ALBERT's headline is confounded: sharing, factorized embeddings, and sentence-order prediction all changed at once; no clean single-factor ablation was published. Bae et al., *Relaxed Recursive Transformers* (ICLR 2025) report recursive Gemma/TinyLlama variants beating reduced-size baselines with layer-wise LoRA, but the baselines are uptrained-from-pruned, not trained from scratch at matched FLOPs. Geiping et al. (2025), *Scaling up Test-Time Compute with Latent Reasoning* (Huginn-3.5B, 800B tokens) show reasoning benchmarks improving with recurrence count at test time — a benchmark number, with no matched-FLOPs non-recurrent control.

## 4. What Is Known

- **Full sharing costs quality at fixed depth and width.** Takase & Kiyono (*Lessons on Parameter Sharing across Layers in Transformers*, SustaiNLP 2023) show that on WMT translation and LM, sharing *all* layers underperforms; partial schedules ($K=2$ or $3$, cycled) recover most of the gap at a fraction of the parameters.
- **ALBERT scale:** ALBERT-base 12M vs BERT-base 108M non-embedding parameters (~9× reduction), GLUE and SQuAD within a few points; ALBERT-xxlarge reaches SQuAD 2.0 F1 ≈ 88 with 235M parameters — but trains and serves *slower* than BERT-large, because $N_{\text{act}}$ rose.
- **Sharing helps most below 1B.** Liu et al., *MobileLLM* (ICML 2024) report immediate block-wise sharing giving ~0.7–0.8 points average zero-shot accuracy at 125M/350M with no added memory-movement cost.
- **Looped models generalise in length where non-shared ones do not.** Yang et al., *Looped Transformers are Better at Learning Learning Algorithms* (ICLR 2024) match a 12-layer Transformer on in-context regression with a single looped block; Fan et al., *Looped Transformers for Length Generalization* (ICLR 2025) show large out-of-distribution length gains on $n$-digit arithmetic and parity when the loop count scales with input length.
- **Compositional tasks favour recurrence.** Ontañón et al., *Making Transformers Solve Compositional Tasks* (ACL 2022) find weight sharing among the largest single contributors to COGS/CFQ generalisation.
- **The parameter/compute ratio is the fundamental tension** (Csordás et al. 2024): to match parameter count with $\rho=L$, width must grow by $\sqrt{L}$, and per-token FLOPs grow with it.

## 5. What Is Not Known

- **Empirically open (the main gap).** No published scaling law of the form $\mathcal{L}(N_{\text{act}}, \rho, D)$ fit across $\geq 2$ decades. Nobody has run the matched-FLOPs, matched-data comparison of $\rho \in \{1,2,4,L\}$ at $\geq 7$B. The experiment is entirely runnable; it costs money and produces a negative-looking headline, so it is unfunded.
- **Empirically open.** Whether test-time depth extrapolation ($L_{\text{infer}} > L_{\text{train}}$) yields real gains beyond a self-consistency baseline at equal FLOPs. Huginn shows the curve; the control is missing.
- **Theoretically open.** No separation theorem between $K=1$ and $K=L$ at *fixed total compute* and finite precision. All existing separations are about number of steps, not about weight tying.
- **Theoretically open.** Whether depth-shared Transformers converge to a fixed point in the DEQ sense (Bai et al., NeurIPS 2019) when trained as ordinary deep networks, and whether that fixed point is what the model uses.
- **Methodologically blocked.** "Layers do different things" has no agreed measurement. Without a validated depth-specialisation metric, the claim that sharing destroys specialisation is untestable.

## 6. Why It Is Hard

**The comparison is non-identifiable under any single budget.** Sharing changes $N$ and $N_{\text{act}}$ in opposite directions. Fix parameters and the shared model uses more FLOPs; fix FLOPs and it uses fewer parameters; fix depth and width and it is strictly a constrained subspace of the unshared model, so it can only lose. Every published comparison picks the budget that flatters its arm, and all three choices are defensible. There is no neutral axis — you must report the full Pareto surface, which multiplies the run count by the grid size in $\rho$.

Secondary: **confounded measurement.** ALBERT-style results bundle sharing with embedding factorization and objective changes. And **evaluation drift** — algorithmic and length-generalisation benchmarks, where sharing wins hardest, correlate weakly with LM loss, so a win there does not transfer to the metric that governs deployment.

## 7. Current Research (as of 2026)

- **Compute-matched depth recurrence at scale.** Geiping and collaborators (Maryland/ELLIS) on recurrent-depth latent reasoning; the open question they inherit is the matched-FLOPs control. *(frontier — verify)*
- **Mixture-of-Recursions** (Bae, Csordás, and co-authors, 2025): per-token adaptive recursion depth with routing, reviving ACT with modern routers. Early results claim Pareto gains at ~1B. *(frontier — verify)*
- **Relaxed sharing.** LoRA or low-rank per-layer deltas on a shared base — the interpolation between $K=1$ and $K=L$, parameterised by delta rank. This is the most likely practical resolution.
- **On-device serving.** Sub-1B models where weight memory, not FLOPs, binds; sharing is already shipping here.
- **Expressivity.** Merrill & Sabharwal and the formal-languages community on log-depth and chain-of-thought separations.

## 8. Concrete Next Experiment

**Scale.** Train a 4-arm grid at three sizes — $N_{\text{act}} \in \{350\text{M}, 1.4\text{B}, 7\text{B}\}$ — on an identical 300B-token corpus, identical tokenizer, identical data order, identical LR schedule tuned per arm.

**Arms.** $\rho \in \{1, 2, 4, L\}$ at fixed $L$ and fixed $d_{\text{model}}$, so $N_{\text{act}}$ and training FLOPs are *equal across all four arms* and only $N$ differs (by up to $L\times$). Each shared arm carries a per-step depth embedding.

**Control.** $\rho=1$ (standard Transformer) at the same $N_{\text{act}}$ — the FLOPs-matched control, not the parameter-matched one. Report the parameter-matched control ($\rho=L$ widened to equal $N$) as a second, separate curve.

**The deciding number.** $\Delta\mathcal{L}(\rho) = \mathcal{L}(\rho) - \mathcal{L}(1)$ in nats/token, plotted against $N_{\text{act}}$. The question is the **sign of $d\,\Delta\mathcal{L}(\rho{=}L)/d\log N_{\text{act}}$**. If the loss penalty for full sharing *shrinks* with scale, depth sharing is a frontier architecture and the field should invest. If it *grows*, sharing is permanently an edge-deployment compression technique. Current evidence at 350M puts $\Delta\mathcal{L}(L) \approx 0.02$–$0.05$ nats; the experiment is decided if the 7B point falls outside a $\pm 0.01$ nat band around the trend line from the two smaller points.

Cost: roughly $12 \times 6 \times N_{\text{act}} \times 3\times10^{11}$ FLOPs summed over the grid — a few hundred thousand GPU-hours. Cheap relative to what it settles.

## 9. Key References

- **[Foundational]** Dehghani, Gouws, Vinyals, Uszkoreit, Kaiser. *Universal Transformers.* ICLR 2019. — arXiv:1807.03819
- **[Foundational]** Lan, Chen, Goodman, Gimpel, Sharma, Soricut. *ALBERT: A Lite BERT for Self-supervised Learning of Language Representations.* ICLR 2020. — arXiv:1909.11942
- **[Foundational]** Bai, Kolter, Koltun. *Deep Equilibrium Models.* NeurIPS 2019. — arXiv:1909.01377
- **[Theory]** Pérez, Barceló, Marinković. *Attention is Turing-Complete.* JMLR 22(75), 2021.
- **[Theory]** Merrill, Sabharwal. *The Expressive Power of Transformers with Chain of Thought.* ICLR 2024.
- **[SOTA]** Csordás, Irie, Schmidhuber, Potts, Manning. *MoEUT: Mixture-of-Experts Universal Transformers.* NeurIPS 2024.
- **[SOTA]** Bae, Fisch, Harutyunyan, Ji, Kim, Schuster. *Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA.* ICLR 2025.
- **[SOTA]** Geiping, McLeish, Jain, Kirchenbauer, Singh, Bartoldson, Kailkhura, Bhatele, Goldstein. *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach.* 2025.
- **[Empirical]** Takase, Kiyono. *Lessons on Parameter Sharing across Layers in Transformers.* SustaiNLP @ ACL 2023.
- **[Empirical]** Yang, Lee, Nowak, Papailiopoulos. *Looped Transformers are Better at Learning Learning Algorithms.* ICLR 2024.
- **[Empirical]** Fan, Du, Wang, Deng, Ma. *Looped Transformers for Length Generalization.* ICLR 2025.
- **[Empirical]** Liu et al. *MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases.* ICML 2024.
- **[Empirical]** Ontañón, Ainslie, Cvicek, Fisher. *Making Transformers Solve Compositional Tasks.* ACL 2022.
- **[Survey]** Giannou, Rajput, Sohn, Lee, Lee, Papailiopoulos. *Looped Transformers as Programmable Computers.* ICML 2023.

## 10. Worked Example

Take a 24-layer model, $d_{\text{model}} = 1024$, $d_{\text{ff}} = 4096$. One block is $\approx 12 d^2 = 12.6$M parameters.

| Arm | $K$ | $N$ (non-emb) | $N_{\text{act}}$/token | Train FLOPs @ 300B tok |
|---|---|---|---|---|
| A: standard | 24 | 302M | 302M | $5.4\times10^{20}$ |
| B: shared, same width | 1 | 12.6M | 302M | $5.4\times10^{20}$ |
| C: shared, param-matched | 1 | 302M ($d\!=\!5017$) | 7.2B | $1.3\times10^{22}$ |

Arm B costs the same to train as A and is 24× smaller on disk. Arm C has A's parameter count and costs **24× more compute per token**, because widening the single block to absorb 302M parameters raises per-layer FLOPs by the same factor it raised parameters, and that layer runs 24 times.

Now the obstruction. Report B against A: B loses, say, 0.03 nats — "sharing hurts." Report B against a *parameter-matched* standard Transformer (a 1-layer, 12.6M model): B wins by a landslide — "sharing is free depth." Report C against A at equal parameters: C may win on loss, and the paper says "matched parameters, better loss," while quietly spending 24× the FLOPs. All three sentences are true of the same three runs.

MoEUT's contribution is precisely to break this: replace arm C's dense widening with sparse experts, so parameters grow without activated FLOPs growing, landing near A on both axes at ~1B. Whether that holds at 7B and 70B is the open part — and it is open because nobody has paid for the row of the table above at the scales where the answer matters.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*