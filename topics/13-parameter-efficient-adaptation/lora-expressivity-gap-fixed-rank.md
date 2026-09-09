---
id: 13-parameter-efficient-adaptation/lora-expressivity-gap-fixed-rank
title: "LoRA Expressivity Gap at Fixed Rank"
topic: 13-parameter-efficient-adaptation
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# LoRA Expressivity Gap at Fixed Rank

> **Topic:** Parameter-Efficient Adaptation · **ID:** `13-parameter-efficient-adaptation/lora-expressivity-gap-fixed-rank` · **Status:** partially-solved

## 1. Problem Statement

Low-Rank Adaptation (LoRA) constrains the fine-tuning update of each weight matrix to rank $r$. The question is what that constraint costs, and why.

Three variants, routinely conflated:

- **Theory variant.** Given a frozen pretrained network $f_{W_0}$, a target function class, and rank budget $r$, what is $\min_{\text{rank} \le r} \; \mathcal{L}$ minus $\min_{\text{unconstrained}} \; \mathcal{L}$? Solving it means a tight two-sided bound as a function of $r$, depth $L$, width $D$, and the spectrum of the required update.
- **Measurement variant.** Given an observed loss gap between LoRA at rank $r$ and full fine-tuning, decompose it into an *approximation* term (no rank-$r$ update achieves the loss) and an *optimization* term (a rank-$r$ update exists but SGD/Adam under the LoRA parameterization does not find it). Solving it means an estimator that separates the two.
- **Method variant.** Find a rank-$r$-parameter-count adapter that closes the measured gap. This is where DoRA, rsLoRA, LoRA+, AdaLoRA and MoRA live.

The problem is *partially solved*: exact-representation thresholds are proved for restricted architectures, and the empirical gap is well characterized in the high-data regime. What is unresolved is which of approximation or optimization dominates at ranks people actually use.

## 2. Formal Setting

Frozen weight $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$; LoRA update

$$\Delta W = \tfrac{\alpha}{r} B A, \qquad B \in \mathbb{R}^{d_{\text{out}} \times r},\; A \in \mathbb{R}^{r \times d_{\text{in}}},\; A \sim \mathcal{N}(0,\sigma^2),\; B = 0 .$$

Let $\mathcal{M}$ index adapted modules, $\mathcal{L}_r = \{\Delta : \operatorname{rank}(\Delta_m) \le r \;\forall m \in \mathcal{M}\}$, and $\mathcal{L}_\infty$ the unconstrained set.

**Expressivity gap** (the quantity of interest):
$$G_{\text{expr}}(r) = \min_{\Delta \in \mathcal{L}_r} \mathcal{L}_{\text{val}}(W_0 + \Delta) \;-\; \min_{\Delta \in \mathcal{L}_\infty} \mathcal{L}_{\text{val}}(W_0 + \Delta).$$

**Measured gap** (the quantity actually reported): $\hat G(r) = \mathcal{L}_{\text{val}}(\hat W_{\text{LoRA}}(r)) - \mathcal{L}_{\text{val}}(\hat W_{\text{FT}})$, each arm tuned over learning rate, $\alpha$, epochs, module set. Always $\hat G(r) \ge G_{\text{expr}}(r)$; the slack is the optimization term $G_{\text{opt}}(r)$.

**Computable upper bound.** For the full-FT solution $\Delta_{\text{FT}}$ and per-module truncated SVD $\Pi_r$,
$$G_{\text{expr}}(r) \;\le\; \mathcal{L}_{\text{val}}(W_0 + \Pi_r \Delta_{\text{FT}}) - \mathcal{L}_{\text{val}}(W_0 + \Delta_{\text{FT}}) \;=:\; \tilde G(r),$$
since $\Pi_r\Delta_{\text{FT}} \in \mathcal{L}_r$. $\tilde G(r)$ is measurable with one full-FT run plus $|\{r\}|$ forward passes.

**Spectral quantities, as measured.** Singular values $\sigma_1 \ge \dots \ge \sigma_n$ of $\Delta_{\text{FT}}^{(m)}$; retained energy $E_m(r) = \sum_{i\le r}\sigma_i^2 / \sum_i \sigma_i^2$; effective rank $\operatorname{erank} = \exp\big(-\sum_i p_i \log p_i\big)$, $p_i = \sigma_i / \sum_j \sigma_j$.

**Assumptions, with the violated ones flagged.**
1. $\hat W_{\text{FT}}$ approximates the unconstrained optimum — *violated*: full FT is itself a stochastic, budget-limited optimizer.
2. Rank constrains function change — *violated*: composing rank-$r$ adapters across $L$ layers yields functional perturbations of rank up to $\min(rL, D)$, so per-matrix rank understates the model's expressivity.
3. The gap is scale-free — *violated*: it depends strongly on dataset size relative to adapter capacity (§4).
4. Validation loss ranks methods consistently with downstream metrics — *violated*: LoRA/full-FT orderings flip between NLL and pass@1 in reported sweeps.

## 3. State of the Art

**Theory SOTA (established).** Zeng & Lee, *The Expressive Power of Low-Rank Adaptation* (ICLR 2024, arXiv:2310.17513): for fully connected and transformer networks, a frozen model of depth $L$ can be made to *exactly* represent any target model of depth $\bar L$ once the LoRA rank reaches order $\tfrac{D}{2}\lceil L/\bar L\rceil$ — half the width, discounted by the depth ratio. Below threshold they bound the residual by the tail singular values of the discrepancy matrices. This is a constructive existence result about the *loss landscape*, not about what Adam finds.

**Empirical SOTA (established, independently reproduced).** Biderman et al., *LoRA Learns Less and Forgets Less* (TMLR 2024, arXiv:2405.09673): at 7B scale, LoRA trails full FT on code and math IFT and continued pretraining, but forgets pretraining abilities less. Shuttleworth et al., *LoRA vs Full Fine-tuning: An Illusion of Equivalence* (arXiv:2410.21228): LoRA solutions contain *intruder dimensions* — high-singular-value directions near-orthogonal to the pretrained spectrum — absent from full-FT solutions at matched loss.

**Claimed but not fully ablated.** DoRA (ICML 2024), rsLoRA (arXiv:2312.03732), LoRA+ (ICML 2024) and AdaLoRA (ICLR 2023) each report closing part of the gap. All of them change *optimization* (scaling, per-matrix LR ratio, rank allocation) without enlarging $\mathcal{L}_r$, so their gains bound $G_{\text{opt}}$ from below and say nothing about $G_{\text{expr}}$ — a point their ablation tables do not make.

**Benchmark-number-only.** The Thinking Machines Lab post *LoRA Without Regret* (2025, non-peer-reviewed) reports LoRA matching full FT when applied to all layers including MLP/MoE and when dataset size stays under adapter capacity, with roughly $10\times$ the full-FT learning rate. Widely cited, no independent replication at the reported scales.

## 4. What Is Known

- **Exact-representation threshold exists and scales with width, not task difficulty** (Zeng & Lee, ICLR 2024). Ranks in the thousands for a $D=4096$ model — orders above practice.
- **Intrinsic dimension of adaptation is tiny for small tasks.** Aghajanyan et al. (ACL 2021, arXiv:2012.13255): RoBERTa-large reaches 90% of full-FT MRPC performance in a $d_{90}=207$-dimensional random subspace; QQP needs 774. Measured at 355M parameters, GLUE-scale data.
- **Full-FT updates are high-rank.** Biderman et al. report spectra of $\Delta_{\text{FT}}$ at Llama-2-7B whose rank at 90% energy exceeds typical LoRA ranks by one to two orders of magnitude; MLP matrices are the highest-rank.
- **The gap is data-size dependent, not task dependent.** Same paper: on instruction tuning ($\sim$100k examples) rank 256 over all modules nearly closes the gap; on continued pretraining (20B tokens of StarCoder-Python) it does not.
- **Module coverage dominates rank.** Attention-only LoRA is consistently worse than attention+MLP at matched parameter count, reproduced across Biderman et al., DoRA, and the 2025 Thinking Machines report.
- **Optimization pathology is real and fixable.** LoRA's $\alpha/r$ scaling collapses effective step size as $r$ grows; $\alpha/\sqrt{r}$ (rsLoRA) removes it. Hayou et al. (ICML 2024) show the $B$/$A$ learning-rate ratio should scale with width. Both are pure-$G_{\text{opt}}$ interventions.
- **Regularization side-effect.** LoRA's lower forgetting and higher output diversity are reproduced; the constraint is not purely a cost.

## 5. What Is Not Known

- **Empirically open.** The decomposition of $\hat G(r)$ into $G_{\text{expr}}$ and $G_{\text{opt}}$ at $r \in \{16,\dots,256\}$ on a 7–8B model with $\ge 10^9$ training tokens. The experiment is a single full-FT run plus SVD projections (§8). Nobody has published it with $\tilde G(r)$ reported alongside $\hat G(r)$.
- **Theoretically open.** No lower bound. There is no theorem stating that some natural adaptation target *requires* rank $\ge r_0$ for a pretrained transformer — only upper bounds on the rank that suffices. Without a lower bound, "LoRA is expressivity-limited" has no proof.
- **Theoretically open.** Whether cross-layer composition (assumption 2) means per-matrix rank $r$ with $L$ layers is equivalent in power to per-matrix rank $rL$ in one layer. Zeng & Lee's depth discount hints yes; no tight statement exists.
- **Methodologically blocked.** "Capacity of a rank-$r$ adapter" has no agreed operational definition. Bits-per-parameter estimates (Allen-Zhu & Li, arXiv:2404.05405, $\approx$2 bits/param for knowledge storage) are measured on synthetic factual corpora and have not been validated as a predictor of the LoRA gap.
- **Methodologically blocked.** Whether intruder dimensions are a cause of degradation or a benign reparameterization; the correlational evidence does not separate them.

## 6. Why It Is Hard

**The obstruction is confounded measurement plus non-identifiability.** Every published gap is $\hat G(r)$, which sums an approximation term and an optimization term that the experiment cannot separate. The two respond to the same knobs in opposite directions: raising $r$ enlarges $\mathcal{L}_r$ (shrinks $G_{\text{expr}}$) *and*, under $\alpha/r$ scaling, shrinks the effective learning rate (grows $G_{\text{opt}}$). Papers reporting flat or non-monotone loss-vs-rank curves are reading the sum of two curves moving oppositely.

Compounding: LoRA's optimum is non-identifiable — $(BA) = (BQ)(Q^{-1}A)$ for any invertible $Q$ — so spectral diagnostics on the *learned* $BA$ are basis-dependent, and $B=0$ initialization puts the optimizer at a saddle where the first steps are determined by random $A$. Finally, the control arm is expensive: an honest full-FT comparison at 8B with $10^9$ tokens is $\sim$$10^{21}$ FLOPs per seed, and the gap sits inside seed variance unless three or more seeds are run.

## 7. Current Research (as of 2026)

- **Optimization-side fixes** (rsLoRA, LoRA+, DoRA, LoRA-RITE, preconditioned/Riemannian LoRA optimizers). Consensus is shifting toward "most of the measured gap is $G_{\text{opt}}$" — *(frontier — verify)*.
- **Capacity-matched adapter design**: MoRA (high-rank square updates at fixed parameter count) and sparse/structured alternatives test whether rank *per se*, or parameter count, binds.
- **High-rank-by-composition training**: ReLoRA (ICLR 2024) and GaLore (ICML 2024) accumulate low-rank steps into a high-rank update — direct probes of assumption 2.
- **Spectral forensics** of $\Delta_{\text{FT}}$ vs $\Delta_{\text{LoRA}}$ following Shuttleworth et al.; active at MIT CSAIL and Princeton *(frontier — verify)*.
- **RL fine-tuning**, where gradient information per step is small: multiple groups report LoRA at low rank matching full FT for RLHF/RLVR, consistent with a data-capacity account rather than an expressivity one *(frontier — verify)*.

## 8. Concrete Next Experiment

**Decomposing $\hat G(r)$ with an SVD-projection control.**

- **Scale.** Llama-3.1-8B, continued pretraining on 2B tokens of a held-out domain corpus (e.g. Python from The Stack v2). One full-FT run, 3 seeds. LoRA arms at $r \in \{8, 32, 128, 512\}$ on all linear modules, $\alpha/\sqrt{r}$ scaling, LR tuned per arm over a 4-point grid, 3 seeds.
- **Control arm.** $\Pi_r \Delta_{\text{FT}}$ — the full-FT update truncated per matrix to rank $r$. Zero training; evaluation only. Gives $\tilde G(r) \ge G_{\text{expr}}(r)$.
- **Deciding number.** The ratio $\rho(r) = \tilde G(r) / \hat G(r)$ in validation nats/token at $r=32$.
  - $\rho(32) > 0.7$: the gap is genuinely expressivity-bound. Optimization fixes are capped; rank must rise.
  - $\rho(32) < 0.2$: the gap is optimization. A rank-32 solution within 0.01 nats of full FT exists and Adam misses it; the research target is the optimizer, not the parameterization.
- **Cost.** $\approx 1.1\times$ one full-FT run plus 12 LoRA runs plus 4 evaluation passes. Report $\hat G$, $\tilde G$, $E_m(r)$ per module type, and seed spread; the result is only meaningful if $\hat G$ exceeds $2\times$ the seed standard deviation.

## 9. Key References

- **[Foundational]** Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022. — arXiv:2106.09685
- **[Foundational]** Aghajanyan, Gupta, Zettlemoyer. *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning.* ACL 2021. — arXiv:2012.13255
- **[Theory SOTA]** Zeng, Lee. *The Expressive Power of Low-Rank Adaptation.* ICLR 2024. — arXiv:2310.17513
- **[Empirical SOTA]** Biderman, Portes, Ortiz, Paul, Greengard, Jennings, King, Havens, Chiley, Frankle, Blakeney, Cunningham. *LoRA Learns Less and Forgets Less.* TMLR 2024. — arXiv:2405.09673
- **[SOTA]** Shuttleworth, Andrews, Agrawal, Solar-Lezama, et al. *LoRA vs Full Fine-tuning: An Illusion of Equivalence.* 2024. — arXiv:2410.21228
- **[Method]** Liu, Wang, Yin, Molchanov, Wang, Cheng, Chen. *DoRA: Weight-Decomposed Low-Rank Adaptation.* ICML 2024. — arXiv:2402.09353
- **[Method]** Kalajdzievski. *A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA.* 2023. — arXiv:2312.03732
- **[Method]** Hayou, Ghosh, Yu. *LoRA+: Efficient Low Rank Adaptation of Large Models.* ICML 2024. — arXiv:2402.12354
- **[Method]** Zhang, Chen, Bukharin, He, Cheng, Chen, Zhao. *AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning.* ICLR 2023. — arXiv:2303.10512
- **[Method]** Lialin, Muckatira, Shivagunde, Rumshisky. *ReLoRA: High-Rank Training Through Low-Rank Updates.* ICLR 2024. — arXiv:2307.05695
- **[Capacity]** Allen-Zhu, Li. *Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws.* 2024. — arXiv:2404.05405
- **[Survey]** Han, Gao, Liu, Zhang, Zhang. *Parameter-Efficient Fine-Tuning for Large Models: A Comprehensive Survey.* TMLR 2024. — arXiv:2403.14608

## 10. Worked Example

**Llama-3-8B, rank 16, all linear modules.** Per layer: $d=4096$, GQA key/value width $1024$, MLP intermediate $14336$. LoRA parameters $= r(d_{\text{in}} + d_{\text{out}})$ per matrix:

```
q_proj    16 × (4096+4096)  =   131,072
k_proj    16 × (4096+1024)  =    81,920
v_proj    16 × (4096+1024)  =    81,920
o_proj    16 × (4096+4096)  =   131,072
gate/up/down  3 × 16 × (4096+14336) =  884,736
per layer                   = 1,310,720
× 32 layers                 = 41,943,040   (0.52% of 8.03B)
```

Now the obstruction. Two accounts predict opposite things from the *same* 41.9M number.

- **Expressivity account.** The rank-16 constraint per matrix caps retained energy of $\Delta_{\text{FT}}$. For a `down_proj` with $\operatorname{erank} \approx 300$ (the regime Biderman et al. report at 7B), $E(16)$ is a small fraction, so $\tilde G(16)$ should be large. Prediction: the gap appears immediately, at any dataset size.
- **Capacity account.** At $\approx$2 bits/parameter (Allen-Zhu & Li), 41.9M parameters store $\approx 84$ Mbit $\approx 10.5$ MB. A 2B-token corpus carrying even 0.1 bits/token of genuinely new information demands 200 Mbit — $2.4\times$ over budget. Prediction: no gap below $\sim$840M tokens at that rate, then a gap that grows linearly with excess data, *independent of the spectrum*.

Both accounts fit every published LoRA-vs-full-FT curve, because those curves report only $\hat G(r)$ at a single dataset size. They differ on one measurement: $\tilde G(16)$ at 200M tokens. Expressivity predicts $\tilde G(16) \approx \hat G(16) > 0$; capacity predicts $\tilde G(16) \approx 0$ with $\hat G(16) \approx 0$ too, and both rising only past 840M tokens. That single evaluation — a truncated-SVD projection of an existing full-FT checkpoint, costing four forward passes — is what no published sweep reports, and it is why the problem is still open.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*