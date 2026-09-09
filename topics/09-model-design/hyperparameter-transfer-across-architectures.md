---
id: 09-model-design/hyperparameter-transfer-across-architectures
title: "Hyperparameter Transfer Across Architecture Families"
topic: 09-model-design
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Hyperparameter Transfer Across Architecture Families

> **Topic:** Architecture & Model Design · **ID:** `09-model-design/hyperparameter-transfer-across-architectures` · **Status:** partially-solved

## 1. Problem Statement

μP (maximal update parametrization) makes the optimal learning rate approximately invariant to width *within one architecture*. The open problem is the cross-family case: given a tuned hyperparameter vector for architecture family $A$ (say a pre-LN Transformer), predict the optimum for family $B$ (Mamba/SSM, MoE, hybrid attention–recurrence, diffusion transformer) without re-running the sweep.

Three variants, different difficulty:

- **Measurement.** Is there a well-defined "same hyperparameter" across families at all? Learning rate is shared, but $\Delta_t$ init in an SSM, router temperature in an MoE, and QK-norm gain in a Transformer have no counterpart in the other. Partially blocked.
- **Method.** Find a parametrization $\mathcal{P}$ (per-tensor init scale, LR multiplier, forward multiplier) such that the argmin of validation loss over the shared hyperparameters is family-invariant. Empirically open.
- **Theory.** Prove that a family-invariant parametrization exists, or that the infinite-width limits of two families induce genuinely different optima. Theoretically open.

Solved means: tune on family $A$ at $\le 10^8$ params, apply the map, and land within the noise floor of a direct sweep on family $B$ at $\ge 10^{10}$ params.

## 2. Formal Setting

A model is $f_\theta$ with $\theta = \{W^{(l)}\}_{l=1}^{L}$, layer $l$ having fan-in $n_l$ and fan-out $m_l$. A parametrization assigns each tensor a triple

$$W^{(l)} = \alpha_l \, \tilde W^{(l)}, \quad \tilde W^{(l)}_{ij}\sim\mathcal{N}(0,\sigma_l^2), \quad \eta_l = c_l\,\eta_0 ,$$

with **measured** quantities: $\alpha_l$ read from the forward code, $\sigma_l^2$ the empirical variance of the init tensor, $\eta_l$ the per-tensor LR actually passed to the optimizer.

μP fixes the width exponents so that, for hidden matrices, $\sigma_l^2 = \Theta(1/n_l)$, $\alpha_l=\Theta(1)$, and under Adam $c_l = \Theta(1/n_l)$; readout uses $\alpha_l = \Theta(1/n_l)$, $c_l=\Theta(1/n_l)$; embeddings $\Theta(1)$. The defining desideratum is measurable: the coordinate-wise activation change per step

$$\delta h^{(l)}_t = \tfrac{1}{m_l}\big\|h^{(l)}_t - h^{(l)}_{t-1}\big\|_1 \;=\; \Theta(1) \quad \text{as } n\to\infty,$$

logged at fixed $t$ (say steps 1, 10, 100) across widths. Transfer quality is

$$\mathrm{Gap}(A\!\to\!B) = \mathcal{L}_B\big(\eta^\star_A\big) - \min_{\eta}\mathcal{L}_B(\eta),$$

in nats/token, against a seed-noise floor $\varepsilon$ estimated from $\ge 3$ seeds. Declare transfer if $\mathrm{Gap} < \varepsilon$ *and* $|\log_2(\eta^\star_A/\eta^\star_B)| < 1$.

Assumptions, with the violated ones flagged:

1. A single width scalar $n$ indexes the family. **Violated** for MoE ($n$, experts $E$, top-$k$), for SSMs ($d_{\text{model}}$, $d_{\text{state}}$, $d_{\text{conv}}$, low-rank $\Delta$ dimension), and for GQA ($n_{\text{kv}}$ fixed while $n$ grows).
2. Every tensor is classifiable as input-like / hidden-like / output-like. **Violated** by tensors whose fan-in scales but whose fan-out is pinned (SSM $\Delta$-projection, router logits: fan-out $=E$).
3. Training is one epoch at fixed token budget with the loss dominated by the feature-learning limit. **Violated**: data-to-parameter ratio changes the optimum (Chinchilla-regime effects), and LR schedule length interacts with $\eta^\star$.
4. The optimum is unique and smooth. Approximately true in $\log\eta$; the basin is asymmetric — overshooting is far more costly than undershooting.

## 3. State of the Art

**Established (ablated, independently reproduced).** μP width transfer for Transformers, Yang et al., *Tensor Programs V*, NeurIPS 2021 — LR transferred from a 40M proxy to a 6.7B GPT-3 variant, beating the published 6.7B baseline. Independently stress-tested by Lingle, *A Large-Scale Exploration of μ-Transfer* (2024): base LR transfers cleanly across width for standard decoder-only Transformers up to 1.2B, including with RMSNorm and Lion; the same paper reports transfer **failure** for trainable norm gains and for large attention-logit scales — a within-family counterexample.

**Established, weaker.** Depth transfer for residual networks with $1/\sqrt{L}$ branch scaling: Bordelon et al., ICLR 2024, and Yang, Yu, Zhu, Hayou, *Tensor Programs VI*, ICLR 2024. Holds for blocks of depth 1; the ICLR 2024 depth-μP paper itself notes the limit degrades for multi-layer blocks.

**Claimed but under-ablated.** Everett et al., *Scaling Exponents Across Parameterizations and Optimizers*, ICML 2024, is the largest controlled sweep (tens of thousands of runs, up to 26.8B params) and reports that several parametrizations besides μP transfer once per-layer LRs and Adam's $\epsilon$ are handled, and that the alignment assumption underlying μP does not hold as stated. Cross-*family* claims are the weak spot everywhere: MoE and SSM μP recipes circulate in model reports (Cerebras-GPT 2023 used μP for Transformers; several 2024–2025 SSM/hybrid releases assert "we use μP") without an ablation showing the transferred optimum matches a direct sweep on that family.

**Benchmark-number-only.** Empirical hyperparameter scaling laws — DeepSeek LLM (Bi et al., 2024) fitting $\eta^\star, B^\star$ as power laws in compute; MiniCPM (Hu et al., 2024); StepLaw / *Predictable Scale* (Li et al., 2025). These are fits on one family's grid, reported as a curve, with no held-out family.

## 4. What Is Known

- Within decoder-only Transformers, μP holds optimal LR fixed across widths $256 \to 4096$ (16× width, ~$10^2$× params); Lingle (2024) finds the optimum stable near $\eta^\star \approx 2^{-8}$ for the base width used, drifting less than one octave over that range.
- Transfer is *not* free across all hyperparameters. In *Tensor Programs V*, the transferable set is init scale, LR, multipliers; batch size and schedule transfer only approximately, and regularization (weight decay, dropout) does not transfer — it is data-size-dependent.
- Optimal LR falls with data at fixed model size. DeepSeek LLM (2024) fits $\eta^\star \propto C^{-0.125}$ and $B^\star \propto C^{0.33}$ over $10^{17}$–$10^{20}$ FLOPs — so a family comparison at unmatched token budgets is confounded by construction.
- Attention-logit growth and output-logit growth cause instabilities whose onset shifts with LR and scale; Wortsman et al., ICLR 2024, reproduce both at $\le$ 1.2B and show QK-norm and z-loss move the divergence threshold. Family changes that alter these mechanisms move $\eta^\star$ for reasons unrelated to width.
- Noci et al., ICML 2024 (*Why do Learning Rates Transfer?*) tie transfer to the largest Hessian eigenvalue / sharpness at the edge of stability becoming width-independent under μP — a measurable diagnostic ($\lambda_{\max}\eta \approx 2$–$38/\eta$ regime) rather than a proof of cross-family invariance.

## 5. What Is Not Known

- **Theoretically open.** Whether a parametrization exists whose optimum is invariant across families with structurally different limits — selective SSMs have an input-dependent, non-linear-in-width recurrence; the Tensor Programs machinery covers them only where the operations reduce to matmul + coordinatewise nonlinearity. No proof of existence or impossibility.
- **Empirically open.** The controlled cross-family sweep. Everett et al. (ICML 2024) did the analogous job across *parametrizations and optimizers*; nobody has published the same grid across Transformer / Mamba-2 / MoE at matched tokens, matched budget, and $\ge 3$ seeds, at $\ge 10^{10}$ params.
- **Methodologically blocked.** The correspondence itself. Which SSM scalar is "the learning rate for the hidden layer" is not determined by the architecture; and $\Delta$, $A_{\log}$, router temperature have no Transformer counterpart, so "same hyperparameter, different family" is undefined for part of the vector.

## 6. Why It Is Hard

The obstruction is **non-identifiability of the tensor role**, compounded by **confounded measurement**.

Non-identifiability: μP's exponents come from classifying each tensor by whether both dimensions grow with $n$. Mamba's $x\!\to\!\Delta$ projection has fan-in $d_{\text{inner}} = \Theta(n)$ and fan-out $d_{\text{rank}}$ pinned (typically $\lceil n/16\rceil$ *or* a constant, depending on the config); an MoE router has fan-in $\Theta(n)$, fan-out $E$. Both are formally output-like *and* hidden-like depending on how you take the joint limit. Different valid limits give LR exponents differing by a full factor of $n$ — a 4× discrepancy at 4× width. There is no experiment on a single width that distinguishes them.

Confounding: any family swap changes the token budget per parameter, the arithmetic intensity, and the instability mechanism simultaneously. A measured $\eta^\star$ shift is then attributable to the parametrization, to the data ratio ($\eta^\star \propto C^{-0.125}$, above), or to a different divergence threshold. Absent ground truth ("the true optimal LR at 70B") makes the decisive check itself expensive: settling it requires the very sweep the method is meant to avoid, at least once.

## 7. Current Research (as of 2026)

- **Google DeepMind / Brain lineage** (Everett, Xiao, Pennington, Sohl-Dickstein): parametrization–optimizer exponent grids, per-layer LR, Adam $\epsilon$ scaling. The strongest methodology; extending it to non-Transformer families is the obvious next step *(frontier — verify)*.
- **Microsoft Research (Yang, Hu)** and **Harvard (Pehlevan, Bordelon)**: depth limits, joint width–depth transfer, and Tensor Programs coverage of new primitives.
- **Graphcore/Aleph Alpha (u-μP, Blake et al. 2024)**: unit-scaling composed with μP, giving hyperparameters that are near-1 by construction and FP8-friendly — a candidate *canonical coordinate system* in which cross-family comparison could be defined.
- **Chinese frontier labs (StepFun StepLaw, DeepSeek, MiniCPM)**: empirical $\eta^\star(N, D)$ surfaces fit on thousands of runs; treats transfer as regression rather than theory.
- **Hybrid-model groups** (Jamba, Zamba, Nemotron-H style stacks): practical need is acute — a hybrid's ratio of attention to SSM blocks is itself a hyperparameter that changes $\eta^\star$ *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question.** Does the μP-transferred learning rate from a Transformer proxy land within seed noise on a matched Mamba-2 model?

**Scale.** Widths $n \in \{256, 512, 1024, 2048\}$, depth 24 fixed, trained on 20 tokens/param (Chinchilla-matched, so the data-ratio confound is closed) on one corpus, one tokenizer, cosine schedule, 3 seeds. LR grid: 7 points, half-octave spacing around $2^{-8}$. Two families at matched non-embedding parameter count: pre-LN Transformer with QK-norm; Mamba-2 with $d_{\text{state}}=128$ held fixed. Total ≈ $4\times7\times3\times2 = 168$ runs, largest ~1.3B params — roughly $10^{21}$ FLOPs, a few thousand H100-hours.

**Control arm.** The Transformer's own μP width transfer (known to hold), plus a standard-parametrization (SP) arm on both families, which is known to drift. SP drift bounds the effect size the μP arm must beat.

**Deciding number.** The fitted exponent $\beta$ in $\eta^\star(n) \propto n^{\beta}$ for Mamba-2 under μP, with bootstrap CI over seeds. $|\beta| < 0.1$ ⇒ μP transfers across the family boundary as-is. $\beta \approx -1$ ⇒ the $\Delta$-projection is behaving as a hidden layer needing $1/n$ LR that μP is not supplying — that identifies the missing rule. Anything in between falsifies the "one exponent per tensor class" framing.

Secondary readout at no extra cost: log $\delta h^{(l)}_t$ per block at steps 1/10/100 and check which tensor's coordinate update deviates from $\Theta(1)$ — that localizes the failure to a specific tensor rather than to the family.

## 9. Key References

- **[Foundational]** Greg Yang, Edward J. Hu. *Feature Learning in Infinite-Width Neural Networks (Tensor Programs IV).* ICML, 2021. — arXiv:2011.14522
- **[Foundational/SOTA]** Greg Yang, Edward J. Hu, Igor Babuschkin, Szymon Sidor, Xiaodong Liu, David Farhi, Nick Ryder, Jakub Pachocki, Weizhu Chen, Jianfeng Gao. *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer.* NeurIPS, 2021. — arXiv:2203.03466
- **[SOTA]** Katie Everett, Lechao Xiao, Mitchell Wortsman, Alexander A. Alemi, Roman Novak, Peter J. Liu, Izzeddin Gur, Jascha Sohl-Dickstein, Leslie Pack Kaelbling, Jaehoon Lee, Jeffrey Pennington. *Scaling Exponents Across Parameterizations and Optimizers.* ICML, 2024. — arXiv:2407.05872
- **[SOTA]** Greg Yang, Dingli Yu, Chen Zhu, Soufiane Hayou. *Tensor Programs VI: Feature Learning in Infinite-Depth Neural Networks.* ICLR, 2024. — arXiv:2310.02244
- **[Replication]** Lucas Lingle. *A Large-Scale Exploration of μ-Transfer.* 2024. — arXiv:2404.05728
- **[Theory]** Lorenzo Noci, Alexandru Meterez, Thomas Hofmann, Antonio Orvieto. *Why do Learning Rates Transfer? Reconciling Optimization and Scaling Limits for Deep Learning.* ICML, 2024. — arXiv:2402.17457
- **[Depth]** Blake Bordelon, Lorenzo Noci, Mufan Bill Li, Boris Hanin, Cengiz Pehlevan. *Depthwise Hyperparameter Transfer in Residual Networks: Dynamics and Scaling Limit.* ICLR, 2024.
- **[Method]** Charlie Blake, Constantin Eichenberg, Josef Dean, Lukas Balles, Luke Y. Prince, Björn Deiseroth, Andres Felipe Cruz-Salinas, Carlo Luschi, Samuel Weinbach, Douglas Orr. *u-μP: The Unit-Scaled Maximal Update Parametrization.* 2024. — arXiv:2407.17465
- **[Empirical laws]** DeepSeek-AI (Xiao Bi et al.). *DeepSeek LLM: Scaling Open-Source Language Models with Longtermism.* 2024. — arXiv:2401.02954
- **[Stability]** Mitchell Wortsman, Peter J. Liu, Lechao Xiao, Katie Everett, Alex Alemi, Ben Adlam, John D. Co-Reyes, Izzeddin Gur, Abhishek Kumar, Roman Novak, Jeffrey Pennington, Jascha Sohl-Dickstein, Kelvin Xu, Jaehoon Lee, Justin Gilmer, Simon Kornblith. *Small-scale proxies for large-scale Transformer training instabilities.* ICLR, 2024. — arXiv:2309.14322
- **[Target family]** Albert Gu, Tri Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* COLM, 2024. — arXiv:2312.00752
- **[Deployment]** Nolan Dey et al. *Cerebras-GPT: Open Compute-Optimal Language Models Trained on the Cerebras Wafer-Scale Cluster.* 2023. — arXiv:2304.03208

## 10. Worked Example

Take a μP Transformer with base width $n_0 = 256$ and tuned $\eta^\star_A = 2^{-8} \approx 3.9\times10^{-3}$ (Adam, base-width units). μP says: keep $\eta_0$ fixed, scale hidden-matrix LRs by $n_0/n$. Going to $n = 2048$, the hidden LR becomes $3.9\times10^{-3}\times(256/2048) = 4.9\times10^{-4}$, and the empirical optimum stays put — this is the part that works.

Now the same move on Mamba-2. The block has $d_{\text{inner}} = 2n$. Two tensors have no clean role:

| tensor | fan-in | fan-out | μP class |
|---|---|---|---|
| $W_{qkv}$ (Transformer) | $n$ | $3n$ | hidden — unambiguous |
| $W_\Delta$ (SSM) | $d_{\text{inner}} = 2n$ | $d_{\text{rank}}$ | ambiguous |
| $A_{\log}$ | — | $2n$ (a vector) | undefined |

If $d_{\text{rank}}$ is a config constant (say 64), $W_\Delta$ is *output-like*: μP requires $\sigma^2 = \Theta(1/n^2)$ and forward multiplier $\Theta(1/n)$. If $d_{\text{rank}}=\lceil n/16\rceil$, it is *hidden-like*: $\sigma^2=\Theta(1/n)$, multiplier $\Theta(1)$. Reference implementations use the second but initialize $\Delta$'s bias from a fixed range so that $\Delta \in [10^{-3}, 10^{-1}]$ — a width-independent init on a width-dependent tensor.

Trace the consequence. Pre-activation $\Delta$ has variance $\propto d_{\text{inner}}\sigma^2$. Under the output-like reading this is $\Theta(1/n)$; under the hidden-like reading, $\Theta(1)$. At $n{:}\,256\to2048$ the two readings differ by $8\times$ in $\mathrm{Var}(\Delta)$, i.e. about $2.8\times$ in $\Delta$ itself. The SSM's effective memory horizon is $\tau \approx 1/(\Delta\,|A|)$, so the wrong choice changes the token horizon the model can hold by roughly $3\times$ at 8× width — and the optimizer compensates by moving $\eta^\star$.

The obstruction is now visible: **both readings are internally consistent μP derivations**, they agree exactly at the base width $n_0$ where you tuned, and they diverge only at the target scale where you cannot afford to check. A single-width proxy sweep contains zero information about which is right. Distinguishing them needs at least three widths on the target family with the LR grid re-run — which is Section 8's experiment, and is precisely the cost μ-transfer was introduced to avoid.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*