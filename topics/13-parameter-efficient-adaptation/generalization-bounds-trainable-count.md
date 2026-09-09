---
id: 13-parameter-efficient-adaptation/generalization-bounds-trainable-count
title: "Generalization Bounds That Use Adapter Count"
topic: 13-parameter-efficient-adaptation
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Generalization Bounds That Use Adapter Count

> **Topic:** Parameter-Efficient Adaptation · **ID:** `13-parameter-efficient-adaptation/generalization-bounds-trainable-count` · **Status:** open

## 1. Problem Statement

Parameter-efficient fine-tuning (PEFT) updates $D \ll P$ parameters of a $P$-parameter pretrained model — LoRA rank-$r$ factors, adapter bottlenecks, prefix vectors, sparse masks. The folk claim is that small $D$ explains why PEFT does not overfit small downstream sets. The problem: **produce a generalization bound whose only capacity term is $D$ (or a simple function of it) that is both non-vacuous and rank-ordering-correct at realistic fine-tuning scale.**

Three variants, with different difficulty:

- **Theory.** Prove $\mathbb{E}[L(\hat w) - \hat L(\hat w)] \le \varepsilon(D, m, \delta)$ for PEFT-constrained hypothesis classes, with $\varepsilon$ independent of $P$ and of the pretrained weights' norms. Open.
- **Measurement.** Decide whether the *empirical* generalization gap is monotone in $D$ at fixed data and fixed optimizer. Runnable; the published evidence says roughly **no**.
- **Method.** Use a $D$-dependent bound as a model-selection signal: pick rank/placement without a validation set. Nobody has shown this beats validation-set selection.

Solving it means: a bound $\le 0.5$ on 0–1 error (or $\le$ the true gap $\times 3$) for a $\ge$1B-parameter model fine-tuned on $m \le 10^5$ examples, *and* Kendall $\tau > 0.5$ against measured gaps across a rank sweep.

## 2. Formal Setting

Pretrained weights $w_0 \in \mathbb{R}^P$, frozen. A PEFT parameterization is a map $\phi: \mathbb{R}^D \to \mathbb{R}^P$, $w = w_0 + \phi(\theta)$. For LoRA on a set $\mathcal{M}$ of matrices with shapes $d^{\text{out}}_j \times d^{\text{in}}_j$ and rank $r$:

$$D = \sum_{j \in \mathcal{M}} r\,(d^{\text{in}}_j + d^{\text{out}}_j).$$

**Measured as:** count of tensor entries with `requires_grad=True` at step 0, excluding frozen biases, LayerNorms, and the classifier head — the third exclusion is where published counts most often disagree.

Sample $S = \{z_i\}_{i=1}^m \sim \mathcal{D}^m$. Empirical risk $\hat L(w) = \frac1m \sum_i \ell(w, z_i)$, population risk $L(w) = \mathbb{E}_{z}[\ell(w,z)]$, $\ell \in [0,1]$. **Gap measured as** $\hat L$ on the exact training split minus $L$ estimated on a held-out split of $n \ge 10^4$ drawn from the same pipeline — not a public test set that the pretraining corpus may contain.

**Counting / Occam bound.** Quantize $\theta$ to $b$ bits per coordinate, giving a prefix-free code of length $Db$ and a finite class of size $2^{Db}$. With probability $\ge 1-\delta$,

$$L(\hat w) \le \hat L(\hat w) + \sqrt{\frac{Db\ln 2 + \ln(2\sqrt m/\delta)}{2m}}.$$

**PAC-Bayes form** (McAllester 1999; Catoni 2007): posterior $Q$ over $\theta$, prior $P$ chosen before seeing $S$,

$$\mathbb{E}_{Q}[L] \le \mathbb{E}_{Q}[\hat L] + \sqrt{\frac{\mathrm{KL}(Q\|P) + \ln(2\sqrt m/\delta)}{2m}},$$

where the $D$-dependence enters only through $\mathrm{KL}$, which for isotropic Gaussians scales as $\tfrac{D}{2}\log(\sigma_P^2/\sigma_Q^2) + \|\theta\|^2/2\sigma_P^2$ — i.e. $D$ multiplies a *precision* term, not a raw count.

**Assumptions, and which break.**
1. *$\phi$ is fixed before seeing $S$.* Violated by AdaLoRA, rank pruning, and any placement search — the effective class is a union over $\phi$'s, adding $\log|\Phi|$ bits nobody counts.
2. *$b$ bits suffice.* Violated: LoRA at $b=4$ post-hoc typically loses accuracy, so the bound's $\hat L$ and the deployed $\hat L$ differ.
3. *Prior independent of $S$.* Violated by data-dependent priors built from $w_0$ pretrained on corpora overlapping the downstream distribution.
4. *$\ell$ bounded.* Violated for next-token cross-entropy; token-level bounds need an explicit clip or a bits-per-token normalization.
5. *i.i.d. sampling.* Violated for instruction-tuning mixtures with template-correlated examples.

## 3. State of the Art

**Theory SOTA — established.**
- Non-vacuous PAC-Bayes bounds *exist* for large models when the compression is into a random low-dimensional subspace plus quantization: Lotfi et al., *PAC-Bayes Compression Bounds So Tight That They Can Explain Generalization* (NeurIPS 2022), and *Non-Vacuous Generalization Bounds for Large Language Models* (ICML 2024), which reach non-vacuous token-level bits-per-dimension bounds for LLMs in the few-hundred-million-parameter range using SubLoRA (subspace + LoRA + quantization). Established: the bound is computed and non-vacuous. Not established: that it tracks $D$.
- Zeng & Lee, *The Expressive Power of Low-Rank Adaptation* (ICLR 2024), characterizes the rank needed to represent a target model. This is approximation, not generalization — it bounds what LoRA *can* fit, which if anything argues the class is larger than $D$ suggests.
- Fu et al., *On the Effectiveness of Parameter-Efficient Fine-Tuning* (AAAI 2023), gives a bound for sparse PEFT scaling with the number of tuned parameters. Established as a theorem under its stated sparsity and smoothness assumptions; **claimed but unablated** as an explanation of LoRA's behaviour, since it is not evaluated as a numeric bound against measured gaps.

**Empirical SOTA.**
- Hu et al., *LoRA* (ICLR 2022): downstream accuracy is nearly flat across $r \in \{1,2,4,8,64\}$ on WikiSQL and MultiNLI with GPT-3 175B. A $64\times$ change in $D$ moves accuracy by well under a point.
- Biderman et al., *LoRA Learns Less and Forgets Less* (TMLR 2024): on Llama-2 7B/13B code and math continued training, LoRA underperforms full fine-tuning on target-domain learning while forgetting less — the effect tracks rank *weakly* and the regularization is better described as constrained-perturbation than as low count.
- Jiang et al., *Fantastic Generalization Measures* (ICLR 2020) and Dziugaite et al., *In Search of Robust Measures of Generalization* (NeurIPS 2020): parameter-count-flavoured complexity measures rank-order poorly; sharpness and PAC-Bayes measures do better. Measured on CNNs, not PEFT — the transfer is assumed, not shown.

## 4. What Is Known

- **Adapter counts, real scale.** LoRA $r=4$ on query/value of GPT-3 175B: $D = 4.7$M, $\approx 2.6\times10^{-5}$ of $P$. LoRA on RoBERTa-base: $D=0.3$M vs 125M full. Adapters (Houlsby et al., ICML 2019) reach within 0.4 points of full BERT-large GLUE with 3.6% of parameters per task.
- **Accuracy is near-flat in rank.** Hu et al. report GPT-3 175B WikiSQL validation accuracy within $\approx 0.4$ points across $r=1$ to $r=64$; the counting bound over the same sweep grows by $\sqrt{64}=8\times$ in its capacity term. Any $D$-monotone bound is therefore *anti-correlated* with measurement over the most-cited sweep.
- **Intrinsic dimension is small but not tiny.** Aghajanyan et al. (ACL 2021): $d_{90}$ — the random-subspace dimension reaching 90% of full fine-tuning performance — is 896 for RoBERTa-base on MRPC and 207 for RoBERTa-large; larger pretrained models have *lower* $d_{90}$, at $m \approx 3.7$k examples.
- **Non-vacuous LLM bounds exist but do not isolate $D$.** Lotfi et al. (ICML 2024; NeurIPS 2024 token-level extension) obtain non-vacuous compression bounds at up to roughly the 1B-parameter range, with $m$ on the order of $10^9$ *tokens*. The slack is dominated by the quantized description length of the whole compressed model, not by adapter count at a fixed base.
- **Scaling factor matters more than count.** Kalajdzievski (2023) shows LoRA's $\alpha/r$ scaling destabilizes high ranks; much of the apparent "rank does not help" effect is an optimization artifact, not a capacity one.

## 5. What Is Not Known

- **Theoretically open.** No bound is known of the form $O(\sqrt{D/m})$ for LoRA that is (a) independent of $P$ and $\|w_0\|$, and (b) valid without a data-independent prior. Whether such a bound can exist is itself unresolved: the LoRA class is a *nonlinear* $D$-dimensional manifold in weight space whose local Lipschitz constant depends on $w_0$, so $D$ alone plausibly cannot control it.
- **Empirically open.** No published experiment holds $m$, optimizer, LR schedule, and $\alpha/r$ fixed while sweeping $D$ over $\ge 3$ decades and reporting a *train-minus-held-out gap* (not accuracy) with seed error bars at $\ge$7B scale. The sweep is $O(10^2)$ GPU-days — affordable, unrun.
- **Methodologically blocked.** "Generalization gap" for an instruction-tuned LLM has no agreed measurement: the held-out set is usually contaminated by pretraining, and $\ell$ is unbounded cross-entropy. Until the gap has a definition that survives contamination auditing, no bound can be *validated*, only computed.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability of capacity from count under a data-dependent prior**. $w_0$ was chosen using data that overlaps $\mathcal{D}$. A $D$-parameter adapter around a well-placed $w_0$ and the same adapter around a random $w_0$ have identical $D$ and vastly different achievable risk. Every bound that is tight in practice smuggles this in through the prior — Lotfi et al.'s priors are centered at the pretrained model — at which point the bound's tightness is attributable to the prior, not to $D$, and the two contributions are not separately measurable from the final number.

Second obstruction: **the evaluation does not measure what it names**. Rank sweeps report task accuracy, which is bounded above by the task ceiling and saturates; the gap term the bound predicts is invisible under saturation. A flat accuracy curve is consistent both with "capacity does not matter" and with "capacity matters but the task is too easy to show it".

## 7. Current Research (as of 2026)

- **Compression-based PAC-Bayes at LLM scale.** Wilson's group (NYU) — SubLoRA, token-level bounds. Direction: pushing non-vacuity past 7B and to instruction data. *(frontier — verify)*
- **Rank-adaptive PEFT with capacity accounting.** AdaLoRA (Zhang et al., ICLR 2023) and successors allocate rank by importance; whether the allocation search cost belongs in the bound is an open accounting question actively raised in reviews. *(frontier — verify)*
- **Norm-based rather than count-based analyses of LoRA**, e.g. treating $\|BA\|_F$ or the spectral perturbation $\|\Delta W\|_2/\|W_0\|_2$ as the capacity term. Preliminary; no non-vacuous numbers published. *(frontier — verify)*
- **Contamination-clean downstream benchmarks** as a prerequisite for measuring gaps at all.

## 8. Concrete Next Experiment

**Scale.** Llama-2 7B (or an equivalently sized open base with a documented pretraining corpus). Fine-tune on a single held-in-house dataset of $m = 20{,}000$ examples with a disjoint $n = 20{,}000$ held-out split from the same collection, contamination-audited by 13-gram overlap against the base corpus.

**Sweep.** LoRA rank $r \in \{1,2,4,8,16,64,256\}$ on all attention and MLP projections → $D$ from $\approx 2.0$M to $\approx 500$M, three decades. Fix $\alpha/r$ constant (rank-stabilized scaling), fix LR, schedule, epochs, batch size; 3 seeds each. Record train loss, held-out loss, and gap $g(r)$.

**Control arm.** Same $D$ but random-subspace projection (Li et al., ICLR 2018 style, or the SubLoRA subspace) instead of low-rank structure. This arm has identical parameter count and no low-rank inductive bias, so it separates "count" from "structure".

**Deciding number.** Kendall $\tau$ between the counting bound's capacity term $\sqrt{Db/2m}$ and measured $g(r)$ across the 7 ranks, pooled over seeds. **$\tau \ge 0.5$** ⇒ adapter count is a usable capacity signal and the theory variant is worth pursuing. **$\tau \le 0.2$, or $\tau < 0$** ⇒ count-based bounds are ranking-invalid at this scale and the field should move to norm- or sharpness-based terms. Secondary readout: whether $g$(LoRA) $\ne$ $g$(random subspace) at matched $D$ — a difference falsifies any bound depending on $D$ alone.

## 9. Key References

- **[Foundational]** Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR, 2022. — arXiv:2106.09685
- **[Foundational]** Neil Houlsby et al. *Parameter-Efficient Transfer Learning for NLP.* ICML, 2019. — arXiv:1902.00751
- **[Foundational]** Gintare Karolina Dziugaite, Daniel M. Roy. *Computing Nonvacuous Generalization Bounds for Deep (Stochastic) Neural Networks with Many More Parameters than Training Data.* UAI, 2017. — arXiv:1703.11008
- **[Foundational]** Armen Aghajanyan, Luke Zettlemoyer, Sonal Gupta. *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning.* ACL, 2021. — arXiv:2012.13255
- **[SOTA]** Sanae Lotfi, Marc Finzi, Yilun Kuang, Tim G. J. Rudner, Micah Goldblum, Andrew Gordon Wilson. *Non-Vacuous Generalization Bounds for Large Language Models.* ICML, 2024. — arXiv:2312.17173
- **[SOTA]** Sanae Lotfi et al. *PAC-Bayes Compression Bounds So Tight That They Can Explain Generalization.* NeurIPS, 2022. — arXiv:2211.13609
- **[SOTA]** Yuchen Zeng, Kangwook Lee. *The Expressive Power of Low-Rank Adaptation.* ICLR, 2024. — arXiv:2310.17513
- **[SOTA]** Dan Biderman et al. *LoRA Learns Less and Forgets Less.* TMLR, 2024. — arXiv:2405.09673
- **[Related]** Zihao Fu, Haoran Yang, Anthony Man-Cho So, Wai Lam, Lidong Bing, Nigel Collier. *On the Effectiveness of Parameter-Efficient Fine-Tuning.* AAAI, 2023.
- **[Related]** Chunyuan Li, Heerad Farkhoor, Rosanne Liu, Jason Yosinski. *Measuring the Intrinsic Dimension of Objective Landscapes.* ICLR, 2018. — arXiv:1804.08838
- **[Related]** Wenfeng Zhang et al. *AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning.* ICLR, 2023. — arXiv:2303.10512
- **[Survey]** Yaowei Zheng / Zeyu Han et al. *Parameter-Efficient Fine-Tuning for Large Models: A Comprehensive Survey.* TMLR, 2024. — arXiv:2403.14608
- **[Survey]** Yiming Jiang, Behnam Neyshabur, Hossein Mobahi, Dilip Krishnan, Samy Bengio. *Fantastic Generalization Measures and Where to Find Them.* ICLR, 2020. — arXiv:1912.02178
- **[Survey]** Ning Ding et al. *Parameter-efficient fine-tuning of large-scale pre-trained language models.* Nature Machine Intelligence, 2023.

## 10. Worked Example

LoRA $r=8$ on Llama-2-7B, $d=4096$, 32 layers, applied to $W_q$ and $W_v$:

$$D = 32 \times 2 \times 8 \times (4096 + 4096) = 4{,}194{,}304 \approx 4.19\text{M}.$$

Fine-tune on MRPC, $m = 3{,}668$. At $b=16$ bits: description length $Db = 6.71\times10^7$ bits. Counting bound capacity term:

$$\sqrt{\frac{6.71\times10^7 \times 0.693}{2 \times 3668}} \approx \sqrt{6.34\times10^3} \approx 79.6.$$

The bound on a 0–1 loss is $\approx 80$. The trivial bound is 1. **Vacuous by about two orders of magnitude.** Measured gap is roughly 0.02–0.05.

How small must $D$ be for non-vacuity? Requiring capacity term $\le 0.25$ at $b=16$:

$$D \le \frac{2m(0.25)^2}{b\ln 2} = \frac{2\cdot 3668 \cdot 0.0625}{16 \times 0.693} \approx 41 \text{ parameters}.$$

Now plug in the *best case the literature offers*: Aghajanyan's intrinsic dimension, $d_{90} = 896$ for MRPC. At $b=16$, capacity term $= \sqrt{896\cdot16\cdot0.693/7336} = 1.18$ — still vacuous. At $b=8$: $0.83$. At $b=4$: $0.59$. Only at $b=2$ does it cross 0.5 — and 2-bit adapters do not reach the accuracy that made $d_{90}$ meaningful.

**The obstruction, made visible.** The gap between what LoRA actually needs ($\sim 10^3$ effective dimensions) and what a counting bound can afford ($\sim 10^1$–$10^2$ at $m=3.7$k) is 1–2 orders of magnitude, and it does not close by shrinking $r$: at $r=1$, $D$ is still 524k, capacity term $\approx 28$. The entire distance is covered in practice by the prior centered at $w_0$ — which is exactly the term $D$ does not describe.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*