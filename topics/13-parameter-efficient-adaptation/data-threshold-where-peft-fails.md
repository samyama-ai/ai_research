---
id: 13-parameter-efficient-adaptation/data-threshold-where-peft-fails
title: "Data Scaling Threshold Where PEFT Fails"
topic: 13-parameter-efficient-adaptation
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Data Scaling Threshold Where PEFT Fails

> **Topic:** Parameter-Efficient Adaptation · **ID:** `13-parameter-efficient-adaptation/data-threshold-where-peft-fails` · **Status:** empirically-open

## 1. Problem Statement

Parameter-efficient fine-tuning (PEFT) — LoRA, adapters, prompt tuning — updates $10^{-3}$ to $10^{-2}$ of a model's parameters and, on small adaptation sets, matches full fine-tuning. It does not match full fine-tuning on large adaptation sets. The problem is to locate and explain the crossover.

- **Measurement variant.** Given a base model, a task distribution, and a PEFT method with a fixed capacity knob (LoRA rank $r$), find the adaptation-set size $n^\*$ at which the held-out loss gap between PEFT and full fine-tuning first exceeds a stated tolerance $\varepsilon$, with both arms tuned to their own optimal hyperparameters.
- **Method variant.** Construct a PEFT scheme whose gap stays below $\varepsilon$ for all $n$ at cost sublinear in the full-fine-tuning cost — i.e. push $n^\*$ to infinity without paying for it.
- **Theory variant.** Predict $n^\*$ from quantities knowable before training: model size $N$, rank $r$, task intrinsic dimension, distribution shift from pretraining.

Solving it means: a formula $n^\*(N, r, \text{task})$ that predicts the crossover on a held-out task family to within a factor of 2, plus a demonstration that the crossover is not an artifact of learning-rate tuning.

## 2. Formal Setting

Base model $f_{\theta_0}$, $\theta_0 \in \mathbb{R}^N$. Adaptation set $D_n = \{(x_i,y_i)\}_{i=1}^n$ i.i.d. from target distribution $\mathcal{P}$; measure $n$ in **tokens of loss-bearing target text**, not examples — example counts are not comparable across tasks.

Full fine-tuning searches $\mathbb{R}^N$. PEFT searches a submanifold $\Theta_\phi = \{\theta_0 + \Delta(\phi) : \phi \in \mathbb{R}^p\}$, $p \ll N$. For LoRA on a weight $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$: $\Delta W = \tfrac{\alpha}{r} BA$, $B \in \mathbb{R}^{d_{\text{out}} \times r}$, $A \in \mathbb{R}^{r \times d_{\text{in}}}$, so $p = r \sum_{m \in \mathcal{M}} (d_{\text{in}}^{(m)} + d_{\text{out}}^{(m)})$ over the adapted module set $\mathcal{M}$.

Measured quantities:

$$L_{\text{peft}}(n) = \mathbb{E}_{(x,y)\sim\mathcal{P}}\big[\ell(f_{\hat\phi(D_n)}(x), y)\big], \qquad \Delta(n) = L_{\text{peft}}(n) - L_{\text{full}}(n)$$

with $\ell$ per-token cross-entropy in nats on a held-out split from the same $\mathcal{P}$, and each arm's learning rate, schedule, and epoch count selected by independent sweep at each $n$. The threshold:

$$n^\*(r,\varepsilon) = \min\{\, n : \Delta(n') > \varepsilon \ \ \forall n' \ge n \,\}$$

The monotone-tail requirement matters: a single noisy crossing is not a threshold.

Zhang et al. (ICLR 2024) fit a **multiplicative joint law** of the form $\hat L(N_{\text{eff}}, n) = A\,N_{\text{eff}}^{-a} n^{-b} + E$, where $N_{\text{eff}}$ is model size for full fine-tuning and trainable-parameter count for PEFT. Under any such law with a method-dependent irreducible term $E_{\text{peft}} > E_{\text{full}}$, $n^\*$ is finite for every $\varepsilon < E_{\text{peft}} - E_{\text{full}}$.

Assumptions, with the ones known to break flagged:

1. Train and eval are i.i.d. from $\mathcal{P}$. **Usually holds** by construction.
2. Both arms are at their own optimum. **Violated in most published comparisons** — LoRA's optimal LR is roughly an order of magnitude above full fine-tuning's, and shared-LR studies systematically understate LoRA.
3. Loss is the quantity of interest. **Violated** when the deliverable is a benchmark accuracy; loss and accuracy cross at different $n$.
4. Adaptation is capacity-limited, not optimization-limited. **Unknown** — the low-rank factorization is a nonconvex reparameterization, and whether the gap is expressivity or trainability is unresolved.
5. $\Delta(n)$ is monotone in $n$. **Not established**; assumed in every threshold definition in use.

## 3. State of the Art

**Empirical SOTA — established.** Zhang et al., *When Scaling Meets LLM Finetuning: The Effect of Data, Model and Finetuning Method* (ICLR 2024) is the only study that fits a joint data/model/method scaling law across full fine-tuning, LoRA, and prompt tuning, on bilingual translation and summarization with encoder-decoder models from 1B to 16B and up to $\sim$1M examples. Finding: fine-tuning data scaling is multiplicative with model scaling, PEFT benefits from model scale more than from data scale, and full fine-tuning overtakes PEFT once the adaptation set is large.

Biderman et al., *LoRA Learns Less and Forgets Less* (TMLR 2024) separates regimes by data volume: on continued pretraining (up to $\sim$20B tokens, Llama-2 7B/13B, code and math) LoRA lags full fine-tuning by a wide margin; on instruction tuning (order $10^5$ examples) the gap narrows and LoRA forgets less of the base distribution. This is the cleanest existing evidence that the gap is data-volume driven rather than task driven.

**Claimed but unablated.** Thinking Machines Lab, *LoRA Without Regret* (2025) argues LoRA matches full fine-tuning whenever adapter capacity exceeds the information content of the dataset, and places the boundary near "trainable parameters $\approx$ dataset tokens". The claim is stated over SFT and RL runs but the capacity boundary itself is a heuristic fit, not an ablation with a controlled rank ladder at fixed data.

**Benchmark-number-only results.** DoRA (Liu et al., ICML 2024), rsLoRA (Kalajdzievski, 2023), and most LoRA variants report GLUE/commonsense/MT-bench deltas at a single dataset size. They say nothing about $n^\*$; a variant that wins at $n = 10^4$ may have identical asymptotics.

**Theory SOTA.** Zeng & Lee, *The Expressive Power of Low-Rank Adaptation* (ICLR 2024): for a target transformer of depth $L_{\text{tgt}}$ and a frozen model of depth $L$, LoRA of rank $r \ge \lceil \text{width}/2 \rceil$ (roughly, under stated width/depth ratios) can exactly represent the target. This bounds expressivity in the exact-fit sense and does not bound $L_{\text{peft}}(n)$ under SGD.

## 4. What Is Known

- **Intrinsic dimension is small for small tasks.** Aghajanyan et al. (ACL 2021) report $d_{90}$ — the subspace dimension reaching 90% of full fine-tuning performance — in the hundreds to low thousands for GLUE tasks, and *decreasing* with model size at fixed task. Measured on BERT/RoBERTa-scale models with datasets of $10^3$–$10^5$ examples.
- **Prompt tuning closes its gap with model scale, not data scale.** Lester et al. (EMNLP 2021): at 10B T5, prompt tuning matches full fine-tuning on SuperGLUE; at 100M–1B it does not. SuperGLUE tasks are small ($\le 10^5$ examples), so this is a low-$n$ result.
- **LoRA's optimal learning rate is much higher than full fine-tuning's.** Reported across Biderman et al. and subsequent LoRA hyperparameter work; sweeps that omit this find spuriously early thresholds.
- **LoRA and full fine-tuning reach structurally different solutions.** Shuttleworth et al., *LoRA vs Full Fine-tuning: An Illusion of Equivalence* (2024): LoRA updates introduce high-ranking singular vectors approximately orthogonal to the pretrained spectrum ("intruder dimensions"), absent in full fine-tuning at matched task loss. Measured on RoBERTa and Llama-scale models, $10^4$–$10^5$ examples.
- **Memorization capacity per parameter is roughly 2 bits.** Allen-Zhu & Li, *Physics of Language Models 3.3: Knowledge Capacity Scaling Laws* (ICML 2025), measured on synthetic biography corpora across GPT-2-style models. Gives a capacity-side upper bound on what any adapter of $p$ parameters can absorb.

## 5. What Is Not Known

- **Empirically open.** Nobody has run a full $(r, n)$ grid on a modern decoder at $10^9$-token adaptation scale with per-cell learning-rate sweeps in both arms. This is the central gap: the experiment is entirely runnable, costs perhaps $10^4$–$10^5$ GPU-hours, and has no publication incentive because the answer is "PEFT eventually loses".
- **Empirically open.** Whether $n^\*$ scales linearly in $p$ (capacity story), sublinearly (optimization story), or is $r$-independent above some $r_{\min}$ (intrinsic-dimension story). The three predictions differ by orders of magnitude and no dataset distinguishes them.
- **Theoretically open.** No generalization bound for LoRA that is $n$-dependent and tight enough to predict a crossover. Expressivity results (Zeng & Lee) are existence statements about the optimum, not about what SGD finds from $\theta_0$.
- **Methodologically blocked.** "Effective capacity of an adapter" has no agreed measurement. Trainable-parameter count ignores that $BA$ is rank-constrained regardless of $p$; the number of bits an adapter can store has been measured only for full models.
- **Methodologically blocked.** Whether $\Delta(n)$ measured on task loss transfers to $\Delta(n)$ on downstream capability. Forgetting means the two arms differ on the base distribution even at equal task loss; there is no accepted scalarization.

## 6. Why It Is Hard

Three named obstructions.

1. **Confounded measurement.** The gap $\Delta(n)$ is a difference of two independently-tuned optimization outcomes. LoRA's optimum sits at a different learning rate, a different effective batch-size sensitivity, and a different epoch count. Any protocol that shares hyperparameters between the arms measures the tuning gap, not the capacity gap — and most published comparisons do exactly this. Getting a clean $\Delta(n)$ multiplies the compute by the size of two sweeps.
2. **Compute cost concentrated in the arm you cannot skip.** Establishing a threshold requires the *full fine-tuning* control at every $n$ on the ladder. That control is the expensive arm, and it is the one PEFT exists to avoid. The experiment's cost is dominated by the baseline.
3. **Non-identifiability of expressivity versus trainability.** If LoRA at $r=8$ fails at $n=10^9$ tokens, the merged update $\Delta W$ may be outside the reachable set, or reachable but unreached by Adam from $B=0$. Distinguishing them needs a merged-weight probe (initialize full fine-tuning at the LoRA solution and continue) that nobody runs as standard.

## 7. Current Research (as of 2026)

- **Joint scaling laws for fine-tuning methods.** Following Zhang et al. (ICLR 2024, Google DeepMind), extension to decoder-only models and to token-counted rather than example-counted data. Active but sparse.
- **Capacity-matched adapter design.** rsLoRA-style rank scaling and high-rank/periodically-merged schemes (ReLoRA, Lialin et al., ICLR 2024) aim to raise $n^\*$ by increasing effective rank at fixed step cost. Whether ReLoRA's merge schedule actually removes the ceiling at $10^9$-token scale is untested *(frontier — verify)*.
- **Post-training-scale LoRA.** Thinking Machines Lab and several open-source RL groups argue LoRA suffices for RL post-training because policy-gradient runs carry very few bits per episode; if true, $n^\*$ in bits is far above the RL data regime and only SFT/CPT hits the wall *(frontier — verify)*.
- **Spectral diagnostics.** Intruder-dimension analysis (Shuttleworth et al.) as an early-warning signal for the gap, measurable without the full fine-tuning control. Unvalidated as a predictor of $n^\*$ *(frontier — verify)*.

## 8. Concrete Next Experiment

**Scale.** Llama-3-8B base. One domain with abundant clean text (e.g. Python from The Stack v2). Continued-pretraining ladder $n \in \{10^7, 10^8, 10^9\}$ target tokens, single epoch, fixed token budget per cell.

**Arms.** LoRA on all linear modules at $r \in \{8, 64, 512\}$, plus the control arm: full fine-tuning at each $n$. Every cell gets an independent 4-point learning-rate sweep; report the best. 12 PEFT cells + 3 control cells + sweeps ≈ 60 runs; at $10^9$ tokens the largest cell is ~$5 \times 10^{21}$ FLOPs, so the whole grid is order $10^4$ H100-hours.

**Probe for obstruction 3.** At the largest $n$ and each $r$, take the merged LoRA weights and continue with *full* fine-tuning for 5% more tokens. If the loss drops to the full-fine-tuning curve, the failure was trainability; if it does not, expressivity.

**The deciding number.** The slope $s$ in $\log n^\*(r) = s \log r + c$, with $n^\*$ defined at $\varepsilon = 0.02$ nats/token. $s \approx 1$ confirms the capacity story (threshold linear in trainable parameters). $s \approx 0$ above $r=8$ confirms the intrinsic-dimension story (rank is irrelevant once minimal). $0 < s < 1$ falsifies both and points at optimization. Three points in $r$ over 1.8 decades give $s$ to about $\pm 0.15$ — enough to separate the hypotheses.

## 9. Key References

- **[Foundational]** Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR, 2022. — arXiv:2106.09685
- **[Foundational]** Houlsby, Giurgiu, Jastrzebski, Morrone, de Laroussilhe, Gesmundo, Attariyan, Gelly. *Parameter-Efficient Transfer Learning for NLP.* ICML, 2019. — arXiv:1902.00751
- **[Foundational]** Aghajanyan, Gupta, Zettlemoyer. *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning.* ACL, 2021. — arXiv:2012.13255
- **[SOTA]** Zhang, Liu, Cherry, Firat. *When Scaling Meets LLM Finetuning: The Effect of Data, Model and Finetuning Method.* ICLR, 2024. — arXiv:2402.17193
- **[SOTA]** Biderman, Portes, Ortiz, Paul, Greengard, Jennings, King, Havens, Chiley, Frankle, Blakeney, Cunningham. *LoRA Learns Less and Forgets Less.* TMLR, 2024. — arXiv:2405.09673
- **[SOTA]** Zeng, Lee. *The Expressive Power of Low-Rank Adaptation.* ICLR, 2024. — arXiv:2310.17513
- **[SOTA]** Shuttleworth, Andreas, Torralba, Sharma. *LoRA vs Full Fine-tuning: An Illusion of Equivalence.* 2024. — arXiv:2410.21228
- **[SOTA]** Lester, Al-Rfou, Constant. *The Power of Scale for Parameter-Efficient Prompt Tuning.* EMNLP, 2021. — arXiv:2104.08691
- **[SOTA]** Allen-Zhu, Li. *Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws.* ICML, 2025. — arXiv:2404.05405
- **[SOTA]** Dettmers, Pagnoni, Holtzman, Zettlemoyer. *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS, 2023. — arXiv:2305.14314
- **[SOTA]** Liu, Wang, Yin, Molchanov, Wang, Cheng, Chen. *DoRA: Weight-Decomposed Low-Rank Adaptation.* ICML, 2024. — arXiv:2402.09353
- **[Survey]** Lialin, Deshpande, Rumshisky. *Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning.* 2023. — arXiv:2303.15647
- **[Frontier]** Thinking Machines Lab. *LoRA Without Regret.* Lab blog post, 2025. — capacity-threshold claim; not peer reviewed.

## 10. Worked Example

Llama-3-8B, LoRA rank $r=16$ on all seven linear projections. Hidden size $d=4096$, MLP width 14336, GQA with KV width 1024, 32 layers. Per layer:

$$p_{\text{layer}} = 16\big[(4096{+}4096) + 2(4096{+}1024) + (4096{+}4096) + 2(4096{+}14336) + (14336{+}4096)\big] = 1{,}310{,}720$$

Over 32 layers, $p = 41.9\text{M}$ — 0.52% of 8.03B.

Now ask the same question two ways.

**Capacity heuristic (bits).** At 2 bits/parameter (Allen-Zhu & Li), the adapter holds at most $8.4 \times 10^7$ bits $\approx$ 10.5 MB. Domain text at a conservative 0.5 nats/token of *new* information beyond the base model is $0.72$ bits/token, so the adapter saturates at roughly $1.2 \times 10^8$ tokens. Predicted $n^\* \sim 10^8$.

**Parameter-count heuristic.** "Trainable parameters $\approx$ dataset tokens" puts $n^\* \approx 4.2 \times 10^7$ tokens.

**Intrinsic-dimension heuristic.** $d_{90}$ in the hundreds implies $r=16$ is already well past the minimal subspace, so $n^\*$ is unbounded and the observed gap is optimization, not capacity. Predicted $n^\* = \infty$.

Three defensible readings of published results place the threshold at $4\times10^7$, $1.2\times10^8$, and infinity. The spread is not a factor of 2 — it is unbounded. That is the obstruction: every input to these estimates was measured on a *different* model family, at a different scale, with a different loss, and none of them was measured by varying $n$ against a tuned full-fine-tuning control. Until the ladder in §8 is run, the practitioner asking "will LoRA hold at 100M tokens of domain data?" gets three answers from the literature and no way to choose between them.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*