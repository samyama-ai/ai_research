---
id: 03-training-dynamics/attention-entropy-collapse
title: "Attention Entropy Collapse Control"
topic: 03-training-dynamics
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Attention Entropy Collapse Control

> **Topic:** Training Dynamics & Optimization · **ID:** `03-training-dynamics/attention-entropy-collapse` · **Status:** open

## 1. Problem Statement

During transformer training the softmax attention distribution of some heads sharpens until it is effectively one-hot. The row entropy falls toward $0$, the attention logits grow without bound, and the loss curve spikes or diverges. Practitioners suppress this with QK-LayerNorm, logit soft-capping, or spectral reparameterization. None of these is derived from a stated criterion, and none comes with a rule saying when it is needed.

Three variants, of different difficulty:

- **Measurement.** Given a training run, decide *from the entropy trajectory alone* whether a loss spike is imminent. Input: per-layer, per-head attention entropies $\{H^{(\ell,h)}_t\}$ over steps $t$. Output: a binary alarm at step $t$ with horizon $\Delta$. Solved means an alarm with useful precision/recall at $\Delta \gtrsim 100$ steps that is not just a lagging indicator of the spike itself.
- **Method.** Find an intervention (loss term, reparameterization, normalization, LR schedule) that keeps entropy above a floor **and** does not cost final loss. Solved means: matched-compute, matched-tokens, no measurable degradation on held-out loss at $\ge 1$B parameters, with the entropy floor demonstrably binding.
- **Theory.** Prove conditions on architecture, initialization, data and optimizer under which $\min_{\ell,h} \mathbb{E}[H^{(\ell,h)}_t]$ stays bounded away from $0$ for $t \le T$. Nothing of this form exists for a realistic optimizer.

The variants are routinely conflated. "Entropy collapse causes instability" is a measurement claim; "QK-norm fixes instability" is a method claim; neither implies the other.

## 2. Formal Setting

A head $h$ in layer $\ell$ with queries $Q = XW_Q$, keys $K = XW_K$, $X \in \mathbb{R}^{n \times d}$, head dim $d_h$. Logits and weights:

$$ L_{ij} = \frac{\langle q_i, k_j\rangle}{\sqrt{d_h}}, \qquad a_{ij} = \frac{\exp(L_{ij})}{\sum_{j' \le i}\exp(L_{ij'})} .$$

**Row entropy**, as measured:

$$ H_i = -\sum_{j\le i} a_{ij}\log a_{ij} \in [0, \log i].$$

Because the causal mask makes the bound position-dependent, the only comparable quantity across positions is the **normalized entropy** $\tilde H_i = H_i / \log i$ for $i \ge 2$. Reported "attention entropy" is usually $\bar H^{(\ell,h)} = \frac{1}{n-1}\sum_{i\ge 2} H_i$ over a fixed batch, *not* normalized — this is a live source of cross-paper incomparability.

**Collapse.** Fix $\epsilon$ (typically $0.05$). Head $(\ell,h)$ is collapsed at step $t$ if $\tilde H^{(\ell,h)}_t < \epsilon$. Run-level statistic: $C_t = \frac{1}{LH}|\{(\ell,h): \tilde H^{(\ell,h)}_t < \epsilon\}|$.

**Logit norm**, the companion diagnostic: $M_t = \max_{\ell,h,i,j} |L_{ij}|$. Since $|L_{ij}| \le \|q_i\|\|k_j\|/\sqrt{d_h} \le \|x_i\|\|x_j\|\,\|W_Q\|_2\|W_K\|_2/\sqrt{d_h}$, spectral-norm control of $W_QW_K^\top$ upper-bounds logit growth; this is the mechanism behind $\sigma$Reparam (Zhai et al., ICML 2023) and the reason $H_i \ge \log i - 2M$ up to constants for near-uniform inputs.

**Assumptions and where they fail.**
1. *Entropy is measured on a fixed held-out batch.* Violated in practice: most logging uses the training batch, so $C_t$ moves with data as well as weights.
2. *Low entropy means degenerate.* Violated: attention sinks (Xiao et al., ICLR 2024) put near-all mass on token 0 by design and are functional, not pathological. Any collapse detector that does not exclude sink heads reports mostly sinks.
3. *Heads are exchangeable.* Violated: entropy is strongly layer-dependent — early layers are broad, later layers sharp — so a global mean $\bar C_t$ is a mixture statistic and its trend can invert without any head changing.
4. *bf16 logits are the object of study.* Violated: at $|L| \gtrsim 10^4$ the softmax saturates in bf16 before it saturates mathematically, so measured entropy is partly a numerics artifact.

## 3. State of the Art

**Established (ablated, reproduced).**
- **QK-LayerNorm** — LayerNorm on $Q$ and $K$ before the dot product. Introduced for scale by Dehghani et al. (ViT-22B, ICML 2023) after logit divergence at 22B, and independently ablated by Wortsman et al. (ICLR 2024) in a controlled LR sweep. It is now default in OLMo 2, Gemma 2/3, and Chameleon. The ablation is the strong part of the evidence: removing it reintroduces the divergence at the same LR.
- **$\sigma$Reparam** (Zhai et al., ICML 2023) — reparameterize $W = \frac{\gamma}{\sigma(W)}W$ with $\sigma$ the spectral norm. Comes with an entropy lower bound in terms of $\|W_QW_K^\top\|_2$ and matching experiments on ViT/ASR/LM.

**Claimed but unablated, or benchmark-number-only.**
- **Logit soft-capping** $L \leftarrow c\tanh(L/c)$, $c=50$ in Gemma 2 (2024). Reported as used; no public ablation isolating its effect on stability, and it was dropped in Gemma 3 in favor of QK-norm.
- **nGPT** (Loshchilov et al., 2024) normalizes everything to the hypersphere and reports 4–20× fewer steps to a target loss. Entropy stability is a side claim, not an ablation.
- **Entropy-regularization losses** (add $-\lambda \bar H$ to the objective). Appear in several papers as a component; no matched-compute study at $\ge 1$B showing zero loss cost.

**Theory SOTA** is the $\sigma$Reparam bound plus Adam-instability analysis (Molybog et al., 2023), which explains the *update* blow-up but not the entropy trajectory. There is no theorem covering Adam + LayerNorm + real data.

## 4. What Is Known

- **Entropy collapse precedes loss spikes in some runs.** Zhai et al. (ICML 2023) show ViT and LM runs where mean attention entropy drops toward $0$ and training loss diverges, and $\sigma$Reparam removes both. Scale: ViT-B/ViT-L class, and small LMs — not billion-parameter LMs.
- **Attention-logit growth is reproducible at small scale with high LR.** Wortsman et al. (ICLR 2024) reproduce the ViT-22B-style divergence in models from ~40M to ~4.8B parameters by raising LR, and show max logits growing by orders of magnitude before divergence; QK-norm extends the stable LR range by roughly an order of magnitude across that whole range. This is the single most useful reproducible fact in the area.
- **Sinks are ubiquitous and low-entropy.** Xiao et al. (ICLR 2024) and Gu et al. (ICLR 2025) show first-token sink heads in essentially all pretrained decoder LMs; Gu et al. tie sink formation to optimization and data, and show it emerges early in pretraining. So $\tilde H \approx 0$ is the *normal* state of a large minority of heads.
- **Layerwise structure.** Since Clark et al. (BlackboxNLP 2019) and Vig & Belinkov (2019) it is repeatedly observed that entropy decreases with depth and that a subset of heads is near-deterministic in healthy models.
- **Pre-LN vs Post-LN** (Xiong et al., ICML 2020) determines whether warmup is needed at all; residual-branch scaling therefore confounds any entropy comparison across architectures.

## 5. What Is Not Known

- **Theoretically open.** No proof, either way, that $\tilde H_t$ stays bounded below for Adam-trained pre-LN transformers under any nontrivial data assumption. Also open: whether logit-norm control is *necessary* for stability or merely one sufficient route.
- **Empirically open.** Whether entropy collapse is causal or epiphenomenal at $\ge 10$B parameters. The intervention arm (train to divergence, restore only entropy while holding logit norm, or vice versa) is runnable today; nobody has published it. Also empirically open: whether an entropy floor costs held-out loss.
- **Methodologically blocked.** "Attention entropy" as a run-level scalar is not well defined: normalization by $\log i$, the sink exclusion rule, the batch it is measured on, and the layer aggregation are all unstated free choices, and different choices reverse the sign of the reported trend. Until a reference measurement is fixed, cross-paper comparison is not meaningful.

## 6. Why It Is Hard

The obstruction is **confounded measurement compounded by non-identifiability**. Entropy, max logit, gradient norm, and Adam's second-moment estimate all move together in the ~100 steps before a spike. Interventions that lower one lower all four, so no published experiment separates "entropy was restored" from "logit norm was capped" from "the effective learning rate was reduced". QK-norm, $\sigma$Reparam and soft-capping each do all three at once.

Second: the failure is **rare and scale-gated**. At small scale you must force it with an unrealistic LR, which changes the mechanism; at realistic scale each observation costs a run. Wortsman et al.'s LR-sweep proxy is the best available workaround and is explicitly a proxy.

Third: **no ground truth for "pathological"**. Sink heads are low-entropy and useful. Without a functional definition of a bad low-entropy head, any detector's precision is undefined.

## 7. Current Research (as of 2026)

- **Normalization-by-default.** Open-weight releases (OLMo 2, Gemma 3, and the Qwen line) ship QK-norm as standard; the research question has shifted from "does it help" to "what does it cost". *(frontier — verify current model-card details.)*
- **Sink mechanism.** Follow-ons to Gu et al. (ICLR 2025) study whether sinks are a compressed no-op and whether removing them changes long-context behavior — directly relevant, since it defines the null class for any collapse detector.
- **Optimizer-side control.** Muon and other spectral-norm-constrained optimizers control $\|W\|_2$ directly, which by the $\sigma$Reparam bound implies an entropy floor. Whether observed stability gains route through this channel is untested. *(frontier — verify.)*
- **Instability forecasting.** Small groups are training spike classifiers on logged run telemetry; results so far are internal and unpublished. *(frontier — verify.)*

## 8. Concrete Next Experiment

**Question.** Is entropy collapse causal for the loss spike, or a co-symptom of logit growth?

**Scale.** 1.4B-parameter decoder LM, ~30B tokens, pre-LN, bf16, AdamW. Choose the LR by sweep so that ~40% of seeds spike (Wortsman et al.'s protocol). Run 12 seeds per arm; ~5 arms. Roughly 5k A100-hours total — an academic-lab budget, not a frontier-lab one.

**Arms.**
1. *Control:* baseline, no intervention.
2. *Logit-only:* soft-cap $L \leftarrow 50\tanh(L/50)$. Caps logit norm; entropy floor follows only indirectly.
3. *Entropy-only:* per-head penalty $\lambda\max(0, \epsilon - \tilde H^{(\ell,h)})^2$ with $\epsilon = 0.05$, sink heads excluded, $\lambda$ tuned so the floor binds *without* reducing $M_t$ below the control's pre-spike trajectory.
4. *Both.*
5. *QK-norm* (reference arm — known to work).

**Deciding number.** Spike rate over 12 seeds per arm. If arm 3 (entropy floor, logit norm free) drops the spike rate from the control's ~40% to $\le 10\%$ — a difference detectable at $p<0.05$ with $n=12$ by Fisher's exact test — entropy is causal. If arm 3 stays within 1 s.e. of control while arm 2 drops, entropy is epiphenomenal and the community should log $M_t$, not $H_t$. Secondary number: held-out loss gap at 30B tokens between arm 5 and control, which prices the standard fix.

**Precondition.** Publish the measurement spec first — normalization by $\log i$, sink exclusion (head is a sink if $\bar a_{i0} > 0.5$), fixed 64-sequence held-out probe batch, per-layer reporting. Without it the result is not comparable to anything.

## 9. Key References

- **[Foundational]** Zhai, Likhomanenko, Littwin, Busbridge, Ramapuram, Zhang, Gu, Susskind. *Stabilizing Transformer Training by Preventing Attention Entropy Collapse.* ICML 2023. — arXiv:2303.06296
- **[SOTA]** Wortsman, Liu, Xiao, Everett, Alemi, Adlam, Co-Reyes, Gur, Kumar, Novak, Pennington, Sohl-Dickstein, Xu, Lee, Gilmer, Kornblith. *Small-scale proxies for large-scale Transformer training instabilities.* ICLR 2024. — arXiv:2309.14322
- **[Foundational]** Dehghani et al. *Scaling Vision Transformers to 22 Billion Parameters.* ICML 2023. — arXiv:2302.05442
- **[Foundational]** Henry, Dachapally, Pawar, Chen. *Query-Key Normalization for Transformers.* Findings of EMNLP 2020. — arXiv:2010.04245
- **[SOTA]** Xiao, Tian, Chen, Han, Lewis. *Efficient Streaming Language Models with Attention Sinks.* ICLR 2024. — arXiv:2309.17453
- **[SOTA]** Gu, Pagliardini, Song, Jaggi et al. *When Attention Sink Emerges in Language Models: An Empirical View.* ICLR 2025. — arXiv:2410.10781
- **[Foundational]** Xiong, Yang, He, Zheng, Zheng, Xing, Zhang, Lan, Wang, Liu. *On Layer Normalization in the Transformer Architecture.* ICML 2020. — arXiv:2002.04745
- **[Theory]** Molybog et al. *A Theory on Adam Instability in Large-Scale Machine Learning.* 2023. — arXiv:2304.09871
- **[Survey]** Clark, Khandelwal, Levy, Manning. *What Does BERT Look At? An Analysis of BERT's Attention.* BlackboxNLP @ ACL 2019. — arXiv:1906.04341

## 10. Worked Example

A 12-layer, 12-head model, $d_h = 64$, sequence length $n = 1024$. Take layer 8.

Healthy head: $\bar H \approx 3.1$ nats. Mean $\log i$ over $i=2..1024$ is about $6.3$, so $\tilde H \approx 0.49$. Sink head: $a_{i0} \approx 0.92$, remaining mass roughly uniform over $i-1$ positions. Then

$$H_i \approx -0.92\log 0.92 - 0.08\log\frac{0.08}{i-1} \approx 0.077 + 0.08(2.53 + \log(i-1)),$$

giving $H_{1024} \approx 0.83$ nats, $\tilde H \approx 0.12$. Collapsing head at step 9,000 of a spiking run: $a_{i,j^*} = 0.997$, $\tilde H \approx 0.004$.

Now the obstruction. Report the run-level mean over all 144 heads. Suppose 30 heads are sinks at $\tilde H = 0.12$, 110 are healthy at $0.49$, and 4 are collapsing at $0.004$. Mean $\tilde H = (30 \cdot 0.12 + 110\cdot 0.49 + 4\cdot 0.004)/144 = 0.400$. Before collapse, with those 4 heads at $0.49$: $0.414$. The signal is a **3.4% shift in the run-level mean** — smaller than the batch-to-batch variation of the same statistic measured on training batches (typically 5–8% for a 64-sequence probe). The four collapsing heads are individually unmistakable at $\tilde H = 0.004$, but only if you look per head *and* have already excluded the 30 sinks that also sit near the floor.

That is the whole difficulty in one calculation: the aggregate statistic everyone logs is dominated by healthy heads and cannot see the event, while the per-head statistic that can see it is indistinguishable from the sinks unless the sink exclusion rule is fixed in advance. Section 8's experiment is only interpretable if that rule is published before the runs start.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*