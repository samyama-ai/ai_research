---
id: 02-attention/attention-entropy-regularization
title: "Attention Entropy Regularization for Training Stability"
topic: 02-attention
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Attention Entropy Regularization for Training Stability

> **Topic:** Attention Mechanisms · **ID:** `02-attention/attention-entropy-regularization` · **Status:** partially-solved

## 1. Problem Statement

Transformer training runs that diverge are frequently preceded by *attention entropy collapse*: the softmax distribution of one or more heads concentrates on a single key, attention logits grow without bound, and the loss spikes. The question is whether **directly regularizing attention entropy** is a correct and sufficient fix — and if not, what the entropy statistic is actually a proxy for.

Three variants, with different difficulty:

- **Measurement.** Given a training run, define an entropy statistic $\tilde H$ that (a) is comparable across layers, heads, sequence positions and context lengths, and (b) predicts an impending loss spike some number of steps ahead. Decision predicate: does $\tilde H$ beat "max attention logit" as an early-warning signal at matched false-positive rate?
- **Method.** Add a penalty $\lambda \cdot (-\bar H)$ (or a floor constraint $\tilde H \ge \tau$) to the training loss. Solved iff, at fixed compute, it widens the stable learning-rate window as much as qk-layernorm or $\sigma$Reparam while costing no more final loss.
- **Theory.** Prove that entropy bounded away from $0$ implies bounded attention-logit growth and non-divergence under a stated optimizer, or exhibit a counterexample.

Status is *partially-solved* because the **indirect** interventions (spectral reparameterization, QK normalization) work and are deployed at scale; the **direct** entropy penalty is the one that is neither established nor cleanly refuted.

## 2. Formal Setting

For layer $\ell$, head $h$, query position $i$ over keys $j \in \mathcal{K}_i$ (unmasked set, $T_i = |\mathcal{K}_i|$):

$$ s_{ij} = \frac{\langle W_Q x_i,\; W_K x_j\rangle}{\sqrt{d_k}}, \qquad a_{ij} = \frac{\exp(s_{ij})}{\sum_{k\in\mathcal{K}_i}\exp(s_{ik})}, \qquad H_i = -\sum_{j\in\mathcal{K}_i} a_{ij}\log a_{ij}. $$

**As measured.** $H_i$ is computed in nats from the fp32 post-softmax probabilities *before* dropout, on a fixed held-out batch of $B=64$ sequences at the training context length, not on the training batch (attention dropout and packing otherwise contaminate the statistic). Because $H_i \le \log T_i$ and causal masking makes $T_i = i$, the raw mean $\frac{1}{T}\sum_i H_i$ confounds entropy with position. The comparable quantity is the normalized entropy

$$ \tilde H_{\ell,h} = \frac{1}{|\mathcal{I}|}\sum_{i \in \mathcal{I}} \frac{H_i}{\log T_i}, \qquad \mathcal{I} = \{i : T_i \ge 64\}, $$

reported per head, with the run-level summary $\min_{\ell,h} \tilde H_{\ell,h}$ (collapse is a per-head event; the mean hides it). FlashAttention never materializes $a_{ij}$, so measurement requires either a recompute pass or the online log-sum-exp statistics; the latter gives $\max_j s_{ij}$ and the partition function cheaply but not $H_i$.

**Elementary bound.** If $|s_{ij}| \le M$ for all $j$, then $a_{ij} \le e^{2M}/T_i$, so

$$ H_i \ge \log T_i - 2M. $$

With $\|x\| \le \gamma$, $M \le \gamma^2 \sigma(W_Q^\top W_K)/\sqrt{d_k}$: entropy is controlled by the spectral norm of the query–key product and the input norm. This is the mechanism $\sigma$Reparam exploits (Zhai et al., ICML 2023).

**Assumptions, and which are violated.** (i) *Low entropy is pathological* — violated: attention sinks are low-entropy and functionally load-bearing (§4). (ii) *Inputs are norm-bounded* — violated: residual-stream norm grows roughly monotonically with depth and training step in pre-LN models. (iii) *Heads are exchangeable, so a single $\lambda$ suits all* — violated: induction heads and sink heads occupy opposite ends of the entropy range in the same layer. (iv) *Entropy is scale-free* — violated by (ii) and by context-length extension, where $\log T_i$ changes but head function does not.

## 3. State of the Art

**Established (ablated, multi-domain, or independently reproduced):**

- **$\sigma$Reparam** (Zhai et al., *Stabilizing Transformer Training by Preventing Attention Entropy Collapse*, ICML 2023, arXiv:2303.06296): reparameterize $W = \frac{\gamma}{\sigma(W)}W$ with learned scalar $\gamma$. Proves a lower bound on attention entropy of the above form and shows across ViT/ImageNet, machine translation, LibriSpeech ASR and RL that warmup and, in some configurations, LayerNorm can be removed without divergence. This is a *bound-based* intervention, not an entropy penalty.
- **qk-layernorm** (Henry et al., *Query-Key Normalization for Transformers*, Findings of EMNLP 2020; scaled up in Dehghani et al., *Scaling Vision Transformers to 22 Billion Parameters*, ICML 2023): LayerNorm on $q$ and $k$ before the dot product bounds logit magnitude directly. ViT-22B reports divergence from attention-logit growth at 8B parameters, fixed by qk-layernorm.
- **Attention-logit growth as the reliable predictor** (Wortsman et al., *Small-scale proxies for large-scale Transformer training instabilities*, ICLR 2024): the max attention logit growing into the $10^3$–$10^4$ range precedes loss divergence; qk-layernorm restores stability across roughly three orders of magnitude of learning rate up to 1.2B parameters.

**Claimed but unablated, or benchmark-only:**

- Direct entropy maximization as a regularizer exists mainly in the *bias/robustness* literature, not the stability literature: Attanasio et al., *Entropy-based Attention Regularization Frees Unintended Bias Mitigation from Lists* (Findings of ACL 2022) adds a negative-entropy term at BERT-base scale and reports gains on hate-speech transfer sets. No LM-scale stability ablation.
- "Entropy collapse causes divergence" is widely repeated as causal. The published evidence is correlational plus the success of interventions that bound entropy *via* logits — which also bound many other things.
- Practitioner reports of entropy floors in production LLM runs circulate but are, as of this writing, benchmark numbers in blog posts without controls.

Note a naming collision: *policy* entropy collapse in RLHF/GRPO (e.g. Cui et al., 2025) is a different quantity and does not transfer.

## 4. What Is Known

- **Bound.** $H_i \ge \log T_i - 2\max_j|s_{ij}|$ (elementary); the spectral-norm version is Zhai et al. (ICML 2023). Entropy control and logit control are formally linked in one direction: bounded logits $\Rightarrow$ high entropy. The converse is false — high mean entropy is compatible with one enormous logit on a rare token.
- **Rank collapse is a separate failure.** Dong et al., *Attention is Not All You Need: Pure Attention Loses Rank Doubly Exponentially with Depth* (ICML 2021): pure self-attention converges to rank-1 at a doubly exponential rate in depth. High-entropy (near-uniform) attention *accelerates* this; skip connections and MLPs counteract it. So entropy maximization and rank preservation pull in opposite directions.
- **Low entropy is often correct.** Xiao et al., *Efficient Streaming Language Models with Attention Sinks* (ICLR 2024): many heads in Llama-2-7B place the majority of their mass on the first token; evicting the first few tokens' KV raises perplexity by more than an order of magnitude. Gu et al., *When Attention Sink Emerges in Language Models* (ICLR 2025) trace sink formation to optimization and data properties. Barbero et al. (2025) argue sinks limit over-mixing. Bondarenko et al., *Quantizable Transformers* (NeurIPS 2023) show the "no-op" low-entropy heads generate the activation outliers that break INT8 quantization, and that clipped-softmax/gated-attention fixes them — a case where suppressing the low-entropy head helped, measured at BERT/OPT-125M–1.3B scale.
- **Scale of the stability result:** 1.2B parameters (Wortsman et al.), ViT-22B (Dehghani et al.), ViT-B/L and 100M-class translation models ($\sigma$Reparam).

## 5. What Is Not Known

- **Theoretically open.** No theorem states that an entropy floor $\tilde H \ge \tau$ implies non-divergence for AdamW at learning rate $\eta$. The implication runs the wrong way (logits $\to$ entropy); nothing rules out divergence with healthy mean entropy. Also open: whether an entropy penalty has a stationary point that is not degenerate given the rank-collapse pressure of Dong et al.
- **Empirically open.** A head-wise entropy floor versus qk-layernorm versus $\sigma$Reparam, at matched compute, on a learning-rate sweep at $\ge$1B parameters, has not been published. The experiment is a few thousand GPU-hours — runnable, unrun.
- **Methodologically blocked.** There is no agreed normalization making entropy comparable across context lengths and heads, and no ground-truth label for "pathological low entropy" versus "sink doing its job". Until a discriminating statistic exists, any penalty is applied to a quantity that mixes the failure mode with a useful mechanism.

## 6. Why It Is Hard

**The statistic is non-identifying.** A single scalar $\tilde H_{\ell,h}$ cannot distinguish (a) a head whose logits are diverging, from (b) a sink head whose mass sits on token 0 by design, from (c) a sharp induction head correctly copying one token. All three read as low entropy. A penalty applied to $\tilde H$ therefore taxes (b) and (c) to fix (a), and — as §10 shows — the gradient concentrates almost entirely on the largest-mass key, i.e. on exactly the sink. The intervention is misnamed: "entropy regularization" is, at realistic attention sparsity, sink suppression.

Compounding this: the measurement is not free under FlashAttention, the failure it targets is rare and run-specific (so each data point costs a full diverging run), and the working alternatives (qk-layernorm) are one line of code with no tuned hyperparameter, setting a high bar.

## 7. Current Research (as of 2026)

- QK normalization is now near-default in large open models; the research question has shifted from *whether to bound logits* to *which normalization interacts best with muP-style scaling and long context*.
- Sink mechanism work (Oxford/Google DeepMind — Barbero et al.; Gu et al.) is reframing low entropy as functional, which directly undercuts the penalty framing. *(frontier — verify)*
- Softmax alternatives that permit "no-op" attention without logit blowup: clipped softmax and gated attention (Qualcomm AI Research), softmax-off-by-one style variants, and differential-attention constructions (Microsoft Research, 2024). These sidestep the entropy statistic entirely.
- Entropy-as-diagnostic tooling inside training dashboards, used as a *monitor* rather than a loss term. *(frontier — verify)*

## 8. Concrete Next Experiment

**Scale.** Decoder-only LM, 1.3B parameters, 30B tokens, context 4096, AdamW, cosine schedule, single seed per cell (three seeds for the two best cells).

**Arms** (identical data order and init):
1. **Control A** — baseline, no intervention.
2. **Control B** — qk-layernorm (the standard to beat).
3. **Treatment** — per-head hinge penalty $\mathcal{L}_{\text{reg}} = \lambda \sum_{\ell,h}\max(0,\ \tau - \tilde H_{\ell,h})^2$ with $\tau = 0.15$, $\lambda \in \{10^{-3},10^{-2},10^{-1}\}$.
4. **Treatment−** — same penalty with token 0 excluded from the entropy sum (sink-exempt), to separate "entropy control" from "sink suppression".

**Sweep.** Learning rate over $\{1,2,4,8,16,32\}\times 10^{-4}$.

**Deciding number.** The **width of the stable learning-rate window in $\log_{10}$ units**, where "stable" means final validation loss within $0.02$ nats of the best cell in that arm and no loss spike exceeding $0.5$ nats. Treatment wins only if its window is $\ge$ the qk-layernorm window and its best-cell loss is not worse by more than $0.01$ nats. Secondary readout: if Treatment− matches Treatment, entropy control is doing the work; if only Treatment (which suppresses sinks) is stable, the mechanism is sink suppression and the name is wrong.

## 9. Key References

- **[Foundational]** Vaswani et al. *Attention Is All You Need.* NeurIPS, 2017. — arXiv:1706.03762
- **[SOTA]** Zhai, Likhomanenko, Littwin, Busbridge, Ramapuram, Zhang, Gu, Susskind. *Stabilizing Transformer Training by Preventing Attention Entropy Collapse.* ICML, 2023. — arXiv:2303.06296
- **[SOTA]** Wortsman, Liu, Xiao, Everett, Alemi, Adlam, Co-Reyes, Gur, Kumar, Novak, Pennington, Sohl-Dickstein, Xu, Lee, Gilmer, Kornblith. *Small-scale proxies for large-scale Transformer training instabilities.* ICLR, 2024. — arXiv:2309.14322
- **[SOTA]** Dehghani et al. *Scaling Vision Transformers to 22 Billion Parameters.* ICML, 2023. — arXiv:2302.05442
- **[Foundational]** Henry, Dachapally, Pawar, Chen. *Query-Key Normalization for Transformers.* Findings of EMNLP, 2020. — arXiv:2010.04245
- **[Theory]** Dong, Cordonnier, Loukas. *Attention is Not All You Need: Pure Attention Loses Rank Doubly Exponentially with Depth.* ICML, 2021. — arXiv:2103.03404
- **[Counter-evidence]** Xiao, Tian, Chen, Han, Lewis. *Efficient Streaming Language Models with Attention Sinks.* ICLR, 2024. — arXiv:2309.17453
- **[Counter-evidence]** Gu, Pagliardini, Jaggi, et al. *When Attention Sink Emerges in Language Models: An Empirical View.* ICLR, 2025.
- **[Related]** Bondarenko, Nagel, Blankevoort. *Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing.* NeurIPS, 2023. — arXiv:2306.12929
- **[Related]** Attanasio, Nozza, Hovy, Baralis. *Entropy-based Attention Regularization Frees Unintended Bias Mitigation from Lists.* Findings of ACL, 2022.

## 10. Worked Example

One head, context $T_i = 1024$, with a sink: $a_{i0} = 0.9$, remaining $0.1$ spread uniformly over $1023$ keys ($a_{ij} = 9.78\times10^{-5}$).

$$ H_i = -0.9\log 0.9 - 0.1\log\!\left(\tfrac{0.1}{1023}\right) = 0.0948 + 0.9233 = 1.018\ \text{nats}, \qquad \tilde H = \tfrac{1.018}{6.931} = 0.147. $$

That is a strong "collapse" reading — well under any plausible floor $\tau = 0.15$–$0.3$. Now where does the penalty's gradient go? Using $\partial H_i/\partial s_{ij} = -a_{ij}(\log a_{ij} + H_i)$:

- sink logit: $-0.9\,(\log 0.9 + 1.018) = -0.822$
- any tail logit: $-9.78\times10^{-5}(\log(9.78\times10^{-5}) + 1.018) = +8.03\times10^{-4}$
- summed over all 1023 tail logits: $+0.821$

The gradient magnitude on the single sink logit is about $1000\times$ that on any individual tail logit, and the entire tail together only balances it. An entropy floor here does one thing: it pushes down $s_{i0}$. It is a sink-deletion operator wearing an information-theoretic name.

The obstruction is now visible. Xiao et al. show that deleting exactly this structure in a 7B model costs more than an order of magnitude of perplexity. So the regularizer, applied to the head where the statistic screams loudest, attacks the mechanism that head exists to provide — while a genuinely diverging head, whose logits are at $10^4$ but spread over a handful of keys, may register a *higher* $\tilde H$ and be taxed less. Any credible version of this method must first supply a statistic that separates the two cases; §8's Treatment− arm is the cheapest test of whether such a separation buys anything.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*